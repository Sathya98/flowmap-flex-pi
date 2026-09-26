"""CUDA-graph capture of a whole training microstep: forward (with its JVPs), loss, backward.

A flow-map microstep is launch-bound: tens of thousands of small kernels, each
issued by Python and the dispatcher, with the GPU idle most of the time. A CUDA
graph records the kernel sequence once and replays it with one launch. It works
at kernel level, so forward AD, fused JVP kernels and checkpoint recomputation
inside the backward are all captured as they run.

Contract (the caller's side):
- ``step()`` reads only *static* tensors (allocated before capture, refilled in
  place with ``copy_`` before each replay) and returns ``(loss, outputs)``;
  ``outputs`` is a dict of tensors (e.g. metrics) read after the replay.
- No host synchronization inside ``step()`` (``float(t)``, ``.item()``, ``bool(t)``,
  data-dependent shapes, pageable CPU→GPU copies). CPU-side control flow runs
  once at capture and is frozen: branches must be fixed per graph.
- Randomness on the default CUDA generator is fine (each replay draws fresh
  numbers); CPU randomness is frozen at capture.
- Gradients accumulate into the parameters' existing ``.grad`` tensors on every
  replay (as eager accumulation does). Zero them in place (``grad.zero_()``), never
  set them to None, or the graph writes into memory nobody reads.
- Collective hooks (ZeRO-2 bucketed reduce) cannot be captured: reduce once per
  update outside the graph.

Decisions that depend on tensor *contents* (e.g. the fused attention-JVP's mask
row groups) are frozen at capture. Code that freezes one calls ``guard(condition)``
with a device-side check that the assumption still holds; every replay ANDs it into
a flag, and ``CapturedStep.check()`` raises if any replay broke an assumption.
"""
import torch

_WARMUP = False     # inside CapturedStep's eager warmup (code may record what capture needs)
_GUARD = None       # the capturing step's flag, while capturing
_KEEP = None        # objects the capturing step's kernels read (kept alive with the graph)


def warming_up():
    return _WARMUP


def keep_alive(obj):
    """While capturing: hold ``obj`` (e.g. index tensors a frozen plan reads) as long as the graph."""
    if _KEEP is None:
        raise RuntimeError("flowmap_core.graphs.keep_alive() is only valid during CapturedStep capture")
    _KEEP.append(obj)


def guard(condition):
    """While capturing: AND a device bool into the step's validity flag (no host sync)."""
    if _GUARD is None:
        raise RuntimeError("flowmap_core.graphs.guard() is only valid during CapturedStep capture")
    _GUARD.logical_and_(condition.to(_GUARD.device).reshape(()))


class CapturedStep:
    """``CapturedStep(step, params).capture()`` then ``replay()`` per microstep."""

    def __init__(self, step, params, warmup=2, pool=None, autocast_dtype=None):
        self.step, self.params = step, [p for p in params if p.requires_grad]
        self.warmup, self.pool, self.autocast_dtype = warmup, pool, autocast_dtype
        self.graph = self.loss = self.outputs = None

    def _run(self):
        if self.autocast_dtype is None:
            loss, outputs = self.step()
        else:
            # The autocast weight-cast cache would hold tensors from outside the graph.
            with torch.autocast("cuda", dtype=self.autocast_dtype, cache_enabled=False):
                loss, outputs = self.step()
        loss.backward()
        return loss.detach(), outputs

    def capture(self):
        """Warm up on a side stream (allocates grads, autotunes kernels), then capture."""
        global _WARMUP, _GUARD, _KEEP
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        _WARMUP = True
        try:
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self._run()
        finally:
            _WARMUP = False
        torch.cuda.current_stream().wait_stream(side)
        # Parameters this step never reaches keep grad None; the same control flow
        # at capture leaves them untouched.
        self.unused = [p for p in self.params if p.grad is None]
        self.zero_grad()
        self.valid = torch.ones((), dtype=torch.bool, device="cuda")
        self.graph = torch.cuda.CUDAGraph()
        self.kept = []
        _GUARD, _KEEP = self.valid, self.kept
        try:
            with torch.cuda.graph(self.graph, pool=self.pool):
                self.loss, self.outputs = self._run()
        finally:
            _GUARD = _KEEP = None
        return self           # capture only records: no kernel ran, the grads are still zero

    def replay(self):
        self.graph.replay()
        return self.loss, self.outputs

    def check(self):
        """Raise if any replay since the last check violated a capture-time assumption (one sync)."""
        ok = bool(self.valid)
        self.valid.fill_(True)
        if not ok:
            raise RuntimeError("A CUDA-graph replay saw inputs that differ from a capture-time "
                               "assumption (e.g. an attention mask's contents); its results are invalid")

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def pool_handle(self):
        """Share this graph's memory pool with another graph that never runs concurrently."""
        return self.graph.pool()

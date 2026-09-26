"""CPU evaluation EMAs for replicated (DDP/ZeRO-2) trainable parameters.

Owned by rank zero, updated after successful optimizer updates, and persisted
with full training state. Self-distillation targets never use these weights.

For CUDA weights the update runs in the background: the weights are copied
into pinned host buffers on the current stream (so no later kernel, e.g. the
next optimizer step, can change them before the copy lands), and a thread
folds that snapshot into the shadows while training continues. The arithmetic
is the synchronous one, so results are bit-identical. Every reader waits for a
pending fold first.
"""
from contextlib import contextmanager
import threading
import time

import torch


class EvaluationEMA:
    CHUNK = 1 << 26   # elements per fold operation: few, long ops (GIL-free) in the thread

    def __init__(self, model, decays, background=None):
        self.decays = tuple(decays)
        self.updates = 0
        self.background = background     # None: in the background iff the weights are on CUDA
        self.fold_seconds = None          # duration of the last fold (timing harness)
        self._staging = self._copied = self._pending = self._error = self._chunk = None
        trainable = [(n, p.detach()) for n, p in model.named_parameters() if p.requires_grad]
        # Flat shadows, grouped by source dtype so that each group's staging buffer
        # maps onto one contiguous shadow range; per-parameter views keep the
        # state_dict layout.
        self._groups, offset = [], 0
        for dtype in sorted({p.dtype for _, p in trainable}, key=str):
            members = [(n, p) for n, p in trainable if p.dtype == dtype]
            size = sum(p.numel() for _, p in members)
            self._groups.append((dtype, members, offset, size))
            offset += size
        self._flat = {d: torch.empty(offset, dtype=torch.float32) for d in self.decays}
        views = {}
        for _, members, start, _ in self._groups:
            for name, value in members:
                views[name] = (start, value.shape)
                start += value.numel()
        self._shadow = {d: {} for d in self.decays}
        for name, value in trainable:       # model order
            start, shape = views[name]
            for decay in self.decays:
                view = self._flat[decay][start:start + value.numel()].view(shape)
                view.copy_(value)
                self._shadow[decay][name] = view

    @property
    def shadow(self):
        self.wait()
        return self._shadow

    def wait(self):
        """Block until the last update is folded in; re-raise a failure from its thread."""
        if self._pending is not None:
            self._pending.join()
            self._pending = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("Background EMA update failed") from error

    def _fold(self):
        start = time.perf_counter()
        if self._copied is not None:
            self._copied.synchronize()
        if self._chunk is None:     # reused: no large allocation (mmap + page faults) per chunk
            self._chunk = torch.empty(min(self.CHUNK, max(size for *_, size in self._groups)),
                                      dtype=torch.float32)
        for (_, _, offset, size), staged in zip(self._groups, self._staging):
            for a in range(0, size, self.CHUNK):
                b = min(a + self.CHUNK, size)
                current = self._chunk[:b - a].copy_(staged[a:b])
                for decay in self.decays:
                    self._flat[decay][offset + a:offset + b].lerp_(current, 1 - decay)
        self.fold_seconds = time.perf_counter() - start

    @torch.no_grad()
    def update(self, model):
        self.wait()
        parameters = dict(model.named_parameters())
        cuda = any(parameters[n].is_cuda for _, members, _, _ in self._groups for n, _ in members)
        if self._staging is None:
            self._staging = [torch.empty(size, dtype=dtype, pin_memory=cuda)
                             for dtype, _, _, size in self._groups]
        # Snapshot into (pinned) host buffers on the current stream: later kernels,
        # such as the next optimizer step, are ordered after these copies.
        for (_, members, _, _), staged in zip(self._groups, self._staging):
            start = 0
            for name, _ in members:
                value = parameters[name].detach()
                staged[start:start + value.numel()].view(value.shape).copy_(value, non_blocking=cuda)
                start += value.numel()
        self._copied = None
        if cuda:
            self._copied = torch.cuda.Event()
            self._copied.record()
        self.updates += 1
        if not (self.background if self.background is not None else cuda):
            self._fold()
            return

        def fold():
            try:
                self._fold()
            except BaseException as error:  # surfaced by wait()
                self._error = error
        self._pending = threading.Thread(target=fold, name="flowmap-ema", daemon=True)
        self._pending.start()

    def state_dict(self):
        return dict(decays=self.decays, updates=self.updates, shadow=self.shadow)

    def load_state_dict(self, state):
        self.wait()
        if tuple(state['decays']) != self.decays:
            raise ValueError('EMA decay factors changed on resume')
        for decay in self.decays:
            if state['shadow'][decay].keys() != self.shadow[decay].keys():
                raise ValueError('EMA parameter names changed on resume')
            for name, target in self.shadow[decay].items():
                saved = state['shadow'][decay][name]
                if saved.shape != target.shape:
                    raise ValueError(f'EMA parameter shape changed: {name}')
                target.copy_(saved)
        self.updates = int(state['updates'])

    @contextmanager
    def apply(self, model, decay):
        """Temporarily install EMA for checkpoint export, always restore live weights."""
        parameters = dict(model.named_parameters())
        backup = {}
        try:
            with torch.no_grad():
                for name, value in self.shadow[decay].items():
                    parameter = parameters[name]
                    backup[name] = parameter.detach().to(device='cpu', copy=True)
                    parameter.copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)

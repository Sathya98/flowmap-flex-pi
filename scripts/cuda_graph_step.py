#!/usr/bin/env python
"""How much does a CUDA graph cut a flow-map training microstep? (1 GPU, no DeepSpeed)

For each mode (objective variant of the config, see profile_flowmap_step.MODES):

1. eager: the production step (``model.training_loss`` + backward under bf16 autocast),
   median of --repeats timed steps;
2. syncs: host-sync call sites of one eager step in capture-ready form (training
   glue cache on, metrics deferred as GPU tensors); each one would break capture;
3. capture: ``flowmap_core.graphs.CapturedStep`` over the same step;
4. equivalence: eager step vs graph replay from the same CUDA RNG seed: loss and
   every gradient (bit-identical expected with ``--backend explicit``);
5. replay: median microstep time of the graph, and memory.

Batch shape and the self-distillation branch are fixed per graph (LSD needs one
graph per branch). The ZeRO-2 gradient reduce is not part of this measurement.

    python scripts/cuda_graph_step.py --out runs/diagnostics/cuda_graph_x \\
        --modes lmd_full,lsd_off,lsd_diag --backend tvm --batch-size 1
"""
import argparse
import datetime
import importlib.util
import json
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from flowmap_core.attention import set_jvp_attention_backend
from flowmap_core.graphs import CapturedStep
from flowmap_core.step_profile import SyncSites
from flexpi.datasets.latent_cache import CachedLatentDataset
from flexpi.models.helpers import flowmap_training as ft
from flexpi.utils import misc

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("profile_step", REPO / "scripts" / "profile_flowmap_step.py")
prof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prof)
GiB = 2 ** 30
log = prof.log


def to_gpu(sample):
    return {k: v.cuda() if torch.is_tensor(v) and k != "_flowmap_diagonal_mask" else v for k, v in sample.items()}


def set_capture_ready(model, on):
    for m in (model, model.flow_map_teacher):
        if m is not None:
            m._glue_cache_train = on
            m._glue_cache = {}


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def grads(model):
    """Gradients parked on the host (native dtype): the graph pool and an eager step do not
    both fit next to each other on the GPU."""
    return [p.grad.detach().to("cpu", copy=True) if p.grad is not None else None for p in model.parameters()]


def compare(model, ref, other=None):
    """Current grads (or the host list ``other``) against ``ref``: bit equality and the worst
    per-tensor relative L2 difference."""
    worst, identical, n = 0.0, True, 0
    current = (p.grad for p in model.parameters()) if other is None else other
    for x, y in zip(current, ref):
        if x is None or y is None:
            identical &= x is None and y is None
            continue
        n += 1
        x, y = x.cuda(), y.cuda()
        identical &= torch.equal(x, y)
        x, y = x.float(), y.float()
        worst = max(worst, float((x - y).norm() / y.norm().clamp_min(1e-30)))
    return dict(identical=bool(identical), max_rel_l2=worst, tensors=n)


def run_mode(model, base_flow, sample, name, args):
    out = {}
    diag = prof.set_mode(model, base_flow, name)
    sample = prof.with_mask(sample, diag)
    params = [p for p in model.parameters() if p.requires_grad]

    def eager_step():
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            loss, metrics = model.training_loss(sample)
        loss.backward()
        return loss, metrics

    # 1. production eager step
    set_capture_ready(model, False)
    model.zero_grad(set_to_none=True)
    for _ in range(args.warmup):
        eager_step()
    t = timed(eager_step, args.repeats)
    out["eager_s"] = statistics.median(t)
    out["eager_peak_gib"] = torch.cuda.max_memory_allocated() / GiB
    log(f"  {name}: eager {out['eager_s']:.3f}s (min {min(t):.3f})  peak {out['eager_peak_gib']:.1f} GiB")

    # 2. remaining host syncs in capture-ready form
    set_capture_ready(model, True)

    def ready_step():
        with ft.deferred_metrics(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            loss, metrics = model.training_loss(sample)
        loss.backward()
        return loss, metrics

    ready_step()                                   # fills the glue cache and TVM plans
    t = timed(ready_step, args.repeats)
    out["eager_ready_s"] = statistics.median(t)
    with SyncSites() as sites:
        ready_step()
        torch.cuda.synchronize()
    out["syncs"] = sites.report()
    log(f"  {name}: capture-ready eager {out['eager_ready_s']:.3f}s; syncs {out['syncs']['total']}: "
        f"{list(out['syncs']['sites'].items())[:6]}")

    # 3. eager references for the equivalence check, taken before capture
    def reference(seed, steps):
        model.zero_grad(set_to_none=False)
        torch.cuda.manual_seed(seed)
        for _ in range(steps):
            with ft.deferred_metrics(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss, metrics = model.training_loss(sample)
            loss.backward()
        return float(loss), {k: float(v) for k, v in metrics.items()}, grads(model)

    ref_loss, ref_metrics, ref_grads = reference(1234, 1)
    _, _, ref_grads2 = reference(99, 2)
    # Noise floor: the same eager references again (nondeterministic kernels show up here).
    floor_loss, _, floor_grads = reference(1234, 1)
    out["eager_floor"] = dict(loss_equal=floor_loss == ref_loss, step1=compare(model, ref_grads, floor_grads))
    del floor_grads
    _, _, floor_grads2 = reference(99, 2)
    out["eager_floor"]["step2"] = compare(model, ref_grads2, floor_grads2)
    del floor_grads2
    log(f"  {name}: eager-vs-eager floor {out['eager_floor']}")

    # 4. capture
    def graph_step():
        with ft.deferred_metrics():
            loss, metrics = model.training_loss(sample)
        return loss, {k: v for k, v in metrics.items() if torch.is_tensor(v)}

    model.zero_grad(set_to_none=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.perf_counter()
        graph = CapturedStep(graph_step, params, warmup=args.warmup, autocast_dtype=torch.bfloat16).capture()
        out["capture_s"] = time.perf_counter() - t0
    except Exception as err:
        out["capture_error"] = f"{type(err).__name__}: {err}"[:2000]
        out["capture_traceback"] = traceback.format_exc()[-6000:]
        log(f"  {name}: CAPTURE FAILED {out['capture_error'][:300]}")
        log(out["capture_traceback"][-3000:])
        out["fatal"] = True       # a failed capture leaves the CUDA RNG/stream state unusable
        return out
    out["unused_params"] = len(graph.unused)
    out["capture_peak_gib"] = torch.cuda.max_memory_allocated() / GiB
    log(f"  {name}: captured in {out['capture_s']:.1f}s ({len(graph.unused)} params without grad), "
        f"reserved {torch.cuda.memory_reserved() / GiB:.1f} GiB")

    # equivalence: replay from the same CUDA seed as the eager references
    graph.replay()
    graph.check()                 # frozen plans still valid for these inputs
    graph.zero_grad()
    torch.cuda.manual_seed(1234)
    loss_g, metrics_g = graph.replay()
    torch.cuda.synchronize()
    out["equivalence"] = dict(loss_eager=ref_loss, loss_graph=float(loss_g),
                              metrics={k: [ref_metrics[k], float(v)] for k, v in metrics_g.items()},
                              grads=compare(model, ref_grads))
    log(f"  {name}: loss eager {ref_loss!r} graph {float(loss_g)!r}; grads {out['equivalence']['grads']}")
    del ref_grads
    graph.zero_grad()             # two replays accumulate like two eager microsteps
    torch.cuda.manual_seed(99)
    graph.replay()
    graph.replay()
    out["accumulation"] = compare(model, ref_grads2)
    del ref_grads2
    log(f"  {name}: 2-step accumulation graph vs eager {out['accumulation']}")

    # 5. replay speed
    model.zero_grad(set_to_none=False)
    t = timed(graph.replay, args.repeats)
    graph.check()
    out["graph_s"] = statistics.median(t)
    out["graph_pool_gib"] = torch.cuda.memory_reserved() / GiB
    out["speedup"] = out["eager_s"] / out["graph_s"]
    log(f"  {name}: graph {out['graph_s']:.3f}s (min {min(t):.3f})  speedup {out['speedup']:.2f}x  "
        f"reserved {out['graph_pool_gib']:.1f} GiB")
    del graph
    model.zero_grad(set_to_none=True)
    set_capture_ready(model, False)
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="flowmap_libero_lmd_full")
    parser.add_argument("--cache", default="data/latent_cache/libero_fulljoint_v2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default="lmd_full,lsd_off,lsd_diag")
    parser.add_argument("--backend", default="tvm", choices=("tvm", "explicit"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    args, overrides = parser.parse_known_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    torch.manual_seed(0)
    misc.register_work_dir(out_dir)
    log(f"{torch.cuda.get_device_name()}; torch {torch.__version__}; {args}")
    model = prof.build_model(cfg)
    model.train()
    set_jvp_attention_backend(args.backend)
    raw = instantiate(cfg.data.train)
    cached = CachedLatentDataset(args.cache, raw, require_complete=False)
    ready = np.flatnonzero(cached.cache.ready())
    positions = np.sort(np.random.default_rng(0).choice(ready, args.batch_size, replace=False))
    sample = to_gpu(default_collate([cached[int(p)] for p in positions]))
    base_flow = model.flow_map
    results = dict(args=vars(args), overrides=overrides, gpu=torch.cuda.get_device_name(),
                   created=datetime.datetime.now().isoformat(), modes={})
    for name in args.modes.split(","):
        log(f"mode {name}, batch {args.batch_size}, backend {args.backend}")
        try:
            results["modes"][name] = run_mode(model, base_flow, sample, name, args)
        except torch.cuda.OutOfMemoryError as err:
            results["modes"][name] = dict(error="OOM", detail=str(err)[:500])
            log(f"  {name}: OOM")
            model.zero_grad(set_to_none=True)
            set_capture_ready(model, False)
            torch.cuda.empty_cache()
        model.flow_map = base_flow
        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
        if results["modes"][name].get("fatal"):
            log("stopping after a failed capture")
            break
    log(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

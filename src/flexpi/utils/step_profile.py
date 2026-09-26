"""Timing-harness helpers (trainer FLEXPI_STEP_* env flags): where does a step's wall time go?

``trace_summary`` reads a torch.profiler chrome trace and reports GPU busy time
(the union of kernel/memcpy/memset intervals over all streams) against the traced
wall window, plus kernel, launch and CPU-op counts. ``SyncSites`` counts host-GPU
synchronizations by Python call site through ``torch.cuda.set_sync_debug_mode``.
"""
import collections
import gzip
import json
import shutil
import traceback
import warnings
from pathlib import Path

import torch

GPU_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def _union(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def trace_summary(trace_path, top=25):
    """Busy/idle and counts from a chrome trace; times in ms. Gzips the trace afterwards."""
    trace_path = Path(trace_path)
    events = json.loads(trace_path.read_text())["traceEvents"]
    timed = [e for e in events if e.get("ph") == "X" and "dur" in e]
    gpu = [e for e in timed if e.get("cat") in GPU_CATEGORIES]
    cpu_ops = [e for e in timed if e.get("cat") == "cpu_op"]
    launches = [e for e in timed if e.get("cat") == "cuda_runtime" and "Launch" in e.get("name", "")]
    runtime = collections.Counter(e["name"] for e in timed if e.get("cat") == "cuda_runtime")
    starts = [e["ts"] for e in timed]
    stops = [e["ts"] + e["dur"] for e in timed]
    wall = (max(stops) - min(starts)) / 1e3 if timed else 0.0
    busy = _union((e["ts"], e["ts"] + e["dur"]) for e in gpu) / 1e3
    by_kernel = collections.Counter()
    for e in gpu:
        by_kernel[e["name"][:90]] += e["dur"] / 1e3
    summary = dict(
        wall_ms=wall, gpu_busy_ms=busy, gpu_busy_fraction=busy / wall if wall else 0.0,
        gpu_events=len(gpu), kernels=sum(e.get("cat") == "kernel" for e in gpu),
        kernel_sum_ms=sum(e["dur"] for e in gpu) / 1e3,
        launches=len(launches), launch_cpu_ms=sum(e["dur"] for e in launches) / 1e3,
        cpu_ops=len(cpu_ops), cuda_runtime_calls=dict(runtime.most_common(12)),
        top_gpu_ms=dict((k, round(v, 2)) for k, v in by_kernel.most_common(top)))
    with open(trace_path, "rb") as src, gzip.open(str(trace_path) + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    trace_path.unlink()
    return summary


class SyncSites:
    """Count synchronizing CUDA calls by the innermost flexpi (or other non-torch) frame."""

    def __init__(self):
        self.sites = collections.Counter()

    def __enter__(self):
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("always")
        self._show = warnings.showwarning

        def record(message, category, filename, lineno, *args, **kwargs):
            if "synchronizing" not in str(message):
                return self._show(message, category, filename, lineno, *args, **kwargs)
            frames = [f for f in traceback.extract_stack()[:-1]
                      if not any(k in f.filename for k in ("/torch/", "warnings.py", "step_profile.py"))]
            where = frames[-1] if frames else None
            self.sites[f"{where.filename.split('src/')[-1]}:{where.lineno} {where.name}"
                       if where else f"{filename}:{lineno}"] += 1
        warnings.showwarning = record
        torch.cuda.set_sync_debug_mode("warn")
        return self

    def __exit__(self, *exc):
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = self._show
        self._warnings.__exit__(*exc)
        return False

    def report(self):
        return dict(total=sum(self.sites.values()), sites=dict(self.sites.most_common(30)))


def write_report(out_dir, name, **sections):
    path = Path(out_dir) / f"{name}.json"
    path.write_text(json.dumps(sections, indent=1, default=str))
    return path

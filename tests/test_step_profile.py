"""Timing-harness trace summary (utils/step_profile.py) on a synthetic chrome trace."""
import gzip
import json

from flexpi.utils.step_profile import trace_summary


def test_busy_time_is_the_union_of_gpu_intervals_across_streams(tmp_path):
    events = [
        dict(ph="X", cat="cpu_op", name="aten::mm", ts=0, dur=1000),
        dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel", ts=10, dur=5),
        dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel", ts=20, dur=5),
        dict(ph="X", cat="kernel", name="gemm", ts=100, dur=200),          # 100-300
        dict(ph="X", cat="kernel", name="gemm", ts=250, dur=100),          # other stream, 250-350
        dict(ph="X", cat="gpu_memcpy", name="Memcpy DtoH", ts=600, dur=100),
        dict(ph="X", cat="kernel", name="relu", ts=1900, dur=100),         # wall ends at 2000
        dict(ph="i", cat="kernel", name="marker", ts=5000),                # not a timed event
    ]
    trace = tmp_path / "t_trace.json"
    trace.write_text(json.dumps(dict(traceEvents=events)))
    s = trace_summary(trace)
    assert s["wall_ms"] == 2.0
    assert s["gpu_busy_ms"] == (250 + 100 + 100) / 1e3
    assert s["gpu_busy_fraction"] == 0.225
    assert (s["kernels"], s["gpu_events"], s["launches"], s["cpu_ops"]) == (3, 4, 2, 1)
    assert s["top_gpu_ms"]["gemm"] == 0.3
    assert not trace.exists()
    assert json.loads(gzip.open(str(trace) + ".gz").read())["traceEvents"] == events

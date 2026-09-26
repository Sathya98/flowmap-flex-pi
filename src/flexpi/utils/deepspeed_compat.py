"""DeepSpeed workarounds.

ZeRO-1/2 hook cost (DeepSpeed 0.18.x). Every parameter's gradient hook calls
``count_used_parameters_in_backward(all_params)``, which visits every trainable
parameter (``view_as`` → autograd node) on each call. With ~1.7k parameter
tensors that is ~2.8M views and ~13 s of CPU per backward, paid on every ZeRO-2
microstep (runs/diagnostics/trainer_timing_27179569: 14.9 s backward against
1.6 s without DeepSpeed). Upstream master only refreshes the count when needed
(``should_refresh_expected_hook_count``); this reproduces that here.

The count is cached per autograd graph task and refreshed once the hooks fired
in that task reach it, so parameters that join later (reentrant checkpointing)
are still counted. ``FLEXPI_DS_HOOK_COUNT_CACHE=0`` disables the patch. It is
skipped automatically on DeepSpeed versions that carry the upstream fix
(``DeepSpeedZeroOptimizer.should_refresh_expected_hook_count``, 0.18.7+).
"""
import os

import torch

_PATCHED = False


def cached_count(original):
    state = {"task": None, "count": 0, "calls": 0}

    def count_used_parameters_in_backward(parameters):
        task = torch._C._current_graph_task_id()
        if task != state["task"] or state["calls"] >= state["count"]:
            state.update(task=task, count=original(parameters), calls=0)
        state["calls"] += 1
        return state["count"]

    count_used_parameters_in_backward.__wrapped__ = original
    return count_used_parameters_in_backward


def patch_zero_hook_count():
    """Idempotent; a no-op without DeepSpeed or when the hook helper is absent."""
    global _PATCHED
    if _PATCHED or os.environ.get("FLEXPI_DS_HOOK_COUNT_CACHE", "1") == "0":
        return False
    try:
        import deepspeed.runtime.zero.stage_1_and_2 as zero12
    except ImportError:
        return False
    # DeepSpeed >= 0.18.7 already refreshes the count only when needed.
    if hasattr(getattr(zero12, "DeepSpeedZeroOptimizer", None), "should_refresh_expected_hook_count"):
        return False
    original = getattr(zero12, "count_used_parameters_in_backward", None)
    if original is None:
        return False
    zero12.count_used_parameters_in_backward = cached_count(original)
    _PATCHED = True
    return True

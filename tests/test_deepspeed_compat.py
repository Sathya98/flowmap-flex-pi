"""The cached ZeRO hook count calls the original once per backward (plus refreshes)."""
from unittest.mock import patch

import torch

from flexpi.utils.deepspeed_compat import cached_count


def test_count_cached_per_graph_task_and_refreshed_when_exhausted():
    calls = []

    def original(params):
        calls.append(len(params))
        return 3

    count = cached_count(original)
    params = [object()] * 5
    with patch.object(torch._C, "_current_graph_task_id", return_value=7):
        assert [count(params) for _ in range(3)] == [3, 3, 3]
        assert len(calls) == 1                     # one scan for three hooks
        count(params)                              # a 4th hook exceeds the count: refresh
        assert len(calls) == 2
    with patch.object(torch._C, "_current_graph_task_id", return_value=8):
        count(params)                              # new backward: new scan
        assert len(calls) == 3


def test_matches_reentrant_growth():
    """If more params join later, the refreshed count is returned."""
    counts = iter([2, 4])
    count = cached_count(lambda params: next(counts))
    with patch.object(torch._C, "_current_graph_task_id", return_value=1):
        assert [count([]) for _ in range(4)] == [2, 2, 4, 4]

"""Reject missing, duplicated or incomplete LIBERO comparisons."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('flowmap_libero_eval',
    Path(__file__).resolve().parents[1]/'scripts/flowmap_libero_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_rollouts_and_incomplete_or_duplicate_results(tmp_path):
    row = dict(task_suite='libero_spatial', task_id=0, total_episodes=2,
               successes=1, success_episodes=[0], failure_episodes=[1])
    path = tmp_path/'task0_results.json'
    path.write_text(json.dumps(row))
    expected = [('libero_spatial', 0)]
    assert module.validate_results(tmp_path, 2, expected) == dict(tasks=1, trials=2, successes=1)
    with pytest.raises(ValueError, match='task coverage'):
        module.validate_results(tmp_path, 2, expected+[('libero_spatial', 1)])
    with pytest.raises(ValueError, match='episodes'):
        module.validate_results(tmp_path, 3, expected)
    row['failure_episodes'] = [0]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='episodes'):
        module.validate_results(tmp_path, 2, expected)
    row['failure_episodes'] = [1]
    path.write_text(json.dumps(row))
    (tmp_path/'duplicate_results.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Duplicate'):
        module.validate_results(tmp_path, 2, expected)

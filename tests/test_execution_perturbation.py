import numpy as np
from experiments.libero.execution_perturbation import ExecutionPerturbation, select_calibrated_candidate


def test_delay_stops_at_observation_and_preserves_remaining_plan():
    actions = np.arange(32*7, dtype=float).reshape(32,7)/300
    p = ExecutionPerturbation('delay', 4)
    p.begin_chunk(actions)
    np.testing.assert_array_equal(p.apply(actions[0]), actions[0])
    p.begin_chunk(actions)
    actual = np.array([p.apply(a) for a in actions])
    np.testing.assert_array_equal(actual[:4,:6], 0.)
    np.testing.assert_array_equal(actual[4:16], actions[:12])
    np.testing.assert_array_equal(actual[16:], actions[16:])
    p.begin_chunk(actions)
    np.testing.assert_array_equal(p.apply(actions[0]), actions[0])


def test_translation_bias_changes_only_translation_and_window():
    actions = np.zeros((32,7)); actions[:,0] = .9
    p = ExecutionPerturbation('translation_bias', .25)
    p.begin_chunk(actions); p.begin_chunk(actions)
    actual = np.array([p.apply(a) for a in actions])
    np.testing.assert_array_equal(actual[:16,0], 1.)
    np.testing.assert_array_equal(actual[:,1:], actions[:,1:])
    np.testing.assert_array_equal(actual[16:], actions[16:])
    assert p.report()['altered_steps'] == 16


def test_clean_is_identity_and_matching_plan_hash():
    a = np.ones((32,7))
    clean = ExecutionPerturbation()
    disturbed = ExecutionPerturbation('delay', 8)
    for p in (clean, disturbed): p.begin_chunk(a); p.begin_chunk(a)
    np.testing.assert_array_equal([clean.apply(x) for x in a], a)
    assert clean.planned_sha256 == disturbed.planned_sha256


def test_calibration_selection_is_frozen_rule():
    rows = [dict(order=i, failure_rate=r) for i,r in enumerate((0.,0.,.5,1.))]
    assert select_calibrated_candidate(rows)['order'] == 2
    assert select_calibrated_candidate(rows[:2])['order'] == 1
    assert select_calibrated_candidate([rows[0],rows[-1]])['order'] == 3

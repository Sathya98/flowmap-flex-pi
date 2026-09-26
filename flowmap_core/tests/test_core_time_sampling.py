"""Off-diagonal (s, t) sampling modes and the curriculum schedule."""
import pytest
import torch

from flowmap_core.flowmap import (FlowMapObjectiveConfig, sample_inference_grid_pairs,
                                  sample_level_pair_strip, training_time_pairs)


def test_uniform_jump_has_uniform_jump_sizes_and_valid_pairs():
    torch.manual_seed(0)
    s, t = sample_level_pair_strip(200000, 1.0, "cpu", torch.float32, sampling="uniform_jump")
    h = s - t
    assert bool(((t >= 0) & (s <= 1) & (h > 0)).all())
    assert abs(float((h > 0.9).float().mean()) - 0.1) < 0.005          # vs 0.01 under uniform area
    s, t = sample_level_pair_strip(200000, 1.0, "cpu", torch.float32, sampling="uniform_triangle")
    assert abs(float(((s - t) > 0.9).float().mean()) - 0.01) < 0.002
    s, t = sample_level_pair_strip(1000, 0.3, "cpu", torch.float32, sampling="uniform_jump")
    assert float((s - t).max()) <= 0.3 + 1e-6


def test_inference_grid_trains_exactly_the_sampling_segments():
    torch.manual_seed(0)
    s, t = sample_inference_grid_pairs(100000, (1, 2), "cpu")
    sources = torch.unique(s)
    torch.testing.assert_close(sources, torch.tensor([0.5, 1.0]))
    one = (s == 1) & (t < 0.5)                                         # only a 1-jump reaches below .5 from s=1
    assert abs(float(one.float().mean()) - 0.25) < 0.01                # half of the 1-jump pairs
    assert abs(float((s == 0.5).float().mean()) - 0.25) < 0.01         # half of the 2-jump pairs
    assert bool(((t < s) & (t >= 0)).all()) and bool((t[s == 0.5] >= 0).all())
    assert float((s - t)[s == 0.5].max()) <= 0.5 + 1e-6
    # With a shifted schedule the sources are the shifted nodes, as in inference_nodes.
    cfg = FlowMapObjectiveConfig(time_sampling="inference_grid", grid_steps=(2,), schedule_shift=3.0)
    s, _ = training_time_pairs(cfg, 1000, "cpu")
    torch.testing.assert_close(torch.unique(s), cfg.inference_nodes(2, "cpu")[:2].flip(0))


def test_curriculum_schedule():
    cfg = FlowMapObjectiveConfig(strip_width_start=0.25, strip_anneal_updates=1000, uniform_jump_from_update=1000)
    assert cfg.has_time_schedule
    assert [cfg.strip_width_at(k) for k in (0, 500, 1000, 5000)] == [0.25, 0.625, 1.0, 1.0]
    assert cfg.time_sampling_at(999) == "uniform_triangle" and cfg.time_sampling_at(1000) == "uniform_jump"
    torch.manual_seed(0)
    s, t = training_time_pairs(cfg, 5000, "cpu", update=0)
    assert float((s - t).max()) <= 0.25 + 1e-6
    with pytest.raises(ValueError, match="optimizer update"):
        training_time_pairs(cfg, 4, "cpu")
    assert not FlowMapObjectiveConfig().has_time_schedule
    assert training_time_pairs(FlowMapObjectiveConfig(), 4, "cpu")[0].shape == (4,)


@pytest.mark.parametrize("kwargs, match", [
    (dict(time_sampling="nope"), "time_sampling"),
    (dict(grid_steps=(1, 1)), "grid_steps"),
    (dict(strip_width_start=0.25), "strip schedule"),
    (dict(strip_width_start=0.9, strip_anneal_updates=10, strip_width=0.5), "strip_width_start"),
    (dict(time_sampling="inference_grid", strip_width_start=0.25, strip_anneal_updates=10), "no strip"),
    (dict(time_sampling="inference_grid", grid_steps=(1,), strip_width=0.5), "strip_width"),
])
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FlowMapObjectiveConfig(**kwargs)

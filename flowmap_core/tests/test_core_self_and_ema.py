"""Self-distillation helpers (diagonal mask, slice_batch, TimeLossWeight) and EvaluationEMA."""
from dataclasses import dataclass

import torch
from torch import nn

from flowmap_core.ema import EvaluationEMA
from flowmap_core.flowmap_self import TimeLossWeight, slice_batch, update_diagonal_mask


def test_effective_batch_split_and_reproducible_resume():
    for step in (0, 1, 47):
        masks = torch.cat([update_diagonal_mask(1, 48, 4, r, m, step, 42)
                           for m in range(48) for r in range(4)])
        assert masks.sum() == 144
        assert torch.equal(masks, torch.cat([update_diagonal_mask(1, 48, 4, r, m, step, 42)
                                            for m in range(48) for r in range(4)]))
    assert not torch.equal(update_diagonal_mask(192, 1, 1, 0, 0, 0, 42),
                           update_diagonal_mask(192, 1, 1, 0, 0, 1, 42))


def test_diagonal_mask_is_stratified_by_microstep():
    # 4 ranks x microbatch 1 x 48 microsteps: 48 off-diagonal = 12 whole microsteps.
    for batch, accum in ((1, 48), (2, 24)):
        grid = torch.stack([torch.stack([update_diagonal_mask(batch, accum, 4, r, m, 3, 42)
                                         for r in range(4)]) for m in range(accum)])
        per_micro = grid.reshape(accum, -1)
        assert int(grid.sum()) == 144
        assert bool((per_micro.all(1) | (~per_micro).all(1)).all())   # one branch per microstep
        assert int((~per_micro).all(1).sum()) == 48 // (4 * batch)
    # Not divisible: exactly one mixed microstep carries the remainder.
    grid = torch.stack([torch.stack([update_diagonal_mask(1, 10, 3, r, m, 0, 7)
                                     for r in range(3)]) for m in range(10)]).reshape(10, 3)
    assert int((~grid).sum()) == 30 - int(30 * .75)
    assert int((grid.any(1) & (~grid).any(1)).sum()) == 1
    # Which microsteps are off-diagonal changes with the step.
    a = [bool(update_diagonal_mask(1, 48, 4, 0, m, 0, 42)) for m in range(48)]
    b = [bool(update_diagonal_mask(1, 48, 4, 0, m, 1, 42)) for m in range(48)]
    assert a != b


def test_weight_matches_reference_algebra():
    head = TimeLossWeight()
    s, t = torch.tensor([.8, .3]), torch.tensor([.2, .1])
    frequency = torch.exp(-torch.log(torch.tensor(10000.)) * torch.arange(64) / 64)
    def embed(time):
        phase = (1-time)[:, None] * frequency
        return torch.cat((phase.cos(), phase.sin()), -1) * 2**.5
    w = head.weight / (head.weight.norm(dim=1, keepdim=True)/128**.5 + 1e-4) / 128**.5
    expected = ((embed(s)+embed(t))/2**.5 @ w.T).squeeze(-1)
    torch.testing.assert_close(head(s, t), expected)


def test_background_ema_matches_synchronous_updates():
    torch.manual_seed(0)
    EvaluationEMA.CHUNK, chunk = 5, EvaluationEMA.CHUNK     # several chunks per dtype group
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2).to(torch.bfloat16))
    model[1].bias.requires_grad_(False)
    sync = EvaluationEMA(model, (.9, .99), background=False)
    background = EvaluationEMA(model, (.9, .99), background=True)
    for step in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        sync.update(model)
        background.update(model)
        with torch.no_grad():   # the snapshot was taken at update(): later edits must not leak in
            for p in model.parameters():
                p.mul_(-3)
    assert background.updates == sync.updates == 3 and background.fold_seconds is not None
    assert len(background._groups) == 2       # fp32 and bf16 parameters, one flat range each
    for decay in (.9, .99):
        assert background.shadow[decay].keys() == sync.shadow[decay].keys()
        for name, value in sync.shadow[decay].items():
            assert torch.equal(background.shadow[decay][name], value), name
    with torch.no_grad():
        model[0].weight.add_(1)
    background.update(model)
    state = background.state_dict()      # waits for the pending fold
    assert state['updates'] == 4 and background._pending is None
    EvaluationEMA.CHUNK = chunk


def test_slice_batch_selects_examples_in_nested_values():
    @dataclass
    class Flags:
        present: torch.Tensor
        scalar: torch.Tensor

    value = dict(x=torch.arange(8).reshape(4, 2), flags=Flags(torch.tensor([1, 0, 1, 1]), torch.tensor(3.)),
                 other=torch.zeros(3), name="kept")
    one = slice_batch(value, 2, 4)
    assert one["x"].tolist() == [[4, 5]] and one["flags"].present.tolist() == [1]
    assert one["other"].shape == (3,) and one["name"] == "kept" and one["flags"].scalar.ndim == 0
    many = slice_batch(value, torch.tensor([3, 0]), 4)
    assert many["x"].tolist() == [[6, 7], [0, 1]] and many["flags"].present.tolist() == [1, 1]

"""FlexPi-free use: a tiny time-conditioned transformer ("DiT") trained with every objective.

The model uses the core's forward-AD pieces (attention, LayerNorm, dual-aware
checkpoint), as a real DiT integration would. predict(x, s, t) is the average
velocity of the map X(s, t, x) = x + (t - s) v; the teacher is the instantaneous
velocity, here the student's own diagonal (self-distillation) or a frozen copy.
"""
import copy

import pytest
import torch
from torch import nn

from flowmap_core.attention import scaled_dot_product_attention
from flowmap_core.checkpoint import checkpoint
from flowmap_core.flowmap import (FlowMapObjectiveConfig, affine_flow_map, dX_dt_finite_difference,
                                  dX_dt_forward_ad, map_residuals, sample_level_pair_strip)
from flowmap_core.normalization import ForwardADLayerNorm

OBJECTIVES = ("lmd", "emd", "pfmm", "lsd", "esd", "psd_m", "psd_u")


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = ForwardADLayerNorm(dim), ForwardADLayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.modulation = nn.Linear(dim, 2 * dim)

    def forward(self, x, cond):
        shift, scale = self.modulation(cond)[:, None].chunk(2, -1)
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x) * (1 + scale) + shift).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        x = x + self.out(scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class TinyDiT(nn.Module):
    def __init__(self, channels=4, dim=32, heads=2, layers=2, use_checkpoint=False):
        super().__init__()
        self.inp, self.head = nn.Linear(channels, dim), nn.Linear(dim, channels)
        self.time = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.use_checkpoint = use_checkpoint

    def forward(self, x, s, t):
        # Two-time conditioning: source time and the jump (the time_delta of the FlexPi DiTs).
        cond = self.time(torch.stack([s, t - s], -1).to(x.dtype))
        h = self.inp(x)
        for block in self.blocks:
            h = checkpoint(block, h, cond) if self.use_checkpoint else block(h, cond)
        return self.head(h)


def residual_fn(model, teacher_model):
    predict = lambda state, s, t: (model(state[0], s, t),)
    teacher = lambda state, time: (teacher_model(state[0], time, time),)
    return predict, teacher


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_every_objective_trains_a_generic_dit(objective):
    torch.manual_seed(0)
    model = TinyDiT()
    cfg = FlowMapObjectiveConfig(enabled=True, objective=objective)
    teacher_model = model if cfg.self_distillation else copy.deepcopy(model).requires_grad_(False)
    x = torch.randn(3, 5, 4)
    s, t = sample_level_pair_strip(3, cfg.strip_width, "cpu", torch.float32)
    predict, teacher = residual_fn(model, teacher_model)
    residual = map_residuals(predict, teacher, (x,), s, t, cfg)[0]
    residual.square().mean().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.norm()) for g in grads) > 0
    if teacher_model is not model:
        assert all(p.grad is None for p in teacher_model.parameters())


def test_forward_ad_time_derivative_matches_finite_difference():
    torch.manual_seed(1)
    model = TinyDiT().double()
    x, s, t = torch.randn(2, 5, 4, dtype=torch.float64), torch.tensor([.9, .6], dtype=torch.float64), \
        torch.tensor([.3, .2], dtype=torch.float64)
    map_fn = lambda z: affine_flow_map(x, model(x, s, z), s, z)
    y, d_ad = dX_dt_forward_ad(map_fn, t)
    y_fd, d_fd = dX_dt_finite_difference(map_fn, t, eps=1e-5)
    torch.testing.assert_close(y, y_fd)
    torch.testing.assert_close(d_ad, d_fd, atol=1e-7, rtol=1e-6)


def test_checkpointed_reverse_over_forward_matches_plain():
    """LMD with teacher-input gradients: the dual-aware checkpoint changes no gradient."""
    grads = []
    for use_checkpoint in (False, True):
        torch.manual_seed(2)
        model = TinyDiT(use_checkpoint=use_checkpoint).double()
        teacher_model = copy.deepcopy(model).requires_grad_(False)
        x = torch.randn(2, 5, 4, dtype=torch.float64)
        s, t = torch.tensor([.8, .7], dtype=torch.float64), torch.tensor([.1, .4], dtype=torch.float64)
        cfg = FlowMapObjectiveConfig(enabled=True, objective="lmd", lmd_teacher_gradient="full")
        predict, teacher = residual_fn(model, teacher_model)
        map_residuals(predict, teacher, (x,), s, t, cfg)[0].square().sum().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)


def test_bf16_forward_ad_through_layernorm_and_attention():
    torch.manual_seed(3)
    model = TinyDiT().bfloat16()
    x = torch.randn(2, 5, 4, dtype=torch.bfloat16)
    s, t = torch.tensor([.9, .5]), torch.tensor([.2, .1])
    cfg = FlowMapObjectiveConfig(enabled=True, objective="lsd")
    predict, teacher = residual_fn(model, model)
    residual = map_residuals(predict, teacher, (x,), s, t, cfg)[0]
    assert residual.dtype == torch.bfloat16
    residual.float().square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())

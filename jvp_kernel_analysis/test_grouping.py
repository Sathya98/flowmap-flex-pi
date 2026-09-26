"""CPU check: the row-group decomposition equals masked attention (primal, tangent, all grads).

The fused Triton call is replaced by an explicit FP64 attention JVP, so this
tests only the grouping/gather/scatter logic. Run from jvp_kernel_analysis/:
    python -m pytest -q test_grouping.py
"""
import math
from unittest.mock import patch

import torch

import fused_attention_jvp as faj


def explicit_jvp(q, k, v, tq, tk, tv, mask=None):
    """Masked attention and its JVP via forward-over-nothing: analytic formulas."""
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    ds = (tq @ k.transpose(-2, -1) + q @ tk.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    empty = torch.isneginf(s).all(-1, keepdim=True)
    p = torch.softmax(torch.where(empty, torch.zeros_like(s), s), -1)
    p = torch.where(empty, torch.zeros_like(p), p)
    o = p @ v
    dp = p * (ds - (p * ds).sum(-1, keepdim=True))
    return o, dp @ v + p @ tv


class Explicit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        with torch.enable_grad():
            leaves = [a.detach().requires_grad_() for a in args]
            o, to = explicit_jvp(*leaves)
        ctx.leaves, ctx.outs = leaves, (o, to)
        return o.detach(), to.detach()

    @staticmethod
    def backward(ctx, do, dto):
        return torch.autograd.grad(ctx.outs, ctx.leaves, (do, dto))


def block_mask(sizes, allowed, anchor_rows=()):
    """Token groups of `sizes`; group i sees group j iff allowed[i][j]."""
    edges = [0]
    for n in sizes:
        edges.append(edges[-1] + n)
    mask = torch.zeros(edges[-1], edges[-1], dtype=torch.bool)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if allowed[i][j]:
                mask[edges[i]:edges[i + 1], edges[j]:edges[j + 1]] = True
    return mask


def check(mask, batch=1, heads=2, dim=8):
    torch.manual_seed(0)
    S = mask.shape[-1]
    args = [torch.randn(batch, heads, S, dim, dtype=torch.float64, requires_grad=True) for _ in range(6)]
    wo, wt = torch.randn(batch, heads, S, dim, dtype=torch.float64), torch.randn(batch, heads, S, dim, dtype=torch.float64)
    ref_o, ref_t = explicit_jvp(*args, mask=mask if mask.dim() == 2 else mask.reshape(batch, 1, S, S))
    ref_g = torch.autograd.grad((ref_o * wo).sum() + (ref_t * wt).sum(), args)
    with patch.object(faj.FusedAttentionJVP, "apply", Explicit.apply):
        o, t = faj.attention_jvp(*args, attn_mask=mask)
        g = torch.autograd.grad((o * wo).sum() + (t * wt).sum(), args)
    torch.testing.assert_close(o, ref_o)
    torch.testing.assert_close(t, ref_t)
    for a, b in zip(g, ref_g):
        torch.testing.assert_close(a, b)


# 7 token groups: ff_v, rem_v, ff_d, rem_d, ff_p, rem_p, action
SIZES = [5, 9, 3, 6, 5, 9, 4]
FULL_JOINT = [  # anchors -> anchors; futures -> all visual; action -> all
    [1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1]]


def test_full_joint_mask_is_three_groups_and_exact():
    mask = block_mask(SIZES, FULL_JOINT)
    groups, _ = faj._row_groups(mask)
    assert len(groups) == 3
    check(mask)


def test_flex_like_regime_and_fully_masked_rows():
    allowed = [row[:] for row in FULL_JOINT]
    allowed[1][3] = allowed[3][1] = 0          # XOR drop: rem_v <-> rem_d
    allowed[6][3] = 0                          # action does not see rem_d
    for j in range(7):                         # pointmap absent: rows and columns killed
        allowed[4][j] = allowed[5][j] = allowed[j][4] = allowed[j][5] = 0
    check(block_mask(SIZES, allowed))


def test_per_sample_masks_in_a_batch():
    a = block_mask(SIZES, FULL_JOINT)
    allowed = [row[:] for row in FULL_JOINT]
    allowed[6] = [1, 0, 1, 0, 1, 0, 1]         # action sees anchors only (base mask)
    b = block_mask(SIZES, allowed)
    check(torch.stack([a, b])[:, None], batch=2)


def test_broadcast_cross_attention_mask():
    """[B,1,1,Skv] masks (cross-attention) broadcast over every query row."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 7, 8, dtype=torch.float64, requires_grad=True)
    k, v, tq, tk, tv = (torch.randn(1, 2, 5, 8, dtype=torch.float64, requires_grad=True) if i != 2 else
                        torch.randn(1, 2, 7, 8, dtype=torch.float64, requires_grad=True) for i in range(5))
    mask = torch.tensor([True, True, False, True, True])[None, None, None]
    ref = explicit_jvp(q, k, v, tq, tk, tv, mask=mask)
    with patch.object(faj.FusedAttentionJVP, "apply", Explicit.apply):
        out = faj.attention_jvp(q, k, v, tq, tk, tv, attn_mask=mask)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b)


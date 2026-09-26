"""Fused attention-JVP backend (helpers/jvp_attention): grouping exactness and routing.

The Triton kernels need a GPU; here the fused call is replaced by an explicit
FP64 attention JVP, so these tests check the mask row-grouping, gather/scatter,
backward through the tangent, and the helpers/attention.py backend switch.
GPU validation of the kernels themselves: jvp_kernel_analysis/validate_tvm.py.
"""
import math
from unittest.mock import patch

import pytest
import torch
from torch.autograd import forward_ad as fw

from flexpi.models.helpers import attention as attn
from flexpi.models.helpers import jvp_attention as ja
from flexpi.models.helpers.flowmap import FlowMapConfig


def explicit_jvp(q, k, v, tq, tk, tv, mask=None):
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    ds = (tq @ k.transpose(-2, -1) + q @ tk.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    empty = torch.isneginf(s).all(-1, keepdim=True)
    p = torch.softmax(torch.where(empty, torch.zeros_like(s), s), -1)
    p = torch.where(empty, torch.zeros_like(p), p)
    dp = p * (ds - (p * ds).sum(-1, keepdim=True))
    return p @ v, dp @ v + p @ tv


class Explicit(torch.autograd.Function):
    """Stand-in for FusedAttentionJVP: same signature, explicit math, full backward."""

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


def block_mask(sizes, allowed):
    edges = [0]
    for n in sizes:
        edges.append(edges[-1] + n)
    mask = torch.zeros(edges[-1], edges[-1], dtype=torch.bool)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if allowed[i][j]:
                mask[edges[i]:edges[i + 1], edges[j]:edges[j + 1]] = True
    return mask


# 7 token groups: ff_v, rem_v, ff_d, rem_d, ff_p, rem_p, action
SIZES = [5, 9, 3, 6, 5, 9, 4]
FULL_JOINT = [[1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 1, 1]]


def check(mask, batch=1, heads=2, dim=8):
    torch.manual_seed(0)
    S = mask.shape[-1]
    args = [torch.randn(batch, heads, S, dim, dtype=torch.float64, requires_grad=True) for _ in range(6)]
    wo, wt = (torch.randn(batch, heads, S, dim, dtype=torch.float64) for _ in range(2))
    ref = explicit_jvp(*args, mask=mask if mask.dim() == 2 else mask.reshape(batch, 1, S, S))
    ref_g = torch.autograd.grad((ref[0] * wo).sum() + (ref[1] * wt).sum(), args)
    with patch.object(ja.FusedAttentionJVP, "apply", Explicit.apply):
        o, t = ja.attention_jvp(*args, attn_mask=mask)
        g = torch.autograd.grad((o * wo).sum() + (t * wt).sum(), args)
    torch.testing.assert_close(o, ref[0])
    torch.testing.assert_close(t, ref[1])
    for a, b in zip(g, ref_g):
        torch.testing.assert_close(a, b)


def test_full_joint_mask_is_three_groups_and_exact():
    mask = block_mask(SIZES, FULL_JOINT)
    assert len(ja._row_groups(mask)[0]) == 3
    check(mask)


def test_flex_regime_with_absent_stream_is_exact():
    allowed = [row[:] for row in FULL_JOINT]
    allowed[1][3] = allowed[3][1] = 0          # XOR drop: rem_v <-> rem_d
    allowed[6][3] = 0                          # action does not see rem_d
    for j in range(7):                         # pointmap absent: rows and columns killed
        allowed[4][j] = allowed[5][j] = allowed[j][4] = allowed[j][5] = 0
    check(block_mask(SIZES, allowed))


def test_per_sample_masks_in_a_batch():
    allowed = [row[:] for row in FULL_JOINT]
    allowed[6] = [1, 0, 1, 0, 1, 0, 1]         # action sees anchors only (base mask)
    check(torch.stack([block_mask(SIZES, FULL_JOINT), block_mask(SIZES, allowed)])[:, None], batch=2)


def test_broadcast_cross_attention_mask():
    torch.manual_seed(0)
    q, tq = (torch.randn(1, 2, 7, 8, dtype=torch.float64) for _ in range(2))
    k, v, tk, tv = (torch.randn(1, 2, 5, 8, dtype=torch.float64) for _ in range(4))
    mask = torch.tensor([True, True, False, True, True])[None, None, None]
    ref = explicit_jvp(q, k, v, tq, tk, tv, mask=mask)
    with patch.object(ja.FusedAttentionJVP, "apply", Explicit.apply):
        out = ja.attention_jvp(q, k, v, tq, tk, tv, attn_mask=mask)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b)


def _dual_run(q, k, v, tangents, mask):
    """Helper-level reverse-over-forward: loss on (primal, tangent) of the attention output."""
    leaves = [x.clone().requires_grad_() for x in (q, k, v, *tangents)]
    with fw.dual_level():
        out = attn.scaled_dot_product_attention(
            *(fw.make_dual(p, t) for p, t in zip(leaves[:3], leaves[3:])), attn_mask=mask)
        o, to = fw.unpack_dual(out)
        loss = o.square().sum() + to.square().sum()
    return o.detach(), to.detach(), torch.autograd.grad(loss, leaves)


@pytest.fixture
def restore_backend():
    yield
    attn.set_jvp_attention_backend("explicit")


def test_backend_switch_routes_forward_ad_and_matches_explicit(restore_backend):
    torch.manual_seed(1)
    mask = block_mask(SIZES, FULL_JOINT)[None, None]
    q, k, v, tq, tk, tv = (torch.randn(1, 2, mask.shape[-1], 8, dtype=torch.float64) for _ in range(6))
    ref = _dual_run(q, k, v, (tq, tk, tv), mask)                 # explicit backend
    attn.set_jvp_attention_backend("tvm")
    with patch.object(ja, "supported", return_value=True), \
         patch.object(ja.FusedAttentionJVP, "apply", side_effect=Explicit.apply) as fused:
        out = _dual_run(q, k, v, (tq, tk, tv), mask)
    assert fused.call_count == 3                                 # one mask-free call per row group
    for a, b in zip(out[:2] + tuple(out[2]), ref[:2] + tuple(ref[2])):
        torch.testing.assert_close(a, b)


def test_unsupported_calls_fall_back_to_explicit(restore_backend):
    attn.set_jvp_attention_backend("tvm")
    q, k, v, tq, tk, tv = (torch.randn(1, 2, 6, 8, dtype=torch.float32) for _ in range(6))
    with patch.object(ja.FusedAttentionJVP, "apply", side_effect=AssertionError("CPU fp32 must not use the kernel")):
        _dual_run(q, k, v, (tq, tk, tv), None)
    assert not ja.supported(q) and not ja.supported(q.bfloat16())   # CPU tensors


def test_config_validates_backend():
    assert FlowMapConfig(jvp_attention="tvm").jvp_attention == "tvm"
    with pytest.raises(ValueError, match="jvp_attention"):
        FlowMapConfig(jvp_attention="flash")
    with pytest.raises(ValueError):
        attn.set_jvp_attention_backend("flash")

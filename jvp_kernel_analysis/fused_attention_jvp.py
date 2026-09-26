"""Evaluation shim: the fused attention JVP now lives in the shared core package.

Implementation: ``flowmap_core.jvp_attention`` (``flowmap_core/src/flowmap_core/jvp_attention``;
FlexPi's ``flexpi.models.helpers.jvp_attention`` is an alias of it; selected in training
by ``flow_map.jvp_attention: tvm``). This module re-exports it so
``validate_tvm.py`` and ``test_grouping.py`` exercise exactly the shipped code.
"""
import torch.nn.functional as F
from torch.autograd import forward_ad as fw

from flowmap_core.jvp_attention import (  # noqa: F401
    FusedAttentionJVP, RowGroups, _grouped, _make_plan, _plan, _row_groups, attention_jvp,
    dual_attention, supported,
)


def scaled_dot_product_attention(q, k, v, attn_mask=None):
    """Drop-in for ``wan_video_dit.scaled_dot_product_attention`` forcing the fused JVP."""
    if fw._current_level < 0:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return dual_attention(q, k, v, attn_mask=attn_mask)

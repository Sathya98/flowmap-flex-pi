"""SDPA for ordinary and forward-AD (dual-number) forwards.

Ordinary forwards use fused SDPA. Under forward AD the backend is chosen by
``flow_map.jvp_attention`` (``set_jvp_attention_backend``):

- ``explicit`` (default): out-of-place FP32 attention that materialises L×L
  (PyTorch 2.7's math SDPA JVP can fail on backward through its tangent).
- ``tvm``: fused Triton attention JVP with backward through the tangent
  (``flowmap_core.jvp_attention``); calls it cannot handle (CPU, fp32, head dim > 128,
  additive masks) fall back to ``explicit``.
"""
import math

import torch
import torch.nn.functional as F

JVP_BACKENDS = ("explicit", "tvm")
_jvp_backend = "explicit"


def set_jvp_attention_backend(name):
    """Process-wide attention backend for forward-AD forwards (student and teacher alike)."""
    global _jvp_backend
    if name not in JVP_BACKENDS:
        raise ValueError(f"jvp_attention must be one of {JVP_BACKENDS}, got {name!r}")
    _jvp_backend = name


def get_jvp_attention_backend():
    return _jvp_backend


def scaled_dot_product_attention(q, k, v, attn_mask=None):
    if torch.autograd.forward_ad._current_level < 0:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    if _jvp_backend == "tvm":
        from . import jvp_attention
        if jvp_attention.supported(q, attn_mask):
            return jvp_attention.dual_attention(q, k, v, attn_mask=attn_mask)
    return _explicit_attention(q, k, v, attn_mask)


def _explicit_attention(q, k, v, attn_mask=None):
    dtype = q.dtype
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    q, k, v = (x.to(compute_dtype) for x in (q, k, v))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if attn_mask is not None:
        scores = (scores.masked_fill(~attn_mask, float('-inf'))
                  if attn_mask.dtype == torch.bool else scores + attn_mask)
    # Match SDPA's zero output for entirely masked rows, without NaN gradients.
    empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
    scores = torch.where(empty, torch.zeros_like(scores), scores)
    # torch.softmax's forward-AD rule itself uses an in-place operation in
    # PyTorch 2.7. Keep its normalization explicit as well.
    weights = torch.exp(scores - scores.detach().amax(dim=-1, keepdim=True))
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = torch.where(empty, torch.zeros_like(weights), weights)
    return (weights @ v).to(dtype)

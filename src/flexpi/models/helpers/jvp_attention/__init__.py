"""Fused attention JVP with backward through the tangent (TVM Triton kernels).

Selected by ``flow_map.jvp_attention: tvm`` (``helpers/attention.py`` routes
forward-AD attention here). The vendored kernels (``tvm/``, CC BY-NC-SA 4.0,
unmodified) compute attention o = softmax(q kᵀ/√d) v and its JVP tō for tangents
(tq, tk, tv) in one pass, and backprop a loss through BOTH outputs to all six
inputs without materialising L×L. They take no mask, so a masked attention is
split into row groups: queries whose mask rows are identical see the same keys,
and each group is one mask-free call on the gathered rows/columns (exact; no
wasted work). Our masks are built from a few token groups: 3 row groups in the
full-joint flow-map configs, at most 7 per sample under flex-joint regimes.

Validation (1 H100, jvp_kernel_analysis/, runs/diagnostics/tvm_validation_*):
errors vs FP64 ≤ 7e-3 (explicit FP32 path: ~2.3e-3); full LMD step grads vs the
explicit path cosine 0.99997; per call 6.7 ms vs 25.6 ms, +0.27 vs +2.86 GiB.
"""
import math
import weakref

import torch
import torch.nn.functional as F
from torch.autograd import forward_ad as fw


def _kernels():
    # Imported lazily: Triton's autotuner needs a GPU driver at import time.
    from .tvm.flash_attn_triton import _flash_attn_backward as primal_backward
    from .tvm.flash_jvp_backward import _flash_attn_backward as tangent_backward
    from .tvm.ryu_triton import flash_attention_jvp_multihead_triton_kernel_wrapper as jvp_forward
    return jvp_forward, primal_backward, tangent_backward


def supported(q, attn_mask=None):
    """Whether the fused kernels handle this call (else use the explicit path)."""
    return (q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and q.shape[-1] <= 128
            and (attn_mask is None or attn_mask.dtype == torch.bool))


class FusedAttentionJVP(torch.autograd.Function):
    """(q, k, v, tq, tk, tv) → (o, tō), all [B, H, S, D]; Sq may differ from Skv."""

    @staticmethod
    def forward(ctx, q, k, v, tq, tk, tv):
        q, k, v, tq, tk, tv = (x.contiguous() for x in (q, k, v, tq, tk, tv))
        B, H, S, _ = q.shape
        S_kv = k.shape[2]
        S_up = math.ceil(S / 128) * 128
        # Same buffer initialisation as TVM's SDPAFunction/SDPAJVPForwardFunction:
        # rows past S (the 128 padding) must hold benign statistics for the backward.
        M = torch.full((B, H, S_up), math.log(S_kv), device=q.device, dtype=torch.float32)
        MU = torch.zeros((B, H, S_up), device=q.device, dtype=torch.float32)
        LI = torch.full((B, H, S_up), float(S_kv), device=q.device, dtype=torch.float32)
        jvp_forward, _, _ = _kernels()
        with torch.no_grad():
            o, to, M, MU, LI = jvp_forward(q, k, v, tq, tk, tv, M=M, MU=MU, LI=LI, return_M=True)
        ctx.save_for_backward(q, k, v, tq, tk, tv, o, M, MU, LI)
        ctx.set_materialize_grads(False)
        return o, to

    @staticmethod
    def backward(ctx, do, dto):
        q, k, v, tq, tk, tv, o, M, MU, LI = ctx.saved_tensors
        # The backward kernels take [B, S, H, D] views (stride(-1) == 1).
        f = lambda x: x.transpose(1, 2)
        grads = [None] * 6
        _, primal_backward, tangent_backward = _kernels()
        with torch.no_grad():
            if do is not None:
                dq, dk, dv = (torch.empty_like(f(x)) for x in (q, k, v))
                primal_backward(f(do.contiguous()), f(q), f(k), f(v), f(o), M, dq, dk, dv,
                                bias=None, causal=False, softmax_scale=None)
                grads[:3] = [f(x) for x in (dq, dk, dv)]
            if dto is not None:
                out = [torch.empty_like(f(x)) for x in (q, k, v, tq, tk, tv)]
                tangent_backward(f(dto.contiguous()), f(q), f(k), f(v), f(o), f(tq), f(tk), f(tv),
                                 M, MU, LI, *out)
                out = [f(x) for x in out]
                grads[:3] = [g if a is None else a + g for a, g in zip(grads[:3], out[:3])]
                grads[3:] = out[3:]
        return tuple(grads)


class RowGroups:
    """Mask-free decomposition of a [Sq, Skv] boolean mask, cached by content."""

    def __init__(self, size=8):
        self.size, self._cache = size, []

    def __call__(self, mask):
        for cached, groups in self._cache:
            if cached.shape == mask.shape and torch.equal(cached, mask):
                return groups
        uniq, inverse = torch.unique(mask.to(torch.uint8), dim=0, return_inverse=True)
        found = []
        for g in range(uniq.shape[0]):
            rows = torch.nonzero(inverse == g).flatten()
            cols = torch.nonzero(uniq[g]).flatten()
            found.append((rows, None if cols.numel() == mask.shape[1] else cols))
        groups = (found, torch.argsort(torch.cat([rows for rows, _ in found])))
        self._cache = [(mask.clone(), groups)] + self._cache[:self.size - 1]
        return groups


_row_groups = RowGroups()


def _grouped(q, k, v, tq, tk, tv, plan):
    """Masked attention JVP for inputs sharing one row-group plan (None: no mask)."""
    if plan is None:
        return FusedAttentionJVP.apply(q, k, v, tq, tk, tv)
    groups, inverse = plan
    outs, touts = [], []
    for rows, cols in groups:
        sel = lambda x: x.index_select(2, rows)
        if cols is not None and cols.numel() == 0:   # fully masked rows: zero, as in the explicit path
            o_g = torch.zeros_like(sel(q))
            outs.append(o_g)
            touts.append(torch.zeros_like(o_g))
            continue
        kv = (k, v, tk, tv) if cols is None else tuple(x.index_select(2, cols) for x in (k, v, tk, tv))
        o_g, to_g = FusedAttentionJVP.apply(sel(q), kv[0], kv[1], sel(tq), kv[2], kv[3])
        outs.append(o_g)
        touts.append(to_g)
    o = torch.cat(outs, dim=2).index_select(2, inverse)
    to = torch.cat(touts, dim=2).index_select(2, inverse)
    return o, to


def _make_plan(attn_mask, batch, seq_q):
    """("batch", groups) if every sample shares the mask, else ("each", [groups per sample])."""
    if attn_mask.dim() == 2:
        return "batch", _row_groups(attn_mask.expand(seq_q, -1))
    masks = attn_mask.reshape(attn_mask.shape[0], *attn_mask.shape[-2:])
    masks = masks.expand(batch if masks.shape[0] == 1 else -1, seq_q, -1)    # broadcast rows
    if masks.shape[0] == 1 or bool((masks == masks[:1]).all()):
        return "batch", _row_groups(masks[0])
    return "each", [_row_groups(m) for m in masks]


# Plans are cached per mask *tensor*: the model builds a mask once per forward and
# passes (views of) it to every layer, so lookups need no GPU sync. Keyed by the
# view's base tensor (flash_attention re-views the mask on every call), guarded by
# a weak reference so a freed mask is never matched by a recycled id.
_plans = {}


def _plan(attn_mask, batch, seq_q):
    base = attn_mask._base if attn_mask._base is not None else attn_mask
    ref, entry = _plans.get(id(base), (None, None))
    if ref is None or ref() is not base:
        if len(_plans) > 64:
            for key in [k for k, (r, _) in _plans.items() if r() is None]:
                del _plans[key]
        entry = {}
        _plans[id(base)] = (weakref.ref(base), entry)
    key = (tuple(attn_mask.shape), attn_mask.stride(), attn_mask.storage_offset(), base._version, batch, seq_q)
    if key not in entry:
        entry[key] = _make_plan(attn_mask, batch, seq_q)
    return entry[key]


def attention_jvp(q, k, v, tq, tk, tv, attn_mask=None):
    """Explicit primal/tangent form; attn_mask None, [Sq,Skv], [B,1,Sq,Skv] or [B,Sq,Skv] (True = attend)."""
    if attn_mask is None:
        return _grouped(q, k, v, tq, tk, tv, None)
    if attn_mask.dtype != torch.bool:
        raise TypeError("fused attention JVP supports boolean masks only")
    kind, plan = _plan(attn_mask, q.shape[0], q.shape[2])
    if kind == "batch":
        return _grouped(q, k, v, tq, tk, tv, plan)            # one mask for the batch: batched calls
    parts = [_grouped(*(x[b:b + 1] for x in (q, k, v, tq, tk, tv)), plan[b]) for b in range(q.shape[0])]
    return tuple(torch.cat(p, dim=0) for p in zip(*parts))


def dual_attention(q, k, v, attn_mask=None):
    """Attention on forward-AD duals: fused SDPA if nothing carries a tangent, else the fused JVP."""
    (qp, qt), (kp, kt), (vp, vt) = (fw.unpack_dual(x) for x in (q, k, v))
    if qt is None and kt is None and vt is None:
        return F.scaled_dot_product_attention(qp, kp, vp, attn_mask=attn_mask)
    tangents = [t if t is not None else torch.zeros_like(p) for p, t in ((qp, qt), (kp, kt), (vp, vt))]
    o, to = attention_jvp(qp, kp, vp, *tangents, attn_mask=attn_mask)
    return fw.make_dual(o, to)

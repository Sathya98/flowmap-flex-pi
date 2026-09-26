#!/usr/bin/env python
"""GPU validation of the TVM fused attention-JVP kernels for full-grad LMD/LSD.

Stages (``--skip`` any of kernel,real,step,grad):
  kernel  synthetic LIBERO-sized block masks (full joint = 3 row groups; a flex-like
          regime): TVM grouped vs the current explicit FP32 path, both against an
          FP64 reference. Primal, tangent and all six gradients of a random loss
          through (o, tō).
  real    the longest real joint attention, captured from the model: same errors,
          plus fwd+bwd time and peak memory of explicit FP32 vs TVM vs fused SDPA.
  step    whole training microsteps, explicit vs TVM backend, microbatch 1 and 2,
          for lmd_full, lsd_off, esd_off, lmd_semigrad: time, peak memory or OOM.
  grad    lmd_full, one microstep, same seed: loss and full parameter-gradient
          agreement (relative error, cosine) of TVM vs explicit, calibrated
          against explicit vs explicit (kernel nondeterminism).

The backend is switched by patching ``wan_video_dit.scaled_dot_product_attention``,
which every attention call (MoT joint, HBridge per-stream, cross-attention, action
expert) goes through. Nothing in src/ is modified.

    python jvp_kernel_analysis/validate_tvm.py --out runs/diagnostics/tvm_validation_x
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.autograd import forward_ad as fw
from torch.utils.data import default_collate

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import fused_attention_jvp as faj  # noqa: E402

spec = importlib.util.spec_from_file_location("profile_step", REPO / "scripts" / "profile_flowmap_step.py")
prof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prof)
from flexpi.models import wan_video_dit  # noqa: E402
from flexpi.models.helpers.attention import scaled_dot_product_attention as explicit_sdpa  # noqa: E402

log, GiB = prof.log, 2 ** 30
ORIGINAL_SDPA = wan_video_dit.scaled_dot_product_attention
STEP_MODES = ("lmd_full", "lsd_off", "esd_off", "lmd_semigrad")
# LIBERO full-joint token groups: ff_v, rem_v, ff_d, rem_d, ff_p, rem_p, action
LIBERO_SIZES = [224, 448, 147, 147, 224, 448, 32]
FULL_JOINT = [[1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 1, 1]]


def set_backend(name):
    wan_video_dit.scaled_dot_product_attention = faj.scaled_dot_product_attention if name == "tvm" else ORIGINAL_SDPA


def block_mask(sizes, allowed, device):
    edges = np.cumsum([0] + sizes)
    mask = torch.zeros(edges[-1], edges[-1], dtype=torch.bool, device=device)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if allowed[i][j]:
                mask[edges[i]:edges[i + 1], edges[j]:edges[j + 1]] = True
    return mask


def reference_jvp(q, k, v, tq, tk, tv, mask):
    """Masked attention + analytic JVP in the inputs' dtype (FP64 for the reference)."""
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    ds = (tq @ k.transpose(-2, -1) + q @ tk.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask, float("-inf")) if mask is not None else s
    empty = torch.isneginf(s).all(-1, keepdim=True)
    p = torch.softmax(torch.where(empty, torch.zeros_like(s), s), -1)
    p = torch.where(empty, torch.zeros_like(p), p)
    dp = p * (ds - (p * ds).sum(-1, keepdim=True))
    return p @ v, dp @ v + p @ tv


def explicit_jvp(q, k, v, tq, tk, tv, mask):
    """The production path: helpers/attention.py under forward AD (explicit FP32)."""
    with fw.dual_level():
        out = explicit_sdpa(*(fw.make_dual(p, t) for p, t in zip((q, k, v), (tq, tk, tv))), attn_mask=mask)
        o, to = fw.unpack_dual(out)
    return o, to


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def compare(q, k, v, mask, seed=0):
    """Errors of explicit-FP32 and TVM against FP64: o, tō and the six input gradients."""
    gen = torch.Generator(device=q.device).manual_seed(seed)
    tq, tk, tv = (torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * x.float().std().to(x.dtype)
                  for x in (q, k, v))
    wo = torch.randn(q.shape, generator=gen, device=q.device, dtype=torch.float64)
    wt = torch.randn(q.shape, generator=gen, device=q.device, dtype=torch.float64)
    m4 = None if mask is None else mask.reshape(1, 1, *mask.shape[-2:])

    def run(fn, dtype):
        leaves = [x.detach().to(dtype).requires_grad_() for x in (q, k, v, tq, tk, tv)]
        o, to = fn(*leaves, m4)
        grads = torch.autograd.grad((o.double() * wo).sum() + (to.double() * wt).sum(), leaves)
        return o.detach(), to.detach(), grads

    ref = run(reference_jvp, torch.float64)
    out = {}
    for name, fn in (("explicit_fp32", explicit_jvp), ("tvm", lambda *a: faj.attention_jvp(*a[:6], attn_mask=a[6]))):
        o, to, grads = run(fn, q.dtype)
        out[name] = {"o": rel(o, ref[0]), "tangent": rel(to, ref[1]),
                     **{f"d{n}": rel(g, r) for n, g, r in zip(("q", "k", "v", "tq", "tk", "tv"), grads, ref[2])}}
    return out


def time_call(fn, warmup=3, repeats=10):
    return prof.bench(fn, warmup, repeats)


def stage_kernel(device):
    torch.manual_seed(0)
    results = {}
    flex = [row[:] for row in FULL_JOINT]
    flex[1][3] = flex[3][1] = 0     # XOR drop rem_v <-> rem_d
    flex[6][3] = 0                  # action does not see rem_d
    for name, allowed in (("full_joint", FULL_JOINT), ("flex_regime", flex)):
        mask = block_mask(LIBERO_SIZES, allowed, device)
        q, k, v = (torch.randn(1, 24, mask.shape[0], 128, device=device, dtype=torch.bfloat16) for _ in range(3))
        results[name] = {"groups": len(faj._row_groups(mask)[0]), **compare(q, k, v, mask)}
        log(f"  kernel {name}: {json.dumps(results[name])}")
    return results


def stage_real(model, batch, diag):
    cap = prof.capture_attention(model, batch, diag)
    q, k, v = (cap[n].reshape(cap[n].shape[0], cap[n].shape[1], cap["heads"], -1).transpose(1, 2).contiguous()
               for n in "qkv")
    mask = cap["mask"]
    m2 = mask.reshape(-1, *mask.shape[-2:])[0] if mask is not None else None
    res = {"L": int(cap["L"]), "groups": len(faj._row_groups(m2)[0]) if m2 is not None else 1,
           "errors": compare(q, k, v, m2)}
    tq, tk, tv = (torch.randn_like(x) * x.float().std().to(x.dtype) for x in (q, k, v))
    m4 = None if m2 is None else m2.reshape(1, 1, *m2.shape)

    def fwd_bwd(fn):
        def run():
            leaves = [x.detach().requires_grad_() for x in (q, k, v, tq, tk, tv)]
            o, to = fn(*leaves, m4)
            (o.float().square().mean() + to.float().square().mean()).backward()
        return run

    def fused(*a):
        o = torch.nn.functional.scaled_dot_product_attention(a[0], a[1], a[2], attn_mask=a[6])
        return o, o * 0
    res["timing"] = {name: time_call(fwd_bwd(fn)) for name, fn in (
        ("fused_sdpa_fwd_bwd", fused), ("explicit_fp32_jvp_fwd_bwd", explicit_jvp),
        ("tvm_jvp_fwd_bwd", lambda *a: faj.attention_jvp(*a[:6], attn_mask=a[6])))}
    log(f"  real L={res['L']} groups={res['groups']} errors={json.dumps(res['errors'])}")
    for name, t in res["timing"].items():
        log(f"  real {name:28s} {t['ms']:8.2f} ms  +{t['peak_delta_gib']:.2f} GiB")
    return res


def stage_step(model, base_flow, batches1, batches2, warmup, repeats):
    results = {}
    for backend in ("explicit", "tvm"):
        set_backend(backend)
        for mb, batches in ((1, batches1), (2, batches2)):
            key = f"{backend}_mb{mb}"
            results[key] = prof.time_steps(model, base_flow, batches, STEP_MODES, warmup, repeats)
            log(f"  step {key}: " + ", ".join(
                f"{m} {r.get('step_s', float('nan')):.2f}s/{r.get('peak_gib', float('nan')):.1f}GiB"
                if "error" not in r else f"{m} {r['error']}" for m, r in results[key].items()))
    set_backend("explicit")
    return results


def grads_of(model, base_flow, batch, seed):
    prof.set_mode(model, base_flow, "lmd_full")
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model.training_loss(batch)
    loss.backward()
    model.flow_map = base_flow
    grads = {n: p.grad.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return float(loss), grads


def agreement(a, b):
    dot = na = nb = diff = 0.0
    for name, ga in a.items():
        ga, gb = ga.double(), b[name].double()
        dot += float((ga * gb).sum())
        na += float(ga.square().sum())
        nb += float(gb.square().sum())
        diff += float((ga - gb).square().sum())
    return {"rel_err": math.sqrt(diff / max(nb, 1e-30)), "cosine": dot / math.sqrt(max(na * nb, 1e-30)),
            "norm_ratio": math.sqrt(na / max(nb, 1e-30))}


def stage_grad(model, base_flow, batch, seed=1234):
    set_backend("explicit")
    loss_a, ref = grads_of(model, base_flow, batch, seed)
    loss_b, again = grads_of(model, base_flow, batch, seed)
    set_backend("tvm")
    loss_t, tvm = grads_of(model, base_flow, batch, seed)
    set_backend("explicit")
    res = {"loss": {"explicit": loss_a, "explicit_repeat": loss_b, "tvm": loss_t},
           "explicit_vs_explicit": agreement(again, ref), "tvm_vs_explicit": agreement(tvm, ref)}
    log(f"  grad {json.dumps(res)}")
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="flowmap_libero_lmd_full")
    parser.add_argument("--cache", default="data/latent_cache/libero_fulljoint_v2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip", default="")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=4)
    args, overrides = parser.parse_known_args()
    skip = set(filter(None, args.skip.split(",")))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {"gpu": torch.cuda.get_device_name(), "config": args.config_name, "overrides": overrides}

    def save():
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    if "kernel" not in skip:
        log("kernel: synthetic LIBERO-sized masks, 24 heads × 128, bf16")
        results["kernel"] = stage_kernel("cuda")
        save()
    if not {"real", "step", "grad"} <= skip:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from flexpi.datasets.latent_cache import CachedLatentDataset
        from flexpi.utils import misc
        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
            cfg = compose(config_name=args.config_name, overrides=overrides)
        misc.register_work_dir(out)
        t0 = time.perf_counter()
        model = prof.build_model(cfg)
        log(f"student + teacher built in {time.perf_counter() - t0:.0f}s")
        cached = CachedLatentDataset(args.cache, instantiate(cfg.data.train), require_complete=False)
        ready = np.flatnonzero(cached.cache.ready())
        picks = np.random.default_rng(0).choice(ready, 8, replace=False)
        batches1 = [default_collate([cached[int(p)]]) for p in picks[:4]]
        batches2 = [default_collate([cached[int(p)] for p in picks[i:i + 2]]) for i in (0, 2, 4, 6)]
        base_flow = model.flow_map
        if "real" not in skip:
            log("real: longest joint attention captured from an LSD-diagonal forward")
            results["real"] = stage_real(model, batches1[0], prof.set_mode(model, base_flow, "lsd_diag"))
            model.flow_map = base_flow
            save()
        if "step" not in skip:
            log(f"step: {STEP_MODES}, explicit vs tvm, microbatch 1 and 2")
            results["step"] = stage_step(model, base_flow, batches1, batches2, args.warmup, args.repeats)
            save()
        if "grad" not in skip:
            log("grad: lmd_full parameter-gradient agreement")
            results["grad"] = stage_grad(model, base_flow, batches1[0])
            save()
    log(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()

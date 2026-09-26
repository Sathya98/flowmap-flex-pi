#!/usr/bin/env python
"""Profile where one flow-map training microstep spends time and memory (1 GPU).

Answers three questions from .claude/context/07-efficiency-notes.md:

1. inputs:  how much of the per-example floor is data decoding + frozen encoders,
            versus reading the latent cache (single process, no workers).
2. attention (micro): one real joint-layer attention call, captured from the
            model: fused SDPA vs the explicit forward-AD path in
            helpers/attention.py (FP32), with/without tangents and backward, plus
            TF32 and bf16 variants (tangent error reported against FP32).
3. step (macro): forward (incl. JVP) and backward of one microstep, batch 1,
            per objective: plain FM, LSD diagonal, PFMM-16, LMD full / detached /
            semigradient, LSD and ESD off-diagonal. One torch.profiler table for
            the --profile-mode step.

Excluded: optimizer step, EMA, ZeRO communication and dataloader workers; this
is the per-example compute of one GPU. The model and teacher are built as the
trainer builds them (strict checkpoint load, frozen teacher outside the module
tree, configure_trainable); each objective is a variant of the config's
flow_map block, so one model load serves every mode.

    python scripts/profile_flowmap_step.py --config-name flowmap_libero_lmd_full \\
        --cache data/latent_cache/libero_fulljoint_v2 --out runs/diagnostics/profile_x
"""
import argparse
import dataclasses
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.autograd import forward_ad as fw
from torch.utils.data import default_collate

from flexpi.datasets.latent_cache import CachedLatentDataset
from flexpi.models import mot as mot_module
from flexpi.models import wan_video_dit
from flexpi.models.helpers.adaptation import clone_teacher, configure_trainable, validate_teacher_payload
from flexpi.utils import misc
from flexpi.utils.config_resolvers import register_default_resolvers

REPO = Path(__file__).resolve().parents[1]
register_default_resolvers()
GiB = 2 ** 30

# objective variants of the config's flow_map block; "diag" feeds the
# self-distillation mixture mask (True = plain-FM diagonal example)
MODES = {
    "fm_base": dict(enabled=False),
    "lsd_diag": dict(objective="lsd", lmd_teacher_gradient="detached", diag=True),
    "pfmm16": dict(objective="pfmm", lmd_teacher_gradient="detached", teacher_steps=16),
    "lmd_full": dict(objective="lmd", lmd_teacher_gradient="full", detach_derivatives=False),
    "lmd_detached": dict(objective="lmd", lmd_teacher_gradient="detached", detach_derivatives=False),
    "lmd_semigrad": dict(objective="lmd", lmd_teacher_gradient="detached", detach_derivatives=True),
    "lsd_off": dict(objective="lsd", lmd_teacher_gradient="detached", diag=False),
    "esd_off": dict(objective="esd", lmd_teacher_gradient="detached", diag=False),
}


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def sync():
    torch.cuda.synchronize()


def build_model(cfg):
    """Student + frozen teacher, as Wan22Trainer builds them."""
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda",
                        skip_dit_load_from_pretrain=True, load_text_encoder=False)
    model.load_checkpoint(str(cfg.pretrained_ckpt), optimizer=None, strict_shape=True)
    teacher = clone_teacher(model)
    payload = teacher.load_checkpoint(str(cfg.model.flow_map.teacher_checkpoint or cfg.pretrained_ckpt),
                                      strict_shape=True)
    validate_teacher_payload(teacher, payload)
    del payload
    teacher.to("cuda")
    teacher.eval().requires_grad_(False)
    object.__setattr__(model, "flow_map_teacher", teacher)
    configure_trainable(model)
    return model


def set_mode(model, base_flow, name):
    fields = dict(MODES[name])
    diag = fields.pop("diag", None)
    model.flow_map = dataclasses.replace(base_flow, **fields)
    if model.flow_map.uses_time_weighting and not hasattr(model, "flow_map_loss_weight"):
        from flexpi.models.helpers.flowmap_self import TimeLossWeight
        model.flow_map_loss_weight = TimeLossWeight().to("cuda")
    return diag


def with_mask(batch, diag):
    if diag is None:
        return batch
    # One mixture flag per example (every example of the batch on the same branch).
    return {**batch, "_flowmap_diagonal_mask": torch.full((batch["action"].shape[0],), diag)}


# --------------------------------------------------------------------------- inputs

def time_inputs(model, raw, cached, positions, windows):
    """Per-example cost of producing build_inputs' output: raw decode+encode vs cache."""
    out = {"raw_decode_s": [], "raw_encode_s": [], "cache_read_s": [], "cache_inputs_s": []}
    for p in positions:
        g = int(windows[p])
        t0 = time.perf_counter()
        batch = default_collate([raw._get(g)])
        t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.build_inputs(batch)
        sync()
        t2 = time.perf_counter()
        batch = default_collate([cached[p]])
        t3 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.build_inputs(batch)
        sync()
        t4 = time.perf_counter()
        for key, value in zip(out, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            out[key].append(value)
    return {k: float(np.median(v)) for k, v in out.items()} | {"n": len(positions)}


# ------------------------------------------------------------------------ attention

def capture_attention(model, batch, diag):
    """q/k/v/mask of the longest joint attention call in one no-grad forward."""
    seen = {}
    original = mot_module.flash_attention

    def spy(q, k, v, num_heads, ctx_mask=None, **kw):
        if q.shape[1] > seen.get("L", 0):
            seen.update(L=q.shape[1], q=q.detach().clone(), k=k.detach().clone(), v=v.detach().clone(),
                        mask=None if ctx_mask is None else ctx_mask.detach().clone(), heads=num_heads)
        return original(q, k, v, num_heads, ctx_mask=ctx_mask, **kw)

    mot_module.flash_attention = spy
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.training_loss(with_mask(batch, diag))
    finally:
        mot_module.flash_attention = original
    return seen


def explicit_attention(q, k, v, mask, compute_dtype):
    """helpers/attention.py's forward-AD path with a selectable compute dtype."""
    dtype = q.dtype
    q, k, v = (x.to(compute_dtype) for x in (q, k, v))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
    scores = torch.where(empty, torch.zeros_like(scores), scores)
    weights = torch.exp(scores - scores.detach().amax(dim=-1, keepdim=True))
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = torch.where(empty, torch.zeros_like(weights), weights)
    return (weights @ v).to(dtype)


def bench(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    sync()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        sync()
        times.append(start.elapsed_time(end))
    return {"ms": float(np.median(times)), "ms_min": float(np.min(times)),
            "peak_delta_gib": (torch.cuda.max_memory_allocated() - base) / GiB}


def micro_attention(cap, warmup, repeats):
    heads = cap["heads"]
    q, k, v = (rearrange(cap[n], "b s (h d) -> b h s d", h=heads) for n in "qkv")
    mask = cap["mask"]
    if mask is not None and mask.dim() == 3:
        mask = mask.unsqueeze(1)
    gen = torch.Generator(device="cuda").manual_seed(0)
    tq, tk, tv = (torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=gen) * x.std() for x in (q, k, v))
    results = {"L": int(cap["L"]), "heads": heads, "head_dim": int(q.shape[-1]),
               "mask_density": None if mask is None else float(mask.float().mean())}

    def fused(grad):
        def run():
            args = [x.detach().requires_grad_(grad) for x in (q, k, v)]
            with torch.set_grad_enabled(grad):
                out = torch.nn.functional.scaled_dot_product_attention(*args, attn_mask=mask)
                if grad:
                    out.float().square().mean().backward()
        return run

    def dual(compute_dtype, grad, tangents=True):
        def run():
            prim = [x.detach().requires_grad_(grad) for x in (q, k, v)]
            tan = [x.detach().requires_grad_(grad) for x in (tq, tk, tv)]
            with torch.set_grad_enabled(grad), fw.dual_level():
                args = [fw.make_dual(p, t) for p, t in zip(prim, tan)] if tangents else prim
                out = explicit_attention(*args, mask, compute_dtype)
                primal, tangent = fw.unpack_dual(out)
                if grad:
                    loss = primal.float().square().mean()
                    if tangent is not None:
                        loss = loss + tangent.float().square().mean()
                    loss.backward()
        return run

    cases = {
        "fused_sdpa_fwd": fused(False),
        "fused_sdpa_fwd_bwd": fused(True),
        "explicit_fp32_fwd": dual(torch.float32, False, tangents=False),
        "explicit_fp32_fwd_bwd": dual(torch.float32, True, tangents=False),
        "explicit_fp32_jvp_fwd": dual(torch.float32, False),
        "explicit_fp32_jvp_fwd_bwd": dual(torch.float32, True),
        "explicit_bf16_jvp_fwd_bwd": dual(torch.bfloat16, True),
    }
    for name, fn in cases.items():
        results[name] = bench(fn, warmup, repeats)
        log(f"  attention {name:28s} {results[name]['ms']:8.2f} ms  +{results[name]['peak_delta_gib']:.2f} GiB")
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        results["explicit_tf32_jvp_fwd_bwd"] = bench(dual(torch.float32, True), warmup, repeats)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
    log(f"  attention explicit_tf32_jvp_fwd_bwd     {results['explicit_tf32_jvp_fwd_bwd']['ms']:8.2f} ms")

    # Tangent accuracy of the cheaper variants against FP32.
    def tangent(compute_dtype, tf32=False):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            with torch.no_grad(), fw.dual_level():
                out = explicit_attention(*(fw.make_dual(p, t) for p, t in zip((q, k, v), (tq, tk, tv))), mask, compute_dtype)
                return fw.unpack_dual(out)[1].float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False

    ref = tangent(torch.float32)
    for name, t in (("bf16", tangent(torch.bfloat16)), ("tf32", tangent(torch.float32, tf32=True))):
        results[f"tangent_rel_err_{name}"] = float((t - ref).norm() / ref.norm())
    return results


# ----------------------------------------------------------------------------- step

def time_steps(model, base_flow, batches, modes, warmup, repeats):
    results = {}
    for name in modes:
        diag = set_mode(model, base_flow, name)
        fwd, bwd, peaks, deltas = [], [], [], []
        try:
            for i in range(warmup + repeats):
                batch = with_mask(batches[i % len(batches)], diag)
                sync()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = model.training_loss(batch)
                sync()
                t1 = time.perf_counter()
                loss.backward()
                sync()
                t2 = time.perf_counter()
                if i >= warmup:
                    fwd.append(t1 - t0)
                    bwd.append(t2 - t1)
                    peaks.append(torch.cuda.max_memory_allocated() / GiB)
                    deltas.append((torch.cuda.max_memory_allocated() - base) / GiB)
                del loss
            results[name] = {"fwd_s": float(np.median(fwd)), "bwd_s": float(np.median(bwd)),
                             "step_s": float(np.median(np.add(fwd, bwd))),
                             "peak_gib": float(max(peaks)), "activation_peak_gib": float(max(deltas))}
            r = results[name]
            log(f"  step {name:14s} fwd {r['fwd_s']:.3f}s bwd {r['bwd_s']:.3f}s = {r['step_s']:.3f}s  "
                f"peak {r['peak_gib']:.1f} GiB (+{r['activation_peak_gib']:.1f})")
        except torch.cuda.OutOfMemoryError as err:
            results[name] = {"error": "OOM", "detail": str(err)[:300]}
            log(f"  step {name}: OOM")
        except Exception as err:   # record and keep profiling the other modes
            results[name] = {"error": type(err).__name__, "detail": str(err)[:300]}
            log(f"  step {name}: {type(err).__name__}: {str(err)[:200]}")
        model.zero_grad(set_to_none=False)
        torch.cuda.empty_cache()
    model.flow_map = base_flow
    return results


def profile_step(model, base_flow, batch, name, out_dir):
    diag = set_mode(model, base_flow, name)
    batch = with_mask(batch, diag)
    original = wan_video_dit.scaled_dot_product_attention

    def labelled(*a, **kw):
        label = "attn_explicit_fwdad" if fw._current_level >= 0 else "attn_fused"
        with torch.profiler.record_function(label):
            return original(*a, **kw)

    for _ in range(2):   # warm the kernels outside the profile
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.training_loss(batch)[0].backward()
    wan_video_dit.scaled_dot_product_attention = labelled
    try:
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=acts) as prof:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.training_loss(batch)
            loss.backward()
            sync()
    finally:
        wan_video_dit.scaled_dot_product_attention = original
        model.flow_map = base_flow
    events = prof.key_averages()
    table = events.table(sort_by="self_device_time_total", row_limit=40)
    (out_dir / f"profiler_{name}.txt").write_text(table)
    total = sum(e.self_device_time_total for e in events)
    by_label = {e.key: e.device_time_total / 1e6 for e in events if e.key.startswith("attn_")}
    gemm = sum(e.self_device_time_total for e in events
               if e.key in ("aten::mm", "aten::bmm", "aten::addmm", "aten::baddbmm", "aten::_scaled_mm"))
    return {"mode": name, "device_total_s": total / 1e6, "gemm_share": gemm / max(total, 1),
            "attention_fwd_ranges_s": by_label}


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="flowmap_libero_lmd_full")
    parser.add_argument("--cache", default="data/latent_cache/libero_fulljoint_v2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--profile-mode", default="lmd_full")
    parser.add_argument("--samples", type=int, default=8, help="cached windows cycled through the step timing")
    parser.add_argument("--input-samples", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--skip", default="", help="comma list of inputs,attention,step,profile")
    args, overrides = parser.parse_known_args()
    skip = set(filter(None, args.skip.split(",")))
    modes = [m for m in args.modes.split(",") if m]
    unknown = set(modes) - set(MODES)
    if unknown:
        parser.error(f"unknown modes {sorted(unknown)}")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    torch.manual_seed(0)
    np.random.seed(0)
    gpu = torch.cuda.get_device_name()
    log(f"{gpu}; config {args.config_name}; cache {args.cache}")
    misc.register_work_dir(out_dir)
    t0 = time.perf_counter()
    model = build_model(cfg)
    log(f"student + teacher built in {time.perf_counter() - t0:.0f}s; "
        f"{torch.cuda.memory_allocated() / GiB:.1f} GiB allocated")
    raw = instantiate(cfg.data.train)
    cached = CachedLatentDataset(args.cache, raw, require_complete=False)
    ready = np.flatnonzero(cached.cache.ready())
    if len(ready) < max(args.samples, args.input_samples):
        raise SystemExit(f"only {len(ready)} cached windows are ready")
    rng = np.random.default_rng(0)
    positions = np.sort(rng.choice(ready, max(args.samples, args.input_samples), replace=False))
    windows = np.asarray(cached.cache.array("windows"))
    batches = [default_collate([cached[int(p)]]) for p in positions[:args.samples]]
    base_flow = model.flow_map
    results = {"gpu": gpu, "config": args.config_name, "cache": args.cache, "overrides": overrides,
               "positions": positions.tolist(), "created": datetime.datetime.now().isoformat()}

    def save():
        (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    if "inputs" not in skip:
        log("inputs: raw decode + frozen encoders vs latent cache (median per example)")
        results["inputs"] = time_inputs(model, raw, cached, positions[:args.input_samples], windows)
        log(f"  {results['inputs']}")
        save()
    if "attention" not in skip:
        # Capture from a plain (no-JVP) flow-map forward: same attention layout,
        # and q/k/v carry no tangents.
        log("attention: longest joint attention call, captured from an LSD-diagonal forward")
        cap = capture_attention(model, batches[0], set_mode(model, base_flow, "lsd_diag"))
        model.flow_map = base_flow
        results["attention"] = micro_attention(cap, warmup=3, repeats=10)
        del cap
        save()
    if "step" not in skip:
        log(f"step: batch 1, warmup {args.warmup}, repeats {args.repeats}")
        results["step"] = time_steps(model, base_flow, batches, modes, args.warmup, args.repeats)
        save()
    if "profile" not in skip:
        log(f"profile: torch.profiler on one {args.profile_mode} step")
        results["profile"] = profile_step(model, base_flow, batches[0], args.profile_mode, out_dir)
        log(f"  {results['profile']}")
        save()
    log(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

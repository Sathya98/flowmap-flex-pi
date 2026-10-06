#!/usr/bin/env python
"""Latent-space generation quality of flow-map checkpoints vs the flow-matching baseline.

Full joint generation (video, DINO, pointmap, action) on fixed latent-cache clips,
sampled exactly as the flow map is defined in training: start at sigma = 1 (noise,
clean first-frame anchors), jump through the NFE grid with x <- x + (t - s) v(x, s, t)
(``flowmap_training.predict_streams``), compare the final latents with the cached
ground truth. The FM baseline is the released checkpoint in the same model: its
jump input is zero-initialised, so v(x, s, t) = b_s(x) and the sampler is plain Euler.

Metrics per stream (future frames only; padded action steps excluded), over clips,
two samples per clip with fixed per-clip noise shared by every model:
- rmse: RMS error of one sample vs ground truth. Rewards blur (a mean prediction wins).
- energy: energy score E|X - y| - 1/2 E|X - X'| (RMS distances). A proper scoring rule:
  the true conditional distribution scores best, a blurry mean does not.
- spread: RMS distance between the two samples (0 for a deterministic/blurry sampler).

    python scripts/flowmap_generation_eval.py --out DIR --clips 128 \\
        --spec "fm=checkpoints/released/.../step_048060.pt@1,2,4,8,16,32" \\
        --spec "lsd_grid_s500_ema=runs/.../ema_0.995/step_000500.pt@1,2,3,4" ...
"""
import argparse
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from flexpi.datasets.latent_cache import CachedLatentDataset
from flexpi.models.helpers.flowmap import STREAMS, affine_flow_map
from flexpi.models.helpers.flowmap_training import predict_streams
from flexpi.utils import misc
from flexpi.utils.config_resolvers import register_default_resolvers

REPO = Path(__file__).resolve().parents[1]
register_default_resolvers()


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def zero_jump_input(model):
    """Payloads without jump-input weights (the FM release) must not inherit a previous load's."""
    for expert in (model.video_expert, model.action_expert):
        delta = getattr(expert, "time_embedding_delta", None)
        if delta is not None:
            torch.nn.init.zeros_(delta[-1].weight)
            torch.nn.init.zeros_(delta[-1].bias)


def jump_input_is_zero(model):
    return all(float(e.time_embedding_delta[-1].weight.abs().max()) == 0
               for e in (model.video_expert, model.action_expert))


@torch.no_grad()
def generate(model, inputs, steps, seed):
    """One joint sample per clip: noise at sigma=1 -> clean latents over ``steps`` jumps."""
    cfg = model.flow_map
    active = tuple(n for n in STREAMS if n in cfg.streams)
    clean = dict(video=inputs["input_latents"], dino=inputs["dino_features"],
                 pointmap=inputs["pointmap_raw"], action=inputs["action"])
    clean["video"] = torch.cat((inputs["first_frame_latents"], clean["video"][:, :, 1:]), dim=2)
    gen = torch.Generator(device=clean["action"].device).manual_seed(seed)
    x = {}
    for name in active:
        noise = torch.randn(clean[name].shape, generator=gen, device=clean[name].device, dtype=torch.float32)
        data = clean[name].float()
        if name != "action":          # the anchor frame stays clean (zero velocity), as in training
            noise = torch.cat((data[:, :, :1], noise[:, :, 1:]), dim=2)
        x[name] = noise
    b = clean["action"].shape[0]
    nodes = cfg.inference_nodes(steps, clean["action"].device)
    for i in range(steps):
        s = nodes[i].expand(b).contiguous()
        t = nodes[i + 1].expand(b).contiguous()
        full = dict(clean)
        full.update(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = predict_streams(model, full, s, t, inputs["context"], inputs["context_mask"],
                                inputs["fuse_vae_embedding_in_latents"], active)
        x = {name: affine_flow_map(x[name], v[name].float(), s, t) for name in active}
    return x, clean, active


def rms(a, b, valid=None):
    diff = (a - b).float().square()
    if valid is not None:
        diff = diff * valid
        return (diff.flatten(1).sum(1) / valid.expand_as(diff).flatten(1).sum(1).clamp_min(1)).sqrt()
    return diff.flatten(1).mean(1).sqrt()


def evaluate(model, batches, steps):
    per = {}
    for inputs in batches:
        x1, clean, active = generate(model, inputs, steps, seed=1000 + inputs["_seed"])
        x2, _, _ = generate(model, inputs, steps, seed=2000 + inputs["_seed"])
        for name in active:
            y = clean[name].float()
            a, b2, valid = x1[name], x2[name], None
            if name == "action":
                pad = inputs.get("action_is_pad")
                valid = None if pad is None else (~pad).float()[:, :, None]
            else:
                a, b2, y = a[:, :, 1:], b2[:, :, 1:], y[:, :, 1:]
            d1, d2, d12 = rms(a, y, valid), rms(b2, y, valid), rms(a, b2, valid)
            rec = per.setdefault(name, dict(rmse=[], energy=[], spread=[]))
            rec["rmse"] += ((d1 + d2) / 2).tolist()
            rec["energy"] += ((d1 + d2) / 2 - d12 / 2).tolist()
            rec["spread"] += d12.tolist()
    return {name: {k: dict(mean=float(np.mean(v)), sem=float(np.std(v) / math.sqrt(len(v)))) for k, v in rec.items()}
            for name, rec in per.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="flowmap_robotwin_lsd_grid")
    parser.add_argument("--cache", default="data/latent_cache/robotwin_s8_v1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--spec", action="append", required=True, help="label=checkpoint@nfe,nfe,...")
    parser.add_argument("--clips", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name=args.config_name)
    misc.register_work_dir(out)
    torch.manual_seed(0)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda",
                        skip_dit_load_from_pretrain=True, load_text_encoder=False)
    model.eval().requires_grad_(False)
    raw = instantiate(cfg.data.train)
    cached = CachedLatentDataset(args.cache, raw, require_complete=False)
    ready = np.flatnonzero(cached.cache.ready())
    positions = np.sort(np.random.default_rng(args.clip_seed).choice(ready, args.clips, replace=False))
    log(f"{torch.cuda.get_device_name()}: {args.clips} clips (seed {args.clip_seed}) from {args.cache}")
    batches = []
    for start in range(0, len(positions), args.batch_size):
        chunk = positions[start:start + args.batch_size]
        batch = default_collate([cached[int(p)] for p in chunk])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = model.build_inputs(batch)
        inputs["_seed"] = int(start)
        batches.append(inputs)
    results = dict(created=datetime.datetime.now().isoformat(), clips=positions.tolist(), config=args.config_name,
                   runs={})
    for spec in args.spec:
        label, rest = spec.split("=", 1)
        ckpt, nfes = rest.rsplit("@", 1)
        t0 = time.perf_counter()
        zero_jump_input(model)
        model.load_checkpoint(ckpt, optimizer=None, strict_shape=True)
        model.eval().requires_grad_(False)
        entry = dict(checkpoint=ckpt, jump_input_zero=jump_input_is_zero(model), nfe={})
        log(f"{label}: loaded {ckpt} in {time.perf_counter() - t0:.0f}s (jump input zero: {entry['jump_input_zero']})")
        for nfe in (int(k) for k in nfes.split(",")):
            t0 = time.perf_counter()
            entry["nfe"][nfe] = evaluate(model, batches, nfe)
            m = entry["nfe"][nfe]
            log(f"  {label} NFE {nfe:2d} ({time.perf_counter() - t0:.0f}s): " + "  ".join(
                f"{n} rmse {m[n]['rmse']['mean']:.4f} es {m[n]['energy']['mean']:.4f} spread {m[n]['spread']['mean']:.4f}"
                for n in ("video", "dino", "action")))
        results["runs"][label] = entry
        (out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    log(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()

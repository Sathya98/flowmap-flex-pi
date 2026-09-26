#!/usr/bin/env python
"""Precompute the frozen-encoder inputs of FlexPi training into a latent cache.

Caches, per dataset window, what ``FlexPi.build_inputs`` computes with frozen
modules: Wan2.2 VAE latents of the RGB and pointmap composites and DINOv3
features (stored once per frame), plus actions/proprio/pad flags and the prompt.
Training then reads these instead of decoding video/depth and re-encoding.
See ``src/flexpi/datasets/latent_cache.py`` and ``docs/LATENT_CACHE.md``.

    # 1. create the cache (one GPU): checks determinism and the checkpoint
    #    geometry, probes shapes, plans windows and DINO rows
    python scripts/cache_latents.py init --out data/latent_cache/libero_fulljoint_v2 \\
        --config-name flowmap_libero_lmd_full [--window-stride 8]
    # 2. encode (one process per GPU; resumable, skips finished windows)
    srun --ntasks=4 --gpus-per-task=1 python scripts/cache_latents.py encode --out ...
    # 3. compare cached inputs against fresh build_inputs on sampled windows
    python scripts/cache_latents.py verify --out ...
    # train from it
    ... scripts/train.py --config-name=... +data.latent_cache_dir=<out>

Hydra overrides after the flags go into the composed config (``init`` only;
later commands reuse the resolved ``config.yaml`` stored in the cache).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate

from flexpi.datasets import latent_cache as lc
from flexpi.datasets.lerobot import base_lerobot_dataset
from flexpi.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from flexpi.utils import misc
from flexpi.utils.config_resolvers import register_default_resolvers

REPO = Path(__file__).resolve().parents[1]
register_default_resolvers()
# Never substitute a random index for one that fails to decode: a cache row
# must hold the window its index names. Failures are recorded instead.
base_lerobot_dataset.MAX_GETITEM_ATTEMPT = 1


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


class ExactDataset(torch.utils.data.Dataset):
    """``raw._get(i)`` without the random-index fallback of ``__getitem__``."""

    def __init__(self, raw):
        self.raw = raw

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        try:
            return int(i), self.raw._get(int(i))
        except Exception as err:  # recorded by the caller, never silently replaced
            return int(i), f"{type(err).__name__}: {err}"


def collate(items):
    ok = [(i, s) for i, s in items if isinstance(s, dict)]
    failed = [(i, s) for i, s in items if not isinstance(s, dict)]
    return [i for i, _ in ok], (default_collate([s for _, s in ok]) if ok else None), failed


def run_standalone():
    """Hide the launcher's MPI/torchrun variables from accelerate.

    Encode processes are independent (each takes its own slice of windows);
    with them visible, the dataset's ``PartialState()`` tries to join a
    multi-node process group and fails for want of ``MASTER_ADDR``.
    """
    for key in list(os.environ):
        if key.startswith(("PMI_", "PMIX_", "OMPI_", "MV2_", "MPI_LOCAL")) or key in (
                "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
            del os.environ[key]


def build(cfg, out):
    for key in ("skip_padding_as_possible", "exterior_view_aug_prob", "sample_language_annotations"):
        if cfg.data.train.get(key):
            raise ValueError(f"data.train.{key} makes samples random; they cannot be cached")
    misc.register_work_dir(out)   # the dataset copies its norm stats here
    raw = instantiate(cfg.data.train)
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))
    # Only the frozen encoders matter; skip the DiT/text-encoder weight loads.
    model = instantiate(cfg.model, model_dtype=dtype, device="cuda",
                        skip_dit_load_from_pretrain=True, load_text_encoder=False)
    model.eval().requires_grad_(False)
    return raw, model, dtype


@torch.no_grad()
def encode(model, dtype, batch):
    with torch.autocast("cuda", dtype=dtype):   # training runs build_inputs under autocast
        inputs = model.build_inputs(batch)
    return inputs


def loader(raw, indices, batch_size, num_workers):
    kwargs = dict(prefetch_factor=4) if num_workers else {}
    return DataLoader(ExactDataset(raw), batch_size=batch_size, sampler=list(indices),
                      num_workers=num_workers, collate_fn=collate, pin_memory=True, **kwargs)


def mark_done(cache, positions):
    """Flush every array, then mark ``positions`` done (never before their data is on disk)."""
    for name in cache.manifest["arrays"]:
        cache.array(name, "r+").flush()
    done = cache.done("r+")
    done[positions] = 1
    done.flush()


def write_rows(cache, positions, batch, inputs, flush=True):
    """Write windows ``positions`` (indices into ``windows``) and the DINO rows they own.

    ``flush=False`` leaves them in the page cache, unmarked; the caller must
    ``mark_done`` them later (a crash before that just re-encodes them).
    """
    m = cache.manifest
    kept = torch.tensor(m["dino"]["kept_token_idx"])
    const_idx = m["dino"]["const_token_idx"]
    dino = inputs["dino_features"][..., 0].cpu()   # [B, C, F, T]
    if const_idx:
        got = dino[:, :, :, const_idx].float()
        want = cache.dino_const().float()[None, :, None]
        if not torch.allclose(got, want.expand_as(got), rtol=2e-2, atol=2e-2):
            raise RuntimeError("DINO tokens assumed constant (synthetic camera) changed; rebuild without dropping them")
    row_index, writer = cache.array("dino_row_index"), cache.array("dino_writer")
    for b, p in enumerate(positions):
        # Encoder outputs are stored as bf16; the reader restores source_dtype.
        cache.array("video_latents", "r+")[p] = lc.to_numpy(inputs["input_latents"][b].bfloat16())
        if "pointmap_latents" in m["arrays"]:
            cache.array("pointmap_latents", "r+")[p] = lc.to_numpy(inputs["pointmap_raw"][b].bfloat16())
        for j in np.flatnonzero(writer[p]):
            cache.array("dino_frames", "r+")[row_index[p, j]] = lc.to_numpy(dino[b, :, j][:, kept].bfloat16())
        for key in m["window_keys"]:
            cache.array(key, "r+")[p] = lc.to_numpy(batch[key][b])
        prompt = batch["prompt"][b].encode("utf-8")
        if len(prompt) > lc.PROMPT_BYTES:
            raise ValueError(f"Prompt longer than {lc.PROMPT_BYTES} bytes: {batch['prompt'][b]!r}")
        cache.array("prompt", "r+")[p] = prompt
    if flush:
        mark_done(cache, positions)


def cmd_init(args, overrides):
    out = Path(args.out).resolve()
    if (out / "manifest.json").exists():
        sys.exit(f"{out} already holds a cache; use encode to continue it")
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    raw, model, dtype = build(cfg, out)
    n = len(raw)
    log(f"dataset: {n} indices")

    # The checkpoint this config trains from must load with strict shapes, or
    # the cached geometry (layout, DINO grid) is not the one it expects.
    ckpt = cfg.get("pretrained_ckpt")
    if ckpt and not args.skip_ckpt_check:
        model.load_checkpoint(str(ckpt), optimizer=None, strict_shape=True)
        log(f"strict-shape load OK: {ckpt}")

    # Determinism: the same index must decode to the same tensors.
    a, b = raw._get(0), raw._get(0)
    for key in ("per_cam", "per_cam_depth"):
        for cam in a.get(key, {}):
            if not torch.equal(a[key][cam], b[key][cam]):
                raise RuntimeError(f"{key}[{cam}] differs between two loads of index 0; not cacheable")

    episodes = raw.lerobot_dataset.episode_data_index
    windows = lc.select_windows(episodes["from"], episodes["to"], args.window_stride)
    num_windows = len(windows)
    probe_pos = sorted({0, num_windows // 3, 2 * num_windows // 3, num_windows - 1})
    probe = windows[probe_pos].tolist()
    indices, batch, failed = next(iter(loader(raw, probe, len(probe), 0)))
    if failed:
        raise RuntimeError(f"Probe windows failed to load: {failed}")
    inputs = encode(model, dtype, batch)
    dino = inputs["dino_features"][..., 0].cpu()   # [B, C, F, T]
    num_tokens, num_frames = dino.shape[-1], dino.shape[2]
    spread = (dino - dino[:1, :, :1]).abs().amax(dim=(0, 1, 2))
    const_idx = torch.nonzero(spread == 0).flatten().tolist()
    if len(const_idx) > num_tokens // 2:
        raise RuntimeError(f"{len(const_idx)}/{num_tokens} DINO tokens constant across probes; suspicious")
    kept_idx = [t for t in range(num_tokens) if t not in set(const_idx)]
    offsets = lc.dino_frame_offsets(raw.video_sample_indices, model.dino_temporal_stride,
                                    model.dino_stride_keep_far, cfg.data.train.global_sample_stride)
    if len(offsets) != num_frames or offsets[0] != 0:
        raise RuntimeError(f"DINO frame offsets {offsets} do not match {num_frames} encoded frames")
    frames = lc.window_frames(windows, episodes["to"], offsets)
    rows, row_index, writer, owner = lc.dino_row_plan(frames)

    def spec(t, rows="window", bf16=False):
        source = str(t.dtype).removeprefix("torch.")
        if bf16:   # encoder output: store bf16, restore the source dtype on read
            return {"shape": list(t.shape), "np_dtype": "<i2", "dtype": "bfloat16",
                    "source_dtype": source, "rows": rows}
        return {"shape": list(t.shape), "np_dtype": np.dtype(lc._NP_DTYPES[t.dtype]).str,
                "dtype": source, "rows": rows}

    window_keys = sorted(k for k, v in batch.items() if isinstance(v, torch.Tensor) and k not in lc._RAW_KEYS)
    arrays = {"video_latents": spec(inputs["input_latents"][0], bf16=True),
              "dino_frames": spec(dino[0, :, 0][:, kept_idx], rows="dino", bf16=True),
              "prompt": {"shape": [], "np_dtype": f"S{lc.PROMPT_BYTES}", "dtype": "bytes", "rows": "window"}}
    if inputs["pointmap_raw"] is not None:
        arrays["pointmap_latents"] = spec(inputs["pointmap_raw"][0], bf16=True)
    for key in window_keys:
        arrays[key] = spec(batch[key][0])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    manifest = {
        "version": lc.CACHE_VERSION, "dataset_len": n, "num_windows": num_windows,
        "window_stride": args.window_stride, "num_dino_rows": len(rows),
        "config_name": args.config_name, "overrides": overrides, "git_commit": commit,
        "created": datetime.datetime.now().isoformat(),
        "fingerprint": lc.encoder_fingerprint(cfg), "arrays": arrays, "window_keys": window_keys,
        "dino": {"frame_offsets": offsets, "num_tokens": num_tokens,
                 "const_token_idx": const_idx, "kept_token_idx": kept_idx},
    }
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    if const_idx:
        np.save(out / "dino_const.npy", lc.to_numpy(dino[0, :, 0][:, const_idx].bfloat16()))
    cache = lc.LatentCache.create(out, manifest, dict(
        windows=windows, dino_rows=rows, dino_row_index=row_index, dino_writer=writer, dino_row_owner=owner))
    write_rows(cache, probe_pos, batch, inputs)
    sizes = {"window": num_windows, "dino": len(rows)}
    total = sum(np.prod([sizes[a["rows"]], *a["shape"]]) * np.dtype(a["np_dtype"]).itemsize for a in arrays.values())
    log(f"created {out}: {num_windows} windows (stride {args.window_stride}) of {n}, {len(rows)} DINO rows, "
        f"{len(const_idx)}/{num_tokens} constant DINO tokens dropped, DINO offsets {offsets}, "
        f"~{total / 2**30:.0f} GiB when full")
    for name, a in arrays.items():
        log(f"  {name:18s} {a['dtype']:9s} {a['shape']} per {a['rows']}")


def cmd_encode(args):
    out = Path(args.out).resolve()
    rank = args.rank if args.rank is not None else int(os.environ.get("SLURM_PROCID", 0))
    world = args.world if args.world is not None else int(os.environ.get("SLURM_NTASKS", 1))
    run_standalone()
    cfg = OmegaConf.load(out / "config.yaml")
    cache = lc.LatentCache.open(out)
    total = cache.manifest["num_windows"]
    lo, hi = rank * total // world, (rank + 1) * total // world
    done = cache.done()
    pending = [p for p in range(lo, hi) if not done[p]]
    if args.limit:
        pending = pending[:args.limit]
    log(f"rank {rank}/{world}: cached windows [{lo}, {hi}), {len(pending)} pending")
    if not pending:
        return
    raw, model, dtype = build(cfg, out)
    if len(raw) != cache.manifest["dataset_len"]:
        raise RuntimeError(f"Dataset now has {len(raw)} indices, cache was built for {cache.manifest['dataset_len']}")
    windows = np.asarray(cache.array("windows"))
    position = {int(windows[p]): p for p in pending}
    failures = out / f"failures_rank{rank}.jsonl"
    start, count, last = time.perf_counter(), 0, 0.
    spent = dict(load=0., encode=0., write=0.)   # where the wall time goes
    unflushed = []                                # written, not yet marked done
    tick = time.perf_counter()
    def chunks():
        # Fresh DataLoader workers and memmaps every --restart-every windows: long
        # runs otherwise grow host memory until the cgroup OOM-kills the step
        # (RoboTwin job 27165240 died after 8 h at 145 GB RSS, shm bus errors).
        for lo in range(0, len(pending), args.restart_every):
            if lo:
                if unflushed:      # never drop a mapping with written-but-unmarked windows
                    mark_done(cache, unflushed)
                    unflushed.clear()
                cache._arrays.clear()
            yield from loader(raw, windows[pending[lo:lo + args.restart_every]], args.batch_size, args.num_workers)

    for indices, batch, failed in chunks():
        spent["load"] += time.perf_counter() - tick
        for i, err in failed:
            log(f"index {i} failed: {err}")
            with open(failures, "a") as f:
                f.write(json.dumps({"index": i, "error": err}) + "\n")
        if batch is None:
            tick = time.perf_counter()
            continue
        t0 = time.perf_counter()
        inputs = encode(model, dtype, batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        unflushed += [position[i] for i in indices]
        write_rows(cache, [position[i] for i in indices], batch, inputs, flush=False)
        if len(unflushed) >= args.flush_every or count + len(indices) == len(pending):
            mark_done(cache, unflushed)
            unflushed = []
        tick = time.perf_counter()
        spent["encode"] += t1 - t0
        spent["write"] += tick - t1
        count += len(indices)
        elapsed = time.perf_counter() - start
        if elapsed - last > 60 or count == len(pending):
            last = elapsed
            rate = count / elapsed
            split = " ".join(f"{k} {v / elapsed:.0%}" for k, v in spent.items())
            log(f"rank {rank}: {count}/{len(pending)} windows, {rate:.2f}/s, "
                f"ETA {(len(pending) - count) / max(rate, 1e-9) / 3600:.1f} h ({split})")
    if unflushed:   # the last batches can end early when windows failed to load
        mark_done(cache, unflushed)
    log(f"rank {rank}: finished {count} windows in {(time.perf_counter() - start) / 3600:.2f} h")


def rel_err(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def cmd_verify(args):
    out = Path(args.out).resolve()
    cfg = OmegaConf.load(out / "config.yaml")
    cache = lc.LatentCache.open(out)
    m = cache.manifest
    done = np.asarray(cache.done()).astype(bool)
    log(f"{int(done.sum())}/{m['num_windows']} windows encoded")
    raw, model, dtype = build(cfg, out)
    cached_ds = lc.CachedLatentDataset(out, raw, require_complete=False)
    rng = np.random.default_rng(args.seed)
    windows = np.asarray(cache.array("windows"))
    row_index = np.asarray(cache.array("dino_row_index"))
    rows, owner = np.asarray(cache.array("dino_rows")), np.asarray(cache.array("dino_row_owner"))
    usable = np.flatnonzero(cache.ready())
    # Half of the checks where the last DINO frame is clamped at the episode end.
    clamped = usable[rows[row_index[usable, -1]] < windows[usable] + m["dino"]["frame_offsets"][-1]]
    half = args.samples // 2
    picks = np.concatenate([rng.choice(usable, min(half, len(usable)), replace=False),
                            rng.choice(clamped, min(half, len(clamped)), replace=False)])
    worst, neighbour_fail = {}, 0
    for start in range(0, len(picks), args.batch_size):
        chunk = picks[start:start + args.batch_size].tolist()
        indices, batch, failed = next(iter(loader(raw, windows[chunk], len(chunk), 0)))
        if failed:
            raise RuntimeError(f"fresh load failed: {failed}")
        fresh = encode(model, dtype, batch)
        cached = encode(model, dtype, default_collate([cached_ds[p] for p in chunk]))
        for key, value in fresh.items():
            if isinstance(value, torch.Tensor):
                if value.dtype != cached[key].dtype:
                    err = float("inf")
                    log(f"{key}: fresh dtype {value.dtype} != cached {cached[key].dtype}")
                elif value.dtype == torch.bool:
                    err = float((value != cached[key]).any())
                else:
                    err = rel_err(cached[key], value)
                worst[key] = max(worst.get(key, 0.), err)
            elif value != cached[key]:
                raise RuntimeError(f"{key}: fresh {value!r} != cached {cached[key]!r}")
        # Each cached DINO frame must match the fresh one better than the
        # neighbouring cached rows (catches an off-by-one frame mapping).
        for b, p in enumerate(chunk):
            for j, r in enumerate(row_index[p]):
                target = fresh["dino_features"][b, :, j, :, 0]
                errs = {q: rel_err(cache.dino_frame(q), target) for q in (r - 1, r, r + 1)
                        if 0 <= q < len(rows) and done[owner[q]]}
                if min(errs, key=errs.get) != r:
                    neighbour_fail += 1
                    log(f"window {windows[p]} DINO frame {j}: row {r} is not the best match {errs}")
    tol = {"dino_features": 2e-2}
    bad = {k: e for k, e in worst.items() if e > tol.get(k, 1e-3)}
    for key, err in sorted(worst.items()):
        log(f"  {key:22s} max rel err {err:.2e}{'  FAIL' if key in bad else ''}")
    log(f"checked {len(picks)} windows ({min(half, len(clamped))} with a clamped last DINO frame)")
    if bad or neighbour_fail or not len(picks):
        sys.exit(1)
    log("OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--out", required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--window-stride", type=int, default=1,
                   help="cache windows starting at every k-th frame of each episode")
    p.add_argument("--skip-ckpt-check", action="store_true",
                   help="do not strict-load cfg.pretrained_ckpt into the configured model")
    p = sub.add_parser("encode")
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int)
    p.add_argument("--world", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="encode at most this many windows (smoke)")
    p.add_argument("--flush-every", type=int, default=256,
                   help="flush to disk and mark done after this many windows")
    p.add_argument("--restart-every", type=int, default=20000,
                   help="recreate dataloader workers and memmaps after this many windows")
    p = sub.add_parser("verify")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args, overrides = parser.parse_known_args()
    if args.cmd != "init" and overrides:
        parser.error(f"unrecognised arguments: {overrides}")
    {"init": lambda: cmd_init(args, overrides), "encode": lambda: cmd_encode(args),
     "verify": lambda: cmd_verify(args)}[args.cmd]()


if __name__ == "__main__":
    main()

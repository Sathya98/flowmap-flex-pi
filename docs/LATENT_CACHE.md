# Latent cache — quick notes

Precompute the frozen-encoder work of `FlexPi.build_inputs` once, so training
reads latents instead of decoding video/depth and re-encoding every batch.
Motivation: the ~13 s/example floor in `.claude/context/07-efficiency-notes.md`
is shared by every objective and contains data decoding plus the frozen VAE and
DINO encoders.

Code: `src/flexpi/datasets/latent_cache.py` (FlexPi layout, reader, `build_inputs_from_cache`) on
top of the generic store `flowmap_core.latent_store.ArrayStore` (memory maps, sparse preallocation, bf16 round trip),
`scripts/cache_latents.py` (init / encode / verify), `scripts/slurm/cache_latents.sbatch`,
`tests/test_latent_cache.py`.

## What is cached

| Stored | Per | Why this granularity |
|---|---|---|
| RGB VAE latents `[48, 3, h, w]` | window | Wan VAE is temporally causal: chunk *k* is encoded with the cached state of chunks < *k*, so one raw frame gets different latents in different windows |
| Pointmap VAE latents (same shape) | window | same VAE |
| DINOv3 features `[C, tokens]` | frame (row) | DINO encodes each image independently; a frame shared by several windows is stored once |
| action, proprio, pad flags, intrinsics, prompt | window | straight from the dataset |

Not cached: the **proprio token** (the proprio encoder trains; it is appended at
train time) and the **umT5 context** (looked up by prompt in the existing text
cache). Tokens that are constant for every frame (LIBERO's synthetic black
camera) are stored once.

A *window* is one training sample: the clip starting at dataset index `g`
(33 raw frames → 9 video frames → 3 latent frames). There is one window per
frame. A window's DINO frames sit at offsets from `g` that depend on the DINO
stride: `[0, 32]` for LIBERO (stride 2, keep-far) and `[0, 16, 32]` for the
RoboTwin checkpoint (stride 1). They are clamped to the episode's last frame,
the same way LeRobot clamps the query.

`window_stride=k` keeps only windows starting at every k-th frame of each
episode. With k dividing 16, the later DINO frames of one window are other
windows' anchors, so the only extra DINO rows are the clamped episode-end
frames.

## Sizes

| Dataset | Windows | Cache |
|---|---|---|
| LIBERO (4 suites, 20 Hz), stride 1 | 277,492 | ~300 GB (258 KB × 2 latents + 602 KB DINO per window) |
| RoboTwin 2.0 3D (50 Hz), stride 1 | ~6.18M | ~4.5 TB (does not fit the quota) |
| RoboTwin, **stride 8** | ~773k | **~570 GB** (138 KB × 2 latents + 452 KB DINO) |

Stride 8 on RoboTwin puts window starts 0.16 s apart, while each window spans
0.66 s. Training sees ~1/8 of the distinct start frames; that is a change in the
data distribution to keep in mind when comparing to raw-data runs.

## Commands

```bash
# Smoke (1 GPU, 64 windows): init + encode + verify
#   args: OUT_DIR CONFIG_NAME [STRIDE] [LIMIT]  (positional: the site sbatch wrapper drops env vars)
sbatch --ntasks=1 --cpus-per-task=18 --mem=120G --time=01:00:00 \
  scripts/slurm/cache_latents.sbatch $PWD/data/latent_cache/libero_fulljoint_v2 flowmap_libero_lmd_full 1 64
# Full build: same OUT_DIR continues the cache (resubmit after a timeout)
sbatch scripts/slurm/cache_latents.sbatch $PWD/data/latent_cache/libero_fulljoint_v2 flowmap_libero_lmd_full
# RoboTwin, stride 8 (stride is fixed at init)
sbatch scripts/slurm/cache_latents.sbatch $PWD/data/latent_cache/robotwin_s8_v1 flowmap_robotwin_base 8

# Train from a cache
... scripts/train.py --config-name=<cfg> +data.latent_cache_dir=data/latent_cache/<name>
```

## Safeguards

- **Exact indices.** Encoding calls `dataset._get(i)` with the random-index
  retry disabled. A window that fails to decode goes to
  `failures_rank*.jsonl` and is never replaced by another window.
- **Determinism check** (init): index 0 is loaded twice and must decode
  identically. Configs with random augmentation (`exterior_view_aug_prob`,
  `skip_padding_as_possible`, `sample_language_annotations`) are refused.
- **Checkpoint geometry** (init): `cfg.pretrained_ckpt` is loaded with strict
  shapes into the configured model. A layout or DINO-grid mismatch fails before
  anything is encoded.
- **Fingerprint** (train): `data.train` + encoder settings + a hash of the norm
  stats must equal the cache's, or training refuses to start.
- **Verify:** fresh `build_inputs` vs cached inputs on random windows, half of
  them with a clamped last DINO frame. Tolerances: rel. err ≤ 1e-3, DINO ≤ 2e-2
  (bf16 batching). Each cached DINO frame must also match better than its
  neighbouring rows.
- **Resumable:** `done.npy` is set only after a window's arrays are flushed.
- **Long runs:** encode restarts its DataLoader workers and drops its memmaps every
  `--restart-every` windows (default 20k), after flushing. Without it, the RoboTwin build
  (job 27165240) grew host memory until the step was OOM-killed after 8 h (MaxRSS 145 GB,
  shm bus errors in workers).

## Limits

- Eval and previews still use the raw dataset (they need pixels).
- A cache is tied to its config. Changing the layout, DINO grid/stride,
  resolution, norm stats or data dirs means building a new cache.
- Weighted multi-dataset sampling indexes raw frames, so it cannot use a strided
  cache (this raises an error).
- Randomly sampled prompt paraphrases cannot be cached. RoboTwin does not sample
  them: each frame carries one of its episode's 100 instructions in
  `task_index`, and a window uses the one on its start frame.

## RoboTwin notes

- **Checkpoint geometry** (`configs/flowmap_robotwin_base.yaml`). The released
  RoboTwin checkpoint predates the current task recipe:
  - legacy asymmetric layout, 294 DINO tokens/frame
  - `dino_pixel_unshuffle: 0` (768-wide DINO layers)
  - `dino_temporal_stride: 1`
  - `dino_pred_x0: false`

  Its saved config names none of these keys, and eval swaps that config in
  verbatim, so the code defaults applied there. The task config's defaults
  differ (uniform layout, unshuffle 2, x0 prediction) and would not match. The
  norm-stats path is the checkpoint's (identical to the dataset's own; the task
  config's path does not exist).
- **Text embeddings** (`data/text_embeds_cache_3d`): 1,039,891 umT5 prompts in
  520 safetensors shards (~1 TB). They are read in place by
  `datasets/text_embed_shards.py` (manifest binary search → one row). Unpacking
  to per-prompt `.pt` files would cost ~1M inodes. The context is the only
  language input (cross-attention in both experts); the ~1M prompts are the 100
  paraphrases per episode.
- **Sequence length.** Legacy geometry gives 360 video + 360 pointmap + 882 DINO
  = 1602 visual tokens/sample, about the same as LIBERO (672 + 672 + 294 = 1638),
  so LIBERO's per-step memory is a fair guide. The current RoboTwin recipe would
  be 1161.

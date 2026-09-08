# 03 — Dataloaders & the data pipeline

Everything that turns a LeRobot dataset on disk into the sample dict the model consumes.
Line numbers are a snapshot; verify before editing. Package root abbreviated `…/` =
`src/flexpi/`.

## 0. The three-layer stack

```
MultiLeRobotDataset (raw parquet + video/depth decode)
  └─ BaseLerobotDataset      …/datasets/lerobot/base_lerobot_dataset.py
        (delta_timestamps, train/val split, per-sample dict, calls processor)
        └─ FlexPiProcessor    …/datasets/lerobot/processors/flexpi_processor.py
              (per-cam image transforms, action/state merge + normalize)
RobotVideoDataset            …/datasets/lerobot/robot_video_dataset.py
  (wraps BaseLerobotDataset; adds depth, camera K, T5 text context, per-cam packaging)
     → sample dict → collate → model.build_inputs → per_cam_compose.compose_from_per_cam (GPU)
```

**The composite is NOT assembled in the worker** — workers emit per-cam dicts; the model
pastes them into the layout composite on GPU via `compose_from_per_cam`.

## 1. `RobotVideoDataset` — the top-level dataset

`…/robot_video_dataset.py:104`. Composes (not subclasses) a `BaseLerobotDataset` in
`self.lerobot_dataset`.

### Key constructor args
`dataset_dirs`, `shape_meta` (`images`/`state`/`action`/optional `depth`), `num_frames=33`,
`action_video_freq_ratio` (**4** in configs), `video_size`, `concat_multi_camera` (a
registered `LayoutSpec` name), `depth_codec` (`ffv1`|`x264rgb`), `text_embedding_cache_dir`,
`context_len=128`, `pretrained_norm_stats`, `val_set_proportion=0.005`, `is_training_set`,
`dataset_weights`/`samples_per_epoch` (size-agnostic mixing), `synthetic_zero_cams`
(fabricate missing cams as zero RGB/depth/identity-K — LIBERO's 3rd cam),
`exterior_view_aug_*`.

### Temporal sampling (`:242-249`)
`video_sample_indices = range(0, num_frames, ratio)` → for `33, 4` = `[0,4,…,32]` = **9
anchor frames**. Asserts `(num_frames-1)%ratio==0` and `((num_frames-1)//ratio)%4==0`. The
model's VAE then compresses 9 → **3 latent frames**; **32 action steps** (`num_frames-1`).
Env `FLEXPI_RGB_DECODE_SUBSAMPLE=1` makes the decoder fetch only the 9 used frames.

### Sample dict (`_get`, `:508-742`; `__getitem__` retries a random index on any exception)
| key | shape | notes |
|---|---|---|
| `per_cam` | `dict[cam → [3, 9, H, W]]` float32 `[-1,1]` | per-cam HW = layout `Slot.src_hw` (head 256×320, wrists 224×224) |
| `per_cam_depth` | `dict[cam → [9, H, W]]` uint16 mm | only if `shape_meta.depth` declared |
| `camera_intrinsics` | `[num_cams, 3, 3]` | only if `meta/camera_intrinsics.json` found; **slot order** |
| `action` | `[32, action_output_dim]` | |
| `proprio` | `[32, proprio_output_dim]` | `sample["proprio"][:-1]` (drops last obs step) — proprio stays **absolute** |
| `prompt` | `str` | formatted instruction |
| `context` | `[128, D]` | umT5 embeddings, padded rows zeroed |
| `context_mask` | `[128]` bool | forced all-ones after zeroing |
| `image_is_pad`/`action_is_pad`/`proprio_is_pad` | bool | end-of-episode pad flags |

RGB has two branches: **uniform** (all cams same HW → stacked `pixel_values`, per-cam
bilinear resize to slot HW) and **heterogeneous** (cams differ → `per_cam_rgb` dict). Three
distinct HW tables coexist and must not be conflated: `_per_cam_hw` (depth grid, from
`shape_meta.depth`), `_rgb_per_cam_hw` (RGB source = `Slot.src_hw`), and composite
`Slot.(h,w)` (paste target).

### Mixing, fps, split (delegated to `BaseLerobotDataset`)
- **fps consistency** asserted across all `dataset_dirs` (raises on mismatch); fps drives
  every `delta_timestamps`.
- **train/val split**: shuffle with fixed `rng(seed=42)`, split at
  `int(len*(1-val_prop))`; `is_training_set` picks head/tail. Explicit `episode_ids` or
  `task_names` (via `task_episode_map.json`) override.
- **`dataset_weights`/`samples_per_epoch`**: per-frame draw ratios (this class validates;
  the trainer's `WeightedResumableEpochSampler` does the weighted draw).
- **Norm stats**: training set with no `pretrained_norm_stats` computes + broadcasts stats;
  val/test must be given stats. Then `processor.set_normalizer_from_stats(...)`.

## 2. Depth pipeline (`…/lerobot/lerobot/datasets/video_utils.py`, `…/utils/depth_codec.py`)

Codecs (`robot_video_dataset.py:88-89`): `ffv1`→`.mkv` (FFV1 `gray16le`, bit-exact uint16),
`x264rgb`→`.mp4` (`libx264rgb` lossless, uint16 packed as `R=hi,G=lo` → recover
`(R<<8)|G`). `self._depth_feat_prefix = f"observation.depth_{codec}"`.

`decode_depth_frames` (`video_utils.py:716`): **decord fast path** for `x264rgb` +
`LEROBOT_DEPTH_DECORD=1` (~3× faster cold, GIL-released `get_batch`); **FFV1 stays on
pyav** (decord can't decode FFV1). Depth is decoded at the **9 RGB-anchor frames** (same
density as RGB → 9 depth → VAE → 3 pointmap latents, 1:1 with RGB). Query timestamps are
reconstructed from the global idx (the processor strips episode/timestamp), rounded to ms,
argmin-selected against loaded PTS with a 1 ms tolerance. Per-cam depth resized with a
linspace-endpoint **nearest** sampler (preserves integer mm; pairs with the K endpoint
rescale).

## 3. `FlexPiProcessor` (`…/processors/flexpi_processor.py`)

`preprocess` (`:186-342`) order:
1. **Instruction** (`augment_instruction`, `:124`): with `drop_high_level_prob=1.0` returns
   the bare low-level task string (no `[Low]:` prefix).
2. **Images** (`:227-303`): per-cam transform list (`ToTensor` → `Resize`), asserts per-frame
   `[C,H,W]==shape_meta`. Uniform → `pixel_values [n_cam,T,C,H,W]`; heterogeneous →
   `per_cam_rgb` dict.
3. **Action/state** (`:311-336`): `delta_action_dim_mask` zeroing → `action_state_transform`
   (rel-action transforms) → `normalizer.forward` (z-score / min-max) →
   `action_state_merger.forward`.

- **Merger `ConcatLeftAlign`** (`…/transforms/action_state_merger.py`): concat per-key
  action tensors in `shape_meta` order, right-pad to `action_output_dim`, emit
  `action_dim_is_pad`. `backward` crops + splits for eval.
- **Normalizer `LinearNormalizer`** (`…/utils/normalizer.py`): per-key
  `SingleFieldLinearNormalizer`, modes `z-score` (scale `1/(std+1e-8)`, clamp ±5),
  `min/max`/`q01/q99` (→`[-1,1]`, clamp ±1). `skip_dims_mask` forces identity on masked
  dims (used to keep rot6d out of z-scoring). Stats JSON written atomically (GPFS-safe);
  `pad_dataset_stats_to_target_dim` for cross-embodiment finetune.

## 4. Composite layouts & slots (`composite_layouts.py`, `per_cam_compose.py`)

`Slot` (frozen dataclass): `key` (placeholder, `""`=black), `top/left/h/w` (bbox on
composite), `src_hw` (native per-cam HW), `tile_mode` (`resize`|`repeat`). `LayoutSpec`:
`name`, `composite_hw`, ordered `slots`, `default_slot_key_map` (placeholder→cam key), plus
**DINO RoPE tables** `dino_cam_patches`/`dino_cam_regions`/`dino_grid_hw` (load-bearing for
RoPE, not just viz). `with_dino_pool(factor)` divides the grids for `dino_pool_factor`.

Registered layouts (`:270-386`):
- **`tshape_robotwin_384x320` (legacy asymmetric)** — head 256×320 top full-width; two wrist
  128×160 below. `dino_cam_patches=((14,14),(7,7),(7,7))` (294 tok).
- **`tshape_robotwin_384x320_uniform`** (`tshape_384x320` aliases here) — same pixels,
  **uniform 14×14/cam** (28×28 grid) → foldable 2×2 → 7×7/cam = **147 tok/frame**. The
  RoboTwin/YAM default.
- **`tshape_libero_2cam_448x512`** — LIBERO default; head 288×512 + two 160×256; empty
  key map (config supplies one); carries a `text_prefix`; third cam is synthetic-zero.

`cam_slots()` order is load-bearing three ways: it indexes the composite, `camera_intrinsics`
(pointmap encoder indexes K by slot position), and the resolved `slot_key_map`. Never
compare layout name string-literals — use `get_layout`/`canonical_layout_name` (old
checkpoints say `"robotwin"`).

`compose_from_per_cam` (`per_cam_compose.py:98`): per `cam_slots()`, take `per_cam[cam_key]`,
resample to `(slot.h,slot.w)` (antialiased bilinear for `resize`, nearest for `repeat`),
paste into `composite[..., slot.top:...,slot.left:...]`. Composite init `-1.0` (true black in
`[-1,1]`). **No K rescale here** — K is already at the per-cam grid.

## 5. Action / rotation transforms

- **`Yam32DRelativeAction`** (`…/transforms/yam_relative_action.py`): the YAM 32D layout —
  `[0:9]` L EEF (pos+rot6d) **relative**, `[9:18]` R EEF relative, `[18:20]` grippers
  **absolute**, `[20:32]` joints (6+6) **scalar delta**. Anchor = state at sample-start
  (`anchor="first"`). SE(3) via `pose9d_to_mat`; forward `T_rel=inv(T_base)@T_act`, backward
  `T_abs=T_base@T_rel`. **State is never modified — proprio stays absolute end-to-end.**
- **`yam_eef.py`**: numpy SE(3)/rot6d single source of truth. `STATE_LAYOUT` (authoritative
  32D map), `ROT6D_SLICES` (override to identity in norm stats — z-scoring rot6d distorts
  SO(3)). rot6d = first two **rows** of R, row-major.
- **`rotation.py`**: torch rotation library (pytorch3d-derived, `(r,i,j,k)` quats):
  axis-angle↔matrix (**rotvec**, used by LIBERO 32D), euler, 6D (row-based Zhou),
  9D (SVD). `col6d_to_row6d` bridges column-convention (AgiBot disk) → FlexPi's row
  convention (mixing them silently yields `Rᵀ`).

## 6. camera_intrinsics + T5 text cache

- **`camera_intrinsics.json`** (`robot_video_dataset.py:452`): loaded once per dataset dir;
  per cam, K rescaled to the depth grid with the **endpoint convention**
  `sx=(w_target-1)/(raw_W-1)` (pairs with the endpoint nearest depth resize). Attached as
  `[num_cams,3,3]` in **slot order** (identity K for synthetic cams). The pointmap encoder
  indexes K by slot position — required for the 3D stream, raises if missing.
- **T5 cache** (`_get_cached_text_context`, `:948`): prompt =
  `"A video recorded from a robot's point of view executing the following instruction:
  {task}"` (optionally with `layout.text_prefix`). `cache_path =
  <cache_dir>/{sha256(prompt)}.t5_len{context_len}.wan22ti2v5b.pt`, loads
  `payload["context"]`/`["mask"]`. Missing → `FileNotFoundError` pointing at
  `precompute_text_embeds.py`. **Keyed by prompt hash + `context_len`** — every instruction
  your episodes carry must have been encoded. No online text encoder at train time
  (`load_text_encoder=false`).

## Cross-cutting notes
- **`shape_meta.depth` is the single depth switch** — its presence turns on depth decode +
  sets the depth grid; pair "no depth block" with `model.enable_pointmap=false` (+
  `data=<benchmark>_nodepth`).
- Cam ordering is load-bearing across disk order (`shape_meta.images`), composite order
  (`layout.cam_slots()`), and the `slot_key_map` binding them.

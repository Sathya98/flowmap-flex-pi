# Extending the flow-map stack to a Wan2.2-TI2V-5B world model (diffsynth-studio)

> **Question.** Can the flow-map training regime (LSD, and LMD) plus the infrastructure and
> optimizations built for FlexPi be used for another DiT flow-matching model: the
> action-conditioned Wan2.2-TI2V-5B **world model** (WM) in
> `../exmachina/diffsynth-studio`? What carries over, and what has to be done?
>
> Written 2026-09-26 from a read of both repos. No code has been ported or run yet. Timings and
> memory for the WM are **estimates** scaled from FlexPi measurements
> (`.claude/context/07-efficiency-notes.md`).

## TL;DR

- **Yes, the method carries over, and most of the code does too.** The WM is the same Wan2.2-TI2V-5B
  DiT as FlexPi's video trunk: identical dimensions (3072 wide, 30 layers, 24 heads, FFN 14336), the same
  DiffSynth-derived source file, the same per-token AdaLN time conditioning, and the same
  flow-matching objective. The flow-map core (`helpers/flowmap.py`, `flowmap_self.py`,
  `attention.py`, `checkpoint.py`, `normalization.py`, `utils/flowmap_ema.py`, ~1,000 lines) is
  pure torch with **no FlexPi imports**.
- **The trainer does not carry over.** `src/flexpi/trainer.py` (2,045 lines, 114
  FlexPi-specific references) is the wrong host. Keep the WM in diffsynth: its checkpoint format,
  rollout eval and the external RLinf `WanEnv` depend on it. Port the core in, and extend
  diffsynth's small accelerate loop.
- **Priorities change because the WM is small per sample.** At 256×256 × 13 frames it has
  **256 tokens** (4 latent frames × 64), against ~1,640 for FlexPi.
  - Per-example compute is tiny (~0.02 s est.), so a microstep is almost entirely CPU launch
    overhead.
  - **The main lever is a large microbatch** (16–32 per GPU est.): batched collate, the batched
    self-distillation loss, and a latent cache so data keeps up.
  - The TVM kernel becomes optional: attention at L = 256 is cheap, and no mask is needed.
  - The DeepSpeed patch is unnecessary: `diff_env` has DeepSpeed 0.19.6, which contains the
    upstream fix.
  - **The WM is the ideal first target for CUDA graphs / compile** (efficiency notes §11): fixed
    shapes, no mask, one stream, no teacher.
- **The payoff is at inference.** Rollout eval uses 5 denoising steps per chunk
  (`scripts/eval_wan_rollout.py --steps 5`). A 1–2-step flow map would make world-model rollouts
  for RL ~2.5–5× faster.
- Effort: **~2–3 weeks** to a validated LSD run with eval (phases below).

---

## 1. The two setups side by side

| | FlexPi (this repo) | WM (diffsynth-studio) |
|---|---|---|
| Backbone | Wan2.2-TI2V-5B DiT + 1B ActionDiT, MoT/HBridge | Wan2.2-TI2V-5B DiT only (`WanModel`, `TI2V2`) |
| Streams | RGB, pointmap, DINO (video tower) + action | RGB latents only |
| Conditioning | umT5 text + proprio (cross-attn), first-frame anchors | **actions**: `action_mlp1` tokens in cross-attn; `action_mlp2` (4 stacked actions) **added to the time embedding** → AdaLN; no text |
| Anchors | first latent frame clean | first **2** latent frames labelled t = 0 (frame 1 gets small noise, `five_frame_condition`) |
| Loss | all present streams | last 2 latent frames only (= the non-anchor frames) |
| Tokens / example | ~1,640 (LIBERO) | **256** (4 × 8 × 8), 256×256 × 13 frames |
| Attention mask | block mask (MoT, flex regimes) | **none** (full bidirectional) |
| Time sampling | continuous σ, shifted | one discrete Wan timestep per sample batch, `training_weight(t)` |
| Batch | 1 × 48 accum × 4 GPUs = 192 | `collate_fn=lambda x: x[0]` → **1 per GPU, accumulation 1** (effective 4 on 4 GPUs) |
| Data | LeRobot → latent cache | RLinf npy rollouts `[T, N, H, W, 3]`, VAE encoded **every step** in `forward_preprocess` |
| Window | 33 frames → 9 latent | `[0, s, …, s+11]` (keyframe + 12 consecutive, s ≤ 250); 5%: fixed `[0,0,0,0,0,1..8]` |
| Augmentation | — | pixel-space `static_video_prob` (repeat frame 0, action 0), `context_noise_sigma` (0 in launchers) |
| Trainer | `Wan22Trainer` (Hydra, EMA, resume, timing harness) | `launch_training_task` (accelerate, ZeRO-2, best/latest safetensors, resume) |
| Env | `fm_env`: torch 2.7.1, Triton 3.3.1, DeepSpeed 0.18.9 | `diff_env`: **torch 2.11.0, Triton 3.6.0, DeepSpeed 0.19.6** |

Sources: `diffsynth/pipelines/wan_video_new.py` (`training_loss` :249, `model_fn_wan_video`
:1626, five-frame timesteps :1822), `diffsynth/trainers/utils.py` (`launch_training_task` :683,
`RLinfNpyDataset` :364), `diffsynth/models/wan_video_dit.py` (`flash_attention` :43, 5B config
:1079), `examples/wanvideo/model_training/train_rlinf.py`, `.claude/finetuning.md` in that repo.

## 2. Component by component: what carries over

| Component (FlexPi location) | Carries over? | Work for the WM |
|---|---|---|
| **Flow-map objective core**: `FlowMapConfig`, `affine_flow_map`, `sample_level_pair_strip`, `tuple_jvp`, `map_residuals` (LSD/ESD/LMD/EMD, FD and forward AD) in `models/helpers/flowmap.py` | **As is.** `map_residuals(predict, velocity, x, s, t, cfg)` takes closures and tuples of streams. | none |
| **Self-distillation extras**: `TimeLossWeight`, `update_diagonal_mask`, `slice_batch` (`flowmap_self.py`) | **As is.** | Add a per-rank-balanced split for large microbatches (§4). |
| **Joint loss wrapper** (`flowmap_training.py`) | **No**: multi-stream, flex flags, FlexPi reductions. The *structure* carries: batched per-branch calls, per-example reduction, learned time weight. | New ~100-line `wm_flowmap_loss`: one stream, 2 anchor frames with zero velocity, masked MSE on the non-anchor frames, per-example `[B]` losses, `exp(−logvar)·L + logvar`, mean. |
| **DiT surgery** (`wan_video_dit.py` diff, ~100 lines) | **Port** into diffsynth's `WanModel`: same file lineage. | (a) `time_embedding_delta` (zero-init last layer) + `embed_time(t, Δ)`, used in `model_fn_wan_video` *before* `+ action_emb`. (b) `ForwardADLayerNorm` for the LayerNorms. (c) route `flash_attention`'s SDPA through `helpers/attention.scaled_dot_product_attention` under forward AD. (d) dual-aware `checkpoint` for `use_gradient_checkpointing`. |
| **Per-sample timesteps in `model_fn_wan_video`** | **Must change.** The batch path builds one shared timestep and `repeat(B, …)`; `training_loss` samples **one** timestep per batch. | Build per-token timesteps from per-sample `[B]` σ (anchors 0) and Δ (anchors 0). Remove the hardcoded `repeat(1,1,64,1)` (64 tokens per latent frame, 256×256 only). |
| **Checkpoint loading** (`model_manager.load_model_from_single_file`) | **Must change.** Models are built under `init_weights_on_device()` (no init), then `load_state_dict(strict=False, assign=True)`. A missing `time_embedding_delta` would stay **uninitialised**. | Add a missing-key gate like the existing `action_mlp1/2` one: zero-init `time_embedding_delta` when absent, keep it when present. |
| **Forward-AD workarounds** (`ForwardADLayerNorm`, explicit FP32 attention JVP, dual-aware checkpoint) | Written against **torch 2.7.1** bugs | Re-validate on torch 2.11 (some may be unnecessary, `checkpoint.py` may need updating), or run the WM in `fm_env` (§5, decision 1). |
| **TVM fused attention JVP** (`jvp_attention/`) | Optional | Mask-free here (no row grouping). At L = 256 the explicit path is cheap and the dual-aware checkpoint keeps its L×L buffers transient. Needs Triton 3.6 re-validation (`jvp_kernel_analysis/validate_tvm.py`); license CC BY-NC-SA 4.0. Start with `explicit`. |
| **Stratified diagonal mask** | Yes, the principle | The WM uses ~4× smaller per-example steps and large microbatches: see §4. |
| **Batched self-distillation loss** | Yes, the principle; it is **essential** here | Part of `wm_flowmap_loss`: one call per branch. |
| **DeepSpeed ZeRO-2 hook patch** (`utils/deepspeed_compat.py`) | **Not needed** | DeepSpeed 0.19.6 has the upstream fix (≥ 0.18.7); our patch detects it and skips itself. Confirm the per-microstep backward time with the timing harness. |
| **Background EMA** (`utils/flowmap_ema.py`) | **As is**, if an EMA is wanted | diffsynth has no EMA today. The LSD reference recipe evaluates EMA weights, so add it to the loop and to `ModelLogger`'s best/latest export. |
| **Latent cache** (`datasets/latent_cache.py`, `scripts/cache_latents.py`) | Storage layer yes (memmap arrays, manifest, fingerprint, deferred flush, restart, verify); the sample reader and encoder no | New window reader for RLinf npy (§3), the diffsynth VAE encode (same Wan2.2 VAE weights), no DINO rows. |
| **Timing harness / profiler** (`FLEXPI_STEP_*`, `utils/step_profile.py`) | `step_profile.py` as is; the harness logic (~60 lines) re-implemented in diffsynth's loop | Needed to measure §4 before optimising. |
| **Flex-joint, MoT/HBridge, DINO/pointmap, text-embed cache, RoboTwin configs** | Not applicable | — |
| **Inference**: flow-map step `X(s,t,x) = x + (t−s)·v(x, s, Δ)` | The logic yes; the pipeline differs | Flow-map sampler in `WanVideoPipeline.__call__` (anchors fixed), `--steps` in `eval_wan_rollout.py`; RLinf `WanEnv` gets it through the pipeline. |

## 3. Data: the latent cache for RLinf rollouts

- **What a window is:** `frame_ids = [0, s, s+1, …, s+11]`, `s ∈ [0, min(250, T−12)]`; 5% of the
  time `[0,0,0,0,0,1,…,8]`. `action[0]` is forced to `[0,0,0,0,0,0,−1]`. The Wan VAE is temporally
  causal, so each window is encoded separately (as in FlexPi, `docs/LATENT_CACHE.md`).
- **Size:** 48 × 4 × 16 × 16 bf16 ≈ **98 KB per window**. At most 252 windows per env trajectory
  (all starts + the fixed pattern): ~25 MB per trajectory; 1,000 trajectories ≈ 25 GB. Small
  enough to use no stride.
- **Augmentations are in pixel space, before the VAE:**
  - `static_video_prob` (5%) repeats frame 0 with action 0. Cache one extra "static" window per
    trajectory (encode frame 0 × 13).
  - `context_noise_sigma` adds pixel noise to the context frame. It is 0 in both launchers; if
    used, it needs an online encode of that frame or a latent-space substitute (a decision).
  - The frame-1 low-noise augmentation is already in latent space: no change.
- **Why it matters more here:** at microbatch 16–32 per GPU, encoding 13 frames × mb per step on
  the training GPU, plus PIL decode in the workers, would dominate a ~1–2 s microstep.

## 4. Performance model for the WM (estimates)

- **Compute per example.** Forward ≈ 2 × 5e9 × 256 ≈ 2.6 TFLOP. The LSD off-diagonal branch is
  ~6 forward-equivalents (primal + tangent, backward through both) ≈ 15 TFLOP; the diagonal branch
  (plain FM) ~3 ≈ 8 TFLOP. At the 75/25 mixture that is ~10 TFLOP, i.e. **~0.02 s per example**
  at ~500 TFLOP/s. A 192-example update is ~1 s of GPU math per GPU.
- **Launch overhead per microstep.** FlexPi's LSD microstep costs ~0.6 s (diagonal) and ~2.4–3.2 s
  (off-diagonal) at mb1, almost all of it CPU dispatch (538k aten ops at LMD mb1, §11). The WM has
  one tower, so maybe half the ops: **~0.3 s / ~1.2–1.5 s**, nearly independent of the
  microbatch until the GPU saturates at mb ≈ tens.
- **So the microbatch is the lever.** At mb1 × 48 accumulation, a 192-example update costs
  ~36 × 0.3 + 12 × 1.3 ≈ **25 s**, almost all overhead. At mb16 × 3 accumulation × 4 GPUs it is a
  few microsteps of ~1.5–2 s, i.e. **~5 s**.
- **Memory.** ZeRO-2 on 4 GPUs for full 5B: bf16 weights 10 GB + grad partition 2.5 GB + fp32
  master/Adam 15 GB ≈ 28 GB static. FlexPi's LSD activations scale to ~6 MB per token, so ~1.6 GB
  per WM example, and **mb ~24–32 fits in 94 GB** without block checkpointing. Measure first.
- **Diagonal mask at large microbatches.** `update_diagonal_mask` packs off-diagonal examples into
  whole microsteps. At mb16 × 4 GPUs with accumulation 3 (per microstep 64, 48 off-diagonal) it
  yields one mixed microstep. Its random slot fill gives ranks ~9–15 off-diagonal examples each,
  a small lockstep imbalance. Add a **balanced** mode: each rank's microbatch is split exactly
  (e.g. 12 diagonal / 4 off-diagonal of 16), so every rank does the same two batched calls per
  microstep. At mb ≥ 4 this is simpler and exactly load-balanced.
- **Host syncs.** The pipeline asserts `isfinite` on ~8 tensors per step (`training_loss`,
  `model_fn_wan_video`). Each is a sync that drains the GPU queue; gate them behind a debug flag.
- **After that: CUDA graphs / compile** (efficiency notes §11). The WM has static shapes, no
  mask, one stream and no teacher, and a single-branch batched call per branch: two graphs.
  Simpler than FlexPi, and it would show whether the ~2.4× (graphs) and fusion estimates hold.

The effective batch is a research decision. The SFT recipe uses 4 (8 upstream) at lr 1e-6/1e-5;
LSD's 75/25 mixture wants a larger batch (FlexPi uses 192).

## 5. Decisions needed before starting

1. **Environment.**
   - (a) Run the WM in `fm_env` (torch 2.7.1): the forward-AD workarounds and TVM are
     validated there. Needs a check that diffsynth imports and trains under 2.7.1; both repos
     pin `transformers==4.49.0`.
   - (b) Stay on `diff_env` (torch 2.11, Triton 3.6): re-validate the forward-AD path (LayerNorm
     JVP, SDPA JVP, dual-aware checkpoint) and TVM. Newer torch may remove workarounds, and
     `torch.compile`'s forward-AD support may differ, which matters for §11.

   **Recommendation:** develop against (a), run the ported CPU tests in (b) as a gate, and pick
   the one that passes. The WM's existing SFT checkpoints are env-independent.
2. **Where the shared code lives.**
   - (a) Vendor copies into diffsynth: fastest, but the copies drift.
   - (b) **Extract a small shared package** (e.g. `faster-wams/flowmap_core`: `flowmap.py`,
     `flowmap_self.py`, `attention.py`, `checkpoint.py`, `normalization.py`, `jvp_attention/`,
     `flowmap_ema.py`, `step_profile.py`, the latent-cache storage layer), installed editable in
     both envs, with FlexPi re-exporting it.

   **Recommendation: (b)**, since the goal is reuse across models. It touches FlexPi imports only
   (moves plus re-exports; its tests must stay green).
3. **Objective and initialisation.** LSD from the SFT WM checkpoint (no teacher; matches FlexPi's
   self-distillation arm). LMD is also feasible: the frozen SFT WM as teacher costs +10 GB bf16,
   affordable at 256 tokens.
4. **Time sampling.** Replace the discrete Wan timestep + `training_weight` with FlexPi's
   continuous σ pairs (strip, `schedule_shift` to match Wan's shift) and the learned time weight.
5. **EMA.** Whether to evaluate EMA weights (recommended; background EMA is ready).
6. **`context_noise_sigma`.** Keep it at 0 (current launchers) or support it with an online
   frame-0 encode.

## 6. Extracting `flowmap_core` (phase 1 in detail)

Step-by-step implementation plan and progress log: `.claude/context/08-flowmap-core-plan.md`.

> **Status (2026-09-26): phase 1 done.** `flowmap_core/` exists in this repo (README lists the
> API) and is editable-installed in `fm_env`. The old FlexPi paths are `sys.modules` aliases of
> the core modules, not `import *` re-exports (so monkeypatches and module globals still work).
> Verified: FlexPi CPU suite unchanged, CPU fingerprint bit-identical, GPU `lmd_full`/`lsd_off`
> bit-identical with `explicit` attention and within run-to-run spread with `tvm`,
> 28 core-only tests with FlexPi unimportable (including a tiny generic DiT trained with all
> seven objectives). Deviations from the text below:
> - the latent-cache split moved a generic `latent_store.ArrayStore` to the core; FlexPi's
>   `LatentCache` (DINO row layout) subclasses it;
> - `teacher_checkpoint` went to the core config (any distillation needs it);
> - FlexPi's `pyproject.toml` notes the in-repo package instead of listing it (it isn't on
>   PyPI; `docs/INSTALL.md` installs it).
> Details and numbers: `.claude/context/08-flowmap-core-plan.md` progress log.

Checked 2026-09-26 against the FlexPi tree. The move is behaviour-preserving: the core modules
are the files FlexPi runs today, moved, with FlexPi importing them back.

**Moves to the core (pure torch, no FlexPi imports):**

| Module (FlexPi path) | Contents | Notes |
|---|---|---|
| `models/helpers/flowmap.py` | residuals (`map_residuals`), `affine_flow_map`, `sample_level_pair_strip`, `tuple_jvp`, FD/AD derivatives | Config split, see below; drop `without_checkpointing` (no callers; it names FlexPi attributes). |
| `models/helpers/flowmap_self.py` | `update_diagonal_mask`, `slice_batch`, `TimeLossWeight` | + balanced per-rank split (§4) later. |
| `models/helpers/attention.py`, `jvp_attention/` (incl. vendored TVM + license) | forward-AD SDPA, explicit/TVM backends | The backend switch is process-global, set by the host model. |
| `models/helpers/checkpoint.py` | dual-aware activation checkpoint | |
| `models/helpers/normalization.py` | `ForwardADLayerNorm` | |
| `utils/flowmap_ema.py` | background EMA | |
| `utils/step_profile.py` | trace summary, sync sites | |
| `utils/deepspeed_compat.py` | ZeRO hook-count patch | No-op on DeepSpeed ≥ 0.18.7. |
| `datasets/latent_cache.py` (storage half) | `LatentCache` (memmaps, manifest, `done`, flush), `select_windows`, `to_numpy`/`from_numpy`/`load`, fingerprint diff | |

**Stays in FlexPi** (imports from the core): `flowmap_training.py` (joint multi-stream loss),
`flowmap_diagnostics.py`, `adaptation.py` (LoRA/adapter/heads, `clone_teacher`), the latent
cache's encoder fingerprint, DINO row plan, `CachedLatentDataset` and `build_inputs_from_cache`,
the trainer, and flex-joint.

**The two couplings to cut:**
1. **`FlowMapConfig`** mixes objective settings with FlexPi ones.
   - Objective settings: objective, strip width, time sampling, diagonal/map weights, learned
     time weighting, EMA decays, `dt_method`, `fd_eps`, `detach_derivatives`,
     `lmd_teacher_gradient`, `jvp_attention`, `schedule_shift`, `num_inference_steps`.
   - FlexPi settings: `streams` validated against FlexPi's four stream names (`STREAMS`), `mode`
     (LoRA/adapter/heads), `rank`, `lora_alpha`, `teacher_checkpoint`, `initialization`.
   - Plan: a core `FlowMapObjectiveConfig` with the objective fields and their validation, and
     FlexPi's `FlowMapConfig` subclassing it with the rest. Field names, defaults and
     `asdict()` output stay identical, so Hydra configs and saved checkpoints (`flexpi.py:2560`
     stores `asdict(self.flow_map)`) load unchanged.
   - (The WM could even use `FlowMapConfig` as is with `streams=("video",)`; the split is for
     cleanliness.)
2. **`latent_cache.py`** splits into the storage layer (core) and the FlexPi sample format
   (stays).

**Compatibility shims.** FlexPi imports these modules at ~12 sites (`flexpi.py`,
`wan_video_dit.py`, `action_dit.py`, `mot.py`, `gradient.py`, `runtime.py`, `trainer.py`,
`flowmap_training.py`, `flowmap_diagnostics.py`, `adaptation.py`). The old module paths become
one-line re-exports (`from flowmap_core.flowmap import *` plus the private names the tests use),
so no import site has to change. They can be migrated later.

**Packaging.** `faster-wams/flowmap_core/` with a `pyproject.toml`, `pip install -e` into
`fm_env` (and `diff_env`). No new dependencies beyond torch (+ triton for TVM, numpy for the
cache). FlexPi's `pyproject.toml` lists it as a dependency.

**Gates:**
- FlexPi's CPU suite (88 tests + 46 subtests; `tests/test_flowmap*.py`, `test_jvp_attention.py`,
  `test_flex_joint_share.py`, `test_latent_cache.py`, `test_step_profile.py`,
  `test_deepspeed_compat.py`) passes **unchanged**. The core-only tests move with the package,
  and FlexPi keeps copies that import through the shims.
- **Bit-identical GPU check:** `scripts/profile_flowmap_step.py` (`lmd_full`, `lsd_off`, TVM on)
  on the LIBERO cache with a fixed seed gives the same loss and gradient norm before and after.
  The code is only moved, so any difference is an import or packaging mistake.
- One `trainer_timing.sbatch` run (`zero2_tvm_mb2`) matches the §12 timings.

**Effort:** ~1–2 days for the split, shims and gates, within phase 1's 2–3 days.

## 7. Work plan

| Phase | Work | Gate | Effort (est.) |
|---|---|---|---|
| 0 | Decisions 1–2; env check (diffsynth under torch 2.7.1, or FlexPi forward-AD tests under 2.11) | environment chosen | 0.5 day |
| 1 | Shared `flowmap_core` package; FlexPi re-exports; move its CPU tests (§6) | FlexPi's 88 tests + 46 subtests unchanged; bit-identical GPU step | 2–3 days |
| 2 | WM DiT surgery (Δ-embedding, LayerNorm, JVP SDPA, checkpoint, reinit gate), per-sample timesteps, `wm_flowmap_loss` | tiny-WanModel CPU tests: untrained Δ path byte-identical to FM; JVP vs finite differences; batched == per-example loop; anchor frames untouched | 2–3 days |
| 3 | RLinf latent cache: window reader, static windows, encode/verify; cached-batch collate for mb > 1 | bit-exact against online encode on sampled windows | 2 days + encode job |
| 4 | Training loop: microbatch collate, accumulation, balanced diagonal split per microstep, EMA (background), sync-free metrics, timing harness | one LSD update runs on 1 GPU at mb 1/8/16; finite loss | 2–3 days |
| 5 | 4-GPU timing and memory sweep (mb 4–32) | per-update time and peak memory in a notes table | 1–2 days |
| 6 | Flow-map sampler in `WanVideoPipeline`; `eval_wan_rollout.py --steps 1/2/4` | rollout PSNR/SSIM/LPIPS vs FM at 5 steps on the **action-matched policy-rollout holdout** (not human demos: see the WM repo's `wm_scaling_investigation.md` F7/F8) | 1–2 days |

Total ≈ 2–3 weeks to a validated short LSD run plus eval. The CUDA-graph/compile work (§11) comes
after, and is best prototyped on the WM.

## 8. Risks and open questions

- **torch 2.11 forward AD.** Unknown until tested; it may break the dual-aware checkpoint or
  remove the need for workarounds.
- **Triton 3.6 and the TVM kernels.** Unvalidated. Only matters if TVM is wanted, since the
  explicit path is viable at L = 256.
- **Action conditioning through the time embedding.** `action_mlp2` is added to `t`, and Δ is
  added alongside it. Correct by construction (the Δ path is zero at init), but the interaction
  of action and Δ conditioning in AdaLN is untested.
- **Anchor frame 1 is noised but labelled t = 0.** Keep it as a fixed conditioning input of the
  map (zero velocity), exactly as in FM training.
- **Autoregressive exposure bias.** Few-step maps may change rollout drift. Evaluate over the full
  horizon (horizon curves), not per chunk only.
- **Large-batch hyperparameters.** Moving from effective batch 4 to ~192 changes the optimisation
  (lr, steps). Budget a short sweep.

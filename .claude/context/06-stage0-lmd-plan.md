# 06 — Stage 0: LMD distillation of flexpi-libero → flow map (implementation plan)

> **Purpose.** A standalone, reusable spec for **Stage 0** of the flow-map conversion (see the
> ladder in `05-flowmap-conversion.md §6`): distill the released **flow-matching** `flexpi-libero`
> teacher into a **flow-map** student that samples in **K=1–2** steps instead of 4, at the *same*
> chunk size and modalities. Base stays noise→data (Axis 1 only — amortize the ODE); the
> chunk→chunk coupling is Stage 1, later.
>
> Grounded in: the papers (`literature/flowmap_notes.md`, `literature/_docling/{flow_map_matching_24,self_dist_flow_maps_25}.md`), the heads map (`01-flowmatching-heads.md`), and two code investigations (2026-09-11). Line numbers are a snapshot — verify before editing.

---

## 0. Scope, hardware, prerequisites

- **What we distill:** all four joint-stepped heads (video/pointmap/dino/action). Action-only is a
  cheaper *first-signal* sub-config, but LIBERO's released eval runs the **joint** path (all
  `infer_joint=true`, K=4), so the headline K→1–2 number needs the visual heads too.
- **Teacher:** `checkpoints/released/flexpi-libero/checkpoints/weights/step_010860.pt` (+
  `dataset_stats.json`, `config.yaml`). Frozen.
- **Hardware:** a full node — **8×H100/80GB**. Memory is not the constraint (see §6). Text-embed
  cache already built (`data/text_embeds_cache/libero`); LIBERO dataset present.
- **Nothing distillation-related exists in the repo yet** — teacher/student, EMA, JVP, LoRA: all
  greenfield. The two clean precedents to copy are the `dino_pred_x0` config wiring and the
  `flex_joint.enabled` dispatch in `training_loss`.

## 1. Objective — Lagrangian Map Distillation (LMD)

Flow map (affine parameterization, FMM eq. 4.1):
```
X_{s,t}(x) = x + (t − s) · v_{s,t}(x)
```
where `s,t` are noise levels (σ∈[0,1]; the scheduler's σ runs 1=noise → 0=clean). Per stream, the
**LMD loss** (FMM Cor. 3.7) regresses the map's time-derivative onto the frozen teacher velocity:
```
L_LMD = E_{(s,t)} E ‖ ∂_t X̂_{s,t}(I_s) − b_t^teacher( X̂_{s,t}(I_s) ) ‖²
```
- `I_s = add_noise(x0, ε, s)` — the existing noise→data interpolant (Stage 0 base = noise).
- Teacher `b_t^teacher` is frozen ⇒ the stop-grad is automatic. **No flow-matching diagonal term
  is needed** (that's only for teacher-free *self*-distillation).
- **LMD not EMD:** Eulerian needs the spatial Jacobian `∇X̂` (near-singular → blurry/unstable);
  Lagrangian uses only `∂_t X̂`, trains faster, lower FID. Both bound `W₂` to the teacher.
- Flex-reduce each stream's loss exactly like the FM losses (`_flex_reduce_per_sample_loss`); keep
  `m_in`/`m_out` + cross-modal forcing semantics untouched.

## 2. The flow-map contract at inference (nearly free)

Because rectified flow gives `dx/dσ = v`, a "jump" and an Euler step are the **same update**
`x + v·δ`. So flow-map inference = the existing `WanContinuousFlowMatchScheduler.step` with
`build_inference_schedule(K=1|2)` (big `δ`) and a head that also sees the **target level**. Add a
sibling `FlowMapScheduler` sharing `_phi`/`add_noise`/`build_inference_schedule`; the only real new
thing is threading the second time input to the head. The expensive inference stack
(KV-cache/CUDA-graph/TensorRT/FlexAttention) is structurally untouched.

## 3. Three training modes (one pipeline, a `flow_map.mode` switch)

All warm-start from the teacher; all use LMD; frozen teacher = automatic stop-grad. They differ in
**trainable set + map form + how `∂_t` is taken**:

| Mode | Trainable | Map parameterization | `∂_t X̂` estimator | JVP through backbone? |
|---|---|---|---|---|
| **`adapter`** (ψ) | tiny `ψ` head / stream | higher-order `x+(t−s)b̂_s+½(t−s)²ψ_{s,t}` | AD through `ψ` only | **no** |
| **`lora`** | LoRA (~1–2%) + 2nd time-embed | affine `x+(t−s)v_{s,t}` | finite-diff (default) / AD | yes |
| **`full`** | all `mot`+heads (+2nd time-embed) | affine `x+(t−s)v_{s,t}` | finite-diff (default) / AD | yes |

- **`adapter`**: `b̂_s` = frozen teacher velocity at the *start* level `s` (constant in `t`), so
  `∂_t X̂ = b̂_s + (t−s)ψ + ½(t−s)²∂_tψ` differentiates **only through the small `ψ`** → no
  forward-AD through attention, memory-trivial, and `ψ=0` recovers the teacher's Euler step exactly
  (can't-do-worse-than-teacher floor). Under-capacity risk at aggressive K=1.
- **`lora`/`full`**: single-network affine map; the head gains a second time embedding (zero-init
  so student≈teacher at start). `∂_t` runs through the backbone (see §4). `full` is the most
  expressive / standard recipe; `lora` is the middle ground. **No LoRA tooling in the repo** — add
  `peft` or a manual low-rank wrap on the attn/FFN projections.

This sweep also happens to test the JVP-free vs JVP paths. Run all three, A/B/C in §9.

## 4. `∂_t X̂` — three estimators

The `∂_t` is w.r.t. the **scalar** target time, so a JVP carries **one tangent per op** →
~**2× a forward** in time+memory, **independent of trainable param count** (param-count cost is the
ordinary backward you pay in any finetune). Options:

1. **`ψ`-only AD** (`adapter` mode): trivial, safe, no attention involved. Preferred where it applies.
2. **Finite-difference in `t`** (`lora`/`full` default): `∂_t X̂ ≈ (X̂_{s,t+ε} − X̂_{s,t})/ε`, two
   forwards through the *existing* grad-safe path — **keeps flash attention**, no forward-AD op
   coverage needed, compatible with **gradient checkpointing**. O(ε) truncation error. Robust first
   choice, especially for the long visual streams.
3. **True forward-mode AD** (`torch.autograd.forward_ad`, dual on the timestep — *not*
   `torch.func.jvp`, to avoid functionalizing 6B params): FMM's stated preference, but requires
   pinning the **MATH SDPA backend** (fused SDPA has no forward-AD formula) which *materializes the
   S×S attention* → quadratic in sequence length. Fine for action (~32 tokens), costly for the
   visual streams (thousands). Complex RoPE (`view_as_complex`/`view_as_real`) has forward-AD
   formulas in torch 2.7 but **validate empirically**. Keep `mot_checkpoint_mixed_attn=false` for
   this path (checkpointing ⊥ forward-AD).

**Decision (revised after A100 validation, §12e):** **forward-AD is the default for the action
stream** (exact; validated end-to-end through the real `ActionDiT`; MATH-SDPA negligible at ~32
tokens). **Central finite differences (eps≈1e-4 σ) for the visual streams** where MATH-SDPA is
quadratic-expensive (~1.6e-3 rel. error, 3 fused-SDPA forwards, checkpoint-compatible). `ψ`-AD for
`adapter`. **Never forward differences in fp32** — see the calibration rule in §12e.

## 5. Teacher velocity query `b_t(x_t | conditioning)`

Grad-safe primitives already exist:
- **Action-only:** prefill the clean first-frame anchor KV cache **once** per batch (reuse the
  anchor block of `_base_infer_action`, `flexpi.py:1725-1855`; `mot.prefill_video_cache`,
  `mot.py:383-508`), then call **`_predict_action_grad_safe`** (`backbone.py:1411-1450`) at any
  `(x_t, t)`, batched, under `no_grad` for the teacher. Cheapest — video not recomputed.
- **Joint:** **`_predict_joint_noise_unified_impl`** (`flexpi.py:1468-1603`) — drop the `no_grad`
  wrapper, returns the per-stream velocity 4-tuple (`:1603`), applies DINO x0→v internally
  (`:1590-1597`). Supply all streams' noised latents + per-stream timesteps + `context`/mask +
  `fuse_vae_embedding_in_latents`; **keep `flex_block_attention=False`** (dense SDPA mask); the
  **caller writes the frame-0 clean anchors** (the impl doesn't re-clamp). This is the same velocity
  primitive as the training `self.mot(...)` call (`flexpi.py:1242`).

The teacher is a frozen `FlexPi` instance called through these; `no_grad`.

## 6. Distributed setup (8×H100)

- **ZeRO-2 data-parallel** (repo already ships `scripts/accelerate_configs/accelerate_zero1_ds.yaml`
  + a zero2 config; `zero3_init_flag:false`). Full-`full` per-GPU: weights 12 GB + grads (ZeRO-2
  shards → ~1.5 GB) + optimizer (72 GB / 8 = 9 GB) + **frozen teacher ~12 GB** + activations ≈
  **~35 GB + activations** on 80 GB. Comfortable. `adapter`/`lora` optimizer is tiny → ZeRO-1 or
  plain DDP is enough.
- **Teacher placement:** one frozen replica per rank, **not** passed to `accelerator.prepare` (so
  DeepSpeed never wraps/shards/optimizes it), `eval()` + `requires_grad_(False)`, called under
  `no_grad`. **Share the frozen encoders** (VAE/DINOv3/umT5) by reference. For `adapter`/`lora` the
  teacher *is* the shared frozen backbone — no separate copy.
- **No tensor/pipeline parallelism** — 6 B fits per GPU; only the optimizer needs sharding.
- **Gradient checkpointing** (`mot_checkpoint_mixed_attn=true`) is compatible with the finite-diff
  `∂_t` path (not with forward-AD) → use it to fit long visual-stream activations / bigger batch.
- Trainer wiring: `Wan22Trainer` gathers trainable params from `self.model` only
  (`trainer.py:121`), so the teacher is auto-excluded from AdamW/DeepSpeed. Add teacher
  instantiate+load+freeze in `Wan22Trainer.__init__` and the teacher `no_grad` forward in the step
  (`trainer.py:1560-1565`).

## 7. Code touchpoints (file:line)

1. **Config flag** — `model.flow_map` + a `flow_map:` block (`mode`, `K`, `psi_*` dims, `strip_width`,
   `dt_method`, `eps`). Thread via the `dino_pred_x0` path: yaml (`configs/model/flexpi.yaml`) →
   `create_flexpi` kwarg (`runtime.py:72,101`, passed `:216`) → `from_wan22_pretrained`
   (`flexpi.py:585/759`) → `_base_init` store (`flexpi.py:150`-style). Model built at
   `runtime.py:427`; eval builds the same way (`eval_libero_single.py:1488`).
2. **Objective dispatch** — at the top of `training_loss` (`flexpi.py:2945`, mirroring the
   `flex_joint.enabled` branch `:2954-2955`): `if self.flow_map: return self._flowmap_training_loss(...)`.
   Leave `_base_training_loss` (`:1110-1383`) as the FM default.
3. **`_flowmap_training_loss`** — reuse `build_inputs` + the pre-dit/MoT scaffolding; per sample draw
   a level pair `(σ_a,σ_b)` (strip; §8), build `I_{σ_a}` (`add_noise`), `X̂`, teacher `b` at
   `(X̂,σ_b)` (§5), LMD residual (§1), `∂_t` per §4. Per-stream losses flex-reduced like `:1371-1383`.
4. **Scheduler** — `FlowMapScheduler` beside `WanContinuousFlowMatchScheduler`
   (`models/schedulers/scheduler_continuous.py`): reuse `_phi`/`add_noise`/`build_inference_schedule`
   (`:44-109`); `step` is the existing Euler with big `δ` (`:119-153`).
5. **Head second time input** (`lora`/`full`) — add an `s`-embedding alongside `t` in the DiT time
   path (`sinusoidal_embedding_1d`/`time_embedding`/`time_projection`), zero-init the new path. `ψ`
   head (`adapter`) — small per-stream head on the frozen backbone features + `(s,t)`; register in
   `_POINTMAP_CKPT_KEYS`/`_DINO_CKPT_KEYS`-style key sets (`flexpi.py:2477-2478`).
6. **Inference** — behind the flag, use `build_inference_schedule(K=1|2)` and feed the head `σ_b`;
   action loop `flexpi.py:1856-1868`, joint per-stream step `flexpi.py:3641-3664` (+ frame-0
   re-clamp already there).
7. **Checkpoint** — loader is `strict=False` with a "missing key → keep fresh init" path
   (`flexpi.py:2519-2522`; super `backbone.py:1485-1518`); load released weights into both teacher
   and student; new zero-init params tolerated. Add ψ/time-embed keys to the save/load sets
   (`flexpi.py:2477-2478,2486-2537`).

## 8. Methodology from the papers

- **Affine param** `x+(t−s)v` (FMM eq. 4.1) for `lora`/`full`; higher-order frozen-`b̂`+`ψ` (SD
  "general representations") for `adapter`.
- **Don't learn one-step directly.** FMM: direct one-step is worse; train on a **strip
  `|σ_a−σ_b| ≤ 1/K`** and reach one step by **PFMM** (progressive collapse of the K-step map). So
  **target K=2 first**, get K=1 by PFMM if wanted — do not force K=1 from scratch.
- **Loss weighting `w_{s,t}`** for the large gradient-scale variance across level pairs (SD uses a
  learned EDM2-style weight). Start from the existing `training_weight` (`scheduler:65-73`) and
  generalize to two times.
- **`∂_t` via forward-mode AD** is FMM's stated preference; we substitute finite-diff where AD forces
  the slow MATH attention (§4) — a deliberate, justified divergence on this attention stack.
- Stop-grad on the teacher = free (frozen). No FM diagonal term (pure distillation).

## 9. Validation ladder + eval matrix

1. **Init sanity:** with `ψ=0` (or zero-init `s`-path), the map ≈ one teacher Euler step (assert
   numerically).
2. **Overfit a tiny batch** → LMD loss → 0.
3. **Short distill run** on LIBERO (text cache ready), one mode, small step budget.
4. **A/B/C eval** — LIBERO, teacher K=4 (Euler) baseline vs each mode at **K=2 and K=1 (PFMM)**:
   - first, **action-only** regime (`infer_joint=false`, the ~60 ms fast path) for cheapest signal;
   - then **joint** regime (released config) for the headline number.
   Metrics: success rate + wall-clock latency + (sanity) action MSE-vs-teacher.

| | K=4 (teacher) | K=2 | K=1 (PFMM) |
|---|---|---|---|
| adapter | baseline | ? | ? |
| lora | baseline | ? | ? |
| full | baseline | ? | ? |

## 10. Risks / open decisions

- **`adapter` capacity** at K=1 (may need PFMM or graduating to `lora`/`full`).
- **finite-diff `ε`** choice (truncation vs fp noise) — sweep; bf16 may need fp32 for the difference.
- **Complex-RoPE forward-AD** correctness (only if we enable true AD) — validate empirically.
- **Joint-path caveats** for AD only: in-place anchor-timestep zeroing (`flexpi.py:1189,1210`) and
  the DINO σ-clamp (`helpers/dino.py:177`) sit in the `t`-tangent for those streams; finite-diff
  sidesteps both.
- **DINO predicts x0** (not v) — its "velocity" is the converted quantity; the map/LMD target must
  use the same v-space convention (`_dino_x0_to_velocity`).
- Decide whether to also generalize `w_{s,t}` now or start unweighted.

## 11. First implementation steps (ordered)

1. Add the `model.flow_map` config flag + `flow_map` block (4-touchpoint), default off.
2. `FlowMapScheduler` (mostly reuse) + head second-time-input plumbing (zero-init).
3. Teacher: instantiate + load + freeze in `Wan22Trainer`; wire `no_grad` velocity query (action
   primitive first).
4. `_flowmap_training_loss` for **`adapter`, action-only** (JVP-free) — the fastest correct first
   signal; run the init-sanity + overfit checks.
5. Extend to `lora`/`full` (affine param + finite-diff `∂_t`) and to the joint streams.
6. `FlowMapScheduler` inference wiring + the A/B/C eval matrix on 8×H100.

See also: `05-flowmap-conversion.md` (the full ladder + invariants), `01-flowmatching-heads.md`
(per-head detail), `literature/flowmap_notes.md` (theory).

---

## 12. Step A (implemented 2026-09-11): Δ time-embedding + `∂_t` machinery for `lora`/`full`

The first code increment. Scope: make every expert *capable* of two-time conditioning and add the
`∂_t` estimators + affine-map helpers. **Not** in this step: the `model.flow_map` config flag, the
LMD loss, the teacher wiring, inference (next increments). **Invariant: with the Δ path disabled
(the default), the model is byte-identical to the flow-matching model.**

### 12a. Architecture choice (from the papers, §8)
`t`-conditioning = positional(`s`) **+** positional(`Δ`), `Δ = t_target − t_input`, fed to the
standard AdaLN/FiLM — the Boffi 2025 recipe (embed `s` and `(t−s)`, *add*, positional not Fourier).
FlexPi's `sinusoidal_embedding_1d` is positional and `Δ` is signed (cos/sin handle sign), so no
new embedding function is needed.

### 12b. What changed, and why it's safe
- **`ActionDiT` / `WanVideoDiT`** (`action_dit.py`, `wan_video_dit.py`):
  - new ctor kwarg `use_time_delta: bool = False`; when True, a `time_embedding_delta` module
    (same shape as `time_embedding`: Linear→SiLU→Linear) whose **final Linear is zero-init**;
    when False the attribute is `None`.
  - new method **`embed_time(timestep, timestep_delta=None)`** returning the `t` conditioning.
    Disabled ⇒ exactly `time_embedding(sinusoidal(timestep))`. Enabled ⇒ that **plus the
    centered Δ term `f(Δ) − f(0)`**, `f = time_embedding_delta ∘ sinusoidal`. Centering makes the
    path contribute **exactly zero at Δ=0 for any weights** — so Δ=0 is *always* the teacher's
    own single-time path (the tangent condition `v_{t,t}=b_t` baked into the architecture) and
    clean anchor tokens (Δ=0) stay untouched throughout training. `f(Δ)−f(0)` spans exactly the
    functions vanishing at Δ=0 (the "jump corrections"), so no expressivity is lost. Zero-init
    additionally makes an *untrained* model equal the teacher for any Δ. A missing Δ means Δ=0.
    **Found by the CPU test:** without centering, pinning anchors to Δ=0 only removed the
    Δ-*dependence*, not the constant `f(0)` offset — anchors still moved by ~0.11 once the path
    went live. Keep this in mind if anchor behavior ever looks off.
  - `pre_dit`/`forward` accept `timestep_delta` and route `t` through `embed_time`. Because the
    video/pointmap `Head` FiLM reads `pre_state["t"]`, the heads see the fused `t` with no extra
    plumbing. The action head is a plain Linear (no FiLM) — nothing to thread.
  - **Video per-token Δ**: built exactly like the per-token timesteps — `Δ` broadcast per frame, then
    **frame-0 (anchor) forced to Δ=0**, mirroring `token_timesteps[:,0,:]=0`.
- **Two compatibility traps, fixed:**
  1. `ActionDiT.from_pretrained` demands every non-skipped key from the Wan-derived backbone
     payload (`strict` merge, `action_dit.py:~146,194-201`) → `"time_embedding_delta."` added to
     `ACTION_BACKBONE_SKIP_PREFIXES` so the new zero-init module is *kept fresh*, not demanded.
  2. `_validate_dit_config` whitelists keys from `inspect.signature(WanVideoDiT.__init__)`
     (`loader.py:62-70`) → adding the ctor kwarg auto-registers it. The Wan2.2 weight load is
     `strict=False` (`loader.py:120`), so the missing `time_embedding_delta` keys are tolerated.
- **`flexpi.py:_build_stream_t_mod`** (DINO/pointmap block conditioning) gains an optional
  `delta_per_token` (default `None` ⇒ identical) and routes through `video_expert.embed_time`.
  All three visual streams **share** `video_expert`'s time modules (as they already share
  `time_embedding`), so one Δ module serves video+dino+pointmap.
- **`_base_training_loss` is untouched** (the FM baseline). The pointmap head-FiLM `pt_t`
  (`flexpi.py:~1216-1221`) is computed inline there; the future `_flowmap_training_loss` computes
  it via `embed_time` with Δ.

### 12c. `models/helpers/flowmap.py` (new; pure functions, no state)
- `delta_timestep(t_input, t_target)` → signed Δ in timestep units.
- `affine_flow_map(x_s, v_st, sigma_s, sigma_t)` → `x_s + (σ_t − σ_s)·v_st` (σ∈[0,1], per-sample
  broadcast). With `v_st = b_s` (teacher) this is one Euler step — the init-sanity check.
- `dX_dt_finite_difference(map_fn, σ_t, eps=1e-4, scheme="central")` → `(X̂(t), (X̂(t+ε) − X̂(t−ε))/2ε)`,
  O(ε²), three fused-SDPA forwards (the third gives the *exact* `X̂(t)` teacher-query point);
  checkpoint-compatible. **The visual-stream estimator.** `scheme="forward"` (O(ε), 2 forwards) is
  kept for diagnostics only.
- `dX_dt_forward_ad(map_fn, t_target)` → exact JVP via `torch.autograd.forward_ad` (dual on the
  target time), wrapped in `sdpa_kernel([MATH])` because fused SDPA has no forward-AD formula.
  Requires checkpointing **off**. Opt-in.
- `lmd_residual(dX_dt, b_teacher)` → `∂_t X̂ − b_t^teacher(X̂)`; the loss is its MSE, flex-reduced
  by the caller.
- `sample_level_pair_strip(...)` → `(σ_a, σ_b)` with `σ_b < σ_a` (denoise direction) and
  `|σ_a−σ_b| ≤ strip`, for the "train a strip, target K=2" recipe.

### 12d. Verification (CPU, no GPU needed)
1. Δ-off vs Δ-on-at-init produce **bit-identical** outputs on tiny `ActionDiT`/`WanVideoDiT`; with a
   *live* Δ path, Δ=0 still equals Δ-off (centering) and frame-0 anchors are exactly unaffected.
2. `dX_dt_finite_difference` ≈ `dX_dt_forward_ad` on a toy two-time net.
3. `affine_flow_map` with the teacher velocity == one Euler `scheduler.step`.

### 12e. A100 validation results (2026-09-11) — and a numerics finding that changed §4
Tests: `scratch_tests/test_flowmap_stepA.py` (CPU, 21/21), `test_flowmap_stepA_gpu.py`,
`diag_fd_truncation.py` (all under `/gpfs/scratch1/shared/faster-wams/scratch_tests/`).
- **bf16-on-CUDA** (the training dtype): Δ-on@init == Δ-off *bit-identical*; centering exact
  (Δ=0 with live weights == Δ-off).
- **Forward-mode AD runs through the real `ActionDiT`** — SDPA under the MATH backend, complex
  RoPE, AdaLN — and its `X̂` matches the FD path exactly. The plan's flagged risk (§4 item 3,
  complex-RoPE forward-AD) is **cleared empirically**.
- **Finding:** forward-difference `∂_t` disagreed with exact AD by **4.0%** at eps=1e-4 σ. Diagnosed
  as pure O(ε) truncation, not a bug — three fingerprints: error *halves as ε halves*
  (3.99→2.01→1.04 %), *persists in fp64* with the same slope (precision-independent), and
  *central differences collapse it 25×* (1.6e-3 at the same ε; O(ε²) — 2.85e-5 at ε=1e-5 in fp64).
  Root cause: `sinusoidal_embedding_1d` embeds Δ in **timestep units** with a top frequency of
  **1 rad/unit**, and the ×N (=1000) chain from σ to timestep makes forward-FD truncation
  ≈ ε·N/2 (0.1 timestep units ⇒ ~5%). fp32 roundoff floors forward-FD at ~8e-3 (ε≈1e-5), so it has
  no good ε. **Calibration rule: `ε·N ≪ 1` timestep unit; use central differences (ε≈1e-4 σ) or AD.**
- Consequence: §4 decision revised — AD default on action, central-FD on visual, forward-FD never.

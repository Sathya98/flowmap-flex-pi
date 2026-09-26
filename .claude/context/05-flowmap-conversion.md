# 05 — WAM → flow-map conversion: working notes & committed plan

> **Status: direction chosen (2026-09-10).** Grounded in the four papers in
> `/gpfs/scratch1/shared/faster-wams/literature/` (see `literature/flowmap_notes.md` for the
> distilled theory) and a code read of the temporal/chunk structure. The committed shape is a
> **3-stage ladder** (§6). Line numbers are a snapshot — verify before editing.

Companion docs: `01-flowmatching-heads.md` (the heads we touch), `00-overview.md` (streams,
masks, objective), `03-dataloaders.md` (the temporal cascade).

---

## 1. Goal

Replace/augment Flex-π's **flow-matching** heads — an ODE **velocity field** `v_θ(x_t,t)`
solved with several Euler steps — with **flow-map** heads `X_{s,t}(x)=x+(t−s)·v_{s,t}(x)` that
jump between levels directly (few-/one-step), while keeping the multi-stream MoT backbone, the
two-mask compute-flex mechanism, the dataloaders, and the training harness intact.

Two payoffs, on **two different axes** (see §2):
1. **Amortize the denoise ODE** → fewer NFEs (K=4 → 1–2) at fixed base = noise.
2. **Change the base from noise to the *previous chunk*** → shorter/straighter transport,
   fewer steps *and* temporal consistency — the user's core hypothesis.

## 2. The core insight — two axes, and where the hypothesis lives

The word "flow" (and "steps") means two different things here; keeping them separate is the
whole key to the design.

- **Axis 1 — denoise axis** (`t`: noise→clean). Flow matching / flow maps live here. "Fewer
  denoising steps" = fewer NFEs on this axis. A flow map `X_{s,t}` amortizes the solve.
  **LMD distillation of the released teacher operates purely here.**
- **Axis 2 — temporal axis** (physical time; chunk `k`→`k+1`). This is the receding-horizon
  rollout.

The hypothesis — *"the change in a small chunk is captured in fewer steps from the previous
chunk than from noise"* — is **Axis 2 reshaping the geometry of Axis 1**. In the stochastic-
interpolant framework it means: **replace the base `ρ0 = N(0,I)` with `ρ0 = previous chunk`**,
i.e. a **data-dependent coupling**
```
I_τ = α_τ·x_prev + β_τ·x_next + γ_τ·z ,   I_0 = x_prev ,  I_1 = x_next
```
Because `x_prev` already sits on the data manifold next to `x_next`, the transport is short and
nearly straight → genuinely fewer steps. This is the SI framework's sample↔sample generality
(family: bridge-matching / I2SB / OT-coupled flow matching; in video: CausVid, Self-Forcing,
Diffusion-Forcing). Flow matching (noise→data) is the one-sided corner of this.

## 3. Code-grounded temporal structure (the constraints) — LIBERO preset

Per plan, conditioned on **one clean observed frame** (frame 0) + proprio + language, the model
jointly denoises:

| Stream | Noised latent shape | Anchor | Generated |
|---|---|---|---|
| video | `[B,48,F=3,h/16,w/16]` (`[B,48,3,28,32]` @448×512) | frame 0 clean | **2 future latent frames** |
| dino | `[B,768,F_d=2,294,1]` (LIBERO `dino_temporal_stride=2`) | frame 0 clean | 1 future |
| pointmap | `[B,48,F=3,h,w]` | frame 0 clean | 2 future |
| action | `[B,32,32]` = horizon 32 × action_dim 32 | **none — fully noised** | all 32 steps |

- Cascade **33→9→3** = raw window frames → RGB anchors (÷`ratio=4`) → VAE latent frames
  (÷4, +1). `robot_video_dataset.py:242,249`; `latent_t=(9−1)//4+1=3` at `flexpi.py:2240,3330`.
- Action = 7-D rotvec `[Δpos3,Δrotvec3,grip1]` scattered into 32 channels (ids
  `[0..5,18]`); one token per step (`action_dit.py:276,286`; `flexpi.py:3361-3364`).
- Per-stream noise + frame-0 re-clamp in `_base_training_loss`: video `:1121-1126`, action
  `:1129-1131` (no clamp), dino `:1135-1139`, pointmap `:1147-1157`. Loss drops the anchor:
  video `pred[:,:,1:]` `:1270-1272`, dino `:1316,1320`, pointmap `:1345`, action all 32
  `:1283-1290`. `λ_*=1`.

**Inference = receding-horizon, single-shot-per-plan, NOT video-autoregressive** (agent-verified):
- LIBERO eval runs the **joint** path (`_infer_action_joint`, dispatch `flexpi.py:3134-3167`;
  denoise loop `:3524-3684`, K=4). Clean anchor written to slot 0 of each visual stream
  (`:3389-3392` video, `:3416` dino, `:3437` pointmap) and **re-clamped after every step**
  (`:3652/:3658/:3664`). Action fully denoised.
- Rollout `run_single_episode` (`experiments/libero/eval_libero_single.py:1048-1179`): predict
  32 actions, **execute `replan_steps=10`, discard 22, re-plan from a fresh single observation**
  (`:1093-1095,1120,1126`). Predicted future *frames* are used only for PSNR viz, **never fed
  back** (`:844-852`).

**Two consequences that drive the plan:**
1. **Actions have no anchor and are regenerated from noise every 10 env steps — including the
   22 steps that overlap the plan just made.** That redundancy is exactly what a
   "from-previous-chunk" map reclaims. Highest-value, lowest-complication target.
2. **Video is not autoregressive** — the previous window enters only via the clean frame-0
   anchor + attention; futures start from noise and are thrown away. Exploiting visual temporal
   closeness needs *new* autoregressive plumbing (feed predicted frames forward).

## 4. The chunk→chunk reframing, per stream

`x_prev` has a concrete, shape-matched meaning (prev/next always share the teacher's chunk
shape, so they interpolate directly — see §5 on the *only* real shape blocker):

- **Action (start here).** `x_prev = concat(a_prev[replan_steps:], pad)` — the previous plan
  **shifted by `replan_steps` (10)**, tail-padded → same `[32,32]`. The map keeps the 22
  overlapping steps ≈ identity and synthesizes the last 10 + corrects for the new observation.
  Short transport by *overlap*, not smoothness. No anchor to complicate it.
- **Video/dino/pointmap (later).** `x_prev =` previous window's future latent frames (temporally
  overlapping). Closeness by *visual smoothness* — the strongest hypothesis case — but partly
  already captured by the frame-0 anchor, and requires autoregressive feedback. Defer to Stage 2.

Scheduler change: today `add_noise` gives `x_τ=(1−σ)x_0+σε`; chunk→chunk replaces the base with
the interpolant above (`I_0=x_prev`). Head gains a **second time input** (`s` as well as `t`, or
step-size `d`). At inference, initialize the ODE state at `x_prev` (τ=0) and jump to τ=1.

## 5. The teacher tension, and how to keep a "distillation" PoC

**The released `flexpi_libero` teacher is a noise→data velocity field.** Its flow map transports
*from noise*. A chunk→chunk student solves a **different ODE** (base = prev chunk, velocity
`c_t=E[İ_τ|I_τ]` over the temporal coupling). So **LMD against the teacher's `b_t` yields the
noise→data map, not the chunk→chunk map** — the "closer base" benefit cannot be distilled from
the frozen velocity.

**Reconciliation (how CausVid/Self-Forcing do it): distill the teacher's *samples*, self-distill
the *map*.** Endpoints: real window → `x_prev`, context, fresh anchor; `x_next` from the teacher's
K-many-step generation (or ground-truth actions). Then learn the bridge `x_prev→x_next` with
**LSD self-distillation on the interpolant** (map target = `İ_τ`, not `b_t^teacher`). This matches
the teacher's output *distribution* — genuinely "student–teacher" — while targeting the correct
ODE.

**"Different chunk sizes" — answered:**
- *prev vs next* (within the coupling): no problem, both are the teacher's shape.
- *student vs teacher* (shrinking the student's output): the **real blocker** — changes
  `num_frames`/`action_horizon`/VAE latent count, so the 32-step teacher can't supply a
  shape-matched target; that's retraining the token layout, not distilling. **Keep student chunk
  = teacher chunk (32) for Stages 0–1.** Note "smaller chunks" is also a *weak* latency lever:
  total NFE ≈ (steps/plan)×(#plans), and #plans is set by `replan_steps`, not chunk size — once
  the map floors steps/plan at ~1, smaller chunks only add replans + drift. The flow map is the
  latency win; chunk size is near-orthogonal (even counterproductive).

## 6. The committed staged plan

- **Stage 0 — LMD, noise→data, chunk size = teacher's, action stream.** Distill `flexpi_libero`
  into a few-step flow-map student on the *denoise axis only*. Tests K=4→1–2, has the Wasserstein
  guarantee, and **de-risks all shared machinery**: second time-input on the head, the `∂_t X̂`
  JVP, a sibling `FlowMapScheduler`, checkpoint-key wiring, keeping flex `m_in`/`m_out` +
  one-way attention intact. **This is the honest PoC; do it first.**
- **Stage 1 — action chunk→chunk bridge (LSD, teacher-defined endpoints).** Flip the base to the
  shifted previous chunk; realize the hypothesis. Reuse Stage 0's head/scheduler; new pieces are
  the interpolant and the pairing. **Roll out on self-predictions during training** to control
  exposure bias (§7.3).
- **Stage 2 — autoregressive video (optional).** Feed predicted frames forward; extend the bridge
  to the visual streams. Biggest change; only after Stages 0–1 work.

Sequences the risk: **plumbing → geometry → autoregression.**

## 7. Risks & mitigations

1. **Latency arithmetic favors large chunks + fewer steps**, not small chunks (see §5). Fix chunk
   size, cut steps.
2. **Stochasticity/multimodality — keep the `γ_τ z`.** A deterministic `x_prev→x_next` can't
   represent multiple futures. Video needs `z`; actions under strong conditioning may tolerate
   small `γ` — keep it as a knob.
3. **Exposure bias / drift** — the map consumes its *own* imperfect previous chunk at inference
   but is trained on teacher/GT `x_prev`. This is the known failure mode of few-step
   autoregressive world models; **Self-Forcing's fix = roll out the student on its own
   predictions during Stage 1 training.** Design it in from the start.
4. **Cold start** — plan 0 has no previous chunk. Keep the Stage-0 noise→data map for plan 0,
   switch to chunk→chunk after (two modes, or a base-agnostic two-time map).
5. **Warm-start helps least when correction is most needed** (surprising observations → large
   transport). Don't hard-pin K; the flow map's variable-K is the escape hatch.
6. **Multi-stream flex interaction** — changing one stream's base while others stay noise→data
   complicates the joint denoise / cross-modal forcing / shared scheduler. **Start action-only**
   (visuals left noise→data) to keep the blast radius small.

## 8. The insertion points (grounded in code)

### 8a. Shared contract — `models/schedulers/scheduler_continuous.py`
`WanContinuousFlowMatchScheduler` owns **all four streams'** target (`training_target =
noise − sample`) and step (`step`, Euler/DPM++2M). Flow map changes both: add a flow-map target
and a `flow_map_step(x_s,s,t)`; consider a sibling **`FlowMapScheduler`** sharing `φ`/`add_noise`
so only target/step differ. For Stage 1, `add_noise` gains the two-sided interpolant base.

### 8b. Precedent to copy — DINO x0-prediction
DINO already predicts a **non-velocity** quantity (`x0`) and adapts to the scheduler via
`_dino_x0_to_velocity` (`helpers/dino.py:154-178`, σ-clamp 0.05) with a **zero-init** final layer
(`helpers/dino.py:181-189`). A flow-map head copies this adapter pattern; add it **behind a config
flag** (like `dino_pred_x0` / the DPM++ solver) so flow-matching stays the A/B default.

### 8c. Training target — `flexpi.py:_base_training_loss` (`:1110-1383`)
Per-stream noising + target `:1121-1157` — where a flow map draws a **pair** `(s,t)` and builds
its target (Stage 1: build `I_τ` from `x_prev`). Loss assembly + λ-combine `:1269-1382` — a
flow-map/consistency loss slots in per stream, **flex-reduced the same way**
(`_flex_reduce_per_sample_loss`); keep `m_in`/`m_out` + cross-modal forcing untouched. A frozen
teacher forward (for LMD / endpoint generation) mirrors `_predict_joint_noise_unified_impl` under
`no_grad`.

### 8d. Inference step loops — the payoff
- **Action fast path** `_base_infer_action` (`:1610-1872`): Euler loop `:1856-1868`
  (`_predict_action_noise_with_cache` → `infer_action_scheduler.step`); KV-cache prefill of clean
  anchors `:1726-1785`. CUDA-graph / loop-compile variant `_run_action_prefill_denoise_loop`
  (`:1922`) — a flow-map step must stay capture-friendly.
- **Joint path** `_infer_action_joint` (dispatch `:3134-3167`; loop `:3524-3684`): per-stream
  `scheduler.step` + re-clamp `:3652/:3658/:3664`; prediction via
  `_predict_joint_noise_unified_impl` (`:1468-1603`). Fewer, larger jumps shrink the 193 ms;
  interacts with `StepSkipController`, the CUDA-graph loop, FlexAttention BlockMask, TensorRT
  (`models/inference_opt/`) — those assume a fixed per-step MoT forward; re-check them.

### 8e. Heads that change (from the `01` matrix)
Video + pointmap share the Wan `Head` (pointmap deep-copied, `flexpi.py:274-276`) → build one
flow-map head, instantiate twice. **Action** — Linear head + fast KV-cache loop → the Stage-0/1
target. DINO — Linear + x0 adapter → easiest to port; may not need a flow map.

## 9. Invariants to preserve
- **Two-mask compute-flex** (`m_in`/`m_out`) + **cross-modality forcing** — the value prop. A
  flow-map objective is still summed over present/cross-modal streams; `m_out` stays a visibility
  mask, not a loss mask.
- **One-way attention** (visual→action) — keeps the action-only KV-cache path exact.
- **Frozen encoders** (VAE, DINOv3, umT5) and the **first-frame-clean (t=0) anchor** per stream.
- **Checkpoint compatibility** — new head params need entries in the ckpt key sets
  (`_POINTMAP_CKPT_KEYS`, `_DINO_CKPT_KEYS`) and the save/load in `flexpi.py:2477-2537`, with
  sane strict-shape behavior for warm starts.

## 10. Stage 0 implementation surface (for the next session)
Rough edit list to scope tomorrow (action stream, behind a flag):
1. **`FlowMapScheduler`** sibling: reuse `φ`/`add_noise`; add `flow_map_target` and
   `flow_map_step(x_s,s,t)`.
2. **Second time-input** on `ActionDiT` (embed `s` alongside `t`, or step-size `d=t−s`) — mirror
   `time_embedding`/`time_projection`; zero-init the new path so the student starts ≈ teacher.
3. **LMD loss** in `_base_training_loss` action branch: draw `(s,t)`, compute `∂_t X̂` via a
   forward-mode **JVP in t**, regress against `b_t^teacher(X̂)` from a `no_grad` teacher
   (frozen `flexpi_libero`), flex-reduced as usual. Keep the FM diagonal term (η≈0.75) so the
   student also retains `v_{t,t}=b_t`.
4. **Inference**: a 1–2 step `flow_map_step` in the action Euler loop (`:1856-1868`), guarded by
   the flag; keep Euler K=4 as the A/B baseline.
5. **A/B smoke on the H100**: `eval_flexpi_libero_single.sh` (K=4 baseline) vs the flow-map
   student at K=1,2 on `libero_object` TASK_ID=0 NUM_TRIALS=2 — compare success + latency.

## 11. Open decisions remaining
1. **Single-network self-distillation** (`v_{s,t}` does both, η-split) vs **higher-order frozen
   teacher** `X_{s,t}=x+(t−s)b̂_s+½(t−s)²ψ_{s,t}` (freeze `flexpi_libero` as `b̂`, train only `ψ`)?
   The latter fits FlexPi's "keep the released weights" instinct and is a strong Stage-0 default.
2. Stage-1 endpoints: **ground-truth** `x_next` vs **teacher-generated** `x_next` (or both)?
3. Stage-1 `γ_τ z`: deterministic bridge vs keep noise (per stream)?
4. First A/B benchmark: **LIBERO** (set up, receding-horizon overlap is ideal) — confirmed
   starting point; RoboTwin later.

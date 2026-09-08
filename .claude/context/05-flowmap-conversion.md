# 05 — WAM → flow-map conversion: working notes

> **Status: pre-planning scaffold.** This captures the goal and the exact code surfaces a
> flow-map conversion touches, so we can plan against real insertion points. The *specific
> flow-map method* (and how much of it we adopt) is still to be decided with the user —
> treat the "design space" section as options, not a committed plan.

## 1. Goal

Replace/augment Flex-π's **flow-matching** heads — which learn an ODE **velocity field**
`v_θ(x_t, t)` solved with several Euler steps at inference — with **flow-map** heads that
learn a map jumping between noise levels directly, `x_s → x_t` (few-step, ideally one-step
for the latency-critical action stream), while keeping the multi-stream MoT backbone,
the two-mask compute-flex mechanism, the dataloaders, and the training harness intact.

Motivation is the same as the paper's latency story, sharpened: action-only already runs
~60 ms at K=4 Euler steps; a one/two-step flow map on the action head (and the joint path)
could cut the denoise loop further and/or improve quality-per-step, especially for the
joint regime (currently ~193 ms with the TensorRT KV-split stack).

## 2. What "flow map" means here (design space — TBD)

A flow map `F_θ(x_s, s, t)` predicts the solution of the flow ODE from level `s` to level
`t` in one evaluation, so inference is a handful of large jumps instead of many small Euler
steps. Concrete families we might draw from (pick per stream — they need not match):

- **Consistency / consistency-trajectory** style: self-consistency along the PF-ODE, so a
  single (or few) evals map any `x_t` to the clean `x_0`. Distillation *or* from-scratch.
- **Shortcut models**: condition the head on a step-size `d` in addition to `t`, train so
  one step of size `d` equals two steps of size `d/2` (self-distillation), giving a
  step-count knob at inference from one checkpoint — conceptually close in spirit to
  Flex-π's "one checkpoint, many regimes" ethos.
- **Flow-map matching / Lagrangian or Eulerian self-distillation**: learn `F_θ(x_s,s,t)`
  with a consistency loss against the model's own short-step rollouts.

Key open question: **distill from the trained flow-matching checkpoint, or train the
flow-map objective jointly/from-scratch?** Distillation is lower-risk (keep the released
weights as a teacher); joint/from-scratch is cleaner but couples to the multi-stream loss.

## 3. The insertion points (grounded in the code)

All references are the same ones detailed in `01-flowmatching-heads.md`.

### 3a. The shared contract — `models/schedulers/scheduler_continuous.py`
`WanContinuousFlowMatchScheduler` is where **all four streams'** training target
(`training_target` = `noise − sample`) and inference step (`step`, Euler / DPM++2M) live.
A flow map changes both:
- **Target**: add a flow-map target (e.g. a consistency target `F_θ(x_s,s,t)` vs a
  teacher/self short-step rollout) alongside/instead of the rectified-flow velocity.
- **Step**: add a `flow_map_step(x_s, s, t)` that takes the large jump; keep Euler as a
  fallback. Consider a sibling `FlowMapScheduler` sharing `φ`/`add_noise` so the noising
  path stays identical and only the target/step differ.
- The head may need a **second time input** (`t` *and* target level `s`, or step-size `d`).
  The DiT time path is per-stream (`time_embedding`/`time_projection`); adding a second
  conditioning is a per-expert change (video/pointmap share `Head`; action + DINO are
  Linear).

### 3b. The precedent to copy — DINO x0-prediction
DINO already has a head predicting a **non-velocity** quantity (`x0`) that is converted to
the scheduler's velocity contract via `_dino_x0_to_velocity` (`helpers/dino.py:154-178`,
σ-clamp 0.05) and a **zero-initialized** final layer (`helpers/dino.py:181-189`). A
flow-map head can reuse exactly this adapter pattern: predict the map, adapt to whatever
the shared step loop expects. Study this path first.

### 3c. Training target construction — `flexpi.py:_base_training_loss`
- Per-stream noising + target: `flexpi.py:1120-1157` (video/action/dino/pointmap) — where
  a flow-map would draw a **pair** of levels (`s`, `t`) instead of a single `t`, and build
  the flow-map target.
- Loss assembly: `flexpi.py:1269-1382` — the four per-stream losses + λ-combine. A
  flow-map/consistency loss slots in per stream here, flex-reduced the same way
  (`_flex_reduce_per_sample_loss`). **Keep the flex `m_in`/`m_out` semantics** — the loss is
  still summed over present-or-cross-modal streams.
- If distilling: a frozen teacher (the released flow-matching ckpt) is another module to
  hold; its short-step rollout is the target. Consider a `no_grad` teacher forward mirroring
  `_predict_joint_noise_unified_impl`.

### 3d. Inference step loops — the payoff
- **Action fast path** `_base_infer_action` (`flexpi.py:1610-1872`): the Euler action loop
  at `:1856-1868` (`_predict_action_noise_with_cache` → `infer_action_scheduler.step`).
  One/two-step flow map here is the highest-value latency win; note the CUDA-graph /
  loop-compile variants (`_run_action_prefill_denoise_loop`, `:1922`) and that this path is
  kept **pure Euler / graph-capturable** — a flow-map step must stay capture-friendly.
- **Joint path** `_infer_action_joint` (`flexpi.py:3266-3695`): per-stream `scheduler.step`
  at `:3641-3664`, prediction via `_predict_joint_noise_unified_impl` (`:1468-1603`). Fewer,
  larger jumps here directly shrink the 193 ms number; interacts with `StepSkipController`,
  the CUDA-graph loop, FlexAttention BlockMask, and the TensorRT engines
  (`models/inference_opt/`) — those assume a fixed per-step MoT forward, so re-check them.

### 3e. Heads that change (from the `01` matrix)
- **Video + pointmap** share the Wan `Head` shape (pointmap is a deep copy) in VAE latent
  space → build one flow-map head, instantiate twice; keep the deep-copy relationship
  (`flexpi.py:274-276`).
- **Action** — Linear head, own fast KV-cache loop → the prime one/few-step target.
- **DINO** — Linear head, already x0 + adapter → likely the easiest to port; may not even
  need a flow map (it's conditioning-oriented).

## 4. Invariants to preserve
- **Two-mask compute-flex** (`m_in`/`m_out`) and **cross-modality forcing** — the whole
  value proposition. A flow-map objective must still be summed over present/cross-modal
  streams and leave `m_out` as a visibility mask, not a loss mask.
- **One-way attention** (visual→action) — keeps the action-only KV-cache path exact.
- **Frozen encoders** (VAE, DINOv3, umT5) and the first-frame-clean (t=0) anchor convention
  in every stream.
- **Checkpoint compatibility knobs** — new head params need entries in the ckpt key sets
  (`_POINTMAP_CKPT_KEYS`, `_DINO_CKPT_KEYS`, and the mode/head save in
  `save_checkpoint`/`load_checkpoint`, `flexpi.py:2477-2537`) and sensible strict-shape
  behavior for warm starts.

## 5. Open questions for the user
1. Which flow-map family (consistency / shortcut / flow-map-matching) and **distill vs
   from-scratch**?
2. All four streams, or start with **action-only** (biggest latency win, smallest blast
   radius) and expand?
3. Target step count(s) at inference — one-step, or a small K with a step-size knob?
4. Do we keep the released flow-matching checkpoints as teachers (needs them downloaded)?
5. Benchmark to move first — RoboTwin (sim, fast iteration) or a real-YAM latency target?

## 6. First moves once the method is chosen
- Reproduce a baseline flow-matching run on the smallest benchmark (RoboTwin subset) to fix
  a reference number and confirm the harness end-to-end (weights, text cache, intrinsics).
- Prototype the scheduler-level target/step + one head (action) behind a config flag, so
  flow-matching stays the default and the two are A/B-comparable — mirror how
  `dino_pred_x0` and the DPM++ solver were added as opt-in switches.

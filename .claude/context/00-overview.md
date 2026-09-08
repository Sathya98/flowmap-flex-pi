# 00 — Architecture overview

The code-level companion to `docs/OVERVIEW.md`. Where that doc explains *why*, this one
pins *where* in the tree. Paths are repo-relative; line numbers are a snapshot — verify
before relying.

Paper: *Flex-π: A Multi-Stream World-Action Model with Compute Flexibility*
(arXiv 2608.10860).

## 1. The idea in one paragraph

A world-action model predicts the future to act better. Flex-π's bet: the frozen
Wan-2.2 VAE, trained only on RGB, encodes **3D pointmaps almost losslessly** — so
geometry and RGB share one latent space and one backbone can co-denoise both. Flex-π
therefore supervises **three visual futures** (RGB, pointmap, DINOv3 semantics) plus
**actions**, all as flow-matching token streams, and drops streams at random during
training so the model must synthesize the ones it never saw (**cross-modality forcing**).
Result: one checkpoint, 56 deployable input/output regimes, selected by a runtime flag.

## 2. The five streams

| Stream | Symbol | Encoder | Trainable? | Tokenizer file |
|---|---|---|---|---|
| RGB future | `z^o` | Wan-2.2 VAE | frozen | `models/wan_video_vae.py` |
| Pointmap future | `z^p` | **same** Wan-2.2 VAE (after depth→XYZ unproject) | frozen | `models/pointmap_encoder.py` |
| DINO semantics | `d` | DINOv3 ViT-B/16 | frozen | `models/dino_encoder.py` |
| Action chunk | `a` | linear projection | trainable | `models/action_dit.py` |
| Proprio `s` + language `l` | — | state Linear + umT5 (frozen) | Linear trainable | `backbone.py:86`, `backbone.py:597` |

Proprio and language are **global conditioning**, injected as shared cross-attention
context (one extra proprio token appended to the umT5 context), not as sequence tokens.
See `03-dataloaders.md` for how each stream's raw input is produced.

### Token budget (shipped 3-cam RoboTwin/YAM layout, `tshape_384x320`)
- 33 raw frames → subsample stride 4 → **9 RGB anchor frames** → VAE 4× temporal →
  **3 latent frames**.
- 384×320 composite → 24×20 latent → patchify `[1,2,2]` → 12×10 = **120 video
  tokens/latent-frame**.
- DINO: uniform 14×14 per cam, folded 2×2 (`dino_pixel_unshuffle: 2`) → 7×7 →
  **3×49 = 147 DINO tokens/frame** (feature dim 768→3072, no spatial detail lost).
- Action horizon **H = 32** (= `num_frames − 1`).

## 3. The backbone — Mixture-of-Transformers

`models/mot.py` (`MoT`), composed by `models/backbone.py` (`FlexPiBackbone`) and
subclassed by `models/flexpi.py` (`FlexPi`).

- **Visual trunk** `self.video_expert` — `WanVideoDiT`, 3072-d, 30 layers, ~5B params,
  initialized from Wan-2.2-5B. All visual streams (video+dino+pointmap) share it: DINO
  and pointmap tokens are **concatenated into the single "video" expert stream** by the
  caller before MoT is entered. MoT itself only ever sees two experts: `"video"` and
  `"action"` (`mot.py:58-59`).
- **Action expert** `self.action_expert` — `ActionDiT`, 1024-d, 30 layers, ~1B params,
  initialized from Wan-2.2 by *resampling* (donor ckpt
  `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`). Fully separate
  Q/K/V, FFN, norms, AdaLN, cross-attn — nothing weight-tied to the trunk.
- **Experts must match on depth (30) and attention head geometry** (`mot.py:81-99`); they
  differ only in `hidden_dim`/FFN width. That shared 30-layer / head-dim axis is what
  lets MoT concatenate their Q/K/V for one joint self-attention per layer.
- **Per layer**: MoT builds per-expert Q/K/V (`_build_expert_attention_io`), concatenates
  along sequence, runs one joint `_mixed_attention`, splits back, then each expert runs
  its own FFN/cross-attn/AdaLN post-block. The *only* cross-expert coupling is the shared
  self-attention.

### HBridge band (`hbridge.bottom_ratio=0.25`, `top_ratio=0.25`)
`_is_outer_layer` (`mot.py:117-127`): with 30 layers, `n_bottom=7`, `n_top=7`,
`n_middle=16`.
- **Outer layers 0–6 and 23–29 (stream-specific)**: the concatenated Q/K/V is sliced into
  per-sub-stream blocks; each block self-attends under its own mask — **no cross-stream
  attention**. Early encoding and late decoding stay stream-local.
- **Inner layers 7–22 (joint)**: one self-attention over the full concatenation under the
  joint mask — this is where visual↔action and video↔dino↔pointmap mixing happens.
- HBridge off (or `sub_stream_lens=None`) → joint attention at every layer.

## 4. The two masks (`m_in` / `m_out`) — the compute-flex mechanism

Two independent per-sample binary masks over the three visual streams, sampled every step
in `models/helpers/flex_joint.py` (`sample_flex_batch_flags`):

- **`m_in` — presence** (`present_v/d/p`): which streams are given as *input* at time *t*.
  Rejection-sampled so at least one visual anchor always remains (closes the train/deploy
  gap where all-absent would raise).
- **`m_out` — joint generation** (`j_v/j_d/j_p`): which future streams the action tokens
  read, and how future streams attend one another. Coupled `j_X &= present_X` unless
  `cross_modal_predict_X`.

**Critical: `m_out` is not a loss mask.** Every future stream is denoised and incurs its
flow-matching loss on every sample regardless of `m_out`; `m_out` only selects what is
*mutually visible*. Because the two masks are drawn independently, a stream dropped from
input is still denoised at output — the model synthesizes that modality's future from the
streams that remain. That is **cross-modality forcing** (`cross_modal_predict_*: true` by
default). Removing it costs ~21% success on RoboTwin.

### How the masks become attention
- Base static `[S,S]` mask: `flexpi.py:_base_build_mot_attention_mask_unified`
  (`flexpi.py:630-709`). Sequence order:
  `[ff_v | rem_v | ff_d | rem_d | ff_p | rem_p | action]` (`ff_*` = first-frame anchor at
  t=0, `rem_*` = noised future).
- Per-sample `[B,S,S]` flex mask: `_build_mot_attention_mask_unified` (`flexpi.py:2737`)
  clones the base per sample and applies `_apply_joint_flag_deltas` (action-row widening +
  rem↔rem XOR drops when two streams disagree on their joint flag, `flexpi.py:2632`) and
  `_apply_presence_absent_edits` (kill rows/cols of absent streams, `flexpi.py:2681`).
- **One-way rule**: action *rows* get True into visual columns; **no visual row ever sets
  True in the action columns**. Info flows visual→action only.
- Token zeroing for absent streams: `_flex_zero_absent_{video,dino,pointmap}_tokens`
  (`flexpi.py:1013-1079`) — whole stream zeroed if `cm=False`, only ff_X zeroed if
  `cm=True` (so rem_X stays a denoise target).

Flex probabilities: `configs/model/flexpi.yaml` `flex_joint`. RoboTwin/YAM ship the 0.5
defaults; LIBERO ships all six at 1.0.

## 5. The objective

Flow matching on the linear (rectified-flow) path, summed over all four streams
(`flexpi.py:_base_training_loss`, `1110-1383`):

```
L = λ_a·L_FM(a_t) + Σ_{i∈{o,d,p}} λ_i·L_FM(i_{t+1})     (all λ = 1)
```

Shared scheduler `WanContinuousFlowMatchScheduler` (`models/schedulers/scheduler_continuous.py`):
target velocity `v = noise − sample` (rectified flow), per-stream shift (video/dino/pointmap
= 6.0, action = 1.0). Each stream draws its **own** timestep independently. The DINO head is
the exception — it predicts **x0** (`dino_pred_x0: true`) and converts analytically back to v
(`helpers/dino.py:_dino_x0_to_velocity`, σ clamp 0.05) so its loss stays v-comparable. Full
head detail: `01-flowmatching-heads.md`.

Frozen throughout: Wan-2.2 VAE, DINOv3, umT5. Trainable: visual trunk, action expert, the
per-stream projectors/heads, and the proprio encoder.

## 6. A training step (the 7 stages)

Per batch (trainer drives only the optimizer; the work is on the model):
1. Decode RGB + depth from the loader; unproject depth → pointmap with per-dataset
   intrinsics (`PointmapEncoder.encode_composite`).
2. VAE-encode the RGB composite and the pointmap composite (frozen, shared weights).
3. DINOv3-encode the RGB composite; fold 2×2.
4. Look up the umT5 text embedding from the precomputed cache (no online text encoder).
5. Sample `m_in`/`m_out` and per-stream flow-matching noise + timesteps.
6. Forward the MoT under the resulting attention mask.
7. Sum the four flow-matching losses; backward.

AdamW, lr 1e-4, wd 1e-2 (task configs), cosine + 5% warmup, bf16, DeepSpeed ZeRO-1 via
`accelerate`. Full detail: `02-training-pipeline.md`.

## 7. Inference regimes

`infer_joint_*` picks what gets **generated**; `infer_present_*` picks what gets
**encoded as input** — 56 combinations from one checkpoint. The model runs *K* Euler steps
of the flow-matching ODE over the **active output streams only** and emits an H=32 action
chunk, of which `EVALUATION.replan_steps` are executed before replanning (32 everywhere
except LIBERO = 10). Eval pins **K=4**.

- **Action-only** (all `infer_joint_*=false`): fast KV-cache path
  `FlexPi._base_infer_action` — prefill first-frame anchors, run the Euler action ODE.
  ~60 ms on an RTX 5090.
- **Full joint** (all true): `FlexPi._infer_action_joint` — one masked MoT forward per
  step producing all active streams' predictions, each stepped by its own scheduler.
  ~193 ms with the TensorRT KV-split stack.

Regime dispatch: `infer_action` → `_infer_action_dispatch` (`flexpi.py:2980`, `3108`).
Latency stacks + TensorRT: `docs/INFERENCE_OPTIMIZATION.md`,
`src/flexpi/models/inference_opt/`.

## 8. Where things are (quick index)

| | |
|---|---|
| Top-level model | `models/flexpi.py` (`FlexPi`) |
| Shared base / composition | `models/backbone.py` (`FlexPiBackbone`) |
| MoT core / HBridge / masks | `models/mot.py` (`MoT`) |
| Visual trunk (+ video head) | `models/wan_video_dit.py` (`WanVideoDiT`, `Head`) |
| Action expert (+ action head) | `models/action_dit.py` (`ActionDiT`, `ActionHead`) |
| DINO / pointmap tokenizers | `models/dino_encoder.py`, `models/pointmap_encoder.py` |
| Flow-matching scheduler | `models/schedulers/scheduler_continuous.py` |
| Flex sampling | `models/helpers/flex_joint.py` |
| Composite layouts / slots | `composite_layouts.py`, `per_cam_compose.py` |
| Model factory | `runtime.py` (`create_flexpi`) |
| Trainer | `trainer.py` (`Wan22Trainer`) |
| Dataloaders | `datasets/lerobot/` |
| Architecture config | `configs/model/flexpi.yaml` |
| Training entry | `scripts/train.py` + `scripts/train_flexpi_*.sh` |
| Eval | `experiments/{robotwin,libero,yam}/` |
| Deploy | `scripts/serve_yam_flexpi.py` |

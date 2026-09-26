# Flowmap-Flex-π — working repository

This repo is a fork of **Flex-π** (`Sathya98/flowmap-flex-pi`), a 6B-parameter
multi-stream **world-action model** for robot manipulation. It jointly denoises four
streams — future **RGB**, **3D pointmaps**, **DINOv3 semantics**, and **actions** — with
**flow matching** inside a **Mixture-of-Transformers**, and deploys as anything from an
action-only VLA (~60 ms) to full joint generation (~193 ms) from **one checkpoint**.

## 🎯 Project goal (why this fork exists)

We are **converting the flow-matching WAM architecture to a flow-map based one** — i.e.
replacing/augmenting the per-modality **flow-matching heads** (which learn an ODE
*velocity field* solved with several Euler steps at inference) with **flow-map heads**
(few-step / consistency-style maps that jump between noise levels directly). The stream
plumbing, MoT backbone, dataloaders, and training harness stay; the *heads and the
denoise/step loop* are the surface we change. See
[docs/FLOW_MAPS.md](docs/FLOW_MAPS.md) for the implementation, objectives,
experiment matrix, and validation limits. Earlier context documents are design history.

> Current status: opt-in multi-stream flow-map training and inference on `dev-pedro`,
> with distillation/self-distillation and full/LoRA/adapter/head tuning. CPU regression
> coverage uses tiny real models; full-scale GPU and robotics validation is pending.

The main study is now **full joint + full parameter tuning**: released FM baseline,
task-checkpoint distillation, and AGIBOT-initialized self-distillation across objectives
and NFEs. Train on LIBERO and RoboTwin; evaluate LIBERO models unchanged on LIBERO-Plus.
See [FULL_JOINT_STUDY.md](docs/FULL_JOINT_STUDY.md). Stream subsets and LoRA/adapters/heads
are future ablations. The general AGIBOT checkpoint must be acquired separately.

Porting the flow-map stack (LSD/LMD, latent cache, batching, EMA, kernels) to the
Wan2.2-TI2V-5B **world model** in `../exmachina/diffsynth-studio`: see
[docs/wm_wan_extension.md](docs/wm_wan_extension.md) (plan; nothing ported yet).

## 📚 Context docs (read these on demand)

Detailed, line-referenced notes live under `.claude/context/`. They are the distilled
result of a full read of the codebase — trust them, but verify a file/line still exists
before relying on it (the tree moves).

| Doc | What it covers |
|---|---|
| [`00-overview.md`](.claude/context/00-overview.md) | Architecture at a glance: the four streams, the MoT backbone, the two masks (`m_in`/`m_out`), cross-modality forcing, HBridge, the objective, inference regimes. |
| [`01-flowmatching-heads.md`](.claude/context/01-flowmatching-heads.md) | **The centerpiece.** Every per-modality flow-matching head (video, pointmap, DINO, action): input projectors, output heads, prediction target (v vs x0), the shared scheduler, per-stream noising/timesteps, and where each lives. Written for the flow-map conversion. |
| [`02-training-pipeline.md`](.claude/context/02-training-pipeline.md) | The training/finetuning pipeline: `create_flexpi` factory, `Wan22Trainer` loop, the 7-step per-batch flow, optimizer/schedule, checkpointing, resume-vs-finetune, launchers, Hydra config graph, run naming. |
| [`03-dataloaders.md`](.claude/context/03-dataloaders.md) | The LeRobot data pipeline: `RobotVideoDataset` → `FlexPiProcessor`, the sample dict, the 33→9→3 temporal cascade, depth codecs, the composite-layout/slot system, action/rotation transforms, camera intrinsics, the T5 text cache. |
| [`04-repo-map.md`](.claude/context/04-repo-map.md) | File-by-file map, eval/deploy entry points, dependencies, and the external services this involves (HuggingFace downloads, SLURM, Wan2.2/DINOv3/DA3 weights). |
| [`05-flowmap-conversion.md`](.claude/context/05-flowmap-conversion.md) | The conversion working notes: what "flow map" means here, the surgical insertion points across the heads and the step loop, open questions. |
| [`07-efficiency-notes.md`](.claude/context/07-efficiency-notes.md) | Training cost of the flow-map objectives: measured s/update and memory (PFMM, LMD), why LMD's forward AD costs memory, LSD/ESD vs LMD JVP/gradient structure, estimated timings, the self-distillation mask imbalance, the forward-AD attention path, the 1-GPU profiling benchmark (`scripts/profile_flowmap_step.py`) and its results, the trainer-timing harness (`FLEXPI_STEP_TIMING`) and the DeepSpeed ZeRO-2 hook bug it found (patched in `utils/deepspeed_compat.py`), the TVM fused attention-JVP validation and integration (`jvp_kernel_analysis/`), the microstep profile and CUDA-graph/compile plan and the background EMA fix (§11), the per-update timeline from baseline to now (§12), and the TODO list. |

## 🗺️ Architecture at a glance

```
per-cam RGB ─VAE(frozen)─► z^o ─┐
per-cam depth ─unproject─VAE──► z^p ─┤ concat → "video" MoT stream ─┐
per-cam RGB ─DINOv3(frozen)──► d  ─┘   (order: ff/rem of v,d,p)     │  Mixture-of-
                                                                    ├─ Transformers ─► per-stream heads → flow-matching loss
action chunk ─Linear──────────► a  ────── separate "action" stream ─┘  (HBridge: 0–6,23–29 stream-specific; 7–22 joint)
proprio(state)+language(umT5, frozen) → shared cross-attn CONTEXT (both experts)
```

- **Visual trunk** = Wan2.2-TI2V-5B DiT, 3072-d, 30 layers (~5B). **Action expert** =
  ActionDiT, 1024-d, 30 layers (~1B). Only MoT couples them, via shared self-attention
  over concatenated Q/K/V at the joint (inner) layers.
- **Attention is one-way**: action attends the visual streams; no visual token ever
  attends action. This is what makes the action-only KV-cache path exact.
- **Two per-sample masks** give one checkpoint every regime: `m_in` (presence — which
  streams are given as input) and `m_out` (joint — which futures are mutually visible).
  `m_out` is **not a loss mask**: every present-or-cross-modal stream is denoised and
  incurs its loss regardless. Independent draws + **cross-modality forcing** force each
  modality to be predictable from the others.
- **Objective**: `L = λ_a·L_FM(a) + Σ_{i∈{o,d,p}} λ_i·L_FM(i)`, all λ = 1. Rectified-flow
  target `v = noise − sample`; DINO head predicts **x0** and converts analytically to v.

## ⚙️ Key commands

```bash
# Train (edit the config block at the top of the launcher first)
bash scripts/train_flexpi_robotwin.sh            # RoboTwin 2.0
bash scripts/train_flexpi_libero.sh              # LIBERO 4-suite
DATASET_DIRS="[./data/<yam_set>]" bash scripts/train_flexpi_yam.sh   # real YAM

# Required before first train: cache umT5 text embeddings (train dies without it)
python scripts/precompute_text_embeds.py task=<TASK_CONFIG>

# Any knob is a Hydra override appended after the launcher (wins over the script):
bash scripts/train_flexpi_robotwin.sh batch_size=4 model.flex_joint.p_jv=1.0

# Evaluate
bash scripts/eval_flexpi_robotwin.sh             # set CKPT + DATASET_STATS at top
CKPT=... DATASET_STATS=... GPUS=0,1,2,3,4,5,6,7 bash scripts/eval_flexpi_libero_4suite.sh

# Serve (real robot, WebSocket policy server)
python scripts/serve_yam_flexpi.py --ckpt-path <run>/checkpoints/weights/step_NNNNNN.pt
```

- One `task=` picks **everything** (data + model + hyperparams) via Hydra `defaults`.
  `task=` is the shared key across train / precompute / eval.
- Env vars only reach the launcher's config block if written `VAR="${VAR:-...}"`; the
  `FLEX_P_*` knobs are in-script assignments (edit the file). CLI Hydra overrides always
  win — every launcher appends `"$@"` last.

## ⚠️ Gotchas worth remembering

- **Text-embed cache is mandatory** and keyed by prompt-hash + `context_len`; it must
  cover *every* prompt in *every* dataset dir you train on (narrowing `TASK_NAMES` does
  not narrow the cache). Missing → training dies on batch 1.
- **The pointmap/3D stream needs `meta/camera_intrinsics.json`** per dataset dir. To run
  without depth: `model.enable_pointmap=false data=<benchmark>_nodepth`.
- **LeRobot v2.1 datasets, canonical cam keys**, depth as `observation.depth_{codec}.*`.
  All combined datasets must share fps / camera layout / action dim (enforced; fps
  mismatch raises).
- **`weight_decay` is `1e-2`** in the task configs (not the `0.0` in `train.yaml`).
- **Gradient checkpointing is OFF in all three shipped task configs**
  (`mot_checkpoint_mixed_attn: false`); the base `model/flexpi.yaml` default is `true`.
- **`num_inference_steps` (K)**: eval configs pin **K=4**; the code's fast-path default is
  20. Action-only success peaks at K=4.
- Full troubleshooting table: `docs/TRAINING.md §5`.

## 📎 Attribution

Branch off `main` into `dev` for conversion work. Commits are co-authored per the
session's attribution footer. Upstream docs (`docs/OVERVIEW.md`, `TRAINING.md`,
`ROBOTWIN.md`, `LIBERO.md`, `YAM.md`, `INSTALL.md`, `INFERENCE_OPTIMIZATION.md`) remain
the source of truth for install/run recipes; the `.claude/context/` docs are the
code-level map.

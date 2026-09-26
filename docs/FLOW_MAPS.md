# Flow-map experiments

This document supersedes the earlier `.claude/context/05-*` and `06-*` design
notes. Flow-map code is opt-in; `model.flow_map.enabled=false` retains the FM
baseline. No training jobs are launched by the matrix generator.

## Main study: full joint, full tuning

The main experiment is now fixed: generate action + RGB + DINO + pointmap in
all runs, with full parameter fine-tuning of the WAM (encoders remain frozen).
Stream subsets, LoRA, adapters, and head tuning are ablations or future work.
The earlier action-first / RGB-first research matrix is superseded.

| Family | Starting weights | Teacher | Objectives |
| --- | --- | --- | --- |
| FlexPi FM baseline | Released benchmark-finetuned FlexPi | none | ordinary FM, evaluation only |
| Distilled Flow Maps | Same benchmark-finetuned FlexPi | frozen copy of that task model | LMD, EMD, PFMM |
| Self-distilled Flow Maps | **AGIBOT-pretrained FlexPi, before benchmark fine-tuning** | current model | LSD, ESD, PSD-M, PSD-U |

There are two training settings: LIBERO and RoboTwin 2.0. Evaluate each LIBERO
checkpoint unchanged on LIBERO-Plus; never fine-tune on Plus or select checkpoints
using its test scores. Evaluate every method at NFE **1, 2, 4, 8, 16**, with the
same full-joint inputs, action horizon, replanning, episodes and evaluation seeds.
One joint network evaluation counts as one NFE, not four because there are four
streams. Use scale-1 guidance and disable step skipping for these comparisons.

“From scratch” here means starting benchmark-specific flow-map training from
AGIBOT weights, with no benchmark-finetuned FM teacher. It does not mean random
weights or Wan-only initialization. Diagonal FM supervision is part of direct
self-distillation, not a separately trained teacher or a mixed-stream model.

See [FULL_JOINT_STUDY.md](FULL_JOINT_STUDY.md) for checkpoint provenance, table
layouts, protocol controls, release gaps and commands. Training-cost accounting
must distinguish inherited teacher training from additional distillation cost.

## Objectives

All times are sigma noise levels (1=noise, 0=clean). The map is
`X(s,t,x) = x + (t-s) v(x,s,t)`. The identity boundary is exact. Every generated
stream moves along a **shared sigma schedule**, including teacher queries and
Eulerian spatial JVPs. This deliberately does not reuse the legacy independent
visual/action shifts; `schedule_shift` controls the new common schedule.
Training defaults to `time_sampling=uniform_triangle`: uniform area over
`0 <= t <= s <= 1` (or its `strip_width` restriction). At full width this
matches Boffi's sorted-uniform proposal after reversing time direction.
`time_sampling=conditional` retains the previous uniform-source proposal for
explicit ablations. This setting is recorded in configs and checkpoints.

| objective | supervision | off-diagonal condition |
| --- | --- | --- |
| `lmd` | frozen FM teacher | Lagrangian target-time derivative |
| `emd` | frozen FM teacher | Eulerian source-time plus joint spatial derivative |
| `pfmm` | frozen FM teacher | regress its `teacher_steps` Euler composition into one map |
| `lsd` | data + current diagonal velocity | Lagrangian self-distillation |
| `esd` | data + current diagonal velocity | Eulerian self-distillation |
| `psd_m` | data + current map | two-jump composition with midpoint split |
| `psd_u` | data + current map | two-jump composition with uniform split |

Self-distillation uses 75% diagonal FM examples and 25% off-diagonal examples.
At effective batch 192, the trainer assigns exactly 144 diagonal and 48
self-distillation examples across ranks and accumulation. Diagonal times are
independent uniforms; off-diagonal times follow the uniform triangle. Each
example contributes one of the two terms. Self-distillation includes supervision
against noise-data;
`diagonal_weight` must be positive. Pure teacher distillation defaults to no
extra data loss (`distill_diagonal_weight=0`). Losses are reduced per stream and
sample, with padding/presence/cross-modal masks, then weighted with the existing
`loss.lambda_*` settings. `map_weight` controls off-diagonal supervision.
For self-distillation, a learned two-time weight then applies
`exp(-logvar)*raw_joint_loss + logvar` before batch averaging. Log raw residuals
as well as the weighted objective: the latter can be negative and is not by
itself evidence of improving predictions. Disable only as an explicit ablation
with `learned_time_weighting=false`.

Self-distillation, and external distillation with `distill_ema=true`, maintain
CPU FP32 evaluation EMAs at 0.999 and 0.9999. Full
training state includes `flowmap_ema.pt`; inference files are under
`checkpoints/weights/ema_0.999{,9}/step_*.pt`. The study's preselected default is
0.9999. EMA never supplies training targets. Inline training validation still
uses current weights; use the exported EMA checkpoint for benchmark evaluation.
EMA adds two FP32 trainable-parameter copies on rank zero and two exported
weight files per save, with the same retention limits as raw checkpoints.
DDP and ZeRO-2 are supported; ZeRO-3 is rejected rather than saving partial EMA.
Old full-state self-distillation runs without EMA cannot resume this recipe;
weight-only initialization is a new run, not exact continuation.

For the existing pilots and self-distillation, gradient placement follows Boffi
2025 Eq. 94: teacher predictions
and Eulerian spatial JVPs are detached, while temporal derivatives remain
fully differentiable. Dual-number attention has two backends
(`flow_map.jvp_attention`):
- `explicit` (default) uses out-of-place FP32 math, including softmax, to support
  backward through the temporal JVP on PyTorch 2.7. Its attention memory is quadratic.
- `tvm` uses fused Triton kernels from Terminal Velocity Matching
  (`src/flexpi/models/helpers/jvp_attention`, CC BY-NC-SA 4.0) that compute
  attention, its JVP, and the backward through both without materialising L×L.
  Masks are split exactly into mask-free row groups (3 in full joint).
  Measured per call it is 3.8× faster and uses ~10× less memory; per LMD/LSD step it
  saves 8–10 GiB, which lets LMD run at microbatch 2 on one GPU. Grads agree with
  `explicit` at cosine 0.99997.

Ordinary forwards retain fused SDPA. Attention inside JVP forwards is checkpointed
with the dual-aware `helpers/checkpoint.py`.
Profile real hardware before selecting a full-scale batch size.

External LMD additionally supports `lmd_teacher_gradient=full`, retaining the
teacher-input derivative while freezing teacher parameters. Future study commands
select this original LMD gradient path; `detached` labels the earlier pilot.
External time weighting is opt-in with `distill_learned_time_weighting=true` and
uses the same per-example learned head as self-distillation. See
[FLOW_MAP_CONTROLS.md](FLOW_MAP_CONTROLS.md) for matched configs and commands.

`detach_derivatives=true` selects the optional expanded semigradient variant
from Appendix F.4. This saves derivative activation memory but changes the
optimization dynamics: our small fixed-batch experiments showed rising residuals
for LSD/ESD with this variant, so it is an explicit ablation, not the default.
PSD and PFMM require no JVP and are useful computational comparisons. PFMM uses
fixed multi-Euler-step teacher composition and a velocity-normalized residual;
it does not implement staged teacher replacement/interval widening. Those are
optional strategies: Section 3.12 of the 2024 paper explicitly allows a fixed
teacher. Our adaptation uses a numerical FM-velocity integrator as the teacher
map, rather than a separately trained flow-map network. Its normalized residual
also differs from unweighted endpoint MSE by the squared interval factor.
`pfmm_loss_space=velocity` preserves that pilot recipe; `endpoint` multiplies
residuals by the interval before squaring. The same endpoint error on a jump of
0.1 receives 100 times its endpoint-loss weight in velocity space, so this
choice changes optimization and must be recorded as a separate experiment.

`central_fd` and `forward_fd` are diagnostic Lagrangian alternatives requiring
float32 model computation with autocast disabled. BF16 finite differences are
rejected, rather than relying on upcasting already-rounded outputs. The generic
FD helper uses one-sided stencils at boundaries; objective sampling keeps the
FD stencil within the valid denoising interval. Calibrate epsilon empirically.

Sources: [Flow Map Matching](https://arxiv.org/abs/2406.07507) and
[Learning flow maps via self-distillation](https://arxiv.org/abs/2505.18825).
The earlier inverse-consistency direct FMM objective and higher-order psi
parameterization are not implemented; they are not aliases for LSD or adapters.

## Adaptation and initialization

- `mode=full`: train the denoisers and conditioning layers; encoders stay frozen.
- `mode=lora`: low-rank updates to transformer-block linear projections and
  time-difference embeddings. `rank` and `lora_alpha` control capacity/scaling.
- `mode=adapter`: parallel nonlinear bottleneck adapters on the same projections,
  plus time-difference embeddings. Uses the same affine map as LoRA.
- `mode=heads`: time-difference embeddings and active stream output heads.

Base projection weight/bias names remain compatible with released checkpoints.
Adaptation configuration is saved in checkpoints and checked on load. Adapter
weights are included in normal MoT checkpoints and full training-state resumes.

A zero-initialized delta embedding reproduces a teacher **Euler step at the
chosen interval**, not the teacher's multi-step accuracy. Centering delta time
conditioning preserves the current network's diagonal; it does not freeze the
teacher's velocity when other weights are updated.

The main self-distillation initialization is AGIBOT-pretrained FlexPi, loaded
through `pretrained_ckpt`, with `initialization=pretrained` and no external teacher.

Other initialization comparisons, outside the main study:

- Released FlexPi + external teacher: distilled WAM fine-tuning.
- Released FlexPi + LSD/ESD/PSD: self-distillation fine-tuning.
- No `pretrained_ckpt`, standard Wan/ActionDiT initialization + self-distillation:
  direct WAM training from foundation-model weights, without a WAM teacher.
- `initialization=random`, `mode=full`, no `pretrained_ckpt`: randomly initialized
  denoisers/conditioning layers with pretrained frozen encoders. Report that
  encoder pretraining explicitly; this is not a wholly pretrained-free system.

Distillation requires `teacher_checkpoint`, falling back to `pretrained_ckpt`.
The teacher is independently loaded, frozen, and held outside the student's
module tree/optimizer/ZeRO/checkpoint. Only frozen encoders and immutable layout metadata are shared. Incomplete teacher checkpoints are rejected. It uses
its own proprioception projection. Resuming requires the original teacher path.

## Commands

```bash
# Print the full-joint plan; unresolved AGIBOT rows are explicitly marked pending.
python scripts/flowmap_matrix.py --format json

# Once the general pretrained checkpoint is available, generate all 14 train commands.
python scripts/flowmap_matrix.py --agibot-checkpoint /path/to/agibot/weights.pt

# Three training seeds, with blank result tables; this does not launch jobs.
python scripts/flowmap_matrix.py --agibot-checkpoint /path/to/agibot/weights.pt \
  --seeds 42 43 44 --tables-dir runs/flowmap_fulljoint/tables --format json
```

`--benchmark libero|robotwin` limits the training setting. `--family distillation`
needs only released task weights. `--include-fm-control` adds an FM fine-tuning
control from the same AGIBOT initialization under the same fixed full-joint recipe.
Use `--libero-teacher`, `--robotwin-teacher` and the corresponding `--*-stats`
arguments to override the paired released files. The separate
`scripts/flowmap_ablations.py` retains the earlier subset/adaptation sweep for
future work; it is not the main experiment matrix.

For evaluation, load the run's saved model configuration. Set
`EVALUATION.num_inference_steps=1|2|4|8|16` explicitly; the existing evaluation preset
pins K=4 even though direct `infer_action` defaults to the flow-map K setting.
Do not request joint generation of an untrained stream. Legacy FM TensorRT,
compiled-loop, quantization, step-skip and DPM++ paths are rejected for map
experiments; they do not carry the second-time contract. Eager action inference
retains KV-cache prefill, and joint inference uses the full two-time predictor.
CFG other than scale 1 is rejected because guided maps require their own training.

`strip_width=1` covers all jump lengths. Narrow strips must cover every inference
jump: the sampler now covers all start levels, and inference checks the actual
shifted interval widths. With shift 6, K=2 has a jump of 6/7, not 1/2.

## Validation and reporting

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

Tests use tiny real ActionDiT/WanVideoDiT/MoT modules and synthetic encoded inputs;
no pretrained weights or dataset downloads. Full GPU training, distributed
execution, real checkpoint distillation, and LIBERO/RoboTwin success require
separate smoke/benchmark runs. CPU tests cannot establish those outcomes.

Report task success with uncertainty, RGB/geometry/semantic future quality,
multiple-noise-seed diversity, K, end-to-end and denoising-only latency, peak
memory, trainable parameters, and training GPU-hours. Use identical datasets,
seeds, hardware, and observation modalities. Hold training compute fixed when
comparing objectives whose forward/JVP costs differ.

Keep previous-chunk bridges separate from these noise-to-data experiments.
Changing the base distribution changes the teacher ODE; interior bridge noise
that vanishes at endpoints does not by itself supply conditional inference
randomness. FlexPi's visual tokens also do not attend action tokens: this work
accelerates its existing WAM, not arbitrary action-conditioned simulation.

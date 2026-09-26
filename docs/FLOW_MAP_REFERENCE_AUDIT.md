# Flow Map reference-code audit — 2026-09-17

This audit directly reads the reference source and resolved local Hydra configs.
It does not claim to reproduce the image experiments or establish convergence.

Update: [FLOW_MAP_CONTROLS.md](FLOW_MAP_CONTROLS.md) specifies the new controlled
comparisons. External distillation now supports EMA and optional learned time
weighting. Full teacher-input gradients are available for LMD. Future study
commands explicitly select full-gradient LMD and endpoint-loss PFMM; existing
pilots retain their original variants. The queued-pilot table below describes
those original runs, not the newly prepared configs.

## Reference versions inspected

- Official 2025 self-distillation repository `nmboffi/flow-maps`, commit
  `2f115a07fa9073553193e4b265dfc303827af2b0`.
  Files: `py/common/{losses,loss_args,flow_map,state_utils,updates}.py`,
  `py/configs/cifar10.py`, and the EDM2 network implementation.
- Boffi's minimal 2024 implementation `nmboffi/flow_map_matching_public`, commit
  `d44007a921db19bea73af7d16ddbea1921149431`.
  Files: `py/common/losses.py`, `py/launchers/learn.py`.
  Its README explicitly calls it unofficial and says it does not reproduce
  the paper's exact experiments.

## Queued pilot settings

PFMM job 26861582 and LMD job 26862643 use:

| Setting | Value |
| --- | --- |
| Initialization and frozen teacher | released LIBERO step 010860 |
| Streams | action, RGB/video, DINO, pointmap; fixed full joint |
| Adaptation | full denoiser/conditioning parameters; frozen encoders |
| Microbatch / accumulation / GPUs | 1 / 48 / 4 = effective batch 192 |
| Updates / example presentations | 50 / 9600 |
| Optimizer | AdamW, betas (0.9, 0.95), default epsilon 1e-8 |
| Peak LR / weight decay | 1e-5 / 0.01 |
| Schedule | 2 warmup updates, then cosine toward 1e-7 at horizon 50 |
| Precision / seed | BF16 model computation; FP32 map-state/time arithmetic / 42 |
| Per-stream loss multipliers | all 1.0, after per-stream masked mean reductions |
| Map weight / extra diagonal data weight | 1.0 / 0.0 |
| Time proposal | uniform triangle (updated before either pilot started) |
| Strip / schedule shift | 1.0 / 1.0; no interval curriculum |
| EMA / learned loss weighting | absent / absent |
| Checkpoints | updates 10, 20, 30, 40, 50 |

PFMM regresses the average of 16 frozen-teacher Euler velocities. LMD uses
an exact temporal JVP and a detached frozen-teacher velocity evaluated at the
student's mapped state. LMD does not use 16 Euler steps in its training target,
even though the inherited config contains `teacher_steps=16`.
The configured inference default of 2 steps does not determine training NFE;
the study's evaluation sweep remains 1, 2, 4, 8, 16.

## Matches and differences

1. **LMD gradient variant.** Our derivative residual is the Lagrangian one.
   In the existing pilot, temporal derivatives remain differentiable; the entire
   teacher evaluation is detached. This follows the 2025 `convex` stop-gradient construction with
   an external frozen teacher. The older `lagrangian_distill` implementation
   does not detach its teacher input, so it propagates a spatial derivative
   through the frozen teacher. Frozen parameters and detached input gradients
   are distinct. The existing pilot must be labeled as the stop-gradient variant.
   `lmd_teacher_gradient=full` now retains the original teacher-input gradient.
2. **Time sampling.** The official 2025 code sorts two independent uniforms,
   producing a uniform triangle. After translating time direction to our
   noise-to-data sigma convention, this has source density 2s. At audit time
   our conditional proposal had uniform source density and joint density 1/s.
   The user then requested uniform-triangle sampling: this is now the default
   and is used by both pending pilots. The old proposal is an explicit ablation.
3. **Loss weighting.** Self-distillation now uses a learned two-time log variance:
   `exp(-w(s,t))*raw_joint_loss + w(s,t)`, applied per example before batch
   averaging. The head follows the reference deterministic positional embeddings,
   magnitude-preserving sum and normalized linear projection. Reference times
   are `1-sigma`. Raw joint and per-stream residual losses remain logged.
   FlexPi keeps masked per-stream means and stream multipliers; the image
   reference sums coordinates. Optimizer and architecture remain FlexPi's, not
   EDM2's projected-parameter training recipe. External-teacher pilots retain
   fixed weighting; `distill_learned_time_weighting=true` enables its ablation.
4. **Self-distillation mixture.** Now 75% diagonal data-FM examples and 25%
   off-diagonal self-distillation examples, rather than both losses per sample.
   The trainer shuffles an exact split across the accumulated distributed update
   (144/48 at batch 192), and carries accumulation across epoch boundaries.
   Diagonal sigma is independently uniform; off-diagonal pairs use the uniform
   triangle. Standalone calls without a trainer mask sample Bernoulli assignments
   (fixed-time diagnostic calls default to off-diagonal). This is not training
   without data supervision: noise-minus-data supplies the diagonal targets.
   Evaluation EMAs 0.999 and 0.9999 are maintained on rank zero in FP32 CPU
   memory, updated only on successful optimizer steps, and saved/restored with
   full training state. Separate inference checkpoints contain each EMA's weights.
   Self targets still use current student parameters. EMA supports DDP/ZeRO-2;
   ZeRO-3 is explicitly rejected. The study selects 0.9999 before evaluation;
   the trainer's inline raw-weight diagnostics remain raw-weight diagnostics.
5. **Progressive distillation.** The older reference composes two teacher maps,
   refreshes teacher parameters periodically, and expands intervals. Our PFMM
   is a fixed FM-teacher Euler-composition regression variant, not that schedule.
   **Clarification:** the 2024 paper Section 3.12 explicitly allows a fixed
   teacher and presents staged replacement as optional. Its absence is not
   itself a departure from PFMM. Our principal adaptations are an Euler FM
   integrator instead of a trained map teacher, and velocity-normalized MSE.
   `pfmm_loss_space=endpoint` now supplies the unnormalized endpoint alternative.
   The queued pilot remains `velocity`. With h=|t-s|, endpoint_loss=h²*velocity_loss;
   identical endpoint errors at h=.1 and h=1 get relative weights 100 and 1
   under velocity loss. This is an optimization difference, not a cosmetic unit
   change, and changing it requires a separate controlled run.
   Source: https://arxiv.org/html/2406.07507v1
6. **Optimizer and model recipe.** The 2025 CIFAR config uses RAdam at 1e-2,
   batch 512, 400,000 updates, inverse-square-root decay, FP32, EMA factors
   0.999 and 0.9999, and an EDM2 architecture with weight projection. These are
   image-model training settings, not justified drop-in settings for fine-tuning
   a pretrained 6B-parameter Flex-Pi transformer. Our optimizer/batch follow the
   local Flex-Pi family of presets; the lower LR and short schedule are pilot
   choices, not Boffi-validated hyperparameters for this model.

## DeepSpeed clipping wiring found during this audit

The trainer's `max_grad_norm=1.0` was not propagated into the JSON-backed
DeepSpeed configuration. Installed Accelerate's `clip_grad_norm_` only obtains
the engine's norm under DeepSpeed; it does not clip. A missing DeepSpeed
`gradient_clipping` key defaults to zero. Therefore the trainer's displayed
threshold did not establish that clipping was active. Inspection of job
26858864's saved optimizer state confirmed `clip_grad=0.0`, with the
`gradient_clipping` key absent from its saved DeepSpeed config.

The trainer now explicitly sets DeepSpeed's `gradient_clipping` to
`cfg.max_grad_norm` before `accelerator.prepare`. Constructor regression tests
exercise the installed Accelerate config processing and DeepSpeed's config
reader for clipping values 0, 0.25, and 1, plus the non-DeepSpeed path.
The constructor test passed all 12 initialization/backend subcases, and both
accumulation-metric regression tests passed after the change.

An attempted Slurm hold was denied by the scheduler; both jobs were still
pending at the check. They load source at execution time and will pick up this
fix. This changes their effective clipping from the omitted-key default to the
intended value 1.0. Submission manifests record this pre-start correction.

These pilots test the corrected implementation and selected recipe. They do
not justify describing any objective as intrinsically unreliable, nor launching
the complete benchmark matrix without a decision on the recipe differences.

## Checkpoint availability, rechecked 2026-09-17

The live upstream README explicitly says: "We plan to release large-scale
checkpoints pre-trained on YAM, AgiBot World, and DROID. Stay tuned!"
Source: https://github.com/geyan21/flex-pi#-model-download
The live HF API `https://huggingface.co/api/models?author=flex-pi&limit=100`
returns only `flexpi-libero`, `flexpi-libero-fulljoint-star`, and `flexpi-robotwin`.
The paper describes AGIBOT pretraining but provides no downloadable general
FlexPi checkpoint. The AGIBOT dataset and unrelated AGIBOT policies are not a
substitute. The requested pretrained-only study arm remains blocked on release
or author-provided weights; no task-finetuned checkpoint is relabeled as its base.

## Validation of the self-distillation alignment

CPU regression suite: 40 tests and 58 subcases pass. Coverage includes the
144/48 effective-batch split, independent diagonal sampling and skipping map
JVPs on diagonal examples, two-time weighting before batch averaging, all four
self-objective gradient contracts, interval-squared PFMM loss/gradient scaling,
EMA continuation/export/restore and successful-update-only EMA integration. Existing all-stream, padding, BF16, inference, initialization
and external-teacher diagnostics regressions also pass. The original tiny
optimization check now also checks unweighted loss, not only the learned-weight
objective. This does not establish convergence, real 6B GPU memory fit, or
closed-loop task success; those require the planned hardware pilots/evaluation.

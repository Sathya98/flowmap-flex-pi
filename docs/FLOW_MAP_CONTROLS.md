# Distillation controls and next runs

Prepared on dev-pedro, 2026-09-17. These configs do not submit jobs.

## Recommended primary choices

- LMD: `lmd_teacher_gradient=full`. Differentiate through the frozen teacher's
  input, as in the older LMD implementation. Teacher parameters stay frozen.
  The current detached-target implementation is a named ablation, and remains
  appropriate for the reference self-distillation gradient choices.
- PFMM: `pfmm_loss_space=endpoint`. This matches the older reference's loss
  space; our fixed FM teacher's Euler integration remains an adaptation of its
  learned-map composition target. Velocity loss is a named ablation.
- Save raw weights and evaluation EMAs at 0.999 and 0.9999 for BOTH training
  families. For external distillation this is `distill_ema=true`. The future
  study matrix preselects EMA 0.9999 for its main comparison and records raw/EMA
  alternatives. Raw weights are mandatory pilot diagnostics: after 50 updates,
  decay 0.9999 still assigns about 99.5% weight to initialization.
- Retain learned time weighting for the self-distillation reference recipe.
  External distillation retains fixed weighting in its primary recipe, with
  `distill_learned_time_weighting=true` available as a controlled ablation.
  Comparing these primary recipes does not isolate the effect of the objective
  alone. Compare external weighting on/off (or all objectives without learned
  weighting) before making that attribution. Weighted objective values are
  not directly comparable across recipes; log raw residuals and use fixed
  prediction/rollout evaluations.

The recommended LMD/PFMM choices follow the reference formulations rather than
an observed performance advantage. Their GPU fit and convergence are unverified.
Do not alter self-distillation's 75/25 mixture, independent diagonal times,
uniform triangle, or detached self-targets for these external-teacher controls.

## Existing jobs keep their recorded variants

The pending jobs observed during preparation are PFMM 26861582 and LMD 26862643,
with dependent diagnostics 26861592 and 26862671. Their inherited smoke config
now explicitly pins `pfmm_loss_space=velocity`, `lmd_teacher_gradient=detached`,
`distill_ema=false`, and `distill_learned_time_weighting=false`. These settings
preserve their previous behavior. No jobs were cancelled, replaced or submitted.
They remain useful raw-weight implementation checks, but are not an EMA-matched
comparison against the new recipe.

## Matched pairs, ready to submit

All four configs use the same LIBERO release, four joint streams, full tuning,
50 updates, seed 42, effective batch 192, LR 1e-5, and EMA/save settings. Each pair
changes one objective setting, plus its output name. The launcher includes CPU
preflight and post-run checks for update logs, raw checkpoints and EMA files.
Its 14-hour allocation is provisional; full LMD needs additional teacher-backward
memory and dual EMA/checkpoint overhead has not been profiled on the 6B model.

```bash
# LMD: same teacher target values, different student gradients.
sbatch scripts/slurm/flowmap_libero_control.sbatch flowmap_libero_lmd_fullgrad_pilot
sbatch scripts/slurm/flowmap_libero_control.sbatch flowmap_libero_lmd_detached_control

# PFMM: same 16-step Euler teacher, different interval weighting.
sbatch scripts/slurm/flowmap_libero_control.sbatch flowmap_libero_pfmm_endpoint_pilot
sbatch scripts/slurm/flowmap_libero_control.sbatch flowmap_libero_pfmm_velocity_control

# Optional weighting ablation against the full-gradient LMD config above.
sbatch scripts/slurm/flowmap_libero_control.sbatch flowmap_libero_lmd_fullgrad_pilot \
  model.flow_map.distill_learned_time_weighting=true
```

Run a 2-update smoke of full LMD first by appending `max_steps=2 save_every=0`;
this tests memory and execution only. Then run the matched pair(s) according to
available compute, inspect fixed-case raw residuals, predicted endpoints and
teacher integration convergence, and use task rollouts before deciding the
full training recipe. Fifty updates do not establish convergence. Longer pilots
need a schedule planned for their longer budget.

The new controls do not change checkpoint lineage: external distillation starts
from a task-finetuned model. The main pretrained-only self-distillation arm still
needs the unreleased AGIBOT initialization. Self-distillation from a task-finetuned
model is a separately labeled adaptation experiment, not a substitute for that arm.

## Validation

The full CPU suite passes 45 tests and 58 subcases. New checks cover analytical
full-versus-detached LMD gradients, full-joint teacher/student backward, learned
external weighting before reduction and checkpoint persistence, external EMA
initialization/ZeRO-3 rejection, and exact config-pair differences. Both primary
configs pass the local LIBERO checkpoint/architecture/data/text-cache preflight.
The Slurm launcher passes Bash syntax checks. No new GPU results are claimed.

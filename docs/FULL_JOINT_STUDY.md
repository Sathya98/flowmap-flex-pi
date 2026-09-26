# Full-joint Flow Maps study

The main comparison uses all four generated streams and full WAM fine-tuning.
Keep the frozen encoders and the task's observation/action representations.
Smaller stream subsets and parameter-efficient tuning belong in ablations.

## Checkpoint lineage

| Training setting | Released FM baseline / frozen teacher | Self-distillation initialization | Evaluation |
| --- | --- | --- | --- |
| LIBERO | `flex-pi/flexpi-libero`, `step_010860.pt` | general AGIBOT-pretrained FlexPi | LIBERO and LIBERO-Plus |
| RoboTwin 2.0 | `flex-pi/flexpi-robotwin`, `step_048060.pt` | general AGIBOT-pretrained FlexPi | clean and randomized RoboTwin |

The task releases are documented in their [LIBERO](https://huggingface.co/flex-pi/flexpi-libero)
and [RoboTwin](https://huggingface.co/flex-pi/flexpi-robotwin) model cards. Preserve
`config.yaml`, `dataset_stats.json`, and `checkpoints/weights/` when acquiring them.
The baseline is normal FlexPi deployed full joint. The separate
[FlexPi* LIBERO release](https://huggingface.co/flex-pi/flexpi-libero-fulljoint-star)
(`step_021690`) was fine-tuned without stream dropout: label it separately if added;
do not silently replace the normal baseline with it.

The [paper's training protocol](https://arxiv.org/html/2608.10860v3#S4.SS1)
uses AGIBOT-pretrained FlexPi before benchmark fine-tuning. As checked on
2026-09-17, the [public organization](https://huggingface.co/flex-pi) lists the
three task-finetuned models above; a general AGIBOT model was not found there.
The [live upstream README](https://github.com/geyan21/flex-pi#-model-download)
explicitly lists AGIBOT-pretrained weights as a planned release.
An actual AGIBOT checkpoint path and its architecture/provenance are therefore
still needed. Never substitute a task teacher, random initialization, or raw Wan
weights for this arm. If transfer changes I/O shapes, record every reinitialized
parameter and use identical transfer rules for the matched FM control.

Distillation starts the student from the task checkpoint and freezes a separate
copy as teacher. Self-distillation loads AGIBOT weights and starts a fresh
benchmark optimizer/schedule; no task-finetuned weights or external teacher.
Self-distillation follows the reference 75% independently uniform diagonal FM /
25% uniform-triangle self-target mixture, learned two-time loss weighting, and
current-student detached targets. Evaluation EMAs 0.999/0.9999 are both saved;
preselect 0.9999 for the main tables and use its exported checkpoint, never
select EMA decay on LIBERO-Plus. See FLOW_MAP_REFERENCE_AUDIT.md for exact
remaining architecture, reduction and optimizer differences.

Only LIBERO and RoboTwin have training runs. Plus is held-out evaluation of the
LIBERO checkpoints and must not be used for checkpoint selection or tuning.

Future matrix commands select full teacher-input gradients for LMD, endpoint
loss for PFMM, and raw plus EMA checkpoints for both Flow Map training families.
Primary external distillation uses fixed weighting; learned weighting remains
an explicit controlled comparison. Table templates record these choices. See
[FLOW_MAP_CONTROLS.md](FLOW_MAP_CONTROLS.md) before submitting the full study;
existing queued pilots intentionally retain their earlier variants.

## Rows and columns

At each NFE in **1, 2, 4, 8, 16**, include these eight rows:

| Row | Objective |
| --- | --- |
| FlexPi baseline | FM |
| Distilled Flow Maps | LMD |
| Distilled Flow Maps | EMD |
| Distilled Flow Maps | PFMM (fixed Euler-teacher composition variant) |
| Self-distilled Flow Maps | LSD |
| Self-distilled Flow Maps | ESD |
| Self-distilled Flow Maps | PSD-M |
| Self-distilled Flow Maps | PSD-U |

Use separate tables for LIBERO (Spatial/Object/Goal/Long/average), LIBERO-Plus
(seven perturbation categories and task-weighted total), and RoboTwin
(clean/randomized/average). The generator writes blank CSV templates; no paper
score is copied into a new NFE cell. Report success with uncertainty, episode
counts, seeds, end-to-end and denoising-only latency, and peak memory. Keep a
future-prediction quality table for RGB/DINO/geometry if making world-model
quality claims in addition to control claims.

With one training seed, this is **14 new training runs** (seven objectives times
two settings), plus two released baselines. They produce **120 aggregate table
rows** across the three evaluations and five NFEs. Tasks/episodes multiply that
rollout cost. NFE is swept at evaluation; it does not require retraining each map.
Use multiple training seeds for final conclusions; a single seed is a pilot.

## Controls and limits

- Flex-joint arm (not part of the main table): `configs/flowmap_libero_lmd_flex.yaml`
  and `configs/flowmap_libero_lsd_flex.yaml`.
  - They sample presence and joint flags per sample with the released recipe (p = 0.5,
    cross-modal prediction on), so one flow map serves every inference regime.
  - The LMD teacher (released LIBERO FlexPi) was trained with the same dropout.
  - The LSD flex config starts from the LIBERO checkpoint, not AGIBOT.
  - Optional: `flex_joint.share_within_microbatch` draws one regime per microbatch.
- All new main runs have every stream present and jointly generated. The released
  flexible baselines were trained with dropout, so training regime is an additional
  difference. Include `--include-fm-control` to fine-tune FM from the same AGIBOT
  checkpoint with the same full-joint recipe before attributing gains solely to
  the objective. Keep the released baseline row as requested.
- Match data, preprocessing, normalization, geometry, task coverage and selection
  rules. The generator defaults to the task presets: LIBERO 20 epochs and RoboTwin
  6 epochs, with `--epochs` for explicit budget changes. Verify saved release
  configurations before expensive runs; these commands are a proposed protocol,
  not a claim of bit-for-bit reproduction of unpublished training details.
- Match evaluation seeds, step limits, reset/warmup, action chunk/replanning,
  instructions, depth source and simulator versions. Use the same training
  normalization statistics for a task across all methods.
- Compare eager BF16 against eager BF16 initially. Existing FM TensorRT engines
  do not implement two-time maps. Report any optimized FM system as a separate
  deployment comparison, rather than mixing software stacks in a speedup claim.
- The map implementation uses a common sigma schedule. Retain and record the
  released FM schedule for the primary baseline; a schedule-matched FM diagnostic
  helps separate schedule effects. Count actual joint forwards as NFE.
- Record both benchmark fine-tuning GPU-hours and incremental distillation cost.
  A distilled model inherits its teacher's training cost; self-distillation does
  not receive those task-finetuned weights. Equal epochs are not equal FLOPs for
  objectives with different derivative/teacher costs.
- The implemented PFMM variant uses a fixed FM teacher's Euler composition;
  optional staged teacher replacement and interval widening are not implemented.
  The paper also permits fixed-teacher distillation. Our numerical FM teacher
  and velocity-normalized residual must be distinguished from its trained-map
  teacher recipe. Training now defaults to uniform-triangle time sampling.

## Generate the plan

```bash
# Review task-checkpoint paths and the pending AGIBOT dependency.
python scripts/flowmap_matrix.py --format json

# Distillation arm can be planned from released task weights alone.
python scripts/flowmap_matrix.py --family distillation

# Main study, after obtaining the actual general pretrained weights.
python scripts/flowmap_matrix.py --agibot-checkpoint /path/to/agibot.pt \
  --num-processes 8 --seeds 42 43 44 --include-fm-control \
  --tables-dir runs/flowmap_fulljoint/tables --format json
```

No command above downloads weights or starts a training/evaluation job. The
checkpoint files and datasets are prerequisites; the plan does not assert they
exist locally. Pin source revisions/hashes when downloading the actual weights.
The local evaluation entry points cover LIBERO and RoboTwin; a complete
LIBERO-Plus launcher/protocol still needs to be integrated and verified. Merely
naming its table in this plan does not mean those rollouts have been implemented
or run. Use a predeclared validation rule to select a checkpoint per training
run, then record its exact path in the tables before evaluating every NFE.


## First LIBERO distillation: debugging versus tuning

`flowmap_libero_pfmm_smoke` is only an infrastructure check: 50 optimizer
updates, effective batch 4 on four GPUs, and four Euler teacher steps. Its
checkpoint is not a tuned result. Start the tuning runs independently from the
released teacher, without resuming the smoke optimizer or its short schedule.

`configs/flowmap_libero_pfmm.yaml` provides a full-budget starting recipe:
20 epochs (no 50-step cap), learning rate 1e-5, effective batch 192 on four GPUs
(microbatch 1, accumulation 48), and 16 teacher steps. The batch matches the
existing LIBERO task preset evaluated on four GPUs; it does not establish the
original release's global batch. Adjust microbatch/accumulation together after
measuring GPU memory. Training already uses 5% warmup, cosine decay and gradient
clipping. The epoch budget is a starting point, not evidence of convergence.

Before committing to this budget:

1. Check finite losses/gradients, peak memory, throughput and checkpoint saving
   with the smoke job. Measure again at the chosen teacher budget.
2. Compare teacher targets at 8, 16 and 32 Euler steps using the **same**
   observations, noise, source/target times and all four streams. Include long
   intervals and compare per-stream normalized endpoint differences, masking
   anchors and padding. Use 64 steps if the 16/32 difference remains material.
   Choose a tolerance before comparing results; report the convergence curves.
   Sixteen is provisional, not a demonstrated accuracy threshold. This is
   teacher integration NFE; student evaluation still sweeps 1/2/4/8/16.
3. Screen learning rates 3e-6, 1e-5 and 3e-5 at equal data/compute budgets.
   Compare checkpoints on a fixed set of LIBERO development rollout seeds at
   NFE 1/2/4, including the released baseline under the same protocol. Reserve
   separate rollout seeds for final reporting. Do not select on LIBERO-Plus.
   Low imitation loss alone does not establish good task performance. Budget
   comparable tuning effort across objectives and record the search cost.
4. Run the selected recipe for the initial 20-epoch budget, inspect development
   performance throughout, and extend only if it is still improving. Replicate
   the selected settings with multiple training seeds for final reporting.

The current LIBERO data config has no separate validation dataset, and the
trainer's built-in evaluation reuses training samples. Full-run inline evaluation
is therefore disabled; the development rollout protocol above must be run
separately. Neither target-convergence calibration nor automated rollout-based
checkpoint selection is implemented by this Slurm script. Retained checkpoints
are limited to four weights/two states: evaluate or archive candidates before
rotation if they are needed for selection.

After GPU and teacher-convergence checks, the full run is launched with:

```bash
sbatch scripts/slurm/flowmap_libero_pfmm.sbatch
# Example after calibration chooses a different teacher budget / learning rate:
sbatch scripts/slurm/flowmap_libero_pfmm.sbatch \
  model.flow_map.teacher_steps=32 learning_rate=3e-6
```

The two-day allocation is not a measured time-to-completion estimate. Periodic
checkpoints permit resuming via `resume=/absolute/path/to/checkpoints/state/step_NNNNNN`.
The experiment generator explicitly passes `--teacher-steps` (default 16) to
PFMM runs rather than inheriting the four-step debugging value. LMD and EMD use
teacher vector-field queries and do not use this Euler composition setting.

## RoboTwin runs, 2026-09-26 (time-sampling comparison)

Four 2,000-update runs (effective batch 192, one 4×H100 node each), from the released
RoboTwin checkpoint `step_048060`: {LMD with full teacher gradients, LSD} × two ways of
sampling the off-diagonal pairs (s, t) (`flow_map.time_sampling`, flowmap_core):

- **grid** (`configs/flowmap_robotwin_{lmd,lsd}_grid.yaml`): exactly the maps that 1-step and
  2-step sampling compose, 50/50 throughout — 1-jump: s = 1, t ~ U[0, 1]; 2-jump:
  s ∈ {1, 0.5}, t ~ U[s − 0.5, s]. NFE ≥ 4 queries untrained sources (generalization only).
- **curriculum** (`..._curriculum.yaml`): the maximum jump grows 0.25 → 1.0 over updates
  0–1000 (Boffi et al., self-distillation App. F.2, untested there), then all jumps with the
  jump size uniform instead of uniform-by-area (jumps > 0.9: ~10% of pairs instead of 1%).

Deviation: **LSD starts from the task release**, not AGIBOT (unavailable); label it so.
Settings shared in `configs/flowmap_robotwin_study.yaml`; run directories
`runs/flowmap_fulljoint/robotwin_{lmd,lsd}_{grid,curriculum}_s42_20260926`.

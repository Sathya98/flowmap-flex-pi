# Diagnosing noisy full-joint PFMM losses

These are controlled numerical checks, not policy success measurements. Training
now defaults to uniform-triangle time sampling, as requested on 2026-09-17.
Loss weighting and DINO parameterization retain their previous settings.
The GPU diagnostics never update weights.

## What the code currently does

All four streams use a common source sigma `s` (1 = noise, 0 = data) and target
sigma `t <= s`. At full strip width, the default uniform-area sampler has joint
density 2 and draws `s = sqrt(u), t = s*(1-w)` for independent uniform `u,w`.
This is distributionally equivalent to sorting two independent uniforms, as in
Boffi's reference. Restricted strips also use uniform area, without endpoint
clipping. The old `conditional` option draws `s ~ Uniform(0,1)`, then
`t | s ~ Uniform(0,s)`, giving joint density `1/s`. Earlier completed smoke
runs used that old proposal; both pending 50-update pilots use the new default.
The fixed diagnostic panel keeps BOTH proposals so old/new results remain
comparable on identical cases.

The released DINO head predicts clean features. Its output is converted using
`v = (x_s - head_output) / max(s, 0.05)`. The flow-map student retains this
checkpoint-compatible parameterization, with additional time-delta conditioning.
For nonzero intervals the head should be interpreted through the resulting map;
its implicit target is not necessarily the clean demonstration feature.

Consequently, fixed head-output errors are amplified by `1/max(s,0.05)^2` in
velocity MSE (up to 400). The old conditional sampler puts 5% of samples below
0.05; the new uniform-triangle default puts 0.25% there. This is a mechanism for
variance, not proof of its contribution on the actual model. The floor also
means the conversion is not the exact unclipped flow-matching identity below
0.05. Do not remove it or reinterpret pretrained head weights without testing.

The joint predictor computes the conversion in FP32, then rounds it back to
BF16 in these smoke runs. Diagnostics separately measure output-rounding error;
this does not measure all neural-network BF16 error.

## What Boffi et al. report

Source: [How to build a consistency model: Learning flow maps via self-distillation,
v2, Section 3 and Appendices F/G](https://arxiv.org/html/2505.18825v2).

- Loss gradients vary with the time pair. Their learned two-time log-variance
  weighting reduces that variability (Equations 19–20).
- Their experiments sample uniformly over the triangle, mixed with diagonal
  FM samples for self-distillation (75% diagonal in the reported setups).
- They remove the squared-interval prefactor from the progressive objective
  because it contributes gradient variance (Equation 22).
- ESD was unstable in their image experiments. This does not establish that
  our external-teacher EMD will fail, or explain our PFMM smoke by itself.

Our code already uses normalized velocity residuals for its progressive
objectives. It does **not** currently implement the paper's learned time weights
or its diagonal/off-diagonal mixture. The PFMM implementation is a fixed FM
Euler-teacher composition variant. The paper permits fixed-teacher PFMM;
staged replacement/interval expansion are optional. Our specific adaptation
uses numerical velocity integration instead of a learned map teacher.
These differences must remain explicit in experiment descriptions.

## Checks implemented

`tests/test_flowmap_diagnostics.py` checks:

1. Source-time distributions, including low-sigma probability mass.
2. DINO endpoint-error amplification and the sigma floor.
3. Teacher Euler convergence against an analytic ODE solution.
4. Exact loss and gradient repeatability on tiny real model components for all
   seven objectives (also freezes the random intermediate time for PSD-U).
5. Probe agreement with the production loss and the relation between velocity,
   mapped-endpoint and implicit clean-endpoint residuals.

`scripts/flowmap_diagnostics.py` loads the real released teacher and compares
its initialized student with all three 50-update smoke checkpoints:

- one fully unpadded episode-start example per LIBERO suite, recorded by index and prompt;
- exact repeats and separate data-only, noise-only and time-only changes;
- paired conditional and uniform-triangle time proposals;
- identical inputs/times/noise across checkpoints;
- 8/16/32/64-step teacher targets on six fixed intervals;
- per-stream losses, teacher velocity magnitude along the integration path,
  normalized target differences, DINO conversion diagnostics, timing and memory.

Checkpoints are compared at the same **16-step evaluation teacher budget**,
regardless of their training teacher budget. Student conditioning is rebuilt
when loading each checkpoint. A separate frozen teacher remains unchanged.
The small panel is exploratory and is not a substitute for held-out rollout
selection. Summary group means include only identical named cases, not training
minibatches sampled at unrelated times. GPU diagnostics do not compute full
parameter gradients; repeatable backward tests use the tiny CPU models.

Outputs are incremental `measurements.jsonl`, `summary.json`, a resolved config,
and a manifest containing cases, checkpoint paths and source hashes/snapshots.
`map_endpoint_residual_mse` uses `(s-t)^2` times velocity residual MSE.
`head_vs_implied_teacher_endpoint_mse` uses the target implied by the converted
DINO head; it is distinct from `head_vs_clean_data_mse`. Neither alone is a
rollout-quality score. No sampler, stream weight, or training default is changed
based on these diagnostics automatically.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_flowmap_diagnostics.py -v
# Once all three referenced smoke checkpoints exist:
sbatch scripts/slurm/flowmap_libero_diagnostics.sbatch
```


## Gradient accumulation logging

The original trainer logged only the last microbatch at an optimizer boundary,
averaged across ranks. This is misleading when accumulation exceeds one.
All three initial smoke jobs used accumulation **one**, so their loss variability
was not caused by this reporting bug; their effective batch was only four.

The trainer now accumulates detached scalars across the complete update. It
reports a sample-weighted mean for total/per-stream losses, the minimum and
maximum local microbatch means across all ranks, and the actual example count.
The range describes microbatch means, not individual-example errors. Gradient
computation and optimizer scaling are unchanged. Per-stream ranges and gradient
norms are retained in `train_update_metrics.jsonl` at each logging boundary,
including when W&B is disabled. Windows reset at every optimizer boundary.

`tests/test_update_metrics.py` checks multi-rank reduction, unequal microbatch
sizes, reset, and the actual trainer loop with two accumulated updates whose
means differ from their last microbatch losses.

A batch-192 pilot is prepared (microbatch 1 x accumulation 48 x four GPUs):

```bash
# First validate two accumulated updates and measure runtime (~24 minutes of
# steady training at teacher_steps=16, plus startup/save overhead).
sbatch --time=01:00:00 scripts/slurm/flowmap_libero_pfmm_accum_smoke.sbatch max_steps=2
# A separate, fresh 50-update run; do not resume the two-step cosine schedule.
sbatch scripts/slurm/flowmap_libero_pfmm_accum_smoke.sbatch
```

Steady timings from updates 31–50 imply about 10.0 hours for 50 updates at
teacher_steps=16, or 11.1 hours at teacher_steps=32, with accumulation 48.
These are linear extrapolations, not measured accumulated-run runtimes; startup,
I/O and hardware variation add uncertainty. The 50-update script requests 14h.
Fifty batch-192 updates expose 48 times as many examples as the batch-4 smoke;
any optimization comparison must account for that difference. Neither run is
an adequate replacement for the full tuning budget.


## Stability gates after the first diagnostic run

Review of job 26857419 found two blockers. Its main example had fully padded
future images, so zero visual losses could not support any DINO/RGB/geometry
conclusion. On the action coordinates, the fixed-input losses increased from
about 0.0037 for initialization to 0.77–0.88 after the old smoke training.
This is not a stable-training result. Those old smoke checkpoints are retained
for diagnosis, not recommended as initialization for longer training.

The trainer previously loaded pretrained/weight-only-resume tensors *after*
`accelerator.prepare`. Installed DeepSpeed ZeRO2 clones separate optimizer
master weights during initialization and copies them back during `step`.
Loading the module alone afterwards leaves those masters stale. The first
step can therefore overwrite the intended warm start even at learning rate zero.
Weight-only initialization now happens before optimizer/master creation; full
training-state resume remains after preparation and restores optimizer state.
A regression test checks both orderings, including a zero-LR master-copy model.

The GPU gates are:

1. Two updates with learning rate zero; compare every checkpoint tensor shared
   with the pretrained release and require exact equality at the saved dtype.
2. Only after that passes, two updates at effective batch 192, nonzero LR;
   require finite update means/ranges and gradient norms, 192 reported examples,
   and complete weights/state checkpoints. Record peak allocated/reserved GPU
   memory. This validates mechanics, not convergence.
3. Run the corrected fixed-input panel on the newly trained checkpoint, compared
   with the released initialization and old smokes. Every selected example must
   have no padded image or action frames. Decide whether a longer stability
   pilot is warranted from these results; do not automatically launch a full tune.

`flowmap_check_pilot.py` runs the numeric/checkpoint gates, and writes
`pilot_check.json`. Failed gates fail their Slurm job and prevent its dependent
job from starting. No task success or final convergence claim follows from
these checks alone.


## Matched LMD pilot

`flowmap_libero_lmd_pilot` compares Lagrangian map distillation with the longer
PFMM pilot: fresh released LIBERO initialization, the same frozen released
teacher, all four streams, full parameter tuning, 50 optimizer updates,
effective batch 192, peak LR 1e-5, and checkpoints every 10 updates. It uses
exact forward AD with differentiable temporal derivatives (`dt_method=ad`,
`detach_derivatives=false`), and no additional diagonal data loss. LMD queries
the frozen velocity at its predicted mapped state; it does not use PFMM's
16-step Euler composition as its training target.

```bash
sbatch scripts/slurm/flowmap_libero_lmd_pilot.sbatch
# Pass each retained checkpoint to a dependent diagnostic job:
sbatch --dependency=afterok:TRAIN_JOB scripts/slurm/flowmap_libero_pilot_diagnostics.sbatch \
  --lmd-probe --student step10=runs/flowmap_pilots/libero_lmd_batch192_TRAIN_JOB/checkpoints/weights/step_000010.pt
```

The optional `--lmd-probe` adds exact LMD residuals on the same fixed cases,
including the released initialization. Existing PFMM teacher-composition
endpoint/velocity errors remain a common external-reference metric, so raw
LMD training loss is not compared numerically to raw PFMM training loss.
`lmd_*` summary metrics are the LMD residual; unprefixed metrics retain their
previous meaning. The added diagnostic does not update weights or compute
full parameter gradients; the training run checks the latter.

Exact temporal derivatives disable activation checkpointing in the dual-number
forward and use explicit attention. Full-model GPU memory fit and runtime have
not yet been established. An out-of-memory failure would be a feasibility
result for this implementation, not evidence that the objective is numerically
unstable. The job requests 14 hours as a cap, not a measured runtime estimate.
No LSD job is part of this comparison.

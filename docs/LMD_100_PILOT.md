# Full-joint LMD: 100-update decision point

Pedro approved a 100-update pilot on 2026-09-18 before deciding on longer
training or additional objectives. Action, RGB, DINO and pointmap remain jointly
trained. The objective, teacher-input gradient, batch size (192), peak LR (1e-5)
and full-parameter tuning are unchanged.

Run: `runs/flowmap_fulljoint/libero_lmd_100_s42_20260918`.
Submitted job **26887233**, dependent on successful checkpointed completion of
parent job **26885139**. At submission it is pending that dependency.

`flowmap_libero_lmd_100` inherits the tested full configuration and sets:

- Total horizon: 100 optimizer updates, including updates restored from the parent.
- LR schedule: the existing 5% warmup rule gives five warmup updates, followed by
  cosine decay. On resume the schedule is deliberately rebuilt at the restored
  global step. Early parent updates actually used the old 100-update warmup;
  they are not retroactively equivalent to a fresh short-schedule experiment.
- Checkpoints and fixed training-clip previews: every 25 updates and at allocation
  boundaries. Protect raw and EMA weights at 25, 50, 75 and 100; retain the latest
  two full states with optimizer, scheduler, RNG, data progress and EMA state.
- Offline W&B with process-local credential isolation.

The previous run has a `STOP` marker. It finishes its current optimizer update,
saves full state and previews, and cannot queue another 2,000-update segment.
The new run records `resume_from_run` in its manifest and starts only after the
parent job succeeds. Before loading, it validates that parent's complete state
and previews. It restores Adam moments and progress, then changes the schedule.
This is a recorded schedule intervention, not an exact same-schedule continuation.

At approximately 13–13.3 minutes/update, the complete 100 updates require about
22 active GPU-allocation hours on four H100s, plus startup, saves, evaluations
and queue delays. Allocation boundaries may require a second segment. The chain
stops at 100 and does not automatically launch other objectives.

## Comparison after training

Use the dedicated `fm_env_libero` environment; the training environment does not
contain the simulator packages. The existing evaluation environment has MuJoCo
3.3.2, robosuite 1.4.0 and bddl 3.6.0. No rollout result is claimed by this check.

In a four-GPU allocation, with `RUN_DIR` set to the new run's absolute path:

```bash
source /gpfs/scratch1/shared/faster-wams/env_flexpi_libero.sh
GPUS=0,1,2,3 NUM_TRIALS=10 bash "$RUN_DIR/code/scripts/flowmap_libero_compare.sh" "$RUN_DIR"
```

This uses the unchanged task-finetuned FM teacher and the raw 100-update LMD
checkpoint on all 40 LIBERO tasks, with the same seed, initial states, full-joint
streams and NFEs 1, 2 and 4. Ten trials/task is a preliminary comparison;
`NUM_TRIALS=50` selects the paper's full evaluation trial count. Compare task
success with FM at the same NFE and the usual four-step FM reference. Inspect
future predictions too, but training-clip previews are not held-out success rates.
No simulator jobs are automatically submitted by the training chain.

Decide whether to extend LMD or test EMD/PFMM using those results. A lower
training loss alone is insufficient to establish successful distillation.

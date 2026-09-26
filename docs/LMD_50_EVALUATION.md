# Step-50 LMD versus normal Flex-Pi FM

Requested by Pedro on 2026-09-20 after checkpoint writes failed in the 100-update
pilot. The latest complete checkpoint is update 50; update 98 exists only in
metrics and previews, not as recoverable model weights.

Evaluation directory:
`runs/flowmap_fulljoint/libero_lmd_100_s42_20260918/comparison/step_50_20260920`.
It contains a frozen code snapshot and a manifest with source hashes, checkpoint
identities, saved-config hashes, trial counts and seed. Normalization statistics
are identical between the baseline and student.

- Baseline: released `flexpi-libero`, `step_010860.pt`, normal flow matching.
- Student: raw full-joint LMD `step_000050.pt`; EMA is not used for this pilot.
- Both: action/RGB/DINO/pointmap generated jointly; all visual inputs present.
- NFEs: 1, 2 and 4, no dynamic step skipping, seed 42, same simulator initial states.
- LIBERO: all four suites, 10 tasks per suite, 10 trials per task per setting.
  This is 400 episodes/setting and 2,400 total, a preliminary comparison rather
  than the paper's 50-trial protocol.
- Environment: `fm_env_libero`, MuJoCo 3.3.2; run-local LIBERO path configuration.
- Evaluation writes rollout videos and JSON summaries, not model checkpoints.

Job **26939813** runs one simulator episode with FM at NFE 4 and LMD at NFE 1.
Both must produce complete results; task success is measured, not used as a
smoke gate. Array **26939816** depends on successful smoke completion, and runs
one four-H100 case at a time:

| Array index | Model | NFE |
|---|---|---|
| 0 | FM | 4 |
| 1 | LMD | 1 |
| 2 | FM | 1 |
| 3 | LMD | 2 |
| 4 | FM | 2 |
| 5 | LMD | 4 |

Each completed case writes `sweep/<model>_nfe<n>/summary_4suite.json`, per-task
results and `validated.json`. Validation rejects missing tasks, duplicate task
results, and missing/duplicated episode identities. Failed jobs do not count as
zero-success policies and do not overwrite earlier evaluation directories.

Compare LMD against FM at the same NFE and against the usual four-step FM
reference. Report per-suite and aggregate task success. The earlier training-set
future previews are separate diagnostics, not held-out LIBERO success results.

The training output is on scratch-shared, but the account's user quota still
applies there. No training restart or checkpoint deletion is part of this
evaluation submission.

## First sweep and retry

The validated 400-episode results are LMD NFE 1: **395/400 (98.75%)**, and
LMD NFE 4: **391/400 (97.75%)**. Both cover all 40 tasks with ten trials each.
They ran on gcn127 and gcn134 respectively.

All three FM cases and LMD NFE 2 were incomplete: depth-map range assertions
and renderer/process aborts occurred on gcn108 in all four cases. Their partial
summary percentages are not full-benchmark scores. Node association is evidence
for an environment problem, not a confirmed diagnosis of the underlying fault.

Retry array **26946382**, indices 0,2,3,4 with one four-H100 case at a time,
excludes gcn108. It reuses the exact frozen code, models, seeds, initial states
and protocol. Results go to the sibling directory
`comparison/step_50_20260920_retry_gcn108`; original outputs are preserved.
The FM comparison remains pending until the retries pass full coverage checks.

The FM NFE-4 retry completed on gcn160: **397/400 (99.25%)**. This gives a
complete standard-baseline comparison with LMD NFE 1 at 395/400; the difference
is two episodes in this ten-trial-per-task pilot, not evidence of equivalence.
FM NFE 1 failed again on gcn100 with simulator depth assertions. It is queued
unchanged on gcn160 as **26947406_2**, after the current array ends. The pending
FM NFE-2 task was replaced by **26947407_4**, also on gcn160, after that retry.
LMD NFE 2 continues as **26946382_3** on gcn125. All old outputs remain intact.

## Completed pilot

All six cases have now passed full 40-task/400-episode validation. The final
complete results, excluding all partial failed attempts, are:

| NFE | Normal FM | Step-50 LMD |
|---|---|---|
| 1 | 393/400 (98.25%) | 395/400 (98.75%) |
| 2 | 394/400 (98.50%) | 392/400 (98.00%) |
| 4 | 397/400 (99.25%) | 391/400 (97.75%) |

The comparison directory contains `step_50_summary.md`, `.csv` and `.json`, with
per-suite results, paired episode outcomes and source directories in the JSON.
Normal FM already performs strongly at one step. The two-episode one-step gain
does not establish a substantial distillation advantage in this small pilot.
These policy-success numbers do not measure future-stream quality or latency.
All submitted evaluation jobs have finished; no further evaluations are queued.

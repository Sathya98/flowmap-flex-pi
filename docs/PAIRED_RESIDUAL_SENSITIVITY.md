# Paired observed-future sensitivity experiment

Launch `sbatch scripts/slurm/paired_residual_libero.sbatch` from the repository.
The original released model is loaded once. No model training is performed.

Use LIBERO spatial task 0, 32-action chunks and four denoising steps. Video and
action are jointly generated; DINO and pointmap supply observation anchors.
Only chunk 1 (the second chunk, zero-indexed) is disturbed and measured.

1. Calibrate on initial states 0–3, independently of the evaluation set. Run clean
   controls and try delays of 4, 8, 16 actions. If none yields a mixture of
   terminal success/failure, try translation biases of 0.1, 0.25, 0.5 in normalized
   controller-command units. Bias direction cycles +x, -x, +y, -y by state index.
   Stop at the first mixed level. If no level is mixed, select the first failing
   level, or the last tested level if all succeed. Record this limitation.
   No JVP scores are computed or used in calibration.
2. Save `selection.json` before evaluating eight disjoint initial states 4–11.
   Each gets a clean and a perturbed rollout. Reset the simulator and all seeds
   for both members; policy/environment seed is `42 + initial_state_index`.
   Alternate which condition runs first. Verify the complete predicted action
   chunk is byte-identical before applying the disturbance.
3. Disturb actions 32–47 only. A delay holds zero pose increments and the previous
   gripper command, then plays lagged commands until action 48. Discard the lagged
   backlog at action 48 and resume the original command at index 16. A translation
   bias changes only xyz controller commands, clipping to [-1,1]. All commands at
   action 48 and beyond are untouched. Full planned/executed traces are saved.
4. After 48 executed actions, compute the VAE residual and the same conditional
   replay JVP as in RESIDUAL_SENSITIVITY.md, scoring original actions 48–63.
   Use four common unit Rademacher probes within each pair. Record one prespecified
   measurement per episode. Already-completed tasks have no prospective score.
5. Primary outcome: task success/failure by the standard 400-action limit.
   Secondary outcome, fixed before calibration: task completion by action 96.
   The latter identifies delayed completion and is not terminal failure.

Report raw AUROC for E, S_res, S_rand, and R. Compare fixed L2=1 logistic models
with log10 inputs: E alone; E + S_rand; E + S_rand + S_res. Use leave-one-pair-out
predictions, holding out both members and fitting standardization only on training
folds. Compare AUROC, Brier score, and log loss. Eight pairs are exploratory and
cannot validate calibration or establish a reliable incremental benefit.

S_rand should match within a pair: the predicted future, Jacobian, and random
probes are identical before the execution disturbance. E and S_res additionally
observe the realized error. Therefore, separating paired outcomes does not alone
show that Hutchinson was falsely certain; the stronger comparison is whether
directional sensitivity adds beyond E. Conditional replay can differ from the
sampled action plan, and its RMS difference is also logged.

Outputs under `eval_results/paired_residual_JOBID/`: protocol and source snapshots,
calibration outcomes, frozen selection, all episode outcomes and videos, action
traces, raw JVP measurements, paired CSV, scatter PNG/PDF, and summary JSON.

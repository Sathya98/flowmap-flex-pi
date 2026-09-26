# Observed-future action sensitivity pilot

Run `sbatch scripts/slurm/residual_sensitivity_libero.sbatch` from the repository.
The pilot uses the released Flex-π LIBERO checkpoint, spatial task 0, six initial
states, four denoising steps, and three measured chunks per episode. Video and
actions are jointly generated; DINO and pointmap provide observation anchors.
The full 32-action chunk executes without replanning. This longer open-loop
execution differs from the usual 10-action replanning evaluation.

## Alignment and derivative definition

This checkpoint's DINO stride keeps only the far future at action 32, leaving
no remaining actions. Instead, capture RGB at actions 0, 4, 8, 12, 16 and encode
that causal clip with the model's VAE. Compare observed latent slot 1 with
predicted latent slot 1, both aligned to action 16. Slot 0 is the current
observation; predicted slot 2 describes action 32 and is held fixed.

Define `a(f)` by replaying action denoising from the *same initial action noise*,
with the predicted clean visual trajectory fixed as conditioning (visual time
zero). Vary only slot 1, and return actions 16:32 in the seven active normalized
action channels [0, 1, 2, 3, 4, 5, 18]. Forward-mode autodiff traverses all four
Euler steps; no Jacobian is materialized, and no finite differences are used.
The replay's text, proprioception, current visual anchors, and unobserved future
slot remain fixed. Executed actions are omitted from the score, but their latent
coordinates are regenerated inside the replay.

This explicitly defines a **conditional replay sensitivity**, not an intrinsic
derivative between the joint sampler's two outputs. The original rollout actions
are executed unchanged. `replay_vs_original_rms` records how far the replay plan
is from that sampled plan. A substantial discrepancy limits the diagnostic's
interpretation as sensitivity of the original plan. Clean future conditioning
also differs from the noisy visual states used during joint generation.

For `r = observed - predicted`, log:

- `E = ||r||₂`.
- `S_res = ||J (r / ||r||₂)||₂²`.
- `S_rand = mean ||J z||₂²`, with four independent unit Rademacher directions
  on exactly the same latent coordinates.
- `R = S_res / (S_rand + 1e-12)`.

The random baseline estimates `||J||_F² / d`, not the unnormalized Hutchinson
trace. Zero residuals have undefined `R` (`null`). Individual random scores and
a flag for a denominator above epsilon are retained. Computation uses the
checkpoint's BF16 arithmetic and the repository's forward-AD-compatible attention
and normalization paths; norm reductions use FP32.

## Outputs and limitations

`eval_results/residual_sensitivity_JOBID/` contains an incremental JSONL stream,
resolved configuration, source snapshots, manifest, rollout videos, CSV scores,
summary JSON, and `sensitivity_scatter.png` / `.pdf`. Measurement records become
`labeled` records only after the episode reaches success or the evaluation time
limit. Already-successful observations are excluded.

The scatter labels each chunk by **eventual episode success**, and summary AUROC
uses per-episode mean scores to avoid treating correlated chunks as independent
rollouts. `chunk_success` separately records whether the task completed during
the remaining executed part of that chunk; non-completion is not automatically
a failed chunk. AUROC is `null` when only one outcome class is present.

Six episodes and four probes are a quick diagnostic, not validation of a
confidence estimator. Large `R` establishes directional alignment locally; it
does not by itself establish calibration or falsely low uncertainty. A useful
follow-up needs more initial states and held-out failure prediction.

Validation: `PYTHONPATH=src python -m pytest -q tests/test_residual_sensitivity.py`.
Regenerate plots: `python scripts/summarize_residual_sensitivity.py PATH/TO/residual_sensitivity.jsonl`.

# Full-joint LIBERO LMD run

**2026-09-18 update:** Pedro selected a 100-update decision point. The original
2,000-update chain has been requested to stop with a full checkpoint. Its state
continues in `runs/flowmap_fulljoint/libero_lmd_100_s42_20260918`, with the shorter
schedule and 25-update saves described in [LMD_100_PILOT.md](LMD_100_PILOT.md).
The 2,000-update settings below describe the original run.

The old PFMM/LMD pilots and their diagnostics (26861582, 26861592, 26862643,
26862671) were cancelled at the user's request. This run starts from the released
LIBERO task-finetuned model and keeps that checkpoint as a separate frozen teacher.
It uses the full teacher-input LMD gradient, four streams and full WAM tuning.

The pending online job **26868229** was cancelled before training started at the
user's request. Its previously created W&B run contains setup metadata only;
no further use of those credentials or that run is authorized. Replacement runs
use isolated offline tracking and a fresh local identity. The run directory records
submission details, frozen source hashes, full local config and subsequent job IDs.

The first offline hardware job, **26868647**, failed on 2026-09-18 at 02:51
Amsterdam time before its first optimizer update. Native LayerNorm's forward-AD
path returned a float32 tangent alongside a BF16 primal; cross-attention's BF16
linear layer rejected it. No checkpoint or continuation was produced. This was
an execution failure, not evidence of unstable training losses.

Hardware job **26883597** was submitted on 2026-09-18 at 08:54 UTC.
Its snapshot is
`runs/flowmap_fulljoint/libero_lmd_fullgrad_s42_20260918T073439Z`; its
`submission.json` records the replacement job ID. During forward AD, LayerNorm
now computes normalization explicitly in FP32 and casts primal and tangent
together (FP64 remains available for derivative checks). Ordinary forwards retain
native LayerNorm and checkpoint keys are unchanged. CPU regression coverage
includes all four streams in BF16 and numerical checks of gradients through the
JVP; all seven replacement regression tests passed. A small CUDA DiT
JVP/backward check now runs before full-model loading;
its result is saved in `hardware_jvp_preflight.json`. The replacement passed this
check on an H100: native LayerNorm reproduced the BF16-primal/FP32-tangent
mismatch, and the corrected block completed BF16 JVP and parameter backward with
finite gradients. Full-model memory fit, optimizer updates and restart remain
separate hardware checks.

Job **26883597** then failed at 11:01 Amsterdam time, before completing an
optimizer update, with a CUDA OOM in the full student's explicit attention.
The process used 93.20 GiB on a 93.34 GiB GPU. No full checkpoint or continuation
was created. Its failure record and logs are retained, and it has a STOP marker.

Memory-fix hardware job **26884210** was submitted on 2026-09-18 at 09:20 UTC.
Its replacement snapshot is
`runs/flowmap_fulljoint/libero_lmd_fullgrad_s42_20260918T091838Z`; its
`submission.json` records the new hardware job. It restores activation
checkpointing for temporal JVPs by passing primal/tangent pairs explicitly
through non-reentrant checkpoint regions. Backward recomputes the local JVP;
both components remain differentiable. Full teacher-input gradients, all four
streams, resolution, batch size and objective settings are unchanged.
Four checkpoint regressions passed, including FP32/BF16 full-joint loss and
parameter-gradient agreement and a numerical mixed-derivative check. The small
attention test retained 65,536 bytes versus 263,296 without checkpointing, about
75% less; this is not a prediction of total GPU memory. The existing objective
suite also passed (31 tests, 34 subtests). The hardware preflight now additionally
compares checkpointed/uncheckpointed BF16 block outputs and parameter gradients.

The CPU checkpoint/extension and real offline SDK tests passed, as did continuation
and account-isolation checks. Those logs remain under the first offline run's
`validation/`; replacement-specific checks are under the new run's `validation/`.
GPU memory fit and full-state GPU resume remain subject to the hardware gate and
its continuation. The initial two updates use the full 2,000-update schedule.

## Budget and restart behavior

- Initial budget: 2,000 optimizer updates, effective batch 192 (1 × 48 × 4 H100s).
  This is 384,000 sampled examples, about 1.38 passes over 277,492 training frames.
  It is a provisional distillation budget, not a convergence guarantee.
- LR 1e-5, AdamW (0.9, 0.95), weight decay .01, gradient clipping 1; 100-update
  warmup followed by cosine decay over the full 2,000-update horizon.
- The first Slurm job stops after 2 updates using that SAME scheduler horizon.
  It saves complete state and generates previews. Only successful validation of
  checkpoints, finite metrics and previews triggers submission of continuation.
- Continuations resume optimizer, scheduler, RNG, dataloader progress and EMAs.
  Each 24-hour allocation pauses after about 20 hours at an optimizer boundary,
  checkpoints and submits the next segment only if it made valid progress.
  Initialization, evaluation and saves need additional time; no wall-time estimate
  for exact LMD is established yet. The queue chain stops at 2,000 updates.
- Crashes, OOMs, nonfinite metrics or missing artifacts do not auto-retry.
  Cancel the active job to stop the chain, or create `RUN_DIR/STOP` for a graceful
  checkpointed stop. A prepared continuation checks that marker before startup.
- To recover after a failure with an existing complete checkpoint, submit the
  same snapshot with `RUN_DIR train`. A failure before the first checkpoint needs
  the `hardware` phase; if code changed, prepare a fresh snapshot first.
  It loads the latest complete full state. To extend to, for example, 4,000 total
  updates, use `RUN_DIR train 4000`. Do this after the current chain has stopped,
  so two jobs cannot write the same run. The launcher detects a changed horizon
  and explicitly re-anchors the cosine schedule at the restored step. Adam moments,
  EMA state and data progress are retained. Re-anchoring can raise the LR; it is
  a deliberate schedule change, not an exact continuation of the old cosine.

## Checkpoints and visual monitoring

Full state and raw/EMA weights are saved every 250 updates and on each segment
boundary. Keep the latest 2 full states; keep the latest 3 weights of each type,
plus step 2 and every 250-update milestone through 2000. Two CPU FP32 EMAs (0.999,
0.9999) add host memory and disk overhead. Early EMA scores largely reflect
initialization, so previews use RAW weights and are labeled accordingly.

Full states contain DeepSpeed model and all four optimizer shards (including
Adam moments/master weights), scheduler, per-rank RNG states, step/epoch/data
position and both EMAs. BF16 has no GradScaler; Accelerate saves one if enabled
by another precision mode. Continuation checks optimizer shards, scheduler and
all RNG files as well as weights and previews. Use `checkpoints/state/step_*` to
resume; exported `checkpoints/weights/*.pt` files alone are not full resumes.

W&B runs OFFLINE, with no configured workspace and no authentication. Process-local
isolation removes inherited W&B environment settings and redirects credential
lookup to an empty run-local netrc; configuration/cache/data stay under `tracking/`.
Shared credentials, shell profiles and other users' jobs are never modified.
Its ID is persisted in `wandb_run.json`; offline W&B creates a separate local file
per segment, while JSONL histories and optimizer-step axes span the whole run.
There is no live cloud dashboard or automatic sync. Automatic source/git and
machine metadata collection is disabled. Losses and videos remain on disk.
Charts use optimizer step as an explicit axis, so a checkpoint rollback does not
silently discard replayed logs. Log per-stream update mean/min/max losses,
gradient norm, LR, sample throughput and peak GPU memory. No expensive parameter
histogram watches are enabled.

Every 100 updates and at segment endpoints, generate futures at NFEs 1, 2 and 4
for fixed clips from all four LIBERO suites. Reuse Flex-Pi's RGB reconstruction/
GT, DINO PCA and pointmap visualizations; all four rank videos are saved locally.
Save the videos under `eval/` and metrics under `preview_metrics.jsonl`.
The evaluation RNG is isolated and restored, so previews do not change training
noise or interval sampling. These are TRAINING-DATA diagnostics, not held-out
benchmark success rates. Closed-loop LIBERO/Plus evaluation is still needed.

### Local dashboard

From the repository root, run:

```bash
python3 scripts/flowmap_local_report.py runs/flowmap_fulljoint/libero_lmd_fullgrad_s42_20260918T091838Z --serve 8765
```

The server binds only to `127.0.0.1`. Forward port 8765 from the same cluster host
using VS Code Remote SSH or an SSH tunnel, then open `http://localhost:8765` on
your computer. Refresh the page to read new metrics; video playback is not
interrupted automatically. No W&B account or extra Python package is needed.
The report serves only its generated page and preview MP4s, not checkpoint or
credential files. Ctrl+C stops the viewer without affecting training. Omit
`--serve 8765` to generate `dashboard.html` for viewing/copying offline; copy the
`eval/` folder alongside it to retain videos. This HTML snapshot must be regenerated
to include later updates.

All Flow Map trainers enforce isolated tracking, even if a launcher omits the
explicit isolation flag. For future online logging to the user's own account it requires
an explicit `wandb.workspace` and a `FLOWMAP_WANDB_API_KEY_FILE` pointing to a
user-owned mode-600 key file. It injects that key only into the training process;
it never runs `wandb login` or falls back to the shared login. The current Slurm
launcher deliberately forces offline mode. Online operation needs a separately
prepared launch with the user's workspace and key; never put the key in YAML,
command-line arguments, source control or chat. Separate Unix accounts are still
needed for access isolation between people sharing a machine.

## Performance and readiness

BF16, ZeRO-2, GPU fused AdamW, frozen encoders, cached text features, pinned
memory and persistent/prefetched workers are enabled. Regular forwards use fused
SDPA. Exact temporal JVPs use differentiable explicit attention with activation
checkpointing that carries primal/tangent pairs. Recomputing the JVP trades
compute for lower activation memory. Full LMD also backpropagates through the
teacher input. The first hardware stage must
measure memory and throughput; this recipe is not claimed to be fully optimized
or known to fit. Do not switch derivatives or stream subsets silently to fit.

The launch uses a frozen copy of src/configs/scripts, with SHA-256 hashes and
checkpoint/data symlinks. Later working-tree edits cannot change the queued code.

```bash
python scripts/prepare_flowmap_full_run.py
# Use the emitted absolute RUN_DIR. Only this first job is manually submitted.
sbatch --time=02:00:00 RUN_DIR/code/scripts/slurm/flowmap_libero_lmd_full.sbatch RUN_DIR hardware
# Resume after a failure (same total horizon):
sbatch RUN_DIR/code/scripts/slurm/flowmap_libero_lmd_full.sbatch RUN_DIR train
# Or extend after stopping the previous chain (4,000 total updates):
sbatch RUN_DIR/code/scripts/slurm/flowmap_libero_lmd_full.sbatch RUN_DIR train 4000
```

#!/bin/bash -l
#SBATCH --job-name=fm-libero-lmd-full
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out

set -euo pipefail
run_dir="${1:?Pass the absolute prepared run directory}"
phase="${2:?Pass hardware or train}"
target_steps="${3:-}"
[[ -z "$target_steps" || "$target_steps" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid training horizon' >&2; exit 2; }
case "$phase" in hardware|train) ;; *) exit 2 ;; esac
[[ "$run_dir" = /* ]] || { echo 'Run directory must be absolute' >&2; exit 2; }
[[ ! -e "$run_dir/STOP" ]] || exit 0
# Runs from the submit directory: sbatch from the repo root; the env scripts sit next to it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d scripts/slurm ]] || { echo 'Submit from the flowmap-flex-pi root' >&2; exit 2; }
source ../env_flexpi.sh
repo_root="$PWD"   # continuations are submitted from here
# Execute the frozen source snapshot, not a later working-tree revision.
export FLEXPI_ROOT="$run_dir/code"
export FLOWMAP_RUN_DIR="$run_dir"
# The frozen flowmap_core must shadow the editable install of the live tree.
[[ -d "$FLEXPI_ROOT/flowmap_core/src/flowmap_core" ]] || { echo 'Run snapshot lacks flowmap_core; re-prepare it' >&2; exit 2; }
export PYTHONPATH="$FLEXPI_ROOT/flowmap_core/src:$FLEXPI_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN
export WANDB_CACHE_DIR="$run_dir/wandb_cache"
export WANDB_DATA_DIR="$run_dir/wandb_data"
# The offline run also isolates inherited credentials inside _init_wandb.
# Never call wandb login or modify the shared user's global configuration here.
unset WANDB_API_KEY WANDB_ENTITY WANDB_USERNAME WANDB_RUN_ID WANDB_SWEEP_ID
export WANDB_MODE=offline
cd "$FLEXPI_ROOT"
config_name="$(python - "$run_dir/manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('config_name','flowmap_libero_lmd_full'))
PY
)"
parent_run="$(python - "$run_dir/manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('resume_from_run') or '')
PY
)"
python scripts/flowmap_smoke_preflight.py --config-name="$config_name"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [[ "$phase" = hardware ]]; then
  python scripts/flowmap_jvp_preflight.py --output "$run_dir/hardware_jvp_preflight.json"
fi
initial_step="$(python scripts/flowmap_run_state.py step "$run_dir")"
resume_path="$(python scripts/flowmap_run_state.py latest "$run_dir")"
if [[ -z "$resume_path" && -n "$parent_run" ]]; then
  [[ "$phase" = train && -e "$parent_run/STOP" ]] || {
    echo 'A parent run must be stopped before a full-state continuation' >&2; exit 2;
  }
  python scripts/flowmap_run_state.py validate "$parent_run"
  resume_path="$(python scripts/flowmap_run_state.py latest "$parent_run")"
  initial_step="$(python scripts/flowmap_run_state.py step "$parent_run")"
  # Preserve the full diagnostic history; preview paths still point to the parent.
  python - "$parent_run" "$run_dir" <<'PY'
from pathlib import Path
import shutil,sys
parent,run=map(Path,sys.argv[1:])
for name in ('train_update_metrics.jsonl','preview_metrics.jsonl'):
    if not (run/name).exists():
        shutil.copy2(parent/name,run/name)
PY
fi
planned_horizon="$(python - "$run_dir/manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['max_steps'])
PY
)"
saved_horizon="$planned_horizon"
if [[ -n "$resume_path" ]]; then
  saved_horizon="$(python - "$resume_path/trainer_state.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('max_steps',2000))
PY
)"
fi
target_steps="${target_steps:-$planned_horizon}"
overrides=("max_steps=$target_steps" wandb.mode=offline wandb.workspace=null wandb.isolated=true)
[[ "$initial_step" -lt "$target_steps" ]] || { echo 'Requested horizon already reached'; exit 0; }
if [[ -n "$resume_path" ]]; then
  overrides+=("resume=$resume_path" pretrained_ckpt=null)
  if [[ "$saved_horizon" != "$target_steps" ]]; then
    # A deliberate horizon change preserves Adam moments and progress while
    # re-anchoring the LR schedule for the newly requested total horizon.
    overrides+=(resume_reanchor_lr_schedule=true)
  fi
elif [[ "$phase" = train ]]; then
  echo 'Full training requires the successful hardware checkpoint' >&2
  exit 2
fi
if [[ "$phase" = hardware ]]; then
  [[ "$initial_step" = 0 ]] || { echo 'Hardware phase already has state' >&2; exit 2; }
  overrides+=(stop_after_steps=2 max_runtime_seconds=3600)
fi
mkdir -p "$run_dir/launches"
python - "$run_dir" "$phase" "$initial_step" "$config_name" "$resume_path" "${overrides[@]}" <<'PY'
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); job=os.environ['SLURM_JOB_ID']
(root/'launches'/f'{job}.json').write_text(json.dumps(dict(
    job=job,phase=sys.argv[2],initial_step=int(sys.argv[3]),config_name=sys.argv[4],
    resume_path=sys.argv[5],overrides=sys.argv[6:]),indent=2)+'\n')
PY
srun --ntasks=1 --kill-on-bad-exit=1 python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 4 --num_machines 1 \
  --main_process_port "$((15000 + SLURM_JOB_ID % 30000))" \
  scripts/train.py --config-name="$config_name" "${overrides[@]}"
# A failed/OOM run never queues more compute. Generic over configs (LIBERO or RoboTwin):
# the run manifest names the config. Checkpoints AND previews must pass.
python scripts/flowmap_run_state.py validate "$run_dir" --initial-step "$initial_step"
if [[ "$phase" = hardware ]]; then
  [[ "$(python scripts/flowmap_run_state.py step "$run_dir")" = 2 ]] || exit 2
fi
if [[ "$(python scripts/flowmap_run_state.py complete "$run_dir")" = 1 || -e "$run_dir/STOP" ]]; then
  exit 0
fi
next_job="$(cd "$repo_root" && sbatch --parsable --time=24:00:00 --job-name="$SLURM_JOB_NAME" --dependency="afterok:$SLURM_JOB_ID" \
  "$FLEXPI_ROOT/scripts/slurm/flowmap_libero_lmd_full.sbatch" "$run_dir" train "$target_steps")"
printf '%s\n' "$next_job" >> "$run_dir/continuation_jobs.txt"
echo "Checkpoint validated; continuation job=$next_job"

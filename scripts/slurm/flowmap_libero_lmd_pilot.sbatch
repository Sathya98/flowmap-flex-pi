#!/bin/bash -l
#SBATCH --job-name=fm-libero-lmd-b192
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out

set -euo pipefail
# Runs from the submit directory: sbatch from the repo root; the env scripts sit next to it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d scripts/slurm ]] || { echo 'Submit from the flowmap-flex-pi root' >&2; exit 2; }
source ../env_flexpi.sh
cd "$FLEXPI_ROOT"
export PYTHONPATH="$FLEXPI_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN

python scripts/flowmap_smoke_preflight.py --config-name=flowmap_libero_lmd_pilot "$@"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[train] branch=$(git branch --show-current) revision=$(git rev-parse HEAD) job=$SLURM_JOB_ID"
echo '[pilot] LMD with exact temporal derivatives; same frozen LIBERO teacher and initialization as PFMM.'
echo '[pilot] 50 updates, effective batch 192. Exact-JVP memory fit and runtime are not yet measured.'
srun --ntasks=1 --kill-on-bad-exit=1 python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 4 --num_machines 1 \
  --main_process_port "$((15000 + SLURM_JOB_ID % 30000))" \
  scripts/train.py --config-name=flowmap_libero_lmd_pilot "$@"

python scripts/flowmap_check_pilot.py "runs/flowmap_pilots/libero_lmd_batch192_${SLURM_JOB_ID}"

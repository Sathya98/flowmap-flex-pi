#!/bin/bash -l
#SBATCH --job-name=fm-libero-eval
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.out
set -euo pipefail
eval_dir="${1:?Pass the absolute prepared evaluation directory}"
phase="${2:?Pass smoke or sweep}"
[[ "$eval_dir" = /* ]] || exit 2
case "$phase" in smoke|sweep) ;; *) exit 2 ;; esac
# Runs from the submit directory: sbatch from the repo root; the env scripts sit next to it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d scripts/slurm ]] || { echo 'Submit from the flowmap-flex-pi root' >&2; exit 2; }
source ../env_flexpi_libero.sh
export FLEXPI_ROOT="$eval_dir/code"
export PYTHONPATH="$FLEXPI_ROOT/third_party/LIBERO:$FLEXPI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
unset WANDB_API_KEY WANDB_ENTITY WANDB_USERNAME WANDB_RUN_ID WANDB_SWEEP_ID
export LIBERO_CONFIG_PATH="$eval_dir/libero_config"
export MUJOCO_GL=egl
cd "$FLEXPI_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv
srun --ntasks=1 --kill-on-bad-exit=1 python scripts/flowmap_libero_eval.py \
  "$eval_dir" "$phase" --case "${SLURM_ARRAY_TASK_ID:-0}"

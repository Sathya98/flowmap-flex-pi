#!/bin/bash -l
#SBATCH --job-name=fm-libero-pfmm-smoke
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --chdir=/gpfs/scratch1/shared/faster-wams/flowmap-flex-pi
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out

set -euo pipefail
source /gpfs/scratch1/shared/faster-wams/env_flexpi.sh
cd "$FLEXPI_ROOT"
export PYTHONPATH="$FLEXPI_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN

TEACHER_STEPS="${SMOKE_TEACHER_STEPS:-16}"
python scripts/flowmap_smoke_preflight.py "model.flow_map.teacher_steps=$TEACHER_STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[pilot] branch=$(git branch --show-current) revision=$(git rev-parse HEAD) job=$SLURM_JOB_ID"
echo "[pilot] 50 optimizer updates; full joint; full tuning; teacher steps=$TEACHER_STEPS"

srun --ntasks=1 --kill-on-bad-exit=1 python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 4 --num_machines 1 \
  --main_process_port "$((15000 + SLURM_JOB_ID % 30000))" \
  scripts/train.py --config-name=flowmap_libero_pfmm_smoke \
  "model.flow_map.teacher_steps=$TEACHER_STEPS"

RUN_DIR="runs/flowmap_pilots/libero_pfmm_${SLURM_JOB_ID}"
test -s "$RUN_DIR/checkpoints/weights/step_000050.pt"
test -s "$RUN_DIR/checkpoints/state/step_000050/trainer_state.json"
echo "[pilot] completed and saved: $RUN_DIR"
echo '[pilot] Next: inspect finite losses/gradient norms, then evaluate baseline and pilot at NFE 1/2/4.'

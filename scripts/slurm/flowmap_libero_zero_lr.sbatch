#!/bin/bash -l
#SBATCH --job-name=fm-libero-zero-lr
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

python scripts/flowmap_smoke_preflight.py --config-name=flowmap_libero_pfmm_zero_lr "$@"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[train] branch=$(git branch --show-current) revision=$(git rev-parse HEAD) job=$SLURM_JOB_ID"
echo '[pilot] Zero-LR master-weight regression: two updates, pretrained tensors must remain identical.'
# Budget based on about 10 hours of steady training at teacher_steps=16, plus overhead.
# Resume a timed-out run from checkpoints/state/step_* with resume=<path>.
srun --ntasks=1 --kill-on-bad-exit=1 python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 4 --num_machines 1 \
  --main_process_port "$((15000 + SLURM_JOB_ID % 30000))" \
  scripts/train.py --config-name=flowmap_libero_pfmm_zero_lr "$@"

python scripts/flowmap_check_pilot.py "runs/flowmap_pilots/libero_pfmm_zero_lr_${SLURM_JOB_ID}" --zero-lr

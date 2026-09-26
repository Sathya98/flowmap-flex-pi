#!/bin/bash -l
#SBATCH --job-name=fm-libero-pilot-diagnostics
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --chdir=/gpfs/scratch1/shared/faster-wams/flowmap-flex-pi
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out
set -euo pipefail
source /gpfs/scratch1/shared/faster-wams/env_flexpi.sh
cd "$FLEXPI_ROOT"
export PYTHONPATH="$FLEXPI_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/flowmap_smoke_preflight.py model.flow_map.teacher_steps=16
if (( $# == 0 )); then
  echo 'Pass --student label=/path/to/weights.pt for each pilot checkpoint.' >&2
  exit 2
fi
srun --ntasks=1 --kill-on-bad-exit=1 python -u scripts/flowmap_diagnostics.py \
  --output-dir "runs/flowmap_diagnostics/${SLURM_JOB_ID}" \
  "$@"

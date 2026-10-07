#!/bin/bash -l
#SBATCH --job-name=fm-libero-diagnostics
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out
set -euo pipefail
# Runs from the submit directory: sbatch from the repo root; the env scripts sit next to it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d scripts/slurm ]] || { echo 'Submit from the flowmap-flex-pi root' >&2; exit 2; }
source ../env_flexpi.sh
cd "$FLEXPI_ROOT"
export PYTHONPATH="$FLEXPI_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/flowmap_smoke_preflight.py model.flow_map.teacher_steps=16
srun --ntasks=1 --kill-on-bad-exit=1 python -u scripts/flowmap_diagnostics.py \
  --output-dir "runs/flowmap_diagnostics/${SLURM_JOB_ID}" \
  --student smoke_t8=runs/flowmap_pilots/libero_pfmm_26854898/checkpoints/weights/step_000050.pt \
  --student smoke_t16=runs/flowmap_pilots/libero_pfmm_26854907/checkpoints/weights/step_000050.pt \
  --student smoke_t32=runs/flowmap_pilots/libero_pfmm_26854915/checkpoints/weights/step_000050.pt \
  "$@"

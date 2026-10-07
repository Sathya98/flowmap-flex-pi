#!/bin/bash
#SBATCH --job-name=validate-tvm
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out
# GPU validation of the TVM fused attention-JVP kernels (jvp_kernel_analysis/validate_tvm.py).
#   sbatch scripts/slurm/validate_tvm.sbatch [extra args, e.g. --skip grad]
set -euo pipefail
# Runs from the submit directory: sbatch from the repo root; the env scripts sit next to it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d scripts/slurm ]] || { echo 'Submit from the flowmap-flex-pi root' >&2; exit 2; }
source ../env_flexpi.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
out="runs/diagnostics/tvm_validation_${SLURM_JOB_ID}"
echo "validate_tvm: out=$out extra=$*"
srun --ntasks=1 python jvp_kernel_analysis/validate_tvm.py --out "$out" "$@"

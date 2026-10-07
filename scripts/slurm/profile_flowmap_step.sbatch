#!/bin/bash
#SBATCH --job-name=profile-flowmap-step
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
# One-GPU profile of a flow-map training microstep (scripts/profile_flowmap_step.py).
# H100, like the timed LMD/PFMM runs in .claude/context/07-efficiency-notes.md.
#
#   sbatch scripts/slurm/profile_flowmap_step.sbatch [CONFIG_NAME] [CACHE_DIR] [extra args...]
set -euo pipefail
config_name="${1:-flowmap_libero_lmd_full}"
cache="${2:-data/latent_cache/libero_fulljoint_v2}"
shift $(( $# > 2 ? 2 : $# ))
# Runs from the submit directory: sbatch from the repo root; the env scripts sit next to it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d scripts/slurm ]] || { echo 'Submit from the flowmap-flex-pi root' >&2; exit 2; }
source ../env_flexpi.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
out="runs/diagnostics/profile_flowmap_step_${SLURM_JOB_ID}"
echo "profile: config=$config_name cache=$cache out=$out extra=$*"
srun --ntasks=1 python scripts/profile_flowmap_step.py \
  --config-name "$config_name" --cache "$cache" --out "$out" "$@"

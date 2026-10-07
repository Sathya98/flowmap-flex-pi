#!/bin/bash
#SBATCH --job-name=cache-latents
#SBATCH --account=gus25006
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=460G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out
# Build (or continue) a latent cache: init on one GPU if the cache is new,
# encode on every allocated GPU (resumable: resubmit to continue), then verify.
# See docs/LATENT_CACHE.md.
#
#   sbatch scripts/slurm/cache_latents.sbatch OUT_DIR CONFIG_NAME [STRIDE] [LIMIT]
#   STRIDE: cache every k-th window start of each episode (default 1; init only)
#   LIMIT:  encode at most this many windows per GPU (smoke; default 0 = all)
#   smoke:  sbatch --ntasks=1 --cpus-per-task=18 --mem=120G --time=01:00:00 ... OUT CFG 8 64
# Arguments, not environment variables: the site sbatch wrapper does not pass
# the caller's environment through.
set -euo pipefail
out="${1:?Pass the cache directory}"
config_name="${2:?Pass the Hydra config name}"
stride="${3:-1}"
limit="${4:-0}"
echo "cache_latents: out=$out config=$config_name stride=$stride limit=$limit tasks=$SLURM_NTASKS"
# Runs from the submit directory: sbatch from the repo root; the env scripts sit next to it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d scripts/slurm ]] || { echo 'Submit from the flowmap-flex-pi root' >&2; exit 2; }
source ../env_flexpi.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -e "$out/manifest.json" ]]; then
  srun --ntasks=1 --kill-on-bad-exit=1 python scripts/cache_latents.py init \
    --out "$out" --config-name "$config_name" --window-stride "$stride"
fi
srun --ntasks="$SLURM_NTASKS" --kill-on-bad-exit=1 python scripts/cache_latents.py encode \
  --out "$out" --num-workers $(( SLURM_CPUS_PER_TASK - 2 )) --batch-size 4 --limit "$limit"
srun --ntasks=1 python scripts/cache_latents.py verify --out "$out" --samples 64

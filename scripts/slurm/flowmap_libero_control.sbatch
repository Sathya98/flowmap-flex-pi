#!/bin/bash -l
#SBATCH --job-name=fm-libero-control
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --chdir=/gpfs/scratch1/shared/faster-wams/flowmap-flex-pi
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out

set -euo pipefail
pilot_config="${1:?Pass a control config name, followed by optional Hydra overrides}"
shift
case "$pilot_config" in
  flowmap_libero_pfmm_endpoint_pilot|flowmap_libero_pfmm_velocity_control|flowmap_libero_lmd_fullgrad_pilot|flowmap_libero_lmd_detached_control) ;;
  *) echo "Unsupported control config: $pilot_config" >&2; exit 2 ;;
esac
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

python scripts/flowmap_smoke_preflight.py --config-name="$pilot_config" "$@"
pilot_run_dir="$(python - "$pilot_config" "$@" <<'PY'
import sys
from pathlib import Path
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = compose(config_name=sys.argv[1], overrides=sys.argv[2:])
print(Path(cfg.output_dir).resolve())
PY
)"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Full LMD teacher-input backward and dual EMA overhead need GPU profiling.
srun --ntasks=1 --kill-on-bad-exit=1 python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 4 --num_machines 1 \
  --main_process_port "$((15000 + SLURM_JOB_ID % 30000))" \
  scripts/train.py --config-name="$pilot_config" "$@"
python scripts/flowmap_check_pilot.py "$pilot_run_dir"

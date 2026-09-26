#!/bin/bash -l
#SBATCH --job-name=paired-residual
#SBATCH --account=gus25006
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --chdir=/gpfs/scratch1/shared/faster-wams/flowmap-flex-pi
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out
set -euo pipefail
source /gpfs/scratch1/shared/faster-wams/env_flexpi_libero.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
output="eval_results/paired_residual_${SLURM_JOB_ID}"
mkdir -p "$output"
cp scripts/slurm/paired_residual_libero.sbatch "$output/launch.sbatch"
srun --ntasks=1 --kill-on-bad-exit=1 python experiments/libero/eval_libero_single.py \
  ckpt=checkpoints/released/flexpi-libero/checkpoints/weights/step_010860.pt \
  eval_config_source=saved seed=42 mixed_precision=bf16 gpu_id=0 \
  EVALUATION.dataset_stats_path=checkpoints/released/flexpi-libero/dataset_stats.json \
  "EVALUATION.output_dir=$output" EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 EVALUATION.num_trials=16 EVALUATION.replan_steps=32 \
  EVALUATION.num_steps_wait=30 EVALUATION.num_inference_steps=4 \
  EVALUATION.visualize_future_video=false EVALUATION.torch_compile=false \
  EVALUATION.dynamic_step_skip=false +EVALUATION.residual_sensitivity=true \
  +EVALUATION.residual_random_probes=4 +EVALUATION.residual_max_chunks=2 \
  +EVALUATION.paired_residual_experiment=true +EVALUATION.paired_holdout_pairs=8 \
  +EVALUATION.infer_present_video=true +EVALUATION.infer_present_dino=true \
  +EVALUATION.infer_present_pointmap=true +EVALUATION.infer_joint_video=true \
  +EVALUATION.infer_joint_dino=false +EVALUATION.infer_joint_pointmap=false \
  +EVALUATION.camera_intrinsics_path=data/libero_mujoco3.3.2_depth/libero_spatial_no_noops_lerobot/meta/camera_intrinsics.json \
  "$@"
python scripts/summarize_paired_residual.py "$output"

#!/usr/bin/env bash
# Preliminary paired LIBERO evaluation. Run inside a GPU allocation after step 100.
set -euo pipefail
run_dir="${1:?Pass the absolute Flow Map run directory}"
[[ "$run_dir" = /* ]] || { echo 'Run directory must be absolute' >&2; exit 2; }
cd "$(dirname "$0")/.."
step="${STEP:-100}"
[[ "$step" =~ ^[1-9][0-9]*$ ]] || exit 2
printf -v step_tag 'step_%06d.pt' "$step"
student="$run_dir/checkpoints/weights/$step_tag"
teacher="$(pwd)/checkpoints/released/flexpi-libero/checkpoints/weights/step_010860.pt"
student_stats="$run_dir/dataset_stats.json"
teacher_stats="$(pwd)/checkpoints/released/flexpi-libero/dataset_stats.json"
for path in "$student" "$teacher" "$student_stats" "$teacher_stats"; do
  [[ -s "$path" ]] || { echo "Missing evaluation input: $path" >&2; exit 2; }
done
# Ten paired trials/task is a pilot comparison, not the paper's 50-trial protocol.
export NUM_TRIALS="${NUM_TRIALS:-10}"
export GPUS="${GPUS:-0,1,2,3}"
export TASKS_PER_SUITE=10
export INFER_JOINT_VIDEO=true INFER_JOINT_DINO=true INFER_JOINT_POINTMAP=true
export INFER_PRESENT_VIDEO=true INFER_PRESENT_DINO=true INFER_PRESENT_POINTMAP=true
for variant in fm lmd; do
  if [[ "$variant" = fm ]]; then
    ckpt="$teacher"; stats="$teacher_stats"
  else
    ckpt="$student"; stats="$student_stats"
  fi
  for nfe in 1 2 4; do
    out="$run_dir/comparison/step_${step}/${variant}_nfe${nfe}_trials${NUM_TRIALS}"
    [[ ! -e "$out" ]] || { echo "Refusing to overwrite comparison: $out" >&2; exit 2; }
    CKPT="$ckpt" DATASET_STATS="$stats" OUTPUT_DIR="$out" \
      bash scripts/eval_flexpi_libero_4suite.sh \
        eval_config_source=saved EVALUATION.num_inference_steps="$nfe" \
        EVALUATION.dynamic_step_skip=false seed=42
  done
done

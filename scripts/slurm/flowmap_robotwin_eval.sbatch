#!/bin/bash -l
#SBATCH --job-name=fm-rt-eval
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --chdir=/gpfs/scratch1/shared/faster-wams/flowmap-flex-pi
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out
# RoboTwin simulator evaluation of one checkpoint over several NFEs (flow map or FM).
#
#   sbatch scripts/slurm/flowmap_robotwin_eval.sbatch NAME CKPT STATS "NFES" "TASKS" [EPISODES] [PHASES]
#   sbatch ... flowmap_robotwin_eval.sbatch SPECFILE - - "NFES" "TASKS" ...   # several checkpoints in turn
#
# SPECFILE: one "NAME|CKPT|STATS" per line. MAX_TASKS_PER_GPU (default 2) is the number of
# simulator workers per GPU; use 1 on 40 GB A100s (sbatch --partition=gpu_a100).
# Several jobs may share one spec file (e.g. on H100 and A100, whichever starts first): a
# checkpoint whose results are complete is skipped, one locked by another RUNNING job is
# skipped, and a lock left by a dead or timed-out job is taken over (the manager resumes).
#
# NAME labels the results: evaluate_results/robotwin/<ckpt tag>/<NAME>_nfe<K>/ (summary.json).
# Eager BF16 only: flow maps refuse torch.compile/TRT (backbone guard), and the FM
# baseline runs the same stack so the comparison is like for like. Everything else
# matches scripts/eval_flexpi_robotwin.sh (unseen instructions, 32-step replanning,
# full joint generation, seed 42 so every model sees the same episodes).
set -uo pipefail
nfes="${4:-1 2 3 4}"; tasks="${5:?comma-separated task list}"
episodes="${6:-10}"; phases="${7:-clean}"; per_gpu="${MAX_TASKS_PER_GPU:-2}"
if [[ -f "${1:?NAME or SPECFILE}" ]]; then specs=$(grep -v '^\s*#' "$1" | grep .); else specs="$1|${2:?CKPT}|${3:?DATASET_STATS}"; fi
while IFS='|' read -r name ckpt stats; do
  [[ -f "$ckpt" && -f "$stats" ]] || { echo "Missing $ckpt or $stats" >&2; exit 2; }
done <<< "$specs"
source /gpfs/scratch1/shared/faster-wams/env_flexpi_robotwin.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
echo "[fm-rt-eval] $(wc -l <<< "$specs") checkpoint(s)  nfes=[$nfes]  tasks=[$tasks]  episodes=$episodes  phases=[$phases]  per_gpu=$per_gpu  rev=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Watchdog: a simulator worker whose log has not changed for STALL_MIN minutes is stuck
# (seen: RoboTwin's expert check looping forever on one seed at 100% CPU). Kill it; the
# manager then stops and the next attempt resumes. The worker's stdout is a pipe to its
# parent (eval_robotwin_single.py), which writes the logs: watch the newest log it holds open.
STALL_MIN="${STALL_MIN:-30}"
( while sleep 120; do
    for pid in $(pgrep -u "$USER" -f "script/eval_policy.py"); do
      ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ') || continue
      newest=0
      for fd in /proc/"$ppid"/fd/*; do
        log=$(readlink "$fd" 2>/dev/null) || continue
        [[ "$log" == *.log && -f "$log" ]] || continue
        m=$(stat -c %Y "$log"); (( m > newest )) && newest=$m
      done
      if (( newest > 0 && $(date +%s) - newest > STALL_MIN * 60 )); then
        echo "[watchdog] $(date +%T) killing stalled worker $pid (logs unchanged for >${STALL_MIN} min)"
        kill "$pid"
      fi
    done
  done ) &
watchdog=$!
trap 'kill $watchdog 2>/dev/null' EXIT
LOCKS=evaluate_results/robotwin/.locks
mkdir -p "$LOCKS"
complete() {   # every (NFE, task) result of checkpoint $1 exists
  local nfe task
  for nfe in $nfes; do for task in ${tasks//,/ }; do for phase in ${phases//,/ }; do
    [[ $phase = random ]] && f=_result_random.txt || f=_result_clean.txt
    compgen -G "evaluate_results/robotwin/*/${1}_nfe${nfe}/${task}/$f" > /dev/null || return 1
  done; done; done
}
claim() {      # take the checkpoint's lock unless a running job holds it
  local lock="$LOCKS/$1" owner
  if ! mkdir "$lock" 2>/dev/null; then
    owner=$(cat "$lock/owner" 2>/dev/null)
    if [[ -n "$owner" && "$owner" != "$SLURM_JOB_ID" ]] && squeue -h -j "$owner" -t RUNNING 2>/dev/null | grep -q .; then
      return 1
    fi
    echo "[fm-rt-eval] taking over stale lock of $1 (owner ${owner:-?})"
  fi
  echo "$SLURM_JOB_ID" > "$lock/owner"
}
# Two passes: the second takes over checkpoints left behind by jobs that died or timed out
# while this one was busy (the manager resumes their finished tasks).
for pass in 1 2; do
while IFS='|' read -r name ckpt stats; do
if complete "$name"; then echo "##### $name: complete, skipping"; continue; fi
if ! claim "$name"; then echo "##### $name: running in job $(cat "$LOCKS/$name/owner"), skipping"; continue; fi
echo "##### $name  ckpt=$ckpt  $(date)"
for nfe in $nfes; do
  # The manager stops every worker when one fails; rerunning it resumes (finished tasks are
  # skipped), so a killed or crashed task costs one retry, not the whole NFE.
  for attempt in 1 2 3; do
    echo "=== NFE $nfe attempt $attempt  $(date)"
    if python experiments/robotwin/run_robotwin_manager.py \
      task=robotwin_unified_flex_3cam_384_1e-4 seed=42 \
      ckpt="$ckpt" EVALUATION.dataset_stats_path="$stats" \
      EVALUATION.num_inference_steps="$nfe" EVALUATION.eval_num_episodes="$episodes" \
      EVALUATION.instruction_type=unseen EVALUATION.replan_steps=32 \
      EVALUATION.offload_text_encoder=true EVALUATION.attn_backend=sdpa \
      EVALUATION.torch_compile=false EVALUATION.compile_encoders=false \
      +EVALUATION.use_per_cam=true \
      +EVALUATION.infer_joint_video=true +EVALUATION.infer_joint_dino=true +EVALUATION.infer_joint_pointmap=true \
      +EVALUATION.infer_present_video=true +EVALUATION.infer_present_dino=true +EVALUATION.infer_present_pointmap=true \
      MULTIRUN.num_gpus=4 MULTIRUN.max_tasks_per_gpu="$per_gpu" \
      "MULTIRUN.phases=[$phases]" "MULTIRUN.tasks=[$tasks]" \
      EVALUATION.output_dir="evaluate_results/robotwin/_/${name}_nfe${nfe}" < /dev/null; then
      break
    fi
    echo "[fm-rt-eval] NFE $nfe attempt $attempt failed; resuming"
  done
done
rm -rf "$LOCKS/$name"
done <<< "$specs"
done
echo "[fm-rt-eval] done $(date)"

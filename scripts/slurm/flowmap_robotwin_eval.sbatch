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
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out
# RoboTwin simulator evaluation of one checkpoint over several NFEs (flow map or FM).
#
#   sbatch scripts/slurm/flowmap_robotwin_eval.sbatch NAME CKPT STATS "NFES" "TASKS" [EPISODES] [PHASES]
#   sbatch ... flowmap_robotwin_eval.sbatch SPECFILE - - "NFES" "TASKS" ...   # several checkpoints in turn
#
# SPECFILE: one "NAME|CKPT|STATS" per line. MAX_TASKS_PER_GPU (default 2) is the number of
# simulator workers per GPU; use 1 on 40 GB A100s (sbatch --partition=gpu_a100).
# POLICY_SERVER=1: one policy server (the only model copy) per GPU and ENVS_PER_GPU (default 4)
# simulator workers per GPU sending it their replans, one episode per work unit.
# Pass these through sbatch: sbatch --export=ALL,POLICY_SERVER=1,ENVS_PER_GPU=12 ...
# GPU memory: the server holds 15.3 GB; a simulator 3.6 GB once it has set up a scene, 5.3-6 GB
# once it has run an expert check (cuRobo, built on first use). 40 GB A100: ENVS_PER_GPU=6 with a
# warm expert cache (37.3 GB peak measured), 3 when the tasks' seeds are checked for the first
# time (cold cache, 30.2 GB peak at 3). 94 GB H100: ~12 either way (then CPU-bound).
# ENVS_PER_GPU_COLD (default ENVS_PER_GPU) applies to each checkpoint's first NFE, which visits
# every episode's seed and so fills the expert cache; the later NFEs run warm.
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
server="${POLICY_SERVER:-0}"; envs_per_gpu="${ENVS_PER_GPU:-4}"; envs_cold="${ENVS_PER_GPU_COLD:-$envs_per_gpu}"
if [[ -f "${1:?NAME or SPECFILE}" ]]; then specs=$(grep -v '^\s*#' "$1" | grep .); else specs="$1|${2:?CKPT}|${3:?DATASET_STATS}"; fi
while IFS='|' read -r name ckpt stats; do
  [[ -f "$ckpt" && -f "$stats" ]] || { echo "Missing $ckpt or $stats" >&2; exit 2; }
done <<< "$specs"
# Runs from the submit directory: sbatch from the repo root; the env scripts sit next to it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d scripts/slurm ]] || { echo 'Submit from the flowmap-flex-pi root' >&2; exit 2; }
source ../env_flexpi_robotwin.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
echo "[fm-rt-eval] $(wc -l <<< "$specs") checkpoint(s)  nfes=[$nfes]  tasks=[$tasks]  episodes=$episodes  phases=[$phases]  per_gpu=$per_gpu  policy_server=$server envs_per_gpu=$envs_per_gpu (first NFE: $envs_cold)  rev=$(git rev-parse --short HEAD)"
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
first_nfe=1
for nfe in $nfes; do
  envs_now=$envs_per_gpu; (( first_nfe )) && envs_now=$envs_cold; first_nfe=0
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
      MULTIRUN.policy_server="$([[ $server = 1 ]] && echo true || echo false)" MULTIRUN.envs_per_gpu="$envs_now" \
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

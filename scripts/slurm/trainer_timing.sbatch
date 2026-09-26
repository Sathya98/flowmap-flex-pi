#!/bin/bash
#SBATCH --job-name=trainer-timing
#SBATCH --account=gus25006
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --chdir=/gpfs/scratch1/shared/faster-wams/flowmap-flex-pi
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.out
# Where does a training microstep's wall time go? Runs the real trainer
# (scripts/train.py, FLEXPI_STEP_TIMING harness in trainer.py) on the same node:
#   single       1 GPU, no DeepSpeed                 (compute + trainer overhead)
#   zero2        4 GPUs, production ZeRO-2 config    (what the LMD runs paid)
#   zero2_overlap 4 GPUs, ZeRO-2 + overlap_comm + contiguous_gradients
#   ddp          4 GPUs, DDP; no gradient sync until the update boundary
#   zero1        4 GPUs, ZeRO-1 (optimizer sharded, full grads; reduce once per update)
#   zero2_profile 4 GPUs, production ZeRO-2 + torch.profiler over microstep 6 (rank 0)
#   nccl         no training: GPU topology + NCCL bandwidth (scripts/nccl_bandwidth.py)
#   lsd_zero2    4 GPUs, patched ZeRO-2, LSD self-distillation, one full 48-microstep
#                update (stratified diagonal/off-diagonal mixture, §6 of the notes)
#   zero2_ds0189 as zero2 but DeepSpeed 0.18.9 (upstream hook fix; our patch skips itself),
#                from the side install /gpfs/scratch1/shared/faster-wams/deepspeed_0.18.9
#   resume_ds0185 / resume_ds0189  resume the LMD-100 step-50 ZeRO-2 state (saved with
#                0.18.5) and run past one optimizer step: checkpoint compatibility
#   zero2_tvm    as zero2 with flow_map.jvp_attention=tvm (fused attention JVP)
#   zero2_tvm_mb2  as zero2_tvm at microbatch 2 (accumulation 5): ZeRO-2 memory at mb2
#   lsd_zero2_tvm  as lsd_zero2 with the fused attention JVP
#   prof_lmd_tvm / prof_lmd_tvm_mb2 / prof_lsd_tvm_mb2  three updates each, TVM:
#                torch.profiler over one microstep (not LSD: its branch is random) and
#                over the last microstep + post-update work of update 2, with host-sync
#                call sites; synced post-update phase timings every update
#                (<variant>/{step,update}_profile_summary.json, "phases" lines).
#   ema_{sync,bg}_{lsd,lmd}  EMA fold synchronous vs background thread (FLEXPI_EMA_BACKGROUND),
#                TVM mb2 at the production accumulation 24, four updates, no profiler.
# single/ddp stop before their first optimizer step (full AdamW states would not
# fit next to student + teacher); zero2 variants include two optimizer steps.
# Batches come from the LIBERO latent cache. No checkpoints are written.
#
#   sbatch scripts/slurm/trainer_timing.sbatch [CONFIG_NAME] [VARIANTS...]
set -uo pipefail
config_name="${1:-flowmap_libero_lmd_full}"
shift || true
variants=("$@")
[[ ${#variants[@]} -gt 0 ]] || variants=(single zero2 zero2_overlap ddp)
source /gpfs/scratch1/shared/faster-wams/env_flexpi.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
unset WANDB_API_KEY WANDB_ENTITY WANDB_USERNAME WANDB_RUN_ID WANDB_SWEEP_ID
export WANDB_MODE=disabled
export FLEXPI_STEP_TIMING=1
root="runs/diagnostics/trainer_timing_${SLURM_JOB_ID}"
mkdir -p "$root"
echo "trainer timing: config=$config_name variants=${variants[*]} out=$root"

run() {   # name accelerate_config num_processes accumulation max_microsteps [profile_microstep] [overrides]
  local name=$1 acc=$2 procs=$3 accum=$4 micro=$5 prof=${6:-0} extra=${7:-} dir="$root/$1"
  mkdir -p "$dir"
  echo "=== $name: $procs GPU(s), $acc, accumulation $accum, $micro microsteps"
  local devices=0,1,2,3; [[ $procs = 1 ]] && devices=0
  # DS_PREFIX (optional): a directory put first on PYTHONPATH, e.g. another DeepSpeed.
  PYTHONPATH="${DS_PREFIX:+$DS_PREFIX:}$PYTHONPATH" \
  CUDA_VISIBLE_DEVICES=$devices FLEXPI_STEP_TIMING_MAX_MICROSTEPS=$micro FLEXPI_STEP_PROFILE_MICRO=$prof \
  srun --ntasks=1 --kill-on-bad-exit=1 python -m accelerate.commands.launch \
    --config_file "scripts/accelerate_configs/$acc.yaml" \
    --num_processes "$procs" --num_machines 1 \
    --main_process_port "$((15000 + (SLURM_JOB_ID + ${#name}) % 30000))" \
    scripts/train.py --config-name="$config_name" \
    output_dir="$dir" wandb.enabled=false eval_every=0 eval_at_start=false eval_on_pause=false \
    save_every=0 gradient_accumulation_steps="$accum" \
    +data.latent_cache_dir=data/latent_cache/libero_fulljoint_v2 $extra \
    2>&1 | tee "$dir/train.log" | grep -E "STEPTIMING|Traceback|Error|error:|OOM"
  echo "=== $name exit ${PIPESTATUS[0]}"
}

DS0189=/gpfs/scratch1/shared/faster-wams/deepspeed_0.18.9
RESUME=runs/flowmap_fulljoint/libero_lmd_100_s42_20260918/checkpoints/state/step_000050
for v in "${variants[@]}"; do
  case $v in
    single)        run single accelerate_single_gpu 1 48 20 ;;
    zero2)         run zero2 accelerate_zero2_ds 4 10 25 ;;
    zero2_overlap) run zero2_overlap accelerate_zero2_overlap_ds 4 10 25 ;;
    ddp)           run ddp accelerate_ddp 4 48 20 ;;
    zero1)         run zero1 accelerate_zero1_ds 4 10 25 ;;
    zero2_profile) run zero2_profile accelerate_zero2_ds 4 10 8 6 ;;
    lsd_zero2)     run lsd_zero2 accelerate_zero2_ds 4 48 49 0 \
                     "model.flow_map.objective=lsd model.flow_map.lmd_teacher_gradient=detached" ;;
    zero2_ds0189)  DS_PREFIX=$DS0189 run zero2_ds0189 accelerate_zero2_ds 4 10 25 ;;
    zero2_tvm)     run zero2_tvm accelerate_zero2_ds 4 10 25 0 "model.flow_map.jvp_attention=tvm" ;;
    zero2_tvm_mb2) run zero2_tvm_mb2 accelerate_zero2_ds 4 5 13 0 "model.flow_map.jvp_attention=tvm batch_size=2" ;;
    zero2_tvm_mb2_share) run zero2_tvm_mb2_share accelerate_zero2_ds 4 5 13 0 \
                     "model.flow_map.jvp_attention=tvm batch_size=2 model.flex_joint.share_within_microbatch=true" ;;
    lsd_zero2_tvm_mb2) run lsd_zero2_tvm_mb2 accelerate_zero2_ds 4 24 25 0 \
                     "model.flow_map.objective=lsd model.flow_map.lmd_teacher_gradient=detached model.flow_map.jvp_attention=tvm batch_size=2 model.flex_joint.share_within_microbatch=true" ;;
    lsd_zero2_tvm) run lsd_zero2_tvm accelerate_zero2_ds 4 48 49 0 \
                     "model.flow_map.objective=lsd model.flow_map.lmd_teacher_gradient=detached model.flow_map.jvp_attention=tvm" ;;
    prof_lmd_tvm) ( export FLEXPI_STEP_PROFILE_UPDATE=2 FLEXPI_STEP_SYNC_DEBUG=1
                    run prof_lmd_tvm accelerate_zero2_ds 4 10 31 6 "model.flow_map.jvp_attention=tvm" ) ;;
    prof_lmd_tvm_mb2) ( export FLEXPI_STEP_PROFILE_UPDATE=2 FLEXPI_STEP_SYNC_DEBUG=1
                    run prof_lmd_tvm_mb2 accelerate_zero2_ds 4 5 16 7 "model.flow_map.jvp_attention=tvm batch_size=2" ) ;;
    prof_lsd_tvm_mb2) ( export FLEXPI_STEP_PROFILE_UPDATE=2 FLEXPI_STEP_SYNC_DEBUG=1
                    run prof_lsd_tvm_mb2 accelerate_zero2_ds 4 24 73 0 \
                      "model.flow_map.objective=lsd model.flow_map.lmd_teacher_gradient=detached model.flow_map.jvp_attention=tvm batch_size=2" ) ;;
    ema_sync_lsd) ( export FLEXPI_EMA_BACKGROUND=0
                    run ema_sync_lsd accelerate_zero2_ds 4 24 97 0 "model.flow_map.objective=lsd model.flow_map.lmd_teacher_gradient=detached model.flow_map.jvp_attention=tvm batch_size=2" ) ;;
    ema_bg_lsd) ( export FLEXPI_EMA_BACKGROUND=1
                    run ema_bg_lsd accelerate_zero2_ds 4 24 97 0 "model.flow_map.objective=lsd model.flow_map.lmd_teacher_gradient=detached model.flow_map.jvp_attention=tvm batch_size=2" ) ;;
    ema_sync_lmd) ( export FLEXPI_EMA_BACKGROUND=0
                    run ema_sync_lmd accelerate_zero2_ds 4 24 97 0 "model.flow_map.jvp_attention=tvm batch_size=2" ) ;;
    ema_bg_lmd) ( export FLEXPI_EMA_BACKGROUND=1
                    run ema_bg_lmd accelerate_zero2_ds 4 24 97 0 "model.flow_map.jvp_attention=tvm batch_size=2" ) ;;
    # resume and pretrained_ckpt are mutually exclusive in the trainer; this config
    # names its teacher checkpoint explicitly, so clearing pretrained_ckpt is safe.
    resume_ds0185) run resume_ds0185 accelerate_zero2_ds 4 10 12 0 "resume=$RESUME pretrained_ckpt=null" ;;
    resume_ds0189) DS_PREFIX=$DS0189 run resume_ds0189 accelerate_zero2_ds 4 10 12 0 "resume=$RESUME pretrained_ckpt=null" ;;
    nccl)
      mkdir -p "$root/nccl"
      { nvidia-smi topo -m; nvidia-smi nvlink --status -i 0; nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv; } > "$root/nccl/topology.txt" 2>&1
      cat "$root/nccl/topology.txt"
      NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH srun --ntasks=1 python -m torch.distributed.run --standalone --nproc_per_node=4 \
        scripts/nccl_bandwidth.py > "$root/nccl/bandwidth.log" 2>&1
      grep -vE "NCCL INFO" "$root/nccl/bandwidth.log"
      grep -E "via (P2P|SHM|NET)|NVLS|Channel 00/|Ring 00" "$root/nccl/bandwidth.log" | sort -u | head -20
      ;;
    *) echo "unknown variant $v" ;;
  esac
done

python - "$root" <<'PY'
import re, statistics, sys
from pathlib import Path
root = Path(sys.argv[1]); rows = []
pat = re.compile(r"\[STEPTIMING\] rank=(\d+) micro=(\d+) step=\d+ wait_ms=(\d+) fwd_ms=(\d+) bwd_ms=(\d+) post_ms=(\d+) total_ms=(\d+)")
for log in sorted(root.glob("*/train.log")):
    per = {}
    for m in pat.finditer(log.read_text()):
        rank, micro, *ms = map(int, m.groups())
        per.setdefault(micro, []).append(ms)
    # Ranks run in lockstep: a microstep costs its slowest rank. Skip 3 warmup microsteps.
    steady = [[max(r[i] for r in ranks) for i in range(5)] for micro, ranks in sorted(per.items()) if micro > 3]
    if not steady:
        rows.append(f"| {log.parent.name} | no timing lines (see train.log) |||||")
        continue
    med = [statistics.median(c[i] for c in steady) / 1e3 for i in range(5)]
    worst_post = max(c[3] for c in steady) / 1e3
    # Mean of fwd+bwd (no post) exposes mixtures of cheap and expensive microsteps
    # that a median hides; "slow" = fwd+bwd above 1.5x the fastest microstep.
    work = [(c[1] + c[2]) / 1e3 for c in steady]
    slow = sum(w > 1.5 * min(work) for w in work)
    rows.append(f"| {log.parent.name} | {len(steady)} | " + " | ".join(f"{x:.2f}" for x in med)
                + f" | {worst_post:.2f} | {statistics.mean(work):.2f} | {slow}/{len(work)} |")
table = ["| variant | microsteps | wait s | fwd s | bwd s | post s | total s | max post s | mean fwd+bwd s | slow |",
         "|---|---|---|---|---|---|---|---|---|---|"] + rows
import json
updates = ["", "| variant | DeepSpeed | patch | update | loss | grad_norm | peak alloc / reserved GiB |", "|---|---|---|---|---|---|---|"]
for log in sorted(root.glob("*/train.log")):
    text = log.read_text()
    ds = re.search(r"DeepSpeed ([0-9.]+)[:\s]", text) or re.search(r"Patched DeepSpeed ([0-9.]+)", text)
    patched = "yes" if "Patched DeepSpeed" in text else "no"
    metrics = log.parent / "train_update_metrics.jsonl"
    for line in (metrics.read_text().splitlines() if metrics.exists() else []):
        r = json.loads(line)
        updates.append(f"| {log.parent.name} | {ds.group(1) if ds else '?'} | {patched} | {r['step']} | "
                       f"{r['metrics']['loss']['mean']:.6f} | {r['grad_norm']:.2f} | "
                       f"{r.get('peak_allocated_gib', 0):.1f} / {r.get('peak_reserved_gib', 0):.1f} |")
(root / "summary.md").write_text("Median per microstep (slowest rank), after 3 warmup microsteps.\n"
                                 "max post = worst microstep, i.e. one with an optimizer step.\n\n"
                                 + "\n".join(table + updates) + "\n")
print("\n".join(table + updates))
PY

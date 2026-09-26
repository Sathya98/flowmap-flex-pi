# 07 — Flow-map training efficiency notes (compute, memory, cost per update)

> **Purpose.** Record what we know (measured) and what we estimate about the training cost
> of the flow-map objectives, so the profiling work can pick up from here. Written
> 2026-09-23 from the LIBERO slurm logs and run dirs. **Not profiled yet** — anything marked
> *estimate* is a cost-model inference, not a measurement. Line numbers are a snapshot.
>
> **Paths (2026-09-26):** the generic modules named below moved to the shared package
> `flowmap_core/` (`.claude/context/08-flowmap-core-plan.md`). `helpers/attention.py`,
> `helpers/jvp_attention/`, `helpers/checkpoint.py`, `helpers/normalization.py`,
> `helpers/flowmap_self.py`, `utils/flowmap_ema.py`, `utils/step_profile.py` and
> `utils/deepspeed_compat.py` are now aliases of `flowmap_core.{attention, jvp_attention,
> checkpoint, normalization, flowmap_self, ema, step_profile, deepspeed_compat}`. Timings are
> unchanged (S6 in the 08 progress log).

---

## 1. Measured numbers (4×H100, effective batch 192 = 1 × 48 accum × 4 GPUs)

| Run | Job | s/update | s/example | Peak alloc / reserved |
|---|---|---|---|---|
| PFMM smoke K=8 (1 ex/GPU/update) | 26854898 | 14 (median, 49 intervals) | 14 | — |
| PFMM smoke K=16 | 26854907 | 15 | 15 | — |
| PFMM smoke K=32 | 26854915 | 17 | 17 | — |
| PFMM K=16, batch 192 | 26858864 | 722 (**one** interval only) | ~15.0 | 52.3 / 65.5 GiB |
| LMD full-grad, batch 192 | 26887233 | 785 median (90 intervals) ≈ 13.1 min | ~16.4 | 68.6 / 74.1 GiB |

Timings come from the `step=N/M` log timestamps; peak memory from
`train_update_metrics.jsonl`. `distill_ema=true` in the LMD run, so the CPU EMA update is
inside its 785 s.

## 2. What the numbers imply

- **One no-grad teacher forward ≈ 0.125 s** (PFMM slope: +3 s for K 8→32).
- **Per-example floor ≈ 13.0 s** (PFMM intercept at K=0 ≈ plain-FM cost). Both objectives
  pay it. It contains data loading, the frozen encoders (Wan VAE for RGB + pointmap, DINOv3),
  the student's normal forward/backward and ZeRO-2 comms. **Its breakdown is unknown.**
- **LMD's extra cost over the floor is only ≈ 3.35 s/example.** Its 13 min/update comes
  mostly from 48 serial batch-1 microbatches × the floor, not from the JVP.
- ⇒ **The floor is the biggest lever for every objective.** Fixing it (more workers, cached
  latents, larger microbatches) speeds up FM/PFMM/LMD/LSD alike.

## 3. Why LMD needs forward AD and what costs memory

- LMD needs ∂v_θ/∂t: how a whole output tensor changes with respect to one scalar input.
  That is a JVP, so forward mode computes it in one pass (`tuple_jvp`, `helpers/flowmap.py:154`).
  Reverse mode would need one pass per output element.
- ∂_t v sits **inside the loss**, so training is reverse-over-forward (second order).
  PFMM never differentiates with respect to its inputs, so it needs only ordinary backprop.
- Memory drivers:
  1. **Dual activations.** Primal and tangent are both saved, because backward runs through
     the tangent computation.
  2. **Explicit FP32 attention.** `helpers/attention.py` replaces fused SDPA (the SDPA JVP
     breaks under backward in PyTorch 2.7), so it materialises full L×L scores/weights and
     their tangents. This is quadratic in sequence length. **This is where job 26883597
     OOMed** (`attention.py:18`, 93.2/93.3 GiB).
  3. **Checkpointing had no JVP rule.** Fixed in 26884210 by `helpers/checkpoint.py`,
     which passes primal/tangent pairs through non-reentrant checkpoint regions.
  4. **Teacher in the graph.** `lmd_teacher_gradient=full` saves the teacher's activations
     for a backward to its input. The teacher is also a full resident copy of the
     denoiser (`trainer.py:227`, `clone_teacher`).
- Job 26868647 was a separate LayerNorm BF16/FP32 tangent dtype bug
  (fixed in `helpers/normalization.py`), not memory.

### Attention path in detail (checked 2026-09-25)

- Normal training forwards call `F.scaled_dot_product_attention` with a dense boolean
  mask (`mot.py` `_mixed_attention` → `wan_video_dit.flash_attention`). That runs the fused
  memory-efficient kernel, which never forms L×L. Training does **not** use FlexAttention:
  that is only an inference backend (`flexpi.py:3960`, `_prepare_joint_flex_block_mask`).
- No fused SDPA backend (flash, memory-efficient, cuDNN) has a forward-AD rule. PyTorch's
  own fallback, the math backend, breaks under backward in 2.7 (softmax's JVP is in-place).
  So whenever a forward-AD level is active, `helpers/attention.py` computes attention
  explicitly: out of place, **FP32 with no TF32** (nothing in the repo enables it), and it
  materialises scores, weights and their tangents.
- Size: L ≈ 1638 visual + 32 action ≈ 1670 tokens (LIBERO; RoboTwin legacy ≈ 1634), 24 heads.
  One L×L FP32 tensor is ~270 MB per sample; roughly 6–8 are saved per joint layer.
  - Without checkpointing: ~30 GB for the 16 joint layers. That was the OOM.
  - With the dual-aware checkpoint: only one layer is live during recompute.
  - The cost exists only because the tangent is differentiated (`detach_derivatives: false`).
    Under no_grad it would be transient.
- FLOP estimate (*unmeasured*): attention's JVP is ~1 TFLOP per forward in FP32, against
  ~20 TFLOP bf16 for the rest of the 6B model. It is probably not the main term in LMD's
  +3.35 s/example; §8 measures it.
- It is not a fundamental limit. Attention's JVP can be computed flash-style, with an online
  softmax (dO = P·dV + (P ⊙ (dS − rowsum(P⊙dS)))·V, where dS = (dQ·Kᵀ + Q·dKᵀ)/√d). sCM
  (Lu & Song 2024) did this in Triton, but only forward, because its tangent is under
  stop-gradient. Keeping our gradient through the tangent would also need a backward kernel.
- Options, once §8 has numbers:
  1. Detach the derivative (semigradient, sCM/MeanFlow style). This changes the gradient
     away from Boffi 2025 Eq. 94, so it is a design decision.
  2. A fused JVP-attention kernel (forward only if option 1; forward and backward otherwise).
  3. Run the explicit path in bf16 or TF32.

## 4. Objective comparison (JVPs and gradients)

| | LMD full-grad (ran) | LSD | ESD | PFMM |
|---|---|---|---|---|
| Target velocity | frozen teacher | student diagonal, no_grad | student diagonal, no_grad | teacher K-step Euler mean, no_grad |
| Queried at | X(s,t) | X(s,t) | x_s | teacher's own path |
| Differentiable JVP | ∂_t v_θ | ∂_t v_θ | ∂_s v_θ | none |
| Extra no-grad JVP | — | — | spatial ∇_x v_θ·b (all 4 streams) | — |
| Grad through target | **yes** (teacher input) | no | no | no |
| Examples paying for it | 100% | 25% (75% plain FM) | 25% | 100% |
| Denoisers on GPU | student + teacher | student | student | student + teacher |

LSD has LMD-*detached*'s gradient structure, with the teacher swapped for the student's own
diagonal. ESD adds one transient no-grad dual forward (time, but little peak memory).

## 5. Estimated per-update time (batch 192) — *estimates except where marked*

Off-diagonal extras (estimate): LSD ≈ 3.1–3.2 s, ESD ≈ 3.4–3.6 s over the 13.0 s floor.

| Objective | Per update |
|---|---|
| Plain FM | ~10.4 min |
| PFMM K=16 | 12.0 min (**measured**) |
| LMD full-grad | 13.1 min (**measured**) |
| LSD / ESD, current random mask | ~12.2 / ~12.3 min |
| LSD / ESD, stratified mask | ~11.0 / ~11.1 min |

Hypothetical, if the non-model floor were removed (assumes the student's fwd+bwd is ≈0.5 s):
FM 0.5, PFMM-16 2.5, LMD 3.9, LSD 2.6 (random) / 1.3 (stratified), ESD 2.9 / 1.4 s per example.
Stratified LSD ≈ 3× cheaper than LMD. That only shows once the floor is fixed.

## 6. Mask-imbalance issue (self-distillation)

`update_diagonal_mask` (`helpers/flowmap_self.py`) shuffles all 192 slots randomly. ZeRO-2
reduces gradients on every microbatch backward, so the ranks run in lockstep and each
microstep is as slow as its slowest rank. A microstep is JVP-free only when all 4 ranks drew
diagonal examples: 0.75⁴ ≈ 32%, so **~68% of microsteps pay the JVP cost** rather than 25%.
Fix: stratify per microstep (12 of 48 microsteps off-diagonal on every rank).

**Implemented (2026-09-25), always on, no flag.** `update_diagonal_mask` now packs
off-diagonal examples into whole microsteps.
- Every rank and every example of a microstep take the same branch.
- At 75/25, 4 GPUs: 12 of 48 microsteps are off-diagonal at mb1, 6 of 24 at mb2. One
  microstep is mixed only when the count is not a multiple of `batch × world`.
- Unchanged: the 144/48 split, which microsteps (hence which data) go off-diagonal is
  reshuffled every step, and all ranks rebuild the plan from the seed (no collective).
- Tests: `tests/test_flowmap_self.py::test_diagonal_mask_is_stratified_by_microstep`.
- Expected (1-GPU §8 times): average microstep ≈ 0.75 × 0.5 + 0.25 × 2.5 = 1.0 s, against
  ≈ 1.9 s shuffled, i.e. ~2× per LSD/ESD update.
- **Measured: job 27183027** (4×H100, patched ZeRO-2, LSD, LIBERO cache, one full
  48-microstep update, `runs/diagnostics/trainer_timing_27183027/`).
  - Every microstep takes the same time on all 4 ranks, so there is no lockstep waiting.
  - Exactly **12 of 48** microsteps are off-diagonal (2.4–3.2 s); the other 36 are diagonal
    (~0.6 s).
  - Five diagonal microsteps take ~0.87 s (cause unknown, ~3% of the update). The summary's
    "16/46 slow" counts them because the 1.5×-min threshold is too tight.
  - **Mean microstep 1.21 s.** The shuffled counterfactual from the same measured times:
    P(all 4 ranks diagonal) ≈ 0.31, so 0.31 × 0.61 + 0.69 × 2.8 ≈ **2.1 s**. That is
    **~1.8× faster**.
  - LSD update ≈ 48 × 1.21 s + optimizer (first 28 s incl. state init) ≈ **1.1 min**
    (LMD full ≈ 2.5 min).
- Follow-up: the self-distillation loss still loops per example inside a microbatch
  (`flowmap_training.training_loss`), because the learned time weight is applied to a
  batch-reduced loss (`flow_map_loss_weight(s, t).squeeze(0)`). Now that a microbatch is
  single-branch, batching it needs per-example loss reduction, then one call per branch.

## 7. To revisit (TODO)

Done:
- [x] The ~13 s "floor" was a **DeepSpeed 0.18.5 O(N²) ZeRO hook bug**. It is patched (§9):
      microstep 16.0 → 3.0 s, LMD update ≈ 13.1 → ~2.5 min (estimate). Already-prepared
      runs execute a frozen `$run_dir/code` snapshot and do not have the patch.
- [x] Latent cache (`docs/LATENT_CACHE.md`):
  - LIBERO is `data/latent_cache/libero_fulljoint_v2` (complete, verified).
  - RoboTwin stride 8 is `data/latent_cache/robotwin_s8_v1` (75% done; resume job 27181221
    after a host-RAM OOM, see §10).
  - Train with `+data.latent_cache_dir=...`.
- [x] Fused attention-JVP kernel (TVM) validated outside training (§10).

Open, roughly in order:
- [x] TVM step time re-measured with the sync-free cache (job 27180820, §10): mb1 5–9% faster;
      LMD mb2 fits at 1.39 s per example (1.86×).
- [x] TVM kernel integrated behind `flow_map.jvp_attention: explicit|tvm` (default
      explicit; license CC BY-NC-SA accepted for now, 2026-09-26).
      - Code: `helpers/jvp_attention/` (vendored `tvm/` + Function + row groups).
      - `helpers/attention.py` routes forward-AD calls; unsupported calls (CPU, fp32,
        hd > 128, additive masks) fall back to explicit.
      - FlexPi sets the backend at build.
      - Tests: `tests/test_jvp_attention.py`.
      - 4-GPU ZeRO-2 check (LMD mb1/mb2, LSD): job 27198562.
- [ ] Flex-joint flow maps: configs `flowmap_libero_{lmd,lsd}_flex`, CPU tests
      `tests/test_flowmap_flex.py`. 4-GPU TVM timing: jobs 27198797 (LMD flex, tvm and
      explicit) and 27198798 (LSD flex, tvm).
- [ ] Microbatch 2 under real 4-GPU ZeRO-2 with TVM. The 1-GPU peak is 82.7 GiB without
      optimizer state; ZeRO-2 adds ~+12 GiB (67.3 → 79.9 at mb1), so ~90–95 GiB is expected.
      Too tight to rely on without a test.
- [ ] Wider dual-aware checkpointing (whole blocks, not just attention). LMD's +31.9 GiB is
      mostly tangent activations outside attention (semigradient: +9.7). This is the main
      lever for microbatch ≥ 2 headroom.
- [ ] Launch overhead: at mb1 the GPU is busy ~1.5 of ~2.6 s. Try larger microbatches, CUDA
      graphs or `torch.compile` on the per-block forward.
- [x] Stratified self-distillation mask, implemented and measured (§6): 12/48 slow
      microsteps, mean 1.21 s against ~2.1 s shuffled (~1.8×); LSD update ≈ 1.1 min.
- [x] Bumped DeepSpeed 0.18.5 → 0.18.9 (`pyproject.toml`, `fm_env`, 2026-09-26) after the
      resume check (job 27198328) passed. Our hook patch detects the upstream fix and skips
      itself (§9).
- [x] Shared flex-joint regime per microbatch, behind the flag
      `flex_joint.share_within_microbatch` (default off, §10).
- [x] Batched the self-distillation per-example loop (§10, end): one call per branch,
      per-example loss reduction. GPU timing: jobs 27199500 / 27199501.
- [ ] TVM is slower than explicit under flex regimes (§10, end). Profile it.
- [x] Post-update cost found and fixed (§11): the rank-0 CPU EMA (~9 s/update). Flat chunked
      background fold: post 0.2 s; LSD update ~38.5 → ~30 s, LMD ~81 → ~72 s (mb2, TVM).
- [ ] The first EMA fold after startup takes ~30 s (one-time, unexplained).
- [x] CUDA graphs measured (§13): 2.2× at mb1, 1.2–1.3× at mb2 (GPU-bound); trainer integration
      needs a capture-compatible gradient reduce (not ZeRO-2 hooks).
- [ ] Cut GPU time: investigation plan in §14 (profile graphed replay, attribute kernels,
      casts, GEMM efficiency, recompute share, then choose the fusion route).
- [ ] `cache_latents.py verify` on `libero_fulljoint_v2` now fails the 1e-3 VAE tolerance
      (input_latents 5.6e-3, first_frame 8.8e-3, pointmap 4.6e-3; dino 1.1e-2 vs 1.7e-3 then). It
      passed at exactly 0 right after encoding (job 27165239, 2026-09-21). Identical numbers at
      `8d00d25` and after the flowmap_core refactor (jobs 27211483, 27211278), so something between
      the encode and now changed the fresh VAE/DINO encode (or it is nondeterministic across
      runs). bf16-rounding scale; decide whether the cache is still "the same inputs" before
      comparing cached and uncached runs.
- [ ] Remove host syncs from the loss/metrics path (~50 per microstep; minor).
- [ ] ESD 4-GPU timing (LSD is job 27183027).
- [ ] Multi-interval PFMM batch-192 timing (only one interval exists).
- [ ] Semigradient (`detach_derivatives`) is 2× cheaper (§8) but changes the gradient
      (Boffi Eq. 94). It is a design decision, not an optimization.

## 8. Profiling benchmark (1 GPU)

`scripts/profile_flowmap_step.py`, launched by `scripts/slurm/profile_flowmap_step.sbatch`
(H100, 1 GPU, 2 h):

```bash
sbatch scripts/slurm/profile_flowmap_step.sbatch [CONFIG] [CACHE] [--modes a,b] [--skip inputs,...]
#   defaults: flowmap_libero_lmd_full, data/latent_cache/libero_fulljoint_v2
#   output:   runs/diagnostics/profile_flowmap_step_<jobid>/{results.json, profiler_<mode>.txt}
```

It reads any finished windows of a latent cache (a partial cache is fine). The student and
frozen teacher are built as `Wan22Trainer` builds them: strict checkpoint load, teacher
outside the module tree, `configure_trainable`. Each objective is a `dataclasses.replace` of
the config's `flow_map`, so one model load serves every mode. Stages:

1. **inputs:** median per example, single process. Raw `_get` (decode) and `build_inputs`
   (VAE + DINO), against a cache read and `build_inputs`.
2. **attention:** the longest joint attention call, captured from a real forward (q/k/v and
   the real mask). Timed variants:
   - fused SDPA, forward and forward + backward;
   - explicit FP32 without tangents, forward and forward + backward;
   - explicit FP32 JVP, forward and forward + backward;
   - explicit JVP in bf16 and in TF32, forward + backward, each with its tangent error
     against FP32.

   Each reports time and peak memory.
3. **step:** batch 1, forward (incl. JVP) and backward, peak and activation-peak memory, for
   `fm_base`, `lsd_diag`, `pfmm16`, `lmd_full`, `lmd_detached`, `lmd_semigrad`
   (detach_derivatives), `lsd_off` and `esd_off`.
4. **profile:** `torch.profiler` on one `--profile-mode` step (default `lmd_full`): top CUDA
   ops, GEMM share, and time inside explicit vs fused attention (forward ranges).

Excluded: the optimizer step, EMA, ZeRO communication and dataloader workers. So it gives
per-example compute, not batch-192 wall time.

### Results: job 27172448 (H100, 2026-09-25, `flowmap_libero_lmd_full`, partial LIBERO cache)

Output: `runs/diagnostics/profile_flowmap_step_27172448/`. Student and teacher take 24.0 GiB.

**Step, batch 1** (median of 6; fwd includes the JVP):

| Mode | fwd s | bwd s | step s | activation peak GiB |
|---|---|---|---|---|
| fm_base | 0.18 | 0.33 | **0.51** | 7.6 |
| lsd_diag | 0.19 | 0.33 | 0.52 | 7.6 |
| pfmm16 | 1.72 | 0.33 | 2.05 | 7.8 |
| lmd_full | 1.14 | 1.53 | **2.67** | 31.9 |
| lmd_detached | 1.13 | 1.43 | 2.56 | 31.9 |
| lmd_semigrad | 0.98 | 0.33 | **1.31** | 9.7 |
| lsd_off | 1.03 | 1.43 | 2.45 | 31.9 |
| esd_off | 1.78 | 1.43 | 3.21 | 32.0 |

**Attention call** (L = 1670, 24×128, mask density 0.76):

| Variant | Time | Memory |
|---|---|---|
| fused SDPA fwd+bwd | 1.5 ms | +0.1 GiB |
| explicit FP32 fwd+bwd, no tangent | 9.5 ms | |
| explicit FP32 JVP fwd+bwd | **47.6 ms** | +4.2 GiB |
| explicit TF32 JVP fwd+bwd | 40.7 ms | |
| explicit bf16 JVP fwd+bwd | 23.7 ms | |

Tangent error against FP32: TF32 9.5e-4, bf16 2.8e-2.

**Profiler, one lmd_full step:**
- Kernel time is 1.52 s against 2.67 s of wall time. The GPU sits idle about 40% of the step,
  because thousands of small elementwise kernels at batch 1 leave it launch- and CPU-bound.
- Explicit-attention forward ranges (incl. checkpoint recompute; their backward kernels are
  not counted) take 0.72 s. That is **~47% of GPU time**.
- Ignore the `gemm_share` figure: GEMM kernels are named `nvjet_*`, not `aten::mm`, so it
  undercounts.

**Inputs, single process:** raw decode 0.63 s + VAE/DINO 0.15 s per example. A cache read
took 1.39 s, but the build job was writing the same files from another node; re-measure once
it is done. `build_inputs` from the cache: 1 ms.

### What this changes

1. **The ~13 s floor is not model compute or data.** On one GPU, compute per example is
   0.5 s (FM), 2.0 s (PFMM-16) and 2.7 s (LMD). The 4-GPU ZeRO-2 runs measured 15.0 and
   16.4 s. Decode plus encode is 0.8 s, spread over workers. So ~13 s per microstep goes
   somewhere in the distributed training loop. Suspects: ZeRO-2 reducing gradients every
   microstep with `overlap_comm: false` (a 6B model); trainer or DeepSpeed per-microstep
   overhead. There is no CPU offload. §2's "floor = data + encoders" guess is wrong.
   Next: §9 times the real trainer for ~20 microsteps on 1 GPU (no DeepSpeed), then on 4 GPUs
   ZeRO-2, both reading the cache.
2. **Forward-AD attention is the largest compute item in LMD/LSD.** It is ~31× fused per
   call and ~50% of LMD GPU time. A fused JVP kernel could save ~1 s of the 2.7 s. TF32 is
   a free ~15% (error 1e-3). bf16 halves it, but its 2.8% tangent error is probably too
   high for a term inside the loss.
3. **Differentiating the tangent is what costs.** Semigradient LMD is 1.31 s and +9.7 GiB,
   against 2.67 s and +31.9 GiB for full LMD. Its backward equals plain FM. The +22 GiB comes
   from dual activations outside attention, which checkpointing does not wrap.
4. **At batch 1 the GPU is underused (~40% idle).** Larger microbatches or CUDA graphs /
   `torch.compile` would help. One GPU with student + teacher + grads peaks at 67 GiB for
   LMD, so microbatch 2 needs about +32 GiB and fits on a 94 GB H100 only without other
   overheads.
5. **Updated per-update compute bound** (192 examples / 4 GPUs, if the floor were removed):
   FM 0.4 min, PFMM-16 1.6 min, LMD full 2.1 min, LMD semigradient 1.1 min.
   LSD at the 75/25 split, stratified: 0.75·0.52 + 0.25·2.45 = 1.0 s/example, i.e. 0.8 min.

## 9. Trainer timing (real training loop, 1 vs 4 GPUs)

Goal: find where the ~13 s/microstep that is not model compute goes (§8 point 1).

**Harness.** `FLEXPI_STEP_TIMING=1` in `trainer.py` (`train()` loop). It is a no-op when
unset. Each phase boundary synchronizes the GPU, and the loop prints one `[STEPTIMING]` line
per microstep:

- `wait_ms`: `next(data_iter)`
- `fwd_ms`: `training_loss`, including the JVP
- `bwd_ms`: `accelerator.backward`, including any ZeRO gradient reduce
- `post_ms`: everything until the next microstep (optimizer step, EMA, logging)

`FLEXPI_STEP_TIMING_MAX_MICROSTEPS=N` exits after N microsteps without a final checkpoint.
It is for timing runs only.

**Launcher.** `scripts/slurm/trainer_timing.sbatch [CONFIG] [VARIANTS...]`. It runs on H100×4
and defaults to `flowmap_libero_lmd_full`. It runs the real `scripts/train.py` with
`+data.latent_cache_dir=data/latent_cache/libero_fulljoint_v2`, `eval_every=0`,
`save_every=0` and wandb off. The variants run back to back on one node:

| Variant | Setup | Microsteps | Isolates |
|---|---|---|---|
| `single` | 1 GPU, no DeepSpeed, accum 48 | 20, no optimizer step | loop overhead on top of compute |
| `zero2` | 4 GPUs, production `ds_zero2_config.json`, accum 10 | 25, 2 optimizer steps | what the timed LMD runs paid |
| `zero2_overlap` | 4 GPUs, `ds_zero2_overlap_config.json` (overlap_comm + contiguous_gradients) | 25 | whether overlapping the reduce helps |
| `ddp` | 4 GPUs, DDP, accum 48 | 20, no optimizer step | ZeRO-2's per-microstep reduce (DDP skips sync until the boundary) |

`single` and `ddp` stop before their first optimizer step, because full AdamW states do not
fit next to student + teacher on one GPU. Output: `runs/diagnostics/trainer_timing_<job>/`
holds `<variant>/train.log` and `summary.md`. The summary gives per-phase medians per
microstep, using the slowest rank (the ranks run in lockstep), after 3 warmup microsteps,
plus the worst `post`, i.e. a microstep with an optimizer step.

**How to read it:**
- Compare `single` against §8's `lmd_full` 2.67 s. A gap is trainer and loop overhead (data
  wait, metrics, hooks).
- Compare `zero2` against `single`. A gap in `bwd` is ZeRO-2 communication or bookkeeping.
- If `ddp` ≈ `single` while `zero2` is slow, the per-microstep reduce is the cost.
  `zero2_overlap` then shows whether overlapping fixes it.

### Results: job 27176766 (4×H100 gcn151, 2026-09-25, `flowmap_libero_lmd_full`, LIBERO cache)

Output: `runs/diagnostics/trainer_timing_27176766/summary.md`. Medians per microstep, slowest
rank, after 3 warmup microsteps:

| Variant | wait s | fwd s | bwd s | total s | optimizer microstep (post) |
|---|---|---|---|---|---|
| single (1 GPU, no DS) | 0.00 | 1.11 | **1.61** | 2.73 | — |
| ddp (4 GPUs, no sync between microsteps) | 0.00 | 1.39 | **1.63** | 3.00 | — |
| zero2 (production) | 0.00 | 1.39 | **14.93** | 16.04 | 29.1 s first, 7.1 s second |
| zero2_overlap | 0.00 | 1.38 | **15.36** | 16.52 | 32.0 s first, 7.6 s second |

**The ~13 s floor is ZeRO-2's per-microstep backward.**
- It costs ≈ 13.3 s extra per microstep, per rank, on top of a 1.6 s compute backward.
- DDP (which skips the gradient sync until the update) and 1 GPU show no floor. The trainer
  loop, the data (latent cache: ~1 ms wait) and the forward are fine.
- `overlap_comm` + `contiguous_gradients` do not help. So it is not simply exposed
  communication that overlap could hide.
- It is not JVP-specific either: PFMM (no forward AD) had the same floor (15.0 s/example
  against 2.05 s compute).
- 13.3 s for ~12 GB of bf16 gradients is ≈ 1 GB/s effective, far below NVLink/PCIe. Suspects,
  unverified:
  - NCCL on a slow transport on these nodes;
  - DeepSpeed stage-2 per-microstep reduce + fp32 partition accumulation for a 6B model, with
    small (2e8) buckets;
  - interaction between DeepSpeed's grad hooks and our double-backward / checkpoint graph.
- Per update (48 microsteps): 48 × 13.3 s ≈ 10.6 min of the 13.1 min LMD update is this
  overhead. Removing it gives ≈ 48 × 2.8 s + optimizer ≈ **2.3 min per update (~5.7×)**.

### Root cause and fix (jobs 27179569, 27180109)

- **Hardware and NCCL are fine.** 4×H100 all-to-all NVLink (`NV6`); ~300 GB/s NCCL bus
  bandwidth. Replaying ZeRO-2's per-microstep traffic (30 × 2e8 bf16 reduce-scatters) takes
  **32 ms** (`nccl/bandwidth.log`).
- **Profile of one ZeRO-2 microstep** (`zero2_profile/step_profile_*.txt`):
  - `AccumulateGrad` takes **12.3 s of self CPU** (16.6 s total) over 1670 calls.
  - There are ~2.8 M `aten::view`/`view_as` calls, i.e. 1670².
  - NCCL kernels take 58 ms; all GPU kernels take ~1.8 s.
- **Cause: a DeepSpeed 0.18.5 bug.** Every ZeRO-1/2 parameter gradient hook
  (`stage_1_and_2.py:1028`) calls `count_used_parameters_in_backward(all_params)`
  (`runtime/utils.py:1426`). That walks every trainable parameter through `view_as` → its
  autograd node. The cost is O(N_params²) CPU per backward, paid on every ZeRO-2 microstep.
  Upstream master already refreshes the count only when needed
  (`should_refresh_expected_hook_count`).
- **Fix:** `src/flexpi/utils/deepspeed_compat.py` caches the count per autograd graph task
  and refreshes it when the hooks that have fired reach it. That keeps DeepSpeed's reentrant
  "late-joining params" semantics.
  - Applied in `Wan22Trainer.__init__` whenever DeepSpeed is active (log line "Patched
    DeepSpeed ZeRO hook parameter count").
  - Opt out with `FLEXPI_DS_HOOK_COUNT_CACHE=0`.
  - Tests: `tests/test_deepspeed_compat.py`.
- **Verified** (patched production ZeRO-2, job 27180109): backward **1.79 s** (from 14.93),
  microstep **3.00 s** (from 16.04). Peak memory is unchanged (68.6 / 79.9 GiB at updates
  1 / 2). The step-1 loss is identical. Step-1 grad norm is 1246.8, within the run-to-run
  spread of the unpatched variants (1226.8–1243.5).
- **ZeRO-1** (no hooks per microstep) runs at 2.87 s/microstep too, but peaks at
  **88.2 / 89.6 GiB** on a 94 GB card. Keep patched ZeRO-2.
- **New LMD per-update estimate:** 48 × 3.0 s + ~7 s optimizer ≈ **2.5 min** (was 13.1 min).
  §5's per-update table and §2's floor are superseded: the floor was this bug.

**DeepSpeed 0.18.9 (upstream fix, since 0.18.7) against 0.18.5 + our patch** (job 27183780,
same node). 0.18.9 was installed side by side in
`/gpfs/scratch1/shared/faster-wams/deepspeed_0.18.9` and used via `PYTHONPATH`; `fm_env`
was not touched.

| | 0.18.5 + patch | 0.18.9, no patch |
|---|---|---|
| microstep median / mean fwd+bwd | 3.02 / 3.17 s | 3.05 / 3.24 s |
| optimizer microstep (update 1) | 37.0 s | 28.6 s |
| update 1 loss / grad norm | 0.325122 / 1256.6 | 0.325122 / 1246.4 |
| update 2 loss / grad norm | 0.560891 / 1299.1 | 0.560267 / 1307.6 |
| peak alloc/reserved GiB (upd 1, 2) | 68.6/74.0, 79.9/83.3 | 68.6/73.0, 79.9/84.3 |

- Equivalent. Grad norms are within the known ±1.5% run-to-run spread.
- `patch_zero_hook_count` skips itself on 0.18.9 (feature detection of
  `should_refresh_expected_hook_count`).
- The resume-compatibility check (0.18.5 checkpoint under 0.18.9) did not run: `resume` and
  `pretrained_ckpt` are mutually exclusive. Fixed (`pretrained_ckpt=null`) and resubmitted
  as job 27198328.

**Options that were on the table before the root cause was found:**
1. **ZeRO-1** (`ds_zero1_config.json` exists). Reduce only at the accumulation boundary.
   Memory: full bf16 grads per rank (+12 GiB) with the optimizer still sharded. That is about
   24 (student + teacher) + 12 (grads) + 18 (sharded fp32 master + Adam) + 32 (LMD activations)
   ≈ 86 GiB on 94 GB. Tight; measure.
2. DDP gradient accumulation (`no_sync`) with a sharded optimizer (torch
   `ZeroRedundancyOptimizer`), or FSDP `SHARD_GRAD_OP` with `no_sync`.
3. Bigger microbatch (fewer reductions per update). It only halves the cost at microbatch 2.
4. Diagnose the slowness itself: NCCL all-reduce / reduce-scatter bandwidth test on a
   gpu_h100 node (`nvidia-smi topo -m`, `NCCL_DEBUG=INFO`), and torch.profiler on one ZeRO-2
   backward.

## 10. Fused attention JVP: TVM kernel validation (1 H100)

**Code.**
- Now integrated in `src/flexpi/models/helpers/jvp_attention/`, enabled with
  `flow_map.jvp_attention: tvm`.
- The evaluation harness is in `jvp_kernel_analysis/` (its `fused_attention_jvp.py`
  re-exports the shipped code).
- `tvm_kernels/`: TVM's Triton kernels, unmodified, CC BY-NC-SA 4.0.
- `fused_attention_jvp.py`: a Function (q,k,v,tq,tk,tv) → (o, tō) with backward to all six
  inputs, the mask row-grouping, and a drop-in `scaled_dot_product_attention`.
- `test_grouping.py`: CPU, FP64-exact.
- `validate_tvm.py`, launched by `scripts/slurm/validate_tvm.sbatch`.
- Backend switch: patch `wan_video_dit.scaled_dot_product_attention`. Every attention call
  (MoT joint, HBridge per-stream, cross-attention, action expert) goes through it.
- Option survey and integration plan: `jvp_kernel_analysis/README.md`.

**Masks → mask-free calls.** Rows with identical mask rows form a group; each group is one
kernel call on the gathered rows and allowed columns.
- The full-joint mask gives **3 groups** (anchors → anchors; futures → all visual; action →
  all). Predicted density 0.7585, measured 0.7587.
- Flex-joint regimes give at most 7 groups per sample. Fully masked rows give zero output.
- A batch that shares one mask gets batched calls; mixed masks loop per sample.
- Plans are cached per mask tensor (base tensor + weakref), so there is no per-call GPU sync.
- CPU FP64-exact tests cover: full joint, a flex regime (XOR drop, action not seeing rem_d,
  absent stream), per-sample masks, and broadcast cross-attention masks.

### Results: job 27180546 (2026-09-25, `flowmap_libero_lmd_full`, LIBERO cache)

**Accuracy against FP64** (relative error; 24×128 heads, bf16):

| | output | tangent | six gradients |
|---|---|---|---|
| explicit FP32 (production) | 1.7e-3 | 1.7e-3 | 2.3e-3 |
| TVM, synthetic full joint / flex (5 groups) | 2.2e-3 | 2.3–2.7e-3 | 3.3–5.2e-3 |
| TVM, real captured attention, L = 1670 | 1.7e-3 | 4.1e-3 | ≤ 7.1e-3 |

**Full LMD step, same seed:**
- TVM against explicit: **grad cosine 0.99997, rel err 0.72%**, norm ratio 1.0009.
- Loss 0.13605 against 0.13673.
- Explicit against explicit is bit-identical, so the 0.72% is the kernel's numerics. For
  scale, 4-GPU ZeRO-2 grad norms vary ~1.4% run to run.

**Per call** (real L = 1670, fwd+bwd):

| Path | Time | Extra memory |
|---|---|---|
| fused SDPA (no JVP) | 1.74 ms | +0.14 GiB |
| explicit FP32 JVP | 25.55 ms | +2.86 GiB |
| **TVM JVP** | **6.65 ms (3.8×)** | **+0.27 GiB** |

**Steps** (1 GPU, no optimizer state; s per step / peak GiB):

| Mode | explicit mb1 | TVM mb1 | explicit mb2 | TVM mb2 |
|---|---|---|---|---|
| lmd_full | 2.59 / 67.3 | 2.68 / **59.1** | OOM | **3.06 / 82.7** |
| lsd_off | 2.58 / 67.3 | 2.57 / **57.4** | harness bug | harness bug |
| esd_off | 3.34 / 67.4 | 3.14 / **57.5** | harness bug | harness bug |
| lmd_semigrad | 1.32 / 45.1 | 1.25 / 43.6 | 1.53 / 54.8 | 1.39 / 51.9 |

**Reading:**
- **Memory: −8 to −10 GiB per step.** This makes LMD full-grad fit at **microbatch 2: 1.53 s
  per example against 2.59, i.e. 1.7× throughput.** The explicit path OOMs at mb2.
- **No mb1 speedup yet, despite 3.8× faster attention calls.**
  1. The step is CPU/launch-bound at mb1 (GPU busy ~1.5 of ~2.6 s), so faster kernels barely
     move wall time.
  2. The first wrapper synced the GPU on every attention call (`torch.equal` cache check and
     a batch-equality `.all()`). That destroys CPU/GPU overlap.
  3. Smaller: the group gathers/scatters and Triton launch overhead.
- Fixed: the plan cache is keyed by the mask tensor, with no sync. Re-measuring in job
  27180820.
- The "harness bug": `profile_flowmap_step.with_mask` built the self-distillation mixture
  mask with shape (1,). It is fixed (one flag per example).

### Rerun: job 27180820 (sync-free plan cache, harness fix)

Steps, s per step / peak GiB, 1 H100 (the last column is per example):

| Mode | explicit mb1 | TVM mb1 | explicit mb2 | TVM mb2 | TVM mb2 / example |
|---|---|---|---|---|---|
| lmd_full | 2.58 / 67.3 | **2.45** / 59.1 | OOM | **2.77 / 82.7** | **1.39 s (1.86× vs explicit mb1)** |
| lsd_off | 2.57 / 67.3 | **2.34** / 57.4 | 5.13 / 81.8 | 4.92 / 71.9 | 2.46 s |
| esd_off | 3.32 / 67.4 | **3.12** / 57.5 | 6.65 / 82.0 | 6.15 / 72.1 | 3.08 s |
| lmd_semigrad | 1.32 / 45.1 | 1.24 / 43.6 | 1.53 / 54.8 | 1.34 / 51.9 | 0.67 s |

- **mb1:** with the sync gone, TVM is 5–9% faster in every mode. The step is launch-bound, so
  the 3.8× attention speedup shows up only partly.
- **mb2:** LMD full-grad fits only with TVM: 1.39 s per example against 2.58 at mb1
  explicit. This is the main throughput gain.
- **LSD/ESD mb2 do not batch** (4.92 ≈ 2 × 2.46): the self-distillation per-example loop
  (§6 follow-up). Batching it is the next lever for LSD/ESD.
- **Still open:** 82.7 GiB is 1 GPU without optimizer state; ZeRO-2 adds ~12 GiB. mb2 needs a
  4-GPU test once TVM is wired into training.

**Flex-joint: share the regime within a microbatch (implemented behind a flag).**
- `sample_flex_batch_flags` (`helpers/flex_joint.py`) used to draw every flag per sample.
  With `flex_joint.share_within_microbatch: true` (in `configs/model/flexpi.yaml`, default
  **false**; saved configs without the key keep per-sample draws), it draws one sample's six
  flags, including the all-absent rejection, and broadcasts them to the microbatch.
- Result: one mask per microbatch, hence one batched fused-JVP call per row group.
- Tests (`tests/test_flex_joint_share.py`): identical within a microbatch when on, varied
  when off, rejection kept, marginals equal to per-sample sampling (4000 draws, ±0.04).
- Only unit-tested: no flex-joint flow-map run exists yet.
- The gradient stays unbiased: each sample's regime marginal is unchanged, and only the two
  samples of a microbatch become correlated.
- It halves the independent regime draws per update (192 → 96 at mb2). That is negligible
  with ≤ 64 regimes. Possible next step (not done): stratify regimes across microsteps,
  as §6 does for the diagonal mixture, to remove the per-update regime variance entirely.
- It only matters once flex-joint flow maps are trained. The current study uses full joint.

**Side note: RoboTwin cache build** (`docs/LATENT_CACHE.md`).
- Job 27165240 was OOM-killed on host RAM after 8 h: step MaxRSS 145 GB, DataLoader shm bus
  errors. It had 586,564 / 784,973 windows done.
- `scripts/cache_latents.py` now restarts workers and memmaps every `--restart-every`
  (20k) windows, after flushing. Resumed as job 27181221.

### Integrated TVM in the real 4-GPU ZeRO-2 trainer (job 27198562, 2026-09-26, full joint)

| Run | microstep median / mean | peak alloc GiB (upd 1 / 2) | vs explicit |
|---|---|---|---|
| LMD mb1 | 2.93 / 3.12 s | 55.5 / 66.7 | 3.02 / 3.17 s, 68.6 / 79.9: **−13 GiB** |
| **LMD mb2** | **3.11 s per 2 examples** | 78.2 / **89.4** (90.7 reserved) | **1.56 s per example against 3.02: 1.94×** |
| LSD mb1 | mean 1.16 s | 43.4 (upd 1) | 1.21 s, 57.3: **−14 GiB** |

- The teacher is a fixed ~11–12 GiB for LMD (explicit: LSD 57.3 against LMD 68.6 GiB). TVM's
  13–14 GiB activation savings apply to both objectives. LMD mb2 fits with only ~3–4 GiB
  spare; LSD mb2 would have ~12 GiB more headroom but does not batch yet (per-example
  loop).
- Resume check (job 27198328): DeepSpeed 0.18.5 and 0.18.9 both resume the LMD-100 step-50
  state. Update 51 has a **bit-identical** loss (0.425483) and grad norm (842.62), so the
  0.18.9 bump is safe.

### Flex-joint regimes in the 4-GPU trainer (jobs 27198797, 27198798, 27199064, 27199065)

Configs `flowmap_libero_{lmd,lsd}_flex` (p = 0.5 on every presence/joint flag, cross-modal
on). DeepSpeed 0.18.5 + patch. Only 10–22 microsteps per LMD run and the regimes are random,
so these medians are noisy.

| Run | microstep median / mean | peak alloc GiB (upd 1 / 2) | full-joint reference |
|---|---|---|---|
| LMD mb1 explicit | 2.95 / 3.21 s | 68.6 / 79.9 | 3.02 / 3.17 s |
| LMD mb1 TVM | **3.81 / 4.63 s** | 55.5 / 66.8 | 2.93 / 3.12 s |
| LMD mb2 TVM, per-sample regimes | 4.29 / 4.63 s per 2 | 78.3 / 89.5 | 3.11 s per 2 |
| LMD mb2 TVM, shared regime | 4.72 / 5.17 s per 2 | 78.3 / 89.5 | |
| LSD mb1 TVM | 0.61 / 1.65 s (21/46 slow) | 43.4 | mean 1.16 s (12/48 slow) |
| LSD mb2 TVM, shared regime, looped | 1.46 / 2.58 s per 2 | 57.9 | |

- **Memory: everything fits.** Flex regimes cost the same memory as full joint (the sequence
  length is unchanged; absent streams are masked, not dropped).
- **Speed: TVM loses its advantage under flex regimes.** At mb1 it is ~0.9 s slower than the
  explicit path, whereas it is slightly faster under full joint. Unexplained. Suspects, not
  yet profiled:
  - a flex regime splits into up to 7 row groups, against 3 for full joint, so there are more
    small kernel calls plus gathers and scatters, in a step that is already launch-bound;
  - plan construction (`torch.unique`, GPU syncs) runs for every new mask, i.e. every
    microstep under flex, whereas the full-joint mask never changes.
  Next: `FLEXPI_STEP_PROFILE_MICRO` on `zero2_tvm` with the flex config.
- **The shared regime did not help** (4.72 against 4.29 s per 2 at LMD mb2). It turns the
  per-sample plan loop into batched calls, but with 10 random-regime microsteps the
  difference is within the regime-to-regime variance. No speedup is shown.
- LSD flex has more slow microsteps than full joint (21/46 against 12/48, same stratified
  mask). Some diagonal microsteps are presumably slowed by expensive regimes. Unverified.

### Batched self-distillation loss (2026-09-26)

`helpers/flowmap_training.py`: the LSD/ESD (and learned-time-weight) path used to loop over
examples, one network call each. It now makes **one call per branch** (diagonal or
off-diagonal). Stratified microsteps are single-branch, so this is normally one batched call
per microstep; a mixed microstep makes two.

- **Why the loop existed:** the learned time weight `exp(−logvar(s,t))·L + logvar` acts on
  each example's loss, and the diagonal examples must skip the JVP.
- **Loss unchanged:** `reduce_stream(..., keep_batch=True)` returns `[B]` per-example losses,
  with an absent, non-cross-modal stream contributing 0. This is exactly the batch-1 case of
  the flex-aware present mean. The loss is still the mean of per-example losses.
- **Attention masks need not match** within a microbatch: SDPA takes per-sample masks, and
  the TVM path plans each sample. Only the branch has to match, and stratification
  guarantees that.
- Test: `tests/test_flowmap_flex.py::test_batched_self_distillation_equals_per_example_loop`.
  For LSD and ESD, with all-diagonal, all-off-diagonal and mixed batches under random flex
  regimes, the loss and all gradients match the old loop.

GPU timing on DeepSpeed 0.18.9 (patch off), TVM, 4-GPU ZeRO-2:

| Run | mean fwd+bwd | per example | peak alloc GiB |
|---|---|---|---|
| Full joint LSD mb1 (job 27199500) | 1.21 s | 1.21 s | 43.4 |
| **Full joint LSD mb2, batched** (job 27199500) | 1.25 s per 2 | **0.63 s (1.94×)** | 66.0 |
| Flex LSD mb2, looped (job 27199065) | 2.58 s per 2 | 1.29 s | 57.9 |
| Flex LSD mb2, batched, shared regime (job 27199501) | 2.30 s per 2 | 1.15 s | 66.0 |

- Full joint gets the same ~1.94× from mb2 as LMD. Flex gains only 1.12×, consistent with the
  flex TVM slowdown above.
- Batching costs +8 GiB at mb2 (66.0 against 57.9 GiB looped). The loop also kept both
  examples' graphs until backward, but its transient per-call buffers were batch 1. 66 GiB
  leaves ~25 GiB, so LSD mb3–4 should fit (untested).

## 11. Making the microstep faster: CUDA graphs, torch.compile, explicit tangents (plan, 2026-09-26)

**The step is launch-bound, not compute-bound.** Three measurements:
- 1-GPU explicit profile: GPU busy 1.52 of 2.67 s.
- TVM cut attention GPU time ~3.8×, but the wall time only dropped 2.58 → 2.45 s.
- A second example per microstep costs only 0.2–0.3 s (LMD TVM: 2.93 → 3.11 s at 4 GPUs;
  2.45 → 2.77 s at 1 GPU).

FLOP estimate for LMD: ~8 forward-equivalents per example (primal + tangent, backward through
both, teacher) × ~20 TFLOP (5B trunk, ~1,640 tokens) ≈ 160 TFLOP. That is **~0.3 s per
example** at half of H100 bf16 peak, which matches the marginal cost. Today we pay 2.93 s at
mb1 and 1.56 s/example at mb2. The rest is fixed per-microstep cost:
- Python/dispatcher launches of tens of thousands of small kernels;
- forward AD roughly doubles the op count (each op also runs its tangent formula);
- the backward through the JVP, and the checkpoint recompute.

**CUDA graphs** (replay the captured kernel sequence with one launch). They remove CPU launch
cost and fuse nothing. They work at kernel level, so forward AD, TVM and the recompute are
fine. Blockers:
1. ZeRO-2's backward hooks (Python + NCCL) cannot be captured. Accumulate gradients locally
   inside the graph (ZeRO-1 / DDP-no-sync style) and reduce once per update.
2. Host syncs in the step (`float(...)` metrics, `bool(...)` checks).
3. The graph's private memory pool: mb1 has room, mb2 (89.5 GiB) does not.
4. Flex regimes: the TVM plan depends on mask contents, so this is full joint only.
5. LSD needs two graphs (one per branch). Stratification makes each microstep single-branch.

Estimate: the mb1 step drops to about the GPU-busy time, **~2–2.5×** if that is 1.0–1.3 s.
Unmeasured.

**torch.compile.** It fuses pointwise work: norm+modulation, GELU and its derivative, RoPE on
q and tq, residual adds, and their backward. But it most likely cannot trace
`torch.autograd.forward_ad` duals: compiled regions have no forward-mode rule. Unchecked on
torch 2.7.1.

**Explicit tangent forward.** Each block maps `(x, tx) → (y, ty)` with ordinary ops, as TVM
does for attention.
- Weights carry no tangent, so a linear layer is one GEMM on the stacked `[x; tx]`.
- The time tangent enters through the time embedding into the adaLN shift/scale.
- The result is plain reverse-mode autograd, so it can be compiled and graph-captured. It also
  removes forward-AD overhead in eager mode, and removes the dual-aware checkpoint workaround.
- Effort: ~1–2 weeks, validated against the forward-AD path in fp64. Risk: two forward paths
  to keep in sync. Gain on top of graphs: fusion, maybe 20–40% of GPU time (a guess).

**Order and estimated LMD update time** (now ~1.4 min at mb2: 24 × 3.11 s + ~8 s post):

| Step | Effort | Expected gain | Update |
|---|---|---|---|
| 0. Profile a TVM microstep and an update boundary: GPU busy, kernels, syncs, post phases | 1 job (27200222) | turns these ranges into numbers | — |
| 1. Fix the ~8 s post-update cost; remove syncs from the loss | days | ~8 s → <1 s per update | ~1.25 min |
| 2. CUDA-graph the full-joint mb1 microstep, local gradient accumulation | ~1 week | ~2–2.5× microstep | ~50–60 s |
| 3. Explicit tangent blocks + torch.compile | 1–2 weeks | GPU-side fusion | ~30–40 s |

Floor: 48 examples/GPU × 0.3 s ≈ 15 s of compute per update. Realistic total: **2–3×**.

Cheaper alternative: a bigger microbatch is nearly free throughput while launch-bound. Wider
checkpointing to reach mb4 re-dispatches the recomputed forward too, so expect ~1.3–1.4×.

**Post-update cost (what step 0 examines).** Timing runs disable eval and checkpoints. Under
accelerate + DeepSpeed, `engine.step()` (optimizer step, clipping) runs *inside*
`accelerator.backward` at the boundary. The boundary backward is only ~0.3 s longer in steady
state (~2.3 s at update 1, with lazy optimizer-state init). The trainer's own
`optimizer.step`/`clip_grad_norm_` are DeepSpeed no-op wrappers. Yet `post_ms` is ~29–31 s at
update 1 and **7.6–8.4 s at update 2** (jobs 27198562, 27199500), spent after the step. The
harness now times each post phase with syncs (`[STEPTIMING] ... phases` lines, including
`ds_engine_step`) and profiles update 2 (`FLEXPI_STEP_PROFILE_UPDATE`,
`utils/step_profile.py`).

### Step 0 results: job 27200222 (4×H100, ZeRO-2, TVM, DeepSpeed 0.18.9, full joint)

**Microstep anatomy, LMD mb1, one microstep, rank 0** (`prof_lmd_tvm/step_profile_*`):

| Metric | Value |
|---|---|
| Unprofiled microstep wall | ~2.7–3.0 s (median 3.19 incl. profiled outliers) |
| GPU busy (union of kernels) | **1.19 s**, ~40–45% of the unprofiled wall |
| Kernels / launches / CPU-side aten ops | **81.5k / 72.9k / 538k** |
| Median kernel duration | 3.5 µs; 54k of 90k GPU events < 5 µs |
| Host syncs | 45 |

GPU time by kernel type:

| Type | ms | share | kernels |
|---|---|---|---|
| elementwise / copy / index | 555 | 46.7% | 64.4k |
| GEMM | 248 | 20.9% | 7.9k |
| attention (TVM fwd/bwd, fused SDPA) | 193 | 16.3% | 1.9k |
| reduction / norm | 76 | 6.4% | 6.9k |
| NCCL all-reduce | 54 | 4.6% | 31 |
| memcpy / memset | 48 | 4.1% | 8.8k |

Top aten ops by GPU time: `mul` (21.5k calls, 198 ms), `mm` (4.8k, 208 ms), `copy_` (34k,
146 ms: dtype casts and contiguity copies), `add`/`add_` (14.8k, 140 ms), TVM backward
(206 calls, 125 ms).

What this changes in the plan:
- Launch-bound confirmed: the CPU needs ~2.9 s to issue 538k ops, and the GPU is busy 1.19 s.
  CUDA graphs would take the microstep to about 1.2 s, **~2.4×**.
- **The GPU time itself is mostly unfused elementwise work**, not math: GEMM + attention are
  0.44 s, close to the FLOP estimate (~0.3 s); elementwise/norm/copy are 0.68 s across ~80k
  tiny kernels (tangent formulas, casts, their backward). Fusion (explicit tangents +
  compile) targets exactly this. If it cut that part 3–5×, the GPU busy time would fall to
  ~0.6 s. With graphs, the microstep could reach **~0.6 s (~4–5×)**, more than the earlier 2–3×
  guess. Still an estimate.
- Syncs are a minor cost: ~45–53 per microstep (LSD mb2 inventory: 12 batch `send_to_device`,
  11 metric scalars → tensors in `update_metrics.add`, ~10 `float(...)` metric reads in
  `flowmap_training.py`, RoPE/padding helpers).

**Update boundary: the EMA was the whole ~8–30 s "post" cost.** Synced phase timings
(`[STEPTIMING] ... phases`), LMD mb1:

| Update | `engine.step` (in backward) | EMA (rank 0) | all other phases |
|---|---|---|---|
| 1 | 1.11 s | **29.5 s** | < 0.2 s |
| 2 | 0.42 s | **8.8 s** | < 0.1 s |
| 3 | 0.10 s | **9.1 s** | < 0.1 s |

- `utils/flowmap_ema.py` kept two fp32 CPU shadows (decays 0.999, 0.9999) on rank 0 only. Per
  update, it copied every trainable parameter synchronously GPU → pageable memory (1,727
  syncs, 1.0 s of DtoH), converted to fp32, and lerped each shadow on the CPU (~8 s, with
  `OMP_NUM_THREADS=2` in the harness). Ranks 1–3 waited in the next collective.
- LMD uses it too: `uses_ema` is on for self-distillation or `distill_ema`.

**Fix: background EMA** (`utils/flowmap_ema.py`, 2026-09-26):
- Snapshot the weights into pinned host buffers with async copies on the current stream. Later
  kernels, including the next optimizer step, are ordered after the copies.
- Fold the snapshot into the shadows in a background thread. The arithmetic is unchanged, so
  results are bit-identical (test `test_background_ema_matches_synchronous_updates`).
- Every reader (`shadow`, `state_dict`, `apply`, `load_state_dict`) waits for a pending fold,
  and so does an epoch-boundary DataLoader re-fork.
- `FLEXPI_EMA_BACKGROUND=0/1` forces the mode.

First version (one op per parameter), measured in job 27200222 (mb2 LMD and LSD):
- On the critical path, EMA **9 s → 0.24 s** per update, plus a one-time 3.5 s pinned
  allocation at update 1.
- But the fold took 24–72 s. While the first fold ran, every LSD microstep was ~1.6× slower
  (0.67 → 1.0–1.18 s on all ranks); LMD rank 0 was ~10% slower. That erased the gain for LSD.
  During the second LSD fold most microsteps were unaffected.
- Suspected cause: the fold thread (~5k separate ops) competing with the dispatch-bound main
  thread (GIL handoffs, CPU cache). Not isolated.

Final version (shipped): flat fp32 shadows per decay (grouped by source dtype,
per-parameter views preserved for checkpoints), folded in 64M-element chunks (~270 ops) through
one reused fp32 chunk buffer. Background is the default for CUDA weights.

A/B at the production accumulation (mb2 × 24 microsteps × 4 GPUs = 192), TVM, steady-state
updates 3–4 (jobs 27202122, 27205496; `ema_{sync,bg}_{lsd,lmd}`):

| | fold (s) | post per update (s) | **update wall (s)** |
|---|---|---|---|
| LSD, original per-parameter sync EMA (job 27200222) | ~9 | ~9 | ~38.5 (est.) |
| LSD, flat sync, fresh chunk temps (job 27202122) | 5.0–5.5 | 5.2–5.3 | 34.2–34.3 |
| LSD, flat sync, reused chunk buffer (job 27205496) | 2.6–3.1 | 2.7–3.3 | 32.6–33.0 |
| **LSD, background** (job 27205496) | 3.7–7.7, overlapped | **0.2** | **29.7–30.3** |
| LMD, flat sync, fresh chunk temps (job 27202122) | ~5 | 5.0–7.9 | 78.1–79.6 |
| **LMD, background** (job 27202122, fresh temps) | 7.8–11.4, overlapped | **0.2–0.3** | **71.7–73.6** |

- LSD update: ~38.5 → ~30 s (−23%). LMD update: ~81 → ~72 s (−11%).
- Steady-state background folds do not slow the microsteps: the LSD median fwd+bwd is 0.67 s in
  both modes.
- **One-time cost, unexplained:** the first fold after startup takes ~30 s in either mode
  (`ema_prev_fold` 30.2 s sync; 38–41 s background, where it also slows the next update's
  microsteps to ~1.0–1.1 s, +~10 s). It is not the chunk allocation (the reused buffer did not
  change it). It costs ~30–40 s once per training segment.
- Update 1 also pays a one-time ~3.5 s pinned-buffer allocation (~12 GB).

**Per-update budget after the EMA fix** (TVM, mb2, steady state): LSD ~30 s, LMD ~72 s per 192
examples. Almost all of it is microstep compute. The next lever is §11's launch overhead and
fusion.


## 12. Timeline: one full update (192 examples, 4×H100) from baseline to now (2026-09-26)

Per-update wall time includes the post-update work (EMA). Each row adds one change to the row
above. **Measured** means a full update or a per-microstep median/mean × accumulation from the
cited job; *est.* means derived from measured parts.

### LMD full-grad (distillation, teacher in memory)

| # | Change | Setup | Per update | vs baseline | Source |
|---|---|---|---|---|---|
| 0 | Baseline | raw data (VAE + DINO online), DeepSpeed 0.18.5, explicit attention JVP, mb1 × 48, rank-0 CPU EMA | **785 s (13.1 min)** | 1× | measured, job 26887233 (90 updates) |
| 1 | + latent cache | same, cached inputs | ~16.0 s/microstep → ~12.9 min *est.* | ~1.0× | job 27176766 (the DeepSpeed bug hid the saving) |
| 2 | + ZeRO-2 hook patch | O(N²) → cached hook count | 3.00 s/microstep → **~153 s (2.6 min)** | 5.1× | job 27180109 + ~9 s EMA |
| 3 | + DeepSpeed 0.18.9 | upstream fix, patch skips itself | 3.05 s/microstep, same | — | jobs 27183780, 27198328 (resume bit-identical) |
| 4 | + TVM fused attention JVP, mb1 | `flow_map.jvp_attention=tvm` | mean 3.12 against 3.17 s → ~2.6 min; **−13 GiB** (55.5/66.7 GiB) | — | job 27198562 |
| 5 | + mb2 (possible thanks to TVM's memory) | `batch_size=2`, accumulation 24 | 3.11 s per 2 → **~84 s (1.4 min)**; peak 89.4/90.7 GiB | 9.3× | job 27198562 + 9 s EMA |
| 6 | + background EMA | flat chunked fold, off the critical path | **71.7–73.6 s (1.2 min)** | **~10.9×** | measured, job 27202122 |

### LSD (self-distillation, 75/25 diagonal mixture, no teacher)

| # | Change | Setup | Per update | vs baseline | Source |
|---|---|---|---|---|---|
| 0 | Baseline | raw data, DeepSpeed 0.18.5, explicit, mb1 × 48, shuffled diagonal mask | **~12.2 min *est.*** (never run at scale) | 1× | §5 cost model |
| 1 | + latent cache + ZeRO-2 patch | shuffled mask: ~69% of microsteps pay a JVP | ~2.1 s/microstep → ~110 s (1.8 min) *est.* | ~6.7× | from job 27183027's measured branch times |
| 2 | + stratified diagonal mask | whole microsteps on one branch (12/48 off-diagonal) | mean 1.21 s → **~67 s (1.1 min)**; 57.3 GiB | ~11× | measured, job 27183027 + ~9 s EMA |
| 3 | + TVM, mb1 | | mean 1.16 s → ~65 s; **−14 GiB** (43.4 GiB) | ~11× | job 27198562 |
| 4 | + batched self-distillation loss, mb2 | one network call per branch, accumulation 24 | 1.25 s per 2 → **~39 s**; 66.0 GiB | ~19× | job 27199500 + 9 s EMA |
| 5 | + background EMA | | **29.7–30.3 s** | **~24×** | measured, job 27205496 |

Against measured LSD at row 2 (the first time LSD ran at scale): 67 s → 30 s, 2.2×.

### What the current numbers need (not defaults yet)

- `+data.latent_cache_dir=data/latent_cache/libero_fulljoint_v2` (fingerprint-checked).
- `model.flow_map.jvp_attention=tvm` (default `explicit`; TVM code is CC BY-NC-SA 4.0).
- `batch_size=2 gradient_accumulation_steps=24` (the shipped configs use 1 × 48).
- Automatic: DeepSpeed 0.18.9 (`fm_env`), stratified mask, batched self-distillation loss,
  background EMA (`FLEXPI_EMA_BACKGROUND=0` falls back to synchronous).
- Flex-joint configs (`flowmap_libero_{lmd,lsd}_flex`) are slower: TVM under flex regimes
  is currently slower than explicit (§10), so these numbers are for full joint only.

### Still on the table (§11)

- Microstep is launch-bound: 538k CPU ops and 81.5k kernels for 1.19 s of GPU work at LMD mb1.
  CUDA graphs: ~2.4× *est.*
- 57% of GPU time is unfused elementwise/copy/norm work. Explicit tangents + torch.compile could
  bring a microstep to ~0.6 s *est.* (~4–5×).
- One-time ~30 s first EMA fold per segment (unexplained); flex TVM slowdown; ~50 host syncs per
  microstep.


## 13. CUDA graphs: whole-microstep capture measured (1 H100, 2026-09-26)

**Forward vs backward idle, from the step-0 trace** (job 27200222, LMD mb1 TVM, one profiled
microstep; the profiler inflates CPU time ~1.5×; split by thread, forward = main thread until the
autograd thread starts):

| Phase | wall | GPU busy | CPU ops | kernels |
|---|---|---|---|---|
| forward (student JVP + teacher) | 1782 ms | 294 ms (**16%**) | 177k | 25k |
| backward (incl. checkpoint recompute) | 2576 ms | 887 ms (34%) | 360k | 56k |

At mb2: forward 21% busy, backward 38%. The forward is the most launch-bound part.

**What made the step capturable** (all bit-identical in eager; CPU suite + fingerprint unchanged):
- `flowmap_core.graphs.CapturedStep`: side-stream warmup, capture of loss + backward, replay;
  gradients accumulate into the existing `.grad`. `guard()`/`keep_alive()`/`check()` for frozen
  content-dependent decisions. Test: `flowmap_core/tests/test_core_graphs.py` (GPU; tiny DiT
  LMD/LSD replay bit-identical to eager, accumulation too).
- `flowmap_core.jvp_attention`: at capture, a mask that misses the plan cache (FlexPi's
  cross-attention key masks, rebuilt each forward) reuses the latest same-shape eager plan, with a
  device-side equality guard (`CapturedStep.check()` raises if a replay saw a different mask).
- `flowmap_core.checkpoint`: `preserve_rng_state=False` only while capturing (the checkpointed
  attention is RNG-free; get_rng_state is illegal in capture).
- FlexPi: `model._glue_cache_train = True` memoizes shape-only masks/freqs in training too (so
  TVM plans hit by tensor identity); `flowmap_training.deferred_metrics()` keeps loss metrics as
  GPU tensors; RoPE tables cached on device (`wan_video_dit._rope_freqs`, `action_dit`: was 4 host
  copies + syncs per forward); `_aux_per_frame_is_pad` without a list index (was a sync).
- Remaining host syncs in a capture-ready eager step: 0 (explicit) / 2 eager-only plan lookups (TVM).

**Results** (`scripts/cuda_graph_step.py`; jobs 27216144 tvm mb1, 27216145 explicit mb1,
27216146 tvm mb2; full-joint LIBERO, 1 H100, no DeepSpeed):

| Mode | mb | eager | graph | speedup | graph reserved |
|---|---|---|---|---|---|
| LMD full, TVM | 1 | 2.476 s | **1.138 s** | 2.18× | 63.0 GiB |
| LSD off-diagonal, TVM | 1 | 2.362 s | **1.038 s** | 2.27× | 61.3 GiB |
| LSD diagonal, TVM | 1 | 0.527 s | **0.273 s** | 1.93× | 45.4 GiB |
| LMD full, explicit | 1 | 2.681 s | 1.547 s | 1.73× | 71.6 GiB |
| LSD off, explicit | 1 | 2.485 s | 1.447 s | 1.72× | 71.2 GiB |
| LMD full, TVM | 2 | 2.679 s | 2.110 s | 1.27× | 88.2 GiB |
| LSD off, TVM | 2 | 2.511 s | 1.890 s | 1.33× | 85.3 GiB |
| LSD diagonal, TVM | 2 | 0.587 s | 0.499 s | 1.18× | 53.7 GiB |

Correctness: losses bit-identical in every mode. With `explicit` attention, graph gradients are
bit-identical to eager (1 step and 2-step accumulation, all modes; one earlier run, 27216090, saw
6.8e-3 on LMD 2-step once). With TVM, graph-vs-eager per-tensor differences (worst tensor
1.7–4.2% rel L2) are the size of eager-vs-eager with the same seed (2.0–6.7%): TVM's atomics, not
the graph. (Worst-tensor metric; the global grad-norm spread is ~4e-7, §S0 of 08.)

**What it means:**
- A graph brings the microstep down to about its GPU-busy time. At mb1 that is 2.2× — the
  launch overhead measured in step 0.
- **At mb2 the graph is already GPU-bound**: 2.11 s for 2 LMD examples = 1.05 s/example, the
  same as mb1 graphed (1.14 s). Production already runs mb2 eager (1.34 s/example LMD, 1.26 LSD
  off on 1 GPU), which hides half the launch cost. Graph over *production*: ~1.2× per example.
- Memory: mb2 graphed needs 88 GiB on 1 GPU before ZeRO state (+~12 GiB) → does not fit at
  4-GPU ZeRO-2. mb1 graphed (63 GiB) does.
- Projected update (48 examples/GPU; 1-GPU compute only, no comm): LMD mb2 eager 64 s (72 s
  measured at 4 GPUs) → mb1 graphed ~55 s; LSD 26 s (30 s measured) → ~22 s. **~1.2–1.3×**, not
  the 2.4× the mb1 profile suggested.
- The GPU time is now the cost: ~1.05 s/example against a ~0.3 s FLOP estimate; step 0 put 57%
  of it in unfused elementwise/copy/norm kernels (tangent formulas, casts). Fusion is the lever.
- torch.compile cannot trace forward-AD duals on torch 2.7.1 (checked: dynamo "Nested forward mode
  AD is not supported"; inductor's compiled Function has no JVP rule). It applies only to dual-free
  parts (LSD diagonal branch, the LMD teacher query) or after the explicit-tangent rewrite.
- Not yet in the trainer: ZeRO-2 reduces gradients from Python hooks during backward, which
  cannot be captured. Options: in-graph NCCL reduce-scatter of a flat grad buffer + own sharded
  AdamW (a small ZeRO-2 of our own), or ZeRO-1/DDP-no-sync style with one reduce per update.


## 14. Next: understanding and cutting the GPU compute (plan, not started)

After graphs (§13) a microstep costs its GPU time: ~1.05–1.14 s per LMD example (1 H100),
against a ~0.3 s FLOP estimate (§11). Step 0 attributed 57% of GPU time to unfused
elementwise/copy/norm kernels (64k kernels, median 3.5 µs). Before changing code, find out
*which* model code produces that time. Steps, in order:

1. **Profile a graphed replay** (`scripts/cuda_graph_step.py` + torch.profiler around
   `graph.replay()`, or nsys with `--cuda-graph-trace=node`): pure GPU time with no launch gaps.
   Report kernel categories as in step 0, at mb1 and mb2, for LMD full, LSD off and LSD diagonal.
2. **Attribute kernels to model regions.** Add `torch.profiler.record_function` (or NVTX) ranges
   around: student primal+tangent forward, teacher forward, backward, checkpoint recompute; and
   inside a DiT block: adaLN modulation, norm, q/k/v + RoPE, attention, output proj, FFN, residual.
   Profile eagerly (ranges are CPU-side; the graph replay has none) and map kernel time per range.
3. **Casts and copies.** `aten::copy_` was 34k calls / 146 ms and `_to_copy` 12k per microstep.
   List where they come from (per call site, via `record_shapes`/stack): suspects are
   `ForwardADLayerNorm`'s FP32 round trip, the explicit attention's FP32 path (TVM leaves some
   casts), `.float()` in `predict_streams`/loss, RoPE in FP32/complex, autocast re-casts of weights
   under forward AD. Each removable cast pair is two kernels ×(forward, tangent, backward).
4. **GEMM efficiency.** GEMM+attention were ~0.44 s vs ~0.3 s FLOP-ideal. Check achieved
   TFLOP/s per GEMM shape (≈1,640 tokens × 3072 at mb1: skinny); mb2 or stacking primal and
   tangent (`[x; tx]` in one GEMM, the explicit-tangent idea) doubles M.
5. **Checkpoint recompute share.** The mixed-attention checkpoint recomputes attention forward +
   JVP in the backward. With TVM, memory at mb1 graphed is 63 GiB (+~12 GiB ZeRO): measure the
   step with `mot_checkpoint_mixed_attn=false` if it fits.
6. **Decide the fusion route** from the numbers:
   - cheap: delete redundant casts/copies; fuse adaLN modulate+norm (+tangent) with a small Triton
     kernel; TF32 where FP32 math remains;
   - torch.compile on the dual-free parts (LSD diagonal branch = 75% of LSD microsteps; LMD teacher
     query), which works on torch 2.7.1;
   - explicit tangent blocks (`(x, tx) → (y, ty)` with plain ops) + torch.compile for the JVP
     forward (§11, 1–2 weeks); first check whether torch 2.11 (`diff_env`) compiles forward-AD
     duals, which would avoid the rewrite.
7. **Then integrate graphs in the trainer** (§13: needs a capture-compatible gradient reduce
   instead of ZeRO-2's hooks). Graphs matter more once GPU time falls, since launch overhead is
   then again the larger share.

Tools already in place: `flowmap_core.step_profile` (trace summary, sync sites), the trace
phase split in §13, `scripts/profile_flowmap_step.py`, `scripts/cuda_graph_step.py`,
`FLEXPI_STEP_PROFILE_*` in the trainer.

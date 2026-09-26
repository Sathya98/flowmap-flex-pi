# Fused attention-JVP kernels: which one fits us

> **Status (2026-09-26): TVM kernel integrated behind `flow_map.jvp_attention: tvm`**
> (`flowmap_core/src/flowmap_core/jvp_attention/`, aliased as `flexpi.models.helpers.jvp_attention`; license accepted for now; default stays
> `explicit`). Validation results: `.claude/context/07-efficiency-notes.md` §10.
> `fused_attention_jvp.py` here is now a shim re-exporting the shipped code.

Question: can the explicit FP32 forward-AD attention (`src/flexpi/models/helpers/attention.py`)
be replaced by a fused kernel? It costs **47.6 ms fwd+bwd per call against 1.5 ms fused SDPA**,
and its forward ranges alone are ~47% of LMD GPU time
(`.claude/context/07-efficiency-notes.md` §8, job 27172448).

Starting point: the options in `docs/jvp-kernels.md`, plus the TVM paper. Everything here comes
from reading the code (repos cloned under `repos/`). **Nothing has run on a GPU yet.**

Per-repo notes: [`notes_tvm.md`](notes_tvm.md), [`notes_rcm.md`](notes_rcm.md),
[`notes_jvp_flash_attention.md`](notes_jvp_flash_attention.md),
[`notes_dmf_fastgen.md`](notes_dmf_fastgen.md).

## What we need

| Requirement | Why |
|---|---|
| **R1** Backward *through* the JVP (grads w.r.t. q,k,v **and** their tangents) | LMD full-grad and LSD put ∂ₜv inside the loss (`detach_derivatives: false`). A semigradient variant would need only a forward JVP. |
| R2 Our mask | Dense bool `[B,1,L,L]`, density 0.76. **But see the finding below.** |
| R3 Shapes | L ≈ 1670 (unaligned), 24 heads × 128, bf16, B = 1–2, Sq ≠ Skv after decomposition |
| R4 Integration | `torch.autograd.forward_ad` duals + our dual-aware checkpoint (not `torch.func.jvp`) |
| R5 Toolchain | torch 2.7.1 / Triton 3.3.1, H100 + A100 |

## Key finding: our mask decomposes into 3 mask-free calls

In the full-joint flow-map configs, the rows fall into three groups, each attending to one
fixed set of columns (derived from `flexpi.py:636`, `:2739-2784` and the `first_frame_causal`
video mask):

- anchors (ff_v, ff_d, ff_p) → anchors
- futures (rem_*) → all visual
- action → everything

Predicted density 0.7585 against **0.7587 measured**. So the masked attention is exactly three
unmasked attentions on gathered rows/columns, with no wasted work. **Mask support stops being a
requirement** for any kernel that handles Sq ≠ Skv and unaligned lengths.

Caveat: this does not hold for flex-joint training with per-sample masks.

## Comparison

| | R1 backward through JVP | Mask | Unaligned L / Sq≠Skv | forward_ad | License | Verdict |
|---|---|---|---|---|---|---|
| **TVM** (lumalabs/tvm `jvp_utils`) | **Yes**: fused Triton kernel for grads of q,k,v,tq,tk,tv (`flash_jvp_backward.py`) + FA2 Triton primal backward | none (fine given the decomposition) | yes / yes | `Function.jvp` + `setup_context`, so it should work (untested) | **CC BY-NC-SA 4.0** | **Only candidate for full-grad LMD/LSD** |
| **rCM** (NVlabs) | No (backward drops the tangent cotangent; silently wrong) | none, plus a forward-only rectangle "MagiMask" | yes / yes | explicit (q,k,v,tq,tk,tv) → (o,to), no `jvp()` | Apache-2.0 | Best **semigradient** option |
| jvp_flash_attention | No (returns None for tangents; silently wrong; HVP claim unbacked) | bool/additive but needs full `[B,H,N,N]` | no (reads out of bounds; bwd needs N%32) | only via `fwd_dual` | MIT | Not recommended |
| DMF | No | none | **no** (L % 128 required, garbage otherwise) | explicit + fwAD wrapper | **no license file** | Not usable |
| FastGen | no kernel; finite differences under `no_grad` | — | — | — | Apache-2.0 | FD recipe only, semigradient only, fp32 forwards |

**Note:** rCM, jvp_flash_attention and DMF all fail R1 in the same dangerous way. Plugged into
LMD/LSD, they would **train silently with wrong gradients** (the tangent path gets no gradient),
not raise.

## Recommendation

1. **TVM kernel + 3-group decomposition** for LMD full-grad and LSD.
   - It is the only implementation of backward-through-JVP.
   - Its toolchain is identical to ours (torch 2.7.1).
   - It handles unaligned lengths and Sq≠Skv.
   - Estimate from their benchmarks: ~6 ms/call against 47.6 ms, i.e. ~1.2 s off LMD's
     2.67 s/example on one GPU. *Unmeasured.*
2. **If we go semigradient** (`detach_derivatives: true`, 1.31 s/example and +9.7 GiB in §8):
   use the rCM forward-JVP kernel (Apache-2.0) under no_grad for the tangent, with fused masked
   SDPA for the primal. TVM's forward would work too.
3. **Free interim win:** turn on TF32 in the explicit path. ~15% faster (40.7 against 47.6 ms);
   tangent error 9.5e-4 against FP32.
4. **Not worth pursuing:** jvp_flash_attention and DMF (unfixable gaps for R1, plus alignment and
   licensing), and whole-model finite differences (bf16 cancellation; only works semigradient in
   fp32). An attention-level central difference with fp32 fused SDPA is a possible
   differentiable fallback, but it is slower than TVM (two fp32 fused calls) and has an O(ε²) bias.

## Validation plan (1 H100, ~30 min; not started)

Use the real q/k/v/mask captured by `scripts/profile_flowmap_step.py`, with TVM's `jvp_utils`
vendored under `jvp_kernel_analysis/` and **no change to training code yet**:

1. **Correctness under `forward_ad`.** 3-group TVM against the explicit FP32 path. Compare the
   primal, the tangent, and **all six gradients** of `loss(o, tō)`. Also check the tangent
   error against the 2.8% of bf16-explicit.
2. **Speed and memory per call**, against the explicit FP32/TF32 paths and fused SDPA.
3. **End to end:** patch `helpers/attention.py` in the profiler only. Measure `lmd_full` and
   `lsd_off` step time and memory against §8 (2.67 s / 2.45 s), running through our dual-aware
   checkpoint.

## Open questions for you

- **License:** TVM is CC BY-NC-SA 4.0 (non-commercial, share-alike). Is that acceptable for this
  project? If not, we would have to write the second-order backward ourselves: an estimated
  1.5–3 weeks (our own estimate, see `notes_rcm.md`).
- **Full-grad or semigradient?** That decides between TVM (R1) and the simpler Apache-2.0 rCM
  forward kernel.

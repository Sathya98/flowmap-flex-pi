# jvp_flash_attention (Alex Morehead): evaluation for flow-map training

Repo: `repos/jvp_flash_attention` (depth-1 clone, HEAD `f7100a8`). `J` = `jvp_flash_attention/jvp_attention.py`,
`T` = `tests/test_jvp_attention.py`. Everything below comes from reading the code. Nothing was run on a GPU.
Import check: the module imports under fm_env (torch 2.7.1 / triton 3.3.1). `HAS_TENSOR_DESC=False` there, so the TMA path is
never used and H100 also runs the generic `_attn_fwd` kernel.

## Verdict
**Useful only for the semigradient (tangent-detached) variant. It cannot serve LMD full-grad or LSD as shipped.**
The kernel does a fused Triton **forward + JVP** and a fused Triton **primal backward**. **Nothing backpropagates through
the tangent output.** Worse, the gradient loss is expected to be *silent* (details in §1). The fix for full-grad is a new
second-order backward kernel, which is a research-grade job (§4). For the semigradient path we would still have to pad
L=1670 up to a multiple of 32 and patch mask broadcasting, but both are easy.

## 1. What it computes and how it is exposed
- `JVPAttn(torch.autograd.Function)` has `forward` (J:2505), `setup_context` (J:2865), `jvp` (J:3066) and `backward`
  (J:3081). There is no `vjp` or double-backward. Wrappers: `attention = JVPAttn.fwd` (J:3229, primal only, passes
  `q_t=k_t=v_t=None`) and `JVPAttn.fwd_dual` (J:2973).
- `fwd_dual` calls `fwAD.unpack_dual` on q, k and v (J:3010-3012). It then passes the **dual tensors and the explicit
  tangents** to `apply` (J:3043-3060), because `forward()` receives the duals demoted to primals. It therefore **works with
  `torch.autograd.forward_ad` dual numbers**: the tests use `fwAD.dual_level()` + `make_dual` + `fwd_dual` (T:413-433, T:557-577).
  It also works under `torch.func.jvp` (T:595).
- Forward+JVP is one fused kernel (`_attn_fwd`, `ENABLE_JVP=True`). It computes o and
  o_t = P·v_t + P∘(s_t − rowsum(P∘s_t))·V (J:307-329, epilogue J:1041-1044). o_t is saved via `ctx.save_for_forward(o_t)`
  (J:2903), and `jvp()` just returns that saved tensor (J:3078). The incoming gq/gk/gv arguments are **ignored**.
- **README note on `fwd_dual` (README:65-67, issue #10).** `jvp_attention` (= `fwd`) never receives tangents, so the kernel
  runs with JVP off and `jvp()` returns `o_t=None`. `fwd_dual` has to be used so the tangents exist before autograd dispatch.
  What this means for us: every call site, including the recompute inside our dual-aware checkpoint, must call `fwd_dual`.
  With plain `attention` the tangent comes back as None/zero, with no error. Outside a dual level `fwd_dual` degrades to
  plain flash attention with the Triton backward, which is fine for non-JVP passes. Also, only `q_t is not None` gates JVP
  (J:2559). If k or v has no tangent, `None` pointers reach the kernel. Our cat'ed dual Q/K/V should always carry all three.
- **Backward.** `backward(ctx, do, _)` launches the fused Triton FA2-style primal backward (`_attn_bwd`, J:3171-3224) and
  returns **None for q_t, k_t, v_t** (J:3226, BwdOut J:2479-2494). o_t lives inside a non-tensor tuple output (J:2848-2862)
  and is created under the Function's no-grad forward, so it has **no grad_fn**.
  - As a result, a loss term through the tangent contributes **zero gradient** to q, k, v, their tangents and any upstream
    parameters.
  - This is almost certainly **silent**: autograd has no edge to complain about. It should be confirmed on GPU.
  - Backward-through-JVP is neither fused nor recomputed in PyTorch. **It does not exist.**
  - The tests only put the primal output in the loss: `loss_fn(sdpa_out/jvp_out, target)` (T:559-577, T:600).
- **HVP.** It is claimed (README:18, J:13, J:20), but there is no HVP test in the repo, and neither forward-over-reverse
  nor reverse-over-forward is possible with this Function. Treat the claim as unsupported for our purposes.

## 2. Masking, shapes, precision
- `attn_mask` (J:2533-2539, J:2576-2586, J:2624-2651):
  - **bool** (True = attend) or **additive** of q.dtype.
  - The shape must be **exactly (Z,H,N,N)** when `verify_attn_mask=True`. There is no broadcast.
  - The mask is made `.contiguous()` (J:2630), then indexed with the full z/h strides in fwd (J:~880) and bwd (J:2120).
  - Per-batch masks are supported. `causal` and a mask cannot be combined (J:2574).
- **Our [B,1,L,L]** fails the shape assert. With `verify_attn_mask=False`, a size-1 head dim gets stride L·L, so heads>0
  would read **out of bounds**.
  - Option 1: pass `mask.expand(B,H,L,L)`. `.contiguous()` then materializes 2·24·1696² ≈ 138 MB of bool per call, and it
    stays saved in ctx for backward.
  - Option 2: a one-line vendored patch that keeps stride_h=0. The fwd and bwd kernels already take strides.
- **Boolean mask math:**
  - Forward adds `MASK_CONST=-100` to the raw logits, then zeroes p where the mask is False (J:48-50, J:254-256, J:280).
    The tangent logits are also zeroed.
  - Backward also zeroes pT (J:~1535). Boolean masks are therefore exact.
  - Additive masks are *not* exact (−100 instead of −inf) and fail the L=32/64 accuracy tests (README:241-242, 369-370).
  - Masked tiles are still loaded and computed. There is **no block-sparse skipping** (our density is 0.76).
  - Rows with no True entries are handled (J:1025-1034). The verify step only asserts that no head is entirely False (J:2634).
- **Sequence length:**
  - Forward asserts only `N%2==0 and N>=32` (J:2569). The block-pointer loads have **no `boundary_check`** (J:243, J:944),
    and the key loop runs to N_CTX in 32-wide steps (J:237). L=1670 would read past the end of the tensors.
  - Backward hard-asserts `N_CTX % 32 == 0` (J:3142).
  - Fix: **pad to 1696** (53×32). Padded keys get mask=False; padded query rows get ≥1 True and are sliced off. Cost is ~3% more L².
- **Head dim:**
  - Allowed values are {16,32,64,128,256} with Q=K=V (J:2561-2568), so 128 works.
  - The README benchmarks use the test default **hd=64, 12 heads, bsz 2** (T:224-226). hd=128 is **not** covered by the
    published numbers.
- **Softmax scale:** `sm_scale` is optional and defaults to `hd**-0.5` (J:2589). Dropout raises an error (J:2550).
- **Tiling:** fixed `BLOCK_M=BLOCK_N=32`, `num_warps=4`, and autotune commented out (J:2841-2844, J:668-672). The code's own
  comment says BLOCK_M=32 "often leads to reduced numerical accuracy for longer sequences" (J:640-642).
- **Precision:**
  - q, k, v are bf16. Dot-product dtype is fp32 (J:805), so P·V and P·v_t run as fp32 (tf32) dots.
  - QK and the tangent QK dots take bf16 inputs with fp32 accumulation. `p_tqk` is cast to bf16 before the `g_acc` dot (J:325).
  - The accumulators m, l, acc, g_acc, mu and p_tv are fp32. o and o_t are stored in the input dtype.
  - In the backward, each tile's dv/dk partial product is rounded to bf16 before being added to the fp32 accumulator
    (J:1552, J:1564).
- **Tolerances:**
  - Primal and tangent use atol `bf16 3.2e-2` / `fp16 4e-3` / `fp32 2.35e-2`, rtol 1e-5 (T:688-693).
  - Grad and loss use atol 5e-4 (T:525-527).
  - An assert failure is only *printed* (T:665). The ✓/✗ column compares max_error against the tolerance.
  - Reported bf16 max error: **1.56e-2** with a boolean mask at L=1024/2048 (README:317, 329); 3.9e-3 with no mask.
  - The tests use 90%-masked random masks and hd 64. The GPU model is not stated.

## 3. Speed/memory claims (README:245-371, bf16, bsz2, 12 heads, hd64, non-causal, bool mask)
- **What is timed:** `fwd_dual` **forward+JVP only**, without backward (T:770-780). The baseline is
  **SDPA MATH under forward AD** (T:756-768), not fused SDPA.

| L | SDPA-math ms / MB | jvp_attn ms / MB | TFLOP/s |
|---|---|---|---|
| 1024 | 5.38 / 595 | **0.63 / 34** | 20.7 |
| 2048 | 21.6 / 2244 | **1.93 / 66** | 27.3 |

- Without a mask, L=2048 takes 1.14 ms (46 TFLOP/s). The mask costs 1.5-1.7×.
- The fp32 plot (`float32_time_scaling.png`) shows 4.3×/6.0× speedups for bool masks at 1024/2048.
- "Loss/speed matching" training plots exist (README:73-83), but they come from an external model.
- **Our shape (2×24×1696×128)** is ≈2.7× the FLOPs of the 2048 row, so the naive estimate is **~5 ms fwd+JVP per call**.
  The fused primal backward comes on top of that (32-wide tiles, likely 2-3× the forward).
  - 27 TFLOP/s is only a few % of H100 bf16 peak. The kernel is memory-efficient but not fast.
  - That still beats our 47.6 ms FP32 materialized fwd+bwd, but it is far from fused SDPA (1.5 ms).
  - None of this has been measured on our hardware.

## 4. Gaps for our use case, and the effort to close them
| Gap | Needed for | Effort |
|---|---|---|
| **No grad through o_t** (w.r.t. q, k, v, q_t, k_t, v_t); silent zero | LMD full-grad, LSD | **High.** Needs a new second-order backward kernel: dO_t → (dq, dk, dv, dq_t, dk_t, dv_t), with extra row stats (mu, Δ_t) and ~2× the tangent accumulators. Est. 2-4 weeks for a Triton-experienced dev plus a gradcheck harness. Stopgap: a custom Function whose backward recomputes in chunked PyTorch, which is roughly the cost of our current fallback, but only in bwd. |
| Guard against silent tangent-grad drop | all | Trivial: a wrapper that raises if grad is enabled and the tangents require grad (outside semigrad mode) |
| L=1670 not a multiple of 32 (fwd OOB, bwd assert) | all | Easy: pad to 1696 with the mask (~1 h) |
| [B,1,L,L] mask not broadcast | all | Easy: vendored patch for stride_h=0, or expand (+138 MB/call) |
| No block-sparse skipping for block-structured mask | perf | Medium: a per-tile "any True" bitmap and early skip |
| Fixed 32×32 tiles, no autotune; hd128 untested in bench | perf/accuracy | Medium: re-enable autotune and validate hd128/bf16 vs our FP32 reference |
| `pyproject` pins torch>=2.8 (we run 2.7.1) | install | Vendor the single file (MIT); do not pip install |
| Must use `fwd_dual` in the checkpoint recompute too | correctness | Easy |

For the **semigradient** variant (forward JVP only, tangent detached), the table's "easy" rows are enough.
Validate on GPU against the FP32 reference at hd128 / L=1696 / bool mask before adopting.

## 5. License / distribution
- **MIT**, © 2025 Regents of UC / LBNL. It carries a DOE government-use notice (LICENSE, README:499-516). Vendoring is fine.
- The PyPI package is `jvp_flash_attention`, published by the GitHub release workflow (`.github/workflows/publish.yaml`).
  The repo version is **0.14.0** (pyproject.toml, CITATION). The live PyPI index was not checked: pip is unavailable in
  fm_env and there is no network lookup.

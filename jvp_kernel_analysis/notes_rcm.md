# NVIDIA rCM FlashAttention-JVP Triton kernel: evaluation for Flex-π flow maps

Repo: `repos/rcm` (depth-1 clone). Kernel: `rcm/utils/flash_attention_jvp_triton.py` (FA2 Triton tutorial +
JVP, Kaiwen Zheng 2025/03). The only caller is `rcm/utils/jvp_helper.py:24,56`. Static read only: no GPU was run,
and the module does not import here because `flash_attn` is missing from fm_env (hard import at `:53`). AST parses; triton 3.3.1.

## 1. What it computes
- **Forward pass with an attached forward JVP.** The inputs are `(q,k,v,tq,tk,tv)` in `[B,H,L,D]`, and the outputs are `(o, to)` plus an fp32 LSE
  (`:106-229`). `to = (P·tV + (P⊙tS)·V − diag(rowsum(P⊙tS))·O)/l` with `tS=(tQKᵀ+QtKᵀ)·scale` (`:193-218`).
- It is exposed as a plain `torch.autograd.Function` `_attention` that takes the tangents as ordinary tensor arguments (`:447,460`).
  There is **no `jvp()` staticmethod**, so it is not a forward_ad or torch.func rule. You call it with explicit primal and tangent tensors.
- **`backward()` covers the primal output only.** `backward(ctx, dout, *args)` (`:532`) silently drops the cotangent of `to`. It calls
  flash-attn's `_flash_attn_backward` for dq/dk/dv (`:539-575`) and returns `None` for tq/tk/tv (`:576`). Tensors are saved only if q, k or v
  requires grad (`:461,497`). With a MagiMask, any grad raises (`:501-502`).
- rCM itself only ever uses it without grad. The caller detaches `to` (`jvp_helper.py:66`), and the whole tangent pass runs under
  `torch.no_grad()` (`t2v_model_distill_rcm.py:568-583`; causal `t2v_model_causal.py:990`). The primal with grad is a **separate** plain forward
  (`t2v_model_distill_rcm.py:585`). The sCM loss uses the tangent only inside a stop-grad target (`:590-613`). **This is exactly our semigradient case.**

## 2. Masking, shapes, dtypes
- **Dense path (`_attn_fwd`):** no mask, non-causal, one shared KV length. Unaligned lengths are handled with row and column bounds masks (`:142-143,161,171,194`),
  and tests cover L=999, 515 and 1000.
- **MagiMask path (`_attn_fwd_magi`, `:241-406`):** the mask is a union of rectangles. You pass q-ranges and k-ranges, and they are merged into CSR form
  (`magimask.py:126-283`). The q-groups must partition `[0,L)` and each q-group needs at least one k-range (`magimask.py:152,270-273`). Boundaries are
  exact to the token, and non-aligned edges go through `n_mask`/`m_in_slice` (`:289,335`). Fully-masked rows are guarded (`:352,389`). The ranges are
  **shared across batch and heads**, and this path is **forward-only**. rCM builds it only for block-causal and teacher-forcing specs
  (`magimask.py:10-123`), but the kernel accepts arbitrary ranges. There is **no arbitrary boolean or additive mask**.
- **Softmax scale:** configurable, default `D^-0.5` (`:472`). **Head dims:** {16, 32, 64, 128, 256}, and d_qk may differ from d_v in the forward (`:468-469`).
  Backward requires d_qk = d_v (`:534`). Tangent strides must equal primal strides (`:471`).
- **Dtypes:** outputs are stored in `q.dtype`, so `to` is bf16 when the inputs are bf16 (`:474-475`). Accumulation is fp32 (`:150-153`). P and H=P⊙tS are cast
  to the input dtype (bf16) before the PV/HV dots (`:201-202`), as in standard FA2. The epilogue `A+B−μO` runs in fp32 but involves cancellation.
  The bf16 tests use `atol=2e-2, rtol=1e-2`.

## 3. Speed, accuracy and constraints
- The repo publishes no numbers. There is only a `triton.testing` TFLOPS benchmark against FA2 (B=4, H=32, D=64, L=1k-16k; `:659-722`), which counts
  the JVP forward as 3× the FA-forward FLOPs (6 matmuls, `:719-721`).
- Tests are in-file and run from `__main__` only (`:582-937`): fwd/bwd, JVP, cross-attention, and Magi JVP against `torch.func.jvp` on a naive
  reference. There is also a network-level test, `rcm/networks/wan2pt1_jvp_test.py:94-137` (pytest L1). That file carries an **NVIDIA-proprietary header**
  (`:1-12`), unlike the Apache-2.0 kernel.
- **Autotune:** 48 configs (BM∈{64,128} × BN∈{16..128} × stages{3,4,7} × warps{4,8}, `:95-104`). The key includes `SEQ_LEN_*`, which are `tl.constexpr`,
  so **each distinct L triggers a recompile plus a full autotune sweep**. A fixed L≈1670 is fine; a varying L is expensive.
- **Hardware:** there is no sm90-specific code (no TMA or explicit wgmma). The Magi path picks BLOCK_M=64 on cc90 and 128 otherwise (`jvp_helper.py:265`).
  The line "only works on post-Ampere GPUs" (`:933`) is inherited from the Triton tutorial benchmark, and its truth on A100 is unverified. Three fp32
  [BM,Dv] accumulators at D=128 create register pressure, so autotune will likely favour BM=64.
- The backward path needs `flash-attn>=2.7.0.post1` (`:53,70`). It is imported at module level even when you only need the forward.
- **Estimate for our shape (unmeasured):** a JVP forward costs about 2.5-3× an FA forward, so roughly 1.5-3 ms per call. That would be one order of magnitude
  below the 47.6 ms FP32 fallback.

## 4. Porting effort (honest)
**(a) Arbitrary boolean mask, forward JVP only: small, about 1-2 days including tests.**
- Add a mask pointer and strides to `_attn_fwd`, load the `[BM,BN]` tile, set `qk=-inf` and `tS=0` where the mask is off, and keep the `m_ij_safe`/`l_safe`
  guards from the Magi kernel. That is about 30 lines.
- Optional: add a per-tile full/empty/partial map to skip empty tiles. At density 0.76 this saves at most about 24%.
- Zero-kernel-change alternative: if our m_in/m_out mask is a union of stream-block rectangles, it can be expressed as MagiMask ranges, called per sample
  (B≤2) because the ranges have no batch dim. Verify the rectangle decomposition first.

**(b) Backward through the JVP: missing, and it is real work, about 1.5-3 weeks for someone fluent in Triton plus a numerics validation pass.**
- You need gradients of `<do,o> + <dto,to>` with respect to q, k, v, tq, tk and tv.
- Useful split: with respect to (tq,tk,tv), the gradient is just the attention VJP at the primal point with cotangent `dto`, so a masked flash backward
  can be reused for it.
- With respect to (q,k,v), you additionally need the **JVP of the attention backward** (second-order, Hessian-vector-like terms). This needs a new FA-style
  backward kernel that recomputes P, tS and H from the LSE, with about 2-3× the matmuls of an FA backward and extra row statistics
  (rowsum(dto⊙to), rowsum(dto⊙o), μ). That means dq and dkv passes, with masking from (a).
- Before building it, check `amorehead/jvp_flash_attention`, which claims backward-through-JVP support (docs/jvp-kernels.md).

## 5. License
`LICENSE.txt` is **Apache-2.0**, and the kernel file carries an SPDX Apache-2.0 NVIDIA header (`:1-14`) that credits OpenAI's Triton tutorial.
We can vendor it if we keep the notice. Do **not** copy `wan2pt1_jvp_test.py`, which has a proprietary header.

## 6. rCM's layer-by-layer JVP
- Every module subclasses `JVP` (`jvp_helper.py:34-49`), and `forward(..., withT=True)` dispatches to `_forward_jvp`, which threads explicit `(x, t_x)` tuples.
- Each non-attention piece is a **small `torch.func.jvp` over a pure closure**:
  - qkv projection and norms (`wan2pt1_jvp.py:317`)
  - o-proj (`:330`)
  - AdaLN pre-modulation (`:579`)
  - gated residual (`:593`)
  - norm3 (`:596`)
  - cross-attention and FFN (`:608`)
  - head (`:659`), patch embedding (`:1033`) and time embedding (`:1043`)
- Attention goes through the Triton kernel (`jvp_helper.py:52-66`) and RoPE through `apply_rope_with_tangent` (`rope.py:113`). Blocks are chained in a
  plain loop (`wan2pt1_jvp.py:1088`).
- Tangents are `.detach()`ed after every piece (e.g. `:319,331,589,594,609`), so the tangent never carries an autograd graph.
- Motivation:
  - FSDP2 compatibility: parameter all-gather hooks fire per layer instead of inside a whole-model functorch transform.
  - Ulysses context parallelism: an all-to-all of (x,t) pairs (`jvp_helper.py:69-122`).
  - Per-block selective activation checkpointing (SAC) (`wan2pt1_jvp.py:1202-1213`).
  - A slot where a custom kernel can plug in.
- Cost: the network runs twice, a no-grad JVP pass plus a primal pass with grad. By construction it is **semigradient-only**.
- **Relevance to us: low.** forward_ad dual numbers already propagate layer by layer with no whole-graph transform. Our dual-aware checkpoint
  (`src/flexpi/models/helpers/checkpoint.py:33-56`) covers memory, and ZeRO-2 does not shard parameters.
- Two ideas do transfer:
  - For semigradient objectives, run the dual (tangent) pass under `no_grad` and the primal with grad separately, as rCM does. This avoids saving the tangent graph.
  - The unpack_dual → explicit-tangent Function → make_dual pattern (the same one our checkpoint already uses) is how to plug in a tangent-as-argument kernel,
    because there is no custom `jvp()` rule.

## Verdict
- **Semigradient (tangent detached): adopt, with a small port.**
  - Vendor only `_attn_fwd`, add a bool-mask tile load (or MagiMask ranges), and drop the flash_attn import.
  - In `helpers/attention.py`'s dual branch, call the kernel on unpacked primals and tangents under no_grad to get `to`. Compute the primal `o` with fused
    masked SDPA (with grad), then `make_dual(o, to.detach())`.
  - Expected result (unmeasured): about 3-5 ms per call against 47.6 ms.
  - Validate the bf16 `to` error at L=1670 against the FP32/FP64 reference first, since our current path is FP32 for a reason.
- **LMD full-grad / LSD (backward through the tangent): not usable as-is.**
  - `backward` ignores `dto` and returns no tangent gradients. **Using it naively would give silently wrong gradients**, not an error.
  - Making it usable needs a new second-order backward kernel (see 4b). Evaluate amorehead/jvp_flash_attention before writing one.

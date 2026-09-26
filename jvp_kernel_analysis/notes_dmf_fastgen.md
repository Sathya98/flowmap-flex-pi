# DMF (Decoupled MeanFlow) and NVIDIA FastGen: attention-JVP paths

- DMF: `repos/dmf`, commit a6db0f6 (2025-10-29). Paths below are relative to `repos/dmf`.
- FastGen: `repos/FastGen`, commit f7d8456 (2026-08-20). Paths below are relative to `repos/FastGen`.
- Everything here comes from reading the code. Nothing was built or run on a GPU.

## DMF

### 1. Where the JVP lives and what it computes
- `models/flash_attention_3_jvp.py` and `models/flash_attention_2_jvp.py` are near-identical
  (`diff` confirms it). Both wrap one vendored Triton kernel in `models/triton_utils.py`.
- `FlashAttnFunc` (`fa3_jvp.py:163`, `fa2_jvp.py:177`) is a new-style `autograd.Function`: its
  `forward` takes `(q, k, v, tq, tk, tv, scale)`. It is not a `jvp()`-driven design.
  - **With tangents** (`fa3:172-217`), it runs the Triton `_attn_fwd_dual` kernel
    (`triton_utils.py:258`). That kernel produces the primal `o`, the LSE `m` and the tangent
    `to` in one online-softmax pass (inner loop `:72-126`, tangent recurrence `:110-118`). FA3/FA2
    is **not** used for this forward.
  - **Without tangents** (`fa3:218-239`), it calls the FA3/FA2 CUDA forward.
- `setup_context` stashes `to` via `save_for_forward` (`:252`). `jvp()` (`:280-283`) just returns
  it.
- How the model calls it: the wrapper `flash_attn_func` (`fa3:295-320`) calls
  `fwAD.unpack_dual` on q/k/v, passes the primals and tangents explicitly, and re-wraps the result
  with `fwAD.make_dual(out, t_out)` (`:314`). The model reaches it through
  `layers.py:86-99 attn_op(op="fa2"|"fa3")`. `loss.py:170` drives it with
  `torch.func.jvp(model_fn, ...)`.
- **Backward** (`fa3:257-277`) runs only the FA3/FA2 CUDA backward on the primal path, using the
  LSE that Triton produced.
  - It **ignores the incoming grad of `to`** (`*args`).
  - It returns `None` for tq/tk/tv (`fa3:277`, `fa2:276`).
  - So there is **no reverse mode through the tangent**. A loss that depends on `du/dt` with grad
    gets a silently wrong (dropped) gradient, not an error.
- DMF never needs that gradient: its target is `u_tgt = (v + (r-t) du_dt).detach()`
  (`loss.py:171-172`), a pure semigradient.

### 2. What it is built on, and build requirements
- **"FA3+JVP"** means the Dao-AILab FA3 hopper build for the forward (no JVP) and the backward
  (`import flash_attn_3._C`, `torch.ops.flash_attn_3`, `fa3:16-17`), plus the vendored Triton kernel
  for the dual forward.
  - It is not a CuTe DSL kernel and not a fork of the hopper sources.
  - FA3 is an **external dependency**, built from source (README §1). It is sm90 only.
  - The README recommends CUDA 12.8 and torch 2.8.0. The raw 33-argument `fwd` call signature
    (`fa3:72-107`) is tied to a specific FA3 API revision (post-2.8 `attention_chunk` /
    `scheduler_metadata`).
- **"FA2+JVP"** is the same design with `flash_attn_2_cuda` (pip `flash_attn`, unpinned in
  `requirements.txt`; sm80 and later).
  - It re-registers `flash_attn::_flash_attn_forward/_backward` custom ops (`fa2:32,91`). These
    names collide with the ones the upstream `flash_attn_interface` registers if both are
    imported.
- `layers.py:23-24` imports **both** modules unconditionally, so both packages must be installed.
- The Triton kernel (`triton_utils.py:1-12`) is derived from the Triton fused-attention tutorial
  plus the Ryu1845 and Birch-san JVP gists. Triton is not pinned.
  - It uses `make_block_ptr` / `advance` with a fixed config of block 128×128, 2 stages and
    8 warps (`fa3:174`), with no autotuning.

### 3. Masking, shapes and dtypes
- **No masking at all.** `causal=False` is hard-wired (`fa3:232,268`). The header says
  "non-causal only". The dual kernel has no mask or bias input.
- **The sequence length must be a multiple of 128 (for both Q and KV).**
  - `grid = (l_q // block_q, ...)` (`fa3:180`) truncates, so for L=1670 the last 6 query rows of
    `o`, `to` and `m` stay `torch.empty` garbage.
  - The KV loop loads full 128-row tiles with no sequence `boundary_check` and no −inf padding
    (`triton_utils.py:64-69,85-90`; the Q load at `:307,332`). The last KV tile therefore reads
    out of bounds, and those reads are included in the softmax.
  - The tests only cover L ∈ {256, 1024, 2048} (`fa3:341`).
- **Head dims:** 64, 72, 96 and 128 are tested (`fa3:343`). `split_head_dim(128)` gives
  `(128, 0)`, so d=128 takes the unsplit kernel.
- **Dtype:** tested on bf16 only (atol 1.2e-3 against naive attention, `fa3:345-377`). The
  tangent is accumulated in fp32 and stored in the input dtype.

### 5. Claims and license
- No kernel benchmark is published.
- The README says `--qk-norm` was needed at 512px "due to instable JVP computation".
- **No LICENSE file in the repo**, so the default is all rights reserved. The kernel draws on the
  MIT Triton tutorial and on gists with no stated license. Vendoring needs the authors'
  permission.

### Verdict for us: not usable as is
- It fails **three hard requirements**:
  1. No arbitrary mask. Our dense block-structured [B,1,L,L] mask is essential for the regimes
     that `m_in`/`m_out` select.
  2. L≈1670 produces garbage and out-of-bounds reads.
  3. There is no backward through the tangent. LMD full-grad and LSD would be silently wrong.
- For semigradient work it is at best a ~500-line reference for the tangent online-softmax
  recurrence (`triton_utils.py:92-118`). The TVM kernels (`notes_tvm.md`) already cover that,
  plus the reverse pass.

## FastGen

### 1. Where the JVP lives
- There is **no fused attention-JVP kernel.** The JVP options are:
  - **Autodiff:** `torch.func.jvp` over the whole network (`mean_flow.py:317-323`,
    `sCM.py:180`). It runs inside `temp_disable_efficient_attn` (`mean_flow.py:24-45`), which
    forces the **math SDPA** backend, meaning materialized attention.
    - DiT disables timm fused attention for this (`networks/DiT/network.py:251`).
    - EDM uses a custom `AttentionOp` with a `jvp()` over the materialized softmax
      (`networks/EDM/network.py:157-196`).
  - **Finite difference** (below).
- **Both paths run under `@torch.no_grad()`** (`mean_flow.py:293`, `sCM.py:151`). There is an
  explicit `assert not u_theta_jvp.requires_grad` (`mean_flow.py:486,605`).
- A separate gradient-carrying forward then computes `u_theta` (`mean_flow.py:609`).
- **Every FastGen objective is semigradient.** No backward through the JVP exists anywhere.

### 4. How the finite-difference JVP is done
- **MeanFlow and AnyFlow** (`mean_flow.py:235-291`):
  - It computes the directional derivative along `(dx_t/dt, dt=1, dr=0)`: two extra no-grad
    forwards at `t±ε`, `x_t ± ε·dxt_dt`, with `r` fixed.
  - It uses **central** differences, with a one-sided fallback where `t±ε` leaves `[min_t, max_t]`
    or `t−ε ≤ r` (`:254-274`; test at `tests/test_anyflowmodel.py:149-166`).
  - The step arithmetic is float64 (`:245-251,284-289`).
  - `fork_rng` keeps the dropout masks identical across the two evaluations (`:282-285`).
- **sCM** (`sCM.py:113-141`) uses a central difference in TrigFlow angle with a relative step
  `ε_t = eps·|t|`, clamped to at least 1e-6.
- **Step sizes:**
  - Defaults: `jvp_finite_diff_eps = 1e-4` for MeanFlow (`config_mean_flow.py:73`) and `1e-3` for
    sCM (`config_scm.py:63`).
  - The Wan runs use **5e-3** (t ∈ [0,1]; "reference ε=5 in 1000-step units"): see
    `WanT2V/config_mf.py:49-50`, `config_anyflow.py:64-66` and
    `config_anyflow_onpolicy.py:121-122`.
- **Precision:** the Wan configs keep **bf16 weights** (`config_mf.py:26`, `config_anyflow.py:47`)
  but set `precision_amp_jvp="float32"`.
  - `_jvp` casts `x_t` to fp32 (`mean_flow.py:302-303`) and runs both FD forwards under
    fp32 autocast (`:309-313`).
  - So the **difference is taken between fp32-compute forwards, never bf16**.
  - TF32 is on by default (`configs/config.py:214`, `utils/scripts.py:37-44`), so matmuls are
    TF32 unless the run disables it.
- **Who uses FD:** only the Wan-1.3B T2V MeanFlow and AnyFlow configs. DiT and EDM use autodiff
  JVP in fp32 (`DiT/config_mf_b.py:20`, `EDM/config_scd_in64.py:23`).
- **Stated reason:** "for compatibility with Flash Attention and FSDP"
  (`methods/consistency_model/README.md:36,82`). There is no benchmark or accuracy study.

### Is FD viable for us?
- **Pure bf16:** no. The FD error is about `c·u·|f|/ε + O(ε²)`.
  - bf16 has u≈3.9e-3. At ε=5e-3 the rounding term is O(1) relative to `du/dt`: pure noise.
  - TF32 (u≈4.9e-4) still gives about 10% noise, amplified by depth. Only true fp32 (u≈6e-8)
    is clean.
  - That matches our existing bf16 rejection.
- **Semigradient (tangent detached), whole-network FD:** viable only FastGen-style.
  - Recipe: fp32-autocast forwards (TF32 off, ideally), central difference, ε≈1e-3–5e-3 in t,
    one-sided at the boundaries, `fork_rng`.
  - Cost is 2 no-grad fp32 forwards of the 6B model per step. The fused bf16 SDPA stays usable
    because FD needs no JVP.
  - Our per-stream timesteps are compatible: perturb each noised stream along its own `dx/dt`,
    and leave clean or given streams fixed.
- **Full-grad LMD/LSD: not viable.**
  - Reverse mode through FD needs both perturbed forwards kept with their graphs (2× activations)
    and two backwards. The parameter gradient is `(∂θf(x₊) − ∂θf(x₋))/2ε`, which is the same
    cancellation now happening in the backward pass.
  - The upstream `∂L/∂tangent` is also scaled by 1/ε.
  - A clean result would need strict-fp32 fwd+bwd of 6B (non-tensor-core FP32 is roughly 15×
    slower than bf16 on H100), and it adds an O(ε²) gradient bias.
- **Untested idea: FD at the attention op only.**
  - Tangent: `[SDPA_fp32(q+εtq, k+εtk, v+εtv, mask) − SDPA_fp32(q−ε·)]/2ε`, with ε scaled by
    ‖q‖/‖tq‖ (≈ u_fp32^(1/3) ~ 4e-3 → relative error ~1e-5).
  - It is built from differentiable fused SDPA calls, so it keeps reverse mode through the tangent.
    It supports our arbitrary boolean mask and unaligned L (the mem-efficient backend accepts
    fp32 plus `attn_mask`; mask padding still needs checking). It also stays inside our fwAD
    dual wrapper (`make_dual(out, tangent_with_grad)`).
  - Cost is 2 fp32 fused fwd+bwd per call instead of 47.6 ms materialized. Worth a quick
    benchmark and an accuracy check against the FP32 materialized reference. The fp32
    mem-efficient speed on sm80/sm90 is unknown.

### 5. License
- **Apache-2.0** (`LICENSE`), with third-party licenses under `licenses/`. Vendoring is
  unproblematic, but there is no kernel worth vendoring.

### Verdict for us: nothing to reuse for full-grad
- It has no fused attention JVP, and every path is semigradient under `no_grad`.
- Its useful content is the **FD recipe** for semigradient ablations: fp32-autocast forwards,
  central difference, ε=5e-3, boundary fallback, `fork_rng`.
- It is also further evidence that production-scale video MeanFlow (Wan) avoids fused-attention
  JVPs entirely rather than solving them.

# Terminal Velocity Matching (Luma AI): JVP-SDPA Triton kernels

- Paper: Zhou, Parger, Haque, Song, *Terminal Velocity Matching*, arXiv 2511.19797.
- Code: https://github.com/lumalabs/tvm, cloned at `repos/tvm` (commit 331c01d, 2026-02-13).
- Kernel code: `repos/tvm/jvp_utils/`.
  - `functional.py`: autograd wiring.
  - `ryu_triton.py`: fused forward + JVP.
  - `flash_attn_triton.py`: Tri Dao's FA2 Triton backward.
  - `flash_jvp_backward.py`: backward through the JVP.
- License: **CC BY-NC-SA 4.0** (`repos/tvm/LICENSE`): non-commercial and share-alike.
  Fine for academic research, but any derivative (a masked or ported kernel included) must
  stay NC-SA. Check this with the project before vendoring it into the repo.

## What it computes, and how it is wired (`functional.py`)

- `SDPAFunction` (`functional.py:227`) is a `torch.autograd.Function` with `forward`, `jvp` and
  `backward`.
  - `forward` only **allocates** `y` and the LSE buffer `M`.
  - `jvp()` (`:249`) runs one fused Triton kernel. It fills `y` **and** computes the output
    tangent `t_y` (`ryu_triton.py:88`, `_flash_attention_jvp_multihead_kernel`, online softmax,
    no L×L).
  - Consequence (README): **calling it without an active JVP silently returns garbage**,
    because `y` is never filled.
- The tangent comes out of `SDPAJVPForwardFunction.apply` (`:49`). That is a reverse-mode
  Function whose `backward` (`:74`) calls `flash_jvp_backward._flash_attn_backward`, a fused
  Triton kernel. It returns grads for **q, k, v, t_q, t_k, t_v** from `d t_y`.
- The primal output's gradient goes through `SDPAFunction.backward` → the standard FA2 Triton
  backward. Autograd sums both paths.
- So this is **full reverse-over-forward**: backprop through the JVP, which is exactly what
  LMD full-grad and LSD need (`detach_derivatives: false`).
- TVM calls it through `torch.func.jvp` (`training/wrapper.py:100`, the whole DiT inside
  `func.jvp`). We use `torch.autograd.forward_ad` duals. `Function.jvp` + `setup_context` is
  also what `forward_ad` dispatches to, so it should work with our `tuple_jvp` and the
  dual-aware checkpoint. **Untested; it needs a GPU check.**

## Capabilities and limits

| | |
|---|---|
| Mask | **none** (README: no masking, no causal, no custom scale) |
| Scale | fixed `1/sqrt(D)` (`ryu_triton.py:325`). Matches ours. |
| Sq ≠ Skv | yes (separate `L`, `L_kv`; backward has `seqlen_q/seqlen_k`) |
| Unaligned lengths | yes: boundary masks in forward (`ryu_triton.py:181-224`); `EVEN_M/EVEN_N` masks in backward (`flash_jvp_backward.py:143-235`). `L_kv` is `tl.constexpr`, so there is one compile per distinct KV length. |
| Layout | `[B, H, S, D]`. Our helper already transposes to this. |
| Dtypes | fp16 / bf16 ("not tested with FP32") |
| Head dim | ≥ 16, padded to a power of two. 128 is fine. |
| Autotune | forward: a single config `BLOCK_M=64, BLOCK_N=32, num_warps=8` (`ryu_triton.py:73-83`) |
| Hardware | generic Triton, not Hopper-specific (no TMA/wgmma), so it should run on A100 and H100 |
| Toolchain | env pins **torch 2.7.1** (`env.yml`), i.e. Triton 3.3.1. **Identical to ours.** |

## Their benchmarks (README, B=1, hd=128, bf16)

The fused forward+JVP at H=24, S=1024 takes 1.01 ms (vanilla 1.13); at S=4096, 4.25 ms
(vanilla 10.6).
The backward through the JVP at H=24, S=1024 takes 1.85 ms (vanilla 2.03); at S=4096, 22.9 ms
(vanilla 24.5).
The main win is **memory**: at H=24, S=4096 it uses 594 MB against 10.7 GB. Speed is ≈ vanilla
because the backward spills registers (their note).
Paper: up to 65% speedup on H100 end to end, and JVP ≈ 3× a normal step.

**Scaled to us** (work ≈ 2.1 M query-key pairs per call after the decomposition below, ≈ 2× S=1024):
roughly 2 ms forward+JVP and 4 ms backward per call, **~6 ms against our 47.6 ms** explicit FP32
path (§8 of 07-efficiency-notes). That is ~40 ms × ~30–60 attention calls, i.e. **~1.2 s off
LMD's 2.67 s/example**. *Estimate*: their "vanilla" is bf16 math SDPA, while ours is FP32
explicit with a mask.

## Our mask does not need a masked kernel (key finding)

The flow-map configs use full joint with `flex_joint.enabled=false` and all joint flags true,
so there are no XOR drops (`flexpi.py:2769-2777`). The action rows are widened to every stream
(`:2780+`). Video uses `first_frame_causal` (`wan_video_dit.py`: first-frame rows cannot see
later frames; later frames see everything). Visual rows never see action. The rows therefore
fall into **three groups, each attending to one fixed set of columns**:

| Rows (queries) | Columns (keys) | LIBERO sizes |
|---|---|---|
| anchors: ff_v, ff_d, ff_p | anchors only | 595 × 595 |
| futures: rem_v, rem_d, rem_p | all visual | 1043 × 1638 |
| action | everything | 32 × 1670 |

Predicted mask density: (595² + 1043·1638 + 32·1670) / 1670² = **0.7585**. §8's captured mask
measured **0.7587**. The mask *is* this structure.

So the masked joint attention equals three **mask-free** calls on gathered rows/columns. Gather
q/k/v with `index_select`, which supports forward AD, and scatter the outputs back. The work is
identical to the masked computation (no wasted tiles). The HBridge outer layers are per-stream
masks of the same form (anchors → anchors; futures → the whole stream), so they are 1–2
mask-free calls each.

Caveat: this holds for the full-joint flow-map configs. Flex-joint training with per-sample
presence/joint flags would give per-sample masks and would need a real masked kernel or a
per-sample grouping.

## Integration sketch (not implemented)

`helpers/attention.py`, when a forward-AD level is active:
1. Take the static group index tensors, derived once from the mask (or from the stream token
   counts).
2. Gather q/k/v per group and call `sdpa_jvp`.
3. `index_copy` the outputs back.

The fused path is only reachable from under a forward-AD level, so the "garbage without JVP"
hazard cannot trigger. Requirements to check:
- Every q/k/v must carry a tangent in our graph. They all depend on t through the time
  modulation.
- The checkpoint recompute runs the Function again under `dual_level` (`helpers/checkpoint.py`).

## Risks and open items

- **Correctness with `forward_ad` + our checkpoint: must be tested.** Compare against
  `helpers/attention.py` for primal, tangent and **all six gradients** (reverse-over-forward).
- **Accuracy:** bf16 I/O with fp32 accumulation, against our FP32 reference. The tangent enters
  the loss; §8 found bf16-explicit had a 2.8% tangent error, so measure this kernel's.
- **Register spills in the backward** (their note). A100 vs H100 behaviour is unknown, and the
  forward has a single autotune config.
- **License (NC-SA).**
- Their architectural changes for JVP stability (QK-RMSNorm, parameter-free RMSNorm,
  AdaLN normalization, Lipschitz init) are **model changes**, independent of the kernel. They
  are not needed to use the kernel, but they are relevant if LMD/LSD training turns out
  unstable.
- Relation to our objectives: TVM differentiates w.r.t. s and keeps gradients through the JVP
  term, i.e. the ESD/Eulerian-flavoured family. They report detaching the JVP cut per-step time
  from 0.95 to 0.69 s, the same trade-off as our `lmd_semigrad` (1.31 against 2.67 s).

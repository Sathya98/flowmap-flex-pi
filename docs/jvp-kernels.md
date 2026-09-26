JVP through Attention for DiTs / Flow Maps
Implementations
NVIDIA rCM — FlashAttention-2 JVP Triton kernel
Exact forward-mode JVP through attention without materializing the full (N \times N) attention matrix.
Probably the most directly useful reference implementation.
https://github.com/NVlabs/rcm/blob/main/rcm/utils/flash_attention_jvp_triton.py
jvp_flash_attention — Alex Morehead
Standalone Triton implementation of FlashAttention with JVP support.
Integrates with torch.func.jvp.
Also intended to support higher-order derivatives / backward through JVP.
https://github.com/amorehead/jvp_flash_attention
Decoupled MeanFlow (DMF)
Flow-map training for pretrained DiTs.
Includes custom FA2 + JVP and FA3 + JVP paths.
FA3 path is particularly relevant for H100/H200.
https://github.com/kyungmnlee/dmf
NVIDIA FastGen
Includes finite-difference JVP as an alternative when fused forward-mode AD is inconvenient.
Useful baseline: two ordinary FlashAttention forwards can be faster than unfused exact JVP.
https://github.com/NVlabs/FastGen
Relevant papers / methods
sCM — Simplified / Stabilized Consistency Models
One of the key works deriving a FlashAttention-compatible attention JVP.
\operatorname{diag}((P\odot\dot S)\mathbf 1)O
]
Allows tiled computation without storing the full attention matrix.
ICLR 2025:
https://proceedings.iclr.cc/paper_files/paper/2025/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html
Decoupled MeanFlow
Directly relevant if we're turning a pretrained flow DiT into a flow map.
Uses JVP targets with stop-gradient, meaning only a forward-JVP kernel is required.
https://github.com/kyungmnlee/dmf
rCM
Scalable consistency/flow-map training.
Important engineering idea: compute JVP layer-by-layer rather than wrapping the entire DiT in one giant torch.func.jvp.
Particularly useful with FSDP.
https://github.com/NVlabs/rcm
FACM / Chain-JVP
Decomposes full-model JVP into per-layer JVPs.
Designed to avoid expensive distributed/FSDP communication caused by whole-model forward AD.
Relevant for very large DiTs / video models.
Paper:
https://arxiv.org/abs/2510.08431
Terminal Velocity Matching
Relevant if gradients need to propagate through the JVP itself.
Requires something stronger than a forward-only JVP kernel: effectively backward through the fused JVP operation / second-order differentiation.
Upstream issues / current limitations
PyTorch fused SDPA does not generally support torch.func.jvp
This can force attention onto the unfused math implementation.
That path materializes the attention matrix and becomes very slow / memory-heavy.
https://github.com/pytorch/pytorch/issues/165530
FlashAttention upstream JVP support
Generic forward-mode AD support is still not something to assume from stock FlashAttention.
Custom JVP-aware FA2/FA3 kernels are currently the practical solution.
https://github.com/Dao-AILab/flash-attention/issues/1672
Core takeaway

Normal attention:

[
S = QK^\top/\sqrt d,\qquad
P=\mathrm{softmax}(S),\qquad
O=PV
]

JVP:

(\dot QK^\top + Q\dot K^\top)/\sqrt d
]

\operatorname{diag}((P\odot\dot S)\mathbf 1)O
]

So the JVP does NOT mathematically require materializing an (N\times N) attention matrix.

It can be implemented exactly like FlashAttention: tile over (Q,K,V), keep the temporary attention/JVP tiles in SRAM/registers, and accumulate (O) and (\dot O).

What I would try first
Check whether torch.func.jvp causes SDPA to fall back to math attention.
Replace only attention's JVP with:
NVIDIA rCM's Triton kernel, or
jvp_flash_attention.
Carry (x, dx) through the DiT layer-by-layer.
On Hopper, benchmark DMF's FA3-JVP implementation.
Compare against finite differences:
[
J_f(x)v \approx \frac{f(x+\epsilon v)-f(x)}{\epsilon}
]
using two ordinary FlashAttention forwards.
If the JVP target is stop_gradient, don't implement second-order backward unnecessarily.

TL;DR: yes, a specialized kernel is probably what we want; no, we probably don't need to write one from scratch.
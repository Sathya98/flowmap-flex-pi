# Vendored TVM attention-JVP kernels

- Source: https://github.com/lumalabs/tvm (commit 331c01d, 2026-02-13), `jvp_utils/`
  (Zhou, Parger, Haque, Song — *Terminal Velocity Matching*, arXiv 2511.19797).
- Files: `ryu_triton.py` (fused forward + JVP), `flash_attn_triton.py` (FA2 Triton primal
  backward), `flash_jvp_backward.py` (backward through the JVP). **Unmodified.**
- License: **CC BY-NC-SA 4.0** (`LICENSE-TVM`) — non-commercial, share-alike; derivatives
  of these files inherit it. Accepted for this project for now (2026-09-26); alternatives
  may be revisited later (see `jvp_kernel_analysis/README.md`).
- Wiring (ours): `../__init__.py`. Evaluation: `jvp_kernel_analysis/`.

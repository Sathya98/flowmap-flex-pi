# flowmap-core

Model-agnostic pieces for training **flow maps** (two-time maps
`X(s, t, x) = x + (t - s) v(x, s, t)`) on DiT flow-matching models, extracted from
FlexPi (`src/flexpi`) so other models — first the Wan2.2-TI2V-5B world model in
`../exmachina/diffsynth-studio` — can reuse them. FlexPi imports everything here through
module aliases, so the numbers it produces did not change (see
`.claude/context/08-flowmap-core-plan.md` for the extraction and its equivalence checks).

```bash
pip install -e flowmap_core          # from the flowmap-flex-pi repo root; torch is the only hard dependency
pytest flowmap_core/tests            # CPU, torch-only; FlexPi imports are blocked when run alone
```

## What is here

| Module | Contents |
|---|---|
| `flowmap` | `FlowMapObjectiveConfig` (objective, time sampling, weights, EMA decays, `jvp_attention` backend), `map_residuals` (LMD, EMD, PFMM, LSD, ESD, PSD-M/U over a tuple of jointly moving streams), `sample_level_pair_strip`, `affine_flow_map`, `tuple_jvp`, `dX_dt_forward_ad` / `dX_dt_finite_difference`. |
| `flowmap_self` | `update_diagonal_mask` (75/25 diagonal/off-diagonal split stratified by microstep, rank-consistent without a collective), `slice_batch`, `TimeLossWeight` (learned two-time loss weight). |
| `attention` | `scaled_dot_product_attention` that works under forward AD with backward through the tangent; backend `explicit` (FP32 L×L) or `tvm` (`set_jvp_attention_backend`). |
| `jvp_attention` | Fused attention JVP (`attention_jvp`, `dual_attention`) with mask row-grouping (`RowGroups`) and a per-mask plan cache; vendored Triton kernels in `tvm/`. |
| `normalization` | `ForwardADLayerNorm` (BF16-safe under forward AD). |
| `checkpoint` | Activation checkpointing that carries dual tensors (reverse-over-forward). |
| `ema` | `EvaluationEMA`: multi-decay FP32 shadow, flat chunked fold, optional background thread. |
| `step_profile` | `trace_summary` (GPU busy vs wall from a chrome trace), `SyncSites` (host-sync call sites), `write_report`. |
| `deepspeed_compat` | ZeRO-1/2 hook-count patch for DeepSpeed 0.18.0–0.18.6 (a no-op from 0.18.7). Env switch keeps its FlexPi name, `FLEXPI_DS_HOOK_COUNT_CACHE=0`. |
| `graphs` | `CapturedStep`: CUDA-graph capture/replay of a whole microstep (forward with JVPs, loss, backward; grads accumulate into `.grad`), `guard`/`keep_alive`/`check` for frozen content-dependent decisions (the fused JVP's mask plans). 2.2× per microstep at mb1 on FlexPi. |
| `latent_store` | `ArrayStore` (memory-mapped, sparse-preallocated, resumable `.npy` cache with a manifest), `to_numpy`/`from_numpy`/`load` (bf16 as int16), `select_windows`, `fingerprint_diff`. |

## What a model integration provides

- A config subclass of `FlowMapObjectiveConfig` with the model's own fields (FlexPi:
  `FlowMapConfig` adds `mode`, `streams`, `initialization`, `rank`, `lora_alpha`).
- `predict(state, s, t)` and `teacher(state, time)` callables over a tuple of stream
  tensors, and the per-stream loss reduction (masks, padding, weights). FlexPi's is
  `src/flexpi/models/helpers/flowmap_training.py`.
- A time-delta input on the DiT (`t - s` embedded next to `s`), and the model's attention
  routed through `flowmap_core.attention.scaled_dot_product_attention` and its LayerNorms
  replaced by `ForwardADLayerNorm` wherever forward AD passes.
- The cache layout on top of `ArrayStore` (FlexPi: `flexpi.datasets.latent_cache.LatentCache`).

## Versions

Validated on torch 2.7.1 / Triton 3.3.1. The forward-AD workarounds (`attention`,
`normalization`, `checkpoint`) target PyTorch 2.7 behaviour; re-validate on other versions
(the TVM kernels with `jvp_kernel_analysis/validate_tvm.py` in flowmap-flex-pi).

## License

Our code follows the flowmap-flex-pi repository's license. The vendored kernels in
`src/flowmap_core/jvp_attention/tvm/` come from lumalabs/tvm and are **CC BY-NC-SA 4.0**
(`LICENSE-TVM`, provenance in `SOURCE.md`). Using the `tvm` backend inherits those
terms (non-commercial, share-alike); the `explicit` backend does not load them.

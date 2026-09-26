# 08 — `flowmap_core` extraction: implementation plan and progress log

> **Goal.** Move the model-agnostic flow-map training code out of `src/flexpi` into a small
> shared package, `flowmap_core`. FlexPi must keep working **bit-identically**, and the Wan2.2
> world model in `../exmachina/diffsynth-studio` can then reuse the package
> (`docs/wm_wan_extension.md` §6–7).
>
> Status: **S0–S7 done and committed 2026-09-26.** Baseline commit:
> `8d00d25` on `dev-pedro`. The progress log at the bottom has every gate's result. Next work is
> phase 2 of `docs/wm_wan_extension.md` §7 (the WM port).

## Decisions (made; change only with the user)

| # | Decision | Why |
|---|---|---|
| D1 | Package lives **inside this repo**, at `flowmap_core/` (its own `pyproject.toml`, `src/` layout). diffsynth installs it with `pip install -e ../flowmap-flex-pi/flowmap_core`. | One git history while the API settles; it can be split into its own repo later. |
| D2 | Develop and gate in **`fm_env`** (torch 2.7.1, Triton 3.3.1). A `diff_env` (torch 2.11) check comes later, in the WM phase. | The forward-AD workarounds and TVM are validated only on 2.7.1. |
| D3 | Old FlexPi module paths become **`sys.modules` aliases** of the core modules, not `import *` re-exports. | Tests monkeypatch module globals (`tests/test_jvp_attention.py:139` patches `ja.supported`, which `dual_attention` looks up at call time). An alias is the same module object, so patches, private names and module globals (`attention._jvp_backend`) behave exactly as before. |
| D4 | Modules with FlexPi additions (`flowmap.py`, `latent_cache.py`) stay **real FlexPi modules** that import from the core and add their parts. | They carry FlexPi-only names (`STREAMS`, the `FlowMapConfig` fields, `CachedLatentDataset`). |
| D5 | Core module names mirror the FlexPi names. | Easy diffing and review. |
| D6 | The vendored TVM kernels move with their `LICENSE-TVM` and `SOURCE.md` (CC BY-NC-SA 4.0). The package README states the license split: our code vs the vendored kernels. | The user accepted inheriting the license for now. |

## Target layout

```
flowmap_core/
  pyproject.toml            # name=flowmap-core, deps: torch (numpy optional for latent_store)
  README.md                 # scope, API, license note for jvp_attention/tvm
  src/flowmap_core/
    __init__.py
    flowmap.py              # objective math + FlowMapObjectiveConfig (from helpers/flowmap.py)
    flowmap_self.py         # update_diagonal_mask, slice_batch, TimeLossWeight
    attention.py            # forward-AD SDPA + backend switch
    checkpoint.py           # dual-aware activation checkpoint
    normalization.py        # ForwardADLayerNorm
    jvp_attention/          # FusedAttentionJVP, RowGroups, plan cache; tvm/ (vendored + LICENSE-TVM, SOURCE.md)
    ema.py                  # EvaluationEMA (from utils/flowmap_ema.py)
    step_profile.py         # trace_summary, SyncSites, write_report
    deepspeed_compat.py     # ZeRO hook-count patch (no-op on DeepSpeed >= 0.18.7)
    latent_store.py         # storage half of datasets/latent_cache.py
  tests/                    # core-only tests (tiny, torch-only, CPU)
```

## What moves where (from the tree at `8d00d25`)

| FlexPi module | Fate | FlexPi path afterwards |
|---|---|---|
| `models/helpers/flowmap_self.py` | → `flowmap_core.flowmap_self` | alias |
| `models/helpers/attention.py` | → `flowmap_core.attention` | alias |
| `models/helpers/checkpoint.py` | → `flowmap_core.checkpoint` | alias |
| `models/helpers/normalization.py` | → `flowmap_core.normalization` | alias |
| `models/helpers/jvp_attention/` (incl. `tvm/`) | → `flowmap_core.jvp_attention` | alias package: `__init__` aliases itself; `tvm` resolves through the core package |
| `utils/flowmap_ema.py` | → `flowmap_core.ema` | alias |
| `utils/step_profile.py` | → `flowmap_core.step_profile` | alias |
| `utils/deepspeed_compat.py` | → `flowmap_core.deepspeed_compat` | alias |
| `models/helpers/flowmap.py` | objective math + `FlowMapObjectiveConfig` → `flowmap_core.flowmap` | **stays real**: `from flowmap_core.flowmap import *` + private names used anywhere, plus `STREAMS` and `FlowMapConfig(FlowMapObjectiveConfig)` |
| `datasets/latent_cache.py` | storage → `flowmap_core.latent_store` (`LatentCache`, `select_windows`, `to_numpy`/`from_numpy`/`load`, `fingerprint_diff`, constants) | **stays real**: FlexPi keeps `encoder_fingerprint`, `dino_frame_offsets`, `window_frames`, `dino_row_plan`, `CachedLatentDataset`, `build_inputs_from_cache` (re-verify the split by reading the file) |
| `flowmap_training.py`, `flowmap_diagnostics.py`, `adaptation.py`, `residual_sensitivity.py`, `trainer.py`, `flex_joint.py`, `runtime.py` | stay | unchanged except imports if needed |
| `jvp_kernel_analysis/fused_attention_jvp.py` | shim | imports from `flowmap_core.jvp_attention` |

**Alias shim pattern** (the entire file content at the old path):
```python
"""Moved to flowmap_core.<name>; this path is an alias of that module."""
import sys
import flowmap_core.<name> as _module
sys.modules[__name__] = _module
```
`from .helpers.attention import scaled_dot_product_attention` then resolves to the core
module. Check that `flexpi.models.helpers.attention is flowmap_core.attention` and that the
parent package's attribute points to the same module (import the parent, then assert).

## The config split (`FlowMapConfig`)

- **Core `FlowMapObjectiveConfig`** (everything the objective math or a generic trainer needs):
  `enabled, objective, self_diagonal_fraction, learned_time_weighting,
  distill_learned_time_weighting, distill_ema, lmd_teacher_gradient, ema_decays,
  pfmm_loss_space, diagonal_weight, distill_diagonal_weight, map_weight, teacher_steps,
  schedule_shift, num_inference_steps, strip_width, time_sampling, dt_method,
  detach_derivatives, jvp_attention, fd_eps, teacher_checkpoint`, their `__post_init__`
  validation, and the properties `self_distillation`, `uses_time_weighting`, `uses_ema`,
  `needs_teacher`.
- **FlexPi `FlowMapConfig(FlowMapObjectiveConfig)`**: `mode, streams, initialization, rank,
  lora_alpha`, plus their validation (streams ⊆ `STREAMS`, mode in adapter/lora/full/heads,
  random init requires full). `__post_init__` calls `super().__post_init__()` first.
- **Invariants:**
  - `asdict(FlowMapConfig(...))` has the same key→value mapping as before (key order may
    change; checkpoints store a dict: `flexpi.py:2560`, read back at `:2574`).
  - `FlowMapConfig(**dict)` construction (`runtime.py:213`) and Hydra overrides are unchanged.
  - Every validation error still fires. Compare the messages `tests/` matches on
    (`pytest.raises(..., match=...)`).
- Read the current `__post_init__` (`helpers/flowmap.py` ~lines 51–100) before splitting.
  Some validations mix fields (e.g. `lmd_teacher_gradient` vs `objective`; `initialization`
  vs `mode`). Put each check in the lowest class that owns *all* the fields it reads.
- Drop `without_checkpointing` (no callers; it names FlexPi attributes).

## Steps and gates

- [x] **S0. Baseline fingerprints on `8d00d25`, before any move.**
  - Write `scripts/refactor_fingerprint.py`. With fixed seeds on the tiny CPU models from
    `tests/test_flowmap.py` (`tiny_model`, `batch`), for each case:
    - lmd (detached and full-grad);
    - lsd; esd; pfmm;
    - lsd + esd with a mixed diagonal batch;
    - the flex regimes (`tests/test_flowmap_flex.py::flex_model`);
    - the batched self-distillation path;
    - `EvaluationEMA` background vs sync;
    - `update_diagonal_mask` outputs;
    - the explicit JVP attention (fp64);
    - `RowGroups` on the test masks.

    Record loss, every metric, sha256 of every parameter grad (`tensor.numpy().tobytes()`)
    and the `asdict(FlowMapConfig())` mapping. Write JSON to
    `runs/refactor/fingerprint_8d00d25.json`. Run it **twice** to confirm CPU determinism.
  - GPU fingerprint job (1 H100): one `lmd_full` and one `lsd_off` step with
    `jvp_attention=tvm` and `explicit`, on the LIBERO cache, fixed seed. Record loss and
    grad-norm, **twice**, to measure run-to-run spread (Triton/SDPA backward may use atomics).
    Reuse `scripts/profile_flowmap_step.py`'s model and batch setup; add a `--fingerprint`
    mode, or write a small sibling script.
  - Gate: fingerprints recorded; CPU rerun identical; GPU spread known.
- [x] **S1. Create the package skeleton** (`flowmap_core/pyproject.toml`, `src/flowmap_core/`,
  README) and install it: `uv pip install --python ../fm_env/bin/python -e flowmap_core`
  (UV_CACHE_DIR per `env_flexpi.sh`).
  - **Shared env: install only, no other package changes.** Check `uv pip install` does not
    touch other packages (use `--no-deps` if it tries).
  - Gate: `python -c "import flowmap_core"` works from a directory outside the repo.
- [x] **S2. Move the alias-able modules** with `git mv` (keeps history) into
  `flowmap_core/src/flowmap_core/`, fix their internal relative imports, and put alias shims
  at the old paths. Order: normalization, checkpoint, attention + jvp_attention (together:
  `attention` lazily imports `jvp_attention`), flowmap_self, ema, step_profile,
  deepspeed_compat.
  - Gate: the full FlexPi CPU suite passes (88 tests + 46 subtests; command below), **and** the
    CPU fingerprint is identical.
- [x] **S3. Split `flowmap.py`**: objective math + `FlowMapObjectiveConfig` → core; FlexPi keeps
  `STREAMS` and `FlowMapConfig`.
  - Gate: suite + CPU fingerprint identical (including the `asdict` mapping). Load one real
    checkpoint's saved `flow_map` dict with `FlowMapConfig(**d)` (any
    `runs/flowmap_fulljoint/*/checkpoints/weights/*.pt` payload `flow_map`).
- [x] **S4. Split `latent_cache.py`**: storage → `flowmap_core.latent_store`.
  - Gate: `tests/test_latent_cache.py` passes. `scripts/cache_latents.py verify` on
    `data/latent_cache/libero_fulljoint_v2`, `--limit 64`, reports OK (CPU is fine if it
    supports that; else a short GPU job).
- [x] **S5. Core-only tests** under `flowmap_core/tests/`: copies of the model-agnostic tests
  (JVP attention grouping and routing, diagonal mask, EMA, step_profile, deepspeed_compat,
  latent_store round-trip, plus a tiny generic `map_residuals` test on a 2-layer MLP "DiT" with
  a time input, to prove FlexPi-free use). They import only `flowmap_core`.
  - Gate: `pytest flowmap_core/tests` passes with `flexpi` **not importable**. Run with
    `python -S`, or check `'flexpi' not in sys.modules` in a conftest.
- [x] **S6. GPU equivalence.** Rerun the S0 GPU fingerprint job on the refactored tree.
  - Gate: loss and grad-norm equal the baseline within the measured run-to-run spread
    (bit-identical expected for `explicit` if the baseline was deterministic).
  - Then one `trainer_timing.sbatch flowmap_libero_lmd_full zero2_tvm_mb2` (and
    `ema_bg_lsd`): per-update time within noise of efficiency notes §12 (LMD ~72 s at mb2 with
    background EMA; LSD ~30 s).
- [x] **S7. Docs and cleanup.**
  - `flowmap_core/README.md` (API, what's in and out, license split).
  - Update `.claude/context/04-repo-map.md` (new top-level package), `07-efficiency-notes.md`
    pointers, `docs/wm_wan_extension.md` §6 (done + any deviations).
  - Point `jvp_kernel_analysis/fused_attention_jvp.py` at the core.
  - `.claude/CLAUDE.md` index row for this file.
  - Add the core to FlexPi's `pyproject.toml` dependencies (path/editable note).
  - Commit on `dev-pedro` (ask the user before committing and pushing).

## Commands

```bash
source /gpfs/scratch1/shared/faster-wams/env_flexpi.sh
cd /gpfs/scratch1/shared/faster-wams/flowmap-flex-pi
# FlexPi CPU suite (~4 min on the login node; run in the background):
python -m pytest -q tests/test_flowmap*.py tests/test_jvp_attention.py tests/test_flex_joint_share.py \
  tests/test_step_profile.py tests/test_deepspeed_compat.py tests/test_latent_cache.py
python -m pytest -q jvp_kernel_analysis/test_grouping.py
# Find every import of a moved module (update this list if a grep finds more):
grep -rn "helpers\.\(flowmap\|flowmap_self\|attention\|checkpoint\|normalization\|jvp_attention\)\|utils\.\(flowmap_ema\|step_profile\|deepspeed_compat\)\|datasets\.latent_cache\|from \.\(flowmap\|flowmap_self\|attention\|checkpoint\|normalization\|jvp_attention\)\b" \
  --include=*.py src scripts tests jvp_kernel_analysis experiments
```

Known import sites at `8d00d25`:
- `flexpi.py:37,48,1399,2732,3601`; `wan_video_dit.py:7,8`; `action_dit.py:9`; `mot.py:11`;
- `helpers/gradient.py:2`; `helpers/flowmap_training.py:5,6`; `helpers/flowmap_diagnostics.py:7,8`;
  `helpers/adaptation.py:104`;
- `runtime.py:134`; `trainer.py:123,216,1780,1840,1876`;
- plus tests and scripts (`scripts/cache_latents.py`, `scripts/profile_flowmap_step.py`,
  `jvp_kernel_analysis/*`).

## Pitfalls to watch

- **Aliasing and `importlib.reload`/pickling.** Pickled objects record the core module path
  from now on. Check no checkpoint pickles FlowMap classes (the EMA state is tensors plus
  plain types; the config is `asdict`).
- **`jvp_attention` lazily imports `.tvm`** (`_kernels()`). The relative import must resolve
  inside the core package after the move.
- **Module globals:** `attention._jvp_backend` and `jvp_attention._plans` are process-global.
  With aliases there is exactly one copy. Never let two copies coexist (e.g. by importing the
  core through a second path).
- **Tests that `patch.object` module functions** must target the alias, which is the core
  module. Grep `patch.object(` and `monkeypatch.setattr(` over `tests/` after each step.
- **Keep runs in the queue in mind.** Jobs started from the working tree import the code at
  process start; don't refactor under a running job whose results you need. Check `squeue`
  first.
- **Shared env:** `fm_env` is shared. Only add the editable `flowmap_core` install.

## Progress log

- 2026-09-26 — Plan written (this file). Next: S0.
- 2026-09-26 — **S0 done.** `scripts/refactor_fingerprint.py OUT [--compare BASE]` (CPU, 1 thread,
  deterministic algorithms; config/mask/attention/row-groups/EMA/10 objectives/5 flex regimes/6
  batched-self cases) → `runs/refactor/fingerprint_8d00d25.json`; rerun IDENTICAL (~1.5 min).
  GPU: `scripts/profile_flowmap_step.py --fingerprint tvm,explicit --modes lmd_full,lsd_off
  --repeats 2 --samples 1 --input-samples 1` via `profile_flowmap_step.sbatch` (jobs 27210890,
  27210892; `runs/diagnostics/profile_flowmap_step_<job>/results.json`). Baseline:
  | mode/backend | loss | grad norm | grad sha |
  |---|---|---|---|
  | lmd_full/explicit | 2.1453397274017334 | 2588.4034442469065 | 0f653dfdf56bfce1 (all 4 runs) |
  | lsd_off/explicit | 1.8694127798080444 | 731.0100391339295 | 651bca1ffebb4f31 (all 4 runs) |
  | lmd_full/tvm | 2.1385114192962646 (all 4) | 2586.63497–2586.63588 | differs every run |
  | lsd_off/tvm | 1.8674802780151367 (all 4) | 730.63675–730.63705 | differs every run |
  Gate for S6: explicit bit-identical (sha); TVM loss identical, grad norm within ~5e-7 relative.
- 2026-09-26 — **S1 done.** `flowmap_core/` (pyproject, README, `src/flowmap_core/__init__.py`),
  `uv pip install --no-deps -e flowmap_core` into fm_env (only that package installed);
  imports from outside the repo.
- 2026-09-26 — **S2 done.** `git mv` of normalization, checkpoint, attention, flowmap_self,
  jvp_attention/ (with tvm/, LICENSE-TVM, SOURCE.md), utils/flowmap_ema → `ema`, step_profile,
  deepspeed_compat; alias shims at the old paths (`jvp_attention/__init__.py` is the shim for the
  package). Checked: old name, `sys.modules` entry and parent attribute are the core module.
  Suite 97 passed + 46 subtests (the 88 + `jvp_kernel_analysis/test_grouping.py`); CPU fingerprint
  IDENTICAL. Env switch names kept (`FLEXPI_DS_HOOK_COUNT_CACHE`).
- 2026-09-26 — **S3 done.** Core `FlowMapObjectiveConfig` + objective math in
  `flowmap_core.flowmap` (`without_checkpointing` dropped); FlexPi `helpers/flowmap.py` is now
  `STREAMS` + `FlowMapConfig(FlowMapObjectiveConfig)` (mode, streams, initialization, rank,
  lora_alpha) + explicit re-exports. **Deviation:** the combined message "schedule_shift,
  map_weight and lora_alpha must be positive" split into core "schedule_shift and map_weight must
  be positive" and FlexPi "lora_alpha must be finite and positive" (no test matches them).
  Real checkpoint `libero_lmd_100.../step_000025.pt` `flow_map` dict → `FlowMapConfig(**d)` →
  `asdict` equal (except `jvp_attention`, which postdates that checkpoint: default, as before).
  Suite + fingerprint IDENTICAL.
- 2026-09-26 — **S4 done (CPU).** **Deviation:** `LatentCache` itself is FlexPi-specific (DINO
  rows, constant tokens, `ready()` over `dino_row_owner`), so the core got a generic
  `latent_store.ArrayStore` (create by row kind → length, open with class `VERSION`, `array`,
  `done`) plus `NP_DTYPES`, `to_numpy`/`from_numpy`/`load`, `select_windows`,
  `fingerprint_diff`; FlexPi `LatentCache(ArrayStore)` keeps `create(root, manifest, index)`,
  DINO methods, `ready`. `tests/test_latent_cache.py` 5 passed; suite + fingerprint IDENTICAL.
- 2026-09-26 — **S5 done.** `flowmap_core/tests/` (28 tests): conftest blocks any `flexpi` import
  (meta-path finder + autouse check; verified it fires); copies of JVP-attention grouping/routing,
  diagonal mask, TimeLossWeight, background EMA, step_profile, deepspeed_compat; new
  `test_latent_store.py` and `test_generic_dit.py` (tiny 2-block DiT with core attention,
  ForwardADLayerNorm and dual checkpoint: all 7 objectives train; AD vs FD dX/dt in fp64;
  checkpointed LMD-full grads equal plain; bf16 forward AD). Passes from outside the repo.
- 2026-09-26 — **S6 done.** GPU fingerprint on the refactored tree (job 27211277): explicit
  `lmd_full` 2.1453397274017334 / 2588.4034442469065 / sha 0f653dfdf56bfce1 and `lsd_off`
  1.8694127798080444 / 731.0100391339295 / sha 651bca1ffebb4f31 — **bit-identical** to S0. TVM
  losses identical (2.1385114192962646, 1.8674802780151367); grad norms 2586.63501–2586.63539 and
  730.63674–730.63678, inside the S0 spread (≤ 3e-9 relative outside its recorded range).
  Timing (job 27211279, `trainer_timing.sbatch flowmap_libero_lmd_full ema_bg_lmd ema_bg_lsd`),
  steady updates 3–4: LMD 72.3 / 74.2 s, LSD 30.4 / 30.0 s (§12: 71.7–73.6, 29.7–30.3); first two
  updates slower from the known one-time ~37 s first EMA fold, as before.
  `cache_latents.py verify --samples 64` on `libero_fulljoint_v2` (job 27211278) **FAILS** the 1e-3
  VAE tolerance (input_latents 5.59e-3, first_frame 8.80e-3, pointmap 4.64e-3, dino 1.12e-2) —
  but the **pre-refactor tree at 8d00d25 gives exactly the same numbers** (job 27211483, run from
  a temporary worktree), so the refactor is not the cause. Pre-existing: the same verify gave 0.0
  VAE error right after encoding (job 27165239, 2026-09-21). Open item in 07 TODO.
- 2026-09-26 — **S7 done (except commit).** README; 04 repo map (§1 helpers, new §1b, deps);
  CLAUDE.md row; `docs/FLOW_MAPS.md`, `docs/LATENT_CACHE.md`, `docs/INSTALL.md` (install line,
  Dockerfile COPY), `docs/wm_wan_extension.md` §6 status; `jvp_kernel_analysis/` shim,
  `validate_tvm.py` and README point at the core; FlexPi `pyproject.toml` notes the in-repo
  package (not listable: not on PyPI); config comments. Source-provenance lists in
  `helpers/residual_sensitivity.py` and `scripts/flowmap_diagnostics.py` now hash the core files
  (the alias shims carry no code). Core test files are prefixed `test_core_` (basename clash with
  `tests/` when both run in one pytest), and the conftest blocks `flexpi` only when pytest runs
  `flowmap_core` alone. Final: `pytest tests jvp_kernel_analysis/test_grouping.py flowmap_core/tests
  --ignore=tests/test_paired_residual_analysis.py` → 135 passed + 58 subtests (that file needs
  matplotlib, absent from fm_env; unrelated); `pytest flowmap_core/tests` → 28 passed with flexpi
  blocked; CPU fingerprint IDENTICAL.

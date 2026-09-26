# 04 — Repo map, eval/deploy, dependencies, external services

A file-by-file orientation, plus the pieces outside the train loop (eval, deploy,
inference optimization) and the external services this project pulls in.

## 1. `src/flexpi/` — the package

```
models/
  flexpi.py            FlexPi — top-level model (heads, training_loss, infer_action)   ← 01,02
  backbone.py          FlexPiBackbone — composes trunk+action expert+MoT; proprio enc  ← 00,01
  mot.py               MoT — Mixture-of-Transformers, HBridge, joint attention         ← 00
  wan_video_dit.py     WanVideoDiT (visual trunk) + Head (video/pointmap head) + DiTBlock ← 01
  action_dit.py        ActionDiT (action expert) + ActionHead                          ← 01
  dino_encoder.py      DinoEncoder — frozen DINOv3 ViT-B/16 (timm)                     ← 01,03
  pointmap_encoder.py  PointmapEncoder — depth→XYZ→VAE (parameter-free)                ← 01,03
  wan_video_vae.py     Wan-2.2 VAE (frozen; encodes RGB + pointmap, decodes video)
  wan_video_text_encoder.py  umT5 text encoder (frozen; unused at train — cache instead)
  wan22.py             Wan-2.2 component loader (load_wan22_ti2v_5b_components)
  schedulers/scheduler_continuous.py  WanContinuousFlowMatchScheduler                  ← 01
  helpers/
    flex_joint.py      FlexJointConfig + sample_flex_batch_flags (m_in/m_out sampling) ← 00
    dino.py            DINO RoPE freqs, frame-slot selection, x0→v conversion          ← 01
    gradient.py        gradient_checkpoint_forward
    loader.py, io.py, state_dict_converters.py   weight loading / conversion
    flowmap.py         FlowMapConfig (FlexPi fields) over flowmap_core.flowmap         ← 05
    flowmap_training.py, flowmap_diagnostics.py, adaptation.py   flow-map loss/tuning  ← 05
    attention.py, checkpoint.py, normalization.py, flowmap_self.py, jvp_attention/
                       ALIASES of flowmap_core modules (sys.modules shims; see §1b)
  inference_opt/       TensorRT + CUDA-graph + step-skip engines (joint path only)     ← §3
datasets/lerobot/      the data pipeline (RobotVideoDataset, processors, transforms)   ← 03
composite_layouts.py   LayoutSpec / Slot registry                                      ← 03
per_cam_compose.py     compose_from_per_cam (GPU composite assembly)                   ← 03
runtime.py             create_flexpi factory + run_training                            ← 02
trainer.py             Wan22Trainer (optimizer loop, eval hook, checkpointing)         ← 02
vis.py, utils/         viz, config resolvers, samplers, video IO/metrics, logging
```

Cross-references (`← NN`) point to the `.claude/context/` doc with the detail.

## 1b. `flowmap_core/` — shared flow-map package (in-repo, editable-installed)

Model-agnostic flow-map training code, extracted so the Wan2.2 world model
(`../exmachina/diffsynth-studio`) can reuse it (`docs/wm_wan_extension.md` §6,
`.claude/context/08-flowmap-core-plan.md`). `src/flowmap_core/`: `flowmap`
(`FlowMapObjectiveConfig`, `map_residuals`, time sampling, JVP helpers), `flowmap_self`
(diagonal mask, `slice_batch`, `TimeLossWeight`), `attention` + `jvp_attention/` (forward-AD
SDPA, fused TVM JVP kernels, CC BY-NC-SA 4.0), `checkpoint`, `normalization`, `ema`
(`EvaluationEMA`), `step_profile`, `deepspeed_compat`, `latent_store` (`ArrayStore`), `graphs` (`CapturedStep`, whole-microstep
CUDA graphs; harness `scripts/cuda_graph_step.py`).
Old FlexPi paths are `sys.modules` aliases: `flexpi.models.helpers.{attention,checkpoint,
normalization,flowmap_self,jvp_attention}`, `flexpi.utils.{flowmap_ema,step_profile,
deepspeed_compat}` *are* the core modules (patch either name). Core-only tests:
`pytest flowmap_core/tests` (FlexPi import blocked by its conftest). Install:
`uv pip install --no-deps -e flowmap_core` (done in `fm_env`).

## 2. Evaluation — `experiments/` + `scripts/eval_*.sh`

Eval never instantiates the dataset (compose a fresh config with `task=...`; a saved run
`config.yaml` carries loader keys the merged dataset dropped — see `docs/TRAINING.md §5`).

- **RoboTwin** (`experiments/robotwin/`): `run_robotwin_manager.py` (eval manager),
  `eval_robotwin_single.py`, `flexpi_policy/deploy_policy.py` (policy wrapper +
  `deploy_policy.yml`). Harness vendored under `third_party/RoboTwin/`. Full sweep = 50
  tasks × 2 phases × 100 episodes. Launchers `scripts/eval_flexpi_robotwin{,_single}.sh`
  (set `CKPT` + `DATASET_STATS` at top; `INFER_JOINT_*`/`INFER_PRESENT_*` pick the regime).
- **LIBERO** (`experiments/libero/`): `eval_libero_batch.py` / `eval_libero_single.py`,
  `summarize_results_4suite.py`, `action_ensembler.py`, `task_pool.py`. Launchers
  `scripts/eval_flexpi_libero_4suite.sh` (shards 40 tasks across GPUs → `summary_4suite.{csv,json}`),
  `..._single.sh`. LIBERO is a submodule (`third_party/LIBERO`, PYTHONPATH). Replans every 10.
- **YAM (real robot)** (`experiments/yam/flexpi_policy/`): the msgpack WebSocket client +
  bridges, RTC step broker, temporal/speed smoothers, prediction recorder, smoke/unit
  tests. Wire contract + serving in `docs/YAM.md`.

Eval regimes: `+EVALUATION.infer_joint_{video,dino,pointmap}=true|false` and
`infer_present_*`. Eval pins `num_inference_steps=4`.

## 3. Inference optimization — `src/flexpi/models/inference_opt/` + `scripts/inference_opt/`

Training-free speedups on the **joint path only** (`docs/INFERENCE_OPTIMIZATION.md`).
`trt_joint.py`, `trt_joint_split.py` (KV-split engines), `trt_prefill.py`,
`joint_loop_graph.py` (whole-loop CUDA graph), `encoder_graphs.py`, `step_skip.py`
(`StepSkipController`). Benchmarks/engine builds in `scripts/inference_opt/`
(`benchmark_flex_latency.py`, `trt_onnx_*_engine.py`). ms/call on RTX 5090 @4 steps:
eager 447 (fj) / 132 (ao); torch.compile 360 / **60**; +TRT joint 230; +TRT KV-split **193**.

## 4. Deployment — `scripts/serve_yam_flexpi.py` (+ `serve_flexpi_yam.sh`)

WebSocket policy server: robot client sends observations, receives action chunks. Loads
`dataset_stats.json` + `config.yaml` from beside the checkpoint. `serve_flexpi_yam.sh`
wires the regime (`--infer-joint-*`/`--infer-present-*`) and TensorRT knobs. The msgpack
wire contract + emergency-stop rules are in `docs/YAM.md`.

## 5. Scripts of note (`scripts/`)
- `train.py`, `train_flexpi_{robotwin,libero,yam}.sh` — training entry + launchers  ← 02
- `precompute_text_embeds.py`, `run_precompute_text_embeds.sh` — umT5 cache (required)  ← 03
- `preprocess_action_dit_backbone.py` — build the resampled ActionDiT donor ckpt
- `da3_depth/` — add Depth-Anything-3 depth to an RGB-only dataset (`label_depth.py`,
  `setup.sh`, `run.sh`)
- `yam_dataset_builder/` — raw → LeRobot v2.1 builder (`convert.py`, `source_reader.py`)
- `accelerate_configs/` (zero0/1/2, ddp, single-gpu), `ds_configs/` (zero1/2 json)

## 6. Dependencies (`pyproject.toml`)
Python ≥3.10, CUDA 12.8. Core: `torch==2.7.1+cu128`, `torchvision`, `torchcodec==0.5`,
`decord2==3.3.0` (imports as `decord`; the depth/AV1 fast path), `accelerate==1.12.0`,
`deepspeed==0.18.9`, `transformers==4.49.0`, `timm==1.0.26` (DINOv3), `hydra-core==1.3.2`,
`av==16.0.1`, `huggingface-hub==0.29.2`, `wandb`, and the in-repo `flowmap-core`
(editable install of `./flowmap_core`, not on PyPI). Extras:
- `.[libero]` — `mujoco==3.3.2` (pin is load-bearing: 3.8.0 shifts libero_object OOD),
  `bddl`, `gym`; **`robosuite==1.4.0` installed separately with `--no-deps`**.
- `.[serve]` — `msgpack`, `websockets`, `osqp`, `scipy` (YAM server).
- `.[trt]` — `tensorrt==10.16.1.11` (engines tied to the TRT minor).

## 7. External services & artifacts this project involves
- **HuggingFace downloads**:
  - **Weights** (once, into `checkpoints/`, `DIFFSYNTH_MODEL_BASE_PATH`): Wan2.2-TI2V-5B
    (VAE + DiT + umT5), the resampled ActionDiT backbone, DINOv3 (via timm
    `vit_base_patch16_dinov3.lvd1689m`). See `docs/INSTALL.md §2`.
  - **Datasets** (`huggingface.co/flex-pi`): `libero_mujoco3.3.2_depth`, `robotwin_3d`
    (+ `robotwin_3d_text_embeds_cache`), per-task YAM sets. Ready to train as downloaded.
  - **Released checkpoints**: `flexpi-robotwin`, `flexpi-libero`,
    `flexpi-libero-fulljoint-star`.
- **SLURM / multi-node**: launchers speak `NNODES`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT`
  and sync `RUN_ID` via a `TCPStore` so ranks agree on the output dir; `accelerate launch
  --rdzv_backend static --deepspeed_multinode_launcher standard`. Default 8 GPUs, floor 4
  (80 GB each for training). Wrap a launcher in an `sbatch` script exporting those.
- **Simulators (eval only)**: RoboTwin (vendored, `third_party/RoboTwin/`), LIBERO
  (submodule) + robosuite/mujoco. `docs/INSTALL.md §4`.
- **Depth annotation**: Depth-Anything-3 for RGB-only datasets (`scripts/da3_depth/`).
- **Real robot**: YAM bimanual via `raiden`; policy over msgpack WebSocket.

## 8. Third party (`third_party/`)
- `RoboTwin/` — vendored eval harness (see its `README.vendor.md`).
- `LIBERO/` — git submodule (`.gitmodules`), used via PYTHONPATH (its `setup.py` is a no-op).

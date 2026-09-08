# 02 — Training & finetuning pipeline

Code-level companion to `docs/TRAINING.md`. The entry chain is:

```
scripts/train.py  (Hydra @main)
  └─ flexpi.runtime.run_training(cfg)
       ├─ create_flexpi(...)            # build the FlexPi model (Hydra _target_)
       ├─ build_datasets(cfg.data)      # RobotVideoDataset train/val
       └─ Wan22Trainer(...).train()     # optimizer loop
            └─ FlexPi.training_loss(sample)   # ALL encode/forward/loss work
```

**The trainer drives only the optimizer.** Every VAE/DINO encode, the flex sampling, the
MoT forward, and the four losses live on the *model* (`_base_training_loss` /
`build_inputs`). See `01-flowmatching-heads.md §6` for that inner path.

Line numbers are a snapshot; verify before editing.

---

## 1. Model factory — `runtime.py`

`create_flexpi(...)` (`runtime.py:72-255`) is the Hydra `_target_` of
`configs/model/flexpi.yaml`. It normalizes the `DictConfig` blocks (schedulers, loss
lambdas, DINO/pointmap knobs, `flex_joint` → `FlexJointConfig` dataclass, `hbridge`
tuple) and delegates to `FlexPi.from_wan22_pretrained(...)` (`runtime.py:197`). Returns a
fully-built `FlexPi` on the target device/dtype — **no optimizer, no dataloaders**.

`run_training` resolves device/dtype first: `_resolve_train_device()` →
`cuda:{LOCAL_RANK}` under multi-GPU (`runtime.py:286`); `_mixed_precision_to_model_dtype`
maps `bf16 → torch.bfloat16` (`runtime.py:29`); both injected via
`instantiate(cfg.model, model_dtype=..., model_device=...)` (`runtime.py:427`) and threaded
into every sub-encoder.

**Weight loading** (`FlexPi._base_from_wan22_pretrained`, `flexpi.py:746-878`):
1. `load_wan22_ti2v_5b_components` → Wan-2.2-TI2V-5B **VAE**, **umT5 text encoder**
   (skipped when `load_text_encoder=false`), tokenizer, **video DiT** (the ~5B trunk).
2. `ActionDiT.from_pretrained(action_dit_pretrained_path=
   "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt")` — the ~1B action
   expert, resampled from Wan-2.2. `skip_dit_load_from_pretrain=true` → random init.
3. `MoT(mixtures={"video":..., "action":...}, hbridge_*)`.
4. Frozen `DinoEncoder` + parameter-free `PointmapEncoder`.

`build_datasets(cfg.data)` (`runtime.py:258`) builds `data.train`, and reuses it for val
when `val_set_proportion < 1e-6`, else builds `data.val` sharing `pretrained_norm_stats`.

## 2. Trainer — `trainer.py` (`Wan22Trainer`)

**Owns**: `model`, `optimizer` (AdamW), `scheduler` (warmup→cosine), `train_loader`,
`accelerator` (DeepSpeed). Constructor (`trainer.py:31-162`):
- Propagates `cfg.model.composite_layout` onto the unwrapped model before `prepare`
  (`:43-45`).
- `Accelerator(gradient_accumulation_steps, mixed_precision, step_scheduler_with_optimizer=False)`
  (`:89`).
- **Freezes non-trainables before building the optimizer**: `_apply_dit_only_train_mode`
  (`:120`, `:544-583`) sets `model.FROZEN_MODULES`
  (`{vae, text_encoder, dino_encoder, pointmap_encoder}`) to eval + `requires_grad_(False)`,
  everything else train + grad. Trainable params = `[p for p in model.parameters() if
  p.requires_grad]`.
- Optimizer → loader → horizon estimate → scheduler; captures `_scheduler_base_lrs` for
  re-anchoring (`:135`); `accelerator.prepare(model, opt, loader, sched)` (`:152`);
  wandb; `_resume_or_load_checkpoint()` (`:159`).

Loader (`_build_loader`, `:213`): `WeightedResumableEpochSampler` when the dataset exposes
`dataset_weights`, else `ResumableEpochSampler`; `persistent_workers`, `prefetch_factor=4`.

### The step (`trainer.py:1560-1573`)
```python
with self.accelerator.accumulate(self.model):
    loss, loss_dict = train_model.training_loss(sample)   # ← all the work
    self.accelerator.backward(loss)
    if self.accelerator.sync_gradients:
        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
```

### Optimizer / schedule / precision
- **AdamW** (`_build_optimizer`, `:164`): `lr=cfg.learning_rate` (1e-4),
  `weight_decay=cfg.weight_decay`, `betas=(0.9,0.95)`. **`weight_decay=1e-2` in the task
  configs** (the `0.0` in `train.yaml` is overridden).
- **LR schedule** (`_build_scheduler`, `:299`): `LinearLR` warmup (5% of total steps) →
  `CosineAnnealingLR(eta_min=lr*0.01)`, joined by `SequentialLR`. `total_train_steps` from
  `_estimate_total_train_steps` (`:275`, accounts for world size + grad-accum).
  `_reanchor_lr_schedule` (`:500`) rebuilds cosine at the original peak when resuming into
  a longer horizon (`cfg.resume_reanchor_lr_schedule`).
- **bf16**: `mixed_precision="bf16"`, forward under `accelerator.autocast()` (`:1563`).
- **Gradient accumulation**: `gradient_accumulation_steps` (**8** in task configs);
  opt/sched/step fire only on `sync_gradients`.
- **max_grad_norm**: `1.0` (`:1568`).
- **Gradient checkpointing**: `use_gradient_checkpointing: ${model.mot_checkpoint_mixed_attn}`
  on both DiT configs — but **all three shipped task configs set
  `mot_checkpoint_mixed_attn: false`**, so checkpointing is OFF in shipped runs. Two
  independent knobs exist: MoT's shared-attention checkpoint (`mot_checkpoint_mixed_attn`,
  `mot.py:200-223`) and each expert's post-block checkpoint (`expert.use_gradient_checkpointing`).
- **DeepSpeed ZeRO-1** via `accelerate` (`scripts/accelerate_configs/accelerate_zero1_ds.yaml`
  → `scripts/ds_configs/ds_zero1_config.json`, `stage:1`, no offload, batch/accum `"auto"`).

## 3. Checkpointing (`trainer.py:1352-1510`)

- **`checkpoints/weights/step_NNNNNN.pt`** (`_save_weights_checkpoint`, `:1352`): model
  weights only, via `model.save_checkpoint` (`flexpi.py:2486`):
  `{"mot", "step", "torch_dtype", <dino heads>, <mode heads>, optional "proprio_encoder"}`.
  Main process only. **This is what eval/serve load.**
- **`checkpoints/state/step_NNNNNN/`** (`accelerator.save_state`, `:1396`): full DeepSpeed
  state (sharded optimizer + grads + weights + scheduler) + `trainer_state.json`
  (`{global_step, epoch, batch_in_epoch}`) for dataloader-resume.
- **Cadence** (`:1651-1666`): eval at step 0 and every `eval_every`; checkpoint every
  `save_every`; final save at `max_steps`/loop end. Task defaults `save_every=1000`,
  `eval_every=500`. Saves are try/except so a full-FS failure drops the partial and
  continues.
- **keep_last_n**: `keep_last_n_states` / `keep_last_n_weights` (default 3; `<=0` keeps all).

### RESUME vs PRETRAINED_CKPT (`_resume_or_load_checkpoint`, `:336`) — mutually exclusive
- **`resume=<state/step_*/ dir>`** → full state restore (optimizer/scheduler/step/dataloader),
  continue the *same* run. A `.pt` path instead loads weights only.
- **`pretrained_ckpt=<...>`** → weights-only **warm start of a new run**, fresh
  optimizer/scheduler/step, new output dir tagged `_ft`. Accepts a `.pt`, a `state/step_*/`
  dir (→ sibling `weights/*.pt`), or a run dir (→ latest).
- **Strict-shape** (`flexpi.py:2500`, base `backbone.py:1485`): `resume` defaults
  `strict_shape=True` (any mismatch raises); warm-start default `false` re-initializes
  mismatched DINO/pointmap I/O heads with a warning (enables cross-layout / cross-embodiment
  warm starts). `action_dim`, `composite_layout`, `dino_pixel_unshuffle` all count.

## 4. Eval hook during training (`trainer.py:726-1350`)

`evaluate()` (`@torch.no_grad`) at step 0 + every `eval_every`: picks a seeded val index,
computes `val_loss` via `training_loss`, then a rollout `model.infer(...,
num_inference_steps=eval_num_inference_steps)`.
- **Video metrics**: PSNR/SSIM (rollout vs GT, vs VAE-recon, recon vs GT); action L1/L2 vs
  denormalized GT.
- **`eval_video` flag** (default true): `false` skips the expensive rollout + viz mp4,
  keeping val-loss/action metrics — for Flex-π's no-test-time-video deployment.
- **Viz mp4** `eval/step_NNNNNN_rank_RRR.mp4`: rows for RGB pred/recon/GT, DINO PCA,
  pointmap XYZ + depth-proj + VAE-PCA. `share_pca_basis` makes pred/GT share one PCA basis.

## 5. Config graph & launchers

### Hydra (`scripts/train.py`, `configs/`)
`configs/train.yaml` has `defaults: [_self_, data:null, model:null, task:null]` — the three
groups are empty until `task=` fills them. A task config (`configs/task/*.yaml`,
`# @package _global_`) carries its own `defaults` that `override /data:` and `override
/model:`, then overrides top-level knobs. **One `task=` picks the whole run.** Interpolations
like `data.train.concat_multi_camera: ${model.composite_layout}` keep data/model geometry in
lockstep. Any trailing CLI override wins (launchers append `"$@"`).

Three shipped task configs:
| Task | Layout | Cams | Action | Notable |
|---|---|---|---|---|
| `robotwin_unified_flex_3cam_384_1e-4` | `tshape_robotwin_384x320_uniform` | 3 | 14D | flex p=0.5, bs 8 |
| `libero_unified_flex_2cam224_32d_rotvec_1e-4` | `tshape_libero_2cam_448x512` | 2(+1 synth) | 32D rotvec | flex p=1.0, dino_temporal_stride=2, bs 6 |
| `yam_unified_flex_3cam_32d_rel_1e-4` | (from data cfg) | 3 | 32D rel (SE3+joint) | flex p=0.5, bs 9 |

### Launchers (`scripts/train_flexpi_{robotwin,libero,yam}.sh`)
Maintained in lockstep; each translates env vars → Hydra overrides in an `EXTRA_ARGS`
array appended to `accelerate launch scripts/train.py`.
- `TASK_CONFIG` → `task=...`; `FLEX_P_PRESENT_*`/`FLEX_P_J*` → `model.flex_joint.p_*`
  (in-script assignments — **edit the file**, `export` does not reach them).
- `DATASET_DIRS`/`TASK_NAMES`, `NUM_EPOCHS`, `VAL_SET_PROPORTION`,
  `DATASET_WEIGHTS`/`SAMPLES_PER_EPOCH`, `RESUME`, `PRETRAINED_CKPT`
  (+`PRETRAINED_CKPT_STRICT_SHAPE`), `GPUS`/`CUDA_DEVICES`, `WANDB_*`.
- RESUME/PRETRAINED_CKPT mutual exclusion enforced in-shell. LIBERO auto-resumes from its
  own output dir's latest `state/step_*` when neither is set. `train_flexpi_yam.sh` also
  runs the text-embed precompute for you.
- `ACCELERATE_CONFIG` swaps ZeRO-1 for ZeRO-2 / DDP / single-GPU / ZeRO-0.

### Run naming
`runs/<TASK_BASENAME>/[<TASK_NAMES_TAG>/]<RUN_ID>_<REGIME_TAG>[_<RUN_NAME>]`. `REGIME_TAG`
encodes the six flex probabilities `flex_pv..._pd..._pp..._jv..._jd..._jp...`; a 2D run
rewrites pointmap fields to `NA` + `_2d`; a warm start appends `_ft`. The trainer writes
`config.yaml` (fully resolved, `runtime.py:420`) and materializes
`checkpoints/{weights,state}/` + `eval/` there; `dataset_stats.json` is written by the
dataset builder into the run dir and is what eval reads back for normalization.

## 6. Prerequisites (before the first run)
1. `docs/INSTALL.md` — env, Wan2.2 weights (`DIFFSYNTH_MODEL_BASE_PATH`), DINOv3,
   ActionDiT backbone ckpt.
2. **Precompute text embeddings** (`scripts/precompute_text_embeds.py task=<TASK>`) —
   mandatory, keyed by prompt-hash + `context_len`, must cover every prompt in every
   dataset dir. See `03-dataloaders.md §6`.
3. Dataset in LeRobot v2.1 layout with canonical cam keys + (for 3D)
   `meta/camera_intrinsics.json`.

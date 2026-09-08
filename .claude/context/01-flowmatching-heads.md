# 01 — The per-modality flow-matching heads

**This is the surface the flow-map conversion touches.** It documents, per stream, the
input projector, the output head, the prediction target, the timestep/modulation, the
per-stream noising, the loss, and the inference step — plus the one shared scheduler that
ties them together. Read `05-flowmap-conversion.md` alongside this for the plan.

Line numbers are a snapshot; verify before editing.

---

## 0. The shared scheduler (rectified flow) — `models/schedulers/scheduler_continuous.py`

Every stream is denoised with **the same** `WanContinuousFlowMatchScheduler`, one
instance per stream (video/action/dino/pointmap), differing only in `shift`.

- **Forward (noising)** `add_noise` (`:75-82`): rectified-flow linear path
  `x_t = (1−σ)·x0 + σ·ε`, where `σ = t / num_train_timesteps ∈ [0,1]`.
- **Target** `training_target` (`:85-87`): **velocity** `v = ε − x0 = noise − sample`.
  (Timestep-independent — the straight-line rectified-flow target.)
- **Training timestep** `sample_training_t` (`:57-63`): draws `u~U(0,1)`, warps
  `σ = φ(u, shift) = shift·u / (1+(shift−1)·u)`, `t = σ·num_train_timesteps`. Larger
  `shift` biases toward higher noise. Config: video/dino/pointmap `train_shift=6.0`,
  action `train_shift=1.0`, `num_train_timesteps=1000`.
- **Loss weight** `training_weight` (`:65-73`): a Gaussian-in-t reweighting centered at
  mid-noise, normalized to mean 1. Applied per stream.
- **Inference step** `step` (`:119-153`): default **Euler** `x_{σ+δ} = x + v·δ` (δ<0,
  marching σ: 1→0). Optional **DPM-Solver++(2M)** branch (data-prediction form) engages
  only when a caller passes the current `timestep` — the joint loop does; the CUDA-graph
  action fast path stays pure Euler. `build_inference_schedule` (`:89-109`) returns the
  `(timesteps, deltas)` for K steps and resets multistep state.

> **Flow-map implication.** A flow map replaces this "predict `v`, take many small Euler
> steps" contract with "predict the *jump* from σ_s to σ_t directly (few or one step)".
> The scheduler is where the training target and the step rule live for **all** streams,
> so it is the natural place to add a flow-map target/step alongside the rectified-flow
> one, or a sibling `FlowMapScheduler`. Note the DINO x0→v conversion already proves the
> pattern of a head predicting one thing and adapting it to the scheduler's contract.

---

## 1. Video RGB stream `z^o` — the Wan DiT head

Operates in **Wan-VAE latent space** (`in_dim = out_dim = 48` channels). Owned by
`self.video_expert` (`WanVideoDiT`, `models/wan_video_dit.py`).

- **Input projector**: `self.video_expert.patch_embedding` =
  `nn.Conv3d(in_dim=48, hidden_dim=3072, kernel_size=(1,2,2), stride=(1,2,2))`
  (`wan_video_dit.py:373`), applied via `patchify` inside `pre_dit` (`:408-414`,
  `:518-629`). Produces `[B, Sv, 3072]` tokens laid out frame-major (`:609`), plus 3D
  RoPE freqs (`:611-615`).
- **Output head**: `self.video_expert.head` = `Head` (`wan_video_dit.py:297-313`):
  `LayerNorm(non-affine)` → FiLM shift/scale from the per-token `t_mod` →
  `nn.Linear(3072, out_dim·∏patch_size)` = `Linear(3072, 48·4=192)`, then `unpatchify`
  back to `[B, 48, F, H, W]` (`post_dit`, `:631-635`). **Predicts velocity `v`.**
- **Timestep/modulation**: per-token timesteps in `pre_dit`
  (`seperated_timestep=True` + `fuse_vae_embedding_in_latents=True`, `:546-559`).
  **Frame 0 is pinned to t=0** (`:555`) — it is the clean observation (first-frame
  anchor); only future frames carry noise. `time_embedding`/`time_projection` produce the
  6-chunk AdaLN `t_mod`.
- **Self-attention mask**: `build_video_to_video_mask` (`:482-516`), default
  `video_attention_mask_mode="first_frame_causal"` — future frames see the first frame;
  the first frame does not see the future.

## 2. Pointmap stream `z^p` — deep-copied Wan DiT head

Because depth→XYZ goes through the **same VAE**, the pointmap lives in the identical
48-channel latent space and **reuses the video DiT machinery via deep-copied modules**.

- **Input projector**: `self.pt_patch_embedding = copy.deepcopy(video_expert.patch_embedding)`
  (`flexpi.py:274-276`) — a *separate-weights* Conv3d. Applied in `_embed_pointmap`
  (`flexpi.py:440-446`) → `(tokens, ptpf, pt_meta{grid_size})`.
- **Output head**: `self.pt_head = copy.deepcopy(video_expert.head)` (`flexpi.py:276`).
  Applied in `_project_pointmap_out` (`flexpi.py:448-457`): `pt_head(tokens, pt_t)` then
  rearrange/unpatchify → `[B, 48, F, H, W]`. **Predicts velocity `v`.**
- **Timestep/modulation**: a per-stream `pt_t_mod` from `_build_stream_t_mod`
  (`flexpi.py:992-1007`) for the DiT blocks, plus a separate per-token `pt_t` fed to
  `pt_head` for FiLM (`flexpi.py:1211-1221`). Frame 0 pinned to t=0 (`:1210`).
- **RoPE**: reuses the video expert's 3D freqs, sliced over the pointmap grid
  (`_compute_pointmap_freqs_impl`, `flexpi.py:474-490`).
- **Encoder side**: `PointmapEncoder` (`models/pointmap_encoder.py`) is parameter-free —
  depth `[B,T,H,W]` uint16-mm → unproject with per-cam K → min-max normalize XYZ into
  `[-1,1]` → paste into the composite → VAE-encode. Always frozen.
- **Checkpoint keys**: `_POINTMAP_CKPT_KEYS = ("pt_patch_embedding","pt_head")`
  (`flexpi.py:2477`).
- When the run is pointmap-off (`_pointmap_globally_off`), `pt_patch_embedding`/`pt_head`
  are frozen and dispatch is skipped (`flexpi.py:308-340`).

## 3. DINO stream `d` — Linear head, **x0-prediction**

The odd one out: it operates in **DINO feature space** (`dino_dim = 768`, or `768·f² =
3072` under the 2×2 fold), with plain Linear projectors, and predicts **clean features
(x0)** rather than velocity.

- **Input projector**: `_embed_dino` (`flexpi.py:360-365`): raw DINO features
  `[B,768,F_d,N,1]` → reshape `[B,Sd,dino_dim]` → `self.dino_feature_norm`
  (`LayerNorm`, `flexpi.py:226`) → `self.dino_embedder` (`nn.Linear(dino_dim, 3072)`,
  `flexpi.py:227`).
- **Output head**: `self.dino_proj_out` = `nn.Linear(3072, dino_dim)` (`flexpi.py:228`),
  applied in `_project_dino_out` (`flexpi.py:367-369`). **Out dim = `dino_dim`.**
- **x-prediction (`dino_pred_x0: true`) — the existing "head predicts non-v" precedent:**
  - final layer **zero-initialized** (`_zero_init_dino_x0_head_`, `helpers/dino.py:181-189`)
    so `x̂0 ≈ 0` at step 0.
  - head output read as `x̂0`, converted to velocity by `_dino_x0_to_velocity`
    (`helpers/dino.py:154-178`): `v̂ = (x_t − x̂0)/σ`, `σ = t/num_train_timesteps` clamped
    to `_DINO_X0_SIGMA_MIN = 0.05`.
  - **Training** conversion: `flexpi.py:1299-1309` (then v-space MSE).
    **Inference** conversion: `flexpi.py:1590-1597` (inside `_predict_joint_noise_unified_impl`).
  - `dino_pred_x0=false` → `pred_dino` used directly as velocity (`flexpi.py:1311`).
- **Timestep/modulation**: `dino_token_timesteps` frame-0 pinned to 0 (`flexpi.py:1188`),
  `dino_t_mod = _build_stream_t_mod(...)` (`:1190`).
- **RoPE**: reuses video 3D freqs over the composite cam grid (`_compute_dino_freqs`,
  `flexpi.py:395-407`, `helpers/dino.py:71-114`).
- **Encoder side**: `DinoEncoder` (`models/dino_encoder.py`) — frozen DINOv3 ViT-B/16 via
  timm; per-cam 14×14 patches, pooled/folded per `dino_cam_patches` + `dino_pixel_unshuffle`.
- **Checkpoint keys**: `_DINO_CKPT_KEYS = ("dino_embedder","dino_proj_out","dino_feature_norm")`
  (`flexpi.py:2478`).

## 4. Action stream `a` — Linear head, velocity

Owned by `self.action_expert` (`ActionDiT`, `models/action_dit.py`).

- **Input projector**: `self.action_expert.action_encoder = nn.Linear(action_dim, 1024)`
  (`action_dit.py:74`), in `pre_dit` (`:228-302`). One token per action-horizon step;
  1D sinusoidal position added.
- **Output head**: `self.action_expert.head = nn.Linear(1024, action_dim)`
  (`action_dit.py:98`), applied in `post_dit` (`:304-305`). (There is also an unused
  `ActionHead` class with FiLM at `:18-29`; the wired head is the plain Linear.)
  **Predicts velocity `v`.**
- **Timestep**: sinusoidal `time_embedding` + `time_projection` → 6-chunk `t_mod`
  (`:282-284`); own 1D RoPE `freqs` (`:99`).
- **Scheduler**: `train_shift = infer_shift = 1.0` (no high-noise bias — action is
  low-dimensional and precise).

## 5. Head comparison (the conversion matrix)

| Stream | Space | Input proj | Output head | Target | Where |
|---|---|---|---|---|---|
| Video `z^o` | VAE latent (48ch) | Conv3d `[1,2,2]` 48→3072 | Wan `Head`: LN+FiLM+Linear→48·4, unpatchify | **v** | `wan_video_dit.py:373,297` |
| Pointmap `z^p` | VAE latent (48ch) | deep-copied Conv3d | deep-copied Wan `Head` | **v** | `flexpi.py:274-276,448-457` |
| DINO `d` | DINO feat (768/3072) | LN + Linear→3072 | Linear→768 | **x0** → v via `_dino_x0_to_velocity` | `flexpi.py:226-228,367` |
| Action `a` | action space (`action_dim`) | Linear→1024 | Linear→`action_dim` | **v** | `action_dit.py:74,98` |

Takeaways for the conversion:
- Video + pointmap share the exact same head *shape* (one is a deep copy of the other) and
  both live in VAE latent space → a flow-map head can be built once and instantiated
  twice.
- DINO already demonstrates a head predicting a non-velocity quantity that is adapted to
  the shared scheduler contract. A flow-map head can follow the same adapter pattern.
- Action is the low-D, latency-critical stream with its own fast KV-cache inference path;
  it is the highest-value target for few-step flow-map inference and the one with the most
  specialized step loop.

## 6. Training forward & loss — `_base_training_loss` (`flexpi.py:1110-1383`)

1. `build_inputs(sample)` (`flexpi.py:884`, super `backbone.py:680`): VAE-encode RGB →
   `input_latents`; DINOv3-encode on the fly; VAE-encode pointmap composite →
   `pointmap_raw` (unless pointmap-off).
2. **Per-stream noising** — each stream independently: `noise=randn_like`,
   `t=train_*_scheduler.sample_training_t(B)`, `x_t=add_noise(x0,noise,t)`,
   `target=training_target(x0,noise,t)`; **frame-0 re-clamped clean**:
   - video `:1121-1126`, action `:1129-1132`, dino `:1135-1139`, pointmap `:1147-1157`
     (all `None` when pointmap-off, `:1142-1145`).
3. `pre_dit` for video/action (`:1160-1167`); `_embed_dino` (`:1180`); `_embed_pointmap`
   (`:1201`); per-stream `t_mod` + RoPE.
4. **Flex application**: token zeroing (`_flex_zero_absent_*`, `:1175,1186,1204`); mask
   build (`_build_mot_attention_mask_unified`); merge streams
   (`_merge_aux_into_video_stream`, `:1224-1229`).
5. **MoT forward** (`:1242-1253`): `self.mot(embeds_all={"video":merged,"action":action},
   attention_mask, freqs_all, context_all, t_mod_all, sub_stream_lens, sub_stream_self_masks)`.
6. **Split + heads** (`:1256-1267`): slice merged out → video/dino/pointmap;
   `pred_video=video_expert.post_dit`, `pred_action=action_expert.post_dit`,
   `pred_dino=_project_dino_out`, `pred_pointmap=_project_pointmap_out`.
7. **Four MSE-vs-velocity losses**, each `training_weight`-weighted and flex-reduced
   (`_flex_reduce_per_sample_loss`, `:1081-1104` — absent-non-cross-modal samples
   contribute 0):
   - video `:1269-1280` (`_compute_video_loss_per_sample`, `backbone.py:1005`)
   - action `:1282-1290` (`action_is_pad`-masked)
   - dino `:1292-1338` (x0→v convert first if `dino_pred_x0`; drop ff_d frames;
     per-frame pad mask)
   - pointmap `:1340-1369` (skip ff_p; reuse video per-sample loss)
8. **Combine** (`:1371-1382`):
   `total = λ_v·L_v + λ_a·L_a + λ_d·L_d + λ_p·L_p` → `(total, loss_dict)`.

`training_loss` (`flexpi.py:2945`) is the flex wrapper: samples `sample_flex_batch_flags`
into `self._batch_flex`, calls `_base_training_loss`, clears in `finally`.

## 7. Inference stepping — where the ODE loop lives

- **Fast action-only** `_base_infer_action` (`flexpi.py:1610-1872`): prefill the video KV
  cache with first-frame video+DINO+pointmap anchors (all t=0), then Euler action ODE
  (`:1856-1868`: `_predict_action_noise_with_cache` → `infer_action_scheduler.step`).
  Loop-compile variant `_run_action_prefill_denoise_loop` (`:1922`). K default 20.
- **Full joint** `_infer_action_joint` (`flexpi.py:3266-3695`): per step,
  `_predict_joint_noise_unified` → one masked MoT forward → `(pred_video, pred_dino,
  pred_pointmap, pred_action)` (`_predict_joint_noise_unified_impl`, `:1468-1603`); then
  per-active-stream `scheduler.step` + frame-0 re-clamp (`:3641-3664`). DINO x0→v inside
  the impl (`:1590-1597`). Accelerations: `StepSkipController`, CUDA-graph loop, FlexAttention
  BlockMask, TensorRT engines (`models/inference_opt/`).
- **Full 4-stream rollout** `_base_infer_joint` (`flexpi.py:2165-2421`): steps all four,
  decodes video, optional action-only consistency check.

> **Flow-map conversion — the step loops to change.** Any few-step flow-map inference
> replaces the per-stream `scheduler.step` calls in these two loops
> (`_base_infer_action` action Euler at `flexpi.py:1856-1868`; `_infer_action_joint`
> per-stream steps at `flexpi.py:3641-3664`) and the joint-noise prediction adapter
> (`_predict_joint_noise_unified_impl`, `flexpi.py:1468-1603`). Training-side, the target
> construction is in `_base_training_loss` steps 2 + 7 above and in the scheduler's
> `training_target`. See `05-flowmap-conversion.md`.

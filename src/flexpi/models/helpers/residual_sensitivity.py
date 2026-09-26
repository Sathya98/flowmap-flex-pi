"""Observed-future sensitivity of a deterministic, conditional action replay.

The independent variable is the first future VAE slot (observed at action 16
for the released LIBERO checkpoint). All other visual slots, initial action
noise, text and proprioception are fixed. Actions are re-denoised with clean
visual conditioning; this is not the Jacobian between two sampler outputs.
"""
import json
import hashlib
from pathlib import Path
import time

import torch
from torch.autograd import forward_ad as fw


def directional_scores(fn, predicted, observed, *, probes=4, seed=2026, eps=1e-12):
    """Exact forward AD; isotropic unit Rademacher probes on identical support."""
    if predicted.shape != observed.shape or probes < 1:
        raise ValueError('Matching residual shapes and at least one probe required')
    residual = observed.float() - predicted.float()
    magnitude = residual.norm()
    if not torch.isfinite(residual).all():
        raise ValueError('Nonfinite observed residual')
    generator = torch.Generator(device=predicted.device).manual_seed(seed)

    def evaluate(direction):
        with torch.no_grad(), fw.dual_level():
            primal, tangent = fw.unpack_dual(fn(fw.make_dual(predicted, direction)))
            if tangent is None:
                raise RuntimeError('Action replay disconnected from future latent')
            if not torch.isfinite(tangent).all() or not torch.isfinite(primal).all():
                raise RuntimeError('Nonfinite action replay/JVP')
            return float(tangent.float().square().sum()), primal.detach().clone()

    s_res, primal = evaluate((residual / magnitude.clamp_min(eps)).to(predicted))
    random_scores = []
    for _ in range(probes):
        z = torch.randint(0, 2, predicted.shape, device=predicted.device,
                          generator=generator).float().mul_(2).sub_(1)
        z = (z / z.norm()).to(predicted)
        score, _ = evaluate(z)
        random_scores.append(score)
    s_rand = sum(random_scores) / probes
    return dict(E=float(magnitude), S_res=s_res, S_rand=s_rand,
                R=s_res / (s_rand + eps) if magnitude > eps else None,
                random_scores=random_scores, random_probes=probes,
                residual_nonzero=bool(magnitude > eps),
                random_baseline_resolved=s_rand > eps,
                residual_dimensions=predicted.numel(), epsilon=eps), primal


class ResidualSensitivityProbe:
    def __init__(self, model, cfg, episode, action_channels):
        self.model, self.cfg, self.episode = model, cfg, episode
        self.channels = list(action_channels)
        self.rows, self.chunk = [], -1
        self.pending = None
        self.ratio = int(cfg.data.train.action_video_freq_ratio)
        self.observe_at = self.ratio * model.vae.temporal_downsample_factor
        self.probes = int(cfg.EVALUATION.get('residual_random_probes', 4))
        self.max_chunks = int(cfg.EVALUATION.get('residual_max_chunks', 3))
        indices = cfg.EVALUATION.get('residual_chunk_indices', None)
        self.chunk_indices = None if indices is None else set(int(i) for i in indices)
        self.probe_seed = int(cfg.EVALUATION.get('residual_probe_seed', 2026 + 1000*episode))
        self.metadata = dict(cfg.EVALUATION.get('residual_metadata', {}))
        self.path = Path(cfg.EVALUATION.output_dir) / 'residual_sensitivity.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if model.flow_map.enabled or cfg.EVALUATION.get('use_action_ensembler', False):
            raise ValueError('Probe requires original FM policy without action ensembling')
        if cfg.EVALUATION.get('dynamic_step_skip', False) or cfg.EVALUATION.get('torch_compile', False):
            raise ValueError('Probe requires eager denoising without step skipping')
        if int(cfg.EVALUATION.replan_steps) <= self.observe_at:
            raise ValueError('Execute beyond the residual observation to assess subsequent actions')
        if cfg.EVALUATION.get('visualize_future_video', False):
            raise ValueError('Use latent diagnostics without the separate visualization sampler')
        if episode == 0:
            from omegaconf import OmegaConf
            OmegaConf.save(cfg, self.path.parent / 'config.yaml')
            root = Path(__file__).resolve().parents[4]
            sources = {}
            for relative in ('src/flexpi/models/helpers/residual_sensitivity.py',
                             'experiments/libero/eval_libero_single.py',
                             'scripts/summarize_residual_sensitivity.py',
                             'src/flexpi/models/flexpi.py',
                             'src/flexpi/models/helpers/attention.py',
                             'src/flexpi/models/helpers/normalization.py'):
                content = (root / relative).read_bytes()
                target = self.path.parent / 'source_snapshot' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                sources[relative] = hashlib.sha256(content).hexdigest()
            manifest = dict(checkpoint=str(cfg.ckpt), torch_version=torch.__version__,
                            gpu=torch.cuda.get_device_name(), source_sha256=sources,
                            score='squared L2 JVP norm', random_directions='unit Rademacher',
                            observed_actions=self.observe_at)
            (self.path.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    @torch.no_grad()
    def predict(self, kwargs):
        self.chunk += 1
        self.pending = None
        if self.chunk >= self.max_chunks or (self.chunk_indices is not None and self.chunk not in self.chunk_indices):
            return self.model.infer_action(**kwargs)
        if not kwargs.get('joint_video') or kwargs.get('joint_dino') or kwargs.get('joint_pointmap'):
            raise ValueError('Pilot requires joint video/actions with DINO and pointmap anchors')
        if kwargs['action_horizon'] <= self.observe_at:
            raise ValueError('No remaining actions after the first observed VAE slot')
        captured = {}
        original = self.model._predict_joint_noise_unified

        def capture(**inputs):
            if not captured:
                captured.update({k: v.detach().clone() if torch.is_tensor(v) else v
                                 for k, v in inputs.items()})
            return original(**inputs)

        self.model._predict_joint_noise_unified = capture
        try:
            prediction = self.model.infer_action(**dict(kwargs, return_stream_latents=True))
        finally:
            self.model._predict_joint_noise_unified = original
        self.pending = dict(inputs=captured, prediction=prediction, kwargs=kwargs,
                            frames=[kwargs['input_image'].detach().clone()], step=0,
                            record=None)
        return prediction

    @torch.no_grad()
    def observe(self, image_fn, done):
        if self.pending is None:
            return
        p = self.pending
        p['step'] += 1
        if p['step'] <= self.observe_at and p['step'] % self.ratio == 0:
            p['frames'].append(image_fn())
        if p['step'] == self.observe_at:
            if done:
                # Already-successful observations cannot predict future failure.
                self.pending = None
                return
            self.measure()
        if p['record'] is not None:
            p['record']['chunk_success'] = bool(done)
            p['record']['executed_after_observation'] = p['step'] - self.observe_at

    @torch.no_grad()
    def measure(self):
        start = time.perf_counter()
        m, p = self.model, self.pending
        actual = m._encode_video_latents(torch.stack(p['frames'], dim=2), tiled=False)
        future = p['prediction']['video_latents'].to(device=m.device, dtype=m.torch_dtype)
        if actual.shape[2] != 2 or future.shape[2] < 3:
            raise ValueError('Expected first observed future slot and a later predicted slot')
        predicted = future[:, :, 1:2].clone()
        inputs = dict(p['inputs'])
        action_noise = inputs['latents_action'].clone()
        for stream in ('video', 'dino', 'pointmap'):
            inputs['timestep_' + stream] = torch.zeros_like(inputs['timestep_' + stream])
        steps, deltas = m.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=p['kwargs']['num_inference_steps'], device=m.device,
            dtype=m.torch_dtype, shift_override=p['kwargs']['sigma_shift'])

        def replay(slot):
            visual = torch.cat((future[:, :, :1], slot, future[:, :, 2:]), dim=2)
            action = action_noise.clone()
            for step, delta in zip(steps, deltas):
                # Pure Euler update: avoid mutable multistep scheduler history.
                velocity = m._predict_joint_noise_unified_impl(**dict(inputs,
                    latents_video=visual, latents_action=action,
                    timestep_action=step.reshape(1)))[3]
                action = action + delta * velocity
            return action[:, self.observe_at:, self.channels]

        saved = (m.joint_video, m.joint_dino, m.joint_pointmap)
        m.joint_video, m.joint_dino, m.joint_pointmap = True, False, False
        try:
            scores, baseline = directional_scores(replay, predicted, actual[:, :, 1:2],
                probes=self.probes, seed=self.probe_seed + self.chunk)
        finally:
            m.joint_video, m.joint_dino, m.joint_pointmap = saved
        original = p['prediction']['action'][self.observe_at:, self.channels].to(baseline)
        prediction_bytes = p['prediction']['video_latents'].contiguous().numpy().tobytes()
        row = dict(episode=self.episode, chunk=self.chunk, observed_actions=self.observe_at,
            **self.metadata, probe_seed=self.probe_seed + self.chunk,
            prediction_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
            remaining_actions=baseline.shape[1], action_channels=self.channels,
            action_units='normalized active channels', stream='video_vae',
            map='clean_visual_conditioned_action_replay', **scores,
            replay_vs_original_rms=float((baseline[0].float()-original.float()).square().mean().sqrt()),
            seconds=time.perf_counter()-start, chunk_success=False,
            executed_after_observation=0)
        self.rows.append(row)
        p['record'] = row
        p['frames'] = []
        # Persist immediately; terminal labels are written separately at finish.
        with self.path.open('a') as handle:
            handle.write(json.dumps(dict(row, event='measurement'), allow_nan=False)+'\n')
        print('[residual]', json.dumps(row), flush=True)

    def finish(self, success):
        with self.path.open('a') as handle:
            for row in self.rows:
                handle.write(json.dumps(dict(row, event='labeled', episode_success=bool(success)),
                                        allow_nan=False)+'\n')
        self.pending = None

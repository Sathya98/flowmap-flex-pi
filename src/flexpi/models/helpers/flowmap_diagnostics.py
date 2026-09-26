"""Deterministic flow-map probes; never update weights or training defaults."""
from dataclasses import replace
import math
import torch

from .dino import _DINO_X0_SIGMA_MIN, _dino_x0_to_velocity
from .flowmap import STREAMS
from .flowmap_training import training_loss, reduce_stream


def paired_time_proposals(count, seed=2026):
    """Common random numbers: legacy conditional proposal vs uniform triangle.

    For the full denoising triangle: p_conditional(s,t)=1/s, p_triangle(s,t)=2.
    The inverse marginal CDF for the latter is s=sqrt(u), then t=s*(1-w).
    This is diagnostic-only; no change to the training sampler.
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(count, generator=g).clamp_min(1e-6)
    w = torch.rand(count, generator=g).clamp_min(1e-6)
    return {'conditional': (u, u * (1-w)),
            'uniform_triangle': (u.sqrt(), u.sqrt() * (1-w))}


def make_noise(inputs, seed):
    clean = dict(video=inputs['input_latents'], dino=inputs['dino_features'],
                 pointmap=inputs['pointmap_raw'], action=inputs['action'])
    generator = torch.Generator(device=clean['action'].device).manual_seed(seed)
    return {k: torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
            for k, v in clean.items() if v is not None}


def lmd_probe(model, sample, inputs, source, target, noise_seed=2026):
    """Exact LMD residual on the same fixed cases as the PFMM reference probe.

    The frozen teacher stays unchanged. Forward AD works under no_grad; this
    measures objective values, while the training pilot tests parameter backward.
    """
    if model.flow_map_teacher is None:
        raise ValueError('LMD diagnostics require the frozen released teacher')
    original = model.flow_map
    model.flow_map = replace(original, objective='lmd', dt_method='ad',
                             detach_derivatives=False, distill_diagonal_weight=0.0)
    try:
        times = tuple(torch.full((inputs['action'].shape[0],), v,
                                device=inputs['action'].device) for v in (source, target))
        with torch.no_grad():
            loss, metrics = training_loss(model, sample, times=times,
                noise=make_noise(inputs, noise_seed), prepared_inputs=inputs)
        return dict(loss=float(loss), metrics=metrics,
                    finite=all(math.isfinite(v) for v in [float(loss), *metrics.values()]))
    finally:
        model.flow_map = original


def pfmm_probe(model, sample, inputs, source, target, noise_seed=2026, teacher_steps=16):
    """Measure the production loss with fixed encoded data, noise and times.

    Also return detached targets for paired integration-convergence checks.
    No optimizer, no backward, no inference-mode tensors (JVP-compatible).
    """
    if model.flow_map.objective != 'pfmm' or model.flow_map.self_distillation:
        raise ValueError('This numerical teacher-convergence probe requires PFMM')
    device = inputs['action'].device
    b = inputs['action'].shape[0]
    if b != 1:
        raise ValueError('Probe one example at a time to preserve per-time diagnostics')
    s, t = (torch.full((b,), v, device=device) for v in (source, target))
    captured = {}
    def capture_head(_module, _args, output):
        captured['head'] = output.detach()
    hook = model.dino_proj_out.register_forward_hook(capture_head)
    old_steps = model.flow_map.teacher_steps
    model.flow_map.teacher_steps = teacher_steps
    trace = {}
    try:
        with torch.no_grad():
            loss, metrics = training_loss(model, sample, times=(s,t),
                noise=make_noise(inputs, noise_seed), prepared_inputs=inputs, diagnostics=trace)
    finally:
        hook.remove()
        model.flow_map.teacher_steps = old_steps
    report = dict(source=source, target=target, interval=source-target,
                  noise_seed=noise_seed, teacher_steps=teacher_steps,
                  loss=float(loss), metrics=metrics,
                  teacher_queries=trace['teacher_queries'], streams={})
    for name, residual, pred, reference in zip(trace['active'], trace['residuals'],
            trace['student_velocity'], trace['target_velocity']):
        mse = lambda value: float(reduce_stream(model, name, value, inputs))
        report['streams'][name] = dict(velocity_residual_mse=mse(residual),
            map_endpoint_residual_mse=mse((source-target)*residual),
            student_velocity_mse=mse(pred), teacher_velocity_mse=mse(reference))
    if model.dino_pred_x0 and 'dino' in trace['active']:
        i = trace['active'].index('dino')
        x, reference, pred = trace['state'][i], trace['target_velocity'][i], trace['student_velocity'][i]
        # Head output is [B,F*tokens,D]; unpack exactly as FlexPi does.
        raw = captured['head'].reshape(b, x.shape[2], x.shape[3], model.dino_dim)
        raw = raw.permute(0,3,1,2).unsqueeze(-1).float()
        floor = max(source, _DINO_X0_SIGMA_MIN)
        fp32_v = _dino_x0_to_velocity(x.to(model.torch_dtype), raw,
            s.view(-1,1,1,1,1)*model.infer_dino_scheduler.num_train_timesteps,
            model.infer_dino_scheduler.num_train_timesteps)
        mse = lambda value: float(reduce_stream(model, 'dino', value, inputs))
        # Use the rounded state actually supplied to the head; this measures
        # conversion/output rounding separately from neural-network BF16 error.
        implied_target = x.to(model.torch_dtype).float() - floor*reference
        report['dino_parameterization'] = dict(sigma_floor=_DINO_X0_SIGMA_MIN,
            squared_error_amplification=1/floor**2,
            head_vs_implied_teacher_endpoint_mse=mse(raw-implied_target),
            head_vs_clean_data_mse=mse(raw-inputs['dino_features'].float()),
            fp32_conversion_residual_mse=mse(fp32_v-reference),
            conversion_output_rounding_mse=mse(fp32_v-pred))
    report['finite'] = all(math.isfinite(v) for v in [report['loss'], *metrics.values()])
    targets = {name: value.detach().cpu() for name,value in
               zip(trace['active'], trace['target_velocity'])}
    return report, targets


def target_difference(model, inputs, first, reference, interval):
    result = {}
    for name in first:
        a, b = (value[name].to(inputs['action'].device) for value in (first, reference))
        delta = float(reduce_stream(model, name, a-b, inputs))
        scale = float(reduce_stream(model, name, b, inputs))
        result[name] = dict(velocity_difference_mse=delta,
            relative_velocity_difference_rms=math.sqrt(delta/max(scale,1e-20)),
            endpoint_difference_mse=interval**2*delta)
    return result

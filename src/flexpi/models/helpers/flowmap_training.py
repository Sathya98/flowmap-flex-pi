"""FlexPi's joint flow-map objective; encoders and padding conventions are reused."""
from contextlib import contextmanager, nullcontext
import torch

from .flowmap_self import slice_batch
from .flowmap import STREAMS, affine_flow_map, map_residuals, training_time_pairs


_DEFER_METRICS = False


@contextmanager
def deferred_metrics():
    """Loss metrics stay 0-d GPU tensors instead of Python floats: no host sync
    (needed inside CUDA-graph capture; read them after the step)."""
    global _DEFER_METRICS
    previous, _DEFER_METRICS = _DEFER_METRICS, True
    try:
        yield
    finally:
        _DEFER_METRICS = previous


def _metric(value):
    value = value.detach().mean()
    return value if _DEFER_METRICS else float(value)


def predict_streams(model, state, s, t, context, context_mask, fuse, active):
    """All heads use the same sigma endpoints, with each head's own timestep units."""
    kwargs = {}
    for name in STREAMS:
        kwargs['latents_' + name] = state[name].to(model.torch_dtype) if state[name] is not None else None
        n = getattr(model, 'train_' + name + '_scheduler').num_train_timesteps
        kwargs['timestep_' + name] = s * n if name in active else torch.zeros_like(s)
        kwargs['timestep_delta_' + name] = (t - s) * n if name in active else torch.zeros_like(s)
    out = model._predict_joint_noise_unified_impl(
        **kwargs, context=context, context_mask=context_mask, fuse_vae_embedding_in_latents=fuse)
    result = {}
    for name, value in zip(STREAMS, out):
        if value is None:
            result[name] = None
        elif name != 'action':
            # Anchors are fixed coordinates of the joint ODE, not model predictions.
            result[name] = torch.cat((torch.zeros_like(value[:, :, :1]), value[:, :, 1:]), dim=2).float()
        else:
            result[name] = value.float()
    return result


def reduce_stream(model, name, residual, inputs, keep_batch=False):
    """Batch-mean stream loss; ``keep_batch`` returns the ``[B]`` per-example losses.

    Per example, an absent stream without cross-modal prediction contributes zero,
    which is the batch-1 case of the flex-aware present-sample mean.
    """
    b = residual.shape[0]
    if name == 'action':
        error = residual.float().square()
        dims_pad = inputs.get('action_dim_is_pad')
        if dims_pad is not None:
            valid = (~dims_pad).to(error)
            if valid.ndim == 2:
                valid = valid[:, None, :]
            per_step = (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        else:
            per_step = error.mean(-1)
        per_example = model._masked_loss_reduction(per_step, inputs['action_is_pad'])
        return per_example if keep_batch else per_example.mean()
    residual = residual[:, :, 1:]
    if residual.shape[2] == 0:
        raise ValueError(f"Generated {name} stream has no future frames")
    if name in ('video', 'pointmap'):
        per_sample = model._compute_video_loss_per_sample(
            residual, torch.zeros_like(residual), inputs['image_is_pad'], False)
    else:
        per_frame = residual.square().mean(dim=(1, 3, 4))
        pad = inputs['image_is_pad']
        if pad is not None:
            pad = model._aux_per_frame_is_pad(pad, model.dino_temporal_stride, keep_far=model.dino_stride_keep_far)[:, 1:]
        per_sample = model._masked_loss_reduction(per_frame, pad)
    flag = {'video': 'v', 'dino': 'd', 'pointmap': 'p'}[name]
    bf = model._batch_flex
    if keep_batch:
        if bf is not None and not getattr(bf, 'cm_' + flag):
            per_sample = per_sample * getattr(bf, 'present_' + flag).to(per_sample)
        return per_sample
    return model._flex_reduce_per_sample_loss(
        per_sample, torch.ones(b, device=per_sample.device),
        present_mask=None if bf is None else getattr(bf, 'present_' + flag),
        cross_modal_active=False if bf is None else getattr(bf, 'cm_' + flag))


def training_loss(model, sample, tiled=False, *, times=None, noise=None,
                  prepared_inputs=None, diagnostics=None, _self_term=None):
    """Training objective, with opt-in deterministic inputs for diagnostic probes.

    Self-distillation uses disjoint diagonal/off-diagonal examples; the trainer
    supplies the effective-batch mixture mask. Prepared inputs must be rebuilt
    after loading different student conditioning weights.
    """
    cfg = model.flow_map
    teacher = model.flow_map_teacher
    if cfg.needs_teacher and teacher is None:
        raise RuntimeError("Distillation requires flow_map.teacher_checkpoint; construct the teacher before training")
    if 'pointmap' in cfg.streams and model._pointmap_globally_off:
        raise ValueError("Cannot train pointmap flow maps with enable_pointmap=false")
    if cfg.dt_method.endswith('_fd') and (model.torch_dtype != torch.float32 or cfg.objective in ('emd', 'esd')):
        raise ValueError("FD requires a float32 model and a Lagrangian objective; use ad for bf16/Eulerian")
    inputs = model.build_inputs(sample, tiled=tiled) if prepared_inputs is None else prepared_inputs
    if (cfg.self_distillation or cfg.uses_time_weighting) and _self_term is None:
        # The learned time weight acts on each example's loss, so the batch is
        # reduced per example (the mean of batch-1 losses); diagonal examples
        # skip the expensive JVP. Stratified microsteps take a single branch,
        # so this is normally one batched call; a mixed batch makes two.
        b = inputs['action'].shape[0]
        diagonal_mask = sample.get('_flowmap_diagonal_mask') if cfg.self_distillation else torch.zeros(b, dtype=torch.bool)
        if diagonal_mask is None:
            # Standalone probes may provide deterministic off-diagonal times.
            diagonal_mask = (torch.zeros(b, dtype=torch.bool) if times is not None
                             else torch.rand(b) < cfg.self_diagonal_fraction)
        if diagonal_mask.shape != (b,):
            raise ValueError('Self-distillation mixture mask must match batch size')
        diagonal_mask = diagonal_mask.cpu()
        old_flags = model._batch_flex
        losses, averaged = [], {}
        try:
            for term, chosen in (('diagonal', diagonal_mask), ('offdiagonal', ~diagonal_mask)):
                index = chosen.nonzero().flatten()
                if not len(index):
                    continue
                pick = ((lambda value: value) if len(index) == b
                        else (lambda value, index=index: slice_batch(value, index, b)))
                model._batch_flex = pick(old_flags)
                loss, metrics = training_loss(
                    model, pick(sample), tiled=tiled,
                    times=None if times is None else tuple(pick(v) for v in times),
                    noise=None if noise is None else pick(noise),
                    prepared_inputs=pick(inputs), diagnostics=diagnostics, _self_term=term)
                losses.append(loss)
                for key, value in metrics.items():
                    averaged[key] = averaged.get(key, 0.) + value * len(index) / b
            return torch.cat(losses).mean(), averaged
        finally:
            model._batch_flex = old_flags
    if inputs['first_frame_latents'] is None:
        raise ValueError("Flow-map training requires clean first-frame visual anchors")
    active = tuple(name for name in STREAMS if name in cfg.streams)
    clean = dict(video=inputs['input_latents'], dino=inputs['dino_features'],
                 pointmap=None if model._pointmap_globally_off else inputs['pointmap_raw'], action=inputs['action'])
    clean['video'] = torch.cat((inputs['first_frame_latents'], clean['video'][:, :, 1:]), dim=2)
    for name in STREAMS:
        if name not in active and clean[name] is not None:
            clean[name] = clean[name][:, :, :1]
    b = clean['action'].shape[0]
    if _self_term == 'diagonal':
        # Independent uniform FM times, never the triangle's biased source.
        s = torch.rand(b, device=clean['action'].device, dtype=torch.float32)
        t = s
    elif times is None:
        # The trainer sets model._flowmap_update (optimizer step) for scheduled sampling.
        s, t = training_time_pairs(cfg, b, clean['action'].device, getattr(model, '_flowmap_update', None))
    else:
        s, t = (a.to(device=clean['action'].device, dtype=torch.float32) for a in times)
        if s.shape != (b,) or t.shape != (b,) or not (s.is_cuda and torch.cuda.is_current_stream_capturing()) and not bool(
                (torch.isfinite(s) & torch.isfinite(t) & (s > 0) & (s <= 1) &
                 (t >= 0) & (t <= s) & (s-t <= cfg.strip_width + 1e-6)).all()):
            raise ValueError('Diagnostic times require batch-shaped 0 <= t <= s <= 1 within the strip')
    if cfg.dt_method.endswith('_fd') and _self_term != 'diagonal':
        # Keep both target-time stencil points within the denoising domain.
        s = s.clamp_min(3 * cfg.fd_eps)
        t = t.clamp_min(cfg.fd_eps)
        t = torch.minimum(t, s - cfg.fd_eps)
    if noise is None:
        noise = {name: torch.randn_like(clean[name]) for name in active}
    elif any(name not in noise or noise[name].shape != clean[name].shape for name in active):
        raise ValueError('Diagnostic noise must match every active stream shape')
    x = []
    targets = []
    for name in active:
        data = clean[name].float()
        target = noise[name].float() - data
        if name != 'action':
            target = torch.cat((torch.zeros_like(target[:, :, :1]), target[:, :, 1:]), dim=2)
        targets.append(target)
        x.append(affine_flow_map(data, target, torch.zeros_like(s), s))
    x = tuple(x)
    context, mask = inputs['context'], inputs['context_mask']
    teacher_context, teacher_mask = context.detach(), mask
    if teacher is not None and teacher.proprio_encoder is not None:
        with torch.no_grad():
            teacher_context, teacher_mask = teacher._append_proprio_to_context(
                context[:, :-1].detach(), mask[:, :-1], sample['proprio'][:, 0].to(context.device))
    def predict(state, lo, hi, owner=model):
        full = dict(clean)
        full.update(zip(active, state))
        ctx, cmask = (context, mask) if owner is model else (teacher_context, teacher_mask)
        result = predict_streams(owner, full, lo, hi, ctx, cmask,
                                 inputs['fuse_vae_embedding_in_latents'], active)
        return tuple(result[name] for name in active)
    def velocity(state, time):
        value = predict(state, time, time, model if cfg.self_distillation else teacher)
        if diagnostics is not None:
            diagnostics.setdefault('teacher_queries', []).append(dict(
                sigma=time.detach().cpu().tolist(),
                velocity_mse={name: float(reduce_stream(model, name, v.detach(), inputs))
                              for name, v in zip(active, value)}))
        return value
    previous_flags = teacher._batch_flex if teacher is not None else None
    if teacher is not None:
        teacher.eval()
        teacher._batch_flex = model._batch_flex
    try:
        # FD must not silently become bf16 under the trainer's autocast context.
        guard = torch.autocast(device_type=x[0].device.type, enabled=False) if cfg.dt_method.endswith('_fd') else nullcontext()
        with guard:
            residuals = (tuple(torch.zeros_like(v) for v in x) if _self_term == 'diagonal'
                         else map_residuals(predict, velocity, x, s, t, cfg, diagnostics=diagnostics))
            if diagnostics is not None:
                diagnostics.update(inputs=inputs, active=active, state=x, s=s, t=t,
                                   residuals=tuple(r.detach() for r in residuals))
            diagonal_weight = (cfg.diagonal_weight if _self_term == 'diagonal' else 0.) if cfg.self_distillation else cfg.distill_diagonal_weight
            diagonal = predict(x, s, s) if diagonal_weight else None
            # Split terms return per-example losses for the caller to average.
            per_example = _self_term is not None
            total = x[0].new_zeros(b if per_example else ())
            metrics = {}
            for i, name in enumerate(active):
                residual = residuals[i]
                if cfg.objective == 'pfmm' and cfg.pfmm_loss_space == 'endpoint':
                    residual = residual * (s-t).view(b, *([1] * (residual.ndim-1)))
                off = reduce_stream(model, name, residual, inputs, per_example)
                diag = (reduce_stream(model, name, diagonal[i] - targets[i], inputs, per_example)
                        if diagonal is not None else off.new_zeros(()))
                weight = getattr(model, 'loss_lambda_' + name)
                total = total + weight * (cfg.map_weight * off + diagonal_weight * diag)
                metrics['loss_flowmap_' + name] = _metric(off)
                metrics['loss_diagonal_' + name] = _metric(diag)
            if cfg.has_time_schedule:
                metrics['time_max_jump'] = cfg.strip_width_at(getattr(model, '_flowmap_update', 0))
            if cfg.self_distillation or cfg.uses_time_weighting:
                metrics['loss_unweighted'] = _metric(total)
                if cfg.self_distillation:
                    metrics['self_diagonal_fraction'] = float(_self_term == 'diagonal')
                if cfg.uses_time_weighting:
                    logvar = model.flow_map_loss_weight(s, t)
                    total = logvar.neg().exp() * total + logvar
                    metrics['time_logvar'] = _metric(logvar)
            return total, metrics
    finally:
        if teacher is not None:
            teacher._batch_flex = previous_flags

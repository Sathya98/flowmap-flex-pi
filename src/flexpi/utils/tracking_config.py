"""Explicit hyperparameter payload for external tracking; paths stay local."""
from omegaconf import OmegaConf


def tracking_config(cfg):
    fields = (
        'max_steps', 'batch_size', 'gradient_accumulation_steps', 'learning_rate',
        'weight_decay', 'adam_betas', 'max_grad_norm', 'seed', 'mixed_precision',
        'fused_adamw', 'save_every', 'eval_every', 'eval_nfes', 'eval_namespace',
        'model.flow_map.objective', 'model.flow_map.streams',
        'model.flow_map.lmd_teacher_gradient', 'model.flow_map.dt_method',
        'model.flow_map.detach_derivatives', 'model.flow_map.time_sampling',
        'model.flow_map.distill_ema', 'model.flow_map.ema_decays',
        'model.flow_map.distill_learned_time_weighting',
    )
    result = {}
    for field in fields:
        value = OmegaConf.select(cfg, field)
        if value is not None:
            result[field] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return result

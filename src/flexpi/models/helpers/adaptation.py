"""Checkpoint-compatible low-rank and nonlinear projection adapters."""
import math
import copy
import torch
from torch import nn
import torch.nn.functional as F


def clone_teacher(model):
    """Independent denoisers, shared frozen encoders and immutable layout data."""
    memo = {id(getattr(model, name)): getattr(model, name)
            for name in model.FROZEN_MODULES if getattr(model, name, None) is not None}
    # Layout slot maps are immutable mappingproxy objects (not pickleable).
    for name in ('_layout', '_slot_key_map'):
        value = getattr(model, name, None)
        if value is not None:
            memo[id(value)] = value
    teacher = copy.deepcopy(model, memo)
    teacher.flow_map.enabled = False
    teacher.eval().requires_grad_(False)
    return teacher


def validate_teacher_payload(teacher, payload):
    """A distillation teacher must not silently retain random WAM weights."""
    required = {'mot': teacher.mot, 'dino_embedder': teacher.dino_embedder,
                'dino_feature_norm': teacher.dino_feature_norm}
    if 'dino' in teacher.flow_map.streams:
        required['dino_proj_out'] = teacher.dino_proj_out
    if not teacher._pointmap_globally_off:
        required['pt_patch_embedding'] = teacher.pt_patch_embedding
    if 'pointmap' in teacher.flow_map.streams:
        required['pt_head'] = teacher.pt_head
    if teacher.proprio_encoder is not None:
        required['proprio_encoder'] = teacher.proprio_encoder
    for name, module in required.items():
        saved = payload.get(name, {})
        missing = [key for key in module.state_dict() if key not in saved
                   and 'time_embedding_delta.' not in key
                   and not key.endswith(('adapter_down', 'adapter_up'))]
        if missing:
            raise ValueError(f"FM teacher checkpoint is incomplete: {name} missing {missing[:5]}")


class AdaptedLinear(nn.Linear):
    """Preserve the original weight/bias keys when loading released checkpoints.

    LoRA adds BAx; adapter adds B SiLU(Ax), a parallel bottleneck adapter.
    Neither changes the affine flow-map parameterization.
    """
    def __init__(self, base, rank, alpha, nonlinear=False):
        # Avoid allocating a second full-sized base weight.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.nonlinear = nonlinear
        self.scale = 1.0 if nonlinear else alpha / rank
        self.adapter_down = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.adapter_up = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.adapter_down, a=math.sqrt(5))

    def forward(self, x):
        hidden = F.linear(x, self.adapter_down)
        if self.nonlinear:
            hidden = F.silu(hidden)
        return F.linear(x, self.weight, self.bias) + self.scale * F.linear(hidden, self.adapter_up)


def install_adaptation(model):
    cfg = model.flow_map
    if not cfg.enabled or getattr(model, '_flowmap_adaptation_installed', False):
        return
    if cfg.initialization == 'random':
        # Initialize generative/conditioning modules from scratch. Encoders stay frozen.
        seen = set()
        for name, child in model.named_children():
            if name in model.FROZEN_MODULES:
                continue
            for module in child.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                if hasattr(module, 'modulation') and isinstance(module.modulation, nn.Parameter):
                    nn.init.normal_(module.modulation, std=module.modulation.shape[-1] ** -.5)
                if hasattr(module, 'emb_pos') and isinstance(module.emb_pos, nn.Parameter):
                    nn.init.zeros_(module.emb_pos)
        for expert in (model.video_expert, model.action_expert):
            if expert.time_embedding_delta is not None:
                nn.init.zeros_(expert.time_embedding_delta[-1].weight)
                nn.init.zeros_(expert.time_embedding_delta[-1].bias)
    if cfg.mode in ('lora', 'adapter'):
        experts = [model.action_expert]
        if set(cfg.streams) & {'video', 'dino', 'pointmap'}:
            experts.append(model.video_expert)
        for expert in experts:
            for block in expert.blocks:
                for parent in list(block.modules()):
                    for name, child in list(parent.named_children()):
                        if type(child) is nn.Linear:
                            setattr(parent, name, AdaptedLinear(child, cfg.rank, cfg.lora_alpha, cfg.mode == 'adapter'))
    if cfg.uses_time_weighting:
        from .flowmap_self import TimeLossWeight
        model.flow_map_loss_weight = TimeLossWeight().to(model.device)
    model._flowmap_adaptation_installed = True


def configure_trainable(model):
    """Called on EVERY train-mode restoration, including after validation."""
    cfg = model.flow_map
    model.eval()
    model.requires_grad_(False)
    if cfg.mode == 'full':
        for name, child in model.named_children():
            if name not in model.FROZEN_MODULES:
                child.train()
                child.requires_grad_(True)
        for p in model.parameters(recurse=False):
            p.requires_grad_(True)
    else:
        model.mot.train()
        for name, p in model.named_parameters():
            if 'time_embedding_delta.' in name or '.adapter_down' in name or '.adapter_up' in name:
                p.requires_grad_(True)
        if cfg.mode == 'heads':
            model.action_expert.head.requires_grad_(True)
            if 'video' in cfg.streams:
                model.video_expert.head.requires_grad_(True)
            if 'dino' in cfg.streams:
                model.dino_proj_out.requires_grad_(True)
            if 'pointmap' in cfg.streams:
                model.pt_head.requires_grad_(True)
    if hasattr(model, 'flow_map_loss_weight'):
        model.flow_map_loss_weight.train().requires_grad_(True)
    # Do not retain an unused visual delta path in the optimizer for action-only runs.
    if not set(cfg.streams) & {'video', 'dino', 'pointmap'}:
        if model.video_expert.time_embedding_delta is not None:
            model.video_expert.time_embedding_delta.requires_grad_(False)
    for stream, head in (('video', model.video_expert.head),
                         ('dino', model.dino_proj_out), ('pointmap', model.pt_head)):
        if stream not in cfg.streams:
            head.requires_grad_(False)
    for name in model.FROZEN_MODULES:
        child = getattr(model, name, None)
        if isinstance(child, nn.Module):
            child.eval().requires_grad_(False)

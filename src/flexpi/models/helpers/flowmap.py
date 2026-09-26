"""Two-time flow maps in noise coordinates (1=noise, 0=data).

All generated streams follow one joint sigma trajectory. Time coordinates remain
float32 even when latent tensors are bfloat16. Teacher and spatial-JVP targets
are detached; temporal derivatives remain differentiable by default.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import torch

MapFn = Callable[[torch.Tensor], torch.Tensor]
STREAMS = ("video", "dino", "pointmap", "action")


@dataclass
class FlowMapConfig:
    enabled: bool = False
    mode: str = "full"  # full | lora | adapter | heads
    objective: str = "lmd"  # lmd | emd | pfmm | lsd | esd | psd_m | psd_u
    streams: tuple = ("action",)
    teacher_checkpoint: Optional[str] = None
    initialization: str = "pretrained"  # random resets denoisers, not frozen encoders
    rank: int = 16
    lora_alpha: float = 16.0
    self_diagonal_fraction: float = 0.75
    learned_time_weighting: bool = True  # self-distillation reference recipe
    distill_learned_time_weighting: bool = False  # explicit external-teacher ablation
    distill_ema: bool = False  # opt in for new runs; preserves existing pilots
    lmd_teacher_gradient: str = "detached"  # full retains teacher input gradients
    ema_decays: tuple = (0.999, 0.9999)  # evaluation; targets use current student
    pfmm_loss_space: str = "velocity"  # preserves pilots; new primary configs use endpoint
    diagonal_weight: float = 1.0
    distill_diagonal_weight: float = 0.0
    map_weight: float = 1.0
    teacher_steps: int = 4
    schedule_shift: float = 1.0
    num_inference_steps: int = 2
    strip_width: float = 1.0
    time_sampling: str = "uniform_triangle"  # conditional retained for ablations
    dt_method: str = "auto"  # auto/ad: exact JVP; FD only for float32 models
    detach_derivatives: bool = False  # optional lower-memory semigradient variant
    # Attention under forward AD: "explicit" materialises L×L in FP32; "tvm" uses the
    # fused Triton JVP kernels (helpers/jvp_attention, CC BY-NC-SA 4.0), ~3.8× faster
    # per call and ~8-10 GiB less per LMD/LSD step. Process-wide (student + teacher).
    jvp_attention: str = "explicit"
    fd_eps: float = 1e-4

    def __post_init__(self):
        self.streams = tuple(self.streams)
        self.ema_decays = tuple(float(d) for d in self.ema_decays)
        if not 0 < self.self_diagonal_fraction < 1:
            raise ValueError("self_diagonal_fraction must be between zero and one")
        if len(set(self.ema_decays)) != len(self.ema_decays) or any(not 0 < d < 1 for d in self.ema_decays):
            raise ValueError("ema_decays must be unique and between zero and one")
        if self.lmd_teacher_gradient not in ("detached", "full"):
            raise ValueError("lmd_teacher_gradient must be detached or full")
        if self.lmd_teacher_gradient == "full" and (self.objective != "lmd" or self.detach_derivatives):
            raise ValueError("Full teacher-input gradients require LMD with differentiable temporal derivatives")
        if self.pfmm_loss_space not in ("velocity", "endpoint"):
            raise ValueError("pfmm_loss_space must be velocity or endpoint")
        if not self.streams or len(set(self.streams)) != len(self.streams) or not set(self.streams) <= set(STREAMS):
            raise ValueError("streams must contain unique action/video/dino/pointmap names")
        if "action" not in self.streams:
            raise ValueError("FlexPi experiments require the action stream")
        if self.mode not in ("adapter", "lora", "full", "heads"):
            raise ValueError(f"Unknown flow_map.mode: {self.mode}")
        if self.objective not in ("lmd", "emd", "pfmm", "lsd", "esd", "psd_m", "psd_u"):
            raise ValueError(f"Unknown flow_map.objective: {self.objective}")
        if self.initialization not in ("pretrained", "random"):
            raise ValueError("initialization must be pretrained or random")
        if self.initialization == "random" and self.mode != "full":
            raise ValueError("Random initialization requires full training")
        if self.time_sampling not in ("uniform_triangle", "conditional"):
            raise ValueError(f"Unknown time_sampling: {self.time_sampling}")
        if self.dt_method not in ("auto", "ad", "central_fd", "forward_fd"):
            raise ValueError(f"Unknown dt_method: {self.dt_method}")
        if self.jvp_attention not in ("explicit", "tvm"):
            raise ValueError(f"Unknown jvp_attention: {self.jvp_attention}")
        for name in ("rank", "teacher_steps", "num_inference_steps"):
            value = getattr(self, name)
            if int(value) != value or value < (2 if name == "teacher_steps" else 1):
                raise ValueError(f"Invalid {name}: {value}")
        for name in ("diagonal_weight", "distill_diagonal_weight", "map_weight", "schedule_shift", "lora_alpha", "fd_eps", "strip_width"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.fd_eps < 0.5 or not 0 < self.strip_width <= 1:
            raise ValueError("Require 0 < fd_eps < .5 and 0 < strip_width <= 1")
        if self.schedule_shift == 0 or self.map_weight == 0 or self.lora_alpha == 0:
            raise ValueError("schedule_shift, map_weight and lora_alpha must be positive")
        if self.self_distillation and self.diagonal_weight == 0:
            raise ValueError("Self-distillation requires a positive diagonal FM weight")

    @property
    def self_distillation(self):
        return self.objective in ("lsd", "esd", "psd_m", "psd_u")

    @property
    def uses_time_weighting(self):
        return self.learned_time_weighting if self.self_distillation else self.distill_learned_time_weighting

    @property
    def uses_ema(self):
        return self.enabled and bool(self.ema_decays) and (self.self_distillation or self.distill_ema)

    @property
    def needs_teacher(self):
        return self.enabled and not self.self_distillation

    def inference_nodes(self, steps, device, shift=None):
        shift = self.schedule_shift if shift is None else float(shift)
        if steps < 1 or not math.isfinite(shift) or shift <= 0:
            raise ValueError("Flow-map steps and shift must be positive")
        u = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
        nodes = shift * u / (1 + (shift - 1) * u)
        if float((nodes[:-1] - nodes[1:]).max()) > self.strip_width + 1e-6:
            raise ValueError("Inference jump exceeds flow_map.strip_width; increase training coverage or steps")
        return nodes


def delta_timestep(timestep_input, timestep_target):
    return timestep_target - timestep_input


def _broadcast_like(per_sample, like):
    per_sample = per_sample.to(device=like.device, dtype=like.dtype)
    return per_sample if per_sample.ndim == 0 else per_sample.view(-1, *([1] * (like.ndim - 1)))


def affine_flow_map(x_s, v_st, sigma_s, sigma_t):
    return x_s + _broadcast_like(sigma_t - sigma_s, x_s) * v_st


def dX_dt_finite_difference(map_fn, sigma_t, eps=1e-4, scheme="central"):
    sigma_t = sigma_t.double() if sigma_t.dtype == torch.float64 else sigma_t.float()
    if not 0 < eps < .5:
        raise ValueError("eps must be in (0, .5)")
    x = map_fn(sigma_t)
    if x.dtype in (torch.float16, torch.bfloat16):
        raise ValueError("Finite differences require float32/64 model computation. Use ad for mixed precision; casting rounded outputs is insufficient.")
    lo, hi = sigma_t < eps, sigma_t > 1 - eps
    step = torch.full_like(sigma_t, eps)
    h = torch.where(hi, -step, step)
    y1 = map_fn(sigma_t + h)
    if scheme == "forward":
        return x, (y1 - x) / _broadcast_like(h, x)
    if scheme != "central":
        raise ValueError("scheme must be central or forward")
    h2 = torch.where(lo, 2 * step, torch.where(hi, -2 * step, -step))
    y2 = map_fn(sigma_t + h2)
    deriv = (y1 - y2) / (2 * eps)
    deriv = torch.where(_broadcast_like(lo, x).bool(), (-3*x + 4*y1 - y2)/(2*eps), deriv)
    deriv = torch.where(_broadcast_like(hi, x).bool(), (3*x - 4*y1 + y2)/(2*eps), deriv)
    return x, deriv


def tuple_jvp(fn, primals, tangents):
    """Forward AD of a tuple-valued function; attention.py supplies safe math."""
    import torch.autograd.forward_ad as fw
    with fw.dual_level():
        result = fn(*(fw.make_dual(x, dx) for x, dx in zip(primals, tangents)))
        pairs = [fw.unpack_dual(y) for y in result]
    return tuple(p.primal for p in pairs), tuple(
        torch.zeros_like(p.primal) if p.tangent is None else p.tangent for p in pairs)


def dX_dt_forward_ad(map_fn, sigma_t):
    y, dy = tuple_jvp(lambda t: (map_fn(t),), (sigma_t,), (torch.ones_like(sigma_t),))
    return y[0], dy[0]


def lmd_residual(dX_dt, b_teacher):
    return dX_dt - b_teacher


def sample_level_pair_strip(batch_size, strip, device, dtype, sigma_min=0., sigma_max=1.,
                            generator=None, sampling="uniform_triangle"):
    """Uniform area on sigma_min <= t <= s <= sigma_max, s-t <= strip.

    At full width this is the same distribution as sorting two independent
    uniforms (Boffi's triangle, with reversed time direction). Inverse-CDF
    sampling also supports narrow strips without rejection or clipping bias.
    The old uniform-source conditional proposal remains an explicit ablation.
    """
    if not 0 < strip <= 1 or not 0 <= sigma_min < sigma_max <= 1:
        raise ValueError("Invalid strip or sigma bounds")
    if sampling not in ("uniform_triangle", "conditional"):
        raise ValueError(f"Unknown time sampling: {sampling}")
    dtype = torch.float64 if dtype == torch.float64 else torch.float32
    u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator).clamp_min(1e-6)
    w = torch.rand(batch_size, device=device, dtype=dtype, generator=generator).clamp_min(1e-6)
    length = sigma_max - sigma_min
    width = min(strip, length)
    if sampling == "uniform_triangle":
        # Source density is proportional to the available target interval.
        # Its unnormalized CDF is y^2/2 for y <= width, then width*y-width^2/2.
        area = width * (length - width / 2)
        q = u * area
        offset = torch.where(q <= width**2 / 2, (2*q).sqrt(), q / width + width / 2)
    else:
        offset = length * u
    s = sigma_min + offset
    t = s - offset.clamp(max=width) * w
    return s, t


@contextmanager
def without_checkpointing(model):
    saved = []
    for module in model.modules():
        for name in ("use_gradient_checkpointing", "mot_checkpoint_mixed_attn"):
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, False)
    try:
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def map_residuals(predict, teacher, x, s, t, cfg, diagnostics=None):
    """Joint vector-field objectives, returning one residual tensor per stream.

    predict(x_tuple, s, t) returns average velocities; teacher(x_tuple,t)
    returns instantaneous velocities. JVPs include ALL jointly moving streams.
    Default stopgrad placement follows Boffi 2025 Eq. 94: teacher and spatial
    JVPs are detached, temporal JVPs are differentiable. The optional expanded
    semigradient detaches all derivatives. FM supervision is added by the caller.
    """
    h = t - s
    objective = cfg.objective
    if objective in ('lmd', 'lsd', 'emd', 'esd') and not cfg.detach_derivatives:
        lagrangian = objective in ('lmd', 'lsd')
        if cfg.dt_method in ('auto', 'ad'):
            time = t if lagrangian else s
            v, dv = tuple_jvp(
                (lambda z: predict(x, s, z)) if lagrangian else (lambda z: predict(x, z, t)),
                (time,), (torch.ones_like(time),))
        else:
            if not lagrangian:
                raise ValueError('Eulerian objectives require ad')
            v = predict(x, s, t)
            plus = predict(x, s, t + cfg.fd_eps)
            minus = predict(x, s, t - cfg.fd_eps) if cfg.dt_method == 'central_fd' else v
            dv = tuple((a-b)/(2*cfg.fd_eps if cfg.dt_method == 'central_fd' else cfg.fd_eps)
                       for a,b in zip(plus,minus))
        if objective == 'lmd' and cfg.lmd_teacher_gradient == 'full':
            # Teacher parameters are frozen by the caller, but its input is a
            # function of the student. Preserve that path for original LMD.
            mapped = tuple(affine_flow_map(a, b, s, t) for a, b in zip(x, v))
            target = teacher(mapped, t)
            return tuple(a + _broadcast_like(h, d)*d - b for a, d, b in zip(v, dv, target))
        with torch.no_grad():
            if lagrangian:
                mapped = tuple(affine_flow_map(a,b,s,t) for a,b in zip(x,v))
                target = teacher(mapped, t)
            else:
                b = teacher(x, s)
                _, spatial = tuple_jvp(lambda *state: predict(state, s, t), x, b)
                target = tuple(a + _broadcast_like(h,d)*d for a,d in zip(b,spatial))
        sign = 1 if lagrangian else -1
        return tuple(a + sign*_broadcast_like(h,d)*d - b.detach() for a,d,b in zip(v,dv,target))
    v = predict(x, s, t)
    mapped = tuple(affine_flow_map(a, b, s, t) for a, b in zip(x, v))
    with torch.no_grad():
        if objective in ("lmd", "lsd"):
            target = teacher(tuple(a.detach() for a in mapped), t)
            if cfg.dt_method in ("auto", "ad"):
                _, dv = tuple_jvp(lambda z: predict(x, s, z), (t,), (torch.ones_like(t),))
            else:
                dt = cfg.fd_eps
                plus = predict(x, s, t + dt)
                minus = predict(x, s, t - dt) if cfg.dt_method == "central_fd" else tuple(a.detach() for a in v)
                if any(a.dtype in (torch.float16, torch.bfloat16) for a in plus):
                    raise ValueError("FD objectives require float32 model computation")
                dv = tuple((a-b)/(2*dt if cfg.dt_method == "central_fd" else dt) for a,b in zip(plus, minus))
            target = tuple(b - _broadcast_like(h, d) * d for b, d in zip(target, dv))
        elif objective in ("emd", "esd"):
            if cfg.dt_method not in ("auto", "ad"):
                raise ValueError("Eulerian objectives require ad (joint spatial/time JVP)")
            b = teacher(x, s)
            _, dv = tuple_jvp(lambda z, *state: predict(state, z, t), (s, *x), (torch.ones_like(s), *b))
            target = tuple(a + _broadcast_like(h, d)*d for a, d in zip(b, dv))
        elif objective == "pfmm":
            state = tuple(a.detach() for a in x)
            target = tuple(torch.zeros_like(a) for a in x)
            for i in range(cfg.teacher_steps):
                lo = s + h * (i / cfg.teacher_steps)
                hi = s + h * ((i + 1) / cfg.teacher_steps)
                b = teacher(state, lo)
                # Average the Euler velocities directly: subtracting nearly
                # equal endpoints and dividing by a tiny h loses precision.
                target = tuple(a + vel / cfg.teacher_steps for a, vel in zip(target, b))
                state = tuple(affine_flow_map(a, vel, lo, hi) for a, vel in zip(state, b))
        else:
            fraction = torch.full_like(s, .5) if objective == "psd_m" else torch.rand_like(s)
            u = s + h * fraction
            first = predict(x, s, u)
            mid = tuple(affine_flow_map(a,b,s,u) for a,b in zip(x, first))
            second = predict(mid, u, t)
            target = tuple(_broadcast_like(fraction,a)*a + _broadcast_like(1-fraction,b)*b for a,b in zip(first,second))
    if diagnostics is not None:
        diagnostics['student_velocity'] = tuple(a.detach() for a in v)
        diagnostics['target_velocity'] = tuple(a.detach() for a in target)
    return tuple(a-b.detach() for a,b in zip(v, target))

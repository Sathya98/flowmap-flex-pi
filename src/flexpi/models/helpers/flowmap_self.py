"""Reference self-distillation sampling and learned two-time loss weighting."""
from dataclasses import fields, replace, is_dataclass
import math
import torch
from torch import nn


def update_diagonal_mask(batch_size, accumulation, world_size, rank, microstep,
                         step, seed, fraction=.75):
    """An exact split of the effective batch, stratified by microstep.

    ZeRO ranks run every microstep in lockstep, so a microstep costs as much as
    its slowest rank; with a per-slot shuffle ~1 - fraction**world_size of
    microsteps paid for an off-diagonal JVP (~68% at 75/25, 4 GPUs) although only
    25% of examples need one. Off-diagonal examples are therefore packed into
    whole microsteps: every rank and every example of a microstep takes the same
    branch, except one mixed microstep when the off-diagonal count is not a
    multiple of batch_size * world_size. Which microsteps (hence which data) take
    which branch is reshuffled every step, so the update's mixture is unchanged.

    All ranks reconstruct the same CPU permutation without a collective. For
    non-integral counts we floor, as in the reference batch split. At least one
    diagonal sample is retained; off-diagonal training requires total size > 1.
    """
    per_microstep = batch_size * world_size
    total = per_microstep * accumulation
    off = total - max(1, int(total * fraction))
    full, partial = divmod(off, per_microstep)
    generator = torch.Generator().manual_seed(seed + step)
    rows = torch.randperm(accumulation, generator=generator)
    mask = torch.ones(accumulation, per_microstep, dtype=torch.bool)
    mask[rows[:full]] = False
    if partial:
        mask[rows[full], torch.randperm(per_microstep, generator=generator)[:partial]] = False
    return mask.reshape(accumulation, world_size, batch_size)[microstep, rank]


def slice_batch(value, index, batch_size):
    """Select examples along the batch dim: ``index`` is an int or a 1-D index tensor."""
    if isinstance(value, torch.Tensor):
        if not value.ndim or value.shape[0] != batch_size:
            return value
        return value[index:index+1] if isinstance(index, int) else value[index.to(value.device)]
    if isinstance(value, dict):
        return {k: slice_batch(v, index, batch_size) for k, v in value.items()}
    if is_dataclass(value):
        return replace(value, **{f.name: slice_batch(getattr(value, f.name), index, batch_size)
                                 for f in fields(value)})
    return value


class TimeLossWeight(nn.Module):
    """EDM2 reference calc_weight: positional features + normalized linear map.

    Sigma runs backwards here, so embed reference times (1-sigma). Computation
    is FP32 even under autocast; the scalar weights the joint masked-mean loss.
    """
    def __init__(self, channels=128):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, channels))
        self.register_buffer('frequencies', torch.exp(
            -math.log(10000) * torch.arange(channels // 2).float() / (channels // 2)))

    def forward(self, s, t):
        with torch.autocast(device_type=s.device.type, enabled=False):
            def embedding(sigma):
                phase = (1 - sigma.float())[:, None] * self.frequencies.float()[None]
                return torch.cat((phase.cos(), phase.sin()), -1) * math.sqrt(2)
            features = (embedding(s) + embedding(t)) / math.sqrt(2)
            w = self.weight.float()
            w = w / (w.square().mean(-1, keepdim=True).sqrt() + 1e-4)
            return (features @ (w / math.sqrt(w.shape[-1])).T).squeeze(-1)

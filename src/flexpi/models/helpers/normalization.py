"""LayerNorm compatible with BF16 forward AD and backward through its tangent."""
import torch
from torch import nn


class ForwardADLayerNorm(nn.LayerNorm):
    def forward(self, x):
        if torch.autograd.forward_ad._current_level < 0:
            return super().forward(x)
        # PyTorch 2.7's native LayerNorm JVP can retain FP32 saved-statistics
        # precision in its tangent while returning a BF16 primal. A following
        # BF16 Linear then fails inside the tangent matmul, even though the
        # visible input.dtype is BF16. An ordinary .to(x.dtype) can be a no-op
        # on that dual tensor. Compute both components explicitly in FP32 and
        # cast the result together. FP64 is retained for derivative checks.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        value = x.to(dtype)
        dims = tuple(range(-len(self.normalized_shape), 0))
        centered = value - value.mean(dim=dims, keepdim=True)
        value = centered * torch.rsqrt(centered.square().mean(dim=dims, keepdim=True) + self.eps)
        if self.weight is not None:
            value = value * self.weight.to(dtype)
        if self.bias is not None:
            value = value + self.bias.to(dtype)
        return value.to(x.dtype)

"""Activation checkpointing for reverse differentiation through a forward JVP."""
from contextlib import nullcontext

import torch
from torch.autograd import forward_ad as fw
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils._pytree import tree_flatten, tree_unflatten


def _preserve_rng_state():
    """Checkpoint saves/restores the CUDA RNG around the recompute, which cannot run inside
    CUDA-graph capture. Captured regions must therefore be RNG-free (attention and the
    DiT blocks are); eager behaviour is unchanged."""
    return not (torch.cuda.is_available() and torch.cuda.is_initialized()
                and torch.cuda.is_current_stream_capturing())


def checkpoint(function, *args, use_reentrant=False, **kwargs):
    """Checkpoint ordinary tensors, explicitly carrying both halves of duals.

    PyTorch's checkpoint autograd Function has no JVP rule. Unpack dual inputs
    before entering it, and reconstruct the local JVP inside the recomputed
    region. Reverse autograd then differentiates BOTH outputs, preserving the
    mixed parameter/time derivatives required by LMD. No derivative is detached.

    All tensors with forward tangents must be explicit arguments, including
    tensors inside dict/list arguments; never capture a dual in a closure.
    Uses the forward-grad switch from the pinned PyTorch 2.7 runtime.
    """
    if use_reentrant:
        raise ValueError('Forward-AD checkpointing requires use_reentrant=False')
    if not torch.is_grad_enabled():
        return function(*args, **kwargs)
    if fw._current_level < 0:
        return torch_checkpoint(function, *args, use_reentrant=False,
                                preserve_rng_state=_preserve_rng_state(), **kwargs)

    leaves, input_spec = tree_flatten((args, kwargs))
    positions = [i for i, value in enumerate(leaves) if isinstance(value, torch.Tensor)]
    parts = []
    for i in positions:
        parts.extend(fw.unpack_dual(leaves[i]))
        leaves[i] = None
    output_spec = None

    def explicit_jvp(*values):
        nonlocal output_spec
        # The original forward runs inside the caller's dual level. Backward
        # recomputation happens after it has exited and needs a fresh level.
        level = fw.dual_level() if fw._current_level < 0 else nullcontext()
        with level, fw._set_fwd_grad_enabled(True):
            inputs = list(leaves)
            for j, i in enumerate(positions):
                primal, tangent = values[2*j:2*j+2]
                inputs[i] = fw.make_dual(primal, tangent) if tangent is not None else primal
            call_args, call_kwargs = tree_unflatten(inputs, input_spec)
            outputs, output_spec = tree_flatten(function(*call_args, **call_kwargs))
            return tuple(fw.unpack_dual(value) if isinstance(value, torch.Tensor)
                         else (value, None) for value in outputs)

    # Do not ask the checkpoint autograd Function itself to propagate duals.
    with fw._set_fwd_grad_enabled(False):
        outputs = torch_checkpoint(explicit_jvp, *parts, use_reentrant=False,
                                   preserve_rng_state=_preserve_rng_state())
    return tree_unflatten([
        fw.make_dual(primal, tangent) if tangent is not None else primal
        for primal, tangent in outputs
    ], output_spec)

"""CapturedStep on the tiny generic DiT: a captured LMD/LSD microstep (forward AD, dual
checkpoint, backward) replays bit-identically to eager and accumulates like eager."""
import copy

import pytest
import torch

from flowmap_core.flowmap import FlowMapObjectiveConfig, map_residuals, sample_level_pair_strip
from flowmap_core.graphs import CapturedStep
from test_core_generic_dit import TinyDiT, residual_fn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


@pytest.mark.parametrize("objective", ["lmd", "lsd"])
def test_replay_equals_eager_and_accumulates(objective):
    torch.manual_seed(0)
    model = TinyDiT(use_checkpoint=True).cuda()
    cfg = FlowMapObjectiveConfig(enabled=True, objective=objective,
                                 lmd_teacher_gradient="full" if objective == "lmd" else "detached")
    teacher_model = model if cfg.self_distillation else copy.deepcopy(model).requires_grad_(False)
    predict, teacher = residual_fn(model, teacher_model)
    x = torch.randn(3, 5, 4, device="cuda")            # static input, refilled in place

    def step():
        s, t = sample_level_pair_strip(3, 1.0, "cuda", torch.float32)   # CUDA RNG: fresh per replay
        residual = map_residuals(predict, teacher, (x,), s, t, cfg)[0]
        loss = residual.square().mean()
        return loss, dict(loss=loss.detach())

    def eager(seed):
        torch.cuda.manual_seed(seed)
        loss, _ = step()
        loss.backward()
        return loss.detach()

    graph = CapturedStep(step, model.parameters()).capture()
    for seed in (1, 2):
        x.copy_(torch.randn_like(x))
        graph.zero_grad()
        ref_loss = eager(seed)
        ref = [p.grad.clone() for p in model.parameters()]
        graph.zero_grad()
        torch.cuda.manual_seed(seed)
        loss, outputs = graph.replay()
        assert torch.equal(loss, ref_loss) and torch.equal(outputs["loss"], ref_loss)
        for p, r in zip(model.parameters(), ref):
            assert torch.equal(p.grad, r)
    # Two replays accumulate into .grad exactly as two eager microsteps do.
    graph.zero_grad()
    torch.cuda.manual_seed(3)
    graph.replay()
    graph.replay()
    twice = [p.grad.clone() for p in model.parameters()]
    graph.zero_grad()
    torch.cuda.manual_seed(3)
    for _ in range(2):
        loss, _ = step()
        loss.backward()
    for p, r in zip(model.parameters(), twice):
        assert torch.equal(p.grad, r)

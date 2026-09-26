"""Bit-level CPU fingerprint of the flow-map code, for refactors that must not change numbers.

Runs fixed-seed cases on the tiny CPU models from tests/ (every objective, flex regimes,
batched self-distillation, EMA, diagonal mask, forward-AD attention, row groups) and
records losses, metrics and sha256 of every gradient. Two runs of the same tree must
match exactly; so must runs before and after a pure refactor.

    python scripts/refactor_fingerprint.py OUT.json [--compare BASELINE.json]
"""
import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.autograd import forward_ad as fw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_flowmap import batch, tiny_model  # noqa: E402
from test_flowmap_flex import fixed_inputs, flex_model  # noqa: E402
from test_jvp_attention import FULL_JOINT, SIZES, block_mask  # noqa: E402

from flexpi.models.helpers import attention as attn  # noqa: E402
from flexpi.models.helpers import flowmap_training as ft  # noqa: E402
from flexpi.models.helpers import jvp_attention as ja  # noqa: E402
from flexpi.models.helpers.adaptation import clone_teacher  # noqa: E402
from flexpi.models.helpers.flex_joint import sample_flex_batch_flags  # noqa: E402
from flexpi.models.helpers.flowmap import FlowMapConfig  # noqa: E402
from flexpi.models.helpers.flowmap_self import update_diagonal_mask  # noqa: E402
from flexpi.utils.flowmap_ema import EvaluationEMA  # noqa: E402

STREAMS = ("action", "video", "dino", "pointmap")


def digest(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.tolist() if value.numel() <= 8 else digest(value)
    if isinstance(value, float):
        return repr(value)
    return value


def record(model, loss, metrics=None):
    loss.backward()
    grads = {n: digest(p.grad) for n, p in model.named_parameters() if p.grad is not None}
    teacher = getattr(model, "flow_map_teacher", None)
    return dict(loss=repr(float(loss)), metrics={k: plain(v) for k, v in sorted((metrics or {}).items())},
                grads=grads, teacher_grads=None if teacher is None else
                sum(p.grad is not None for p in teacher.parameters()))


def objective_cases():
    out = {}
    cases = [("lmd", {}), ("lmd_full", dict(lmd_teacher_gradient="full")), ("lmd_detach", dict(detach_derivatives=True)),
             ("emd", {}), ("pfmm", {}), ("pfmm_endpoint", dict(pfmm_loss_space="endpoint")),
             ("lsd", {}), ("esd", {}), ("psd_m", {}), ("psd_u", {})]
    for name, options in cases:
        objective = name.split("_")[0] if not name.startswith("psd") else name
        torch.manual_seed(11)
        model = tiny_model(STREAMS, objective, **options)
        if model.flow_map.needs_teacher:
            object.__setattr__(model, "flow_map_teacher", clone_teacher(model))
        data = batch(b=3)
        if model.flow_map.self_distillation:
            data["_flowmap_diagonal_mask"] = torch.tensor([True, False, True])
        torch.manual_seed(5)
        out[name] = record(model, *model.training_loss(data))
    return out


def flex_cases():
    out = {}
    for objective, cross, share in (("lmd", True, False), ("lmd", False, True), ("lsd", True, False),
                                    ("lsd", False, False), ("esd", False, True)):
        torch.manual_seed(3)
        options = dict(lmd_teacher_gradient="full") if objective == "lmd" else {}
        model = flex_model(objective, cross, share=share, **options)
        data = batch(b=4)
        if model.flow_map.self_distillation:
            data["_flowmap_diagonal_mask"] = torch.tensor([True, False, False, True])
        torch.manual_seed(9)
        out[f"{objective}_cross{int(cross)}_share{int(share)}"] = record(model, *model.training_loss(data))
    return out


def batched_self_cases():
    out = {}
    for objective in ("lsd", "esd"):
        for diagonal in ((False,) * 4, (True,) * 4, (True, False, False, True)):
            torch.manual_seed(0)
            model = flex_model(objective, cross_modal=False)
            data, noise, times = fixed_inputs(4)
            data["_flowmap_diagonal_mask"] = torch.tensor(diagonal)
            model._batch_flex = sample_flex_batch_flags(cfg=model.flex_joint, batch_size=4, device="cpu",
                                                        pointmap_off=False)
            torch.manual_seed(123)
            loss, metrics = ft.training_loss(model, data, times=times, noise=noise)
            model._batch_flex = None
            out[f"{objective}_{''.join('d' if d else 'o' for d in diagonal)}"] = record(model, loss, metrics)
    return out


def ema_cases():
    torch.manual_seed(0)
    chunk, EvaluationEMA.CHUNK = EvaluationEMA.CHUNK, 5
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2).to(torch.bfloat16))
    out = {}
    try:
        for background in (False, True):
            torch.manual_seed(1)
            for p in model.parameters():
                p.data.normal_()
            ema = EvaluationEMA(model, (.9, .99), background=background)
            for _ in range(3):
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(torch.randn_like(p))
                ema.update(model)
            state = ema.state_dict()
            out[f"background{int(background)}"] = dict(
                updates=state["updates"],
                shadow={f"{d}/{n}": digest(v) for d, values in ema.shadow.items() for n, v in values.items()})
    finally:
        EvaluationEMA.CHUNK = chunk
    return out


def mask_cases():
    out = {}
    for args in ((1, 48, 4, 0, 5, 3, 42), (2, 24, 4, 3, 17, 0, 42), (1, 10, 3, 2, 9, 1, 7), (192, 1, 1, 0, 0, 0, 42)):
        out[str(args)] = update_diagonal_mask(*args).tolist()
    return out


def attention_cases():
    out = {}
    torch.manual_seed(1)
    mask = block_mask(SIZES, FULL_JOINT)[None, None]
    q, k, v, tq, tk, tv = (torch.randn(1, 2, mask.shape[-1], 8, dtype=torch.float64) for _ in range(6))
    for name, m in (("masked", mask), ("none", None)):
        leaves = [x.clone().requires_grad_() for x in (q, k, v, tq, tk, tv)]
        with fw.dual_level():
            o, to = fw.unpack_dual(attn.scaled_dot_product_attention(
                *(fw.make_dual(p, t) for p, t in zip(leaves[:3], leaves[3:])), attn_mask=m))
            loss = o.square().sum() + to.square().sum()
        grads = torch.autograd.grad(loss, leaves)
        out[name] = dict(o=digest(o), to=digest(to), grads=[digest(g) for g in grads])
    groups = {}
    allowed = [row[:] for row in FULL_JOINT]
    allowed[6] = [1, 0, 1, 0, 1, 0, 1]
    for name, m in (("full_joint", block_mask(SIZES, FULL_JOINT)), ("base", block_mask(SIZES, allowed))):
        found, inverse = ja.RowGroups()(m)
        groups[name] = dict(rows=[r.tolist() for r, _ in found],
                            cols=[None if c is None else c.tolist() for _, c in found], inverse=inverse.tolist())
    out["row_groups"] = groups
    return out


def config_cases():
    return dict(default={k: plain(v) if not isinstance(v, tuple) else list(v)
                         for k, v in sorted(dataclasses.asdict(FlowMapConfig()).items())},
                properties={o: [getattr(FlowMapConfig(objective=o, enabled=True), p) for p in
                                ("self_distillation", "uses_time_weighting", "uses_ema", "needs_teacher")]
                            for o in ("lmd", "emd", "pfmm", "lsd", "esd", "psd_m", "psd_u")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--compare")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = dict(config=config_cases(), mask=mask_cases(), attention=attention_cases(), ema=ema_cases(),
                  objectives=objective_cases(), flex=flex_cases(), batched_self=batched_self_cases())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        diffs = []

        def walk(a, b, path):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in sorted(set(a) | set(b)):
                    walk(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}/{key}")
            elif a != b:
                diffs.append(f"{path}: {str(a)[:80]} != {str(b)[:80]}")
        walk(json.loads(json.dumps(result, sort_keys=True)), baseline, "")
        print("\n".join(diffs[:40]) if diffs else f"IDENTICAL to {args.compare}")
        sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()

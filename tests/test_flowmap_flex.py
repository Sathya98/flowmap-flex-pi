"""Flow-map training under flex-joint regime sampling (random presence and joint flags).

Covers the configs flowmap_libero_{lmd,lsd}_flex: per-sample regimes with and
without cross-modal prediction, LMD with full teacher-input gradients (the teacher
sees the student's per-sample flags), LSD/ESD with a mixed diagonal batch, and one
regime shared per microbatch; the batched self-distillation loss equals the
mean of batch-1 losses (the former per-example loop).
"""
import torch
from unittest.mock import patch

from test_flowmap import batch, tiny_model

from flexpi.models import flexpi as flexpi_module
from flexpi.models.helpers import flowmap_training as ft
from flexpi.models.helpers.adaptation import clone_teacher
from flexpi.models.helpers.flex_joint import sample_flex_batch_flags
from flexpi.models.helpers.flowmap_self import slice_batch

STREAMS = ("action", "video", "dino", "pointmap")


def flex_model(objective, cross_modal, share=False, **options):
    model = tiny_model(STREAMS, objective, **options)
    fj = model.flex_joint
    for name in ("video", "dino", "pointmap"):
        setattr(fj, "p_present_" + name, 0.5)
        setattr(fj, "cross_modal_predict_" + name, cross_modal)
    fj.p_jv = fj.p_jd = fj.p_jp = 0.5
    fj.share_within_microbatch = share
    if model.flow_map.needs_teacher:
        object.__setattr__(model, "flow_map_teacher", clone_teacher(model))
    return model


def run(model, data, flags_seen):
    original = flexpi_module.sample_flex_batch_flags

    def spy(*args, **kwargs):
        flags = original(*args, **kwargs)
        flags_seen.append(flags)
        return flags

    teacher_flags = []
    if model.flow_map_teacher is not None:
        predict = model.flow_map_teacher._predict_joint_noise_unified_impl

        def teacher_predict(*args, **kwargs):
            teacher_flags.append(model.flow_map_teacher._batch_flex)
            return predict(*args, **kwargs)
        model.flow_map_teacher._predict_joint_noise_unified_impl = teacher_predict
    with patch.object(flexpi_module, "sample_flex_batch_flags", spy):
        loss, metrics = model.training_loss(data)
    loss.backward()
    return loss, metrics, teacher_flags


def check_finite(model, loss):
    assert bool(torch.isfinite(loss))
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert model._batch_flex is None
    if model.flow_map_teacher is not None:
        assert all(p.grad is None for p in model.flow_map_teacher.parameters())


def test_lmd_full_grad_random_regimes_teacher_shares_flags():
    for cross_modal in (True, False):
        for seed in range(3):
            torch.manual_seed(seed)
            model = flex_model("lmd", cross_modal, lmd_teacher_gradient="full")
            flags = []
            loss, _, teacher_flags = run(model, batch(b=4), flags)
            check_finite(model, loss)
            assert len(flags) == 1 and flags[0].B == 4
            # Every teacher query saw the student's per-sample regime.
            assert teacher_flags and all(f is flags[0] for f in teacher_flags)


def test_lsd_and_esd_random_regimes_with_mixed_diagonal_batch():
    for objective in ("lsd", "esd"):
        for cross_modal in (True, False):
            torch.manual_seed(0)
            model = flex_model(objective, cross_modal)
            data = batch(b=4)
            data["_flowmap_diagonal_mask"] = torch.tensor([True, False, True, False])
            flags = []
            loss, metrics, _ = run(model, data, flags)
            check_finite(model, loss)
            assert metrics["self_diagonal_fraction"] == 0.5


def test_shared_regime_per_microbatch_trains():
    for objective in ("lmd", "lsd"):
        torch.manual_seed(1)
        model = flex_model(objective, True, share=True)
        flags = []
        loss, _, _ = run(model, batch(b=4), flags)
        check_finite(model, loss)
        for name in ("present_v", "present_d", "present_p", "j_v", "j_d", "j_p"):
            value = getattr(flags[0], name)
            assert bool((value == value[0]).all())


def fixed_inputs(b):
    data = batch(b=b)
    noise = dict(video=torch.randn_like(data['input_latents']), dino=torch.randn_like(data['dino_features']),
                 pointmap=torch.randn_like(data['pointmap_raw']), action=torch.randn_like(data['action']))
    s = torch.rand(b) * .5 + .5
    return data, noise, (s, s - torch.rand(b) * .4)


def loss_and_grads(model, fn):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(123)                     # diagonal FM times are drawn inside
    loss = fn()
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_batched_self_distillation_equals_per_example_loop():
    b = 4
    for objective in ("lsd", "esd"):
        for diagonal in ([False] * b, [True] * b, [True, False, False, True]):
            torch.manual_seed(0)
            model = flex_model(objective, cross_modal=False)
            assert model.flow_map.uses_time_weighting
            data, noise, times = fixed_inputs(b)
            data["_flowmap_diagonal_mask"] = torch.tensor(diagonal)
            model._batch_flex = sample_flex_batch_flags(
                cfg=model.flex_joint, batch_size=b, device="cpu", pointmap_off=False)
            flags = model._batch_flex
            assert not bool(flags.present_d.all() and flags.present_p.all())  # some stream absent

            def loop():
                # Diagonal examples draw their FM times in index order, as in the batched call.
                losses = []
                for i in range(b):
                    model._batch_flex = slice_batch(flags, i, b)
                    loss, _ = ft.training_loss(
                        model, slice_batch(data, i, b), times=tuple(v[i:i+1] for v in times),
                        noise=slice_batch(noise, i, b), prepared_inputs=slice_batch(data, i, b),
                        _self_term="diagonal" if diagonal[i] else "offdiagonal")
                    losses.append(loss)
                model._batch_flex = flags
                return torch.cat(losses).mean()

            ref, ref_grads = loss_and_grads(model, loop)
            out, grads = loss_and_grads(model, lambda: ft.training_loss(
                model, data, times=times, noise=noise)[0])
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
            assert grads.keys() == ref_grads.keys()
            for name in grads:
                torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-6, msg=name)

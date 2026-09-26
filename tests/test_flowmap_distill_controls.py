"""Distillation objective/gradient controls and matching evaluation machinery."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from test_flowmap import tiny_model, batch
from flexpi.models.helpers.adaptation import clone_teacher
from flexpi.models.helpers.flowmap import FlowMapConfig, map_residuals
from flexpi.models.helpers.flowmap_training import training_loss
from flexpi.models.helpers.flowmap_self import slice_batch


def test_lmd_full_gradient_differs_from_detached_at_same_residual():
    # X = x + (t-s)*p and b_T(X) = k*X, with frozen teacher slope k.
    x, s, t = (torch.tensor([[.6]]), torch.tensor([.9]), torch.tensor([.2]))
    p = nn.Parameter(torch.tensor(.7))
    k = nn.Parameter(torch.tensor(1.3), requires_grad=False)
    def predict(state, s, t): return (torch.ones_like(state[0])*p,)
    def teacher(state, t): return (k*state[0],)
    results = {}
    for variant in ('detached', 'full'):
        residual = map_residuals(predict, teacher, (x,), s, t,
            FlowMapConfig(objective='lmd', lmd_teacher_gradient=variant))[0]
        gradient = torch.autograd.grad(residual.sum(), p)[0]
        results[variant] = residual.detach(), gradient
    torch.testing.assert_close(results['full'][0], results['detached'][0])
    torch.testing.assert_close(results['detached'][1], torch.tensor(1.))
    torch.testing.assert_close(results['full'][1], (1-k*(t-s)).squeeze())
    assert k.grad is None
    with pytest.raises(ValueError, match='require LMD'):
        FlowMapConfig(objective='lsd', lmd_teacher_gradient='full')
    with pytest.raises(ValueError, match='require LMD'):
        FlowMapConfig(objective='lmd', lmd_teacher_gradient='full', detach_derivatives=True)


def test_full_lmd_backprop_joint_student_but_not_teacher():
    torch.set_num_threads(2)
    model = tiny_model(('action','video','dino','pointmap'), objective='lmd',
                       lmd_teacher_gradient='full')
    teacher = clone_teacher(model)
    object.__setattr__(model, 'flow_map_teacher', teacher)
    data = batch(b=1)
    from flexpi.models.helpers.flowmap_diagnostics import make_noise
    noise = make_noise(data, 2026)
    results = {}
    for variant in ('full', 'detached'):
        model.flow_map.lmd_teacher_gradient = variant
        model.zero_grad(set_to_none=True)
        loss, _ = training_loss(model, data, noise=noise,
                               times=(torch.tensor([.8]), torch.tensor([.2])))
        loss.backward()
        assert torch.isfinite(loss)
        assert model.action_expert.head.weight.grad.norm() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        results[variant] = loss.detach(), model.action_expert.head.weight.grad.clone()
    torch.testing.assert_close(results['full'][0], results['detached'][0])
    assert not torch.allclose(results['full'][1], results['detached'][1])


def test_external_weighting_before_reduction_and_checkpoint_roundtrip():
    model = tiny_model(objective='lmd', distill_learned_time_weighting=True)
    object.__setattr__(model, 'flow_map_teacher', clone_teacher(model))
    data = batch(b=2)
    times = (torch.tensor([.9,.5]), torch.tensor([.2,.1]))
    noise = {'action': torch.randn_like(data['action'])}
    actual, metrics = training_loss(model, data, times=times, noise=noise)
    assert 'loss_unweighted' in metrics and 'self_diagonal_fraction' not in metrics
    model.flow_map.distill_learned_time_weighting = False
    expected = []
    logvars = model.flow_map_loss_weight(*times)
    for i in range(2):
        raw, _ = training_loss(model, slice_batch(data,i,2),
            times=tuple(v[i:i+1] for v in times), noise=slice_batch(noise,i,2))
        expected.append(torch.exp(-logvars[i])*raw+logvars[i])
    torch.testing.assert_close(actual, torch.stack(expected).mean())
    actual.backward()
    assert model.flow_map_loss_weight.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.flow_map_teacher.parameters())
    model.flow_map.distill_learned_time_weighting = True
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'model.pt'
        model.save_checkpoint(path)
        restored = tiny_model(objective='lmd', distill_learned_time_weighting=True)
        restored.load_checkpoint(path)
        torch.testing.assert_close(restored.flow_map_loss_weight.weight, model.flow_map_loss_weight.weight)
        payload = torch.load(path, weights_only=False)
        del payload['flow_map_loss_weight']
        torch.save(payload,path)
        with pytest.raises(ValueError, match='missing learned time weights'):
            restored.load_checkpoint(path)


def test_external_ema_initialization_and_legacy_defaults():
    from flexpi.trainer import Wan22Trainer
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = nn.Linear(2,1)
    trainer.model.flow_map = FlowMapConfig(enabled=True, objective='lmd')
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m:m, is_main_process=True,
                                         state=SimpleNamespace(deepspeed_plugin=None))
    trainer._initialize_flowmap_ema()
    assert trainer.flowmap_ema is None
    assert not trainer.model.flow_map.uses_time_weighting
    trainer.model.flow_map.distill_ema = True
    trainer._initialize_flowmap_ema()
    assert trainer.flowmap_ema.decays == (.999,.9999)
    trainer.accelerator.is_main_process = False
    trainer._initialize_flowmap_ema()
    assert trainer.flowmap_ema is None
    trainer.accelerator.is_main_process = True
    trainer.accelerator.state.deepspeed_plugin = SimpleNamespace(
        deepspeed_config={'zero_optimization': {'stage':3}})
    with pytest.raises(ValueError, match='replicated parameters'):
        trainer._initialize_flowmap_ema()


def test_control_configs_change_only_intended_recipe_fields():
    root = Path(__file__).resolve().parents[1]
    names = ['flowmap_libero_pfmm_accum_smoke','flowmap_libero_lmd_pilot',
             'flowmap_libero_pfmm_endpoint_pilot','flowmap_libero_pfmm_velocity_control',
             'flowmap_libero_lmd_fullgrad_pilot','flowmap_libero_lmd_detached_control']
    with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
        configs = [compose(config_name=name) for name in names]
    for config in configs:
        FlowMapConfig(**OmegaConf.to_container(config.model.flow_map, resolve=True))
        assert config.batch_size * config.gradient_accumulation_steps * 4 == 192
        assert config.learning_rate == 1e-5 and config.max_steps == 50
    assert all(not c.model.flow_map.distill_ema for c in configs[:2])
    assert configs[0].model.flow_map.pfmm_loss_space == 'velocity'
    assert configs[1].model.flow_map.lmd_teacher_gradient == 'detached'
    for a,b,field in [(configs[2],configs[3],'pfmm_loss_space'),
                      (configs[4],configs[5],'lmd_teacher_gradient')]:
        left,right = [OmegaConf.to_container(c,resolve=True) for c in (a,b)]
        left.pop('output_dir'); right.pop('output_dir')
        assert left['model']['flow_map'].pop(field) != right['model']['flow_map'].pop(field)
        assert left == right

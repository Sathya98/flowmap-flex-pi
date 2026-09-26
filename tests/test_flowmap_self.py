"""Reference-mixture, weighting, gradients and EMA persistence regressions."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from test_flowmap import tiny_model, batch
from flexpi.models.helpers.flowmap import FlowMapConfig, map_residuals
from flexpi.models.helpers.flowmap_self import update_diagonal_mask, TimeLossWeight
from flexpi.models.helpers.flowmap_training import training_loss
from flexpi.utils.flowmap_ema import EvaluationEMA


def test_effective_batch_split_and_reproducible_resume():
    for step in (0, 1, 47):
        masks = torch.cat([update_diagonal_mask(1, 48, 4, r, m, step, 42)
                           for m in range(48) for r in range(4)])
        assert masks.sum() == 144
        assert torch.equal(masks, torch.cat([update_diagonal_mask(1, 48, 4, r, m, step, 42)
                                            for m in range(48) for r in range(4)]))
    assert not torch.equal(update_diagonal_mask(192, 1, 1, 0, 0, 0, 42),
                           update_diagonal_mask(192, 1, 1, 0, 0, 1, 42))


def test_diagonal_mask_is_stratified_by_microstep():
    # 4 ranks x microbatch 1 x 48 microsteps: 48 off-diagonal = 12 whole microsteps.
    for batch, accum in ((1, 48), (2, 24)):
        grid = torch.stack([torch.stack([update_diagonal_mask(batch, accum, 4, r, m, 3, 42)
                                         for r in range(4)]) for m in range(accum)])
        per_micro = grid.reshape(accum, -1)
        assert int(grid.sum()) == 144
        assert bool((per_micro.all(1) | (~per_micro).all(1)).all())   # one branch per microstep
        assert int((~per_micro).all(1).sum()) == 48 // (4 * batch)
    # Not divisible: exactly one mixed microstep carries the remainder.
    grid = torch.stack([torch.stack([update_diagonal_mask(1, 10, 3, r, m, 0, 7)
                                     for r in range(3)]) for m in range(10)]).reshape(10, 3)
    assert int((~grid).sum()) == 30 - int(30 * .75)
    assert int((grid.any(1) & (~grid).any(1)).sum()) == 1
    # Which microsteps are off-diagonal changes with the step.
    a = [bool(update_diagonal_mask(1, 48, 4, 0, m, 0, 42)) for m in range(48)]
    b = [bool(update_diagonal_mask(1, 48, 4, 0, m, 1, 42)) for m in range(48)]
    assert a != b


def test_diagonal_skips_map_and_uses_independent_uniform_time():
    torch.set_num_threads(2)
    model = tiny_model()
    data = batch(b=1)
    data['_flowmap_diagonal_mask'] = torch.ones(1, dtype=torch.bool)
    observed = {}
    with patch('flexpi.models.helpers.flowmap_training.map_residuals',
               side_effect=AssertionError('Diagonal example must not evaluate map residual')), \
         patch('flexpi.models.helpers.flowmap_training.torch.rand', return_value=torch.tensor([.23])):
        loss, metrics = training_loss(model, data, times=(torch.tensor([.9]), torch.tensor([.1])),
                                      diagnostics=observed)
    torch.testing.assert_close(observed['s'], torch.tensor([.23]))
    torch.testing.assert_close(observed['s'], observed['t'])
    assert metrics['self_diagonal_fraction'] == 1
    assert metrics['loss_flowmap_action'] == 0
    loss.backward()
    assert model.flow_map_loss_weight.weight.grad is not None


def test_weight_before_batch_reduction_and_checkpoint():
    model = tiny_model()
    data = batch(b=2)
    data['_flowmap_diagonal_mask'] = torch.zeros(2, dtype=torch.bool)
    times = (torch.tensor([.9, .4]), torch.tensor([.1, .2]))
    noise = {'action': torch.randn_like(data['action'])}
    loss, metrics = training_loss(model, data, times=times, noise=noise)
    singles = []
    from flexpi.models.helpers.flowmap_self import slice_batch
    for i in range(2):
        singles.append(training_loss(model, slice_batch(data, i, 2),
                        times=tuple(t[i:i+1] for t in times),
                        noise=slice_batch(noise, i, 2))[0])
    torch.testing.assert_close(loss, torch.stack(singles).mean())
    assert 'loss_unweighted' in metrics
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'model.pt'
        model.save_checkpoint(path)
        restored = tiny_model()
        restored.load_checkpoint(path)
        torch.testing.assert_close(model.flow_map_loss_weight.weight,
                                   restored.flow_map_loss_weight.weight)


def test_weight_matches_reference_algebra():
    head = TimeLossWeight()
    s, t = torch.tensor([.8, .3]), torch.tensor([.2, .1])
    frequency = torch.exp(-torch.log(torch.tensor(10000.)) * torch.arange(64) / 64)
    def embed(time):
        phase = (1-time)[:, None] * frequency
        return torch.cat((phase.cos(), phase.sin()), -1) * 2**.5
    w = head.weight / (head.weight.norm(dim=1, keepdim=True)/128**.5 + 1e-4) / 128**.5
    expected = ((embed(s)+embed(t))/2**.5 @ w.T).squeeze(-1)
    torch.testing.assert_close(head(s, t), expected)


def test_pfmm_endpoint_has_interval_squared_scaling():
    model = tiny_model(objective='pfmm')
    from flexpi.models.helpers.adaptation import clone_teacher
    object.__setattr__(model, 'flow_map_teacher', clone_teacher(model))
    data = batch(b=1)
    noise = {'action': torch.randn_like(data['action'])}
    for h in (.1, .8):
        times = (torch.tensor([.9]), torch.tensor([.9-h]))
        model.flow_map.pfmm_loss_space = 'velocity'
        velocity = training_loss(model, data, times=times, noise=noise)[0]
        gv = torch.autograd.grad(velocity, model.action_expert.head.weight, retain_graph=False)[0]
        model.flow_map.pfmm_loss_space = 'endpoint'
        endpoint = training_loss(model, data, times=times, noise=noise)[0]
        ge = torch.autograd.grad(endpoint, model.action_expert.head.weight)[0]
        torch.testing.assert_close(endpoint, velocity*h*h)
        torch.testing.assert_close(ge, gv*h*h, atol=1e-6, rtol=1e-4)


def test_self_targets_detached_and_temporal_derivatives_differentiable():
    # Analytic nonconstant map: a*x+b*s+c*t. Source-time derivative for ESD
    # and target-time derivative for LSD remain differentiable, targets do not.
    x, s, t = torch.tensor([[.6]]), torch.tensor([.9]), torch.tensor([.2])
    for objective in ('lsd', 'esd', 'psd_m', 'psd_u'):
        a, b, c, teacher = [nn.Parameter(torch.tensor(v)) for v in (.7, .4, .2, 1.3)]
        def predict(state, source, target):
            return (a*state[0]+b*source[:,None]+c*target[:,None],)
        def velocity(state, time):
            return (teacher*state[0],)
        residual = map_residuals(predict, velocity, (x,), s, t,
                                FlowMapConfig(objective=objective))[0]
        grads = torch.autograd.grad(residual.sum(), (a,b,c,teacher), allow_unused=True)
        h = float(t-s)
        expected = (float(x), float(s) - (h if objective == 'esd' else 0),
                    float(t) + (h if objective == 'lsd' else 0))
        for actual, value in zip(grads[:3], expected):
            torch.testing.assert_close(actual, torch.tensor(value))
        assert grads[3] is None


def test_ema_resume_export_and_restore_on_failure():
    model = nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    ema = EvaluationEMA(model, (.999, .9999))
    initial = model.weight.detach().clone()
    with torch.no_grad(): model.weight.add_(2)
    ema.update(model)
    assert ema.updates == 1
    assert 'bias' not in ema.shadow[.999]
    torch.testing.assert_close(ema.shadow[.999]['weight'], initial+.002)
    raw = model.weight.detach().clone()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'ema.pt'
        torch.save(ema.state_dict(), path)
        resumed = EvaluationEMA(model, (.999, .9999))
        resumed.load_state_dict(torch.load(path, weights_only=True, mmap=True))
        ema.update(model); resumed.update(model)
        torch.testing.assert_close(ema.shadow[.9999]['weight'], resumed.shadow[.9999]['weight'])
        try:
            with resumed.apply(model, .9999):
                torch.testing.assert_close(model.weight, resumed.shadow[.9999]['weight'])
                raise RuntimeError('Simulated export failure')
        except RuntimeError: pass
        torch.testing.assert_close(model.weight, raw)


def test_trainer_accumulation_ema_update_and_export():
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace
    from flexpi.trainer import Wan22Trainer
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.))
            self.flow_map = FlowMapConfig(enabled=True, objective='lsd')
            self.masks = []
        def training_loss(self, sample):
            self.masks.append(sample['_flowmap_diagonal_mask'].clone())
            return self.weight.square(), {}
        def save_checkpoint(self, path, **kwargs):
            torch.save(self.state_dict(), path)
    class Accelerator:
        num_processes = 1
        process_index = 0
        is_main_process = True
        count = 0
        @contextmanager
        def accumulate(self, model):
            self.count += 1
            self.sync_gradients = self.count % 4 == 0
            self.optimizer_step_was_skipped = self.count == 4
            yield
        def unwrap_model(self, model): return model
        def autocast(self): return nullcontext()
        def backward(self, loss): (loss / 4).backward()
        def gather(self, value): return value
        def clip_grad_norm_(self, parameters, limit):
            return torch.nn.utils.clip_grad_norm_(parameters, limit)
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = Model()
    trainer.accelerator = Accelerator()
    trainer.flowmap_ema = EvaluationEMA(trainer.model, (.999, .9999))
    trainer._flowmap_microstep = 0
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
    trainer._set_dit_only_train_mode = lambda: None
    trainer._estimate_eta = lambda: ('00:00:00', 1.)
    trainer.save_checkpoint = lambda: dict(weights_path=None, state_path='mock-state')
    trainer.global_step = trainer.epoch = trainer.batch_in_epoch = 0
    trainer.max_steps = 2
    trainer.eval_every = trainer.save_every = trainer.log_every = 0
    trainer.batch_size = 1
    trainer.gradient_accumulation_steps = 4
    trainer.seed = 42
    trainer.max_grad_norm = 1.
    trainer.train_sampler = None
    # Accumulation crosses this simulated epoch boundary without resetting.
    trainer.train_loader = [{'action': torch.zeros(1, 1, 1)} for _ in range(3)]
    trainer._wandb_log = lambda _: None
    with tempfile.TemporaryDirectory() as folder:
        trainer.output_dir = trainer.weights_dir = folder
        trainer.train()
        assert trainer.flowmap_ema.updates == 1  # successful updates only
        assert trainer.scheduler.last_epoch == 1
        masks = torch.cat(trainer.model.masks).reshape(2, 4)
        assert torch.equal(masks.sum(1), torch.tensor([3, 3]))
        with torch.no_grad(): trainer.model.weight.fill_(5.)
        trainer._save_weights_checkpoint('step_000002')
        assert trainer.model.weight == 5  # export must restore raw training weights
        raw = torch.load(Path(folder)/'step_000002.pt', weights_only=True)
        averaged = torch.load(Path(folder)/'ema_0.9999/step_000002.pt', weights_only=True)
        assert raw['weight'] == 5 and averaged['weight'] == 1
        trainer._save_trainer_state(folder)
        saved = torch.load(Path(folder)/'flowmap_ema.pt', weights_only=True)
        assert saved['updates'] == 1


def test_background_ema_matches_synchronous_updates():
    torch.manual_seed(0)
    EvaluationEMA.CHUNK, chunk = 5, EvaluationEMA.CHUNK     # several chunks per dtype group
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2).to(torch.bfloat16))
    model[1].bias.requires_grad_(False)
    sync = EvaluationEMA(model, (.9, .99), background=False)
    background = EvaluationEMA(model, (.9, .99), background=True)
    for step in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        sync.update(model)
        background.update(model)
        with torch.no_grad():   # the snapshot was taken at update(): later edits must not leak in
            for p in model.parameters():
                p.mul_(-3)
    assert background.updates == sync.updates == 3 and background.fold_seconds is not None
    assert len(background._groups) == 2       # fp32 and bf16 parameters, one flat range each
    for decay in (.9, .99):
        assert background.shadow[decay].keys() == sync.shadow[decay].keys()
        for name, value in sync.shadow[decay].items():
            assert torch.equal(background.shadow[decay][name], value), name
    with torch.no_grad():
        model[0].weight.add_(1)
    background.update(model)
    state = background.state_dict()      # waits for the pending fold
    assert state['updates'] == 4 and background._pending is None
    EvaluationEMA.CHUNK = chunk


def test_scheduled_and_grid_time_sampling_in_training_loss():
    # Grid arm: every off-diagonal source is a 1- or 2-step grid node.
    torch.manual_seed(0)
    model = tiny_model(objective='lsd', time_sampling='inference_grid', grid_steps=(1, 2))
    data = batch(b=4)
    data['_flowmap_diagonal_mask'] = torch.zeros(4, dtype=torch.bool)
    seen = {}
    loss, metrics = training_loss(model, data, diagnostics=seen)
    loss.backward()
    assert set(seen['s'].tolist()) <= {1.0, 0.5} and 'time_max_jump' not in metrics
    # Curriculum arm: the trainer's update sets the maximum jump; missing update is an error.
    model = tiny_model(objective='lmd', strip_width_start=0.25, strip_anneal_updates=1000,
                       uniform_jump_from_update=1000)
    from flexpi.models.helpers.adaptation import clone_teacher
    object.__setattr__(model, 'flow_map_teacher', clone_teacher(model))
    import pytest
    with pytest.raises(ValueError, match='optimizer update'):
        training_loss(model, batch(b=2))
    model._flowmap_update = 0
    seen = {}
    loss, metrics = training_loss(model, batch(b=8), diagnostics=seen)
    assert float((seen['s'] - seen['t']).max()) <= 0.25 + 1e-6 and metrics['time_max_jump'] == 0.25
    model._flowmap_update = 1500
    assert training_loss(model, batch(b=2))[1]['time_max_jump'] == 1.0


def test_trainer_passes_update_to_wrapped_model():
    """Under DeepSpeed the trainer holds an engine that forwards training_loss to the
    FlexPi module but keeps attribute writes to itself: the update must reach the module."""
    from contextlib import contextmanager, nullcontext
    from flexpi.trainer import Wan22Trainer

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.))
            self.flow_map = FlowMapConfig(enabled=True, objective='lmd', strip_width_start=.25,
                                          strip_anneal_updates=10)
            self.seen = []
        def training_loss(self, sample):
            self.seen.append(self._flowmap_update)
            return self.weight.square(), {}

    class Engine(nn.Module):                      # DeepSpeedEngine-like: forwards reads only
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    class Accelerator:
        num_processes, process_index, is_main_process, count = 1, 0, True, 0
        @contextmanager
        def accumulate(self, model):
            self.count += 1
            self.sync_gradients = self.count % 4 == 0
            self.optimizer_step_was_skipped = False
            yield
        def unwrap_model(self, model): return model.module
        def autocast(self): return nullcontext()
        def backward(self, loss): (loss / 4).backward()
        def gather(self, value): return value
        def clip_grad_norm_(self, parameters, limit):
            return torch.nn.utils.clip_grad_norm_(parameters, limit)

    inner = Inner()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = Engine(inner)
    trainer.accelerator = Accelerator()
    trainer.flowmap_ema = None
    trainer._flowmap_microstep = 0
    trainer.optimizer = torch.optim.SGD(inner.parameters(), lr=0.)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
    trainer._set_dit_only_train_mode = lambda: None
    trainer._estimate_eta = lambda: ('00:00:00', 1.)
    trainer.save_checkpoint = lambda: dict(weights_path=None, state_path='mock-state')
    trainer.global_step = trainer.epoch = trainer.batch_in_epoch = 0
    trainer.max_steps = 2
    trainer.eval_every = trainer.save_every = trainer.log_every = 0
    trainer.batch_size, trainer.gradient_accumulation_steps, trainer.seed = 1, 4, 42
    trainer.max_grad_norm = 1.
    trainer.train_sampler = None
    trainer.train_loader = [{'action': torch.zeros(1, 1, 1)} for _ in range(8)]
    trainer._wandb_log = lambda _: None
    with tempfile.TemporaryDirectory() as folder:
        trainer.output_dir = trainer.weights_dir = folder
        trainer.train()
    assert inner.seen == [0] * 4 + [1] * 4

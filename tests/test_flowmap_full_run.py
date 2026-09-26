"""Long-run continuation, stable logging identities and reproducible previews."""
from contextlib import contextmanager, nullcontext
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from hydra import compose, initialize_config_dir
from flexpi.trainer import Wan22Trainer
from flexpi.utils.evaluation_rng import evaluation_rng
from flexpi.utils.tracking_config import tracking_config


def test_tracking_configuration_excludes_local_paths_and_unlisted_fields():
    cfg=OmegaConf.create(dict(max_steps=2000,output_dir='/private/run',
        pretrained_ckpt='/private/weights',wandb={'secret':'never-upload'},
        model={'flow_map':{'teacher_checkpoint':'/private/teacher','objective':'lmd'}}))
    assert tracking_config(cfg)=={'max_steps':2000,'model.flow_map.objective':'lmd'}


def test_flowmap_logging_cannot_fall_back_to_shared_login(tmp_path,monkeypatch):
    monkeypatch.delenv('FLOWMAP_WANDB_API_KEY_FILE',raising=False)
    trainer=Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator=SimpleNamespace(is_main_process=True)
    trainer.wandb_enabled=True;trainer.output_dir=str(tmp_path)
    trainer.cfg=OmegaConf.create({'model':{'flow_map':{'enabled':True}},
        'wandb':{'mode':'online','workspace':'explicit-workspace','isolated':False}})
    with pytest.raises(ValueError,match='shared login fallback is disabled'):
        trainer._init_wandb()


def test_preview_rng_restores_training_and_matches_across_nfes():
    random.seed(8); np.random.seed(8); torch.manual_seed(8)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    outputs=[]
    for _ in range(3):
        with evaluation_rng(2026,'cpu'):
            outputs.append((random.random(),np.random.rand(),torch.rand(4)))
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1],states[1][1])
    assert torch.equal(torch.get_rng_state(),states[2])
    for a,b,c in outputs:
        assert a==outputs[0][0] and b==outputs[0][1]
        torch.testing.assert_close(c,outputs[0][2])


def test_wandb_resumes_identity_and_uses_optimizer_axis(tmp_path,monkeypatch):
    import sys
    run=MagicMock();run.url='https://wandb.ai/test/flowmap-flex-pi/runs/abc'
    sdk=SimpleNamespace(init=MagicMock(return_value=run),util=SimpleNamespace(generate_id=lambda:'abc'),
        Settings=lambda **kwargs:kwargs)
    monkeypatch.setitem(sys.modules,'wandb',sdk)
    trainer=Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator=SimpleNamespace(is_main_process=True)
    trainer.wandb_enabled=True;trainer.output_dir=str(tmp_path);trainer.global_step=7
    trainer.cfg=OmegaConf.create({'max_steps':2000,'wandb':dict(enabled=True,id=None,
        workspace='test',project='flowmap-flex-pi',name='lmd',group=None,mode='online')})
    trainer._init_wandb();trainer._wandb_log({'train/loss':.5})
    run.log.assert_called_with({'train/loss':.5,'optimizer_step':7})
    trainer._init_wandb()
    assert sdk.init.call_args.kwargs['id']=='abc'
    assert sdk.init.call_args.kwargs['resume']=='allow'
    assert sdk.init.call_args.kwargs['config']['max_steps']==2000
    assert json.loads((tmp_path/'wandb_run.json').read_text())['id']=='abc'
    trainer.cfg.wandb.workspace='different'
    with pytest.raises(ValueError,match='identity changed'):trainer._init_wandb()


def test_segment_pause_preserves_full_schedule(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def training_loss(self,sample):return self.weight.square(),{}
    class Accelerator:
        num_processes=1;process_index=0;is_main_process=True
        optimizer_step_was_skipped=False;count=0;device=torch.device('cpu')
        @contextmanager
        def accumulate(self,model):
            self.count+=1;self.sync_gradients=self.count%2==0
            yield
        def unwrap_model(self,model):return model
        def autocast(self):return nullcontext()
        def backward(self,loss):(loss/2).backward()
        def gather(self,value):return value
        def clip_grad_norm_(self,params,limit):return torch.nn.utils.clip_grad_norm_(params,limit)
    trainer=Wan22Trainer.__new__(Wan22Trainer)
    trainer.model=Model();trainer.accelerator=Accelerator()
    trainer.learning_rate=1e-5
    trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=1e-5)
    trainer.scheduler=trainer._build_scheduler(total_train_steps=2000,warmup_steps=100)
    trainer._set_dit_only_train_mode=lambda:None
    trainer._estimate_eta=lambda:('00:00:00',1.)
    saves=[]
    def save():
        saves.append(trainer.global_step)
        return dict(weights_path='raw.pt',state_path='state')
    trainer.save_checkpoint=save;trainer._wandb_log=lambda _:None
    trainer.global_step=trainer.epoch=trainer.batch_in_epoch=0
    trainer.max_steps=2000;trainer.stop_after_steps=2;trainer.max_runtime_seconds=0
    trainer.eval_every=trainer.save_every=0;trainer.log_every=1
    trainer.batch_size=1;trainer.gradient_accumulation_steps=2;trainer.max_grad_norm=1.
    trainer.train_sampler=None;trainer.output_dir=str(tmp_path)
    trainer.train_loader=[{'action':torch.zeros(1,1,1)}]*4
    trainer.train()
    assert saves==[2] and trainer.global_step==2 and trainer.max_steps==2000
    assert trainer.scheduler.last_epoch==2
    assert trainer.optimizer.param_groups[0]['lr']==pytest.approx(2.98e-7)
    status=json.loads((tmp_path/'segment_status.json').read_text())
    assert not status['complete'] and status['max_steps']==2000


def test_checkpoint_retention_keeps_milestones_and_recent_emas(tmp_path):
    trainer=Wan22Trainer.__new__(Wan22Trainer)
    trainer.weights_dir=str(tmp_path);trainer.cfg=OmegaConf.create({'keep_weight_steps':[2]})
    directories=[tmp_path,tmp_path/'ema_0.999',tmp_path/'ema_0.9999']
    for d in directories:
        d.mkdir(exist_ok=True)
        for step in (2,50,100,150,200):(d/f'step_{step:06d}.pt').touch()
    trainer._prune_old_weights(2)
    for d in directories:
        assert [p.name for p in sorted(d.glob('step_*.pt'))]==[
            'step_000002.pt','step_000150.pt','step_000200.pt']


def test_full_config_has_budget_and_fixed_suite_previews():
    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='flowmap_libero_lmd_full')
    assert cfg.max_steps==2000 and cfg.stop_after_steps is None
    assert cfg.batch_size*cfg.gradient_accumulation_steps*4==192
    assert cfg.model.flow_map.lmd_teacher_gradient=='full' and cfg.model.flow_map.distill_ema
    assert cfg.eval_namespace=='train_preview' and list(cfg.eval_nfes)==[1,2,4]
    assert list(cfg.eval_fixed_indices)==[0,53128,120437,173332]
    assert cfg.save_every==250
    assert cfg.wandb.enabled and cfg.wandb.mode=='offline' and cfg.wandb.isolated
    assert cfg.wandb.workspace is None


@pytest.mark.parametrize('self_distillation', [False, True])
def test_flowmap_epoch_budget_counts_complete_accumulated_batches(self_distillation):
    trainer=Wan22Trainer.__new__(Wan22Trainer)
    trainer.max_steps=None;trainer.batch_size=1;trainer.gradient_accumulation_steps=4
    trainer.num_epochs=2;trainer.train_dataset=list(range(5))
    trainer.accelerator=SimpleNamespace(num_processes=1)
    trainer.model=SimpleNamespace(flow_map=SimpleNamespace(enabled=True,
        self_distillation=self_distillation))
    # Ten microbatches over two epochs need three updates, not two per epoch.
    assert trainer._estimate_total_train_steps()==3


def test_continuation_gate_requires_checkpoint_progress_and_previews(tmp_path):
    spec=importlib.util.spec_from_file_location('run_state',Path(__file__).resolve().parents[1]/'scripts/flowmap_run_state.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state=tmp_path/'checkpoints/state/step_000002';state.mkdir(parents=True)
    (state/'trainer_state.json').write_text(json.dumps(dict(global_step=2,max_steps=2000,flowmap_ema_updates=2)))
    (state/'flowmap_ema.pt').write_bytes(b'x')
    tag=state/'pytorch_model';tag.mkdir()
    (tag/'mp_rank_00_model_states.pt').write_bytes(b'x')
    (state/'scheduler.bin').write_bytes(b'x')
    for rank in range(4):
        (state/f'random_states_{rank}.pkl').write_bytes(b'x')
        (tag/f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'x')
    weights=tmp_path/'checkpoints/weights'
    for d in (weights,weights/'ema_0.999',weights/'ema_0.9999'):
        d.mkdir(parents=True,exist_ok=True);(d/'step_000002.pt').write_bytes(b'x')
    (tmp_path/'segment_status.json').write_text(json.dumps(dict(step=2,max_steps=2000,complete=False,state_path=str(state))))
    with (tmp_path/'train_update_metrics.jsonl').open('w') as f:
        for step in (1,2):f.write(json.dumps(dict(step=step,examples=192,grad_norm=1.,learning_rate=1e-7,
            metrics={'loss':{'mean':.1,'microbatch_min':.05,'microbatch_max':.2}}))+'\n')
    with (tmp_path/'preview_metrics.jsonl').open('w') as f:
        for nfe in (1,2,4):
            for rank in range(4):(tmp_path/f'preview_{nfe}_rank_{rank:03d}.mp4').write_bytes(b'x')
            f.write(json.dumps(dict(step=2,nfe=nfe,has_pointmap=True,dino_mse=.5,val_loss=.1,
                video_path=str(tmp_path/f'preview_{nfe}_rank_000.mp4')))+'\n')
    assert module.validate_segment(tmp_path,minimum_bytes=1)['step']==2
    with pytest.raises(ValueError,match='make progress'):
        module.validate_segment(tmp_path,initial_step=2,minimum_bytes=1)
    shard=tag/'zero_pp_rank_3_mp_rank_00_optim_states.pt';shard.unlink()
    with pytest.raises(ValueError,match='optimizer shards'):
        module.validate_segment(tmp_path,minimum_bytes=1)
    shard.write_bytes(b'x')
    (tmp_path/'preview_4_rank_003.mp4').unlink()
    with pytest.raises(ValueError,match='Missing preview video'):
        module.validate_segment(tmp_path,minimum_bytes=1)

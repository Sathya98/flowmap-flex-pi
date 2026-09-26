"""Real checkpoint round trips and credentials isolated to a single process."""
import copy
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from flexpi.trainer import Wan22Trainer
from flexpi.utils.flowmap_ema import EvaluationEMA
from flexpi.utils.isolated_tracking import isolate_wandb
from flexpi.utils.samplers import ResumableEpochSampler


def test_isolated_offline_ignores_inherited_account(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'environ', dict(os.environ, WANDB_API_KEY='unrelated-test-key',
        WANDB_ENTITY='someone-else', WANDB_MODE='online', WANDB_SERVICE='unrelated-service'))
    isolate_wandb(tmp_path, 'offline')
    assert os.environ['WANDB_MODE']=='offline'
    assert 'WANDB_API_KEY' not in os.environ and 'WANDB_ENTITY' not in os.environ
    assert 'WANDB_SERVICE' not in os.environ
    assert Path(os.environ['NETRC']).read_text()==''
    assert Path(os.environ['WANDB_CONFIG_DIR']).parent==tmp_path/'tracking'


def test_online_requires_explicit_job_key_and_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'environ', dict(os.environ, WANDB_API_KEY='unrelated-test-key'))
    os.environ.pop('FLOWMAP_WANDB_API_KEY_FILE',None)
    with pytest.raises(ValueError,match='shared login fallback is disabled'):
        isolate_wandb(tmp_path,'online','my-workspace')
    key=tmp_path/'test-key';key.write_text('own-test-key');key.chmod(0o600)
    os.environ['FLOWMAP_WANDB_API_KEY_FILE']=str(key)
    isolate_wandb(tmp_path,'online','my-workspace')
    assert os.environ['WANDB_API_KEY']=='own-test-key'
    assert os.environ['WANDB_ENTITY']=='my-workspace'
    key.chmod(0o644)
    with pytest.raises(ValueError,match='mode 600'):
        isolate_wandb(tmp_path,'online','my-workspace')


@pytest.mark.parametrize('saved_horizon', [20, 2000])
def test_real_full_state_restore_and_changed_schedule(tmp_path, saved_horizon):
    from accelerate import Accelerator
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.,2.]))
        def save_checkpoint(self,path,**kwargs):torch.save(self.state_dict(),path)
    trainer=Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator=Accelerator(cpu=True,step_scheduler_with_optimizer=False)
    trainer.model=Model();trainer.learning_rate=1e-3
    trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=trainer.learning_rate)
    trainer.scheduler=trainer._build_scheduler(saved_horizon,int(saved_horizon*.05))
    trainer._scheduler_base_lrs=[trainer.learning_rate]
    trainer.model,trainer.optimizer,trainer.scheduler=trainer.accelerator.prepare(
        trainer.model,trainer.optimizer,trainer.scheduler)
    trainer.max_steps=saved_horizon;trainer.global_step=3;trainer.epoch=2;trainer.batch_in_epoch=5
    trainer.batch_size=1
    trainer.cfg=OmegaConf.create({'keep_last_n_states':2,'keep_last_n_weights':2})
    trainer.weights_dir=str(tmp_path/'weights');trainer.state_dir=str(tmp_path/'state')
    Path(trainer.weights_dir).mkdir();Path(trainer.state_dir).mkdir()
    trainer.flowmap_ema=EvaluationEMA(trainer.model,(.999,.9999))
    trainer.train_sampler=ResumableEpochSampler(list(range(20)),42,1,1)
    def update():
        trainer.optimizer.zero_grad()
        value=torch.rand(2)+random.random()+np.random.rand()
        trainer.accelerator.backward((trainer.model.weight*value).square().sum())
        trainer.optimizer.step();trainer.scheduler.step()
        trainer.flowmap_ema.update(trainer.model)
    for _ in range(3):update()
    saved_optimizer=copy.deepcopy(trainer.optimizer.state_dict())
    saved_scheduler=copy.deepcopy(trainer.scheduler.state_dict())
    checkpoint=trainer.save_checkpoint()
    state=Path(checkpoint['state_path'])
    assert (state/'optimizer.bin').exists() and (state/'scheduler.bin').exists()
    assert (state/'random_states_0.pkl').exists() and (state/'flowmap_ema.pt').exists()
    assert json.loads((state/'trainer_state.json').read_text())['max_steps']==saved_horizon
    update();expected=trainer.model.weight.detach().clone()
    update()  # Move model, optimizer, RNG, scheduler and EMA beyond the checkpoint.
    trainer.global_step=trainer.epoch=trainer.batch_in_epoch=99
    trainer.load_training_state(checkpoint['state_path'])
    assert (trainer.global_step,trainer.epoch,trainer.batch_in_epoch)==(3,2,5)
    assert trainer.train_sampler.epoch_offset==2 and trainer.train_sampler.resume_batch_offset==5
    assert trainer.flowmap_ema.updates==3
    assert trainer.scheduler.state_dict()==saved_scheduler
    for key,values in saved_optimizer['state'].items():
        for name,value in values.items():
            torch.testing.assert_close(trainer.optimizer.state_dict()['state'][key][name],value)
    update()
    torch.testing.assert_close(trainer.model.weight,expected,rtol=0,atol=0)
    # Both extending and shortening the horizon retain Adam's moments/progress.
    before=copy.deepcopy(trainer.optimizer.state_dict()['state'])
    trainer.max_steps=100;trainer.global_step=4
    trainer._reanchor_lr_schedule()
    assert trainer.scheduler.state_dict()['last_epoch']==4
    assert trainer.scheduler.state_dict()['_schedulers'][1]['T_max']==95
    assert trainer.optimizer.param_groups[0]['lr']==pytest.approx(8.4e-4)
    for key,values in before.items():
        for name,value in values.items():
            torch.testing.assert_close(trainer.optimizer.state_dict()['state'][key][name],value)


def test_real_offline_sdk_does_not_authenticate(tmp_path,monkeypatch):
    monkeypatch.setattr(os,'environ',dict(os.environ,WANDB_API_KEY='unrelated-test-key',
        WANDB_ENTITY='someone-else',WANDB_MODE='online'))
    trainer=Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator=SimpleNamespace(is_main_process=True)
    trainer.wandb_enabled=True;trainer.output_dir=str(tmp_path);trainer.global_step=1
    trainer.cfg=OmegaConf.create({'max_steps':2000,'wandb':dict(enabled=True,id=None,
        workspace=None,project='flowmap-local-test',name='offline-test',group=None,
        mode='offline',isolated=True)})
    trainer._init_wandb()
    try:
        assert trainer.wandb_run.settings.mode=='offline'
        assert not trainer.wandb_run.settings.api_key
        assert not trainer.wandb_run.entity
        trainer._wandb_log({'train/loss':.25})
    finally:trainer._finish_wandb()
    assert list(tmp_path.glob('wandb/offline-run-*/run-*.wandb'))
    assert json.loads((tmp_path/'wandb_run.json').read_text())['url'] is None

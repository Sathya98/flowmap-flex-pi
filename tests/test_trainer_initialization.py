"""Guard pretrained initialization against stale mixed-precision master weights."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import itertools
import json
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from flexpi.trainer import Wan22Trainer


class InitializationTests(unittest.TestCase):
    def test_weights_precede_master_creation_and_full_resume_follows_it(self):
        for mode, clip in itertools.product(('pretrained','weight_resume','full_resume'), (None, 0.25, 0.0, 1.0)):
            with self.subTest(mode=mode, deepspeed_clip=clip), tempfile.TemporaryDirectory() as folder:
                events=[]
                class Model(torch.nn.Module):
                    def __init__(self):
                        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.))
                    def load_checkpoint(self,path,**kwargs):
                        events.append('load_weights')
                        with torch.no_grad():self.weight.fill_(7.)
                class FakeAccelerator:
                    def __init__(self,**kwargs):
                        plugin=None
                        if clip is not None:
                            from accelerate.utils import DeepSpeedPlugin
                            config=json.loads((Path(__file__).resolve().parents[1]/'scripts/ds_configs/ds_zero2_config.json').read_text())
                            plugin=DeepSpeedPlugin(hf_ds_config=config)
                        self.state=SimpleNamespace(deepspeed_plugin=plugin)
                        self.distributed_type='test';self.num_processes=1;self.process_index=0
                        self.mixed_precision='no';self.gradient_accumulation_steps=1
                        self.device=torch.device('cpu');self.is_main_process=True
                    def gather(self,x):return x
                    def prepare(self,model,*objects):
                        events.append('prepare_master')
                        if self.state.deepspeed_plugin is not None:
                            from deepspeed.runtime.config import get_gradient_clipping
                            plugin=self.state.deepspeed_plugin
                            # Exercise installed Accelerate's config processing:
                            # its default must not replace the trainer's setting.
                            plugin.deepspeed_config_process(must_match=False, gradient_clipping=1.0,
                                train_batch_size=1, train_micro_batch_size_per_gpu=1,
                                gradient_accumulation_steps=1)
                            self.effective_clip=get_gradient_clipping(plugin.deepspeed_config)
                        self.master=model.weight.detach().float().clone()
                        return (model,*objects)
                    def unwrap_model(self,model):return model
                class Trainer(Wan22Trainer):
                    def _build_loader(self,*args,**kwargs):return [None]
                    def _check_pretrain_norm_mode_compat(self,path):pass
                    def _prepare_flowmap_teacher(self):events.append('prepare_teacher')
                    def load_training_state(self,path):events.append('restore_full_state')
                path=Path(folder)/'weights.pt';path.touch()
                state=Path(folder)/'state';state.mkdir()
                cfg=OmegaConf.create(dict(model={},output_dir=folder,learning_rate=0.,weight_decay=0.,
                    batch_size=1,num_workers=0,num_epochs=1,max_steps=2,log_every=1,save_every=0,
                    eval_every=0,eval_num_inference_steps=2,gradient_accumulation_steps=1,
                    max_grad_norm=1. if clip is None else clip,seed=42,mixed_precision='no',wandb={'enabled':False},
                    resume=str(state if mode=='full_resume' else path) if mode!='pretrained' else None,
                    pretrained_ckpt=str(path) if mode=='pretrained' else None,
                    pretrained_ckpt_strict_shape=True))
                with patch('flexpi.trainer.Accelerator',FakeAccelerator):
                    trainer=Trainer(Model(),[None],cfg=cfg)
                if clip is not None:
                    self.assertEqual(trainer.accelerator.effective_clip,clip)
                if mode=='full_resume':
                    self.assertNotIn('load_weights',events)
                    self.assertLess(events.index('prepare_master'),events.index('restore_full_state'))
                else:
                    self.assertLess(events.index('load_weights'),events.index('prepare_teacher'))
                    self.assertLess(events.index('load_weights'),events.index('prepare_master'))
                    self.assertEqual(events.count('load_weights'),1)
                    # Emulate ZeRO's copy-back at a zero-LR step. Old ordering
                    # copied 0 into a model that had just loaded checkpoint 7.
                    with torch.no_grad():trainer.model.weight.copy_(trainer.accelerator.master)
                    self.assertEqual(float(trainer.model.weight),7.)


if __name__=='__main__':unittest.main()

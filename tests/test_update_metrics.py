"""Regression checks for complete-update loss reporting."""
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from flexpi.utils.update_metrics import UpdateLossMetrics
from flexpi.trainer import Wan22Trainer


class UpdateMetricsTests(unittest.TestCase):
    def test_weighted_mean_rank_range_and_reset(self):
        tracker=UpdateLossMetrics()
        loss=torch.tensor(1.,requires_grad=True)
        tracker.add(loss,{'dino':2.},2)
        tracker.add(torch.tensor(5.),{'dino':10.},1)
        other=torch.tensor([[[12.,4.,4.,3.,1.],[24.,8.,8.,3.,1.]]])
        fake=SimpleNamespace(gather=lambda x:torch.cat((x,other),0))
        report=tracker.finish(fake)
        self.assertAlmostEqual(report['metrics']['loss']['mean'],19/6,places=6)
        self.assertEqual(report['metrics']['loss']['microbatch_min'],1)
        self.assertEqual(report['metrics']['loss']['microbatch_max'],5)
        self.assertEqual(report['examples'],6)
        self.assertEqual(report['microbatches'],3)
        self.assertIsNone(loss.grad)
        tracker.add(torch.tensor(7.),{},1)
        self.assertEqual(tracker.finish(SimpleNamespace(gather=lambda x:x))['metrics']['loss']['mean'],7)

    def test_training_loop_reports_both_accumulated_updates(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.))
            def training_loss(self,sample):
                loss=self.weight*sample['value']
                return loss,{'loss_flowmap_dino':float(loss.detach()*2)}
        class Accelerator:
            num_processes=1
            process_index=0
            is_main_process=True
            optimizer_step_was_skipped=False
            count=0
            @contextmanager
            def accumulate(self,model):
                self.count+=1;self.sync_gradients=self.count%3==0
                yield
            def unwrap_model(self,model):return model
            def autocast(self):return nullcontext()
            def backward(self,loss):(loss/3).backward()
            def gather(self,value):return value
            def clip_grad_norm_(self,parameters,limit):return torch.nn.utils.clip_grad_norm_(parameters,limit)
        trainer=Wan22Trainer.__new__(Wan22Trainer)
        trainer.model=Model();trainer.accelerator=Accelerator()
        trainer.optimizer=torch.optim.SGD(trainer.model.parameters(),lr=0.)
        trainer.scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,lambda _:1.)
        trainer._set_dit_only_train_mode=lambda:None
        trainer._estimate_eta=lambda:('00:00:00',1.)
        trainer.save_checkpoint=lambda:dict(weights_path='unused',state_path='unused')
        trainer.global_step=trainer.epoch=trainer.batch_in_epoch=0
        trainer.max_steps=2;trainer.eval_every=trainer.save_every=0
        trainer.log_every=1;trainer.batch_size=2;trainer.gradient_accumulation_steps=3
        trainer.max_grad_norm=1.;trainer.train_sampler=None
        # Real FlexPi batches can omit video and carry per_cam + action instead.
        trainer.train_loader=[{'action':torch.zeros(2,1,1),'value':v} for v in (1.,3.,5.,2.,4.,6.)]
        logged=[];trainer._wandb_log=logged.append
        with tempfile.TemporaryDirectory() as folder:
            trainer.output_dir=folder
            trainer.train()
            rows=[json.loads(line) for line in (Path(folder)/'train_update_metrics.jsonl').read_text().splitlines()]
        self.assertEqual([r['train/loss'] for r in logged],[3.,4.])
        self.assertEqual([r['train/loss_flowmap_dino'] for r in logged],[6.,8.])
        self.assertEqual([r['metrics']['loss']['microbatch_min'] for r in rows],[1.,2.])
        self.assertEqual([r['metrics']['loss']['microbatch_max'] for r in rows],[5.,6.])
        self.assertEqual([r['examples'] for r in rows],[6,6])
        self.assertEqual(trainer.global_step,2)


if __name__=='__main__':unittest.main()

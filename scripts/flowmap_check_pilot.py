"""Fail a pilot job if logging/checkpoint checks fail; optional zero-LR weight gate."""
import argparse
import json
import math
from pathlib import Path
from omegaconf import OmegaConf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--zero-lr',action='store_true')
    args=parser.parse_args()
    root=Path(args.run_dir)
    cfg=OmegaConf.load(root/'config.yaml')
    rows=[json.loads(line) for line in (root/'train_update_metrics.jsonl').read_text().splitlines()]
    steps=int(cfg.max_steps)
    expected=int(cfg.batch_size)*int(cfg.gradient_accumulation_steps)*4
    assert [r['step'] for r in rows]==list(range(1,steps+1)), 'Missing update logs'
    for row in rows:
        assert row['examples']==expected, (row['examples'],expected)
        assert math.isfinite(row['grad_norm']), 'Nonfinite gradient norm'
        for values in row['metrics'].values():
            assert all(math.isfinite(v) for v in values.values()), 'Nonfinite loss statistic'
            assert values['microbatch_min']-1e-5 <= values['mean'] <= values['microbatch_max']+1e-5
    weight_path=root/f'checkpoints/weights/step_{steps:06d}.pt'
    state_path=root/f'checkpoints/state/step_{steps:06d}/trainer_state.json'
    assert weight_path.is_file() and weight_path.stat().st_size>1_000_000_000
    state=json.loads(state_path.read_text())
    assert state['global_step']==steps
    result=dict(passed=True,updates=steps,examples_per_update=expected,
        losses=[r['metrics']['loss'] for r in rows],grad_norms=[r['grad_norm'] for r in rows])
    flow=cfg.model.flow_map
    uses_ema=bool(flow.get('ema_decays', [])) and (
        flow.objective in ('lsd','esd','psd_m','psd_u') or flow.get('distill_ema', False))
    if uses_ema:
        ema_state=state_path.parent/'flowmap_ema.pt'
        assert ema_state.is_file() and ema_state.stat().st_size>1_000_000_000
        result['ema_updates']=state['flowmap_ema_updates']
        assert 0 <= result['ema_updates'] <= steps
        for decay in flow.ema_decays:
            ema_weights=root/f'checkpoints/weights/ema_{float(decay)}/step_{steps:06d}.pt'
            assert ema_weights.is_file() and ema_weights.stat().st_size>1_000_000_000
    if args.zero_lr:
        assert cfg.learning_rate==0, 'Zero-LR guard requires zero learning rate'
        import torch
        torch.set_num_threads(4)
        reference=torch.load(cfg.pretrained_ckpt,map_location='cpu',weights_only=False)
        actual=torch.load(weight_path,map_location='cpu',weights_only=False)
        checked=0;failed=[]
        modules=('mot','dino_embedder','dino_proj_out','dino_feature_norm',
                 'pt_patch_embedding','pt_head','proprio_encoder')
        for name in modules:
            if name not in reference:continue
            for key,expected_tensor in reference[name].items():
                value=actual[name][key]
                if not torch.equal(value,expected_tensor.to(value.dtype)):
                    failed.append(f'{name}.{key}')
                checked+=1
        result['zero_lr_tensors_checked']=checked
        result['changed_tensors']=failed
        assert checked>0
        result['passed']=not failed
        (root/'pilot_check.json').write_text(json.dumps(result,indent=2)+'\n')
        assert not failed, f'Zero-LR step changed {len(failed)} pretrained tensors: {failed[:8]}'
    else:
        (root/'pilot_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Read-only fixed-panel PFMM diagnostics on released and distilled checkpoints.

Writes incremental JSONL, plus a summary. Does not optimize or change defaults.
Run on a single GPU with room for the BF16 student, frozen teacher and encoders.
"""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from flexpi.utils import misc
from flexpi.utils.logging_config import setup_logging
from flexpi.utils.config_resolvers import register_default_resolvers
from flexpi.models.helpers.adaptation import clone_teacher, validate_teacher_payload
from flexpi.models.helpers.flowmap_diagnostics import paired_time_proposals, pfmm_probe, lmd_probe, target_difference


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [finite_json(v) for v in value]
    return value


def make_cases(sample_count, random_pairs=8):
    cases=[]
    def add(kind, sample, s, t, seed=2026):
        cases.append(dict(case_id=len(cases),kind=kind,sample=sample,
                          source=float(s),target=float(t),noise_seed=seed))
    add('fixed',0,.5,.1)
    add('repeat',0,.5,.1)
    for seed in (2027,2028,2029):add('noise_only',0,.5,.1,seed)
    for index in range(1,sample_count):add('data_only',index,.5,.1)
    for s in (.01,.05,.2,.5,.9,1.):
        for fraction in (.1,.9):add('time_grid',0,s,s*(1-fraction))
    add('full_interval',0,1.,0.)
    for kind,(source,target) in paired_time_proposals(random_pairs).items():
        for s,t in zip(source,target):add(kind,0,s,t)
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--student',action='append',default=[],help='label=/path/to/weights.pt; baseline always included')
    parser.add_argument('--teacher-steps',type=int,default=16)
    parser.add_argument('--random-pairs',type=int,default=8)
    parser.add_argument('--lmd-probe',action='store_true',
                        help='Also measure exact LMD residuals against the same frozen teacher')
    args=parser.parse_args()
    if args.teacher_steps<2 or args.random_pairs<1:parser.error('Invalid probe budget')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for real checkpoint probes')
    root=Path(__file__).resolve().parents[1]
    out=Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'measurements.jsonl').exists():raise FileExistsError('Refusing to overwrite previous diagnostics')
    misc.register_work_dir(out)
    register_default_resolvers()
    setup_logging(log_level=logging.INFO,is_main_process=True)
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='flowmap_libero_pfmm_smoke')
    cfg.output_dir=str(out)
    cfg.model.flow_map.teacher_steps=args.teacher_steps
    OmegaConf.save(cfg,out/'config.yaml')
    checkpoints=[('released',str(cfg.pretrained_ckpt))]
    for specification in args.student:
        label,path=specification.split('=',1)
        if not label or label in {k for k,_ in checkpoints}:raise ValueError('Student labels must be unique')
        if not Path(path).is_file():raise FileNotFoundError(path)
        checkpoints.append((label,path))
    model=instantiate(cfg.model,model_dtype=torch.bfloat16,device='cuda:0')
    payload=model.load_checkpoint(cfg.pretrained_ckpt,strict_shape=True)
    validate_teacher_payload(model,payload)
    del payload
    model.eval().requires_grad_(False)
    # Clone AFTER loading the released checkpoint. The teacher remains fixed
    # as subsequent student checkpoints are loaded. Frozen encoders are shared.
    teacher=clone_teacher(model)
    object.__setattr__(model,'flow_map_teacher',teacher)
    dataset=instantiate(cfg.data.train)
    counts=dataset.per_dataset_num_frames
    episode_index=dataset.lerobot_dataset.episode_data_index
    indices=[];samples=[];offset=0
    for count in counts:
        selected=None
        # Select a real episode start inside each suite. Mid-suite frame
        # indices may land at an episode end and mask every future image.
        starts=[int(v) for v in episode_index['from'] if offset <= int(v) < offset+count]
        for idx in starts:
            sample=dataset[idx]
            if bool(sample['image_is_pad'].any()) or bool(sample['action_is_pad'].any()):
                continue
            indices.append(idx);samples.append(default_collate([sample]));selected=idx
            break
        if selected is None:
            raise RuntimeError(f'No fully unpadded diagnostic example in suite at offset {offset}')
        offset+=count
    cases=make_cases(len(samples),args.random_pairs)
    source_paths = [Path(__file__).resolve(),
        root/'src/flexpi/models/helpers/flowmap.py', root/'flowmap_core/src/flowmap_core/flowmap.py',
        root/'src/flexpi/models/helpers/flowmap_training.py',
        root/'src/flexpi/models/helpers/flowmap_diagnostics.py',
        root/'src/flexpi/models/helpers/dino.py', root/'src/flexpi/models/flexpi.py']
    sources = {}
    for source in source_paths:
        relative = source.relative_to(root)
        content = source.read_bytes()
        saved = out/'source_snapshot'/relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(content)
        sources[str(relative)] = hashlib.sha256(content).hexdigest()
    manifest=dict(source_sha256=sources, arguments=vars(args),checkpoints=checkpoints,indices=indices,cases=cases,
        padding_counts=[dict(image=int(s['image_is_pad'].sum()), action=int(s['action_is_pad'].sum())) for s in samples],
        prompts=[s['prompt'] for s in samples],torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),seed=2026,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    records=[]
    def emit(record):
        records.append(record)
        with (out/'measurements.jsonl').open('a') as handle:
            handle.write(json.dumps(finite_json(record),allow_nan=False)+'\n')
        print('[probe]',record['checkpoint'],record['kind'],record.get('case_id',''),flush=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for label,path in checkpoints:
            if label!='released':
                payload=model.load_checkpoint(path,strict_shape=True)
                del payload
            model.eval().requires_grad_(False)
            # Rebuild conditioning after every checkpoint: proprio weights change.
            inputs=[model.build_inputs(sample) for sample in samples]
            torch.cuda.reset_peak_memory_stats()
            for case in cases:
                idx=case['sample']
                torch.cuda.synchronize();start=time.perf_counter()
                report,_=pfmm_probe(model,samples[idx],inputs[idx],case['source'],case['target'],
                                   case['noise_seed'],args.teacher_steps)
                if args.lmd_probe:
                    lmd=lmd_probe(model,samples[idx],inputs[idx],case['source'],case['target'],case['noise_seed'])
                    report['lmd']=lmd
                    report['metrics'].update({'lmd_'+k:v for k,v in lmd['metrics'].items()})
                    report['metrics']['lmd_loss']=lmd['loss']
                    report['finite']=report['finite'] and lmd['finite']
                torch.cuda.synchronize()
                emit(dict(checkpoint=label,**case,report=report,
                    seconds=time.perf_counter()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            if label=='released':
                # Fixed observations/noise/times for ALL four integration budgets.
                for s,t in ((1.,0.),(.9,.09),(.5,.05),(.2,.02),(.05,.005),(.01,.001)):
                    targets={}
                    for steps in (8,16,32,64):
                        report,target=pfmm_probe(model,samples[0],inputs[0],s,t,teacher_steps=steps)
                        targets[steps]=target
                        emit(dict(checkpoint=label,kind='teacher_budget',source=s,target=t,
                                  teacher_steps=steps,report=report))
                    comparisons={f'{a}_vs_{b}':target_difference(model,inputs[0],targets[a],targets[b],s-t)
                                 for a,b in ((8,16),(16,32),(32,64),(8,64),(16,64))}
                    emit(dict(checkpoint=label,kind='teacher_convergence',source=s,target=t,comparisons=comparisons))
            del inputs
    # Summaries use identical cases, NOT unrelated random minibatch averages.
    summary={}
    for label,_ in checkpoints:
        rows=[r for r in records if r['checkpoint']==label and 'case_id' in r]
        first=next(r for r in rows if r['kind']=='fixed')['report']['metrics']
        repeat=next(r for r in rows if r['kind']=='repeat')['report']['metrics']
        summary[label]=dict(repeat_max_abs_difference=max(abs(first[k]-repeat[k]) for k in first),
            all_finite=all(r['report']['finite'] for r in rows),groups={})
        for kind in sorted({r['kind'] for r in rows}):
            group=[r for r in rows if r['kind']==kind]
            summary[label]['groups'][kind]={k:float(np.mean([r['report']['metrics'][k] for r in group]))
                                           for k in first}
    (out/'summary.json').write_text(json.dumps(finite_json(summary),indent=2,allow_nan=False)+'\n')
    print('[probe] completed:',out,flush=True)


if __name__=='__main__':main()

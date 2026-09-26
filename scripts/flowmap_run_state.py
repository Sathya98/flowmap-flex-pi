"""Validate resumable LMD segments before automatically scheduling continuation."""
import argparse
import json
import math
from pathlib import Path


def latest_state(root):
    states = sorted((root/'checkpoints/state').glob('step_*/trainer_state.json'))
    return states[-1].parent if states else None


def ema_decays(root):
    """The run's exported EMA decays (its frozen config), default (0.999, 0.9999)."""
    config = root/'prepared_config.yaml'
    if not config.is_file():
        return (.999, .9999)
    from omegaconf import OmegaConf
    decays = OmegaConf.select(OmegaConf.load(config), 'model.flow_map.ema_decays')
    return tuple(float(d) for d in decays) if decays is not None else (.999, .9999)


def validate_segment(root, initial_step=0, minimum_bytes=1_000_000_000):
    status=json.loads((root/'segment_status.json').read_text())
    step=int(status['step']); horizon=int(status['max_steps'])
    if not initial_step < step <= horizon:
        raise ValueError('Segment must make progress within the planned horizon')
    state=Path(status['state_path'])
    progress=json.loads((state/'trainer_state.json').read_text())
    if (progress['global_step'] != step or progress.get('max_steps') != horizon
            or not progress.get('flowmap_ema_updates', 0)):
        raise ValueError('Checkpoint progress or EMA updates are missing')
    # The full state, not the exported model weights, makes a run resumable.
    tag=state/'pytorch_model'
    shards=list(tag.glob('*_optim_states.pt'))
    if len(shards) != 4 or any(p.stat().st_size < minimum_bytes for p in shards):
        raise ValueError('Missing/incomplete ZeRO optimizer shards')
    for path in [state/'scheduler.bin',tag/'mp_rank_00_model_states.pt'] + [
            state/f'random_states_{rank}.pkl' for rank in range(4)]:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing full training state: {path}')
    required=[root/f'checkpoints/weights/step_{step:06d}.pt', state/'flowmap_ema.pt']
    required += [root/f'checkpoints/weights/ema_{d}/step_{step:06d}.pt' for d in ema_decays(root)]
    for path in required:
        if not path.is_file() or path.stat().st_size < minimum_bytes:
            raise ValueError(f'Missing/incomplete checkpoint: {path}')
    rows=[json.loads(line) for line in (root/'train_update_metrics.jsonl').read_text().splitlines()]
    current={int(r['step']):r for r in rows if initial_step < r['step'] <= step}
    if set(current) != set(range(initial_step+1,step+1)):
        raise ValueError('Missing optimizer-update metrics in this segment')
    for row in current.values():
        values=[row['grad_norm'], row['learning_rate']]
        values += [v for metric in row['metrics'].values() for v in metric.values()]
        if row['examples'] != 192 or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid accumulated-batch metrics')
    previews=[json.loads(line) for line in (root/'preview_metrics.jsonl').read_text().splitlines()]
    final={r['nfe']:r for r in previews if r['step']==step}
    if not {1,2,4} <= final.keys():
        raise ValueError('Missing final future previews at NFEs 1/2/4')
    for nfe in (1,2,4):
        row=final[nfe]
        if not row.get('has_pointmap') or not math.isfinite(row.get('dino_mse',float('nan'))):
            raise ValueError('Incomplete DINO/pointmap future visualization')
        if not math.isfinite(row['val_loss']):
            raise ValueError('Nonfinite fixed-preview residual')
        for rank in range(4):
            video=Path(row['video_path'].replace('rank_000',f'rank_{rank:03d}'))
            if not video.is_file() or not video.stat().st_size:
                raise ValueError(f'Missing preview video: {video}')
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('latest','validate','step','complete'))
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--initial-step',type=int,default=0)
    args=parser.parse_args()
    if args.command=='latest':
        path=latest_state(args.run_dir)
        print(path or '')
    elif args.command=='validate':
        print(json.dumps(validate_segment(args.run_dir,args.initial_step),indent=2))
    elif args.command=='step':
        path=latest_state(args.run_dir)
        print(json.loads((path/'trainer_state.json').read_text())['global_step'] if path else 0)
    else:
        print(int(json.loads((args.run_dir/'segment_status.json').read_text())['complete']))


if __name__=='__main__':main()

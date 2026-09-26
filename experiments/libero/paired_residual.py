"""Calibrate one bounded disturbance, then evaluate untouched held-out pairs."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf, open_dict

from experiments.libero.execution_perturbation import ExecutionPerturbation, select_calibrated_candidate
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env, save_rollout_video
from flexpi.utils.pytorch_utils import set_global_seed


DIRECTIONS = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.))
CANDIDATES = (('delay', 4), ('delay', 8), ('delay', 16),
              ('translation_bias', .1), ('translation_bias', .25), ('translation_bias', .5))


def run_paired_task(task, initial_states, model, processor, cfg, video_dir, *, episode_fn,
                    action_horizon, input_w, input_h, model_device):
    root = Path(cfg.EVALUATION.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'paired_outcomes.jsonl').exists():
        raise FileExistsError('Refusing to append a second experiment to the same directory')
    pairs = int(cfg.EVALUATION.get('paired_holdout_pairs', 8))
    calibration_count = 4
    if len(initial_states) < calibration_count+pairs:
        raise ValueError('Need distinct initial states for calibration and evaluation')
    if int(cfg.EVALUATION.replan_steps) != 32 or action_horizon != 32:
        raise ValueError('This protocol uses the released 32-action LIBERO horizon')
    OmegaConf.save(cfg, root/'protocol_config.yaml')
    protocol = dict(calibration_initial_states=list(range(calibration_count)),
        holdout_initial_states=list(range(calibration_count, calibration_count+pairs)),
        candidates=[dict(kind=k, strength=s) for k,s in CANDIDATES],
        calibration_rule='First level with mixed terminal outcomes; otherwise first failing level, '
                         'or largest tested level if all succeed. No score-based selection.',
        target_chunk=1, perturb_action_indices=[32,47], residual_observed_after_actions=48,
        unmodified_remaining_action_indices=[48,63],
        primary_outcome='task failure by the standard 400-action episode limit',
        secondary_outcome='task not completed by action 96 (delay, not necessarily terminal failure)',
        primary_analysis='one prespecified chunk per episode; leave-one-pair-out predictions',
        paired_matching='initial state, environment seed, policy seed, probe seed; alternating run order',
        seed_rule='42 + initial state index', direction_rule='cycle +x,-x,+y,-y by initial state index',
        note='Delay queue is flushed at action 48; translation bias clips controller inputs to [-1,1]. '
             'Only the first 16 actions of chunk 1 can be changed. No later disturbances.')
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    source_root = Path(__file__).resolve().parents[2]
    hashes = {}
    for relative in ('experiments/libero/paired_residual.py',
                     'experiments/libero/execution_perturbation.py',
                     'experiments/libero/eval_libero_single.py',
                     'src/flexpi/models/helpers/residual_sensitivity.py',
                     'scripts/summarize_paired_residual.py'):
        content = (source_root/relative).read_bytes()
        target = root/'paired_source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    (root/'paired_source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    env, description = get_libero_env(task, LIBERO_ENV_RESOLUTION, int(cfg.seed), camera_depths=True,
        prompt_source=str(cfg.EVALUATION.get('prompt_source', 'task.language')))
    outcomes = []

    def run(state_index, kind, strength, phase, episode, pair_id, condition):
        local = copy.deepcopy(cfg)
        seed = 42+state_index
        local.seed = seed
        with open_dict(local.EVALUATION):
            local.EVALUATION.residual_sensitivity = phase == 'evaluation'
            local.EVALUATION.residual_chunk_indices = [1]
            local.EVALUATION.residual_max_chunks = 2
            local.EVALUATION.residual_probe_seed = 2026+1000*state_index
            local.EVALUATION.residual_metadata = dict(phase=phase, pair_id=pair_id,
                condition=condition, initial_state_index=state_index,
                perturbation_kind=kind, perturbation_strength=float(strength))
        set_global_seed(seed, get_worker_init_fn=False)
        env.seed(seed)
        perturbation = ExecutionPerturbation(kind, strength,
            direction=DIRECTIONS[state_index % len(DIRECTIONS)])
        model._execution_perturbation = perturbation
        start = time.perf_counter()
        success, images, _, _ = episode_fn(env=env, initial_state=initial_states[state_index],
            task_description=description, model=model, processor=processor, cfg=local,
            episode_idx=episode, action_horizon=action_horizon, input_w=input_w,
            input_h=input_h, model_device=model_device)
        row = dict(phase=phase, episode=episode, pair_id=pair_id, condition=condition,
            initial_state_index=state_index, seed=seed, **model._last_episode_execution,
            **perturbation.report(), seconds=time.perf_counter()-start)
        outcomes.append(row)
        with (root/'paired_outcomes.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        print('[paired-outcome]', json.dumps({k:v for k,v in row.items() if k != 'action_trace'}), flush=True)
        save_rollout_video(video_dir, images, f'{phase}_state{state_index}_{condition}_{kind}{strength}',
                           success=success, task_description=description)
        model._execution_perturbation = None
        return row

    calibration = []
    try:
        controls = {}
        for index in range(calibration_count):
            controls[index] = run(index, 'clean', 0, 'calibration', index, index, 'clean')
        for order, (kind, strength) in enumerate(CANDIDATES):
            records = [run(index, kind, strength, 'calibration',
                           calibration_count*(order+1)+index, index, 'perturbed')
                       for index in range(calibration_count)]
            for row in records:
                if row['planned_action_sha256'] != controls[row['initial_state_index']]['planned_action_sha256']:
                    raise RuntimeError('Calibration pair diverged before disturbance; check reset/seeding')
            candidate = dict(kind=kind, strength=strength, order=order,
                failures=sum(not row['success'] for row in records), trials=len(records),
                failure_rate=float(np.mean([not row['success'] for row in records])),
                deadline_96_failure_rate=float(np.mean([not row['success_by_96'] for row in records])))
            calibration.append(candidate)
            (root/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
            print('[calibration]', json.dumps(candidate), flush=True)
            if 0 < candidate['failure_rate'] < 1:
                break
        selected = select_calibrated_candidate(calibration)
        (root/'selection.json').write_text(json.dumps(dict(selected=selected,
            mixed_calibration=0 < selected['failure_rate'] < 1,
            frozen_before_holdout=True, calibration=calibration), indent=2)+'\n')
        print('[selection]', json.dumps(selected), flush=True)
        for pair in range(pairs):
            state_index = calibration_count+pair
            records = {}
            order = ('clean','perturbed') if pair % 2 == 0 else ('perturbed','clean')
            for condition in order:
                kind, strength = (('clean', 0) if condition == 'clean'
                                  else (selected['kind'], selected['strength']))
                records[condition] = run(state_index, kind, strength, 'evaluation',
                    2*pair + int(condition == 'perturbed'), pair, condition)
            if records['clean']['planned_action_sha256'] != records['perturbed']['planned_action_sha256']:
                raise RuntimeError(f'Pair {pair} diverged before disturbance; invalid matched comparison')
    finally:
        model._execution_perturbation = None
        model._residual_sensitivity_probe = None
        env.close()
    evaluation = [r for r in outcomes if r['phase'] == 'evaluation']
    result = dict(successes=sum(r['success'] for r in evaluation), total_episodes=len(evaluation),
        success_episodes=[r['episode'] for r in evaluation if r['success']],
        failure_episodes=[r['episode'] for r in evaluation if not r['success']],
        task_description=description, paired_holdout_pairs=pairs, selected_perturbation=selected,
        calibration_episodes=sum(r['phase']=='calibration' for r in outcomes))
    (root/'paired_complete.json').write_text(json.dumps(result, indent=2)+'\n')
    return result

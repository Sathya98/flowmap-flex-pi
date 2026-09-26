"""Run frozen, paired FM/LMD LIBERO evaluations and reject incomplete results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
# Put the normal FM reference and one-step student first in the serialized array.
CASES = (('fm', 4), ('lmd', 1), ('fm', 1), ('lmd', 2), ('fm', 2), ('lmd', 4))


def validate_results(directory, trials, expected_tasks):
    found = {}
    for path in Path(directory).rglob('*_results.json'):
        row = json.loads(path.read_text())
        key = (row['task_suite'], int(row['task_id']))
        if key in found:
            raise ValueError(f'Duplicate task result: {key}')
        success = row['success_episodes']
        failure = row['failure_episodes']
        if (row['total_episodes'] != trials or row['successes'] != len(success)
                or len(success) + len(failure) != trials
                or set(success).intersection(failure)
                or set(success + failure) != set(range(trials))):
            raise ValueError(f'Incomplete/inconsistent episodes in {path}')
        found[key] = row
    if set(found) != set(expected_tasks):
        raise ValueError(f'Incomplete task coverage: {len(found)}/{len(expected_tasks)}')
    return {'tasks': len(found), 'trials': trials * len(found),
            'successes': sum(row['successes'] for row in found.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eval_dir', type=Path)
    parser.add_argument('phase', choices=('smoke', 'sweep'))
    parser.add_argument('--case', type=int, default=0, choices=range(len(CASES)))
    args = parser.parse_args()
    root = args.eval_dir.resolve()
    plan = json.loads((root/'manifest.json').read_text())
    code = root/'code'
    if args.phase == 'sweep' and not (root/'smoke_passed.json').is_file():
        raise RuntimeError('Both model rollout smokes must pass before the sweep')
    results = []
    for variant, nfe in (CASES[:2] if args.phase == 'smoke' else (CASES[args.case],)):
        trials = 1 if args.phase == 'smoke' else plan['trials_per_task']
        out = root/args.phase/f'{variant}_nfe{nfe}'
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite an evaluation: {out}')
        out.mkdir(parents=True)
        config = plan['models'][variant]
        common = [f"ckpt={config['checkpoint']}",
                  f"EVALUATION.dataset_stats_path={config['stats']}",
                  f'EVALUATION.output_dir={out}', f'EVALUATION.num_trials={trials}',
                  f'EVALUATION.num_inference_steps={nfe}', 'eval_config_source=saved',
                  'EVALUATION.dynamic_step_skip=false', f"seed={plan['seed']}"]
        if args.phase == 'smoke':
            command = [sys.executable, 'experiments/libero/eval_libero_single.py', *common,
                       'EVALUATION.task_suite_name=libero_spatial', 'EVALUATION.task_id=0',
                       f"+EVALUATION.camera_intrinsics_path={plan['intrinsics']}"]
            for key in ('infer_present_video','infer_present_dino','infer_present_pointmap',
                        'infer_joint_video','infer_joint_dino','infer_joint_pointmap'):
                command.append(f'+EVALUATION.{key}=true')
            expected = [('libero_spatial', 0)]
            env = os.environ.copy()
        else:
            command = ['bash', 'scripts/eval_flexpi_libero_4suite.sh', *common]
            env = dict(os.environ, CKPT=config['checkpoint'], DATASET_STATS=config['stats'],
                       OUTPUT_DIR=str(out), NUM_TRIALS=str(trials), TASKS_PER_SUITE='10',
                       GPUS='0,1,2,3', CAMERA_INTRINSICS_PATH=plan['intrinsics'])
            for key in ('INFER_PRESENT_VIDEO','INFER_PRESENT_DINO','INFER_PRESENT_POINTMAP',
                        'INFER_JOINT_VIDEO','INFER_JOINT_DINO','INFER_JOINT_POINTMAP'):
                env[key] = 'true'
            expected = [(suite, task) for suite in SUITES for task in range(10)]
        (out/'launch.json').write_text(json.dumps(dict(command=command,
            model=variant, nfe=nfe, job=os.environ.get('SLURM_JOB_ID')), indent=2)+'\n')
        subprocess.run(command, cwd=code, env=env, check=True)
        result = dict(model=variant, nfe=nfe, **validate_results(out, trials, expected))
        (out/'validated.json').write_text(json.dumps(result, indent=2)+'\n')
        results.append(result)
    if args.phase == 'smoke':
        (root/'smoke_passed.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

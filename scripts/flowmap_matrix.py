"""Plan full-joint experiments. Never downloads weights or launches jobs."""
import argparse
import csv
import json
from pathlib import Path
import shlex

DISTILLATION = ('lmd', 'emd', 'pfmm')
SELF_DISTILLATION = ('lsd', 'esd', 'psd_m', 'psd_u')
BENCHMARKS = {
    'libero': dict(task='libero_unified_flex_2cam224_32d_rotvec_1e-4',
                   repo='flex-pi/flexpi-libero', step='010860', epochs=20,
                   evaluations=['libero', 'libero_plus']),
    'robotwin': dict(task='robotwin_unified_flex_3cam_384_1e-4',
                     repo='flex-pi/flexpi-robotwin', step='048060', epochs=6,
                     evaluations=['robotwin']),
}
METRICS = {
    'libero': ['spatial', 'object', 'goal', 'long', 'average'],
    'libero_plus': ['layout', 'viewpoint', 'robot_initial_state', 'language',
                    'lighting', 'texture', 'sensor_noise', 'task_weighted_total'],
    'robotwin': ['clean', 'randomized', 'average'],
}


def build_plan(args):
    plan = []
    teachers, statistics = {}, {}
    for name, spec in BENCHMARKS.items():
        release = Path(args.releases_dir) / spec['repo'].split('/')[-1]
        teachers[name] = getattr(args, name + '_teacher') or str(
            release / 'checkpoints' / 'weights' / ('step_' + spec['step'] + '.pt'))
        statistics[name] = getattr(args, name + '_stats') or str(release / 'dataset_stats.json')
    if args.agibot_checkpoint and Path(args.agibot_checkpoint).resolve() in {
            Path(p).resolve() for p in teachers.values()}:
        raise ValueError('AGIBOT initialization cannot be a task-finetuned teacher')
    names = tuple(BENCHMARKS) if args.benchmark == 'all' else (args.benchmark,)
    for name in names:
        spec, teacher, stats = BENCHMARKS[name], teachers[name], statistics[name]
        baseline = dict(benchmark_train=name, method='flexpi_fm', objective='fm',
                        seed=None, checkpoint=teacher, teacher_checkpoint=None,
                        initial_checkpoint=teacher, dataset_stats=stats,
                        evaluations=spec['evaluations'], nfes=args.nfes,
                        streams=['action', 'video', 'dino', 'pointmap'], mode='full',
                        train_command=None, status='released_checkpoint_evaluation')
        plan.append(baseline)
        families = []
        if args.family in ('all', 'distillation'):
            families.extend(('distillation', obj) for obj in DISTILLATION)
        if args.family in ('all', 'self_distillation'):
            families.extend(('self_distillation', obj) for obj in SELF_DISTILLATION)
        if args.include_fm_control:
            families.append(('matched_fm_control', 'fm'))
        for family, objective in families:
            initial = teacher if family == 'distillation' else args.agibot_checkpoint
            for seed in args.seeds:
                run = Path(args.runs_dir) / f'{name}_{family}_{objective}_fulljoint_full_s{seed}'
                row = dict(baseline, method=family, objective=objective, seed=seed,
                           checkpoint=None, run_dir=str(run), initial_checkpoint=initial,
                           teacher_checkpoint=teacher if family == 'distillation' else None,
                           status='planned' if initial else 'needs_agibot_checkpoint')
                row['loss_weighting'] = 'learned' if family == 'self_distillation' else 'fixed'
                if family in ('distillation', 'self_distillation'):
                    row['evaluation_weights'] = ['raw', 'ema_0.999', 'ema_0.9999']
                    row['evaluation_ema_decay'] = 0.9999
                    row['evaluation_checkpoint_pattern'] = str(run / 'checkpoints/weights/ema_0.9999/step_*.pt')
                if initial:
                    command = ['accelerate', 'launch', '--config_file',
                               'scripts/accelerate_configs/accelerate_zero2_ds.yaml',
                               '--num_processes', str(args.num_processes), 'scripts/train.py',
                               'task=' + spec['task'], 'pretrained_ckpt=' + initial,
                               'pretrained_ckpt_strict_shape=' + str(family == 'distillation').lower(),
                               'resume=null', 'model.flow_map.enabled=' + str(objective != 'fm').lower(),
                               'model.flow_map.mode=full', 'model.flow_map.initialization=pretrained',
                               'model.flow_map.streams=[action,video,dino,pointmap]',
                               'model.flex_joint.enabled=false', 'model.enable_pointmap=true',
                               'model.joint_video=true', 'model.joint_dino=true', 'model.joint_pointmap=true',
                               'model.mot_checkpoint_mixed_attn=true',
                               'data.train.pretrained_norm_stats=' + stats,
                               'data.train.val_set_proportion=0.0',
                               'num_epochs=' + str(args.epochs or spec['epochs']),
                               'seed=' + str(seed), 'output_dir=' + str(run), 'eval_video=true']
                    if objective != 'fm':
                        command.extend(['model.flow_map.objective=' + objective,
                                        'model.flow_map.strip_width=1.0',
                                        'model.flow_map.detach_derivatives=false',
                                        'model.flow_map.teacher_checkpoint=' +
                                        (teacher if family == 'distillation' else 'null')])
                    if family == 'distillation':
                        command.extend(['model.flow_map.distill_ema=true',
                                        'model.flow_map.distill_learned_time_weighting=false'])
                    if objective == 'lmd':
                        command.append('model.flow_map.lmd_teacher_gradient=full')
                        row['lmd_teacher_gradient'] = 'full'
                    if family == 'self_distillation':
                        command.extend(['model.flow_map.self_diagonal_fraction=0.75',
                                        'model.flow_map.learned_time_weighting=true',
                                        'model.flow_map.ema_decays=[0.999,0.9999]',
                                        'model.flow_map.time_sampling=uniform_triangle'])
                    if objective == 'pfmm':
                        command.append('model.flow_map.pfmm_loss_space=endpoint')
                        row['pfmm_loss_space'] = 'endpoint'
                        command.append('model.flow_map.teacher_steps=' + str(args.teacher_steps))
                        row['teacher_steps'] = args.teacher_steps
                    row['train_command'] = command
                plan.append(row)
    return plan


def write_tables(plan, folder):
    """Empty measurements; published scores are not NFE-matched results."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for benchmark in METRICS:
        target = folder / (benchmark + '.csv')
        if target.exists() and any(benchmark in r['evaluations'] for r in plan):
            raise FileExistsError(f'Refusing to overwrite a possible results file: {target}')
    for benchmark, metrics in METRICS.items():
        rows = [r for r in plan if benchmark in r['evaluations']]
        if not rows:
            continue
        fields = ['method', 'objective', 'training_seed', 'nfe', 'checkpoint', 'status',
                  'lmd_teacher_gradient', 'pfmm_loss_space', 'loss_weighting', 'evaluation_weights',
                  *metrics, 'episodes', 'evaluation_seed', 'latency_ms', 'denoising_ms',
                  'peak_memory_gb', 'training_gpu_hours', 'notes']
        with (folder / (benchmark + '.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                for nfe in row['nfes']:
                    writer.writerow(dict(method=row['method'], objective=row['objective'],
                                         training_seed=row['seed'], nfe=nfe,
                                         checkpoint=row['checkpoint'], status='not_run',
                                         lmd_teacher_gradient=row.get('lmd_teacher_gradient', ''),
                                         pfmm_loss_space=row.get('pfmm_loss_space', ''),
                                         loss_weighting=row.get('loss_weighting', 'fixed'),
                                         evaluation_weights=('ema_' + str(row['evaluation_ema_decay'])
                                                            if 'evaluation_ema_decay' in row else 'raw')))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('all', *BENCHMARKS), default='all')
    parser.add_argument('--family', choices=('all', 'distillation', 'self_distillation'), default='all')
    parser.add_argument('--agibot-checkpoint', help='FlexPi AGIBOT weights BEFORE benchmark fine-tuning')
    parser.add_argument('--releases-dir', default='runs/releases')
    parser.add_argument('--runs-dir', default='runs/flowmap_fulljoint')
    for name in BENCHMARKS:
        parser.add_argument('--' + name + '-teacher')
        parser.add_argument('--' + name + '-stats', help='Normalization statistics paired with task checkpoint')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--nfes', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--epochs', type=int, help='Budget override; defaults LIBERO=20, RoboTwin=6')
    parser.add_argument('--teacher-steps', type=int, default=16,
                        help='PFMM teacher Euler steps; provisional default, calibrate target convergence')
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--include-fm-control', action='store_true')
    parser.add_argument('--format', choices=('commands', 'json'), default='commands')
    parser.add_argument('--tables-dir', help='Write empty result CSV templates')
    args = parser.parse_args(argv)
    if args.teacher_steps < 2:
        parser.error('PFMM teacher steps must be at least 2')
    if any(n < 1 for n in args.nfes) or args.num_processes < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error('NFEs, processes and epochs must be positive')
    if len(set(args.nfes)) != len(args.nfes) or len(set(args.seeds)) != len(args.seeds):
        parser.error('NFEs and seeds must be unique')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.tables_dir:
        write_tables(plan, args.tables_dir)
    if args.format == 'json':
        print(json.dumps(plan, indent=2))
    else:
        for row in plan:
            print(f"# {row['benchmark_train']} / {row['method']} / {row['objective']} / seed={row['seed']}")
            if row['train_command']:
                print(shlex.join(row['train_command']))
            else:
                print('# ' + row['status'] + ': ' + str(row['checkpoint'] or '--agibot-checkpoint required'))


if __name__ == '__main__':
    main()

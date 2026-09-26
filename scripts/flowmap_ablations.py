"""Print reproducible training commands; never launches jobs or spends GPU time."""
import argparse
import itertools
import shlex


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('action','rgb','all'),default='rgb')
    parser.add_argument('--objectives',nargs='+',choices=('lmd','emd','pfmm','lsd','esd','psd_m','psd_u'),
                        default=['lmd','emd','pfmm','lsd','esd','psd_m','psd_u'])
    parser.add_argument('--modes',nargs='+',choices=('full','lora','adapter','heads'),default=['lora'])
    parser.add_argument('--initialization',choices=('wam','backbone','random'),default='wam')
    parser.add_argument('--teacher',help='Released FM weights (required for distillation)')
    parser.add_argument('--student',help='Warm-start WAM checkpoint; defaults to --teacher for wam initialization')
    parser.add_argument('--task',default='libero_unified_flex_2cam224_32d_rotvec_1e-4')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if any(o in ('lmd','emd','pfmm') for o in args.objectives) and not args.teacher:
        parser.error('--teacher is required for distillation objectives')
    student=args.student or args.teacher
    if args.initialization=='wam' and not student:
        parser.error('--student or --teacher is required for wam initialization')
    if args.initialization=='random' and args.modes!=['full']:
        parser.error('random initialization requires --modes full')
    subsets=[('action',)]
    if args.stage=='rgb':subsets.append(('action','video'))
    if args.stage=='all':
        subsets=[('action',)+extra for n in range(4) for extra in itertools.combinations(('video','dino','pointmap'),n)]
    for streams,objective,mode in itertools.product(subsets,args.objectives,args.modes):
        run='flowmap_'+objective+'_'+mode+'_'+'-'.join(streams)+'_'+args.initialization+'_s'+str(args.seed)
        command=['accelerate','launch','--config_file','scripts/accelerate_configs/accelerate_zero2_ds.yaml',
                 'scripts/train.py','task='+args.task,'model.flow_map.enabled=true',
                 'model.flow_map.objective='+objective,'model.flow_map.mode='+mode,
                 'model.flow_map.streams=['+','.join(streams)+']',
                 'model.flow_map.initialization='+('random' if args.initialization=='random' else 'pretrained'),
                 'model.flex_joint.enabled=false','seed='+str(args.seed),
                 'output_dir=./runs/train/'+run,'eval_video='+str('video' in streams).lower()]
        if args.initialization=='wam':command.append('pretrained_ckpt='+student)
        if objective in ('lmd','emd','pfmm'):command.append('model.flow_map.teacher_checkpoint='+args.teacher)
        print(shlex.join(command))


if __name__=='__main__':main()

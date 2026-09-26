"""Freeze a reviewable full LMD run; no Slurm submission or W&B network writes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--config-name',default='flowmap_libero_lmd_full')
    parser.add_argument('--resume-from-run',type=Path,
                        help='Resume the latest validated full state of a stopped parent run')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    parent=args.resume_from_run.resolve() if args.resume_from_run else None
    if parent is not None and not (parent/'manifest.json').is_file():
        parser.error('--resume-from-run must identify a prepared Flow Map run')
    run=(args.run_dir or root/'runs/flowmap_fulljoint'/f'libero_lmd_fullgrad_s42_{stamp}').resolve()
    run.mkdir(parents=True,exist_ok=False)
    code=run/'code';code.mkdir()
    # flowmap_core is imported by FlexPi (module aliases): freeze it too, or the run
    # would import the live editable install. The launcher puts code/flowmap_core/src first.
    for directory in ('src','configs','scripts','experiments','flowmap_core'):
        shutil.copytree(root/directory,code/directory,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
    for directory in ('checkpoints','data','third_party'):
        (code/directory).symlink_to(root/directory,target_is_directory=True)
    for file in ('pyproject.toml','README.md'):
        shutil.copy2(root/file,code/file)
    with initialize_config_dir(config_dir=str(code/'configs'),version_base=None):
        cfg=compose(config_name=args.config_name,overrides=['output_dir='+str(run)])
    OmegaConf.save(OmegaConf.to_container(cfg,resolve=True),run/'prepared_config.yaml')
    identity=dict(id=uuid.uuid4().hex[:8],entity=cfg.wandb.workspace,project=cfg.wandb.project,
                  mode=cfg.wandb.mode, url=None)
    if cfg.wandb.mode == 'online':
        identity['url']=f"https://wandb.ai/{identity['entity']}/{identity['project']}/runs/{identity['id']}"
    (run/'wandb_run.json').write_text(json.dumps(identity,indent=2)+'\n')
    hashes={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src','configs','scripts','experiments','flowmap_core') for p in (code/folder).rglob('*') if p.is_file()}
    manifest=dict(created_utc=stamp,source_root=str(root),run_dir=str(run),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        uncommitted_source_included=True,source_sha256=hashes,
        config_name=args.config_name,resume_from_run=str(parent) if parent else None,
        max_steps=int(cfg.max_steps),effective_batch=int(cfg.batch_size)*int(cfg.gradient_accumulation_steps)*4,
        initial_hardware_steps=0 if parent else 2,wandb=identity,
        cancelled_jobs=[26861582,26861592,26862643,26862671,26868229])
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(run_dir=str(run),wandb=identity),indent=2))


if __name__=='__main__':main()

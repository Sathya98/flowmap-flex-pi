#!/usr/bin/env python
"""Make run directories resumable after moving the repo: rewrite the old location's
absolute paths in their text records.

Runs record absolute save-time paths: segment_status.json (state/weights), preview and
update metrics (video paths), config.yaml / prepared_config.yaml (output_dir, resume),
manifest.json, launches/*.json, and the frozen code/ snapshot's sbatch files (#SBATCH
--chdir, env scripts), which continuation jobs execute. Checkpoints hold no paths.
Run after copying, from the new repo:

    python scripts/relocate_runs.py runs/flowmap_fulljoint/*/            # dry run: lists changes
    python scripts/relocate_runs.py runs/flowmap_fulljoint/*/ --apply

--old is the previous parent of flowmap-flex-pi (default: the Snellius scratch project,
both spellings); --new defaults to this repo's parent. The repo directory name must stay
the same. Logs, binary files and checkpoints/, wandb*/ are never touched.
"""
import argparse
from pathlib import Path

TEXT_SUFFIXES = {'.json', '.jsonl', '.yaml', '.yml', '.txt', '.sbatch', '.sh', '.py', '.md', '.csv'}   # not logs: history
SKIP_DIRS = {'checkpoints', 'wandb', 'wandb_data', 'wandb_cache', '__pycache__', '.git'}
DEFAULT_OLD = ['/gpfs/scratch1/shared/faster-wams', '/scratch-shared/faster-wams']


def text_files(run):
    for path in run.rglob('*'):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and not SKIP_DIRS & set(path.relative_to(run).parts):
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--old', action='append', help='old parent dir (repeatable)')
    parser.add_argument('--new', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--apply', action='store_true', help='write the changes (default: dry run)')
    args = parser.parse_args()
    new = args.new.rstrip('/')
    olds = [o.rstrip('/') for o in (args.old or DEFAULT_OLD) if o.rstrip('/') != new]
    total = 0
    for run in args.runs:
        run = run.resolve()
        if not (run / 'manifest.json').is_file():
            print(f'skip {run}: no manifest.json (not a prepared run)')
            continue
        changed = 0
        for path in text_files(run):
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            count = sum(text.count(o + '/') for o in olds)
            if not count:
                continue
            for o in olds:
                text = text.replace(o + '/', new + '/')
            if args.apply:
                path.write_text(text)
            changed += 1
            print(f'  {count:5d}  {path.relative_to(run)}')
        print(f'{run.name}: {changed} files {"rewritten" if args.apply else "to rewrite"}')
        total += changed
    print(f'{"rewrote" if args.apply else "dry run:"} {total} files, {" | ".join(olds)} -> {new}')


if __name__ == '__main__':
    main()

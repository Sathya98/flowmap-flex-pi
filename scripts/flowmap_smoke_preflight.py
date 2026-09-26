"""CPU checks for LIBERO pilots initialized from the task-finetuned release."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', default='flowmap_libero_pfmm_smoke')
    parser.add_argument('overrides', nargs='*', help='The same Hydra overrides passed to training')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    resolve = lambda p: root / str(p)
    weights = resolve(cfg.pretrained_ckpt)
    required = [weights, resolve(cfg.data.train.pretrained_norm_stats),
                resolve(cfg.model.action_dit_pretrained_path)]
    saved_path = weights.parents[2] / 'config.yaml'
    required.append(saved_path)
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing pilot prerequisite: {path}')
    if weights.stat().st_size < 1_000_000_000:
        raise ValueError('Teacher file is unexpectedly small; check for an LFS pointer/incomplete download')
    saved = OmegaConf.load(saved_path)
    current = OmegaConf.to_container(cfg.model, resolve=True)
    old = OmegaConf.to_container(saved.model, resolve=True)
    # Compare the actual release architecture, not just the task-config name.
    for key in ('video_dit_config', 'action_dit_config', 'proprio_dim', 'hbridge',
                'composite_layout', 'composite_layout_slot_key_map', 'dino_dim',
                'dino_pixel_unshuffle', 'dino_temporal_stride', 'dino_pred_x0',
                'pointmap_norm_bounds'):
        if current[key] != old[key]:
            raise ValueError(f'Teacher/student configuration mismatch: {key}')
    count = 0
    # Read the exact dataset prompt without importing its ML/video dependencies.
    source = ast.parse((root / 'src/flexpi/datasets/lerobot/robot_video_dataset.py').read_text())
    template = next(ast.literal_eval(node.value) for node in source.body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'DEFAULT_PROMPT'
                        for target in node.targets))
    for directory in cfg.data.train.dataset_dirs:
        dataset = resolve(directory)
        for name in ('info.json', 'tasks.jsonl', 'episodes.jsonl', 'camera_intrinsics.json'):
            if not (dataset / 'meta' / name).is_file():
                raise FileNotFoundError(dataset / 'meta' / name)
        if not (dataset / 'data').is_dir() or not (dataset / 'videos').is_dir():
            raise FileNotFoundError(f'Missing data/videos directory: {dataset}')
        for line in (dataset / 'meta/tasks.jsonl').read_text().splitlines():
            prompt = template.format(task=json.loads(line)['task'])
            digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            cache = resolve(cfg.data.train.text_embedding_cache_dir) / (
                f'{digest}.t5_len{cfg.data.train.context_len}.wan22ti2v5b.pt')
            if not cache.is_file():
                raise FileNotFoundError(f'Missing text cache for {prompt!r}: {cache}')
            count += 1
    assert cfg.model.flow_map.enabled
    assert cfg.model.flow_map.objective in ('pfmm', 'lmd')
    if cfg.model.flow_map.objective == 'lmd':
        assert cfg.model.flow_map.teacher_checkpoint == cfg.pretrained_ckpt
        assert cfg.model.flow_map.dt_method == 'ad'
        assert not cfg.model.flow_map.detach_derivatives
    assert set(cfg.model.flow_map.streams) == {'action', 'video', 'dino', 'pointmap'}
    assert cfg.model.flow_map.mode == 'full' and not cfg.model.flex_joint.enabled
    print(f'Preflight passed: released initialization, matching architecture, {count} task text caches, four data suites.')
    print(f'Objective={cfg.model.flow_map.objective}, time_sampling={cfg.model.flow_map.time_sampling}, strip_width={cfg.model.flow_map.strip_width}')
    print('This does not validate every video, checkpoint tensor, CUDA/DeepSpeed execution, or GPU memory fit.')


if __name__ == '__main__':
    main()

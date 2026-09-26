"""Process-local W&B configuration that never falls back to a shared login."""
import os
from pathlib import Path
import stat


def isolate_wandb(output_dir, mode, entity=None):
    if mode not in ('offline', 'online'):
        raise ValueError('Isolated tracking supports offline or online mode')
    key = None
    if mode == 'online':
        key_path = os.environ.get('FLOWMAP_WANDB_API_KEY_FILE')
        if not entity or not key_path:
            raise ValueError('Online Flow Map logging needs your explicit workspace and '
                             'FLOWMAP_WANDB_API_KEY_FILE; shared login fallback is disabled')
        path = Path(key_path)
        if path.stat().st_uid != os.getuid() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError('The Flow Map key file must be owned by this user and mode 600')
        key = path.read_text().strip()
        if not key:
            raise ValueError('The Flow Map key file is empty')
    # Change only this process and its children, never ~/.netrc or shell profiles.
    for name in list(os.environ):
        if name.startswith('WANDB_'):
            del os.environ[name]
    private = Path(output_dir) / 'tracking'
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    empty_netrc = private / 'empty.netrc'
    empty_netrc.touch(mode=0o600, exist_ok=True)
    empty_netrc.write_text('')
    os.environ['NETRC'] = str(empty_netrc)
    for variable, subdir in [('WANDB_CONFIG_DIR', 'config'), ('WANDB_CACHE_DIR', 'cache'),
                             ('WANDB_DATA_DIR', 'data'), ('WANDB_ARTIFACT_DIR', 'artifacts')]:
        directory = private / subdir
        directory.mkdir(exist_ok=True, mode=0o700)
        os.environ[variable] = str(directory)
    os.environ['WANDB_DIR'] = str(Path(output_dir))
    os.environ['WANDB_MODE'] = mode
    os.environ['WANDB_ERROR_REPORTING'] = 'false'
    if key is not None:
        os.environ['WANDB_API_KEY'] = key
        os.environ['WANDB_ENTITY'] = entity

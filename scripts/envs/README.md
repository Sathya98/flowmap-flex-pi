# Environments

Exact, rebuildable copies of the three uv venvs (Python 3.10.20, torch 2.7.1+cu128,
DeepSpeed 0.18.9). They are built next to the repo, in its parent directory.

| Target | Venv | Activate | Adds |
|---|---|---|---|
| `train` | `fm_env` | `source ../env_flexpi.sh` | training, inference, flow-map dev |
| `libero` | `fm_env_libero` | `source ../env_flexpi_libero.sh` | mujoco 3.3.2, robosuite 1.4.0 (`--no-deps`), opencv-headless 4.8.1.78, LIBERO on `PYTHONPATH` |
| `robotwin` | `fm_env_robotwin` | `source ../env_flexpi_robotwin.sh` | sapien 3.0.0b1 + mplib 0.2.1 (patched as in RoboTwin's `_install.sh`), cuRobo `0a50de1` (compiled), Vulkan ICD |

```bash
bash scripts/envs/install_envs.sh                  # all three, into the repo's parent dir
bash scripts/envs/install_envs.sh libero           # one
PREFIX=/projects/x bash scripts/envs/install_envs.sh
bash scripts/envs/freeze_envs.sh                   # refresh locks/ after changing an env
```

- `locks/<venv>.txt` is the complete package set (`uv pip freeze`, minus the editable
  installs). It is installed with `--no-deps`, so nothing is re-resolved. That is what
  keeps robosuite's `opencv-python` / unpinned `mujoco` metadata from leaking in.
- An existing venv or activation script is never touched; that target is skipped.
- Keep `UV_CACHE_DIR` (default `$PREFIX/.uv_cache`) on the venvs' filesystem: uv
  hardlinks packages from it, so the venvs share one copy of torch, CUDA libs & co.
- Not covered: the RoboTwin assets (16 GB, `third_party/RoboTwin/script/_download_assets.sh`;
  the installer prints the command), weights (`checkpoints/`), datasets and caches.
- Off Snellius, provide CUDA 12.8 (`CUDA_HOME`, nvcc for cuRobo) and FFmpeg 4–7 yourself;
  the scripts load the EESSI modules only when Lmod is present.

## One eval env for both simulators?

Possible but not done. Against `fm_env`, LIBERO adds 32 packages and RoboTwin adds 84.
Where they share a package they agree on its version, except three patch-level ones (fonttools, pyparsing,
setuptools). The one real conflict is `cv2`: sapien requires `opencv-python`, LIBERO
pins `opencv-python-headless==4.8.1.78`, and both install into the same `cv2/`
directory (docs/INSTALL.md §4).

LIBERO's eval path (LIBERO, the flexpi policy, and robosuite's mujoco offscreen renderer)
never imports `cv2`, so a merged env would keep `opencv-python 4.11.0.86` alone. Before
switching, it must reproduce a LIBERO eval. The disk saving is small, because the venvs
already share their files through uv-cache hardlinks.

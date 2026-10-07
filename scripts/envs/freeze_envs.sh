#!/usr/bin/env bash
# Refresh scripts/envs/locks/<venv>.txt from the live venvs (exact versions of every package).
#
#   bash scripts/envs/freeze_envs.sh                  # fm_env fm_env_libero fm_env_robotwin
#   PREFIX=/elsewhere bash scripts/envs/freeze_envs.sh fm_env
#
# Editable installs (flexpi, flowmap_core, cuRobo) are left out; install_envs.sh adds them.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PREFIX=${PREFIX:-$(dirname "$REPO")}
LOCKS=$REPO/scripts/envs/locks
envs=("$@")
[ ${#envs[@]} -eq 0 ] && envs=(fm_env fm_env_libero fm_env_robotwin)

mkdir -p "$LOCKS"
for e in "${envs[@]}"; do
  [ -x "$PREFIX/$e/bin/python" ] || { echo "no venv at $PREFIX/$e" >&2; exit 1; }
  ( uv pip freeze --python "$PREFIX/$e/bin/python" \
      | grep -v -E '^-e |^(flexpi|flowmap-core|nvidia-curobo)[=@ ]' > "$LOCKS/$e.txt" ) &
done
wait
for e in "${envs[@]}"; do echo "$e: $(wc -l < "$LOCKS/$e.txt") packages -> locks/$e.txt"; done

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

if [[ -f "${REPO_ROOT}/.venv-codex/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv-codex/bin/activate"
fi

if [[ "${OMLX_SKIP_UPGRADE:-0}" != "1" ]]; then
  echo "[run-omlx] syncing deps (set OMLX_SKIP_UPGRADE=1 to skip)..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e "${REPO_ROOT}" --upgrade --quiet
  else
    pip install -e "${REPO_ROOT}" --upgrade --quiet
  fi
fi

MODEL_DIR="${OMLX_MODEL_DIR:-$HOME/.omlx/models}"
BASE_PATH="${OMLX_BASE_PATH:-$HOME/.omlx}"
PORT="${OMLX_PORT:-5556}"
API_KEY="${OMLX_API_KEY:-4991}"

# Persist desired values into ~/.omlx/settings.json so the managed
# server (Swift Mac.app or brew services) picks them up on start.
SETTINGS_PATH="${BASE_PATH}/settings.json"
mkdir -p "${BASE_PATH}"
MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" API_KEY="${API_KEY}" SETTINGS_PATH="${SETTINGS_PATH}" \
python - <<'PY'
import json, os
from pathlib import Path

path = Path(os.environ["SETTINGS_PATH"])
data = {}
if path.exists():
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        data = {}

data.setdefault("version", "1.0")
server = data.setdefault("server", {})
server["port"] = int(os.environ["PORT"])
model = data.setdefault("model", {})
model_dirs = [d.strip() for d in os.environ["MODEL_DIR"].split(",") if d.strip()]
model["model_dirs"] = model_dirs
model["model_dir"] = model_dirs[0] if model_dirs else None
auth = data.setdefault("auth", {})
auth["api_key"] = os.environ["API_KEY"]

path.write_text(json.dumps(data, indent=2))
print(f"[run-omlx] persisted port={server['port']} model_dir={model['model_dir']} to {path}")
PY

cd "${REPO_ROOT}"

# Managed lifecycle: Swift Mac.app via UNIX socket, or brew services.
# Falls back to foreground `omlx serve` if neither is installed.
if [[ $# -eq 0 ]] && python -m omlx.cli start; then
  exit 0
fi

echo "[run-omlx] managed start unavailable, falling back to foreground serve..."
exec python -m omlx.cli serve \
  --model-dir "${MODEL_DIR}" \
  --base-path "${BASE_PATH}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  "$@"

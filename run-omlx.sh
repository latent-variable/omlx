#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

if [[ -f "${REPO_ROOT}/.venv-codex/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv-codex/bin/activate"
fi

MODEL_DIR="${OMLX_MODEL_DIR:-$HOME/.omlx/models}"
BASE_PATH="${OMLX_BASE_PATH:-$HOME/.omlx}"
PORT="${OMLX_PORT:-5556}"
API_KEY="${OMLX_API_KEY:-4991}"

cd "${REPO_ROOT}"

exec python -m omlx.cli serve \
  --model-dir "${MODEL_DIR}" \
  --base-path "${BASE_PATH}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  "$@"

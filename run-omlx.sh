#!/bin/sh
# Run oMLX from this source checkout (.venv-codex dev build).
#
# Chunk reuse toggle lives in ~/.omlx/settings.json:
#   "cache": { "chunk_reuse": true, "chunk_reuse_mode": "edge" }
# Env vars override the file when set explicitly:
#   OMLX_CHUNK_REUSE=false ./run-omlx.sh   # force OFF for an A/B run
#   OMLX_CHUNK_REUSE_MODE=...              # reuse | edge (default) | devblock
#
# Uses your ~/.omlx settings (port 5599, model dir, api key) unchanged.
cd "$(dirname "$0")" || exit 1
exec .venv-codex/bin/omlx serve "$@"

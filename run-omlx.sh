#!/bin/sh
# Run oMLX from this source checkout (.venv-codex dev build).
#
#   ./run-omlx.sh                      # normal server (chunk reuse OFF)
#   OMLX_CHUNK_REUSE=true ./run-omlx.sh   # with experimental chunk reuse
#   OMLX_CHUNK_REUSE_MODE=edge          # reuse | edge (default) | devblock
#
# Uses your ~/.omlx settings (port 5599, model dir, api key) unchanged.
cd "$(dirname "$0")" || exit 1
exec .venv-codex/bin/omlx serve "$@"

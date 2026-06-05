#!/usr/bin/env bash
# Launch the previously-built Swift oMLX.app without rebuilding.
# Run ./build-omlx-app.sh first to produce the staged bundle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGED_APP="${SCRIPT_DIR}/apps/omlx-mac/build/Stage/oMLX.app"

if [[ ! -d "${STAGED_APP}" ]]; then
  echo "error: ${STAGED_APP} not found." >&2
  echo "       Run ./build-omlx-app.sh first." >&2
  exit 1
fi

echo "[launch-omlx-app] opening ${STAGED_APP}"
open -gj "${STAGED_APP}"

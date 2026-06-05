#!/usr/bin/env bash
# Build the Swift oMLX.app from source and launch it.
#
# First run:
#   ./build-omlx-app.sh              # full release build (slow: builds
#                                    # venvstacks Python donor + Swift app)
#
# Subsequent runs (Swift-only changes):
#   ./build-omlx-app.sh swift        # reuse existing Python donor; rebuild
#                                    # Swift bundle only (fast)
#   ./build-omlx-app.sh swift-fast   # like swift, but skip embedded
#                                    # native signing (fastest)
#
# Pass additional flags through to apps/omlx-mac/Scripts/build.sh, e.g.:
#   ./build-omlx-app.sh debug
#   ./build-omlx-app.sh --rebuild-donor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
BUILD_SCRIPT="${REPO_ROOT}/apps/omlx-mac/Scripts/build.sh"
STAGED_APP="${REPO_ROOT}/apps/omlx-mac/build/Stage/oMLX.app"

if [[ ! -x "${BUILD_SCRIPT}" ]]; then
  echo "error: ${BUILD_SCRIPT} not found or not executable" >&2
  exit 1
fi

echo "[build-omlx-app] building (args: $*)..."
"${BUILD_SCRIPT}" "$@"

if [[ ! -d "${STAGED_APP}" ]]; then
  echo "error: build finished but ${STAGED_APP} does not exist" >&2
  exit 1
fi

echo "[build-omlx-app] launching ${STAGED_APP}"
open -gj "${STAGED_APP}"

echo "[build-omlx-app] done. Server lifecycle controlled from the app's"
echo "                 menubar / Server screen. UNIX socket:"
echo "                 ~/Library/Application Support/oMLX/control.sock"

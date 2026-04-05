#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCODE_SESSION_ID="${OPENCODE_SESSION_ID:-}"
OPENCODE_DB="${OPENCODE_DB:-$HOME/.local/share/opencode/opencode.db}"
WORKLOAD_ID="${WORKLOAD_ID:-ad_hoc}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/benchmark_results/opencode_latest}"
DASHBOARD_TITLE="${DASHBOARD_TITLE:-oMLX Harness Cache Dashboard: OpenCode}"
OPEN_DASHBOARD="${OPEN_DASHBOARD:-0}"

if [[ -z "$OPENCODE_SESSION_ID" ]]; then
  echo "Set OPENCODE_SESSION_ID to the OpenCode session you want to analyze." >&2
  exit 1
fi

if [[ ! -f "$OPENCODE_DB" ]]; then
  echo "OpenCode DB not found: $OPENCODE_DB" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
.venv/bin/python scripts/run_harness_benchmark.py \
  --opencode-session-id "$OPENCODE_SESSION_ID" \
  --opencode-db "$OPENCODE_DB" \
  --workload-id "$WORKLOAD_ID" \
  --output-dir "$OUTPUT_DIR" \
  --dashboard-title "$DASHBOARD_TITLE"

echo
echo "Dashboard: $OUTPUT_DIR/dashboard.html"

if [[ "$OPEN_DASHBOARD" == "1" ]]; then
  open "$OUTPUT_DIR/dashboard.html"
fi

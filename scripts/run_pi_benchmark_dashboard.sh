#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_PATH="${PI_TRACE_PATH:-$HOME/.pi/agent/pi-prompt-trace.jsonl}"
WORKLOAD_ID="${WORKLOAD_ID:-ad_hoc}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/benchmark_results/pi_latest}"
DASHBOARD_TITLE="${DASHBOARD_TITLE:-oMLX Harness Cache Dashboard: Pi}"
OPEN_DASHBOARD="${OPEN_DASHBOARD:-0}"

if [[ ! -f "$TRACE_PATH" ]]; then
  echo "Pi trace not found: $TRACE_PATH" >&2
  echo "Set PI_TRACE_PATH or generate trace data first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
.venv/bin/python scripts/run_harness_benchmark.py \
  --pi-trace "$TRACE_PATH" \
  --workload-id "$WORKLOAD_ID" \
  --block-size "$BLOCK_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --dashboard-title "$DASHBOARD_TITLE"

echo
echo "Dashboard: $OUTPUT_DIR/dashboard.html"

if [[ "$OPEN_DASHBOARD" == "1" ]]; then
  open "$OUTPUT_DIR/dashboard.html"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_PATH="${PI_TRACE_PATH:-$HOME/.pi/agent/pi-prompt-trace.jsonl}"
WORKLOAD_ID="${WORKLOAD_ID:-ad_hoc}"
BLOCK_SIZES="${BLOCK_SIZES:-512 1024 2048 4096}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/benchmark_results/pi_block_sweep}"
DASHBOARD_TITLE="${DASHBOARD_TITLE:-oMLX Harness Cache Dashboard: Pi Block Sweep}"
OPEN_DASHBOARD="${OPEN_DASHBOARD:-0}"

if [[ ! -f "$TRACE_PATH" ]]; then
  echo "Pi trace not found: $TRACE_PATH" >&2
  echo "Set PI_TRACE_PATH or generate trace data first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

cd "$ROOT_DIR"

compare_args=()
primary_block=""
for block_size in $BLOCK_SIZES; do
  run_dir="$OUTPUT_ROOT/block_${block_size}"
  mkdir -p "$run_dir"
  echo "Running Pi benchmark for block size $block_size"
  .venv/bin/python scripts/run_harness_benchmark.py \
    --pi-trace "$TRACE_PATH" \
    --workload-id "$WORKLOAD_ID" \
    --block-size "$block_size" \
    --output-dir "$run_dir" \
    --dashboard-title "$DASHBOARD_TITLE"
  if [[ -z "$primary_block" ]]; then
    primary_block="$block_size"
  else
    compare_args+=(--compare-report-json "$run_dir/report.json")
  fi
done

combined_dir="$OUTPUT_ROOT/combined"
mkdir -p "$combined_dir"

echo "Building combined comparison dashboard"
.venv/bin/python scripts/run_harness_benchmark.py \
  --pi-trace "$TRACE_PATH" \
  --workload-id "$WORKLOAD_ID" \
  --block-size "$primary_block" \
  "${compare_args[@]}" \
  --output-dir "$combined_dir" \
  --dashboard-title "$DASHBOARD_TITLE"

echo
echo "Combined dashboard: $combined_dir/dashboard.html"

if [[ "$OPEN_DASHBOARD" == "1" ]]; then
  open "$combined_dir/dashboard.html"
fi

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run additive agent-harness benchmark analysis for oMLX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omlx.benchmarking.runner import analyze_pi_trace_to_json, analyze_pi_trace_to_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze harness cache behavior for oMLX")
    parser.add_argument("--pi-trace", required=True, help="Path to pi-prompt-trace.jsonl")
    parser.add_argument("--workload-id", default="ad_hoc")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "report.md"
    json_path = output_dir / "report.json"

    markdown = analyze_pi_trace_to_markdown(
        args.pi_trace,
        workload_id=args.workload_id,
        block_size=args.block_size,
        output_path=markdown_path,
    )
    payload = analyze_pi_trace_to_json(
        args.pi_trace,
        workload_id=args.workload_id,
        block_size=args.block_size,
        output_path=json_path,
    )

    print(markdown)
    print(json.dumps({"report_markdown": str(markdown_path), "report_json": str(json_path), "summary": payload["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

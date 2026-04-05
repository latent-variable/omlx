#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run additive agent-harness benchmark analysis for oMLX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omlx.benchmarking.runner import (
    analyze_opencode_session,
    analyze_pi_session,
    analyze_pi_trace,
    load_benchmark_result,
    render_result_to_json,
    render_result_to_markdown,
    render_results_dashboard,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze harness cache behavior for oMLX")
    parser.add_argument("--pi-trace", help="Path to pi-prompt-trace.jsonl")
    parser.add_argument("--pi-session-jsonl", help="Path to Pi session .jsonl")
    parser.add_argument("--opencode-session-id", help="OpenCode session id to analyze")
    parser.add_argument(
        "--opencode-db",
        default=str(Path.home() / ".local/share/opencode/opencode.db"),
        help="Path to the OpenCode SQLite database",
    )
    parser.add_argument("--workload-id", default="ad_hoc")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument(
        "--compare-report-json",
        action="append",
        default=[],
        help="Additional report.json files to include in the dashboard comparison",
    )
    parser.add_argument(
        "--dashboard-title",
        default="oMLX Harness Benchmark Dashboard",
        help="Title to render at the top of the generated HTML dashboard",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "report.md"
    json_path = output_dir / "report.json"
    html_path = output_dir / "dashboard.html"

    provided_inputs = sum(bool(value) for value in (args.pi_trace, args.pi_session_jsonl, args.opencode_session_id))
    if provided_inputs != 1:
        parser.error("pass exactly one of --pi-trace, --pi-session-jsonl, or --opencode-session-id")

    if args.pi_trace:
        result = analyze_pi_trace(
            args.pi_trace,
            workload_id=args.workload_id,
            block_size=args.block_size,
        )
    elif args.pi_session_jsonl:
        result = analyze_pi_session(
            args.pi_session_jsonl,
            workload_id=args.workload_id,
            block_size=args.block_size,
        )
    else:
        result = analyze_opencode_session(
            args.opencode_session_id,
            db_path=args.opencode_db,
            workload_id=args.workload_id,
        )
    markdown = render_result_to_markdown(result, output_path=markdown_path)
    payload = render_result_to_json(result, output_path=json_path)
    comparison_results = [load_benchmark_result(path) for path in args.compare_report_json]
    render_results_dashboard(
        [result, *comparison_results],
        title=args.dashboard_title,
        output_path=html_path,
    )

    print(markdown)
    print(
        json.dumps(
            {
                "report_markdown": str(markdown_path),
                "report_json": str(json_path),
                "dashboard_html": str(html_path),
                "summary": payload["summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

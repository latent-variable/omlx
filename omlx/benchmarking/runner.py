# SPDX-License-Identifier: Apache-2.0
"""Entry points for harness cache benchmark analysis."""

from __future__ import annotations

import json
from pathlib import Path

from .analysis import load_pi_trace, summarize_opencode_session, summarize_pi_session, summarize_trace
from .reports import render_html_dashboard, render_markdown
from .schema import BenchmarkResult, BenchmarkRunConfig, BenchmarkTurn


def analyze_pi_trace(
    pi_trace_path: str | Path,
    *,
    workload_id: str,
    block_size: int,
) -> BenchmarkResult:
    trace = load_pi_trace(pi_trace_path)
    return summarize_trace(
        trace,
        harness="pi",
        workload_id=workload_id,
        block_size=block_size,
    )


def analyze_opencode_session(
    session_id: str,
    *,
    db_path: str | Path,
    workload_id: str,
) -> BenchmarkResult:
    return summarize_opencode_session(session_id, db_path=db_path, workload_id=workload_id)


def analyze_pi_session(
    pi_session_path: str | Path,
    *,
    workload_id: str,
    block_size: int,
) -> BenchmarkResult:
    return summarize_pi_session(pi_session_path, workload_id=workload_id, block_size=block_size)


def benchmark_result_from_dict(payload: dict) -> BenchmarkResult:
    config = BenchmarkRunConfig(**payload["config"])
    turns = [BenchmarkTurn(**turn) for turn in payload["turns"]]
    return BenchmarkResult(config=config, turns=turns, summary=dict(payload["summary"]))


def load_benchmark_result(path: str | Path) -> BenchmarkResult:
    payload = json.loads(Path(path).read_text())
    return benchmark_result_from_dict(payload)


def analyze_pi_trace_to_markdown(
    pi_trace_path: str | Path,
    *,
    workload_id: str,
    block_size: int,
    output_path: str | Path | None = None,
) -> str:
    result = analyze_pi_trace(pi_trace_path, workload_id=workload_id, block_size=block_size)
    return render_result_to_markdown(result, output_path=output_path)


def render_result_to_markdown(
    result: BenchmarkResult,
    *,
    output_path: str | Path | None = None,
) -> str:
    markdown = render_markdown(result)
    if output_path is not None:
        Path(output_path).write_text(markdown)
    return markdown


def analyze_pi_trace_to_json(
    pi_trace_path: str | Path,
    *,
    workload_id: str,
    block_size: int,
    output_path: str | Path | None = None,
) -> dict:
    result = analyze_pi_trace(pi_trace_path, workload_id=workload_id, block_size=block_size)
    return render_result_to_json(result, output_path=output_path)


def render_result_to_json(
    result: BenchmarkResult,
    *,
    output_path: str | Path | None = None,
) -> dict:
    payload = result.to_dict()
    if output_path is not None:
        Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def render_results_dashboard(
    results: list[BenchmarkResult],
    *,
    title: str = "oMLX Harness Benchmark Dashboard",
    output_path: str | Path | None = None,
) -> str:
    dashboard = render_html_dashboard(results, title=title)
    if output_path is not None:
        Path(output_path).write_text(dashboard)
    return dashboard

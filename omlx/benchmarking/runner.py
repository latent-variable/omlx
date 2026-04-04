# SPDX-License-Identifier: Apache-2.0
"""Entry points for harness cache benchmark analysis."""

from __future__ import annotations

import json
from pathlib import Path

from .analysis import load_pi_trace, summarize_trace
from .reports import render_markdown


def analyze_pi_trace_to_markdown(
    pi_trace_path: str | Path,
    *,
    workload_id: str,
    block_size: int,
    output_path: str | Path | None = None,
) -> str:
    trace = load_pi_trace(pi_trace_path)
    result = summarize_trace(
        trace,
        harness="pi",
        workload_id=workload_id,
        block_size=block_size,
    )
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
    trace = load_pi_trace(pi_trace_path)
    result = summarize_trace(
        trace,
        harness="pi",
        workload_id=workload_id,
        block_size=block_size,
    )
    payload = result.to_dict()
    if output_path is not None:
        Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload

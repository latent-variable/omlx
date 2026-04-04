# SPDX-License-Identifier: Apache-2.0
"""Report formatting helpers for harness cache benchmarks."""

from __future__ import annotations

from .schema import BenchmarkResult


def render_markdown(result: BenchmarkResult) -> str:
    lines = [
        f"# Harness Cache Benchmark: {result.config.harness}",
        "",
        f"- Model: `{result.config.model_id}`",
        f"- Workload: `{result.config.workload_id}`",
        f"- Block size: `{result.config.block_size}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in result.summary.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Turns",
            "",
            "| Turn | Prompt chars | Prefix chars | Block-aligned prefix | Reprocessed est. | Msgs | After abort |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for turn in result.turns:
        lines.append(
            f"| {turn.turn_index} | {turn.prompt_chars_total} | {turn.common_prefix_chars} | "
            f"{turn.block_aligned_prefix_chars} | {turn.reprocessed_chars_estimate} | "
            f"{turn.effective_message_count} | {turn.likely_replay_after_abort} |"
        )
    return "\n".join(lines) + "\n"

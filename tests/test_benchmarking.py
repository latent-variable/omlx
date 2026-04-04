# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from omlx.benchmarking.analysis import block_align, load_pi_trace, summarize_trace
from omlx.benchmarking.reports import render_markdown


def test_block_align() -> None:
    assert block_align(0, 2048) == 0
    assert block_align(2047, 2048) == 0
    assert block_align(4097, 2048) == 4096


def test_trace_summary_detects_abort_followup(tmp_path) -> None:
    trace_path = tmp_path / "pi-prompt-trace.jsonl"
    rows = [
        {
            "seq": 1,
            "ts": "2026-01-01T00:00:00Z",
            "model": {"id": "Qwen", "provider": "omlx"},
            "system_prompt_len": 100,
            "effective_prompt_len": 120,
            "effective_message_count": 0,
            "common_prefix_chars_vs_prev": 0,
            "divergence_index_vs_prev": -1,
        },
        {
            "seq": 2,
            "ts": "2026-01-01T00:01:00Z",
            "model": {"id": "Qwen", "provider": "omlx"},
            "system_prompt_len": 100,
            "effective_prompt_len": 5000,
            "effective_message_count": 6,
            "common_prefix_chars_vs_prev": 118,
            "divergence_index_vs_prev": 118,
            "curr_snippet_at_divergence": 'stopReason\\":\\"aborted\\",\\"errorMessage\\":\\"Operation aborted\\"',
        },
        {
            "seq": 3,
            "ts": "2026-01-01T00:02:00Z",
            "model": {"id": "Qwen", "provider": "omlx"},
            "system_prompt_len": 100,
            "effective_prompt_len": 5500,
            "effective_message_count": 8,
            "common_prefix_chars_vs_prev": 4998,
            "divergence_index_vs_prev": 4998,
        },
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    trace = load_pi_trace(trace_path)
    result = summarize_trace(trace, harness="pi", workload_id="interrupt_then_continue", block_size=2048)

    assert result.summary["turns"] == 3
    assert result.turns[2].likely_replay_after_abort is True
    assert result.turns[2].block_aligned_prefix_chars == 4096

    markdown = render_markdown(result)
    assert "Harness Cache Benchmark: pi" in markdown
    assert "interrupt_then_continue" in markdown

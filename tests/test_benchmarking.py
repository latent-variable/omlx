# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from omlx.benchmarking.analysis import block_align, load_pi_trace, summarize_trace
from omlx.benchmarking.reports import render_html_dashboard, render_markdown
from omlx.benchmarking.runner import benchmark_result_from_dict


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


def test_html_dashboard_renders_comparison_and_turns() -> None:
    payload = {
        "config": {
            "harness": "pi",
            "model_id": "Qwen",
            "workload_id": "small_followups",
            "block_size": 2048,
            "source_paths": {},
            "metadata": {},
        },
        "summary": {
            "turns": 3,
            "max_prompt_chars": 5500,
            "median_reprocessed_chars_estimate": 1404,
            "max_reprocessed_chars_estimate": 4882,
            "turns_after_abort": 1,
        },
        "turns": [
            {
                "turn_index": 0,
                "harness": "pi",
                "workload_id": "small_followups",
                "model_id": "Qwen",
                "block_size": 2048,
                "prompt_chars_total": 120,
                "common_prefix_chars": 0,
                "block_aligned_prefix_chars": 0,
                "reprocessed_chars_estimate": 120,
                "effective_message_count": 0,
                "custom_messages_added": 0,
                "pending_next_turn_messages": 0,
                "system_prompt_modified": False,
                "previous_turn_aborted": False,
                "likely_replay_after_abort": False,
                "source_payload_path": None,
                "metadata": {"divergence_index_vs_prev": -1},
            },
            {
                "turn_index": 1,
                "harness": "pi",
                "workload_id": "small_followups",
                "model_id": "Qwen",
                "block_size": 2048,
                "prompt_chars_total": 5000,
                "common_prefix_chars": 4098,
                "block_aligned_prefix_chars": 4096,
                "reprocessed_chars_estimate": 904,
                "effective_message_count": 6,
                "custom_messages_added": 0,
                "pending_next_turn_messages": 0,
                "system_prompt_modified": False,
                "previous_turn_aborted": False,
                "likely_replay_after_abort": False,
                "source_payload_path": None,
                "metadata": {"divergence_index_vs_prev": 4098},
            },
            {
                "turn_index": 2,
                "harness": "pi",
                "workload_id": "small_followups",
                "model_id": "Qwen",
                "block_size": 2048,
                "prompt_chars_total": 5500,
                "common_prefix_chars": 4998,
                "block_aligned_prefix_chars": 4096,
                "reprocessed_chars_estimate": 1404,
                "effective_message_count": 8,
                "custom_messages_added": 0,
                "pending_next_turn_messages": 0,
                "system_prompt_modified": False,
                "previous_turn_aborted": True,
                "likely_replay_after_abort": True,
                "source_payload_path": None,
                "metadata": {"divergence_index_vs_prev": 4998},
            },
        ],
    }
    result = benchmark_result_from_dict(payload)
    html = render_html_dashboard([result], title="Bench Dashboard")

    assert "Bench Dashboard" in html
    assert "Comparison Table" in html
    assert "Reuse ratio (%)" in html
    assert "small_followups" in html
    assert "dashboard" in html.lower()

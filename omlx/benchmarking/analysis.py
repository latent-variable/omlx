# SPDX-License-Identifier: Apache-2.0
"""Helpers for analyzing harness traces against oMLX cache behavior."""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from pathlib import Path

from .schema import BenchmarkResult, BenchmarkRunConfig, BenchmarkTurn, HarnessTraceRecord, OmlxLogEvent

_SNAPSHOT_RE = re.compile(
    r"Using boundary cache snapshot for (?P<request_id>[a-f0-9-]+): storing "
    r"(?P<stored>\d+)/(?P<prompt>\d+) tokens"
)
_ABORT_RE = re.compile(r"Aborting request (?P<request_id>[a-f0-9-]+)")
_INTERRUPTED_RE = re.compile(r"Prefill interrupted at (?P<done>\d+)/(?P<total>\d+) tokens")


def block_align(value: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return (value // block_size) * block_size


def load_pi_trace(path: str | Path) -> list[HarnessTraceRecord]:
    records: list[HarnessTraceRecord] = []
    for idx, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        model = raw.get("model", {})
        model_id = model.get("id") or raw.get("model_id") or "unknown"
        records.append(
            HarnessTraceRecord(
                seq=int(raw.get("seq", idx)),
                ts=str(raw["ts"]),
                model_id=str(model_id),
                system_prompt_len=int(raw["system_prompt_len"]),
                effective_prompt_len=int(raw["effective_prompt_len"]),
                effective_message_count=int(raw["effective_message_count"]),
                common_prefix_chars_vs_prev=int(raw.get("common_prefix_chars_vs_prev", 0)),
                divergence_index_vs_prev=int(raw.get("divergence_index_vs_prev", -1)),
                custom_messages_added=int(raw.get("custom_messages_added", 0)),
                pending_next_turn_messages=int(raw.get("pending_next_turn_messages", 0)),
                system_prompt_modified=bool(raw.get("system_prompt_modified", False)),
                payload_path=raw.get("payload_path"),
                raw=raw,
            )
        )
    return records


def load_pi_session_messages(path: str | Path) -> list[dict]:
    messages: list[dict] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("type") != "message":
            continue
        message = raw.get("message")
        if isinstance(message, dict):
            messages.append(message)
    return messages


def parse_omlx_log(path: str | Path) -> list[OmlxLogEvent]:
    events: list[OmlxLogEvent] = []
    for line in Path(path).read_text().splitlines():
        snapshot_match = _SNAPSHOT_RE.search(line)
        if snapshot_match:
            events.append(
                OmlxLogEvent(
                    ts=line[:23],
                    event_type="snapshot",
                    request_id=snapshot_match.group("request_id"),
                    stored_tokens=int(snapshot_match.group("stored")),
                    prompt_tokens=int(snapshot_match.group("prompt")),
                    message=line,
                )
            )
            continue

        abort_match = _ABORT_RE.search(line)
        if abort_match:
            events.append(
                OmlxLogEvent(
                    ts=line[:23],
                    event_type="abort",
                    request_id=abort_match.group("request_id"),
                    message=line,
                )
            )
            continue

        interrupted_match = _INTERRUPTED_RE.search(line)
        if interrupted_match:
            events.append(
                OmlxLogEvent(
                    ts=line[:23],
                    event_type="prefill_interrupted",
                    interrupted_tokens=int(interrupted_match.group("done")),
                    total_tokens=int(interrupted_match.group("total")),
                    message=line,
                )
            )
    return events


def summarize_trace(
    trace: list[HarnessTraceRecord],
    *,
    harness: str,
    workload_id: str,
    block_size: int,
    model_id: str | None = None,
) -> BenchmarkResult:
    if not trace:
        raise ValueError("trace must not be empty")

    turns: list[BenchmarkTurn] = []
    previous_aborted = False
    for idx, record in enumerate(trace):
        raw = record.raw
        stop_reason = (
            str(raw.get("curr_snippet_at_divergence", "")).lower()
            if raw.get("curr_snippet_at_divergence")
            else ""
        )
        likely_abort = "stopreason\\\":\\\"aborted\\\"" in stop_reason or "operation aborted" in stop_reason
        block_aligned = block_align(record.common_prefix_chars_vs_prev, block_size)
        reprocessed = max(record.effective_prompt_len - block_aligned, 0)
        turns.append(
            BenchmarkTurn(
                turn_index=idx,
                harness=harness,
                workload_id=workload_id,
                model_id=model_id or record.model_id,
                block_size=block_size,
                prompt_chars_total=record.effective_prompt_len,
                common_prefix_chars=record.common_prefix_chars_vs_prev,
                block_aligned_prefix_chars=block_aligned,
                reprocessed_chars_estimate=reprocessed,
                effective_message_count=record.effective_message_count,
                custom_messages_added=record.custom_messages_added,
                pending_next_turn_messages=record.pending_next_turn_messages,
                system_prompt_modified=record.system_prompt_modified,
                previous_turn_aborted=previous_aborted,
                likely_replay_after_abort=previous_aborted,
                source_payload_path=record.payload_path,
                metadata={
                    "divergence_index_vs_prev": record.divergence_index_vs_prev,
                },
            )
        )
        previous_aborted = likely_abort

    reprocessed = [turn.reprocessed_chars_estimate for turn in turns[1:]]
    reuse_ratios = [
        (turn.block_aligned_prefix_chars / turn.prompt_chars_total) if turn.prompt_chars_total else 0.0 for turn in turns[1:]
    ]
    reprocessed_fractions = [
        (turn.reprocessed_chars_estimate / turn.prompt_chars_total) if turn.prompt_chars_total else 0.0 for turn in turns[1:]
    ]
    summary = {
        "turns": len(turns),
        "max_prompt_chars": max(turn.prompt_chars_total for turn in turns),
        "median_reprocessed_chars_estimate": statistics.median(reprocessed) if reprocessed else 0,
        "max_reprocessed_chars_estimate": max(reprocessed) if reprocessed else 0,
        "turns_after_abort": sum(1 for turn in turns if turn.likely_replay_after_abort),
        "median_reuse_ratio": statistics.median(reuse_ratios) if reuse_ratios else 0.0,
        "median_reprocessed_fraction": statistics.median(reprocessed_fractions) if reprocessed_fractions else 0.0,
    }
    config = BenchmarkRunConfig(
        harness=harness,
        model_id=model_id or trace[0].model_id,
        workload_id=workload_id,
        block_size=block_size,
        metadata={"prompt_unit": "chars", "reuse_metric": "block_aligned_prefix_over_prompt"},
    )
    return BenchmarkResult(config=config, turns=turns, summary=summary)


def summarize_pi_session(
    session_path: str | Path,
    *,
    workload_id: str,
    block_size: int = 2048,
) -> BenchmarkResult:
    messages = load_pi_session_messages(session_path)
    if not messages:
        raise ValueError(f"no Pi messages found in session {session_path}")

    turns: list[BenchmarkTurn] = []
    previous_aborted = False
    model_id = "unknown"
    effective_message_count = 0
    for message in messages:
        role = str(message.get("role", ""))
        if role in {"user", "assistant", "toolResult"}:
            effective_message_count += 1
        if role != "assistant":
            continue

        model_id = str(message.get("model") or model_id)
        usage = message.get("usage", {}) if isinstance(message.get("usage"), dict) else {}
        uncached_prompt_tokens = int(usage.get("input", 0) or 0)
        cached_tokens = int(usage.get("cacheRead", 0) or 0)
        prompt_tokens = uncached_prompt_tokens + cached_tokens
        stop_reason = str(message.get("stopReason", "") or "")
        error_message = str(message.get("errorMessage", "") or "")
        was_aborted = stop_reason == "aborted" or error_message == "Operation aborted"

        turns.append(
            BenchmarkTurn(
                turn_index=len(turns),
                harness="pi",
                workload_id=workload_id,
                model_id=model_id,
                block_size=block_size,
                prompt_chars_total=prompt_tokens,
                common_prefix_chars=cached_tokens,
                block_aligned_prefix_chars=cached_tokens,
                reprocessed_chars_estimate=uncached_prompt_tokens,
                effective_message_count=effective_message_count,
                custom_messages_added=0,
                pending_next_turn_messages=0,
                system_prompt_modified=False,
                previous_turn_aborted=previous_aborted,
                likely_replay_after_abort=previous_aborted,
                source_payload_path=str(session_path),
                metadata={
                    "stop_reason": stop_reason,
                    "error_message": error_message,
                    "output_tokens": int(usage.get("output", 0) or 0),
                    "prompt_unit": "tokens",
                    "reuse_metric": "cache_read_over_input",
                },
            )
        )
        previous_aborted = was_aborted

    if not turns:
        raise ValueError(f"no Pi assistant turns found in session {session_path}")

    reprocessed = [turn.reprocessed_chars_estimate for turn in turns]
    reuse_ratios = [
        (turn.block_aligned_prefix_chars / turn.prompt_chars_total) if turn.prompt_chars_total else 0.0 for turn in turns
    ]
    reprocessed_fractions = [
        (turn.reprocessed_chars_estimate / turn.prompt_chars_total) if turn.prompt_chars_total else 0.0 for turn in turns
    ]
    config = BenchmarkRunConfig(
        harness="pi",
        model_id=model_id,
        workload_id=workload_id,
        block_size=block_size,
        source_paths={"pi_session_jsonl": str(session_path)},
        metadata={"prompt_unit": "tokens", "reuse_metric": "cache_read_over_input"},
    )
    return BenchmarkResult(
        config=config,
        turns=turns,
        summary={
            "turns": len(turns),
            "max_prompt_chars": max(turn.prompt_chars_total for turn in turns),
            "median_reprocessed_chars_estimate": statistics.median(reprocessed) if reprocessed else 0,
            "max_reprocessed_chars_estimate": max(reprocessed) if reprocessed else 0,
            "turns_after_abort": sum(1 for turn in turns if turn.likely_replay_after_abort),
            "median_reuse_ratio": statistics.median(reuse_ratios) if reuse_ratios else 0.0,
            "median_reprocessed_fraction": statistics.median(reprocessed_fractions) if reprocessed_fractions else 0.0,
        },
    )


def summarize_opencode_session(
    session_id: str,
    *,
    db_path: str | Path,
    workload_id: str,
) -> BenchmarkResult:
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        select id, time_created, data
        from message
        where session_id = ?
        order by time_created, id
        """,
        (session_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"no messages found for OpenCode session {session_id}")

    turns: list[BenchmarkTurn] = []
    previous_aborted = False
    model_id = "unknown"
    for idx, row in enumerate(rows):
        payload = json.loads(row["data"])
        role = payload.get("role")
        if role == "user":
            model = payload.get("model", {})
            model_id = model.get("modelID") or model.get("id") or model_id
            continue
        if role != "assistant":
            continue
        model_id = payload.get("modelID") or model_id
        tokens = payload.get("tokens", {})
        cache = tokens.get("cache", {}) if isinstance(tokens, dict) else {}
        uncached_prompt_tokens = int(tokens.get("input", 0) or 0)
        cached_tokens = int(cache.get("read", 0) or 0)
        prompt_tokens = uncached_prompt_tokens + cached_tokens
        reprocessed_tokens = uncached_prompt_tokens
        error = payload.get("error", {}) or {}
        finish = str(payload.get("finish", "") or "")
        error_name = str(error.get("name", "") or "")
        was_aborted = finish == "aborted" or error_name == "MessageAbortedError"
        turns.append(
            BenchmarkTurn(
                turn_index=len(turns),
                harness="opencode",
                workload_id=workload_id,
                model_id=model_id,
                block_size=2048,
                prompt_chars_total=prompt_tokens,
                common_prefix_chars=cached_tokens,
                block_aligned_prefix_chars=cached_tokens,
                reprocessed_chars_estimate=reprocessed_tokens,
                effective_message_count=len(turns) + 1,
                custom_messages_added=0,
                pending_next_turn_messages=0,
                system_prompt_modified=False,
                previous_turn_aborted=previous_aborted,
                likely_replay_after_abort=previous_aborted,
                source_payload_path=f"{db_path}#{session_id}",
                metadata={
                    "message_id": row["id"],
                    "finish": finish,
                    "error_name": error_name,
                    "cache_write_tokens": int(cache.get("write", 0) or 0),
                    "uncached_prompt_tokens": uncached_prompt_tokens,
                    "prompt_unit": "tokens",
                    "reuse_metric": "cache_read_over_input",
                },
            )
        )
        previous_aborted = was_aborted

    if not turns:
        raise ValueError(f"no assistant turns found for OpenCode session {session_id}")

    reprocessed = [turn.reprocessed_chars_estimate for turn in turns]
    reuse_ratios = [
        (turn.block_aligned_prefix_chars / turn.prompt_chars_total) if turn.prompt_chars_total else 0.0 for turn in turns
    ]
    reprocessed_fractions = [
        (turn.reprocessed_chars_estimate / turn.prompt_chars_total) if turn.prompt_chars_total else 0.0 for turn in turns
    ]
    config = BenchmarkRunConfig(
        harness="opencode",
        model_id=model_id,
        workload_id=workload_id,
        block_size=2048,
        source_paths={"opencode_db": str(db_path)},
        metadata={"prompt_unit": "tokens", "reuse_metric": "cache_read_over_input", "session_id": session_id},
    )
    return BenchmarkResult(
        config=config,
        turns=turns,
        summary={
            "turns": len(turns),
            "max_prompt_chars": max(turn.prompt_chars_total for turn in turns),
            "median_reprocessed_chars_estimate": statistics.median(reprocessed) if reprocessed else 0,
            "max_reprocessed_chars_estimate": max(reprocessed) if reprocessed else 0,
            "turns_after_abort": sum(1 for turn in turns if turn.likely_replay_after_abort),
            "median_reuse_ratio": statistics.median(reuse_ratios) if reuse_ratios else 0.0,
            "median_reprocessed_fraction": statistics.median(reprocessed_fractions) if reprocessed_fractions else 0.0,
        },
    )

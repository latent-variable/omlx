# SPDX-License-Identifier: Apache-2.0
"""Helpers for analyzing harness traces against oMLX cache behavior."""

from __future__ import annotations

import json
import re
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
    summary = {
        "turns": len(turns),
        "max_prompt_chars": max(turn.prompt_chars_total for turn in turns),
        "median_reprocessed_chars_estimate": sorted(reprocessed)[len(reprocessed) // 2] if reprocessed else 0,
        "max_reprocessed_chars_estimate": max(reprocessed) if reprocessed else 0,
        "turns_after_abort": sum(1 for turn in turns if turn.likely_replay_after_abort),
    }
    config = BenchmarkRunConfig(
        harness=harness,
        model_id=model_id or trace[0].model_id,
        workload_id=workload_id,
        block_size=block_size,
    )
    return BenchmarkResult(config=config, turns=turns, summary=summary)

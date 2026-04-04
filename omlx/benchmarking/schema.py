# SPDX-License-Identifier: Apache-2.0
"""Shared schema for harness cache benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class WorkloadTurn:
    label: str
    prompt: str
    event_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkloadDefinition:
    workload_id: str
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    turns: list[WorkloadTurn] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkRunConfig:
    harness: str
    model_id: str
    workload_id: str
    block_size: int
    source_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HarnessTraceRecord:
    seq: int
    ts: str
    model_id: str
    system_prompt_len: int
    effective_prompt_len: int
    effective_message_count: int
    common_prefix_chars_vs_prev: int
    divergence_index_vs_prev: int
    custom_messages_added: int = 0
    pending_next_turn_messages: int = 0
    system_prompt_modified: bool = False
    payload_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OmlxLogEvent:
    ts: str
    event_type: str
    request_id: str | None = None
    stored_tokens: int | None = None
    prompt_tokens: int | None = None
    interrupted_tokens: int | None = None
    total_tokens: int | None = None
    message: str = ""


@dataclass(slots=True)
class BenchmarkTurn:
    turn_index: int
    harness: str
    workload_id: str
    model_id: str
    block_size: int
    prompt_chars_total: int
    common_prefix_chars: int
    block_aligned_prefix_chars: int
    reprocessed_chars_estimate: int
    effective_message_count: int
    custom_messages_added: int
    pending_next_turn_messages: int
    system_prompt_modified: bool
    previous_turn_aborted: bool
    likely_replay_after_abort: bool
    source_payload_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkResult:
    config: BenchmarkRunConfig
    turns: list[BenchmarkTurn] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "turns": [turn.to_dict() for turn in self.turns],
            "summary": dict(self.summary),
        }

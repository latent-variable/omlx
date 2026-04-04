# SPDX-License-Identifier: Apache-2.0
"""Reusable workload definitions for agent-harness benchmarks."""

from __future__ import annotations

from .schema import WorkloadDefinition, WorkloadTurn


def builtin_workloads() -> list[WorkloadDefinition]:
    return [
        WorkloadDefinition(
            workload_id="small_followups",
            title="Small Follow-ups",
            description="Grow context with repository exploration, then ask small follow-ups that should preserve most of the prefix.",
            tags=["followup", "cache", "agentic"],
            turns=[
                WorkloadTurn("start", "Read the largest files in this project and summarize each in two sentences.", "analysis"),
                WorkloadTurn("followup_lines", "Can you tell me the number of lines of each?", "follow_up"),
                WorkloadTurn("followup_detail", "Which one looks the most suspicious and why?", "follow_up"),
            ],
        ),
        WorkloadDefinition(
            workload_id="interrupt_then_continue",
            title="Interrupt Then Continue",
            description="Interrupt a long answer and continue, to measure replay and recompute cost after aborts.",
            tags=["interrupt", "continue", "followup"],
            turns=[
                WorkloadTurn("start", "Read the entire file and look for interesting code paths.", "analysis"),
                WorkloadTurn("interrupt", "Interrupt the current response.", "interrupt"),
                WorkloadTurn("continue", "continue", "follow_up"),
            ],
        ),
        WorkloadDefinition(
            workload_id="new_thread_restart",
            title="New Thread Restart",
            description="Start a fresh thread on the same repo to measure cold-ish reuse versus same-thread continuation.",
            tags=["thread", "cold_start"],
            turns=[
                WorkloadTurn("thread_start", "hi", "thread_start"),
                WorkloadTurn("deep_task", "Find a really high issue bug in this project, take your time.", "analysis"),
            ],
        ),
    ]

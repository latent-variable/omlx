# Agent Harness Benchmarking

## Goal

Build an additive benchmark subsystem inside `omlx` that measures how friendly
different agent harnesses are to oMLX's prefix/block caching under realistic
agentic workloads.

This work must stay aligned with upstream `omlx` and avoid changing core engine,
scheduler, or cache semantics unless an instrumentation hook is strictly
optional and narrowly scoped.

## Non-Goals

- Forking or replacing existing `omlx` benchmark logic
- Changing cache behavior to make benchmarks look better
- Building harness-specific benchmark code into the core runtime path
- Depending on one harness's internal implementation details for the overall
  benchmark design

## Principles

- Keep core `omlx` read-mostly
- Add benchmark code as a separate package layer under `omlx/benchmarking/`
- Reuse existing cache metrics, benchmark patterns, and integration tests where
  possible
- Prefer JSONL outputs and small composable modules over monolithic scripts
- Prioritize realistic follow-up-heavy agent workflows over synthetic throughput
  wins

## Benchmark Questions

1. How many tokens does each harness force oMLX to reprocess on follow-up turns?
2. How stable is the prompt prefix across turns, tool calls, interrupts, and
   thread changes?
3. How do block sizes affect partial reuse, recompute waste, and latency?
4. Which harnesses are most cache-friendly for locally hosted agentic workflows?

## MVP Scope

- Shared benchmark schema and result model
- Prompt-prefix and block-alignment analysis helpers
- Reusable workload definitions for realistic agentic scenarios
- A trace-analysis runner that can consume Pi/oMLX logs immediately
- Markdown and JSONL reporting
- Tests for schema, analysis helpers, and basic report generation

## Next Phases

- Live harness adapters for Pi, Codex, OpenCode, and others
- Automated end-to-end runner for harness x workload x block-size matrices
- Admin/report UI integration if useful after the CLI pipeline is stable

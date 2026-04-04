#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 cache-miss harness for scheduler-level debugging.

Runs deterministic requests directly against the Scheduler with a persistent
temporary SSD cache directory, so we can observe how cache reuse behaves as
prompt length grows and the prompt shape changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from mlx_lm import load

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


def _build_messages(seed: str, repeat: int) -> list[dict[str, str]]:
    base = (
        f"[{seed}] You are debugging long-context prefix caching in a local LLM server. "
        "Track invariants around block boundaries, exact-prefix reuse, reconstruction, "
        "and suffix growth. Prefer concise, technical reasoning with explicit hypotheses. "
        "We care about deterministic cache reuse, prompt growth, repeated requests, "
        "and whether cache state becomes unreachable after enough evictions or edits. "
    )
    system = base * repeat
    user = (
        "Explain how a tiered KV cache should behave when the same conversation grows over time. "
        "Call out exact repeats, suffix appends, and edits in the middle of the prompt."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _encode_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    try:
        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        token_ids = tokenizer.encode(text)
    return list(token_ids)


def _fit_to_length(tokenizer: Any, seed: str, target_len: int) -> list[int]:
    repeat = 8
    token_ids = _encode_messages(tokenizer, _build_messages(seed, repeat))
    while len(token_ids) < target_len:
        repeat = max(repeat + 4, int(repeat * 1.3))
        token_ids = _encode_messages(tokenizer, _build_messages(seed, repeat))
    return token_ids[:target_len]


def _same_length_edit(tokenizer: Any, source: list[int], replacement_seed: str) -> list[int]:
    edit_width = min(128, max(32, len(source) // 32))
    edit_start = max(0, (len(source) // 2) - (edit_width // 2))
    replacement = _fit_to_length(tokenizer, replacement_seed, edit_width)
    edited = list(source)
    edited[edit_start : edit_start + edit_width] = replacement[:edit_width]
    return edited


def _run_request(
    model: Any,
    tokenizer: Any,
    prompt_token_ids: list[int],
    model_name: str,
    cache_dir: str,
    block_size: int,
    max_tokens: int,
    request_id: str,
) -> dict[str, Any]:
    config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=max(8192, min(len(prompt_token_ids), 32768)),
        completion_batch_size=1,
        prefill_step_size=2048,
        paged_ssd_cache_dir=cache_dir,
        paged_cache_block_size=block_size,
        paged_ssd_cache_max_size=10 * 1024 * 1024 * 1024,
        model_name=model_name,
    )
    scheduler = Scheduler(config=config, model=model, tokenizer=tokenizer)
    request = Request(
        request_id=request_id,
        prompt=prompt_token_ids,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
        ),
    )

    started = time.perf_counter()
    scheduler.add_request(request)

    last_output = None
    for _ in range(max_tokens + 256):
        step_result = scheduler.step()
        for output in step_result.outputs:
            last_output = output
        if step_result.finished_request_ids:
            break

    elapsed = time.perf_counter() - started
    stats = scheduler.get_ssd_cache_stats() or {}
    cache_debug = stats.get("cache_debug", {})
    prefix_cache = stats.get("prefix_cache", {})
    indexed_blocks = stats.get("indexed_blocks", 0)
    scheduler.shutdown()

    return {
        "request_id": request_id,
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": getattr(last_output, "completion_tokens", 0) if last_output else 0,
        "cached_tokens": getattr(last_output, "cached_tokens", 0) if last_output else 0,
        "elapsed_s": round(elapsed, 3),
        "indexed_blocks": indexed_blocks,
        "cache_debug": cache_debug,
        "prefix_cache": {
            "hits": prefix_cache.get("hits", 0),
            "misses": prefix_cache.get("misses", 0),
            "tokens_saved": prefix_cache.get("tokens_saved", 0),
            "partial_block_skips": prefix_cache.get("partial_block_skips", 0),
            "last_partial_tokens_skipped": prefix_cache.get("last_partial_tokens_skipped", 0),
            "last_tokens_to_next_block": prefix_cache.get("last_tokens_to_next_block", 0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 cache reuse harness")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--keep-cache-dir", action="store_true")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser())
    model, tokenizer = load(model_path)
    cache_dir = tempfile.mkdtemp(prefix="omlx_phase1_cache_")

    print(json.dumps({
        "model_path": model_path,
        "cache_dir": cache_dir,
        "block_size": args.block_size,
        "lengths": args.lengths,
        "max_tokens": args.max_tokens,
    }))

    results: list[dict[str, Any]] = []
    try:
        for length in args.lengths:
            base_tokens = _fit_to_length(tokenizer, f"base-{length}", length)
            suffix_small = _fit_to_length(tokenizer, f"suffix-small-{length}", 128)
            suffix_large = _fit_to_length(tokenizer, f"suffix-large-{length}", 512)

            variants = [
                ("cold_base", base_tokens),
                ("exact_repeat", list(base_tokens)),
                ("suffix_plus_128", list(base_tokens) + suffix_small),
                ("middle_edit_same_len", _same_length_edit(tokenizer, base_tokens, f"edit-{length}")),
                ("suffix_plus_512", list(base_tokens) + suffix_large),
            ]

            print(f"\n=== prompt_length={length} ===")
            for label, prompt_token_ids in variants:
                request_id = f"{length}-{label}"
                result = _run_request(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_token_ids=prompt_token_ids,
                    model_name=model_path,
                    cache_dir=cache_dir,
                    block_size=args.block_size,
                    max_tokens=args.max_tokens,
                    request_id=request_id,
                )
                result["label"] = label
                result["base_length"] = length
                results.append(result)
                print(json.dumps(result, sort_keys=True))
    finally:
        if args.keep_cache_dir:
            print(f"\nKept cache dir: {cache_dir}")
        else:
            shutil.rmtree(cache_dir, ignore_errors=True)

    summary = {
        "total_runs": len(results),
        "lengths": args.lengths,
        "runs_with_cached_tokens": sum(1 for r in results if r["cached_tokens"] > 0),
        "runs_with_reconstruct_failures": sum(
            1 for r in results if r["cache_debug"].get("reconstruct_failures", 0) > 0
        ),
    }
    print("\n=== summary ===")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

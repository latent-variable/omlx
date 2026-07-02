#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""End-to-end chunk-reuse demo against a running oMLX server.

Scenario (agentic): read the same source files in two different "sessions"
(different system prompt + question). Prefix caching serves ~0% of the second
session; chunk reuse should serve the shared file content.

Measures per-request prefill throughput / TTFT and prints the server's
chunk_reuse cache stats. Compares a chunk-reuse hit (B) against a same-size
no-reuse control (C) within one server run to isolate the speedup.

Usage:
  python scripts/chunk_reuse_demo.py --base-url http://localhost:5599 \
    --api-key testkey --model Qwen3.6-35B-A3B-8bit
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "omlx" / "cache"
FILES = ["prefix_cache.py", "paged_cache.py", "scheduler.py"]  # scheduler is large


def _post(base, key, path, body):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data, time.perf_counter() - t0


def _get(base, key, path):
    req = urllib.request.Request(base + path, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _file_block(name, max_lines=None):
    text = (CACHE_DIR / name).read_text()
    if max_lines:
        text = "\n".join(text.splitlines()[:max_lines])
    return f'\n<file path="{name}">\n{text}\n</file>\n'


def chat(base, key, model, system, user, max_tokens=32):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }
    data, wall = _post(base, key, "/v1/chat/completions", body)
    usage = data.get("usage", {})
    text = data["choices"][0]["message"]["content"]
    return {
        "wall_s": wall,
        "prompt_tokens": usage.get("prompt_tokens"),
        "prompt_tps": usage.get("prompt_tokens_per_second"),
        "text": text[:80],
    }, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:5599")
    ap.add_argument("--api-key", default="testkey")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    base, key, model = args.base_url, args.api_key, args.model

    files_a = _file_block("prefix_cache.py", 200) + _file_block("paged_cache.py", 200)
    files_c = _file_block("scheduler.py", 400)  # unique control content

    def stats():
        s = _get(base, key, "/admin/api/cache/stats") if _has_admin(base, key) else None
        return s

    print(f"model={model}\n")

    # Session A: read the files, populate the chunk store.
    a, _ = chat(base, key, model,
                "You are a coding agent debugging cache behavior.",
                "Here are the files:\n" + files_a +
                "\nBriefly: what does compute_block_hash use?")
    print(f"A (cold, populates store): prompt_tokens={a['prompt_tokens']} "
          f"prompt_tps={a['prompt_tps']:.0f} wall={a['wall_s']:.2f}s :: {a['text']!r}")

    time.sleep(1)

    # Session B: NEW session (different system + question), SAME files.
    # Prefix cache misses (different prefix); chunk reuse should hit.
    b, _ = chat(base, key, model,
                "You are a meticulous reviewer doing a fresh audit. Keep it short.",
                "New session. Re-examine these modules:\n" + files_a +
                "\nWhat is the block size in prefix_cache.py?")
    print(f"B (reuse, same files new session): prompt_tokens={b['prompt_tokens']} "
          f"prompt_tps={b['prompt_tps']:.0f} wall={b['wall_s']:.2f}s :: {b['text']!r}")

    # Control C: same-size prompt but UNIQUE content (no reuse possible).
    c, _ = chat(base, key, model,
                "You are a meticulous reviewer doing a fresh audit. Keep it short.",
                "New session. Re-examine this module:\n" + files_c +
                "\nWhat does the scheduler's add_request do?")
    print(f"C (control, unique content): prompt_tokens={c['prompt_tokens']} "
          f"prompt_tps={c['prompt_tps']:.0f} wall={c['wall_s']:.2f}s :: {c['text']!r}")

    print("\n--- interpretation ---")
    if b["prompt_tps"] and c["prompt_tps"]:
        print(f"B effective prompt throughput / C (no-reuse) = "
              f"{b['prompt_tps'] / c['prompt_tps']:.1f}x")
    sc = stats()
    if sc:
        print("\nserver cache stats (chunk_reuse):")
        print(json.dumps(sc.get("chunk_reuse", sc), indent=2))


def _has_admin(base, key):
    try:
        _get(base, key, "/admin/api/cache/stats")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()

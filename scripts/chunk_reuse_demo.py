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
import uuid
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
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    pt = usage.get("prompt_tokens") or 0
    return {
        "wall_s": wall,
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "cached_pct": (100.0 * cached / pt) if pt else 0.0,
        "total_time": usage.get("total_time"),
        "text": text[:80],
    }, data


def _row(label, r):
    return (f"{label}: prompt_tokens={r['prompt_tokens']} "
            f"cached={r['cached_tokens']} ({r['cached_pct']:.0f}%) "
            f"wall={r['wall_s']:.2f}s :: {r['text']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:5599")
    ap.add_argument("--api-key", default="testkey")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    base, key, model = args.base_url, args.api_key, args.model

    files_a = _file_block("prefix_cache.py", 200) + _file_block("paged_cache.py", 200)
    files_c = _file_block("paged_ssd_cache.py", 400)  # unique control content

    # Per-run nonce in the system prompts so the persistent paged SSD prefix
    # cache from *previous* demo runs can't serve these prompts — isolates
    # what THIS run's chunk store contributes.
    nonce = uuid.uuid4().hex[:8]

    print(f"model={model} nonce={nonce}\n")

    # Session A: read the files, populate the chunk store.
    a, _ = chat(base, key, model,
                f"[session {nonce}-a] You are a coding agent debugging cache behavior.",
                "Here are the files:\n" + files_a +
                "\nBriefly: what does compute_block_hash use?")
    print(_row("A (cold, populates store)   ", a))

    time.sleep(1)

    # Session B: NEW session (different system + question), SAME files.
    # Prefix cache misses (different prefix); chunk reuse should hit.
    b, _ = chat(base, key, model,
                f"[session {nonce}-b] You are a meticulous reviewer doing a fresh audit. Keep it short.",
                "New session. Re-examine these modules:\n" + files_a +
                "\nWhat is the block size in prefix_cache.py?")
    print(_row("B (reuse: same files, new sess)", b))

    # Control C: same-size prompt but UNIQUE content (no reuse possible).
    c, _ = chat(base, key, model,
                f"[session {nonce}-c] You are a meticulous reviewer doing a fresh audit. Keep it short.",
                "New session. Re-examine this module:\n" + files_c +
                "\nWhat does the scheduler add_request do?")
    print(_row("C (control: unique content)  ", c))

    print("\n--- interpretation ---")
    print(f"B cached {b['cached_pct']:.0f}% of its prompt via chunk reuse; "
          f"control C cached {c['cached_pct']:.0f}%.")
    if b["wall_s"] and c["wall_s"] and abs(b['prompt_tokens'] - c['prompt_tokens']) < 400:
        print(f"B wall {b['wall_s']:.2f}s vs C {c['wall_s']:.2f}s "
              f"(similar prompt size) → {c['wall_s']/b['wall_s']:.2f}x")

    # Best-effort: dig chunk_reuse stats out of the admin observability blob.
    try:
        blob = _get(base, key, "/admin/api/stats")
        found = _find_key(blob, "chunk_reuse")
        if found:
            print("\nserver chunk_reuse stats:")
            print(json.dumps(found, indent=2))
    except Exception:
        pass


def _find_key(obj, target):
    if isinstance(obj, dict):
        if target in obj:
            return obj[target]
        for v in obj.values():
            r = _find_key(v, target)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, target)
            if r is not None:
                return r
    return None


if __name__ == "__main__":
    main()

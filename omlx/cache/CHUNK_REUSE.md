# Chunk reuse (position-independent KV cache) — integration notes

**Status:** experimental, opt-in (`OMLX_CHUNK_REUSE=true` or
`config.chunk_reuse.enabled`). Default off. Does not touch the prefix cache.

## What it is

The prefix cache reuses KV only when the *entire* preceding context matches
byte-for-byte (chain-hashed 256-token blocks). Chunk reuse reuses a chunk's
KV when the same content appears at a *different* position or under a
*different* prefix — the misses an agent hits constantly: new session over the
same files, compacted history, a file re-read after edits elsewhere.

Mechanism (validated standalone in `LatentPlayground/kv-subset-cache`):

1. Cache a chunk's per-layer KV (content-hashed, no parent chaining).
2. On reuse, re-rotate keys by the position delta (exact; respects partial
   rotary and llama3 scaling).
3. Recompute a small fraction of tokens to correct cross-chunk attention
   (`edge` = leading tokens, cheapest, best on MoE; `devblock` = deviation-
   selected blocks, better on dense models).

Hybrid models (qwen3_5-family: GatedDeltaNet + attention — this is the
Qwen3.6-27B/35B architecture) additionally cache each chunk's linear-layer
per-token inputs (qkv/a/b) and replay the cheap conv + delta scan from live
state; the recurrence is position-free so no re-rotation is needed there.

## Results (standalone, real oMLX source files as chunks)

| model | mode | recompute | speedup | correctness |
|---|---|---|---|---|
| Qwen2.5-7B (full-attn) | devblock | 15% | 3.1x | 5/5 |
| Llama-3.1-8B (full-attn) | devblock | 15% | 3.5x | 5/5 |
| Qwen3.6-35B-A3B (hybrid) | edge | 3% | 3.0x | 5/5 |

## Results (end-to-end, live server, chunk reuse ON, 2026-07-02)

New-session prompt sharing ~3.4k tokens of file content with a prior
session (prefix cache misses; chunk reuse assembles):

| model | cached | wall vs cold | quality |
|---|---|---|---|
| Qwen3.6-35B-A3B-8bit | 98–99% | 2.24s → ~1.0s (~2.2x) | 3/3 factual Qs match baseline |
| Qwen3.6-27B-oQ4-mtp | 99% | ~10.4s → 2.86s (~3x) | 3/3 factual Qs match baseline |

Cold/donor requests include the capture recording prefill (hybrids rerun a
recording forward to grab linear-layer inputs), roughly doubling novel-prompt
prefill; skip-first-sight / async capture is the obvious follow-up.

## Files

- `chunk_reuse.py` — full-attention path (extract / re-rotate / blended prefill)
- `chunk_reuse_hybrid.py` — hybrid path (linear-input replay + KV transplant)
- `chunk_reuse_vlm.py` — same hybrid mechanism on the mlx-vlm runtime
  (M-RoPE `rotary_emb`; how oMLX actually serves Qwen3.5/3.6 incl. MTP/oQ)
- `chunk_reuse_engine.py` — content-defined chunk store + capture/assemble
- `config.py::ChunkReuseConfig` — the toggle and recompute policy

## Integration status (this branch)

- [x] Config flag + env override (`OMLX_CHUNK_REUSE`), default off
- [x] Port validated mechanism into the package
- [x] Content-hash chunk index (gear-hash CDC; in-memory, per-model)
- [x] Scheduler hook: on prefix-cache miss, look up chunks, assemble via
      blended prefill instead of full prefill
- [x] Arch gating: full-attn / hybrid / vlm_hybrid; sliding-window (gemma4)
      and unknown archs decline (normal prefill runs)
- [x] MTP: 27B-oQ4-mtp generates correctly from assembled caches (e2e)
- [ ] Persist chunk store to SSD (currently in-memory, lost on restart)
- [ ] Capture cost: skip-first-sight or async recording prefill
- [ ] Upstream the mlx-vlm capture/replay primitives (long-term home)

## Constraints

- Full-attention and qwen3_5-hybrid (mlx-lm or mlx-vlm runtime) only.
  gemma4 sliding-window not yet supported (window makes reuse local — a
  future LegoLink-at-window variant).
- Text-only prompts for now (image spans decline; M-RoPE rigid-shift makes
  them feasible later).
- Quality is opt-in / approximate: facts hold, exact wording drifts. Not
  bit-exact generation.

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

## Files

- `chunk_reuse.py` — full-attention path (extract / re-rotate / blended prefill)
- `chunk_reuse_hybrid.py` — hybrid path (linear-input replay + KV transplant)
- `config.py::ChunkReuseConfig` — the toggle and recompute policy

## Integration plan (this branch)

- [x] Config flag + env override, default off
- [x] Port validated mechanism into the package
- [ ] Content-hash chunk index alongside the paged block store (reuse the
      existing SSD format; chunk = block sequence keyed by content hash)
- [ ] Scheduler hook: on prefix-cache miss, look up chunks, assemble via
      blended prefill instead of full prefill
- [ ] Arch gating: full-attn → `chunk_reuse`, hybrid → `chunk_reuse_hybrid`,
      sliding-window (gemma4) → decline (fall back to prefix cache)
- [ ] MTP: verify draft path reads reused caches unchanged at decode

## Constraints

- Full-attention and qwen3_5-hybrid only. gemma4 sliding-window not yet
  supported (window makes reuse local — a future LegoLink-at-window variant).
- Quality is opt-in / approximate: facts hold, exact wording drifts. Not
  bit-exact generation.
- The 27B's oQ4e quant only decodes inside oMLX, so it is validated here (in
  the engine) rather than standalone.

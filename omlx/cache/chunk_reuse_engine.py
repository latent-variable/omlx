# SPDX-License-Identifier: Apache-2.0
"""ChunkReuseEngine: content-hashed store + capture/assemble for chunk reuse.

Sits beside the paged SSD prefix cache in its own namespace. The prefix cache
serves exact-prefix hits; this serves the misses an agent hits constantly —
the same file/content reappearing at a different position or under a different
prefix (new session, compacted history, re-read after edits elsewhere).

Flow:
  capture(prompt_ids, prompt_cache)   after a full prefill, segment the prompt
                                      into fixed blocks and store each block's
                                      KV (+ hybrid linear records) by content
                                      hash.
  assemble(prompt_ids)                on a later prompt, find stored chunks as
                                      contiguous subsequences; rebuild a prompt
                                      cache via blended/hybrid prefill (reuse +
                                      RoPE re-rotation + small recompute),
                                      returning (prompt_cache, cached_tokens,
                                      remaining_tokens, stats).

Arch support: full-attention (KVCache-only) and qwen3_5-family hybrids
(GatedDeltaNet + attention). Sliding-window (gemma4) declines. Detection is
by inspecting model.make_cache() layer types + layer flags.

Validated standalone: LatentPlayground/kv-subset-cache (report.md).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from mlx_lm.models.cache import ArraysCache, KVCache

from . import chunk_reuse as cr
from . import chunk_reuse_hybrid as crh

logger = logging.getLogger(__name__)


def _hash_tokens(token_ids: list[int], model_name: str) -> bytes:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"omlx-chunk-v1")
    h.update(bytes(str(tuple(token_ids)), "utf-8"))
    return h.digest()


# Gear table for content-defined chunking (fixed random 64-bit constants).
_GEAR = [((i * 0x9E3779B97F4A7C15) ^ (i << 17) ^ (i >> 3)) & 0xFFFFFFFFFFFFFFFF
         for i in range(0x20000)]


def content_defined_chunks(
    token_ids: list[int], *, min_size: int, avg_size: int, max_size: int
) -> list[tuple[int, int]]:
    """Split token ids into (start, end) chunks at content-defined boundaries.

    Gear-hash CDC: a boundary is cut where the rolling hash of the recent
    window has enough low bits clear. Because cut points depend on content
    (not absolute position), the *same* content chunks identically wherever it
    appears — so a file re-read under a different prefix produces matching
    chunk hashes and becomes reusable. The first ~window tokens after a content
    change may chunk differently until the rolling window clears; subsequent
    chunks converge.
    """
    n = len(token_ids)
    if n < min_size:
        return []
    mask = (avg_size - 1) if (avg_size & (avg_size - 1)) == 0 else ((1 << (avg_size.bit_length() - 1)) - 1)
    chunks = []
    start = 0
    h = 0
    i = 0
    while i < n:
        h = ((h << 1) + _GEAR[token_ids[i] & 0x1FFFF]) & 0xFFFFFFFFFFFFFFFF
        length = i - start + 1
        if length >= min_size and (length >= max_size or (h & mask) == 0):
            chunks.append((start, i + 1))
            start = i + 1
            h = 0
        i += 1
    # trailing remainder is left unchunked (becomes suffix/prefix, recomputed)
    return chunks


@dataclass
class _StoredChunk:
    token_ids: list[int]
    src_pos: int
    # full-attention: per-layer (K, V); hybrid: HybridChunk
    payload: Any
    is_hybrid: bool


@dataclass
class ChunkReuseResult:
    prompt_cache: list
    cached_tokens: int
    remaining_tokens: list[int]
    chunks_used: int
    tokens_reused: int
    recompute_fraction: float


class ChunkReuseEngine:
    """Per-model chunk store + capture/assemble. Thread-safe (RLock)."""

    def __init__(self, model, model_name: str, config):
        # Strip oMLX's VLMModelAdapter once; all forward/introspection uses the
        # real mlx-lm model. Idempotent on already-unwrapped models.
        self.model = cr.unwrap_model(model)
        self.model_name = model_name
        self.config = config
        # content-defined chunk sizing (tokens)
        self.cdc_min = 128
        self.cdc_avg = 256
        self.cdc_max = 512
        self._store: "OrderedDict[bytes, _StoredChunk]" = OrderedDict()
        self._max_chunks = 4096
        self._lock = threading.RLock()

        self.arch = self._detect_arch(self.model)
        # cached rope modules for the full-attention path
        self._ropes = None
        if self.arch == "full":
            try:
                self._ropes = cr.get_layer_ropes(model)
            except Exception as e:  # noqa: BLE001
                logger.info("chunk reuse: full-attn rope probe failed (%s); disabling", e)
                self.arch = "unsupported"

        # metrics
        self.captures = 0
        self.assemble_hits = 0
        self.assemble_misses = 0
        self.tokens_reused_total = 0
        self.tokens_recomputed_total = 0

    # -- arch detection -----------------------------------------------------

    def _detect_arch(self, model) -> str:
        try:
            caches = model.make_cache() if hasattr(model, "make_cache") else None
        except Exception as e:  # noqa: BLE001
            logger.info("chunk reuse: make_cache() failed (%s); assuming full-attn", e)
            caches = None
        if caches is None:
            return "full"  # plain KVCache default
        kinds = sorted({type(c).__name__ for c in caches})
        if set(kinds) <= {"KVCache"}:
            return "full"
        if set(kinds) <= {"KVCache", "ArraysCache"}:
            # hybrid — confirm the qwen3_5 layout is introspectable
            try:
                crh.get_hybrid_layout(model)
                return "hybrid"
            except Exception as e:  # noqa: BLE001
                logger.info(
                    "chunk reuse: hybrid layout introspection failed (%s); "
                    "cache kinds=%s; disabling", e, kinds,
                )
                return "unsupported"
        logger.info("chunk reuse: unsupported cache kinds=%s; disabling", kinds)
        return "unsupported"  # sliding-window / rotating / mamba, etc.

    @property
    def supported(self) -> bool:
        return self.arch in ("full", "hybrid")

    # -- capture ------------------------------------------------------------

    def capture(self, prompt_ids: list[int], prompt_cache: Optional[list]) -> int:
        """Store fixed-size chunks of a prompt. Returns #newly-stored.

        Full-attention: slices KV straight from the finished ``prompt_cache``
        (offset must equal ``len(prompt_ids)``) — free, no extra forward.
        Hybrid: the normal prefill cache does not retain linear-layer inputs,
        so a one-time recording prefill is rerun over the novel prompt to grab
        them. Gated to capture-time and only for prompts with ≥1 new block.
        """
        if not self.supported:
            return 0
        n = len(prompt_ids)
        if n < self.config.min_chunk_tokens:
            return 0
        blocks = self._chunk_bounds(prompt_ids)
        if not blocks:
            return 0

        # Skip all work if every content chunk is already stored.
        with self._lock:
            novel = any(
                _hash_tokens(prompt_ids[s:e], self.model_name) not in self._store
                for s, e in blocks
            )
        if not novel:
            return 0

        recording = None
        if self.arch == "hybrid":
            try:
                recording = crh.capture_prefill(self.model, prompt_ids)
            except Exception as e:  # noqa: BLE001
                logger.debug("hybrid recording prefill failed: %s", e)
                return 0

        stored = 0
        with self._lock:
            for s, e in blocks:
                key = _hash_tokens(prompt_ids[s:e], self.model_name)
                if key in self._store:
                    self._store.move_to_end(key)
                    continue
                try:
                    if self.arch == "hybrid":
                        chunk = crh.extract_hybrid_chunk(recording, prompt_ids, s, e)
                        self._store[key] = _StoredChunk(prompt_ids[s:e], s, chunk, True)
                    else:
                        kv = cr.extract_chunk_kv(prompt_cache, s, e)
                        self._store[key] = _StoredChunk(prompt_ids[s:e], s, kv, False)
                    stored += 1
                except Exception as e:  # noqa: BLE001
                    logger.debug("chunk capture failed at [%d:%d]: %s", s, e, e)
            while len(self._store) > self._max_chunks:
                self._store.popitem(last=False)
        self.captures += stored
        return stored

    def _chunk_bounds(self, token_ids: list[int]) -> list[tuple[int, int]]:
        return content_defined_chunks(
            token_ids, min_size=self.cdc_min, avg_size=self.cdc_avg,
            max_size=self.cdc_max,
        )

    # -- assemble -----------------------------------------------------------

    def _find_chunks(self, prompt_ids: list[int]):
        """Locate stored chunks in a prompt via content-defined boundaries.

        Chunk the prompt with the same CDC used at capture, then keep the
        boundaries whose content hash is in the store. Because CDC boundaries
        are content-determined, a file re-read under a different prefix chunks
        the same way and its hashes match. Returns ordered
        (start, end, _StoredChunk); non-matching spans recompute fresh.
        """
        found = []
        with self._lock:
            for s, e in self._chunk_bounds(prompt_ids):
                key = _hash_tokens(prompt_ids[s:e], self.model_name)
                sc = self._store.get(key)
                if sc is not None:
                    self._store.move_to_end(key)
                    found.append((s, e, sc))
        return found

    def assemble(self, prompt_ids: list[int]) -> Optional[ChunkReuseResult]:
        if not self.supported:
            return None
        found = self._find_chunks(prompt_ids)
        if not found:
            self.assemble_misses += 1
            return None

        # Everything after the last reused chunk is the suffix oMLX prefills.
        last_end = found[-1][1]
        suffix = prompt_ids[last_end:]

        # Build prefix + interleave segments for blended prefill.
        prefix = list(prompt_ids[: found[0][0]])
        chunks = [sc for (_, _, sc) in found]
        interleave = []
        for idx in range(len(found) - 1):
            gap_s = found[idx][1]
            gap_e = found[idx + 1][0]
            interleave.append(list(prompt_ids[gap_s:gap_e]))

        mode = self.config.recompute_mode
        try:
            if self.arch == "hybrid":
                hchunks = [sc.payload for sc in chunks]
                # hybrid path has no interleave arg yet; require contiguous chunks
                if any(seg for seg in interleave):
                    self.assemble_misses += 1
                    return None
                caches, _, stats = crh.hybrid_blended_prefill(
                    self.model, prefix, hchunks, [],  # empty suffix; we prefill it via oMLX
                    mode=mode, edge_k=self.config.edge_tokens,
                    deviation_ratio=self.config.deviation_ratio,
                    dev_block=self.config.dev_block,
                )
            else:
                kchunks = [
                    cr.ChunkReuse(tokens=sc.token_ids, kv=sc.payload, src_pos=sc.src_pos)
                    for sc in chunks
                ]
                caches, _, stats = cr.blended_prefill(
                    self.model, prefix, kchunks, [],
                    ropes=self._ropes, interleave=interleave or None,
                    mode=mode, edge_k=self.config.edge_tokens,
                    deviation_ratio=self.config.deviation_ratio,
                    dev_block=self.config.dev_block,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("chunk assemble failed (%s); falling back to full prefill", e)
            self.assemble_misses += 1
            return None

        cached_tokens = last_end
        self.assemble_hits += 1
        self.tokens_reused_total += stats.reused_tokens
        self.tokens_recomputed_total += stats.recomputed_tokens
        return ChunkReuseResult(
            prompt_cache=caches,
            cached_tokens=cached_tokens,
            remaining_tokens=suffix,
            chunks_used=len(chunks),
            tokens_reused=stats.reused_tokens,
            recompute_fraction=stats.recompute_fraction,
        )

    # -- metrics ------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "arch": self.arch,
            "stored_chunks": len(self._store),
            "captures": self.captures,
            "assemble_hits": self.assemble_hits,
            "assemble_misses": self.assemble_misses,
            "tokens_reused_total": self.tokens_reused_total,
            "tokens_recomputed_total": self.tokens_recomputed_total,
        }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

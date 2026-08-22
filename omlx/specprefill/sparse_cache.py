# SPDX-License-Identifier: Apache-2.0
"""Reuse of SpecPrefill sparse conversation caches across turns.

SpecPrefill's saving comes from *not* computing ~80% of the target KV, so the
cache it leaves behind holds N' physical rows for M logical tokens. Ordinary
prefix matching hashes contiguous runs of token IDs against the KV for those
runs, and that correspondence is exactly what a sparse prefill destroys -- which
is why, before this module, a SpecPrefill request stored nothing at all and
every turn of an agent session re-prefilled the whole conversation.

What makes reuse possible is that ``sparse_prefill`` writes each selected key at
its ORIGINAL RoPE angle. RoPE is relative, so a sparse cache is a valid prefix
for a *growing* conversation: the next turn appends at logical position M while
the cache offset sits at N', and an ``_OffsetAdjustedRoPE(M - N')`` reconciles
the two. ``sparse_prefill`` already accepts ``position_offset`` and reads
``cache_start`` from the live cache; nothing called it that way until now.

Scope and the honest limits:
  - Entries are found through an in-memory index of stored logical lengths.
    A server restart loses the index, so the first turn after a restart
    re-prefills and then the session is warm again. That is the pre-existing
    behaviour, not a regression, and it is why no index is persisted yet.
  - History is frozen. Turn N reuses the selection each earlier turn made
    against the question that was live when it arrived; only the new tokens are
    scored fresh. That is a real quality trade against re-scoring everything
    every turn, and it is the trade this module exists to make.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# How many candidate boundaries a single lookup may probe. Each probe is a
# chain hash over a prompt prefix, so this bounds lookup cost on long prompts.
MAX_LOOKUP_PROBES = 8

# Logical lengths remembered per model. Bounded so a long-running server with
# many distinct sessions cannot grow this without limit.
MAX_INDEXED_LENGTHS = 512

# Below this, a sparse entry is not worth a block: the request would not have
# entered SpecPrefill at this size anyway.
MIN_SPARSE_PREFIX_TOKENS = 512


@dataclass(frozen=True)
class SparsePrefixHit:
    """A restored sparse conversation prefix."""

    cache: list[Any]
    logical_tokens: int
    physical_rows: int

    @property
    def position_offset(self) -> int:
        """Logical position the next token appends at."""
        return self.logical_tokens

    @property
    def rope_adjustment(self) -> int:
        """Constant added to cache offsets to recover true RoPE positions."""
        return self.logical_tokens - self.physical_rows


class SparsePrefixIndex:
    """Logical lengths at which sparse prefixes have been stored, per model.

    Lookup needs to answer "where might an earlier turn have ended?", and a
    sparse entry is keyed on the exact logical token sequence, so the length is
    the one thing a later request cannot derive from its own prompt.
    """

    def __init__(self, max_lengths: int = MAX_INDEXED_LENGTHS) -> None:
        self._max_lengths = max_lengths
        self._lengths: OrderedDict[str, OrderedDict[int, None]] = OrderedDict()
        self._lock = threading.Lock()

    def record(self, model_name: str, logical_length: int) -> None:
        with self._lock:
            lengths = self._lengths.setdefault(model_name, OrderedDict())
            lengths.pop(logical_length, None)
            lengths[logical_length] = None
            while len(lengths) > self._max_lengths:
                lengths.popitem(last=False)

    def candidates(self, model_name: str, prompt_length: int) -> list[int]:
        """Plausible stored lengths for this prompt, longest first.

        Longest-first matters: a longer stored prefix means fewer tokens left
        for the current turn to prefill.
        """
        with self._lock:
            lengths = self._lengths.get(model_name)
            if not lengths:
                return []
            usable = [n for n in lengths if 0 < n <= prompt_length]
        usable.sort(reverse=True)
        return usable[:MAX_LOOKUP_PROBES]

    def clear(self, model_name: str | None = None) -> None:
        with self._lock:
            if model_name is None:
                self._lengths.clear()
            else:
                self._lengths.pop(model_name, None)


def find_sparse_prefix(
    *,
    prefix_cache: Any,
    index: SparsePrefixIndex,
    model_name: str,
    request_id: str,
    prompt_tokens: list[int],
    promote_to_hot_cache: bool = True,
) -> SparsePrefixHit | None:
    """Restore the longest stored sparse prefix of ``prompt_tokens``.

    Returns None when nothing matches, which leaves the caller on exactly the
    path it took before sparse reuse existed.
    """
    if prefix_cache is None or not prompt_tokens:
        return None
    restore = getattr(prefix_cache, "restore_sparse_prefix", None)
    if restore is None:
        return None

    for logical_length in index.candidates(model_name, len(prompt_tokens)):
        if logical_length < MIN_SPARSE_PREFIX_TOKENS:
            continue
        # A stored prefix is only usable if the current prompt still begins
        # with exactly those tokens; the hash check below enforces that, but
        # bail early when the whole prompt is shorter.
        if logical_length > len(prompt_tokens):
            continue
        try:
            restored = restore(
                f"{request_id}:specprefill-sparse-restore",
                prompt_tokens[:logical_length],
                promote_to_hot_cache=promote_to_hot_cache,
            )
        except Exception as error:  # pragma: no cover - defensive
            logger.debug("SpecPrefill: sparse prefix restore failed: %s", error)
            return None
        if restored is None:
            continue
        cache, logical_tokens = restored
        physical_rows = _physical_rows(cache)
        if physical_rows <= 0 or physical_rows > logical_tokens:
            # A payload that claims more rows than logical tokens is not a
            # sparse prefix; refuse rather than install a bogus RoPE offset.
            logger.warning(
                "SpecPrefill: rejecting sparse prefix with %d rows for %d "
                "logical tokens",
                physical_rows,
                logical_tokens,
            )
            continue
        logger.info(
            "SpecPrefill: restored sparse prefix, %d logical tokens over %d "
            "physical KV rows (rope adjustment %d)",
            logical_tokens,
            physical_rows,
            logical_tokens - physical_rows,
        )
        return SparsePrefixHit(
            cache=cache,
            logical_tokens=logical_tokens,
            physical_rows=physical_rows,
        )
    return None


def _physical_rows(cache: list[Any]) -> int:
    """Rows actually present in the restored sliceable layers.

    Hybrid models mix fixed-size recurrent layers (no sequence dimension) with
    full-attention KV layers; only the latter carry the row count that the RoPE
    adjustment is computed from.
    """
    for layer in cache or []:
        offset = getattr(layer, "offset", None)
        if isinstance(offset, int) and offset > 0:
            return offset
    return 0


def should_store_sparse_prefix(logical_tokens: int, physical_rows: int) -> bool:
    """Whether a finished sparse request is worth persisting.

    Refuses the degenerate case where nothing was actually dropped: such a
    cache is dense, belongs in the ordinary prefix cache, and storing it here
    would only hide it behind the sparse domain.
    """
    if logical_tokens < MIN_SPARSE_PREFIX_TOKENS:
        return False
    if physical_rows <= 0 or physical_rows >= logical_tokens:
        return False
    return True

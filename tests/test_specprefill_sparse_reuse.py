# SPDX-License-Identifier: Apache-2.0
"""Sparse SpecPrefill conversation-prefix reuse.

The property under test throughout is that a sparse cache -- N' physical KV
rows standing in for M logical tokens -- can be stored, found again, and
extended, while remaining unreachable from every path that would read it as a
dense cache.
"""

from __future__ import annotations

import pathlib
import tempfile

import mlx.core as mx
import pytest

from omlx.specprefill.sparse_cache import (
    MIN_SPARSE_PREFIX_TOKENS,
    SparsePrefixHit,
    SparsePrefixIndex,
    find_sparse_prefix,
    should_store_sparse_prefix,
)
from tests.test_prefix_cache import TestBlockAwarePrefixCache as _Fixtures


def _sparse_cache_data(rows: int):
    keys = mx.arange(rows, dtype=mx.float32).reshape(1, 1, rows, 1)
    values = (keys + 100).astype(mx.float32)
    return [
        {
            "state": (keys, values),
            "meta_state": (rows,),
            "class_name": "KVCache",
            "cache_type": "KVCache",
        }
    ], keys


@pytest.fixture
def prefix_cache():
    directory = pathlib.Path(tempfile.mkdtemp())
    cache, _paged, ssd = _Fixtures._make_ssd_prefix_cache(directory)
    yield cache
    ssd.close()


class TestSparsePrefixStorage:
    def test_logical_and_physical_lengths_may_differ(self, prefix_cache):
        """The whole point: 40 logical tokens stored over 9 KV rows."""
        logical = list(range(40))
        cache_data, keys = _sparse_cache_data(9)

        assert prefix_cache.store_sparse_prefix("store", logical, cache_data)

        restored = prefix_cache.restore_sparse_prefix(
            "restore", logical, promote_to_hot_cache=False
        )
        assert restored is not None
        layers, logical_tokens = restored
        assert logical_tokens == 40
        assert layers[0].offset == 9
        assert mx.array_equal(layers[0].state[0], keys)

    def test_a_dense_cache_is_refused(self, prefix_cache):
        """Nothing was dropped, so this belongs in the ordinary domain."""
        assert should_store_sparse_prefix(1024, 205) is True
        assert should_store_sparse_prefix(1024, 1024) is False
        assert should_store_sparse_prefix(1024, 2048) is False
        assert should_store_sparse_prefix(MIN_SPARSE_PREFIX_TOKENS - 1, 10) is False

    def test_a_different_logical_sequence_misses(self, prefix_cache):
        logical = list(range(40))
        cache_data, _ = _sparse_cache_data(9)
        prefix_cache.store_sparse_prefix("store", logical, cache_data)

        assert (
            prefix_cache.restore_sparse_prefix(
                "miss", logical[:-1], promote_to_hot_cache=False
            )
            is None
        )
        assert (
            prefix_cache.restore_sparse_prefix(
                "miss2", [*logical[:-1], 999], promote_to_hot_cache=False
            )
            is None
        )


class TestDomainSeparation:
    """A sparse entry is only correct when replayed with its RoPE offset.

    Every one of these is a route by which a request WITHOUT that offset could
    otherwise read N' rows as though they were M tokens.
    """

    def test_ordinary_prefix_matching_cannot_see_a_sparse_entry(self, prefix_cache):
        logical = list(range(40))
        cache_data, _ = _sparse_cache_data(9)
        prefix_cache.store_sparse_prefix("store", logical, cache_data)

        table, remaining = prefix_cache.fetch_cache("ordinary", [*logical, 40])
        assert table is None or table.num_tokens == 0
        assert remaining == [*logical, 40]

    def test_static_exact_prefix_domain_cannot_see_a_sparse_entry(self, prefix_cache):
        logical = list(range(40))
        cache_data, _ = _sparse_cache_data(9)
        prefix_cache.store_sparse_prefix("store", logical, cache_data)

        assert prefix_cache.fetch_exact_prefix("exact", logical) is None

    def test_sparse_domain_cannot_see_a_static_exact_entry(self, prefix_cache):
        tokens = list(range(7))
        cache_data, _ = _sparse_cache_data(7)
        assert prefix_cache.store_exact_prefix("static", tokens, cache_data) is not None

        assert (
            prefix_cache.restore_sparse_prefix(
                "sparse", tokens, promote_to_hot_cache=False
            )
            is None
        )


class TestSparsePrefixIndex:
    def test_candidates_are_longest_first_and_bounded_by_prompt(self):
        index = SparsePrefixIndex()
        for length in (1024, 4096, 2048):
            index.record("m", length)

        assert index.candidates("m", 5000) == [4096, 2048, 1024]
        assert index.candidates("m", 2000) == [1024]
        assert index.candidates("m", 100) == []
        assert index.candidates("other-model", 5000) == []

    def test_index_is_bounded(self):
        index = SparsePrefixIndex(max_lengths=3)
        for length in (1000, 2000, 3000, 4000):
            index.record("m", length)
        assert index.candidates("m", 10000) == [4000, 3000, 2000]

    def test_clear_is_per_model(self):
        index = SparsePrefixIndex()
        index.record("a", 1024)
        index.record("b", 1024)
        index.clear("a")
        assert index.candidates("a", 5000) == []
        assert index.candidates("b", 5000) == [1024]


class TestRopeArithmetic:
    def test_offset_and_adjustment(self):
        hit = SparsePrefixHit(cache=[], logical_tokens=1024, physical_rows=205)
        # Next token appends at logical position M...
        assert hit.position_offset == 1024
        # ...while the cache sits at N', so queries need M - N' added.
        assert hit.rope_adjustment == 819


class TestFindSparsePrefix:
    def test_longest_stored_prefix_wins(self, prefix_cache):
        index = SparsePrefixIndex()
        prompt = list(range(4000))
        for length in (1024, 2048):
            cache_data, _ = _sparse_cache_data(length // 5)
            assert prefix_cache.store_sparse_prefix(
                f"s{length}", prompt[:length], cache_data
            )
            index.record("model", length)

        hit = find_sparse_prefix(
            prefix_cache=prefix_cache,
            index=index,
            model_name="model",
            request_id="r",
            prompt_tokens=prompt,
            promote_to_hot_cache=False,
        )
        assert hit is not None
        assert hit.logical_tokens == 2048
        assert hit.physical_rows == 2048 // 5
        assert hit.rope_adjustment == 2048 - 2048 // 5

    def test_no_index_entry_means_no_lookup(self, prefix_cache):
        prompt = list(range(4000))
        cache_data, _ = _sparse_cache_data(205)
        prefix_cache.store_sparse_prefix("s", prompt[:1024], cache_data)

        # Stored, but the index never learned the length, so it is not found.
        # This is the documented restart behaviour, asserted so it stays a
        # known miss rather than becoming a silent wrong-length match.
        hit = find_sparse_prefix(
            prefix_cache=prefix_cache,
            index=SparsePrefixIndex(),
            model_name="model",
            request_id="r",
            prompt_tokens=prompt,
            promote_to_hot_cache=False,
        )
        assert hit is None

    def test_a_payload_with_more_rows_than_tokens_is_rejected(self, prefix_cache):
        """Fail closed rather than install a negative RoPE adjustment."""
        index = SparsePrefixIndex()
        prompt = list(range(4000))
        cache_data, _ = _sparse_cache_data(2000)
        prefix_cache.store_sparse_prefix("s", prompt[:1024], cache_data)
        index.record("model", 1024)

        hit = find_sparse_prefix(
            prefix_cache=prefix_cache,
            index=index,
            model_name="model",
            request_id="r",
            prompt_tokens=prompt,
            promote_to_hot_cache=False,
        )
        assert hit is None

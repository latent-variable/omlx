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
    sparse_variant_key,
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

        assert prefix_cache.store_sparse_prefix("store", logical, cache_data)[0]

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
                f"s{length}", prompt[:length], cache_data,
                variant_key="keep=0.2;draft=d",
            )[0]
            index.record("model", length)

        hit = find_sparse_prefix(
            prefix_cache=prefix_cache,
            index=index,
            model_name="model",
            request_id="r",
            prompt_tokens=prompt,
            variant_key="keep=0.2;draft=d",
            promote_to_hot_cache=False,
        )
        assert hit is not None
        assert hit.logical_tokens == 2048
        assert hit.physical_rows == 2048 // 5
        assert hit.rope_adjustment == 2048 - 2048 // 5

    def test_no_index_entry_means_no_lookup(self, prefix_cache):
        prompt = list(range(4000))
        cache_data, _ = _sparse_cache_data(205)
        prefix_cache.store_sparse_prefix(
            "s", prompt[:1024], cache_data, variant_key="keep=0.2;draft=d"
        )

        # Stored, but the index never learned the length, so it is not found.
        # This is the documented restart behaviour, asserted so it stays a
        # known miss rather than becoming a silent wrong-length match.
        hit = find_sparse_prefix(
            prefix_cache=prefix_cache,
            index=SparsePrefixIndex(),
            model_name="model",
            request_id="r",
            prompt_tokens=prompt,
            variant_key="keep=0.2;draft=d",
            promote_to_hot_cache=False,
        )
        assert hit is None

    def test_a_payload_with_more_rows_than_tokens_is_rejected(self, prefix_cache):
        """Fail closed rather than install a negative RoPE adjustment."""
        index = SparsePrefixIndex()
        prompt = list(range(4000))
        cache_data, _ = _sparse_cache_data(2000)
        prefix_cache.store_sparse_prefix(
            "s", prompt[:1024], cache_data, variant_key="keep=0.2;draft=d"
        )
        index.record("model", 1024)

        hit = find_sparse_prefix(
            prefix_cache=prefix_cache,
            index=index,
            model_name="model",
            request_id="r",
            prompt_tokens=prompt,
            variant_key="keep=0.2;draft=d",
            promote_to_hot_cache=False,
        )
        assert hit is None


class TestSchedulerGuards:
    """The paths by which a sparse cache could escape into ordinary reuse.

    Each of these is a way a request WITHOUT the matching RoPE offset could
    end up reading N' rows as though they were M tokens, which is silent
    corruption rather than a visible failure.
    """

    @staticmethod
    def _scheduler():
        from omlx.scheduler import Scheduler, SchedulerConfig

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.config = SchedulerConfig(paged_cache_block_size=4)
        return scheduler

    @staticmethod
    def _request(**overrides):
        import types

        base = dict(
            request_id="r",
            prompt_token_ids=list(range(16)),
            specprefill_indices=None,
            specprefill_rope_offset=None,
        )
        base.update(overrides)
        return types.SimpleNamespace(**base)

    def test_boundary_store_refuses_a_restored_sparse_request(self):
        """A request that only RESTORED a prefix selected no indices...

        ...but its cache is every bit as sparse, so the boundary-store
        fallback must refuse it just the same.
        """
        scheduler = self._scheduler()
        scheduler.block_aware_cache = object()

        restored_only = self._request(specprefill_rope_offset=819)
        assert (
            scheduler._prepare_prompt_boundary_cache_store("r", restored_only, 0)
            is None
        )

        freshly_selected = self._request(specprefill_indices=[1, 2, 3])
        assert (
            scheduler._prepare_prompt_boundary_cache_store("r", freshly_selected, 0)
            is None
        )

    def test_restore_is_inert_without_a_draft_model(self):
        scheduler = self._scheduler()
        scheduler.block_aware_cache = object()
        request = self._request(_specprefill_enabled=True)
        # No _specprefill_draft_model attribute at all.
        scheduler._try_sparse_prefix_restore(request)
        assert request.specprefill_rope_offset is None

    def test_restore_is_inert_when_disabled_by_config(self):
        from omlx.scheduler import SchedulerConfig

        scheduler = self._scheduler()
        scheduler.config = SchedulerConfig(specprefill_sparse_reuse=False)
        scheduler.block_aware_cache = object()
        scheduler._specprefill_draft_model = object()
        scheduler._sparse_prefix_index = SparsePrefixIndex()
        request = self._request(_specprefill_enabled=True, vlm_inputs_embeds=None)
        scheduler._try_sparse_prefix_restore(request)
        assert request.specprefill_rope_offset is None

    def test_restore_leaves_image_requests_on_the_dense_path(self):
        """SpecPrefill already declines VLM embeddings; reuse must not re-open it."""
        scheduler = self._scheduler()
        scheduler.block_aware_cache = object()
        scheduler._specprefill_draft_model = object()
        scheduler._sparse_prefix_index = SparsePrefixIndex()
        request = self._request(
            _specprefill_enabled=True, vlm_inputs_embeds=object()
        )
        scheduler._try_sparse_prefix_restore(request)
        assert request.specprefill_rope_offset is None

    def test_abandoning_a_prefix_returns_the_request_to_a_cold_start(self):
        scheduler = self._scheduler()
        scheduler.model = None
        scheduler._specprefill_active_request_id = "r"
        request = self._request(
            specprefill_rope_offset=819,
            specprefill_sparse_logical_tokens=1024,
            specprefill_sparse_physical_rows=205,
            prompt_cache=["something"],
            cached_tokens=1024,
            remaining_tokens=[],
        )
        scheduler._abandon_sparse_prefix(request)

        assert request.specprefill_rope_offset is None
        assert request.specprefill_sparse_logical_tokens == 0
        assert request.specprefill_sparse_physical_rows == 0
        assert request.prompt_cache is None
        assert request.cached_tokens == 0
        assert request.remaining_tokens == request.prompt_token_ids
        assert scheduler._specprefill_active_request_id is None


class TestStoreRestoreWiring:
    """The two scheduler halves must actually meet.

    Every other test exercises one side. This drives the real store method and
    then the real restore method, through a real prefix cache, the way two
    consecutive turns of one conversation would.
    """

    @staticmethod
    def _scheduler(prefix_cache):
        from omlx.scheduler import Scheduler, SchedulerConfig

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.config = SchedulerConfig(model_name="test-model")
        scheduler.block_aware_cache = prefix_cache
        scheduler._specprefill_draft_model = object()
        scheduler._sparse_prefix_index = SparsePrefixIndex()
        scheduler._bypass_hot_cache_under_pressure = lambda: True
        scheduler._release_paged_cache_for_request = lambda _rid: None
        return scheduler

    def test_a_stored_turn_is_found_by_the_next_turn(self, prefix_cache):
        import types

        rows = 205
        cache_data, keys = _sparse_cache_data(rows)
        scheduler = self._scheduler(prefix_cache)
        scheduler._extract_cache_states = lambda _cache: (cache_data, None)

        # --- turn one: 1025 prompt tokens, stored at the prompt boundary ---
        turn_one_prompt = list(range(1025))
        first = types.SimpleNamespace(
            request_id="turn-1",
            prompt_token_ids=turn_one_prompt,
            specprefill_indices=[1, 2, 3],
            specprefill_rope_offset=None,
        )
        scheduler._store_sparse_prefix_after_prefill(first, object())

        # Stored for prompt[:-1] — the kickoff token is BatchGenerator's.
        assert scheduler._sparse_prefix_index.candidates("test-model", 5000) == [1024]

        # --- turn two: the same prompt plus an answer and a new question ---
        second = types.SimpleNamespace(
            request_id="turn-2",
            prompt_token_ids=turn_one_prompt + list(range(2000, 2100)),
            cached_tokens=0,
            remaining_tokens=None,
            prompt_cache=None,
            shared_prefix_blocks=0,
            vlm_inputs_embeds=None,
            specprefill_indices=None,
            specprefill_rope_offset=None,
            specprefill_sparse_logical_tokens=0,
            specprefill_sparse_physical_rows=0,
            _specprefill_enabled=True,
        )
        scheduler._try_sparse_prefix_restore(second)

        assert second.specprefill_sparse_logical_tokens == 1024
        assert second.specprefill_sparse_physical_rows == rows
        # The offset feeding _OffsetAdjustedRoPE and specprefill_position_offset.
        assert second.specprefill_rope_offset == 1024 - rows
        assert second.cached_tokens == 1024
        # Only the genuinely new tokens are left to prefill/score.
        assert second.remaining_tokens == turn_one_prompt[1024:] + list(
            range(2000, 2100)
        )
        assert second.prompt_cache is not None
        assert mx.array_equal(second.prompt_cache[0].state[0], keys)

    def test_a_longer_dense_hit_beats_the_sparse_one(self, prefix_cache):
        """An ordinary hit is exact; it wins whenever it covers as much."""
        import types

        cache_data, _ = _sparse_cache_data(205)
        scheduler = self._scheduler(prefix_cache)
        scheduler._extract_cache_states = lambda _cache: (cache_data, None)

        prompt = list(range(4000))
        first = types.SimpleNamespace(
            request_id="t1",
            prompt_token_ids=prompt[:1025],
            specprefill_indices=[1],
            specprefill_rope_offset=None,
        )
        scheduler._store_sparse_prefix_after_prefill(first, object())

        second = types.SimpleNamespace(
            request_id="t2",
            prompt_token_ids=prompt,
            cached_tokens=2048,  # a dense hit already covers more
            remaining_tokens=prompt[2048:],
            prompt_cache=["dense"],
            shared_prefix_blocks=3,
            vlm_inputs_embeds=None,
            specprefill_indices=None,
            specprefill_rope_offset=None,
            specprefill_sparse_logical_tokens=0,
            specprefill_sparse_physical_rows=0,
            _specprefill_enabled=True,
        )
        scheduler._try_sparse_prefix_restore(second)

        assert second.specprefill_rope_offset is None
        assert second.cached_tokens == 2048
        assert second.prompt_cache == ["dense"]


class TestStaleRopeOwner:
    """A stranded RoPE wrapper must not stall the scheduler.

    `_specprefill_active_request_id` gates admission for EVERY request. If a
    prefill exits between installing the wrapper and reaching decode — the
    eviction pause requeues the request, a memory rejection drops it entirely —
    nothing is left running to clear it, and without reconciliation the
    scheduler defers all work permanently.
    """

    @staticmethod
    def _scheduler(running_ids=()):
        from omlx.scheduler import Scheduler, SchedulerConfig

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.config = SchedulerConfig()
        scheduler.running = {rid: object() for rid in running_ids}
        scheduler.model = None
        return scheduler

    def test_owner_that_never_reached_decode_is_cleared(self):
        """The eviction-pause and memory-rejection case."""
        scheduler = self._scheduler(running_ids=())
        scheduler._specprefill_active_request_id = "evicted-before-decode"

        assert scheduler._reconcile_stale_specprefill_owner() is True
        assert scheduler._specprefill_active_request_id is None

    def test_a_live_owner_is_left_alone(self):
        """The control: a genuinely decoding request keeps its wrapper.

        Without this half, "clear it whenever" would corrupt the remaining
        decode of the request that legitimately owns the offset (#766).
        """
        scheduler = self._scheduler(running_ids=("decoding-now",))
        scheduler._specprefill_active_request_id = "decoding-now"

        assert scheduler._reconcile_stale_specprefill_owner() is False
        assert scheduler._specprefill_active_request_id == "decoding-now"

    def test_no_owner_is_a_no_op(self):
        scheduler = self._scheduler(running_ids=("someone",))
        scheduler._specprefill_active_request_id = None
        assert scheduler._reconcile_stale_specprefill_owner() is False

    def test_installing_the_wrapper_does_not_claim_ownership(self):
        """Arming happens after prefill succeeds, not when the wrapper goes on.

        This is the actual fix for the stranding: everything between
        installing the wrapper and reaching decode can still fail out.
        """
        import types

        from omlx.patches import specprefill as sp

        scheduler = self._scheduler()
        scheduler._specprefill_active_request_id = None

        layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace(rope=object()))]
        scheduler.model = types.SimpleNamespace(layers=layers)

        request = types.SimpleNamespace(
            request_id="r",
            specprefill_rope_offset=819,
            specprefill_sparse_logical_tokens=1024,
            specprefill_sparse_physical_rows=205,
        )
        scheduler._install_sparse_rope_offset(request)

        assert isinstance(layers[0].self_attn.rope, sp._OffsetAdjustedRoPE)
        assert scheduler._specprefill_active_request_id is None


class TestSelectionPolicyIsolation:
    """Two keep rates over the same tokens are NOT interchangeable caches.

    `specprefill_keep_pct` and the drafter are both per-request settings in the
    public API, and each selects a different subset. Without the policy in the
    key, whichever variant landed first would silently answer every later
    request and override the quality/cost setting it explicitly asked for.
    """

    def test_a_different_keep_rate_does_not_reuse_the_cache(self, prefix_cache):
        tokens = list(range(1024))
        cache_data, _ = _sparse_cache_data(205)
        sparse = sparse_variant_key(
            keep_pct=0.2, draft_model_name="qwen-2b", default_keep_pct=0.2
        )
        denser = sparse_variant_key(
            keep_pct=0.5, draft_model_name="qwen-2b", default_keep_pct=0.2
        )
        assert sparse != denser

        assert prefix_cache.store_sparse_prefix(
            "s", tokens, cache_data, variant_key=sparse
        )[0]
        assert (
            prefix_cache.restore_sparse_prefix(
                "same", tokens, variant_key=sparse, promote_to_hot_cache=False
            )
            is not None
        )
        assert (
            prefix_cache.restore_sparse_prefix(
                "denser", tokens, variant_key=denser, promote_to_hot_cache=False
            )
            is None
        )

    def test_a_different_drafter_does_not_reuse_the_cache(self, prefix_cache):
        tokens = list(range(1024))
        cache_data, _ = _sparse_cache_data(205)
        one = sparse_variant_key(
            keep_pct=0.2, draft_model_name="qwen-2b", default_keep_pct=0.2
        )
        other = sparse_variant_key(
            keep_pct=0.2, draft_model_name="qwen-4b", default_keep_pct=0.2
        )
        prefix_cache.store_sparse_prefix("s", tokens, cache_data, variant_key=one)

        assert (
            prefix_cache.restore_sparse_prefix(
                "other", tokens, variant_key=other, promote_to_hot_cache=False
            )
            is None
        )

    def test_an_unset_keep_rate_resolves_to_the_default(self):
        assert sparse_variant_key(
            keep_pct=None, draft_model_name="d", default_keep_pct=0.2
        ) == sparse_variant_key(
            keep_pct=0.2, draft_model_name="d", default_keep_pct=0.2
        )


class TestExactLengthHitsAreRefused:
    """A hit covering the WHOLE prompt leaves no generation-kickoff token.

    The scheduler would send the prompt's final token anyway, over a cache that
    already contains that position, and the next prompt-boundary store would
    label an over-long cache with a shorter logical sequence. A hybrid's
    recurrent state cannot be trimmed back one token, so there is no repair.
    """

    def test_a_prefix_covering_the_entire_prompt_is_skipped(self, prefix_cache):
        index = SparsePrefixIndex()
        prompt = list(range(1024))
        cache_data, _ = _sparse_cache_data(205)
        key = "keep=0.2;draft=d"
        prefix_cache.store_sparse_prefix("s", prompt, cache_data, variant_key=key)
        index.record("model", len(prompt))

        assert (
            find_sparse_prefix(
                prefix_cache=prefix_cache,
                index=index,
                model_name="model",
                request_id="r",
                prompt_tokens=prompt,
                variant_key=key,
                promote_to_hot_cache=False,
            )
            is None
        )

        # One more token in the prompt and it is usable again.
        assert (
            find_sparse_prefix(
                prefix_cache=prefix_cache,
                index=index,
                model_name="model",
                request_id="r",
                prompt_tokens=prompt + [9999],
                variant_key=key,
                promote_to_hot_cache=False,
            )
            is not None
        )


class TestSupersededEntriesAreReleased:
    """Each turn stores the whole cumulative prefix; the shorter ones it
    contains must not stay resident crowding out unrelated blocks."""

    def test_storing_a_longer_prefix_releases_the_shorter_one(self, prefix_cache):
        key = "keep=0.2;draft=d"
        tokens = list(range(4096))
        short, _ = _sparse_cache_data(205)
        long_, _ = _sparse_cache_data(410)

        assert prefix_cache.store_sparse_prefix(
            "turn1", tokens[:1024], short, variant_key=key
        )[0]
        assert prefix_cache.restore_sparse_prefix(
            "probe", tokens[:1024], variant_key=key, promote_to_hot_cache=False
        ) is not None

        stored, superseded = prefix_cache.store_sparse_prefix(
            "turn2", tokens[:2048], long_, variant_key=key, supersedes=[1024]
        )
        assert stored
        assert superseded == [1024]
        assert prefix_cache.get_stats().sparse_prefix_superseded == 1
        # The outcome, not just the counter: the shorter entry is really gone
        # from the durable tier, so it cannot be restored any more.
        assert prefix_cache.restore_sparse_prefix(
            "gone", tokens[:1024], variant_key=key, promote_to_hot_cache=False
        ) is None
        # And the longer one, which contains it, is what a later turn finds.
        assert prefix_cache.restore_sparse_prefix(
            "probe2", tokens[:2048], variant_key=key, promote_to_hot_cache=False
        ) is not None

    def test_a_longer_length_is_never_treated_as_superseded(self, prefix_cache):
        key = "keep=0.2;draft=d"
        tokens = list(range(4096))
        data, _ = _sparse_cache_data(205)
        prefix_cache.store_sparse_prefix(
            "t", tokens[:1024], data, variant_key=key, supersedes=[2048, 1024]
        )
        # 2048 > 1024, so it is not contained and must be left alone.
        assert prefix_cache.get_stats().sparse_prefix_superseded == 0


class TestConversationsDoNotEvictEachOther:
    """Candidate lengths are shared across conversations; entries are not.

    A server runs many sessions at once, so "some conversation stored 1024
    tokens" says nothing about whose. Superseding by LENGTH alone would let one
    conversation strand another's entry on disk — still stored, no longer
    discoverable — sending it back to full re-prefill every turn, which is the
    exact failure this feature exists to remove.
    """

    def test_storing_one_conversation_leaves_another_intact(self, prefix_cache):
        key = "keep=0.2;draft=d"
        alice = list(range(0, 4096))
        bob = list(range(100000, 104096))
        data_a, _ = _sparse_cache_data(205)
        data_b, _ = _sparse_cache_data(410)

        assert prefix_cache.store_sparse_prefix(
            "alice", alice[:1024], data_a, variant_key=key
        )[0]

        # Bob stores a LONGER prefix, and 1024 is a candidate length only
        # because Alice put it there.
        stored, superseded = prefix_cache.store_sparse_prefix(
            "bob", bob[:2048], data_b, variant_key=key, supersedes=[1024]
        )
        assert stored
        assert superseded == [], "Alice's entry is not a prefix of Bob's"

        # Alice is still there, and still findable.
        assert prefix_cache.restore_sparse_prefix(
            "alice-again", alice[:1024], variant_key=key, promote_to_hot_cache=False
        ) is not None
        assert prefix_cache.restore_sparse_prefix(
            "bob-again", bob[:2048], variant_key=key, promote_to_hot_cache=False
        ) is not None

    def test_the_index_only_forgets_confirmed_deletions(self):
        """The scheduler half: forgetting is driven by what was deleted."""
        index = SparsePrefixIndex()
        index.record("m", 1024)   # Alice
        index.record("m", 2048)   # Bob

        # Bob's store reports it superseded nothing (Alice did not match).
        for length in []:
            index.forget("m", length)

        assert index.candidates("m", 8192) == [2048, 1024]

        # A genuine supersession does remove the length.
        for length in [1024]:
            index.forget("m", length)
        assert index.candidates("m", 8192) == [2048]

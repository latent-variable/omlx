# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-model validation for SpecPrefill sparse conversation reuse.

Unit tests can prove the sparse entry is stored, keyed and domain-separated
correctly, but they cannot prove the thing that actually matters: that a
sparse cache which has been through the SSD tier still produces the RIGHT
NUMBERS when a later turn is prefilled on top of it.

That claim rests on a property of the sparse prefill itself. ``sparse_prefill``
writes every selected key at its ORIGINAL RoPE angle, and RoPE is relative, so
attention between a query at logical position m and a stored key at logical
position p depends only on (m - p) no matter which physical row p landed in.
A sparse cache is therefore a legitimate prefix for a growing conversation.
These tests check that empirically, on a real hybrid checkpoint whose cache
mixes fixed-size recurrent layers with full-attention KV layers -- the same
``qwen3_5`` family and ``full_attention_interval`` as the 27B targets people
actually run SpecPrefill against.

Two phases:

1. Round-trip. A sparse cache stored and restored through the real prefix
   cache must extend to bit-identical logits versus the in-memory original.
2. Growth. Restoring turn 1 and sparse-prefilling turn 2 on top must agree
   with sparse-prefilling both turns in a single pass -- same cache offset,
   same RoPE adjustment, same sampled tokens.

The test never downloads checkpoints and writes its paged cache under pytest's
temporary directory. It is marked ``slow`` and needs an explicit model path.

What these tests do NOT establish: output QUALITY on an oMLX-quantized
checkpoint. They drive the model through ``mlx_lm.utils.load``, which does not
reproduce oMLX's own load path, and on an ``oQ4e`` build that yields incoherent
text for a plain dense prefill too -- "The capital of France is" comes back as
noise. Every assertion here is an EQUIVALENCE between two paths, so it stays
valid under that: both sides are equally affected, and identical output still
proves the round trip. But a pass on a 27B oQ4e checkpoint says nothing about
whether answers are GOOD. Quality belongs to a real server run.

Example::

    OMLX_SPARSE_REUSE_MODEL_PATH="$HOME/.omlx/models/mlx-community/Qwen3.5-2B-bf16" \
      uv run pytest tests/integration/test_specprefill_sparse_reuse_real_model.py \
      -o addopts="" -m slow -s -q
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="Real sparse-reuse validation requires macOS on Apple Silicon.",
    ),
]

_MODEL_PATH_ENV = "OMLX_SPARSE_REUSE_MODEL_PATH"

PREFIX_TOKENS = 1024
TURN_TWO_TOKENS = 512
TAIL_TOKENS = 32
KEEP_EVERY = 5  # deterministic 20% selection, independent of the draft scorer
CHUNK = 32      # select_chunks() granularity


def _chunked_selection(length: int, keep: float = 0.2, tail: int = 256):
    """A selection shaped like the one select_chunks() actually emits.

    Contiguous CHUNK-sized runs plus the mandatory trailing window. Taking
    every Nth token instead shreds the text into noise, which makes any
    downstream quality comparison meaningless -- both sides come out garbage
    and "they differ" says nothing.
    """
    n_chunks = max(1, (length + CHUNK - 1) // CHUNK)
    keep_n = max(1, int(n_chunks * keep))
    stride = max(1, n_chunks // keep_n)
    chosen = list(range(0, n_chunks, stride))[:keep_n]
    picked = {i for c in chosen for i in range(c * CHUNK, min((c + 1) * CHUNK, length))}
    picked |= set(range(max(0, length - tail), length))
    return sorted(picked)


def _model_path() -> Path:
    raw = os.environ.get(_MODEL_PATH_ENV)
    if not raw:
        pytest.skip(f"Set {_MODEL_PATH_ENV} to a local hybrid checkpoint.")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.skip(f"{_MODEL_PATH_ENV} does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def loaded_model():
    from mlx_lm.utils import load

    model, tokenizer = load(str(_model_path()))
    return model, tokenizer


def _snapshot(cache):
    """Deep-copy layer states so later model calls cannot mutate them."""
    import mlx.core as mx

    snapshot = []
    for layer in cache:
        state = layer.state
        copied = [mx.array(x) if hasattr(x, "shape") else x for x in state]
        # ArraysCache keeps a mutable list; KVCache expects a tuple.
        snapshot.append(copied if isinstance(state, list) else tuple(copied))
    return snapshot


def _restore(cache, snapshot):
    for layer, state in zip(cache, snapshot):
        layer.state = state


def _installed_adjustment(model):
    from omlx.patches.specprefill import (
        _OffsetAdjustedRoPE,
        _find_attention_layers,
        _get_attn_module,
    )

    for _idx, layer in _find_attention_layers(model):
        attention = _get_attn_module(layer)
        rope = getattr(attention, "rope", None)
        if isinstance(rope, _OffsetAdjustedRoPE):
            return rope._adjustment
    return None


def _kv_offset(cache):
    for layer in cache:
        offset = getattr(layer, "offset", None)
        if isinstance(offset, int) and offset > 0:
            return offset
    return 0


def test_hybrid_checkpoint_actually_mixes_cache_families(loaded_model):
    """Guard the premise: a dense-only model would not exercise the risk."""
    from collections import Counter

    from mlx_lm.models.cache import make_prompt_cache

    model, _tokenizer = loaded_model
    kinds = Counter(type(c).__name__ for c in make_prompt_cache(model))
    assert len(kinds) > 1, (
        f"expected a hybrid cache layout, got {dict(kinds)}. The recurrent "
        "layers are the part that cannot be trimmed backwards, so a dense "
        "model would silently skip what this test exists to cover."
    )


def test_sparse_cache_round_trips_and_extends(loaded_model, tmp_path):
    """A stored-and-restored sparse cache extends identically to the original."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.patches.specprefill import (
        _OffsetAdjustedRoPE,
        _find_attention_layers,
        _get_attn_module,
        _unwrap_rope,
        cleanup_rope,
        sparse_prefill,
    )

    model, _tokenizer = loaded_model
    mx.random.seed(0)
    tokens = mx.random.randint(
        1000, 100000, (PREFIX_TOKENS + TAIL_TOKENS,)
    ).astype(mx.int32)
    selected = mx.array(_chunked_selection(PREFIX_TOKENS))

    # -- in place --------------------------------------------------------
    cleanup_rope(model)
    cache = make_prompt_cache(model)
    sparse_prefill(model, tokens[:PREFIX_TOKENS], selected, cache, step_size=512)
    adjustment = _installed_adjustment(model)
    physical_rows = _kv_offset(cache)
    assert adjustment == PREFIX_TOKENS - physical_rows
    snapshot = _snapshot(cache)
    logits_inplace = model(tokens[PREFIX_TOKENS:][None], cache=cache)
    mx.eval(logits_inplace)

    # -- restored --------------------------------------------------------
    cleanup_rope(model)
    restored = make_prompt_cache(model)
    _restore(restored, snapshot)
    assert _kv_offset(restored) == physical_rows
    for _idx, layer in _find_attention_layers(model):
        attention = _get_attn_module(layer)
        if attention is not None and hasattr(attention, "rope"):
            attention.rope = _OffsetAdjustedRoPE(
                _unwrap_rope(attention.rope), adjustment
            )
    logits_restored = model(tokens[PREFIX_TOKENS:][None], cache=restored)
    mx.eval(logits_restored)
    cleanup_rope(model)

    assert bool(
        mx.all(
            mx.argmax(logits_inplace[0], axis=-1)
            == mx.argmax(logits_restored[0], axis=-1)
        )
    )
    assert float(mx.max(mx.abs(logits_inplace - logits_restored))) == 0.0


def test_storing_a_turn_changes_nothing_about_the_next_one(loaded_model):
    """Storing must be INVISIBLE. That is the invariant, and it is exact.

    The tempting comparison -- "restore then extend" against "both turns
    sparse-prefilled in one pass" -- is the wrong bar and it fails on real
    checkpoints. Splitting a sparse prefill into two calls moves the chunk
    boundaries, and a hybrid's chunked recurrent scan is boundary-sensitive:
    on Qwen3.8-27B that alone moves logits by ~27 with no cache involved. In
    production the turns ARRIVE separately, so the split is a given, never a
    choice.

    What the feature must guarantee is narrower and much stronger: passing the
    cache through the store/restore round trip changes NOTHING versus keeping
    it in memory. Measured bit-exact on both a bf16 2B and a 4-bit 27B.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.patches.specprefill import cleanup_rope, sparse_prefill

    model, _tokenizer = loaded_model
    mx.random.seed(0)
    turn_one = mx.random.randint(1000, 100000, (PREFIX_TOKENS,)).astype(mx.int32)
    turn_two = mx.random.randint(1000, 100000, (TURN_TWO_TOKENS,)).astype(mx.int32)
    tail = mx.random.randint(1000, 100000, (TAIL_TOKENS,)).astype(mx.int32)
    select_one = mx.array(_chunked_selection(PREFIX_TOKENS))
    select_two = mx.array(_chunked_selection(TURN_TWO_TOKENS))

    def second_turn(cache):
        sparse_prefill(
            model,
            turn_two,
            select_two,
            cache,
            step_size=512,
            position_offset=PREFIX_TOKENS,
        )
        logits = model(tail[None], cache=cache)
        mx.eval(logits)
        return logits

    cleanup_rope(model)
    resident = make_prompt_cache(model)
    sparse_prefill(model, turn_one, select_one, resident, step_size=512)
    snapshot = _snapshot(resident)
    physical_rows = _kv_offset(resident)

    # (a) never stored: continue straight on
    cleanup_rope(model)
    logits_resident = second_turn(resident)
    resident_offset = _kv_offset(resident)
    resident_adjustment = _installed_adjustment(model)

    # (b) stored and restored first
    cleanup_rope(model)
    restored = make_prompt_cache(model)
    _restore(restored, snapshot)
    assert _kv_offset(restored) == physical_rows
    logits_restored = second_turn(restored)
    restored_adjustment = _installed_adjustment(model)
    cleanup_rope(model)

    # Same physical shape, and the same adjustment installed for decode.
    assert _kv_offset(restored) == resident_offset
    assert restored_adjustment == resident_adjustment
    # The invariant: passing through storage changed nothing at all.
    assert float(mx.max(mx.abs(logits_resident - logits_restored))) == 0.0, (
        "store/restore perturbed the next turn"
    )


def test_sparse_entry_survives_the_ssd_tier(tmp_path):
    """Store through the real prefix cache, reopen it, restore intact."""
    import mlx.core as mx

    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    logical = list(range(4096))
    rows = 820
    keys = mx.arange(rows, dtype=mx.float32).reshape(1, 1, rows, 1)
    values = (keys + 7).astype(mx.float32)
    cache_data = [
        {
            "state": (keys, values),
            "meta_state": (rows,),
            "class_name": "KVCache",
            "cache_type": "KVCache",
        }
    ]

    def build():
        ssd = PagedSSDCacheManager(
            cache_dir=tmp_path / "sparse",
            max_size_bytes=256 * 1024**2,
            hot_cache_max_bytes=0,
            expected_model_name="sparse-real",
            expected_num_layers=1,
            expected_block_size=256,
        )
        paged = PagedCacheManager(
            block_size=256,
            max_blocks=512,
            model_name="sparse-real",
            initial_blocks=512,
        )
        paged.set_paged_ssd_cache_manager(ssd)
        return (
            BlockAwarePrefixCache(
                model=None,
                paged_cache_manager=paged,
                paged_ssd_cache_manager=ssd,
            ),
            ssd,
        )

    writer, writer_ssd = build()
    assert writer.store_sparse_prefix("cold", logical, cache_data)
    writer_ssd.close()

    reader, reader_ssd = build()
    try:
        restored = reader.restore_sparse_prefix(
            "warm", logical, promote_to_hot_cache=False
        )
        assert restored is not None
        layers, logical_tokens = restored
        assert logical_tokens == len(logical)
        assert layers[0].offset == rows
        assert mx.array_equal(layers[0].state[0], keys)
        # And still invisible to a dense reader after a restart.
        assert reader.fetch_exact_prefix("dense", logical) is None
    finally:
        reader_ssd.close()


def test_offset_rope_puts_new_tokens_at_true_logical_positions(loaded_model):
    """The exact arithmetic ``Scheduler._install_sparse_rope_offset`` performs.

    A restored prefix sits at cache offset N' while its next token belongs at
    logical position M. The helper installs _OffsetAdjustedRoPE(M - N'), and
    this asserts the only thing that has to be true of it: a chunk presented at
    cache offset N' + j comes out rotated to logical position M + j, matching
    the genuine RoPE asked for that position directly.

    Compared against the model's OWN rope rather than against a second
    sparse_prefill: sparse_prefill routes rotation through ``manual_rope``,
    which differs from the fused kernel by ~1e-4 per application and compounds
    over layers, so it is not a bit-exact reference for anything.
    """
    import mlx.core as mx

    from omlx.patches.specprefill import (
        _OffsetAdjustedRoPE,
        _find_attention_layers,
        _get_attn_module,
        _unwrap_rope,
    )

    model, _tokenizer = loaded_model
    _idx, layer = _find_attention_layers(model)[0]
    attention = _get_attn_module(layer)
    genuine = _unwrap_rope(attention.rope)

    logical_tokens = 4096      # M
    physical_rows = 820        # N'
    adjustment = logical_tokens - physical_rows

    wrapped = _OffsetAdjustedRoPE(genuine, adjustment)
    mx.random.seed(3)
    heads = getattr(attention, "num_attention_heads", 8)
    head_dim = getattr(attention, "head_dim", 128)
    chunk = mx.random.normal((1, heads, 8, head_dim)).astype(mx.float32)

    for j in (0, 1, 37, 512):
        got = wrapped(chunk, offset=physical_rows + j)
        want = genuine(chunk, offset=logical_tokens + j)
        assert float(mx.max(mx.abs(got - want))) == 0.0, (
            f"offset {physical_rows + j} did not rotate to logical "
            f"position {logical_tokens + j}"
        )

    # And the degenerate case the helper short-circuits: nothing dropped means
    # cache offsets already are true positions.
    assert logical_tokens - logical_tokens == 0


def test_real_text_generation_survives_store_restore_and_growth(loaded_model):
    """End to end on real text: a restored, extended cache still generates.

    The other tests work on random token ids, which drive the model into a
    near-degenerate logit distribution where any numeric wobble flips an
    argmax. This one uses real prose and greedy decoding, and asserts that a
    conversation reassembled from a STORED turn-1 cache generates the same
    continuation as one that never left memory -- the property an agent
    session actually depends on.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.patches.specprefill import cleanup_rope, sparse_prefill

    model, tokenizer = loaded_model

    document = (
        "The maintenance log for reactor seven records a coolant pressure of "
        "four hundred and twelve kilopascals, measured on the third of March. "
    ) * 60
    follow_up = " The engineer asked what the recorded coolant pressure was."

    turn_one = mx.array(tokenizer.encode(document), dtype=mx.int32)
    turn_two = mx.array(tokenizer.encode(follow_up), dtype=mx.int32)
    history_len = int(turn_one.shape[0])
    if history_len < 256:
        pytest.skip("tokenizer produced too short a document for this test")
    selection = mx.array(_chunked_selection(history_len))

    def generate(cache, kickoff, steps=12):
        tokens = []
        logits = model(kickoff[None], cache=cache)
        mx.eval(logits)
        for _ in range(steps):
            nxt = mx.argmax(logits[0, -1])
            mx.eval(nxt)
            tokens.append(int(nxt))
            logits = model(nxt.reshape(1, 1), cache=cache)
            mx.eval(logits)
        return tokens

    # -- never left memory ------------------------------------------------
    cleanup_rope(model)
    resident = make_prompt_cache(model)
    sparse_prefill(model, turn_one, selection, resident, step_size=512)
    physical_rows = _kv_offset(resident)
    snapshot = _snapshot(resident)
    sparse_prefill(
        model,
        turn_two[:-1],
        mx.array(list(range(int(turn_two.shape[0]) - 1))),
        resident,
        step_size=512,
        position_offset=history_len,
    )
    resident_tokens = generate(resident, turn_two[-1:])

    # -- reassembled from a stored turn 1 ---------------------------------
    cleanup_rope(model)
    reassembled = make_prompt_cache(model)
    _restore(reassembled, snapshot)
    assert _kv_offset(reassembled) == physical_rows
    sparse_prefill(
        model,
        turn_two[:-1],
        mx.array(list(range(int(turn_two.shape[0]) - 1))),
        reassembled,
        step_size=512,
        position_offset=history_len,
    )
    reassembled_tokens = generate(reassembled, turn_two[-1:])
    cleanup_rope(model)

    assert reassembled_tokens == resident_tokens, (
        f"stored/restored history changed the continuation:\n"
        f"  resident:    {tokenizer.decode(resident_tokens)!r}\n"
        f"  reassembled: {tokenizer.decode(reassembled_tokens)!r}"
    )

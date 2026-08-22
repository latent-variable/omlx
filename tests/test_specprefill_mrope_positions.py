# SPDX-License-Identifier: Apache-2.0
"""SpecPrefill position mapping on the M-RoPE (rotary_emb) attention path.

SpecPrefill's design note states the property the whole approach rests on:
selected keys are written at their ORIGINAL RoPE angles, so that attention
during decode depends on true token distances. ``sparse_prefill`` implements
that by wrapping ``attn.rope``.

mlx-vlm's ``Qwen3_5Attention`` does not have ``.rope``. It carries
``rotary_emb`` (M-RoPE, with an ``mrope_section``) and takes positions as an
explicit ``position_ids`` argument, deriving a CONTIGUOUS range from
``cache.offset`` when the caller supplies none. ``has_rope`` was therefore
False for every mlx-vlm Qwen3.5/3.6/3.8 checkpoint, and the whole
position-mapping block -- including ``position_offset`` -- was skipped.

The result was self-consistent rather than corrupt: positions stayed monotonic
and correctly offset from the cache, so the model saw a COMPRESSED document
rather than a scrambled one. What it did not see was the real spacing between
the tokens that were kept, which is what the method is defined to preserve and
what anything reusing that cache later has to be able to rely on.
"""

from __future__ import annotations

import types

import mlx.core as mx
import pytest

from omlx.patches.specprefill import (
    _OffsetAdjustedMRoPE,
    _PositionMappedMRoPE,
    _get_rope_carrier,
    _has_rope_carrier,
    _make_offset_adjusted,
    _make_position_mapped,
    _OffsetAdjustedRoPE,
    _PositionMappedRoPE,
    _unwrap_rope,
    cleanup_rope,
)


class _RecordingMRoPE:
    """Stands in for rotary_emb and records the ids it is handed."""

    def __init__(self):
        self.seen: list[list[int]] = []

    def apply_rotary(self, q, k, position_ids, **kwargs):
        self.seen.append(mx.array(position_ids).reshape(-1).tolist())
        return q, k


class TestCarrierDetection:
    def test_scalar_offset_rope_is_found(self):
        attn = types.SimpleNamespace(rope="R")
        assert _get_rope_carrier(attn) == ("rope", "R")
        assert _has_rope_carrier(attn) is True

    def test_mrope_is_found(self):
        attn = types.SimpleNamespace(rotary_emb="M")
        assert _get_rope_carrier(attn) == ("rotary_emb", "M")
        assert _has_rope_carrier(attn) is True

    def test_rope_wins_when_both_exist(self):
        attn = types.SimpleNamespace(rope="R", rotary_emb="M")
        assert _get_rope_carrier(attn)[0] == "rope"

    def test_an_architecture_with_neither_is_reported_honestly(self):
        """Nemotron-H has no RoPE at all; that must stay a no-op, not a guess."""
        assert _get_rope_carrier(types.SimpleNamespace(q_proj=object())) is None
        assert _has_rope_carrier(None) is False

    def test_factories_pick_the_matching_wrapper(self):
        assert isinstance(
            _make_position_mapped("rotary_emb", object(), mx.array([0, 1]), 0),
            _PositionMappedMRoPE,
        )
        assert isinstance(
            _make_position_mapped("rope", _StubRope(), mx.array([0, 1]), 0),
            _PositionMappedRoPE,
        )
        assert isinstance(
            _make_offset_adjusted("rotary_emb", object(), 5), _OffsetAdjustedMRoPE
        )
        assert isinstance(
            _make_offset_adjusted("rope", object(), 5), _OffsetAdjustedRoPE
        )


class _StubRope:
    """Minimal object satisfying _PositionMappedRoPE's constructor probing."""

    def __init__(self):
        self.dims = 8
        self.base = 10000
        self.scale = 1.0

    def __call__(self, x, offset=0):
        return x


class TestPositionMapping:
    def test_contiguous_ids_are_rewritten_to_original_positions(self):
        """The defect, stated as a test.

        The attention hands over the contiguous range it derived from the
        cache offset. Those are indices into the SELECTED sequence, so the
        wrapper gathers the true position for each one.
        """
        recorder = _RecordingMRoPE()
        selected = mx.array([0, 5, 9, 20, 33])
        wrapper = _PositionMappedMRoPE(recorder, selected, cache_start=0)

        q = mx.zeros((1, 2, 5, 4))
        wrapper.apply_rotary(q, q, mx.arange(5).reshape(1, 1, 5))

        assert recorder.seen[0] == [0, 5, 9, 20, 33]

    def test_a_restored_prefix_shifts_the_index_window(self):
        """cache_start removes KV already present, so indexing stays aligned."""
        recorder = _RecordingMRoPE()
        selected = mx.array([0, 5, 9, 20, 33])
        wrapper = _PositionMappedMRoPE(recorder, selected, cache_start=2)

        q = mx.zeros((1, 2, 3, 4))
        # Two KV rows were already present, so the ids start at 2 and the
        # first SELECTED token is still entry 0 of the selection.
        wrapper.apply_rotary(q, q, mx.arange(2, 5).reshape(1, 1, 3))

        assert recorder.seen[0] == [0, 5, 9]

        # Advancing the window walks further into the selection.
        recorder.seen.clear()
        wrapper.apply_rotary(q, q, mx.arange(4, 7).reshape(1, 1, 3))
        assert recorder.seen[0] == [9, 20, 33]

    def test_out_of_range_indices_are_clamped_not_wrapped(self):
        """Negative indexing would silently read from the END of the array."""
        recorder = _RecordingMRoPE()
        wrapper = _PositionMappedMRoPE(recorder, mx.array([7, 8, 9]), cache_start=5)

        q = mx.zeros((1, 2, 3, 4))
        wrapper.apply_rotary(q, q, mx.arange(3).reshape(1, 1, 3))

        assert recorder.seen[0] == [7, 7, 7]


class TestOffsetAdjustment:
    def test_a_constant_is_added_to_every_id(self):
        recorder = _RecordingMRoPE()
        wrapper = _OffsetAdjustedMRoPE(recorder, 819)

        q = mx.zeros((1, 2, 4, 4))
        wrapper.apply_rotary(q, q, mx.arange(205, 209).reshape(1, 1, 4))

        assert recorder.seen[0] == [1024, 1025, 1026, 1027]


class TestUnwrapAndCleanup:
    def test_both_wrapper_families_peel_to_the_genuine_module(self):
        genuine = _RecordingMRoPE()
        assert _unwrap_rope(_OffsetAdjustedMRoPE(genuine, 5)) is genuine
        assert _unwrap_rope(
            _PositionMappedMRoPE(genuine, mx.array([0]), 0)
        ) is genuine
        # Nested, as #766 produced for the .rope family.
        assert (
            _unwrap_rope(
                _OffsetAdjustedMRoPE(_PositionMappedMRoPE(genuine, mx.array([0]), 0), 5)
            )
            is genuine
        )

    def test_cleanup_restores_a_wrapped_rotary_emb(self):
        genuine = _RecordingMRoPE()
        attn = types.SimpleNamespace(rotary_emb=_OffsetAdjustedMRoPE(genuine, 5))
        model = types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)])

        cleanup_rope(model)

        assert attn.rotary_emb is genuine

    def test_cleanup_leaves_an_unwrapped_module_alone(self):
        genuine = _RecordingMRoPE()
        attn = types.SimpleNamespace(rotary_emb=genuine)
        model = types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)])

        cleanup_rope(model)

        assert attn.rotary_emb is genuine

    def test_cleanup_is_a_no_op_without_any_carrier(self):
        attn = types.SimpleNamespace(q_proj=object())
        model = types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)])
        cleanup_rope(model)  # must not raise

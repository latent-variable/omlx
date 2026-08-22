# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-model check that SpecPrefill maps positions on the M-RoPE path.

Unit tests drive the wrappers directly. This drives a real mlx-vlm
``Qwen3_5Attention`` -- the one that actually has ``rotary_emb`` instead of
``.rope`` -- and records the ``position_ids`` the model receives, which is the
only place the defect was ever visible.

Loading matters: ``mlx_lm.utils.load`` on the same checkpoint yields
``Qwen3NextAttention``, which HAS ``.rope`` and was never affected. The bug
only appears through the mlx-vlm path oMLX actually serves these models with,
which is why it survived.

Example::

    OMLX_MROPE_MODEL_PATH="$HOME/.omlx/models/mlx-community/Qwen3.5-2B-bf16" \
      uv run pytest tests/integration/test_specprefill_mrope_real_model.py \
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
        reason="Real M-RoPE validation requires macOS on Apple Silicon.",
    ),
]

_MODEL_PATH_ENV = "OMLX_MROPE_MODEL_PATH"
SELECTED = [0, 5, 9, 20, 33, 47, 60, 63]
CONV_TOKENS = 64
SYSTEM_TOKENS = 40


def _model_path() -> Path:
    raw = os.environ.get(_MODEL_PATH_ENV)
    if not raw:
        pytest.skip(f"Set {_MODEL_PATH_ENV} to a local mlx-vlm qwen3_5 checkpoint.")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.skip(f"{_MODEL_PATH_ENV} does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def language_model():
    mlx_vlm = pytest.importorskip("mlx_vlm.utils")
    model, _processor = mlx_vlm.load(str(_model_path()))
    return getattr(model, "language_model", model)


class _Spy:
    """Wraps the genuine rotary_emb and records the ids it is handed."""

    def __init__(self, inner):
        self._inner = inner
        self.seen: list[list[int]] = []

    def apply_rotary(self, q, k, position_ids, **kwargs):
        import mlx.core as mx

        self.seen.append(mx.array(position_ids).reshape(-1).tolist())
        return self._inner.apply_rotary(q, k, position_ids, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def test_this_checkpoint_really_uses_the_mrope_carrier(language_model):
    """Guard the premise: on .rope this test would prove nothing."""
    from omlx.patches.specprefill import (
        _find_attention_layers,
        _get_attn_module,
        _get_rope_carrier,
    )

    attn = _get_attn_module(_find_attention_layers(language_model)[0][1])
    carrier = _get_rope_carrier(attn)
    assert carrier is not None, "no RoPE carrier found at all"
    assert carrier[0] == "rotary_emb", (
        f"expected the M-RoPE path, got {carrier[0]} on {type(attn).__name__}. "
        "Loaded via mlx_lm rather than mlx_vlm?"
    )


def test_selected_tokens_receive_their_original_positions(language_model):
    """The defect and the fix, on real weights.

    Before: every selected token got a contiguous id, so the model saw a
    COMPRESSED document rather than the real spacing between what was kept.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    import omlx.patches.specprefill as sp

    layers = sp._find_attention_layers(language_model)
    spies = []
    for _idx, layer in layers:
        attn = sp._get_attn_module(layer)
        spy = _Spy(attn.rotary_emb)
        attn.rotary_emb = spy
        spies.append((attn, spy))

    try:
        tokens = mx.arange(1000, 1000 + CONV_TOKENS).astype(mx.int32)
        selected = mx.array(SELECTED)

        sp.cleanup_rope(language_model)
        cache = make_prompt_cache(language_model)
        spies[0][1].seen.clear()
        sp.sparse_prefill(language_model, tokens, selected, cache, step_size=512)
        sp.cleanup_rope(language_model)

        first = spies[0][1].seen[0][: len(SELECTED)]
        assert first == SELECTED[: len(first)], (
            f"selected tokens were rotated at {first}, not their own positions"
        )
    finally:
        for attn, spy in spies:
            attn.rotary_emb = spy._inner


def test_position_offset_is_honoured_after_a_dense_prefix(language_model):
    """``position_offset`` was computed and then silently dropped.

    With a system prefix already in the cache, the conversation's selected
    tokens belong at ``index + system_tokens``.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    import omlx.patches.specprefill as sp

    layers = sp._find_attention_layers(language_model)
    spies = []
    for _idx, layer in layers:
        attn = sp._get_attn_module(layer)
        spy = _Spy(attn.rotary_emb)
        attn.rotary_emb = spy
        spies.append((attn, spy))

    try:
        sp.cleanup_rope(language_model)
        cache = make_prompt_cache(language_model)
        system = mx.arange(1000, 1000 + SYSTEM_TOKENS).astype(mx.int32)
        language_model(system[None], cache=cache)
        mx.eval([layer.state for layer in cache])

        spies[0][1].seen.clear()
        conversation = mx.arange(2000, 2000 + CONV_TOKENS).astype(mx.int32)
        sp.sparse_prefill(
            language_model,
            conversation,
            mx.array(SELECTED),
            cache,
            step_size=512,
            position_offset=SYSTEM_TOKENS,
        )
        sp.cleanup_rope(language_model)

        expected = [i + SYSTEM_TOKENS for i in SELECTED]
        got = spies[0][1].seen[0][: len(expected)]
        assert got == expected[: len(got)], (
            f"conversation tokens landed at {got}, expected {expected[: len(got)]}"
        )
    finally:
        for attn, spy in spies:
            attn.rotary_emb = spy._inner

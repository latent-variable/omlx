# SPDX-License-Identifier: Apache-2.0
"""Chunk reuse for qwen3_5-family hybrids loaded through mlx-vlm.

oMLX loads Qwen3.5/3.6 (incl. MTP/oQ variants) through mlx-vlm, whose
attention exposes an M-RoPE ``rotary_emb`` (interleaved style, partial
rotary) instead of stock mlx-lm's ``.rope``. Two facts make the same
chunk-reuse mechanism carry over:

1. For text-only positions all three M-RoPE axes carry the same scalar
   position, so the interleaved frequency selection collapses to standard
   RoPE (see mlx_vlm rope_utils.compute_mrope_frequencies) applied with
   split-half pairing on the first ``rotary_emb.dim`` dims. Re-rotating a
   cached key from src_pos to dst_pos is a constant-angle split-half
   rotation with ``delta * inv_freq`` — rotations compose, and any
   attention_scaling already baked into the cached keys is preserved
   because the rotation is linear.
2. mlx-vlm re-exports mlx-lm's KVCache/ArraysCache, so assembled caches
   are byte-compatible with what oMLX's make_prompt_cache produces and
   feed the batch-insert/decode (and MTP verify) paths exactly like a
   paged prefix-cache restore.

Linear (GatedDeltaNet) layers use the same input-replay trick as the
mlx-lm hybrid path: cache per-token pre-conv projections (qkv/a/b) at
capture, replay only conv + gated-delta scan from live state at reuse.

Layer math replicated from the pinned mlx_vlm qwen3_5 module — guarded by
tests/test_chunk_reuse_vlm.py (capture must match the stock forward);
revisit on mlx-vlm upgrades. Long-term home for the capture/replay
primitives is upstream mlx-vlm; this proves the mechanism downstream first.
"""

from __future__ import annotations

import time
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache

from . import chunk_reuse as cr
from .chunk_reuse import BlendStats, _spans_from_indices, insert_kv
from .chunk_reuse_hybrid import CaptureResult, HybridChunk, LinearRecord


@lru_cache(maxsize=1)
def _q35():
    """mlx-vlm qwen3_5 language module (lazy: keep engine import cheap)."""
    from mlx_vlm.models.qwen3_5 import language

    return language


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def _lm(model):
    """The mlx-vlm LanguageModel (owns args / lm_head / make_cache)."""
    model = cr.unwrap_model(model)  # strips oMLX's VLMModelAdapter
    return getattr(model, "language_model", model)


def _inner(model):
    """The Qwen3_5Model with embed_tokens / layers / norm."""
    return _lm(model).model


def get_vlm_hybrid_layout(model):
    """(layers, is_linear flags, fa_idx, rotary_embs by idx).

    Raises unless this is a qwen3_5-style mlx-vlm hybrid whose M-RoPE
    collapses to standard RoPE for text (position-selector styles).
    """
    layers = _inner(model).layers
    flags = [getattr(l, "is_linear", False) for l in layers]
    if not any(flags):
        raise ValueError("not a hybrid model")
    fa_idx = flags.index(False)
    ropes = {}
    for i, l in enumerate(layers):
        if flags[i]:
            continue
        emb = getattr(l.self_attn, "rotary_emb", None)
        if emb is None or not hasattr(emb, "inv_freq") or not hasattr(emb, "dim"):
            raise ValueError("attention layer without an M-RoPE rotary_emb")
        # Text positions collapse only for styles that select frequency by
        # position axis with split-half pairing (interleaved/chunked).
        if getattr(emb, "style", None) not in ("interleaved", "chunked"):
            raise ValueError(f"unsupported M-RoPE style {getattr(emb, 'style', None)!r}")
        ropes[i] = emb
    return layers, flags, fa_idx, ropes


def make_vlm_caches(model):
    return _lm(model).make_cache()


# ---------------------------------------------------------------------------
# Key re-rotation (constant delta, text-only positions)
# ---------------------------------------------------------------------------

def rotate_keys_delta_mrope(rotary_emb, keys, delta: int):
    """Rotate cached post-RoPE keys by ``delta`` positions.

    Split-half pairing on the first ``rotary_emb.dim`` dims (mlx-vlm
    interleaved/chunked text path); remaining dims pass through untouched.
    """
    if delta == 0:
        return keys + 0
    rd = rotary_emb.dim
    freqs = float(delta) * rotary_emb.inv_freq.astype(mx.float32)  # (rd/2,)
    emb = mx.concatenate([freqs, freqs])
    cos, sin = mx.cos(emb), mx.sin(emb)
    k_rot = keys[..., :rd].astype(mx.float32)
    half = rd // 2
    rotated = mx.concatenate([-k_rot[..., half:], k_rot[..., :half]], axis=-1)
    out = (k_rot * cos + rotated * sin).astype(keys.dtype)
    if keys.shape[-1] == rd:
        return out
    return mx.concatenate([out, keys[..., rd:]], axis=-1)


# ---------------------------------------------------------------------------
# GatedDeltaNet capture/replay (math mirrors mlx_vlm qwen3_5.Qwen3_5GatedDeltaNet
# on the plain-cache, no-mask, no-target-verify path)
# ---------------------------------------------------------------------------

def _gdn_scan_vlm(gdn, mixed_qkv, a, b, cache):
    """Conv (from live conv-state) + gated delta scan; updates cache in place."""
    q35 = _q35()
    B, S, _ = mixed_qkv.shape
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=mixed_qkv.dtype
        )
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    if cache is not None:
        cache[0] = mx.contiguous(conv_input[:, -(gdn.conv_kernel_size - 1):, :])
    conv_out = nn.silu(gdn.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]
    state = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, state = q35.gated_delta_update(
        q, k, v, a, b, gdn.A_log, gdn.dt_bias, state, None, use_kernel=True
    )
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    return out


def _gdn_layer(layer, h, cache, record_into=None, layer_idx=None):
    """Full linear-layer block (attn part + residual + MLP), optionally
    recording the pre-conv per-token inputs."""
    x_ln = layer.input_layernorm(h)
    gdn = layer.linear_attn
    mixed_qkv = gdn.in_proj_qkv(x_ln)
    z = gdn.in_proj_z(x_ln)
    b = gdn.in_proj_b(x_ln)
    a = gdn.in_proj_a(x_ln)
    if record_into is not None:
        record_into[layer_idx] = LinearRecord(mixed_qkv, a, b)
    out = _gdn_scan_vlm(gdn, mixed_qkv, a, b, cache)
    B, S = h.shape[0], h.shape[1]
    z4 = z.reshape(B, S, -1, gdn.head_v_dim)
    r = gdn.out_proj(gdn.norm(out, z4).reshape(B, S, -1))
    h = h + r
    return h + layer.mlp(layer.post_attention_layernorm(h))


# ---------------------------------------------------------------------------
# Capture (recording prefill) and chunk extraction
# ---------------------------------------------------------------------------

def _text_position_ids(start: int, length: int) -> mx.array:
    """(3, 1, L) all-axes-equal positions for a text span."""
    pid = mx.arange(start, start + length)[None, :]
    return mx.tile(mx.expand_dims(pid, 0), (3, 1, 1))


def capture_prefill(model, tokens: list[int]) -> CaptureResult:
    """Full prefill recording every linear layer's per-token inputs."""
    q35 = _q35()
    layers, flags, fa_idx, ropes = get_vlm_hybrid_layout(model)
    caches = make_vlm_caches(model)
    records: dict[int, LinearRecord] = {}

    t0 = time.perf_counter()
    inner = _inner(model)
    lm = _lm(model)
    h = inner.embed_tokens(mx.array(tokens)[None])
    position_ids = _text_position_ids(0, len(tokens))
    fa_mask = q35._create_qwen3_5_attention_mask(h, caches[fa_idx])
    ssm_mask = q35._create_qwen3_5_ssm_mask(h, caches[flags.index(True)])
    pe = None
    if not ropes[fa_idx].fused_apply:
        pe = ropes[fa_idx](h, position_ids)
    for i, (layer, c) in enumerate(zip(layers, caches)):
        if flags[i]:
            # decomposed so we can record pre-conv inputs; math mirrors
            # Qwen3_5DecoderLayer with ssm_mask (None for plain B=1 caches)
            if ssm_mask is not None:
                raise ValueError("unexpected ssm mask on plain capture cache")
            h = _gdn_layer(layer, h, c, record_into=records, layer_idx=i)
        else:
            r = layer.self_attn(
                layer.input_layernorm(h), mask=fa_mask, cache=c,
                position_ids=position_ids, position_embeddings=pe,
            )
            h = h + r
            h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = inner.norm(h)
    if lm.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)
    mx.eval(logits)
    return CaptureResult(logits[0, -1, :], caches, records, time.perf_counter() - t0)


def extract_vlm_chunk(cap: CaptureResult, tokens: list[int], start: int, end: int) -> HybridChunk:
    attn_kv, linear = {}, {}
    for i, c in enumerate(cap.caches):
        if isinstance(c, KVCache):
            k, v = c.state
            attn_kv[i] = (k[..., start:end, :] + 0, v[..., start:end, :] + 0)
        elif isinstance(c, ArraysCache):
            linear[i] = cap.records[i].slice(start, end)
        else:  # pragma: no cover — layout guard should have declined
            raise ValueError(f"unexpected cache type {type(c).__name__}")
    return HybridChunk(tokens=list(tokens[start:end]), src_pos=start,
                       attn_kv=attn_kv, linear=linear)


# ---------------------------------------------------------------------------
# Blended prefill
# ---------------------------------------------------------------------------

def _forward_span(model, tokens: list[int], caches, pos: int):
    """Run a token span through the inner model with explicit text positions.

    Bypasses LanguageModel.__call__ so no _rope_deltas/_position_ids instance
    state is read or written.
    """
    inner = _inner(model)
    out = inner(
        mx.array(tokens)[None],
        cache=caches,
        position_ids=_text_position_ids(pos, len(tokens)),
    )
    mx.eval(out)
    return out


def vlm_blended_prefill(
    model,
    prefix_tokens: list[int],
    chunks: list[HybridChunk],
    suffix_tokens: list[int],
    *,
    mode: str = "edge",  # "reuse" | "edge" ("devblock" falls back to "edge")
    edge_k: int = 32,
    **_ignored,
):
    layers, flags, fa_idx, ropes = get_vlm_hybrid_layout(model)
    stats = BlendStats()
    caches = make_vlm_caches(model)
    t0 = time.perf_counter()

    pos = 0
    if prefix_tokens:
        _forward_span(model, prefix_tokens, caches, pos)
        pos += len(prefix_tokens)

    if mode == "devblock":
        mode = "edge"  # deviation probe not ported to the vlm path yet

    for ci, chunk in enumerate(chunks):
        L = len(chunk.tokens)
        delta = pos - chunk.src_pos

        if mode == "reuse":
            recompute_idx: set[int] = set()
        elif mode == "edge":
            recompute_idx = set(range(min(edge_k, L)))
        else:
            raise ValueError(mode)

        for s, e, recompute in _spans_from_indices(recompute_idx, L):
            if recompute:
                _forward_span(model, chunk.tokens[s:e], caches, pos + s)
                stats.recomputed_tokens += e - s
                stats.recompute_spans.append((ci, s, e))
            else:
                for i, c in enumerate(caches):
                    if isinstance(c, KVCache):
                        k_src, v_src = chunk.attn_kv[i]
                        k = rotate_keys_delta_mrope(ropes[i], k_src[..., s:e, :], delta)
                        insert_kv(c, k, v_src[..., s:e, :] + 0)
                    else:
                        rec = chunk.linear[i].slice(s, e)
                        _gdn_scan_vlm(layers[i].linear_attn, rec.qkv, rec.a, rec.b, c)
                stats.reused_tokens += e - s
        pos += L

    # Empty suffix = engine mode: caller (oMLX) prefills the suffix itself.
    last_logits = None
    if suffix_tokens:
        out = _forward_span(model, suffix_tokens, caches, pos)
        lm = _lm(model)
        if lm.args.tie_word_embeddings:
            last_logits = _inner(model).embed_tokens.as_linear(out)[0, -1, :]
        else:
            last_logits = lm.lm_head(out)[0, -1, :]
        pos += len(suffix_tokens)
    mx.eval([c.state for c in caches if c.state is not None])
    stats.prefill_seconds = time.perf_counter() - t0
    stats.total_tokens = pos
    return caches, last_logits, stats

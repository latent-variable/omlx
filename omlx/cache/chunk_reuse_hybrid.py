# SPDX-License-Identifier: Apache-2.0
"""Chunk reuse for HYBRID models (qwen3_5-family: GatedDeltaNet + attention).

The insight: hybrid linear-attention layers have no transplantable per-token
state, but their recurrence is applied to *per-token inputs* that are
functions of the token's hidden state — the same quantity KV reuse already
freezes. So instead of transplanting state, we cache each chunk's per-token
pre-conv projections (qkv, a, b) from the donor and *replay* only the cheap
depthwise conv + gated-delta scan from the live incoming state at reuse time.
All projections, MLPs (incl. MoE), and attention math for reused tokens are
skipped. Attention layers get the usual rotated-KV transplant.

The linear replay is exact given frozen inputs (no positional encoding in the
recurrence); the conv even uses the *live* preceding conv-state, so chunk
boundaries are slightly better than frozen.

Layer math replicated from mlx_lm 0.31.3 models/qwen3_5.py — guarded by
test_capture_matches_stock; revisit on mlx-lm upgrades.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update

from .chunk_reuse import (
    BlendStats,
    _spans_from_indices,
    copy_cache,
    insert_kv,
    rotate_keys_delta_module,
)


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def _lm(model):
    """The text-model wrapper that owns args/head (qwen3_5 nests it)."""
    return getattr(model, "language_model", model)


def _inner(model):
    """The module with embed_tokens / layers / norm."""
    return _lm(model).model


def get_hybrid_layout(model):
    """(layers, is_linear flags, fa_idx of first attention layer, ropes by idx)."""
    layers = _inner(model).layers
    flags = [getattr(l, "is_linear", False) for l in layers]
    if not any(flags):
        raise ValueError("not a hybrid model; use blendkv.blended_prefill")
    fa_idx = flags.index(False)
    ropes = {i: l.self_attn.rope for i, l in enumerate(layers) if not flags[i]}
    return layers, flags, fa_idx, ropes


def make_hybrid_caches(model):
    return [
        ArraysCache(size=2) if getattr(l, "is_linear", False) else KVCache()
        for l in _inner(model).layers
    ]


def copy_hybrid_cache(c):
    if isinstance(c, KVCache):
        return copy_cache(c)
    n = ArraysCache(size=2)
    n.cache = [x + 0 if x is not None else None for x in c.cache]
    return n


# ---------------------------------------------------------------------------
# GatedDeltaNet: capture and replay (math mirrors mlx_lm qwen3_5.GatedDeltaNet)
# ---------------------------------------------------------------------------

def _gdn_conv_qkv(gdn, x_ln):
    """Pre-conv projections: the per-token inputs we cache."""
    qkv = gdn.in_proj_qkv(x_ln)
    a = gdn.in_proj_a(x_ln)
    b = gdn.in_proj_b(x_ln)
    z = gdn.in_proj_z(x_ln)
    return qkv, a, b, z


def _gdn_scan(gdn, qkv, a, b, cache):
    """Conv (from live conv-state) + gated delta scan (from live state).

    Returns (out_pre_norm, q_used) — out is pre RMSNormGated/out_proj.
    Updates cache[0] (conv tail) and cache[1] (state), advances cache.
    """
    B, S, _ = qkv.shape
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=qkv.dtype
        )
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
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

    out, state = gated_delta_update(
        q, k, v, a, b, gdn.A_log, gdn.dt_bias, state, None, use_kernel=True
    )
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    return out


@dataclass
class LinearRecord:
    qkv: mx.array  # (B, S, conv_dim) pre-conv projections
    a: mx.array    # (B, S, Hv)
    b: mx.array    # (B, S, Hv)

    def slice(self, s, e):
        return LinearRecord(self.qkv[:, s:e], self.a[:, s:e], self.b[:, s:e])


@dataclass
class HybridChunk:
    tokens: list[int]
    src_pos: int
    attn_kv: dict          # layer_idx -> (K, V)
    linear: dict           # layer_idx -> LinearRecord


@dataclass
class CaptureResult:
    logits_last: mx.array
    caches: list
    records: dict          # layer_idx -> LinearRecord (full sequence)
    seconds: float


def capture_prefill(model, tokens: list[int]):
    """Full prefill that also records every linear layer's per-token inputs."""
    layers, flags, fa_idx, _ = get_hybrid_layout(model)
    caches = make_hybrid_caches(model)
    records: dict[int, LinearRecord] = {}

    t0 = time.perf_counter()
    inner = _inner(model)
    h = inner.embed_tokens(mx.array(tokens)[None])
    fa_mask = create_attention_mask(h, caches[fa_idx])
    for i, (layer, c) in enumerate(zip(layers, caches)):
        if flags[i]:
            x_ln = layer.input_layernorm(h)
            gdn = layer.linear_attn
            qkv, a, b, z = _gdn_conv_qkv(gdn, x_ln)
            records[i] = LinearRecord(qkv, a, b)
            out = _gdn_scan(gdn, qkv, a, b, c)
            B_, S_ = h.shape[0], h.shape[1]
            z4 = z.reshape(B_, S_, gdn.num_v_heads, gdn.head_v_dim)
            r = gdn.out_proj(gdn.norm(out, z4).reshape(B_, S_, -1))
        else:
            r = layer.self_attn(layer.input_layernorm(h), fa_mask, c)
        h = h + r
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = inner.norm(h)
    lm = _lm(model)
    logits = (inner.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings
              else lm.lm_head(h))
    mx.eval(logits)
    return CaptureResult(logits[0, -1, :], caches, records, time.perf_counter() - t0)


def extract_hybrid_chunk(cap: CaptureResult, tokens: list[int], start: int, end: int) -> HybridChunk:
    attn_kv, linear = {}, {}
    for i, c in enumerate(cap.caches):
        if isinstance(c, KVCache):
            k, v = c.state
            attn_kv[i] = (k[..., start:end, :] + 0, v[..., start:end, :] + 0)
        else:
            linear[i] = cap.records[i].slice(start, end)
    return HybridChunk(tokens=list(tokens[start:end]), src_pos=start,
                       attn_kv=attn_kv, linear=linear)


# ---------------------------------------------------------------------------
# Blended prefill for hybrids
# ---------------------------------------------------------------------------

def _forward(model, tokens, caches):
    logits = model(mx.array(tokens)[None], cache=caches)
    mx.eval(logits)
    return logits[0, -1, :]


def _probe_first_attn_deviation(model, chunk: HybridChunk, caches, fa_idx, flags, ropes, pos):
    """Deviation at the first attention layer, running layers 0..fa_idx on
    throwaway cache copies (the leading linear layers are cheap)."""
    layers = _inner(model).layers
    h = _inner(model).embed_tokens(mx.array(chunk.tokens)[None])
    probes = [copy_hybrid_cache(caches[i]) for i in range(fa_idx + 1)]
    fa_mask = create_attention_mask(h, probes[fa_idx])
    for i in range(fa_idx):
        h = layers[i](h, mask=None, cache=probes[i])
    attn = layers[fa_idx].self_attn
    x = layers[fa_idx].input_layernorm(h)
    B, L, _ = x.shape
    n_kv = getattr(attn, "num_key_value_heads", getattr(attn, "n_kv_heads", None))
    k = attn.k_proj(x).reshape(B, L, n_kv, -1)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    k_fresh = attn.rope(k.transpose(0, 2, 1, 3), offset=pos)
    k_reused = rotate_keys_delta_module(ropes[fa_idx], chunk.attn_kv[fa_idx][0], pos - chunk.src_pos)
    dev = mx.sqrt(mx.sum((k_fresh - k_reused) ** 2, axis=(0, 1, 3)))
    mx.eval(dev)
    return dev


def hybrid_blended_prefill(
    model,
    prefix_tokens: list[int],
    chunks: list[HybridChunk],
    suffix_tokens: list[int],
    *,
    mode: str = "reuse",  # "reuse" | "edge" | "devblock"
    edge_k: int = 16,
    deviation_ratio: float = 0.15,
    dev_block: int = 32,
):
    layers, flags, fa_idx, ropes = get_hybrid_layout(model)
    stats = BlendStats()
    caches = make_hybrid_caches(model)
    t0 = time.perf_counter()

    pos = 0
    if prefix_tokens:
        _forward(model, prefix_tokens, caches)
        pos += len(prefix_tokens)

    for ci, chunk in enumerate(chunks):
        L = len(chunk.tokens)
        delta = pos - chunk.src_pos

        if mode == "reuse":
            recompute_idx: set[int] = set()
        elif mode == "edge":
            recompute_idx = set(range(min(edge_k, L)))
        elif mode == "devblock":
            dev = _probe_first_attn_deviation(model, chunk, caches, fa_idx, flags, ropes, pos)
            n_blocks = (L + dev_block - 1) // dev_block
            scores = [
                float(mx.mean(dev[bi * dev_block : min((bi + 1) * dev_block, L)]).item())
                for bi in range(n_blocks)
            ]
            n_sel = max(1, int(round(deviation_ratio * n_blocks)))
            top = sorted(range(n_blocks), key=lambda x: -scores[x])[:n_sel]
            recompute_idx = set()
            for bi in set(top) | {0}:
                recompute_idx |= set(range(bi * dev_block, min((bi + 1) * dev_block, L)))
        else:
            raise ValueError(mode)

        for s, e, recompute in _spans_from_indices(recompute_idx, L):
            if recompute:
                _forward(model, chunk.tokens[s:e], caches)
                stats.recomputed_tokens += e - s
                stats.recompute_spans.append((ci, s, e))
            else:
                for i, c in enumerate(caches):
                    if isinstance(c, KVCache):
                        k_src, v_src = chunk.attn_kv[i]
                        k = rotate_keys_delta_module(ropes[i], k_src[..., s:e, :], delta)
                        insert_kv(c, k, v_src[..., s:e, :] + 0)
                    else:
                        rec = chunk.linear[i].slice(s, e)
                        _gdn_scan(layers[i].linear_attn, rec.qkv, rec.a, rec.b, c)
                stats.reused_tokens += e - s
        pos += L

    last_logits = _forward(model, suffix_tokens, caches)
    pos += len(suffix_tokens)
    mx.eval([c.state for c in caches])
    stats.prefill_seconds = time.perf_counter() - t0
    stats.total_tokens = pos
    return caches, last_logits, stats

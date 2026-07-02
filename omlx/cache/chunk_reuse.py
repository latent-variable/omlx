# SPDX-License-Identifier: Apache-2.0
"""Position-independent KV chunk reuse (experimental, opt-in via ChunkReuseConfig).

Full-attention models. Extract a chunk's per-layer KV from a donor context,
re-rotate keys by the position delta, splice into a new context, and
selectively recompute a small fraction of tokens (CacheBlend-style). Hybrid
(linear-attention) models are handled in chunk_reuse_hybrid.py.

Validated standalone before integration (LatentPlayground/kv-subset-cache):
Qwen2.5-7B / Llama-3.1-8B held 5/5 factual probes at ~15% recompute, 3.1-3.5x
prefill speedup on real files. See that repo's report.md.

Mechanism:
  - chunk KV extraction from a donor context
  - RoPE re-rotation of cached keys by a constant position delta (respects
    partial rotary, e.g. qwen3_5's rope.dims < head_dim; llama3 scaling)
  - blended prefill: seed a fresh cache with prefix KV computed normally,
    insert reused chunk KV, selectively recompute chosen token spans
    (reuse / edge / deviation / devblock)

Requires plain KVCache on every layer (get_layer_ropes guards this);
sliding-window / rotating caches are rejected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.models.cache import KVCache


# ---------------------------------------------------------------------------
# RoPE re-rotation
# ---------------------------------------------------------------------------

def rotate_keys_delta(keys: mx.array, delta: int, theta: float) -> mx.array:
    """Rotate post-RoPE keys by a constant position delta (plain NeoX RoPE).

    keys: (B, n_kv_heads, L, head_dim), already carrying rotation for their
    source positions. RoPE rotations compose additively, so moving a chunk
    from src_pos to src_pos+delta is a single constant-angle rotation per
    frequency, identical for every token in the chunk.

    NeoX-style (half-split) pairing: dims [0:d/2] pair with [d/2:d].
    Only valid for unscaled RoPE — prefer rotate_keys_delta_module, which
    uses the layer's own rope module (correct under llama3-style scaling too).
    """
    if delta == 0:
        return keys
    d = keys.shape[-1]
    half = d // 2
    freqs = mx.exp(-mx.arange(0.0, half) * (mx.log(mx.array(theta)) * 2.0 / d))
    angle = delta * freqs  # (half,)
    cos = mx.cos(angle).astype(keys.dtype)
    sin = mx.sin(angle).astype(keys.dtype)
    k1 = keys[..., :half]
    k2 = keys[..., half:]
    return mx.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)


def rotate_keys_delta_module(rope_module, keys: mx.array, delta: int) -> mx.array:
    """Rotate post-RoPE keys by a constant delta using the layer's own rope.

    Extracts cos/sin of the delta rotation per frequency by probing the module
    with a unit vector at a single position (a rotation of (1, 0) yields
    (cos θ, sin θ) directly). This picks up the module's true frequencies —
    including llama3-style scaling — and applies one constant-angle rotation
    to every token in the chunk. Exact for any half-split RoPE whose angle is
    linear in position. Negative deltas use cos(-θ)=cos θ, sin(-θ)=-sin θ.
    """
    if delta == 0:
        return keys
    if getattr(rope_module, "traditional", False):
        raise ValueError("traditional (interleaved) RoPE not supported yet")
    D = keys.shape[-1]
    # partial rotary (e.g. qwen3_5: rope.dims = head_dim/4): only the first
    # `dims` features are rotated; the rest pass through untouched
    dims = getattr(rope_module, "dims", D) or D
    half = dims // 2
    probe = mx.concatenate(
        [mx.ones((1, 1, 1, half), dtype=mx.float32), mx.zeros((1, 1, 1, half), dtype=mx.float32)],
        axis=-1,
    )
    out = rope_module(probe, offset=abs(delta))
    cos = out[0, 0, 0, :half].astype(keys.dtype)
    sin = out[0, 0, 0, half:].astype(keys.dtype)
    if delta < 0:
        sin = -sin
    k1 = keys[..., :half]
    k2 = keys[..., half:dims]
    rotated = mx.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    if dims == D:
        return rotated
    return mx.concatenate([rotated, keys[..., dims:]], axis=-1)


def get_layer_ropes(model) -> list:
    """Per-layer rope modules; validates the arch is chunk-reuse compatible.

    Requires every layer to use a plain KVCache (no sliding-window /
    rotating / mamba state) — mirrors the same restriction oMLX applies to
    its own sliceable-cache paths.
    """
    caches = model.make_cache() if hasattr(model, "make_cache") else [KVCache() for _ in inner_module(model).layers]
    bad = [i for i, c in enumerate(caches) if not isinstance(c, KVCache)]
    if bad:
        raise ValueError(
            f"model has non-plain cache layers (e.g. sliding-window) at {bad[:6]}...; "
            "chunk reuse requires plain KVCache on every layer"
        )
    ropes = []
    for layer in inner_module(model).layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        rope = getattr(attn, "rope", None)
        if rope is None:
            raise ValueError("layer without a rope module; unsupported arch")
        ropes.append(rope)
    return ropes


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------

def unwrap_model(model):
    """Return the real mlx-lm model, stripping oMLX's VLMModelAdapter wrapper.

    oMLX wraps every model (text or VLM) in a VLMModelAdapter for
    BatchGenerator compatibility; the adapter holds the real model on
    ``_vlm_model``. Chunk reuse operates on the unwrapped model.
    """
    return getattr(model, "_vlm_model", model)


def inner_module(model):
    """Module holding embed_tokens / layers / norm.

    Handles both flat models (qwen2/llama: ``model.model``) and the
    language_model nesting used by qwen3_5-family and VLMs
    (``model.language_model.model``). Assumes model is already unwrapped.
    """
    lm = getattr(model, "language_model", model)
    return lm.model


def make_caches(model) -> list:
    if hasattr(model, "make_cache"):
        return model.make_cache()  # correct per-layer types (incl. hybrids)
    return [KVCache() for _ in range(len(inner_module(model).layers))]


def copy_cache(cache: KVCache) -> KVCache:
    c = KVCache()
    if cache.offset > 0:
        k, v = cache.state
        c.state = (k + 0, v + 0)  # force materialized copies
    return c


def insert_kv(cache: KVCache, keys: mx.array, values: mx.array) -> None:
    """Append K/V into a cache without a forward pass (advances offset)."""
    cache.update_and_fetch(keys, values)


def extract_chunk_kv(caches: list[KVCache], start: int, end: int):
    """Per-layer (K, V) slices for token range [start, end) from a prefilled cache."""
    out = []
    for c in caches:
        k, v = c.state
        out.append((k[..., start:end, :] + 0, v[..., start:end, :] + 0))
    return out


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _forward(model, tokens: list[int], caches: list[KVCache]) -> mx.array:
    """Run the model over tokens, updating caches. Returns last-position logits."""
    logits = model(mx.array(tokens)[None], cache=caches)
    mx.eval(logits)
    return logits[0, -1, :]


def full_prefill(model, tokens: list[int]):
    """Baseline: prefill everything from scratch."""
    caches = make_caches(model)
    t0 = time.perf_counter()
    last_logits = _forward(model, tokens, caches)
    dt = time.perf_counter() - t0
    return caches, last_logits, dt


# ---------------------------------------------------------------------------
# Blended prefill
# ---------------------------------------------------------------------------

@dataclass
class ChunkReuse:
    """A chunk of tokens whose KV is reused from a donor context."""

    tokens: list[int]
    kv: list[tuple[mx.array, mx.array]]  # per-layer (K, V) from the donor
    src_pos: int  # chunk start position in the donor context


@dataclass
class BlendStats:
    total_tokens: int = 0
    reused_tokens: int = 0
    recomputed_tokens: int = 0
    prefill_seconds: float = 0.0
    recompute_spans: list = field(default_factory=list)

    @property
    def recompute_fraction(self) -> float:
        chunk = self.reused_tokens + self.recomputed_tokens
        return self.recomputed_tokens / chunk if chunk else 0.0


def _spans_from_indices(indices: set[int], length: int) -> list[tuple[int, int, bool]]:
    """Split [0, length) into ordered (start, end, recompute?) runs."""
    spans = []
    i = 0
    while i < length:
        j = i
        flag = i in indices
        while j < length and (j in indices) == flag:
            j += 1
        spans.append((i, j, flag))
        i = j
    return spans


def _probe_layer1_deviation(
    model, chunk: ChunkReuse, caches: list[KVCache], pos: int, ropes: list
) -> mx.array:
    """Per-token L2 deviation between fresh and reused layer-1 keys.

    Runs block 0 over the chunk (attending to the current cache contents,
    i.e. the fresh prefix) on throwaway cache copies, then compares block 1's
    fresh post-RoPE keys against the re-rotated reused ones. ~2/28 of a full
    forward for this model.
    """
    blocks = inner_module(model).layers
    h = inner_module(model).embed_tokens(mx.array(chunk.tokens)[None])
    probe_cache = copy_cache(caches[0])

    from mlx_lm.models.base import create_attention_mask

    mask = create_attention_mask(h, probe_cache)
    h1 = blocks[0](h, mask, probe_cache)

    attn1 = blocks[1].self_attn
    x = blocks[1].input_layernorm(h1)
    B, L, _ = x.shape
    k_fresh = attn1.k_proj(x).reshape(B, L, attn1.n_kv_heads, -1).transpose(0, 2, 1, 3)
    k_fresh = attn1.rope(k_fresh, offset=pos)

    k_reused = rotate_keys_delta_module(ropes[1], chunk.kv[1][0], pos - chunk.src_pos)
    dev = mx.sqrt(mx.sum((k_fresh - k_reused) ** 2, axis=(0, 1, 3)))  # (L,)
    mx.eval(dev)
    return dev


def blended_prefill(
    model,
    prefix_tokens: list[int],
    chunks: list[ChunkReuse],
    suffix_tokens: list[int],
    *,
    theta: float | None = None,  # unused; kept for call-site compatibility
    ropes: list | None = None,
    mode: str = "reuse",  # "reuse" | "edge" | "deviation" | "devblock"
    edge_k: int = 16,
    deviation_ratio: float = 0.15,
    dev_block: int = 32,
    interleave: list[list[int]] | None = None,
):
    """Prefill prefix normally, splice in reused chunk KV, recompute selected spans.

    interleave: optional list of token lists placed between chunks
    (len = len(chunks)-1 segments of fresh tokens between consecutive chunks).
    Layout: prefix | chunk0 | inter0 | chunk1 | ... | suffix.
    """
    stats = BlendStats()
    if ropes is None:
        ropes = get_layer_ropes(model)
    caches = make_caches(model)
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
        elif mode == "deviation":
            dev = _probe_layer1_deviation(model, chunk, caches, pos, ropes)
            n_sel = max(1, int(round(deviation_ratio * L)))
            top = mx.argsort(dev)[::-1][:n_sel].tolist()
            recompute_idx = set(top) | set(range(min(4, L)))  # always keep leading sink tokens
        elif mode == "devblock":
            # Block-granular deviation selection: contiguous recompute runs, so
            # the recompute cost is a handful of forward calls instead of
            # hundreds of scattered single-token passes.
            dev = _probe_layer1_deviation(model, chunk, caches, pos, ropes)
            n_blocks = (L + dev_block - 1) // dev_block
            scores = [
                float(mx.mean(dev[b * dev_block : min((b + 1) * dev_block, L)]).item())
                for b in range(n_blocks)
            ]
            n_sel = max(1, int(round(deviation_ratio * n_blocks)))
            top_blocks = sorted(range(n_blocks), key=lambda b: -scores[b])[:n_sel]
            recompute_idx = set()
            for b in set(top_blocks) | {0}:  # always recompute the leading block
                recompute_idx |= set(range(b * dev_block, min((b + 1) * dev_block, L)))
        else:
            raise ValueError(f"unknown mode {mode}")

        for s, e, recompute in _spans_from_indices(recompute_idx, L):
            if recompute:
                _forward(model, chunk.tokens[s:e], caches)
                stats.recomputed_tokens += e - s
                stats.recompute_spans.append((ci, s, e))
            else:
                for layer, cache in enumerate(caches):
                    k_src, v_src = chunk.kv[layer]
                    k = rotate_keys_delta_module(ropes[layer], k_src[..., s:e, :], delta)
                    insert_kv(cache, k, v_src[..., s:e, :] + 0)
                stats.reused_tokens += e - s
        pos += L

        if interleave and ci < len(chunks) - 1 and interleave[ci]:
            _forward(model, interleave[ci], caches)
            pos += len(interleave[ci])

    # Empty suffix = engine mode: caller (oMLX) prefills the suffix itself.
    last_logits = _forward(model, suffix_tokens, caches) if suffix_tokens else None
    pos += len(suffix_tokens)

    mx.eval([c.state for c in caches if c.state])
    stats.prefill_seconds = time.perf_counter() - t0
    stats.total_tokens = pos
    return caches, last_logits, stats


# ---------------------------------------------------------------------------
# Generation + comparison metrics
# ---------------------------------------------------------------------------

def greedy_generate(model, caches: list[KVCache], first_logits: mx.array, max_tokens: int, eos_ids: set[int]):
    out = []
    logits = first_logits
    for _ in range(max_tokens):
        tok = int(mx.argmax(logits).item())
        if tok in eos_ids:
            break
        out.append(tok)
        logits = _forward(model, [tok], caches)
    return out


def kl_divergence(p_logits: mx.array, q_logits: mx.array) -> float:
    """KL(P || Q) between two logit vectors."""
    p = p_logits - mx.logsumexp(p_logits)
    q = q_logits - mx.logsumexp(q_logits)
    return float(mx.sum(mx.exp(p) * (p - q)).item())


def match_length(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

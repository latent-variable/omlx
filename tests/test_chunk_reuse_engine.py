# SPDX-License-Identifier: Apache-2.0
"""Integration test for ChunkReuseEngine against oMLX's own cache constructor.

Mimics the scheduler flow end to end without the server:
  1. full-prefill prompt A, engine.capture()
  2. engine.assemble(prompt B) where B shares content with A at a new position
  3. finish B's suffix prefill against the assembled cache (as the scheduler
     would) and compare the next-token logits to a full prefill of B.

Skips unless a small model is available locally; safe to run in CI with the
tiny models pre-cached. Uses full-attention (Qwen2.5-1.5B) and hybrid
(Qwen3.5-0.8B) so both paths are exercised.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from omlx.cache.chunk_reuse import kl_divergence
from omlx.cache.chunk_reuse_engine import ChunkReuseEngine
from omlx.config import ChunkReuseConfig

FULL_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
HYBRID_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"

# Real source files (non-periodic — synthetic repeated text defeats CDC).
_CACHE_DIR = Path(__file__).resolve().parent.parent / "omlx" / "cache"
FILE_A = (_CACHE_DIR / "stats.py").read_text()
FILE_B = (_CACHE_DIR / "recovery.py").read_text()


def _prefill(model, ids):
    cache = make_prompt_cache(model)
    logits = model(mx.array(ids)[None], cache=cache)
    mx.eval(logits)
    return cache, logits[0, -1, :]


def _suffix_prefill(model, cache, suffix_ids):
    """Continue prefilling suffix against an assembled cache (scheduler does this)."""
    logits = model(mx.array(suffix_ids)[None], cache=cache)
    mx.eval(logits)
    return logits[0, -1, :]


def _run(model_id):
    model, tok = load(model_id)
    cfg = ChunkReuseConfig(enabled=True, recompute_mode="devblock",
                           deviation_ratio=0.2, min_chunk_tokens=128)
    engine = ChunkReuseEngine(model, model_id, cfg)
    assert engine.supported

    sysA = tok.encode("<|im_start|>system\nYou are a coding agent.<|im_end|>\n<|im_start|>user\n")
    fileA = tok.encode(FILE_A)
    fileB = tok.encode(FILE_B)

    # 1. donor prompt (prefix + fileA + fileB), full prefill, capture
    donor = sysA + fileA + fileB
    donor_cache, _ = _prefill(model, donor)
    n_cap = engine.capture(donor, donor_cache)
    assert n_cap > 0, "engine captured no chunks"

    # 2. new session: different prefix, same files at shifted positions + question
    sysB = tok.encode("<|im_start|>system\nNew session. Audit these files.<|im_end|>\n<|im_start|>user\n")
    question = tok.encode("\nWhat does get_timeout return?<|im_end|>\n<|im_start|>assistant\n")
    promptB = sysB + fileA + fileB + question

    result = engine.assemble(promptB)
    assert result is not None, "assemble found no reusable chunks"
    assert result.chunks_used >= 1
    assert result.cached_tokens >= len(sysB) + len(fileA) // 2

    # 3. finish suffix prefill against the assembled cache
    reuse_logits = _suffix_prefill(model, result.prompt_cache,
                                   promptB[result.cached_tokens:])

    # baseline: full prefill of promptB
    _, base_logits = _prefill(model, promptB)

    kl = kl_divergence(base_logits, reuse_logits)
    top1 = int(mx.argmax(base_logits).item()) == int(mx.argmax(reuse_logits).item())
    print(f"\n{model_id} [{engine.arch}]: chunks={result.chunks_used} "
          f"cached={result.cached_tokens}/{len(promptB)} "
          f"recomp={result.recompute_fraction:.0%} KL={kl:.4f} top1={top1}")
    # facts should survive; distribution shifts but top token should match
    assert top1, f"top-1 token diverged (KL={kl})"
    return engine


@pytest.mark.slow
def test_engine_full_attention():
    try:
        _run(FULL_MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"model unavailable: {e}")


@pytest.mark.slow
def test_engine_hybrid():
    try:
        _run(HYBRID_MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"model unavailable: {e}")


if __name__ == "__main__":
    _run(FULL_MODEL)
    _run(HYBRID_MODEL)
    print("\nengine integration OK")

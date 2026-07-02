# SPDX-License-Identifier: Apache-2.0
"""Integration test for chunk reuse on the mlx-vlm runtime (qwen3_5 hybrids).

Mirrors test_chunk_reuse_engine.py but loads the model exactly the way oMLX
serves Qwen3.5/3.6: through mlx_vlm.utils.load wrapped in VLMModelAdapter
(M-RoPE rotary_emb, mlx-vlm GatedDeltaNet). Additionally validates that the
recording capture prefill (replicated layer math) matches the stock forward.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache

from omlx.cache.chunk_reuse import kl_divergence
from omlx.cache.chunk_reuse_engine import ChunkReuseEngine
from omlx.config import ChunkReuseConfig

VLM_HYBRID_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"

_CACHE_DIR = Path(__file__).resolve().parent.parent / "omlx" / "cache"
FILE_A = (_CACHE_DIR / "stats.py").read_text()
FILE_B = (_CACHE_DIR / "recovery.py").read_text()


def _load_via_mlx_vlm(model_id):
    from mlx_vlm.utils import load as vlm_load

    from omlx.models.vlm import VLMModelAdapter

    vlm_model, processor = vlm_load(model_id)
    tok = getattr(processor, "tokenizer", processor)
    return VLMModelAdapter(vlm_model), tok


def _prefill(adapter, ids):
    cache = make_prompt_cache(adapter)
    logits = adapter(mx.array(ids)[None], cache=cache)
    mx.eval(logits)
    return cache, logits[0, -1, :]


def _suffix_prefill(adapter, cache, suffix_ids):
    logits = adapter(mx.array(suffix_ids)[None], cache=cache)
    mx.eval(logits)
    return logits[0, -1, :]


def _run(model_id):
    adapter, tok = _load_via_mlx_vlm(model_id)
    cfg = ChunkReuseConfig(enabled=True, recompute_mode="edge",
                           edge_tokens=32, min_chunk_tokens=128)
    engine = ChunkReuseEngine(adapter, model_id, cfg)
    assert engine.arch == "vlm_hybrid", f"arch={engine.arch}"

    sysA = tok.encode("<|im_start|>system\nYou are a coding agent.<|im_end|>\n<|im_start|>user\n")
    fileA = tok.encode(FILE_A)
    fileB = tok.encode(FILE_B)

    # 1. donor prompt, full prefill via the adapter (stock path), capture
    donor = sysA + fileA + fileB
    donor_cache, donor_logits = _prefill(adapter, donor)

    # 1a. the recording capture prefill must reproduce the stock forward
    from omlx.cache import chunk_reuse_vlm as crv
    cap = crv.capture_prefill(adapter, donor)
    cap_kl = kl_divergence(donor_logits, cap.logits_last)
    cap_top1 = int(mx.argmax(donor_logits).item()) == int(mx.argmax(cap.logits_last).item())
    print(f"\ncapture-vs-stock: KL={cap_kl:.6f} top1={cap_top1}")
    assert cap_top1 and cap_kl < 1e-2, "recording prefill diverged from stock forward"

    n_cap = engine.capture(donor, donor_cache)
    assert n_cap > 0, "engine captured no chunks"

    # 2. new session: different prefix, same files at shifted positions
    sysB = tok.encode("<|im_start|>system\nNew session. Audit these files.<|im_end|>\n<|im_start|>user\n")
    question = tok.encode("\nWhat does get_timeout return?<|im_end|>\n<|im_start|>assistant\n")
    promptB = sysB + fileA + fileB + question

    result = engine.assemble(promptB)
    assert result is not None, "assemble found no reusable chunks"
    assert result.chunks_used >= 1
    assert result.cached_tokens >= len(sysB) + len(fileA) // 2

    # 3. finish suffix prefill against the assembled cache via the adapter
    reuse_logits = _suffix_prefill(adapter, result.prompt_cache,
                                   promptB[result.cached_tokens:])

    _, base_logits = _prefill(adapter, promptB)

    kl = kl_divergence(base_logits, reuse_logits)
    top1 = int(mx.argmax(base_logits).item()) == int(mx.argmax(reuse_logits).item())
    print(f"{model_id} [{engine.arch}]: chunks={result.chunks_used} "
          f"cached={result.cached_tokens}/{len(promptB)} "
          f"recomp={result.recompute_fraction:.0%} KL={kl:.4f} top1={top1}")
    assert top1, f"top-1 token diverged (KL={kl})"
    return engine


@pytest.mark.slow
def test_engine_vlm_hybrid():
    try:
        _run(VLM_HYBRID_MODEL)
    except ImportError as e:
        pytest.skip(f"mlx-vlm unavailable: {e}")
    except Exception as e:  # noqa: BLE001
        if "not found" in str(e).lower() or "unavailable" in str(e).lower():
            pytest.skip(f"model unavailable: {e}")
        raise


if __name__ == "__main__":
    _run(VLM_HYBRID_MODEL)
    print("\nvlm engine integration OK")

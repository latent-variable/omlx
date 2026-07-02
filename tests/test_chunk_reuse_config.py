# SPDX-License-Identifier: Apache-2.0
"""Tests for the chunk-reuse config flag and module surface.

Model-level exactness/quality is validated in the standalone prototype
(LatentPlayground/kv-subset-cache); these guard the oMLX integration surface:
the toggle defaults off, env override works, and the ported modules import
with the expected public API.
"""

import os
from unittest.mock import patch

from omlx.config import ChunkReuseConfig, OMLXConfig


def test_chunk_reuse_default_off():
    cfg = OMLXConfig()
    assert cfg.chunk_reuse.enabled is False
    assert cfg.chunk_reuse.recompute_mode == "edge"
    assert cfg.chunk_reuse.min_chunk_tokens == 128


def test_chunk_reuse_env_override():
    with patch.dict(os.environ, {"OMLX_CHUNK_REUSE": "true",
                                 "OMLX_CHUNK_REUSE_MODE": "devblock"}):
        cfg = OMLXConfig.from_env()
    assert cfg.chunk_reuse.enabled is True
    assert cfg.chunk_reuse.recompute_mode == "devblock"


def test_chunk_reuse_env_default_false():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OMLX_CHUNK_REUSE", None)
        cfg = OMLXConfig.from_env()
    assert cfg.chunk_reuse.enabled is False


def test_full_attention_module_surface():
    from omlx.cache import chunk_reuse as cr

    for name in ("rotate_keys_delta_module", "get_layer_ropes",
                 "blended_prefill", "extract_chunk_kv", "ChunkReuse",
                 "full_prefill", "BlendStats"):
        assert hasattr(cr, name), name


def test_hybrid_module_surface():
    from omlx.cache import chunk_reuse_hybrid as crh

    for name in ("capture_prefill", "extract_hybrid_chunk",
                 "hybrid_blended_prefill", "get_hybrid_layout", "HybridChunk"):
        assert hasattr(crh, name), name


def test_config_dataclass_shape():
    c = ChunkReuseConfig(enabled=True, recompute_mode="edge", edge_tokens=16)
    assert c.enabled and c.edge_tokens == 16 and c.deviation_ratio == 0.15

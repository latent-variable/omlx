# SPDX-License-Identifier: Apache-2.0
"""
Prompt diagnostics helpers for cache investigations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPT_DUMP_DIR = Path.home() / ".omlx" / "logs" / "prompt_dumps"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    return repr(value)


def dump_prompt_snapshot(
    request_id: str,
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None,
    prompt: str | list[int],
    tokenizer: Any = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a prompt snapshot for a single internal request."""
    PROMPT_DUMP_DIR.mkdir(parents=True, exist_ok=True)

    prompt_text = prompt if isinstance(prompt, str) else None
    prompt_token_ids = prompt if isinstance(prompt, list) else None

    if prompt_text is not None and tokenizer is not None:
        try:
            prompt_token_ids = tokenizer.encode(prompt_text)
        except Exception as e:
            logger.debug("Prompt snapshot encode failed for %s: %s", request_id, e)

    prompt_hash = None
    if prompt_token_ids is not None:
        prompt_hash = hashlib.sha256(
            json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    elif prompt_text is not None:
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    out_path = PROMPT_DUMP_DIR / f"{request_id}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "request_id": request_id,
                "model_name": model_name,
                "messages": _json_safe(messages),
                "tools": _json_safe(tools),
                "chat_template_kwargs": _json_safe(chat_template_kwargs),
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_count": (
                    len(prompt_token_ids) if prompt_token_ids is not None else None
                ),
                "prompt_hash": prompt_hash,
                "extra": _json_safe(extra or {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    logger.info("Prompt snapshot saved for %s: %s", request_id, out_path)
    return out_path

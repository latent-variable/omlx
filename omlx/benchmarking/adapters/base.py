# SPDX-License-Identifier: Apache-2.0
"""Base harness adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class HarnessAdapterConfig:
    harness: str
    cwd: str
    model_id: str
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class HarnessAdapter(Protocol):
    config: HarnessAdapterConfig

    def start(self) -> None:
        """Start or prepare the harness session."""

    def stop(self) -> None:
        """Stop the harness session."""

    def send_turn(self, prompt: str) -> dict[str, Any]:
        """Send one logical turn and return adapter-specific metadata."""

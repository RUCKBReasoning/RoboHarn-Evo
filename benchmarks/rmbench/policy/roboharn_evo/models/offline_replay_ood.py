from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OfflineReplayOODConfig:
    OOD_scenario: str = "none"
    reason: str = "offline replay local OOD fallback"
    confidence: float = 0.0


class OfflineReplayOODAdapter:
    """Local OOD stub used by offline trajectory replay.

    Offline replay is primarily used to check planner, memory, perception, and
    visualization flow over recorded observations. Keeping OOD local prevents a
    VLM OOD request after every recorded action frame.
    """

    def __init__(self, config: OfflineReplayOODConfig) -> None:
        self.config = config

    def evaluate_ood(self, *, skill_payload: str, media: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        del skill_payload, media
        return {
            "OOD_scenario": self.config.OOD_scenario,
            "reason": self.config.reason,
            "confidence": max(0.0, min(1.0, float(self.config.confidence))),
        }

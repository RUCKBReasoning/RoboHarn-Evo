from __future__ import annotations

from typing import Any


PREFERRED_ARMS = {"left", "right", "either"}
SELECTED_ARMS = {"left", "right", "none"}


def normalize_preferred_arm(value: Any, *, default: str = "either") -> str:
    normalized = _normalize_arm_token(value)
    if normalized in PREFERRED_ARMS:
        return normalized
    return default if default in PREFERRED_ARMS else "either"


def normalize_selected_arm(value: Any, *, default: str = "none") -> str:
    normalized = _normalize_arm_token(value)
    if normalized in SELECTED_ARMS:
        return normalized
    return default if default in SELECTED_ARMS else "none"


def normalize_physical_arm(value: Any) -> str:
    normalized = _normalize_arm_token(value)
    return normalized if normalized in {"left", "right", "both"} else ""


def _normalize_arm_token(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "left_arm": "left",
        "right_arm": "right",
        "either_arm": "either",
        "any": "either",
        "any_arm": "either",
        "no_arm": "none",
        "neither": "none",
    }
    return aliases.get(normalized, normalized)

from __future__ import annotations

from typing import Any


def public_manipulation_state(value: Any) -> dict[str, Any]:
    """Project runtime manipulation state into planner-visible state.

    Operation candidate IDs are runtime-private implementation details.  The
    planner selects stable public targets and strategies; runtime selects the
    arm-specific executable candidate underneath them.
    """

    projected = _without_private_candidate_ids(value)
    return projected if isinstance(projected, dict) else {}


def _without_private_candidate_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_private_candidate_ids(item)
            for key, item in value.items()
            if not str(key).lower().endswith("candidate_id")
        }
    if isinstance(value, list):
        return [
            _without_private_candidate_ids(item)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _without_private_candidate_ids(item)
            for item in value
        ]
    return value

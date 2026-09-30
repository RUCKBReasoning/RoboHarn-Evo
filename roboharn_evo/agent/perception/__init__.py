"""Perception public API with lazy exports.

Submodules such as ``query_normalization`` are imported by the candidate
contract itself.  Avoiding eager grounding imports here prevents a circular
dependency while preserving the existing four public names.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "SAM2SegmentationClient": (
        "roboharn_evo.agent.perception.sam2_client",
        "SAM2SegmentationClient",
    ),
    "SAM3SegmentationClient": (
        "roboharn_evo.agent.perception.sam3_client",
        "SAM3SegmentationClient",
    ),
    "ground_segmentation_result": (
        "roboharn_evo.agent.perception.grounding",
        "ground_segmentation_result",
    ),
    "SceneMemoryTracker": (
        "roboharn_evo.agent.perception.scene_memory",
        "SceneMemoryTracker",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

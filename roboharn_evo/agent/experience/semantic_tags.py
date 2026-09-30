from __future__ import annotations

import re
from typing import Any


TASK_FAMILIES = {
    "pick_and_place",
    "open_drawer",
    "close_drawer",
    "articulated_object",
    "tool_use",
    "other",
}

SUBTASK_TYPES = {
    "grasp",
    "place",
    "open",
    "close",
    "move",
    "align",
    "reobserve",
    "recover",
    "other",
}

STATE_TAG_KEYS = ("object_state", "visibility_state", "gripper_state", "motion_state")

STATE_TAG_VALUES = {
    "object_state": {
        "aligned",
        "contacting",
        "dropped",
        "grasped",
        "in_target",
        "inserted",
        "misaligned",
        "obstructed",
        "other",
        "out_of_target",
        "placed",
        "released",
        "target_in_gripper",
    },
    "visibility_state": {
        "object_not_visible",
        "occluded",
        "other",
        "out_of_view",
        "partially_visible",
        "target_visible",
        "visible",
        "visible_object",
    },
    "gripper_state": {
        "closed",
        "empty",
        "holding_object",
        "open",
        "other",
        "partially_closed",
        "slipping",
    },
    "motion_state": {
        "blocked",
        "moving",
        "no_progress",
        "other",
        "overshot",
        "stable",
        "stalled",
    },
}

TAG_SOURCES = {"planner_vlm", "ood_vlm", "recovery_vlm", "fallback", "unknown"}


def _clean_tag(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _normalize_source(value: Any, *, default_source: str) -> str:
    source = _clean_tag(value) or default_source
    return source if source in TAG_SOURCES else default_source


def _normalize_enum(value: Any, allowed: set[str], *, invalid_value: str) -> str:
    if value is None:
        return ""
    tag = _clean_tag(value)
    if not tag:
        return ""
    return tag if tag in allowed else invalid_value


def empty_semantic_tags(*, tag_source: str = "unknown") -> dict[str, Any]:
    return {
        "task_family": "",
        "subtask_type": "",
        "state_tags": {key: "" for key in STATE_TAG_KEYS},
        "tag_source": _normalize_source(tag_source, default_source="unknown"),
    }


def normalize_semantic_tags(raw: Any, *, default_source: str = "unknown") -> dict[str, Any]:
    """Validate VLM/runtime semantic tags against the local taxonomy."""

    if not isinstance(raw, dict):
        return empty_semantic_tags()

    state_tags_payload = raw.get("state_tags", {})
    if not isinstance(state_tags_payload, dict):
        state_tags_payload = {}

    normalized = empty_semantic_tags(
        tag_source=_normalize_source(raw.get("tag_source"), default_source=default_source)
    )
    normalized["task_family"] = _normalize_enum(
        raw.get("task_family"),
        TASK_FAMILIES,
        invalid_value="other",
    )
    normalized["subtask_type"] = _normalize_enum(
        raw.get("subtask_type"),
        SUBTASK_TYPES,
        invalid_value="other",
    )
    normalized["state_tags"] = {
        key: _normalize_enum(
            state_tags_payload.get(key),
            STATE_TAG_VALUES[key],
            invalid_value="",
        )
        for key in STATE_TAG_KEYS
    }
    return normalized


def has_semantic_tags(tags: Any) -> bool:
    normalized = normalize_semantic_tags(tags)
    if normalized["task_family"] or normalized["subtask_type"]:
        return True
    return any(str(value).strip() for value in normalized["state_tags"].values())


def merge_semantic_tags(
    base: Any,
    update: Any,
    *,
    overwrite: bool = True,
    merge_task_fields: bool = True,
    merge_state_fields: bool = True,
) -> dict[str, Any]:
    """Merge validated tag dictionaries while preserving the allowlisted schema."""

    result = normalize_semantic_tags(base)
    incoming = normalize_semantic_tags(update)
    contributed = False

    if merge_task_fields:
        for key in ("task_family", "subtask_type"):
            value = str(incoming.get(key, "")).strip()
            if value and (overwrite or not str(result.get(key, "")).strip()):
                result[key] = value
                contributed = True

    if merge_state_fields:
        result_state_tags = dict(result.get("state_tags", {}))
        incoming_state_tags = dict(incoming.get("state_tags", {}))
        for key in STATE_TAG_KEYS:
            value = str(incoming_state_tags.get(key, "")).strip()
            if value and (overwrite or not str(result_state_tags.get(key, "")).strip()):
                result_state_tags[key] = value
                contributed = True
        result["state_tags"] = result_state_tags

    source = str(incoming.get("tag_source", "unknown")).strip()
    if contributed and source != "unknown":
        result["tag_source"] = source
    elif contributed and str(result.get("tag_source", "unknown")).strip() == "unknown":
        result["tag_source"] = source
    return normalize_semantic_tags(result)


def should_use_fallback_tags(*tag_sets: Any) -> bool:
    return not any(has_semantic_tags(tags) for tags in tag_sets)


def fallback_semantic_tags(
    *,
    global_task: str = "",
    current_subtask: str = "",
    signal_name: str = "",
    state_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build only machine-signal state tags when no structured Agent tags exist.

    Task and subtask text are deliberately not interpreted here.  Their semantic
    categories must come from validated structured Agent/VLM output, rather than
    runtime object-name or verb heuristics.
    """

    del global_task, current_subtask

    inferred_state_tags = {key: "" for key in STATE_TAG_KEYS}
    signal = _clean_tag(signal_name)
    if signal in {"stall_detected", "motion_blocked", "step_budget_exhausted"}:
        inferred_state_tags["motion_state"] = "stalled" if signal != "motion_blocked" else "blocked"
    if signal == "object_not_visible":
        inferred_state_tags["visibility_state"] = "object_not_visible"
    if signal == "grasp_lost":
        inferred_state_tags["object_state"] = "dropped"
        inferred_state_tags["gripper_state"] = "empty"
    if state_tags:
        inferred_state_tags.update(state_tags)

    return normalize_semantic_tags(
        {
            "task_family": "",
            "subtask_type": "",
            "state_tags": inferred_state_tags,
            "tag_source": "fallback",
        }
    )

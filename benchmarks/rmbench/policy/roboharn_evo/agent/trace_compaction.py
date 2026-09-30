from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

from .planner_state_projection import (
    public_manipulation_state,
)


LEGACY_TRACE_PAYLOAD_MODE = "legacy"
RAW_TRACE_PAYLOAD_MODE = "raw"
COMPACT_TRACE_PAYLOAD_MODE = "compact_v1"
VALID_TRACE_PAYLOAD_MODES = {
    LEGACY_TRACE_PAYLOAD_MODE,
    RAW_TRACE_PAYLOAD_MODE,
    COMPACT_TRACE_PAYLOAD_MODE,
}

TRACE_COMPACT_SCHEMA_VERSION = 1
OBSERVATION_PREPROCESS_TRACE_SCHEMA = (
    "trace/observation_preprocess/compact_v1"
)
SCENE_MEMORY_UPDATE_TRACE_SCHEMA = "trace/scene_memory_update/compact_v1"
OPERATION_CANDIDATE_SELECTED_TRACE_SCHEMA = (
    "trace/operation_candidate_selected/compact_v1"
)
RECOVERY_RESULT_TRACE_SCHEMA = "trace/recovery_result/compact_v1"

_POSE_KEYS = (
    "object_contact_pose",
    "tcp_pose",
    "ee_target_pose",
    "approach_pose",
)


def normalize_trace_payload_mode(
    value: Any,
    *,
    field_name: str = "trace_payload_mode",
    default: str = LEGACY_TRACE_PAYLOAD_MODE,
) -> str:
    """Normalize a trace serialization mode without changing runtime state."""

    normalized = str(value or default).strip().lower().replace("-", "_")
    if normalized not in VALID_TRACE_PAYLOAD_MODES:
        choices = ", ".join(sorted(VALID_TRACE_PAYLOAD_MODES))
        raise ValueError(
            f"{field_name} must be one of {{{choices}}}; got {value!r}"
        )
    return normalized


def copy_legacy_trace_payload(payload: Any) -> Any:
    """Return an isolation-safe copy for the legacy/raw serialization path."""

    return deepcopy(payload)


def compact_trace_payload(
    event_name: Any,
    payload: Any,
    *,
    mode: Any = COMPACT_TRACE_PAYLOAD_MODE,
) -> Any:
    """Project one event payload into its trace-only representation.

    This function is deliberately pure.  In particular, the dictionaries held
    by Working Memory, Scene Memory, grounding, and the executor are never
    modified.  Unknown event kinds retain their legacy shape so rollout
    consumers can migrate one event family at a time.
    """

    normalized_mode = normalize_trace_payload_mode(mode)
    if normalized_mode in {
        LEGACY_TRACE_PAYLOAD_MODE,
        RAW_TRACE_PAYLOAD_MODE,
    }:
        return copy_legacy_trace_payload(payload)

    event = str(event_name or "").strip()
    if event == "observation_preprocess":
        return compact_observation_preprocess_event(payload)
    if event == "scene_memory_update":
        return compact_scene_memory_update_event(payload)
    if event == "operation_candidate_selected":
        return compact_operation_candidate_selected_event(payload)
    if event == "recovery_result":
        return compact_recovery_result_event(payload)
    return copy_legacy_trace_payload(payload)


def compact_trace_event(
    event_name: Any,
    payload: Any,
    *,
    mode: Any = COMPACT_TRACE_PAYLOAD_MODE,
) -> Any:
    """Compact an event payload that does not include the JSONL envelope.

    ``event_name`` is kept separate so the same pure function can be used by
    both JSONL writers before either writer adds timestamp/episode metadata.
    """

    return compact_trace_payload(event_name, payload, mode=mode)


def compact_jsonl_event(
    event: Any,
    *,
    mode: Any = COMPACT_TRACE_PAYLOAD_MODE,
) -> Any:
    """Offline convenience wrapper for a complete, already-written event."""

    if not isinstance(event, dict):
        return copy_legacy_trace_payload(event)
    return compact_trace_payload(event.get("event"), event, mode=mode)


def compact_observation_preprocess_event(payload: Any) -> dict[str, Any]:
    """Build a bounded audit view of raw segmentation and RGB-D grounding."""

    source = payload if isinstance(payload, dict) else {}
    segments = [
        item
        for item in source.get("segmentation", []) or []
        if isinstance(item, dict)
    ]
    compact_results = [_compact_segmentation_result(item) for item in segments]
    robot_state = _find_robot_state(source, segments)

    result: dict[str, Any] = {
        "schema": OBSERVATION_PREPROCESS_TRACE_SCHEMA,
        "schema_version": TRACE_COMPACT_SCHEMA_VERSION,
        "full_runtime_payload_retained": True,
    }
    _copy_present(
        source,
        result,
        (
            "event",
            "timestamp",
            "episode_id",
            "seed",
            "env_step",
            "stage",
            "latency_sec",
            "observation_generation",
            "observation_capture_id",
            "generation",
            "capture_id",
            "agent_identity_binding_required",
        ),
    )
    if robot_state:
        result["robot_state"] = robot_state

    queries = [
        _compact_perception_query(item)
        for item in source.get("perception_queries", []) or []
        if isinstance(item, dict)
    ]
    result["perception_queries"] = queries
    result["cameras"] = sorted(
        {
            str(item.get("camera", "") or "").strip()
            for item in segments
            if str(item.get("camera", "") or "").strip()
        }
    )
    result["segmentation_result_count"] = len(segments)
    result["successful_segmentation_count"] = sum(
        item.get("success") is True for item in segments
    )
    result["failed_segmentation_count"] = sum(
        item.get("success") is not True for item in segments
    )
    result["detection_count"] = sum(
        len(item.get("detections", []) or [])
        if isinstance(item.get("detections"), list)
        else int(bool(item.get("success")))
        for item in segments
    )
    result["dropped_detection_count"] = sum(
        len(item.get("dropped_detections", []) or [])
        for item in segments
        if isinstance(item.get("dropped_detections"), list)
    )
    # Keep the legacy collection key so report/audit/overlay readers can keep
    # iterating results; only the elements change to the compact_v1 schema.
    result["segmentation"] = compact_results
    candidate_index = _hoist_operation_candidate_index(compact_results)
    if candidate_index:
        result["operation_candidate_index"] = candidate_index

    errors: list[dict[str, Any]] = []
    for key in ("perception_error", "oracle_error", "error"):
        message = _short_text(source.get(key), 240)
        if message:
            errors.append({"scope": "observation", "kind": key, "error": message})
    errors.extend(_segmentation_error_summaries(segments))
    if errors:
        result["errors"] = errors[:8]
    binding = _compact_binding_requirement(
        source.get("perception_binding_requirement")
    )
    if binding:
        result["perception_binding_requirement"] = binding
    return result


def compact_scene_memory_update_event(payload: Any) -> dict[str, Any]:
    """Build a public, bounded Scene Memory update for rollout auditing."""

    source = payload if isinstance(payload, dict) else {}
    nested = source.get("scene_memory")
    scene = nested if isinstance(nested, dict) else source
    result: dict[str, Any] = {
        "schema": SCENE_MEMORY_UPDATE_TRACE_SCHEMA,
        "schema_version": TRACE_COMPACT_SCHEMA_VERSION,
        "full_runtime_payload_retained": True,
    }
    if scene is not source:
        _copy_present(
            source,
            result,
            (
                "event",
                "timestamp",
                "episode_id",
                "seed",
                "env_step",
                "observation_generation",
                "observation_capture_id",
            ),
        )

    compact_scene = _compact_scene_memory(scene)
    robot_state = _compact_robot_state(
        source.get("robot_state", scene.get("robot_state"))
    )
    if robot_state:
        result["robot_state"] = robot_state
    result["scene_memory"] = compact_scene
    return result


def compact_operation_candidate_selected_event(
    payload: Any,
) -> dict[str, Any]:
    """Record one selected candidate's executable geometry exactly once.

    Selection and execution events after this record should refer to the
    candidate by ID rather than embedding another complete pose bundle.  This
    pure projection guarantees one bundle inside this event; cross-event
    routing remains the responsibility of the trace writer.
    """

    source = payload if isinstance(payload, dict) else {}
    raw_candidate = source.get("candidate")
    if not isinstance(raw_candidate, dict):
        raw_candidate = source.get("operation_candidate")
    candidate = raw_candidate if isinstance(raw_candidate, dict) else source
    result: dict[str, Any] = {
        "schema": OPERATION_CANDIDATE_SELECTED_TRACE_SCHEMA,
        "schema_version": TRACE_COMPACT_SCHEMA_VERSION,
    }
    _copy_present(
        source,
        result,
        (
            "event",
            "timestamp",
            "episode_id",
            "seed",
            "env_step",
            "observation_generation",
            "observation_capture_id",
            "dispatch_index",
            "dispatch_env_step",
            "result_env_step",
        ),
    )
    _copy_present(
        candidate,
        result,
        (
            "candidate_id",
            "instance_id",
            "track_id",
            "object_id",
            "source_detection_index",
            "source_candidate_index",
            "source_camera",
            "source",
            "source_object_id",
            "source_text_prompt",
            "camera",
            "arm",
            "action_mode",
            "target_id",
            "geometry_source",
            "candidate_geometry_revision",
            "supporting_cameras",
            "priority",
            "score",
        ),
    )
    for key in _POSE_KEYS:
        pose = _numeric_vector(candidate.get(key), length=7)
        if pose:
            result[key] = pose
    robot_state = _compact_robot_state(source.get("robot_state"))
    if robot_state:
        result["robot_state"] = robot_state
    return result


def compact_recovery_result_event(payload: Any) -> dict[str, Any]:
    """Remove selected-candidate pose copies from recovery result details.

    Requested targets, observed/executed poses, errors, reachability, collision,
    stall, skip, and batch-halt evidence are intentionally retained.  Only the
    executable candidate's four-pose bundle is canonicalized into the separate
    ``operation_candidate_selected`` event.
    """

    source = payload if isinstance(payload, dict) else {}
    result = deepcopy(source)
    result["schema"] = RECOVERY_RESULT_TRACE_SCHEMA
    result["schema_version"] = TRACE_COMPACT_SCHEMA_VERSION
    raw_results = result.get("results")
    if not isinstance(raw_results, list):
        return result
    for item in raw_results:
        if not isinstance(item, dict) or "details" not in item:
            continue
        item["details"] = _without_repeated_candidate_geometry(
            item.get("details")
        )
    return result


def _compact_segmentation_result(item: dict[str, Any]) -> dict[str, Any]:
    detections = [
        detection
        for detection in item.get("detections", []) or []
        if isinstance(detection, dict)
    ]
    if not detections and item.get("success") is True and isinstance(
        item.get("grounding_3d"), dict
    ):
        # Older preprocessors only exposed the selected detection at result
        # level.  Treat it as one detection without duplicating its grounding.
        detections = [item]
    selected_index = _selected_detection_index(item, detections)
    compact: dict[str, Any] = {}
    _copy_present(
        item,
        compact,
        (
            "object_id",
            "text_prompt",
            "backend",
            "camera",
            "success",
            "entity_scope",
            "placement_relation",
            "expected_count",
            "identity_binding_required",
            "identity_binding_error",
        ),
    )
    role = item.get("query_role", item.get("role"))
    if role not in (None, ""):
        compact["query_role"] = deepcopy(role)
    instance_hint = item.get("query_instance_hint", item.get("instance_hint"))
    if instance_hint not in (None, ""):
        compact["query_instance_hint"] = deepcopy(instance_hint)
    instance_ref = item.get("instance_ref")
    if instance_ref not in (None, ""):
        compact["instance_ref"] = deepcopy(instance_ref)
    reason = _short_text(item.get("query_reason", item.get("reason")), 200)
    if reason:
        compact["query_reason"] = reason
    compact["detection_count"] = len(detections)
    dropped_detection_count = len(
        item.get("dropped_detections", []) or []
    ) if isinstance(item.get("dropped_detections"), list) else 0
    if dropped_detection_count:
        compact["dropped_detection_count"] = dropped_detection_count
    if selected_index is not None:
        compact["selected_detection_index"] = selected_index
    error = _short_text(item.get("error"), 240)
    if error:
        compact["error"] = error
    compact["detections"] = [
        _compact_detection(detection)
        for detection in detections
    ]
    return compact


def _compact_detection(
    detection: dict[str, Any],
) -> dict[str, Any]:
    # Position in the enclosing detections list is the stable detection index;
    # do not repeat it in every row.
    result: dict[str, Any] = {}
    _copy_present(
        detection,
        result,
        ("rank", "score", "area_px", "mask_path"),
    )
    bbox = _numeric_vector(
        detection.get("bbox_xyxy", detection.get("box_xyxy")),
        length=4,
    )
    if bbox:
        result["bbox_xyxy"] = bbox
    centroid = _numeric_vector(detection.get("centroid_px"), length=2)
    if centroid:
        result["centroid_px"] = centroid

    grounding = detection.get("grounding_3d")
    if not isinstance(grounding, dict):
        grounding = {}
    compact_grounding = _compact_grounding(grounding)
    if compact_grounding:
        result["grounding_3d"] = compact_grounding
    quality = _compact_quality(detection.get("quality"))
    if quality:
        result["quality"] = quality

    candidates = _operation_candidates(detection, grounding)
    result["operation_candidate_count"] = max(
        len(candidates),
        _nonnegative_int(
            detection.get(
                "operation_pose_candidate_count",
                grounding.get("operation_pose_candidate_count"),
            ),
            default=0,
        ),
    )
    candidate_index = _compact_operation_candidate_index(candidates)
    if candidate_index:
        result["operation_candidate_index"] = candidate_index
    selected_candidate_id = detection.get(
        "selected_operation_candidate_id",
        grounding.get("selected_operation_candidate_id"),
    )
    if selected_candidate_id not in (None, ""):
        result["selected_operation_candidate_id"] = deepcopy(
            selected_candidate_id
        )
    return result


def _compact_grounding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    _copy_present(value, result, ("success", "valid_ratio"))
    for key in (
        "centroid_world",
        "bbox_world_min",
        "bbox_world_max",
        "top_surface_world",
    ):
        point = _numeric_vector(value.get(key), length=3)
        if point:
            result[key] = point
    return result


def _compact_quality(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    if "actionable" in value:
        result["actionable"] = bool(value.get("actionable"))
    warnings = _short_text_list(value.get("warnings"), limit=4, item_limit=160)
    if warnings:
        result["warnings"] = warnings
    return result


def _operation_candidates(
    detection: dict[str, Any],
    grounding: dict[str, Any],
) -> list[dict[str, Any]]:
    for owner in (grounding, detection):
        for key in ("operation_pose_candidates", "operation_candidates"):
            candidates = owner.get(key)
            if isinstance(candidates, list):
                return [item for item in candidates if isinstance(item, dict)]
    return []


def _compact_operation_candidate_index(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        return {}
    columns = [
        "candidate_id",
        "source_candidate_index",
        "arm",
        "action_mode",
        "priority",
        "score",
        "geometry_source",
    ]
    rows = [
        [deepcopy(candidate.get(column)) for column in columns]
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    return {"columns": columns, "rows": rows} if rows else {}


def _hoist_operation_candidate_index(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deduplicate table schema and categorical strings across detections."""

    indexed_candidates: list[tuple[int, int, dict[str, Any]]] = []
    for result_index, result in enumerate(results):
        detections = result.get("detections")
        if not isinstance(detections, list):
            continue
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            local_index = detection.pop("operation_candidate_index", None)
            if not isinstance(local_index, dict):
                continue
            columns = local_index.get("columns")
            rows = local_index.get("rows")
            if not isinstance(columns, list) or not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, list):
                    continue
                candidate = {
                    str(column): deepcopy(row[index])
                    for index, column in enumerate(columns)
                    if index < len(row)
                }
                indexed_candidates.append(
                    (result_index, detection_index, candidate)
                )
    if not indexed_candidates:
        return {}

    ordered_columns = [
        "candidate_id",
        "source_candidate_index",
        "arm",
        "action_mode",
        "priority",
        "score",
        "geometry_source",
    ]
    candidate_columns = [
        column
        for column in ordered_columns
        if column not in {"priority", "score"}
        or any(
            candidate.get(column) is not None
            for _, _, candidate in indexed_candidates
        )
    ]
    categorical_columns = {
        "candidate_id",
        "arm",
        "action_mode",
        "geometry_source",
    }
    dictionaries: dict[str, list[Any]] = {
        column: [] for column in candidate_columns if column in categorical_columns
    }
    for column in ("priority", "score"):
        if column not in candidate_columns:
            continue
        values = [
            candidate.get(column)
            for _, _, candidate in indexed_candidates
            if candidate.get(column) is not None
        ]
        unique_values: list[Any] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        direct_size = sum(len(json.dumps(value)) for value in values)
        indexed_size = (
            len(json.dumps(unique_values))
            + sum(
                len(str(unique_values.index(value))) for value in values
            )
            + len(column)
            + 6
        )
        if unique_values and indexed_size < direct_size:
            dictionaries[column] = unique_values
    groups: list[list[Any]] = []
    current_group: tuple[int, int] | None = None
    current_rows: list[list[Any]] = []
    for result_index, detection_index, candidate in indexed_candidates:
        group = (result_index, detection_index)
        if current_group != group:
            if current_group is not None:
                groups.append(
                    [current_group[0], current_group[1], current_rows]
                )
            current_group = group
            current_rows = []
        row: list[Any] = []
        for column in candidate_columns:
            value = candidate.get(column)
            if column in dictionaries and value is not None:
                dictionary = dictionaries[column]
                if value not in dictionary:
                    dictionary.append(value)
                value = dictionary.index(value)
            row.append(deepcopy(value))
        current_rows.append(row)
    if current_group is not None:
        groups.append([current_group[0], current_group[1], current_rows])
    return {
        "group_columns": ["result_index", "detection_index", "rows"],
        "columns": candidate_columns,
        "dictionaries": dictionaries,
        "groups": groups,
    }


def _selected_detection_index(
    item: dict[str, Any],
    detections: list[dict[str, Any]],
) -> int | None:
    explicit = item.get("selected_detection_index")
    if explicit is not None:
        index = _nonnegative_int(explicit, default=-1)
        if 0 <= index < len(detections):
            return index
    selected_path = str(item.get("mask_path", "") or "")
    if selected_path:
        for index, detection in enumerate(detections):
            if str(detection.get("mask_path", "") or "") == selected_path:
                return index
    selected_bbox = item.get("bbox_xyxy")
    if isinstance(selected_bbox, (list, tuple)):
        for index, detection in enumerate(detections):
            if detection.get("bbox_xyxy", detection.get("box_xyxy")) == selected_bbox:
                return index
    return 0 if detections and item.get("success") is True else None


def _compact_scene_memory(scene: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    _copy_present(scene, result, ("env_step",))
    result["instances"] = [
        _compact_scene_instance(item)
        for item in scene.get("instances", []) or []
        if isinstance(item, dict)
    ]
    task_focus = _compact_task_focus(scene.get("task_focus"))
    if task_focus:
        result["task_focus"] = task_focus
    operation_targets = _compact_operation_targets(
        scene.get("operation_targets")
    )
    if operation_targets:
        result["operation_targets"] = operation_targets
    reference_regions = _compact_reference_regions(
        scene.get("reference_regions")
    )
    if reference_regions:
        result["reference_regions"] = reference_regions
    manipulation = _compact_manipulation_state(
        scene.get("manipulation_state")
    )
    if manipulation:
        result["manipulation_state"] = manipulation
    uncertainty = _compact_uncertainty(scene.get("uncertainty"))
    if uncertainty:
        result["uncertainty"] = uncertainty
    summary = _short_text(scene.get("summary"), 320)
    if summary:
        result["summary"] = summary
    temporal_memory = scene.get("temporal_memory")
    if isinstance(temporal_memory, (dict, list)) and temporal_memory:
        result["temporal_memory"] = {"present": True}
    return result


def _compact_scene_instance(item: dict[str, Any]) -> dict[str, Any]:
    instance_id = str(
        item.get("instance_id", item.get("track_id", "")) or ""
    ).strip()
    result: dict[str, Any] = {"instance_id": instance_id}
    _copy_present(
        item,
        result,
        (
            "track_id",
            "class",
            "role",
            "query_role",
            "camera",
            "status",
            "stability",
            "position_state",
            "position_source",
            "last_verified_step",
            "action_geometry_state",
            "actionable",
            "missing_steps",
        ),
    )
    world = _numeric_vector(
        item.get("world_m", item.get("last_verified_world_m")), length=3
    )
    if world:
        result["world_m"] = world
    original = _numeric_vector(
        item.get("original_world_m", item.get("first_observed_world_m")),
        length=3,
    )
    if original:
        result["original_world_m"] = original
    confidence = _finite_float(
        item.get(
            "confidence",
            item.get("last_verified_score", item.get("score")),
        )
    )
    if confidence is not None:
        result["confidence"] = confidence
    score = _finite_float(item.get("score"))
    if score is not None:
        result["score"] = score
    bbox = _numeric_vector(item.get("bbox_xyxy"), length=4)
    if bbox:
        result["bbox_xyxy"] = bbox
    cameras = sorted(
        {
            str(camera or "").strip()
            for camera in item.get("supporting_cameras", []) or []
            if str(camera or "").strip()
        }
    )
    if cameras:
        result["supporting_cameras"] = cameras
    warnings = _short_text_list(
        item.get("quality_warnings"), limit=4, item_limit=160
    )
    if warnings:
        result["quality_warnings"] = warnings
    roles = _compact_verified_roles(item.get("verified_roles"))
    if roles:
        result["verified_roles"] = roles

    operations, actual_candidate_count = _compact_instance_operations(item)
    if operations:
        result["operations"] = operations
    declared_candidate_count = _nonnegative_int(
        item.get("operation_pose_candidate_count"), default=0
    )
    result["operation_candidate_count"] = max(
        declared_candidate_count, actual_candidate_count
    )
    return result


def _compact_instance_operations(
    item: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    raw_candidates = item.get("operation_pose_candidates")
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    by_mode: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        mode = str(candidate.get("action_mode", "") or "").strip().lower()
        if not mode:
            continue
        operation = by_mode.setdefault(
            mode,
            {"arms": set(), "candidate_ids": [], "candidate_count": 0},
        )
        operation["candidate_count"] += 1
        arm = str(candidate.get("arm", "") or "").strip().lower()
        if arm:
            operation["arms"].add(arm)
        candidate_id = str(candidate.get("candidate_id", "") or "").strip()
        if candidate_id and candidate_id not in operation["candidate_ids"]:
            operation["candidate_ids"].append(candidate_id)

    for mode, point_key in (
        ("grasp", "grasp_world_m"),
        ("contact", "contact_world_m"),
        ("place", "place_world_m"),
    ):
        if _numeric_vector(item.get(point_key), length=3):
            by_mode.setdefault(
                mode,
                {"arms": set(), "candidate_ids": [], "candidate_count": 0},
            )

    result: dict[str, Any] = {}
    for mode in sorted(by_mode):
        value = by_mode[mode]
        result[mode] = {
            "available": True,
            "arms": sorted(value["arms"]),
            "candidate_ids": list(value["candidate_ids"]),
            "candidate_count": int(value["candidate_count"]),
        }
    return result, sum(
        int(value["candidate_count"]) for value in by_mode.values()
    )


def _compact_task_focus(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    _copy_present(
        value,
        result,
        (
            "current_subtask",
            "target_instances",
            "tool_instances",
            "context_instances",
            "identity_binding_required",
            "identity_binding_roles",
            "identity_binding_errors",
            "source",
        ),
    )
    reason = _short_text(value.get("reason_summary"), 240)
    if reason:
        result["reason_summary"] = reason
    return result


def _compact_operation_targets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = (
        "target_id",
        "action_mode",
        "held_instance_id",
        "target_kind",
        "object_target_world_m",
        "support_instance_id",
        "vacated_by_instance_id",
        "placement_relation",
        "reference_region_id",
        "reference_object_id",
        "reference_instance_ids",
        "expected_count",
        "observed_member_count",
        "reference_evidence_status",
        "support_valid",
        "free",
        "occupied_by",
        "arm_options",
        "candidate_id",
        "candidate_ids",
        "candidate_count",
        "priority",
    )
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        projected: dict[str, Any] = {}
        _copy_present(item, projected, allowed)
        point = _numeric_vector(
            projected.get("object_target_world_m"), length=3
        )
        if point:
            projected["object_target_world_m"] = point
        elif "object_target_world_m" in projected:
            projected.pop("object_target_world_m", None)
        result.append(projected)
    return result


def _compact_reference_regions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = (
        "reference_region_id",
        "reference_object_id",
        "placement_relation",
        "center_world_xy",
        "position_state",
        "support_valid",
        "evidence_status",
        "expected_count",
        "observed_member_count",
        "reference_instance_ids",
    )
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        projected: dict[str, Any] = {}
        _copy_present(item, projected, allowed)
        center = _numeric_vector(projected.get("center_world_xy"), length=2)
        if center:
            projected["center_world_xy"] = center
        elif "center_world_xy" in projected:
            projected.pop("center_world_xy", None)
        result.append(projected)
    return result


def _compact_manipulation_state(value: Any) -> dict[str, Any]:
    public = public_manipulation_state(value)
    allowed = (
        "phase",
        "held_instance_id",
        "released_instance_id",
        "holding_confirmed",
        "transport_authorized",
        "operation_action_mode",
        "attachment_evidence_status",
        "grasp_transport_policy",
        "place_target_id",
        "held_object_target_world_m",
        "object_at_target",
        "object_target_error_m",
        "target_tolerance_m",
        "detachment_verified",
        "stable_across_fresh_observations",
        "placement_recovery_required",
        "pre_release_validated",
        "placement_failure_reason",
        "updated_step",
    )
    result: dict[str, Any] = {}
    for arm, arm_state in public.items():
        if not isinstance(arm_state, dict):
            continue
        projected: dict[str, Any] = {}
        _copy_present(arm_state, projected, allowed)
        target = _numeric_vector(
            projected.get("held_object_target_world_m"), length=3
        )
        if target:
            projected["held_object_target_world_m"] = target
        elif "held_object_target_world_m" in projected:
            projected.pop("held_object_target_world_m", None)
        evidence = _short_text(arm_state.get("evidence"), 240)
        if evidence:
            projected["evidence"] = evidence
        if projected:
            result[str(arm)] = projected
    return result


def _compact_uncertainty(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    result: list[Any] = []
    allowed = (
        "instance_id",
        "track_id",
        "object_id",
        "type",
        "status",
        "reason",
        "camera",
        "score",
        "actionable",
    )
    for item in value[:8]:
        if isinstance(item, dict):
            projected: dict[str, Any] = {}
            _copy_present(item, projected, allowed)
            if "reason" in projected:
                projected["reason"] = _short_text(projected["reason"], 220)
            if projected:
                result.append(projected)
            continue
        text = _short_text(item, 220)
        if text and text not in result:
            result.append(text)
    return result


def _compact_verified_roles(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value[:4]:
        if isinstance(item, str):
            role = item.strip()
            if role:
                result.append({"role": role})
            continue
        if not isinstance(item, dict):
            continue
        projected: dict[str, Any] = {}
        _copy_present(item, projected, ("role", "effect_type", "env_step"))
        if projected:
            result.append(projected)
    return result


def _compact_perception_query(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    _copy_present(
        value,
        result,
        (
            "object_id",
            "text_prompt",
            "role",
            "instance_hint",
            "instance_ref",
            "identity_binding_required",
            "entity_scope",
            "placement_relation",
            "expected_count",
        ),
    )
    reason = _short_text(value.get("reason"), 200)
    if reason:
        result["reason"] = reason
    return result


def _compact_binding_requirement(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    _copy_present(
        value,
        result,
        (
            "required_any_roles",
            "require_instance_binding",
            "force_refresh",
            "failure_reason",
            "contract",
        ),
    )
    return result


def _compact_robot_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    _copy_present(
        value,
        result,
        ("step", "step_limit", "joint_dim", "joint_norm"),
    )
    for arm in ("left", "right"):
        arm_state = value.get(arm)
        if not isinstance(arm_state, dict):
            continue
        projected: dict[str, Any] = {}
        for key, length in (("xyz", 3), ("quat_wxyz", 4), ("rpy", 3)):
            vector = _numeric_vector(arm_state.get(key), length=length)
            if vector:
                projected[key] = vector
        _copy_present(arm_state, projected, ("gripper", "gripper_command"))
        if projected:
            result[arm] = projected
    return result


def _find_robot_state(
    source: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    compact = _compact_robot_state(source.get("robot_state"))
    if compact:
        return compact
    for segment in segments:
        compact = _compact_robot_state(segment.get("robot_state"))
        if compact:
            return compact
        for detection in segment.get("detections", []) or []:
            if not isinstance(detection, dict):
                continue
            compact = _compact_robot_state(detection.get("robot_state"))
            if compact:
                return compact
    return {}


def _segmentation_error_summaries(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in segments:
        error = _short_text(item.get("error"), 200)
        if not error:
            continue
        summary = {
            "object_id": item.get("object_id"),
            "camera": item.get("camera"),
            "error": error,
        }
        if summary not in result:
            result.append(summary)
    return result[:8]


def _without_repeated_candidate_geometry(
    value: Any,
    *,
    candidate_context: bool = False,
) -> Any:
    if isinstance(value, dict):
        local_candidate_context = candidate_context or any(
            key in value
            for key in (
                "candidate_id",
                "operation_candidate_id",
                "selected_operation_candidate_id",
            )
        )
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in {
                "operation_tcp_pose",
                "operation_object_contact_pose",
            }:
                continue
            nested_candidate_context = local_candidate_context or (
                "candidate" in normalized_key
                and normalized_key
                not in {
                    "operation_candidate_id",
                    "selected_operation_candidate_id",
                    "source_candidate_index",
                }
            )
            if nested_candidate_context and normalized_key in _POSE_KEYS:
                continue
            result[str(key)] = _without_repeated_candidate_geometry(
                item,
                candidate_context=nested_candidate_context,
            )
        return result
    if isinstance(value, list):
        return [
            _without_repeated_candidate_geometry(
                item,
                candidate_context=candidate_context,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _without_repeated_candidate_geometry(
                item,
                candidate_context=candidate_context,
            )
            for item in value
        ]
    return deepcopy(value)


def _copy_present(
    source: dict[str, Any],
    target: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key in source and source.get(key) not in (None, ""):
            target[key] = deepcopy(source.get(key))


def _numeric_vector(value: Any, *, length: int) -> list[float | int]:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return []
    result: list[float | int] = []
    for item in value[:length]:
        if isinstance(item, bool):
            return []
        number = _finite_float(item)
        if number is None:
            return []
        result.append(item if isinstance(item, int) else number)
    return result


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value: Any, *, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result >= 0 else default


def _short_text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[: max(0, int(limit))]


def _short_text_list(
    value: Any,
    *,
    limit: int,
    item_limit: int,
) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in value:
        text = _short_text(item, item_limit)
        if text and text not in result:
            result.append(text)
        if len(result) >= max(0, int(limit)):
            break
    return result

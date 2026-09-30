from __future__ import annotations

import math
from typing import Any


LEGACY_PLANNER_CONTEXT_MODE = "legacy"
COMPACT_PLANNER_CONTEXT_MODE = "compact_v1"
VALID_PLANNER_CONTEXT_MODES = {
    LEGACY_PLANNER_CONTEXT_MODE,
    COMPACT_PLANNER_CONTEXT_MODE,
}


def normalize_planner_context_mode(
    value: Any,
    *,
    field_name: str = "planner_context_mode",
    default: str = LEGACY_PLANNER_CONTEXT_MODE,
) -> str:
    normalized = str(value or default).strip().lower().replace("-", "_")
    if normalized not in VALID_PLANNER_CONTEXT_MODES:
        choices = ", ".join(sorted(VALID_PLANNER_CONTEXT_MODES))
        raise ValueError(
            f"{field_name} must be one of {{{choices}}}; got {value!r}"
        )
    return normalized


def build_planner_scene_view(
    scene_memory: Any,
    *,
    manipulation_state: Any = None,
    observation_preprocess: Any = None,
    max_instances: int = 12,
    max_uncertainties: int = 4,
) -> dict[str, Any]:
    """Project private runtime memory into one planner-visible scene view.

    Runtime Scene Memory keeps masks, per-camera grounding, executable SE(3)
    candidates, attachment transforms, and temporal repair history.  A planner
    selects public instances/actions/targets; it does not need those private
    executor details.  This function therefore uses explicit allowlists and
    only aggregates candidate availability by action mode and arm.

    The returned object intentionally retains the familiar Scene Memory public
    field names.  During migration it can occupy the existing ``scene_memory``
    payload slot without duplicating a second compatibility representation.
    """

    scene = scene_memory if isinstance(scene_memory, dict) else {}
    preprocess = (
        observation_preprocess
        if isinstance(observation_preprocess, dict)
        else {}
    )
    focus = _compact_task_focus(scene.get("task_focus"))
    effective_manipulation = (
        manipulation_state
        if isinstance(manipulation_state, dict)
        else scene.get("manipulation_state")
    )
    instances = [
        item
        for item in scene.get("instances", []) or []
        if isinstance(item, dict)
    ]
    selected_instances = _select_instances(
        instances,
        scene=scene,
        focus=focus,
        manipulation_state=effective_manipulation,
        max_instances=max_instances,
    )
    operation_targets = _compact_operation_targets(
        scene.get("operation_targets")
    )
    reference_regions = _compact_reference_regions(
        scene.get("reference_regions")
    )
    spatial_state = _compact_spatial_state(scene.get("spatial_state"))
    uncertainty = _compact_uncertainty(
        scene.get("uncertainty"),
        limit=max_uncertainties,
    )
    perception_status = _compact_perception_status(preprocess)

    view: dict[str, Any] = {
        "schema": "planner_scene_view/compact_v1",
        "env_step": scene.get("env_step", preprocess.get("env_step")),
        "task_focus": focus,
        "instances": [
            _compact_instance(item) for item in selected_instances
        ],
        "operation_targets": operation_targets,
        "reference_regions": reference_regions,
        "spatial_state": spatial_state,
        "manipulation_state": _compact_manipulation_state(
            effective_manipulation
        ),
        "uncertainty": uncertainty,
    }
    if perception_status:
        view["perception_status"] = perception_status
    return view


def compact_finalized_observation_event(
    preprocess_payload: Any,
    *,
    planner_scene_view: dict[str, Any],
) -> dict[str, Any]:
    """Return a bounded trace record while runtime keeps the complete payload."""

    payload = preprocess_payload if isinstance(preprocess_payload, dict) else {}
    segmentation = [
        item
        for item in payload.get("segmentation", []) or []
        if isinstance(item, dict)
    ]
    successful = sum(item.get("success") is True for item in segmentation)
    detections = sum(
        len(item.get("detections", []) or [])
        if isinstance(item.get("detections"), list)
        else int(bool(item.get("success")))
        for item in segmentation
    )
    result: dict[str, Any] = {
        "schema": "trace/observation_preprocess_finalized/compact_v1",
        "schema_version": 1,
        "stage": payload.get("stage", "observation_preprocess"),
        "planner_context_mode": COMPACT_PLANNER_CONTEXT_MODE,
        "observation_generation": payload.get("observation_generation"),
        "observation_capture_id": payload.get("observation_capture_id"),
        "latency_sec": payload.get("latency_sec"),
        "perception_query_count": len(
            payload.get("perception_queries", []) or []
        ),
        "segmentation_result_count": len(segmentation),
        "successful_segmentation_count": successful,
        "detection_count": detections,
        "planner_scene_view": dict(planner_scene_view or {}),
        "full_runtime_state_retained_in_working_memory": True,
    }
    binding_requirement = payload.get("perception_binding_requirement")
    if isinstance(binding_requirement, dict) and binding_requirement:
        result["perception_binding_requirement"] = {
            key: binding_requirement.get(key)
            for key in (
                "required_any_roles",
                "require_instance_binding",
                "force_refresh",
                "failure_reason",
            )
            if key in binding_requirement
        }
    if "agent_identity_binding_required" in payload:
        result["agent_identity_binding_required"] = bool(
            payload.get("agent_identity_binding_required")
        )
    return result


def _compact_task_focus(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        key: value.get(key)
        for key in (
            "current_subtask",
            "target_instances",
            "tool_instances",
            "context_instances",
            "identity_binding_required",
            "identity_binding_roles",
            "identity_binding_errors",
        )
        if key in value
    }
    reason = _short_text(value.get("reason_summary"), limit=240)
    if reason:
        result["reason_summary"] = reason
    return result


def _select_instances(
    instances: list[dict[str, Any]],
    *,
    scene: dict[str, Any],
    focus: dict[str, Any],
    manipulation_state: Any,
    max_instances: int,
) -> list[dict[str, Any]]:
    limit = max(1, int(max_instances))
    priority_refs: list[str] = []

    def add_ref(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in priority_refs:
            priority_refs.append(text)

    for key in ("target_instances", "tool_instances"):
        for item in focus.get(key, []) or []:
            add_ref(item)
    arm_states = (
        manipulation_state.values()
        if isinstance(manipulation_state, dict)
        else ()
    )
    for arm_state in arm_states:
        if not isinstance(arm_state, dict):
            continue
        add_ref(arm_state.get("held_instance_id"))
        add_ref(arm_state.get("released_instance_id"))
    for target in scene.get("operation_targets", []) or []:
        if not isinstance(target, dict):
            continue
        add_ref(target.get("held_instance_id"))
    for item in focus.get("context_instances", []) or []:
        add_ref(item)
    for target in scene.get("operation_targets", []) or []:
        if not isinstance(target, dict):
            continue
        for key in ("support_instance_id", "vacated_by_instance_id"):
            add_ref(target.get(key))
        for item in target.get("reference_instance_ids", []) or []:
            add_ref(item)
    for region in scene.get("reference_regions", []) or []:
        if not isinstance(region, dict):
            continue
        for item in region.get("reference_instance_ids", []) or []:
            add_ref(item)

    def identities(item: dict[str, Any]) -> set[str]:
        return {
            str(item.get(key, "") or "").strip()
            for key in ("instance_id", "track_id", "oracle_id")
            if str(item.get(key, "") or "").strip()
        }

    by_ref: dict[str, dict[str, Any]] = {}
    for item in instances:
        for ref in identities(item):
            by_ref.setdefault(ref, item)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for ref in priority_refs:
        item = by_ref.get(ref)
        if item is None or id(item) in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(id(item))

    remaining = [item for item in instances if id(item) not in selected_ids]
    remaining.sort(
        key=lambda item: (
            0 if str(item.get("status", "")).lower() == "visible" else 1,
            0
            if str(item.get("position_state", "")).lower()
            == "current_verified"
            else 1,
            -_finite_float(item.get("score"), default=0.0),
            str(item.get("instance_id", item.get("track_id", ""))),
        )
    )
    selected.extend(remaining)
    return selected[:limit]


def _compact_instance(item: dict[str, Any]) -> dict[str, Any]:
    instance_id = str(
        item.get("instance_id", item.get("track_id", "")) or ""
    ).strip()
    result: dict[str, Any] = {
        "instance_id": instance_id,
    }
    for key in (
        "class",
        "role",
        "query_role",
        "status",
        "stability",
        "position_state",
        "position_source",
        "last_verified_step",
        "action_geometry_state",
        "actionable",
        "missing_steps",
    ):
        if key in item and item.get(key) not in (None, "", []):
            result[key] = item.get(key)

    world = _xyz(item.get("world_m"))
    if world:
        result["world_m"] = world
    original = _xyz(
        item.get("first_observed_world_m", item.get("original_world_m"))
    )
    if original:
        result["original_world_m"] = original
    confidence = _finite_float(
        item.get("last_verified_score", item.get("score")),
        default=None,
    )
    if confidence is not None:
        result["confidence"] = round(confidence, 4)

    roles: list[dict[str, Any]] = []
    for role in item.get("verified_roles", []) or []:
        if isinstance(role, str):
            role_name = role.strip()
            if role_name:
                roles.append({"role": role_name})
            continue
        if not isinstance(role, dict):
            continue
        projected = {
            key: role.get(key)
            for key in ("role", "effect_type", "env_step")
            if role.get(key) not in (None, "")
        }
        if projected:
            roles.append(projected)
    if roles:
        result["verified_roles"] = roles[:4]

    cameras = sorted(
        {
            str(camera or "").strip()
            for camera in item.get("supporting_cameras", []) or []
            if str(camera or "").strip()
        }
    )
    if cameras:
        result["supporting_cameras"] = cameras

    warnings = [
        _short_text(value, limit=160)
        for value in item.get("quality_warnings", []) or []
        if _short_text(value, limit=160)
    ]
    if warnings:
        result["quality_warnings"] = warnings[:3]

    operations = _compact_instance_operations(item)
    if operations:
        result["operations"] = operations
    return result


def _compact_instance_operations(item: dict[str, Any]) -> dict[str, Any]:
    by_mode: dict[str, set[str]] = {}
    for candidate in item.get("operation_pose_candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        mode = str(candidate.get("action_mode", "") or "").strip().lower()
        if mode not in {"grasp", "contact"}:
            continue
        arm = str(candidate.get("arm", "") or "").strip().lower()
        if arm in {"left", "right"}:
            by_mode.setdefault(mode, set()).add(arm)
        else:
            by_mode.setdefault(mode, set())

    if _xyz(item.get("grasp_world_m")):
        by_mode.setdefault("grasp", set())
    if _xyz(item.get("contact_world_m")):
        by_mode.setdefault("contact", set())

    operations: dict[str, Any] = {}
    for mode in ("grasp", "contact"):
        if mode not in by_mode:
            continue
        point_keys = ["approach_world_m"]
        point_keys.append(
            "grasp_world_m" if mode == "grasp" else "contact_world_m"
        )
        operations[mode] = {
            "available": True,
            "arms": sorted(by_mode[mode]),
            "point_keys": point_keys,
        }
    return operations


def _compact_operation_targets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    keys = (
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
        "holding_status",
        "grasp_transport_policy",
        "priority",
    )
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        projected = {
            key: item.get(key)
            for key in keys
            if key in item and item.get(key) not in (None, "")
        }
        if "object_target_world_m" in projected:
            projected["object_target_world_m"] = _xyz(
                projected["object_target_world_m"]
            )
        result.append(projected)
    return result[:8]


def _compact_reference_regions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    keys = (
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
    return [
        {
            key: item.get(key)
            for key in keys
            if key in item and item.get(key) not in (None, "")
        }
        for item in value[:4]
        if isinstance(item, dict)
    ]


def _compact_spatial_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in (
            "signature",
            "order_by_x",
            "order_by_y",
            "horizontal_overlap_pairs",
            "support_pairs",
        )
        if key in value and value.get(key) not in (None, "")
    }


def _compact_manipulation_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
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
        "evidence",
    )
    result: dict[str, Any] = {}
    for arm in ("left", "right"):
        arm_state = value.get(arm)
        if not isinstance(arm_state, dict):
            continue
        projected = {
            key: arm_state.get(key)
            for key in allowed
            if key in arm_state and arm_state.get(key) not in (None, "")
        }
        if "held_object_target_world_m" in projected:
            projected["held_object_target_world_m"] = _xyz(
                projected["held_object_target_world_m"]
            )
        if "evidence" in projected:
            projected["evidence"] = _short_text(
                projected["evidence"], limit=240
            )
        result[arm] = projected
    return result


def _compact_uncertainty(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _short_text(item, limit=220)
        if text and text not in result:
            result.append(text)
        if len(result) >= max(0, int(limit)):
            break
    return result


def _compact_perception_status(payload: dict[str, Any]) -> dict[str, Any]:
    segments = [
        item
        for item in payload.get("segmentation", []) or []
        if isinstance(item, dict)
    ]
    if not segments:
        return {}
    cameras = sorted(
        {
            str(item.get("camera", "") or "").strip()
            for item in segments
            if str(item.get("camera", "") or "").strip()
        }
    )
    failed = []
    ambiguous = []
    for item in segments:
        object_id = str(item.get("object_id", "") or "").strip()
        if item.get("success") is not True:
            failed.append(
                {
                    "object_id": object_id,
                    "camera": item.get("camera"),
                    "reason": _short_text(
                        item.get("error", "not detected"), limit=140
                    ),
                }
            )
        detections = item.get("detections")
        if isinstance(detections, list) and len(detections) > 1:
            ambiguous.append(
                {
                    "object_id": object_id,
                    "camera": item.get("camera"),
                    "candidate_count": len(detections),
                }
            )
    result: dict[str, Any] = {
        "observation_generation": payload.get("observation_generation"),
        "observation_capture_id": payload.get("observation_capture_id"),
        "cameras": cameras,
        "query_result_count": len(segments),
        "successful_query_count": sum(
            item.get("success") is True for item in segments
        ),
    }
    if failed:
        result["failed_queries"] = failed[:4]
    if ambiguous:
        result["ambiguous_queries"] = ambiguous[:4]
    return result


def _xyz(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return []
    result: list[float] = []
    for item in value[:3]:
        number = _finite_float(item, default=None)
        if number is None:
            return []
        result.append(round(number, 6))
    return result


def _finite_float(value: Any, *, default: float | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _short_text(value: Any, *, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[: max(0, int(limit))]

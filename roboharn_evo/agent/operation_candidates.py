from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from roboharn_evo.agent.relational_place_targets import (
    RELATIONAL_PLACE_TARGET_KIND,
    build_relational_place_target_specs,
    revalidate_relational_place_target,
)

OPERATION_POSE_CANDIDATES_KEY = "operation_pose_candidates"
OPERATION_TARGETS_KEY = "operation_targets"
SPATIAL_STATE_KEY = "spatial_state"
SUPPORTED_OPERATION_ACTION_MODES = {"contact", "grasp", "place"}

_PLACE_CLEARANCE_MARGIN_M = 0.025
_PLACE_HORIZONTAL_MARGIN_M = 0.012
_PLACE_POSITION_DEDUP_M = 0.015
_PLACE_TARGET_REVALIDATION_TOLERANCE_M = 0.02
_PLACE_DEFAULT_TARGET_TOLERANCE_M = 0.025
_PLACE_MIN_TARGET_TOLERANCE_M = 0.02
_PLACE_MAX_TARGET_TOLERANCE_M = 0.05
_MAX_HELD_OBJECT_TO_TCP_OFFSET_M = 0.35


def normalize_grounded_point_key(value: Any) -> str:
    raw = str(value or "approach_world_m").strip().lower()
    aliases = {
        "approach": "approach_world_m",
        "approach_point_world": "approach_world_m",
        "approach_world": "approach_world_m",
        "grasp": "grasp_world_m",
        "grasp_world": "grasp_world_m",
        "grasp_point_world": "grasp_world_m",
        "contact": "contact_world_m",
        "contact_world": "contact_world_m",
        "contact_point_world": "contact_world_m",
        "place": "place_world_m",
        "place_world": "place_world_m",
        "place_point_world": "place_world_m",
        "top_surface": "top_surface_world_m",
        "top_surface_world": "top_surface_world_m",
        "centroid": "world_m",
        "centroid_world": "world_m",
        "world": "world_m",
        "world_m": "world_m",
    }
    return aliases.get(raw, raw)


def operation_action_mode(point_key: Any, explicit_mode: Any = None) -> str:
    explicit = str(explicit_mode or "").strip().lower()
    if explicit in SUPPORTED_OPERATION_ACTION_MODES:
        return explicit
    normalized = normalize_grounded_point_key(point_key)
    if normalized == "place_world_m":
        return "place"
    if normalized == "contact_world_m":
        return "contact"
    # An approach without an explicit mode keeps the manipulation default.
    # Contact sequences carry an internal explicit mode from their contact setup.
    return "grasp"


def uses_operation_pose_candidate(point_key: Any) -> bool:
    return normalize_grounded_point_key(point_key) in {
        "approach_world_m",
        "grasp_world_m",
        "contact_world_m",
        "place_world_m",
    }


def manipulation_state_allows_transport(
    value: Any,
    *,
    active_grasp_transport_policy: Any = None,
) -> bool:
    """Return whether runtime state may use bounded carry/place geometry.

    Strict states require verifier-confirmed attachment.  Evidence-only states
    may instead use exact close-time geometry, but remain explicitly marked as
    provisional rather than being presented as confirmed grasps.
    """

    if not isinstance(value, dict):
        return False
    if not str(value.get("held_instance_id", "") or "").strip():
        return False
    if value.get("transport_authorized") is not True:
        return False
    if value.get("execution_evidence_enabled") is False:
        return _valid_attachment_geometry(value.get("held_object_to_tcp_attachment"))
    if value.get("holding_confirmed") is True:
        return valid_held_object_to_tcp_attachment(
            value.get("held_object_to_tcp_attachment"),
            grasp_candidate_id=value.get("grasp_candidate_id"),
            grasp_attempt_nonce=value.get("grasp_attempt_nonce"),
            grasp_attempt_step=value.get("grasp_attempt_step"),
        )
    if active_grasp_transport_policy is not None and (
        str(active_grasp_transport_policy or "")
        .strip()
        .lower()
        .replace("-", "_")
        != "evidence_only"
    ):
        return False
    return valid_evidence_only_transport_state(value)


def valid_evidence_only_transport_state(value: Any) -> bool:
    """Validate one runtime-authored provisional transport capability.

    ``attachment_evidence_status`` is deliberately descriptive in this mode;
    even contradictory visual evidence cannot silently regain Guard authority.
    The planner may still react to that evidence explicitly.
    """

    if not isinstance(value, dict):
        return False
    if (
        str(value.get("phase", "") or "").strip().lower()
        not in {"holding_provisional", "place_aligned"}
        or value.get("holding_confirmed") is True
        or value.get("transport_authorized") is not True
        or str(value.get("grasp_transport_policy", "") or "")
        .strip()
        .lower()
        != "evidence_only"
        or str(value.get("attachment_evidence_status", "") or "")
        .strip()
        .lower()
        not in {"unknown", "supports", "contradicts"}
        or not str(value.get("held_instance_id", "") or "").strip()
    ):
        return False
    attachment = value.get("held_object_to_tcp_attachment")
    if not isinstance(attachment, dict) or not _valid_attachment_geometry(
        attachment
    ):
        return False
    candidate_id = str(value.get("grasp_candidate_id", "") or "").strip()
    attempt_nonce = str(value.get("grasp_attempt_nonce", "") or "").strip()
    attempt_step = _integer(value.get("grasp_attempt_step"))
    return bool(
        candidate_id
        and attempt_nonce
        and attempt_step is not None
        and str(attachment.get("grasp_candidate_id", "") or "").strip()
        == candidate_id
        and str(attachment.get("grasp_attempt_nonce", "") or "").strip()
        == attempt_nonce
        and _integer(attachment.get("capture_step")) == attempt_step
        and str(attachment.get("source", "") or "").strip()
        == "runtime_close_time_grasp_attachment_assumption"
        and str(attachment.get("authority", "") or "").strip()
        == "runtime_evidence_only_grasp_transport_policy"
    )


def valid_held_object_to_tcp_attachment(
    value: Any,
    *,
    grasp_candidate_id: Any,
    grasp_attempt_nonce: Any,
    grasp_attempt_step: Any,
) -> bool:
    """Return whether runtime-authoritative attachment may drive transport."""

    candidate_id = str(grasp_candidate_id or "").strip()
    attempt_nonce = str(grasp_attempt_nonce or "").strip()
    attempt_step = _integer(grasp_attempt_step)
    evidence_authority = (
        str(value.get("source", "") or "").strip(),
        str(value.get("authority", "") or "").strip(),
    ) if isinstance(value, dict) else ("", "")
    return bool(
        _valid_attachment_geometry(value)
        and evidence_authority
        in {
            (
                "runtime_multiview_grasp_motion_verified",
                "runtime_multiview_grasp_motion",
            ),
            (
                "runtime_single_fixed_camera_grasp_motion_verified",
                "runtime_single_fixed_camera_grasp_motion",
            ),
        }
        and candidate_id
        and attempt_nonce
        and attempt_step is not None
        and str(value.get("grasp_candidate_id", "") or "").strip()
        == candidate_id
        and str(value.get("grasp_attempt_nonce", "") or "").strip()
        == attempt_nonce
        and _integer(value.get("capture_step")) == attempt_step
    )


def valid_pending_held_object_to_tcp_attachment(
    value: Any,
    *,
    grasp_candidate_id: Any,
    grasp_attempt_nonce: Any,
    grasp_attempt_step: Any,
) -> bool:
    """Validate the exact close-time attachment before authority promotion."""

    if not _valid_attachment_geometry(value):
        return False
    candidate_id = str(grasp_candidate_id or "").strip()
    attempt_nonce = str(grasp_attempt_nonce or "").strip()
    attempt_step = _integer(grasp_attempt_step)
    return bool(
        candidate_id
        and attempt_nonce
        and attempt_step is not None
        and str(value.get("grasp_candidate_id", "") or "").strip()
        == candidate_id
        and str(value.get("grasp_attempt_nonce", "") or "").strip()
        == attempt_nonce
        and _integer(value.get("capture_step")) == attempt_step
        and str(value.get("source", "") or "").strip()
        == "pending_lift_verification"
    )


def _valid_attachment_geometry(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if (
        str(value.get("object_proxy_frame", "") or "").strip()
        != "tcp_aligned_at_capture"
    ):
        return False
    offset = _xyz(
        value.get("object_centroid_to_tcp_translation_tcp_m")
    )
    return bool(
        offset is not None
        and math.sqrt(sum(item * item for item in offset))
        <= _MAX_HELD_OBJECT_TO_TCP_OFFSET_M
        and pose7_to_matrix(value.get("capture_tcp_pose")) is not None
    )


def place_target_tolerance_m(held_extent_m: Any) -> float:
    """Return the placement tolerance already used by runtime verification.

    Keeping this in the operation-candidate contract prevents release guards,
    identity leases, and post-release verification from inventing independent
    task-specific thresholds.
    """

    extent = _xyz(held_extent_m)
    if extent is None:
        return _PLACE_DEFAULT_TARGET_TOLERANCE_M
    return min(
        _PLACE_MAX_TARGET_TOLERANCE_M,
        max(
            _PLACE_MIN_TARGET_TOLERANCE_M,
            min(abs(float(extent[0])), abs(float(extent[1]))) / 2.0,
        ),
    )


def pose7_to_matrix(value: Any) -> np.ndarray | None:
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if pose.size != 7 or not np.all(np.isfinite(pose)):
        return None
    quat = pose[3:7]
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    w, x, y, z = quat / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose7(value: Any) -> list[float] | None:
    try:
        transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None
    if not np.all(np.isfinite(transform)):
        return None
    rotation = _orthonormal_rotation(transform[:3, :3])
    quat = matrix_to_quat_wxyz(rotation)
    return _round_list(np.concatenate([transform[:3, 3], quat]))


def matrix_to_quat_wxyz(value: Any) -> np.ndarray:
    rotation = _orthonormal_rotation(value)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(max(0.0, trace + 1.0)) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            quat = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            quat = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            quat = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat = quat / norm
    return -quat if quat[0] < 0.0 else quat


def action_to_tcp_transform(calibration: Any, *, fallback_offset_m: float = 0.0) -> np.ndarray:
    if isinstance(calibration, dict):
        raw_matrix = calibration.get("action_to_tcp_matrix")
        try:
            transform = np.asarray(raw_matrix, dtype=np.float64).reshape(4, 4)
        except Exception:
            transform = None
        if transform is not None and np.all(np.isfinite(transform)):
            result = np.asarray(transform, dtype=np.float64).copy()
            result[:3, :3] = _orthonormal_rotation(result[:3, :3])
            result[3, :] = [0.0, 0.0, 0.0, 1.0]
            return result
        translation = calibration.get("translation_m")
        rotation_quat = calibration.get("quat_wxyz")
        if translation is not None and rotation_quat is not None:
            try:
                pose_values = [*list(translation), *list(rotation_quat)]
            except (TypeError, ValueError):
                pose_values = None
            pose = pose7_to_matrix(pose_values)
            if pose is not None:
                return pose
    transform = np.eye(4, dtype=np.float64)
    try:
        offset = float(fallback_offset_m)
    except (TypeError, ValueError):
        offset = 0.0
    transform[0, 3] = max(0.0, offset) if math.isfinite(offset) else 0.0
    return transform


def operation_pose_candidates(value: Any) -> list[dict[str, Any]]:
    raw = value.get(OPERATION_POSE_CANDIDATES_KEY) if isinstance(value, dict) else None
    if not isinstance(raw, list):
        return []
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "") or "").strip()
        action_mode = str(item.get("action_mode", "") or "").strip().lower()
        target_id = str(item.get("target_id", "") or "").strip()
        arm = str(item.get("arm", "") or "").strip().lower()
        if (
            not candidate_id
            or candidate_id in seen_ids
            or action_mode not in SUPPORTED_OPERATION_ACTION_MODES
            or (action_mode == "place" and not target_id)
            or arm not in {"left", "right"}
            or pose7_to_matrix(item.get("ee_target_pose")) is None
            or pose7_to_matrix(item.get("approach_pose")) is None
        ):
            continue
        seen_ids.add(candidate_id)
        candidates.append(dict(item))
    return candidates


def select_operation_pose_candidate(
    instance: dict[str, Any],
    *,
    arm: str,
    action_mode: str,
    blocked_candidate_ids: Any = None,
    eligible_candidate_ids: Any = None,
    requested_candidate_id: Any = None,
    requested_target_id: Any = None,
    geometric_strategy: Any = None,
    geometry_policy: Any = None,
    ranking_audit: Any = None,
    geometry_scene_state: Any = None,
    v3_action_match: Any = None,
    v3_usage_audit: Any = None,
) -> dict[str, Any] | None:
    normalized_arm = str(arm or "").strip().lower()
    normalized_mode = str(action_mode or "").strip().lower()
    blocked = {
        str(item)
        for item in (blocked_candidate_ids if isinstance(blocked_candidate_ids, (list, tuple, set)) else [])
        if str(item)
    }
    allowed = None
    if eligible_candidate_ids is not None:
        if not isinstance(eligible_candidate_ids, (list, tuple, set)):
            raise TypeError("eligible_candidate_ids must be a sequence or set")
        allowed = {
            str(item).strip()
            for item in eligible_candidate_ids
            if str(item).strip()
        }
    requested = str(requested_candidate_id or "").strip()
    requested_target = str(requested_target_id or "").strip()
    eligible = [
        item
        for item in operation_pose_candidates(instance)
        if str(item.get("arm", "")).strip().lower() == normalized_arm
        and str(item.get("action_mode", "")).strip().lower() == normalized_mode
        and str(item.get("candidate_id", "")) not in blocked
        and (
            allowed is None
            or str(item.get("candidate_id", "")) in allowed
        )
        and (
            not requested_target
            or str(item.get("target_id", "") or "").strip() == requested_target
        )
    ]
    if v3_action_match is not None:
        if geometric_strategy is not None:
            raise ValueError(
                "v3 action knowledge and a direct geometric strategy are mutually exclusive"
            )
        from collections.abc import MutableMapping

        from roboharn_evo.agent.hpk.candidate_features import candidate_semantic_features
        from roboharn_evo.agent.hpk.hierarchical_retriever import ActionKnowledgeMatch
        from roboharn_evo.agent.hpk.schemas import HPKUnresolved

        if not isinstance(v3_action_match, ActionKnowledgeMatch):
            raise TypeError("v3_action_match must be ActionKnowledgeMatch")
        if not isinstance(v3_usage_audit, MutableMapping):
            raise TypeError("v3_usage_audit must be a mutable mapping")
        if geometry_scene_state is not None and not isinstance(
            geometry_scene_state, Mapping
        ):
            raise TypeError("geometry_scene_state must be a mapping or None")
        eligible.sort(key=_candidate_sort_key)
        baseline_order = list(eligible)
        scene = instance if geometry_scene_state is None else geometry_scene_state

        def semantic_features(candidate: dict[str, Any]) -> dict[str, Any]:
            value = candidate_semantic_features(
                scene,
                candidate,
                geometry_policy=geometry_policy,
            )
            if isinstance(value, HPKUnresolved):
                return {
                    "action_mode": str(candidate.get("action_mode", "unknown")),
                    "geometry_source_class": "unknown",
                }
            return value.to_dict()

        expected = v3_action_match.selected_candidate_geometry
        features_by_position = [
            semantic_features(candidate) for candidate in baseline_order
        ]
        preferred_positions = [
            index
            for index, features in enumerate(features_by_position)
            if all(features.get(key) == value for key, value in expected.items())
        ]
        preferred = [baseline_order[index] for index in preferred_positions]
        preferred_ids = {
            str(candidate.get("candidate_id", "")) for candidate in preferred
        }
        ranked_order = preferred + [
            candidate
            for candidate in baseline_order
            if str(candidate.get("candidate_id", "")) not in preferred_ids
        ]
        if requested:
            selected = next(
                (
                    candidate
                    for candidate in baseline_order
                    if str(candidate.get("candidate_id", "")) == requested
                ),
                None,
            )
        else:
            selected = ranked_order[0] if ranked_order else None
        baseline_ref = (
            str(baseline_order[0].get("candidate_id", ""))
            if baseline_order
            else ""
        )
        selected_ref = "" if selected is None else str(selected.get("candidate_id", ""))
        adopted = bool(selected is not None and selected_ref in preferred_ids)
        v3_usage_audit.clear()
        v3_usage_audit.update(
            {
                "mode": "full",
                "knowledge_adopted": adopted,
                "behavior_changed": bool(adopted and selected_ref != baseline_ref),
                "rank_before": [
                    {"rank": index, "geometry": features}
                    for index, features in enumerate(features_by_position)
                ],
                "rank_after": [
                    {"rank": index, "geometry": semantic_features(candidate)}
                    for index, candidate in enumerate(ranked_order)
                ],
                "selected_candidate_features": (
                    None if selected is None else semantic_features(selected)
                ),
                "retrieval_reason": v3_action_match.reason,
                "retrieved_knowledge": v3_action_match.compact_context(),
            }
        )
        return selected
    elif v3_usage_audit is not None:
        raise ValueError("v3_usage_audit requires v3_action_match")

    if geometric_strategy is None:
        if requested:
            return next(
                (item for item in eligible if str(item.get("candidate_id", "")) == requested),
                None,
            )
        eligible.sort(key=_candidate_sort_key)
        return eligible[0] if eligible else None

    # Import only for an explicitly active strategy.  The default/off path
    # keeps the selector's prior imports, result, errors, and side effects.
    from roboharn_evo.agent.hpk.geometry_policy import (
        rank_operation_pose_candidates,
        validate_private_ranking_audit_sink,
    )

    validate_private_ranking_audit_sink(ranking_audit)
    if geometry_scene_state is not None and not isinstance(
        geometry_scene_state,
        dict,
    ):
        if not isinstance(geometry_scene_state, Mapping):
            raise TypeError("geometry_scene_state must be a mapping or None")
    eligible.sort(key=_candidate_sort_key)
    baseline_order = list(eligible)
    ranking = rank_operation_pose_candidates(
        eligible,
        geometric_strategy=geometric_strategy,
        geometry_policy=geometry_policy,
        scene_state=(
            instance
            if geometry_scene_state is None
            else geometry_scene_state
        ),
    )
    selected = ranking.select(requested)
    ranking.write_private_audit(
        ranking_audit,
        selected_candidate=selected,
        requested_candidate_id=requested,
    )
    return selected


def ranked_eligible_operation_pose_candidates(
    instance: dict[str, Any],
    *,
    arm: str,
    action_mode: str,
    blocked_candidate_ids: Any = None,
    requested_target_id: Any = None,
) -> list[dict[str, Any]]:
    """Return the existing baseline order after its existing eligibility gate.

    This is a read-only production seam for evolving safe exploration.  It
    deliberately does not accept a rank, exploration marker, task, seed, or
    result.  Callers may only choose from the same arm/mode/target/blocked set
    that the baseline selector already uses.
    """

    normalized_arm = str(arm or "").strip().lower()
    normalized_mode = str(action_mode or "").strip().lower()
    blocked = {
        str(item)
        for item in (
            blocked_candidate_ids
            if isinstance(blocked_candidate_ids, (list, tuple, set))
            else []
        )
        if str(item)
    }
    requested_target = str(requested_target_id or "").strip()
    eligible = [
        item
        for item in operation_pose_candidates(instance)
        if str(item.get("arm", "")).strip().lower() == normalized_arm
        and str(item.get("action_mode", "")).strip().lower() == normalized_mode
        and str(item.get("candidate_id", "")) not in blocked
        and _candidate_allows_safe_baseline_alternative(item)
        and (
            not requested_target
            or str(item.get("target_id", "") or "").strip() == requested_target
        )
    ]
    eligible.sort(key=_candidate_sort_key)
    return eligible


def _candidate_allows_safe_baseline_alternative(
    candidate: dict[str, Any],
) -> bool:
    """Honor explicit existing Guard/feasibility facts for exploration only."""

    if any(
        key in candidate and candidate.get(key) is not True
        for key in ("valid", "legal", "eligible", "feasible")
    ):
        return False
    if any(
        key in candidate and candidate.get(key) is not True
        for key in ("reachable", "reachable_estimate", "support_valid", "free")
    ):
        return False
    if any(
        candidate.get(key) is True
        for key in ("blocked", "unreachable", "support_invalid", "occupied")
    ):
        return False
    return True


def materialize_operation_candidate(
    instance: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    materialized = dict(instance)
    if not isinstance(candidate, dict):
        return materialized
    action_mode = str(candidate.get("action_mode", "") or "").strip().lower()
    approach_pose = _pose7(candidate.get("approach_pose"))
    target_pose = _pose7(candidate.get("ee_target_pose"))
    object_contact_pose = _pose7(candidate.get("object_contact_pose"))
    tcp_pose = _pose7(candidate.get("tcp_pose"))
    if approach_pose is not None:
        materialized["approach_world_m"] = approach_pose[:3]
        materialized["approach_quat_wxyz"] = approach_pose[3:7]
    if target_pose is not None and action_mode in SUPPORTED_OPERATION_ACTION_MODES:
        materialized[f"{action_mode}_world_m"] = target_pose[:3]
        materialized[f"{action_mode}_quat_wxyz"] = target_pose[3:7]
    if object_contact_pose is not None:
        materialized["selected_object_contact_pose"] = object_contact_pose
    if tcp_pose is not None:
        materialized["selected_tcp_pose"] = tcp_pose
    materialized["selected_operation_candidate_id"] = candidate.get("candidate_id")
    materialized["selected_operation_target_id"] = candidate.get("target_id")
    materialized["selected_operation_action_mode"] = action_mode
    materialized["selected_operation_arm"] = candidate.get("arm")
    materialized["selected_operation_candidate"] = dict(candidate)
    return materialized


def with_dynamic_place_candidates(
    scene_memory: dict[str, Any],
    *,
    manipulation_state: dict[str, Any] | None,
    robot_state: dict[str, Any] | None,
    tcp_calibration_by_arm: dict[str, dict[str, Any]] | None = None,
    active_grasp_transport_policy: Any = None,
) -> dict[str, Any]:
    """Attach runtime-private place candidates and compact public target groups.

    Placement candidates are generated only for attachments authorized by the
    active transport policy: verifier-confirmed state in strict mode, or an
    exact close-time provisional state in evidence-only mode.  The public
    target group is stable and semantic (`target_id`); the arm-specific
    executable pose remains in the held instance's private candidate list.
    """

    payload = dict(scene_memory or {})
    raw_instances = payload.get("instances")
    if not isinstance(raw_instances, list):
        payload[OPERATION_TARGETS_KEY] = []
        payload[SPATIAL_STATE_KEY] = spatial_state_from_scene(payload)
        return payload

    instances: list[dict[str, Any]] = []
    for raw_instance in raw_instances:
        if not isinstance(raw_instance, dict):
            continue
        instance = dict(raw_instance)
        retained = [
            dict(item)
            for item in operation_pose_candidates(instance)
            if str(item.get("action_mode", "") or "").strip().lower() != "place"
        ]
        instance[OPERATION_POSE_CANDIDATES_KEY] = retained
        instance["operation_pose_candidate_count"] = len(retained)
        instances.append(instance)

    state = manipulation_state if isinstance(manipulation_state, dict) else {}
    robot = robot_state if isinstance(robot_state, dict) else {}
    calibrations = (
        tcp_calibration_by_arm
        if isinstance(tcp_calibration_by_arm, dict)
        else {}
    )
    targets_by_id: dict[str, dict[str, Any]] = {}
    for arm in ("left", "right"):
        arm_state = state.get(arm)
        if not manipulation_state_allows_transport(
            arm_state,
            active_grasp_transport_policy=(
                active_grasp_transport_policy
            ),
        ):
            continue
        held_ref = str(arm_state.get("held_instance_id", "") or "").strip()
        held = _resolve_instance(instances, held_ref)
        if held is None:
            continue
        held_id = str(held.get("instance_id", "") or "").strip()
        held_extent = _instance_extent(held)
        action_pose = _robot_action_pose(robot.get(arm))
        current_object = _instance_world(held)
        if (
            not held_id
            or held_extent is None
            or action_pose is None
        ):
            continue
        attachment = arm_state.get("held_object_to_tcp_attachment")
        propagated_object = _object_world_from_attachment(
            current_action_pose=action_pose,
            calibration=calibrations.get(arm),
            attachment=attachment,
        )
        if propagated_object is not None:
            current_object = propagated_object
        if current_object is None:
            continue
        target_specs = _placement_target_specs(
            payload,
            instances=instances,
            held=held,
        )
        private_candidates: list[dict[str, Any]] = []
        for source_index, target in enumerate(target_specs):
            candidate = _place_candidate_for_arm(
                target,
                arm=arm,
                held=held,
                held_extent=held_extent,
                current_object=current_object,
                current_action_pose=action_pose,
                calibration=calibrations.get(arm),
                attachment=attachment,
                scene_instances=instances,
                source_index=source_index,
            )
            if candidate is None:
                continue
            candidate["holding_status"] = (
                "assumed_from_command"
                if arm_state.get("execution_evidence_enabled") is False
                else "verified"
                if arm_state.get("holding_confirmed") is True
                else "provisional_evidence_only"
            )
            candidate["grasp_transport_policy"] = str(
                arm_state.get("grasp_transport_policy", "strict")
                or "strict"
            )
            candidate["post_release_verification_required"] = arm_state.get("execution_evidence_enabled") is not False
            validation = validate_place_candidate(
                {**payload, "instances": instances},
                held_instance=held,
                candidate=candidate,
            )
            candidate.update(validation)
            if validation.get("valid") is not True:
                continue
            private_candidates.append(candidate)
            target_id = str(candidate.get("target_id", "") or "")
            summary = targets_by_id.setdefault(
                target_id,
                {
                    "target_id": target_id,
                    "action_mode": "place",
                    "held_instance_id": candidate.get(
                        "held_instance_id"
                    ),
                    "target_kind": candidate.get("target_kind"),
                    "object_target_world_m": candidate.get(
                        "held_object_target_world_m"
                    ),
                    "support_instance_id": candidate.get(
                        "support_instance_id"
                    ),
                    "support_evidence": candidate.get(
                        "support_evidence"
                    ),
                    "vacated_by_instance_id": candidate.get(
                        "vacated_by_instance_id"
                    ),
                    "support_valid": True,
                    "free": True,
                    "arm_options": [],
                    "holding_status": candidate.get("holding_status"),
                    "grasp_transport_policy": candidate.get(
                        "grasp_transport_policy"
                    ),
                    "priority": candidate.get("priority"),
                },
            )
            if summary.get("holding_status") != candidate.get(
                "holding_status"
            ):
                summary["holding_status"] = "mixed"
                summary["grasp_transport_policy"] = "mixed"
            if candidate.get("target_kind") == RELATIONAL_PLACE_TARGET_KIND:
                summary.update(
                    {
                        key: candidate.get(key)
                        for key in (
                            "placement_relation",
                            "goal_target_role",
                            "goal_target_relation",
                            "reference_region_id",
                            "reference_object_id",
                            "reference_instance_ids",
                            "expected_count",
                            "observed_member_count",
                            "reference_evidence_status",
                        )
                    }
                )
            if arm not in summary["arm_options"]:
                summary["arm_options"].append(arm)

        existing = list(held.get(OPERATION_POSE_CANDIDATES_KEY, []) or [])
        held[OPERATION_POSE_CANDIDATES_KEY] = existing + private_candidates
        held["operation_pose_candidate_count"] = len(
            held[OPERATION_POSE_CANDIDATES_KEY]
        )
    public_targets = list(targets_by_id.values())
    public_targets.sort(
        key=lambda item: (
            _finite_number(item.get("priority"), default=float("inf")),
            str(item.get("target_id", "")),
        )
    )
    for item in public_targets:
        item["arm_options"] = sorted(set(item.get("arm_options", [])))
    payload["instances"] = instances
    payload[OPERATION_TARGETS_KEY] = public_targets
    payload[SPATIAL_STATE_KEY] = spatial_state_from_scene(payload)
    return payload


def capture_held_object_to_tcp_attachment(
    *,
    object_world_m: Any,
    robot_arm_state: Any,
    calibration: Any = None,
) -> dict[str, Any] | None:
    """Capture a rotation-aware local centroid-to-TCP offset.

    Scene Memory does not provide a full object orientation.  We therefore use
    a proxy object frame aligned with TCP at capture time and store the centroid
    offset in TCP coordinates.  Current TCP rotation can then propagate the
    attached centroid through wrist rotations without relying on stale vision.
    """

    object_world = _xyz(object_world_m)
    action_pose = _robot_action_pose(robot_arm_state)
    current_action = pose7_to_matrix(action_pose)
    if object_world is None or current_action is None:
        return None
    current_tcp = current_action @ action_to_tcp_transform(calibration)
    tcp_offset_local = current_tcp[:3, :3].T @ (
        current_tcp[:3, 3] - object_world
    )
    return {
        "object_proxy_frame": "tcp_aligned_at_capture",
        "object_centroid_to_tcp_translation_tcp_m": _round_list(
            tcp_offset_local
        ),
        "capture_tcp_pose": matrix_to_pose7(current_tcp),
        "source": "verified_grasp_scene_and_robot_tcp",
    }


def propagate_held_object_world_m(
    *,
    robot_arm_state: Any,
    attachment: Any,
    calibration: Any = None,
) -> list[float] | None:
    """Return the current attached-object centroid from the calibrated TCP.

    This is the public, read-only counterpart of the placement generator's
    internal propagation.  It lets Scene Memory update a transport-authorized
    object's current position without pretending that an occluded object was
    freshly segmented; the caller remains responsible for preserving whether
    the attachment is confirmed or provisional.
    """

    action_pose = _robot_action_pose(robot_arm_state)
    if action_pose is None:
        return None
    world = _object_world_from_attachment(
        current_action_pose=action_pose,
        calibration=calibration,
        attachment=attachment,
    )
    return (
        None
        if world is None
        else _round_list(world)
    )


def validate_place_candidate(
    scene_memory: dict[str, Any],
    *,
    held_instance: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Recompute occupancy/support from current scene geometry.

    Candidate-carried booleans are deliberately ignored.  This function is used
    both while generating a candidate and immediately before motion/release.
    """

    instances = [
        item
        for item in (scene_memory.get("instances", []) if isinstance(scene_memory, dict) else [])
        if isinstance(item, dict)
    ]
    held_id = str(held_instance.get("instance_id", "") or "").strip()
    target_id = str(candidate.get("target_id", "") or "").strip()
    target_kind = str(candidate.get("target_kind", "") or "").strip().lower()
    target = _xyz(candidate.get("held_object_target_world_m"))
    held_extent = _xyz(candidate.get("held_extent_m"))
    if held_extent is None:
        held_extent = _instance_extent(held_instance)
    support_id = str(candidate.get("support_instance_id", "") or "").strip()
    support = _resolve_instance(instances, support_id) if support_id else None
    support_valid = False
    target_drift_m: float | None = None
    relation_validation: dict[str, Any] = {}

    if target is not None and held_extent is not None:
        if target_kind == "object_top" and support is not None:
            support_top = _instance_top_z(support)
            support_world = _instance_world(support)
            if (
                support_top is not None
                and support_world is not None
                and _object_top_support_valid(
                    support,
                    held_extent=held_extent,
                )
            ):
                expected = np.asarray(
                    [
                        support_world[0],
                        support_world[1],
                        support_top + abs(float(held_extent[2])) / 2.0,
                    ],
                    dtype=np.float64,
                )
                target_drift_m = float(np.linalg.norm(expected - target))
                support_valid = target_drift_m <= _PLACE_TARGET_REVALIDATION_TOLERANCE_M
        elif target_kind in {"free_support", "vacated_pose"}:
            support_plane_z = _support_plane_z(instances)
            if support_plane_z is not None:
                expected_z = support_plane_z + abs(float(held_extent[2])) / 2.0
                target_drift_m = abs(float(target[2]) - expected_z)
                support_valid = target_drift_m <= _PLACE_TARGET_REVALIDATION_TOLERANCE_M
        elif target_kind == RELATIONAL_PLACE_TARGET_KIND:
            relation_validation = revalidate_relational_place_target(
                scene_memory,
                candidate=candidate,
                held_extent_m=held_extent,
                support_plane_z=_support_plane_z(instances),
            )
            relation_drift = relation_validation.get(
                "target_geometry_drift_m"
            )
            if relation_drift is not None:
                target_drift_m = float(relation_drift)
            support_valid = bool(
                relation_validation.get("support_valid")
                and target_drift_m is not None
                and target_drift_m
                <= _PLACE_TARGET_REVALIDATION_TOLERANCE_M
            )

    ignored = {held_id}
    if target_kind == "object_top" and support_id:
        ignored.add(support_id)
    if target_kind == RELATIONAL_PLACE_TARGET_KIND:
        ignored.update(
            str(item or "").strip()
            for item in relation_validation.get(
                "occupancy_exempt_instance_ids",
                [],
            )
            if str(item or "").strip()
        )
    occupied_by = (
        _occupants_at_object_target(
            instances,
            target=target,
            extent=held_extent,
            ignored_instance_ids=ignored,
        )
        if target is not None and held_extent is not None
        else []
    )
    free = not occupied_by
    return {
        "place_target_revalidated": True,
        "target_id": target_id,
        "target_kind": target_kind,
        "support_valid": bool(support_valid),
        "free": bool(free),
        "occupied_by": occupied_by,
        "target_geometry_drift_m": (
            None if target_drift_m is None else round(target_drift_m, 6)
        ),
        "valid": bool(target_id and support_valid and free),
    }


def spatial_state_from_scene(scene_memory: dict[str, Any]) -> dict[str, Any]:
    """Return an identity-bound, coordinate-derived scene arrangement summary."""

    instances = [
        item
        for item in (scene_memory.get("instances", []) if isinstance(scene_memory, dict) else [])
        if isinstance(item, dict)
    ]
    records: list[dict[str, Any]] = []
    for instance in instances:
        instance_id = str(instance.get("instance_id", "") or "").strip()
        world = _instance_world(instance)
        if not instance_id or world is None:
            continue
        records.append(
            {
                "instance_id": instance_id,
                "world_m": _round_list(world),
                "extent_m": (
                    _round_list(extent)
                    if (extent := _instance_extent(instance)) is not None
                    else None
                ),
            }
        )
    records.sort(key=lambda item: item["instance_id"])
    order_by_x = [
        item["instance_id"]
        for item in sorted(
            records,
            key=lambda item: (
                float(item["world_m"][0]),
                float(item["world_m"][1]),
                item["instance_id"],
            ),
        )
    ]
    order_by_y = [
        item["instance_id"]
        for item in sorted(
            records,
            key=lambda item: (
                float(item["world_m"][1]),
                float(item["world_m"][0]),
                item["instance_id"],
            ),
        )
    ]
    overlap_pairs: list[list[str]] = []
    support_pairs: list[list[str]] = []
    for index, first in enumerate(records):
        first_extent = _xyz(first.get("extent_m"))
        first_world = _xyz(first.get("world_m"))
        if first_extent is None or first_world is None:
            continue
        for second in records[index + 1 :]:
            second_extent = _xyz(second.get("extent_m"))
            second_world = _xyz(second.get("world_m"))
            if second_extent is None or second_world is None:
                continue
            if _horizontal_aabb_overlap(
                first_world,
                first_extent,
                second_world,
                second_extent,
                margin=0.0,
            ):
                pair = [first["instance_id"], second["instance_id"]]
                overlap_pairs.append(pair)
                first_top = float(first_world[2] + abs(first_extent[2]) / 2.0)
                second_top = float(second_world[2] + abs(second_extent[2]) / 2.0)
                if abs(float(second_world[2] - abs(second_extent[2]) / 2.0) - first_top) <= 0.025:
                    support_pairs.append([first["instance_id"], second["instance_id"]])
                elif abs(float(first_world[2] - abs(first_extent[2]) / 2.0) - second_top) <= 0.025:
                    support_pairs.append([second["instance_id"], first["instance_id"]])

    signature_payload = [
        [
            item["instance_id"],
            *[round(float(value) / 0.01) for value in item["world_m"]],
        ]
        for item in records
    ]
    signature = hashlib.sha256(
        json.dumps(signature_payload, separators=(",", ":"), sort_keys=False).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return {
        "coordinate_source": "latest_world_m_else_world_m",
        "signature_resolution_m": 0.01,
        "signature": signature,
        "instances": records,
        "order_by_x": order_by_x,
        "order_by_y": order_by_y,
        "horizontal_overlap_pairs": overlap_pairs,
        "support_pairs": support_pairs,
    }


def _placement_target_specs(
    scene_memory: dict[str, Any],
    *,
    instances: list[dict[str, Any]],
    held: dict[str, Any],
) -> list[dict[str, Any]]:
    held_id = str(held.get("instance_id", "") or "").strip()
    held_extent = _instance_extent(held)
    support_plane_z = _support_plane_z(instances)
    if not held_id or held_extent is None or support_plane_z is None:
        return []
    held_half_z = abs(float(held_extent[2])) / 2.0
    surface_target_z = support_plane_z + held_half_z
    specs: list[dict[str, Any]] = build_relational_place_target_specs(
        scene_memory,
        held_instance_id=held_id,
        held_extent_m=held_extent,
        support_plane_z=support_plane_z,
        source_supported_world=held.get("first_observed_world_m"),
    )

    for instance in instances:
        owner_id = str(instance.get("instance_id", "") or "").strip()
        first = _xyz(instance.get("first_observed_world_m"))
        if not owner_id or first is None:
            continue
        target = np.asarray([first[0], first[1], surface_target_z], dtype=np.float64)
        if _occupants_at_object_target(
            instances,
            target=target,
            extent=held_extent,
            ignored_instance_ids={held_id},
        ):
            continue
        specs.append(
            {
                "target_id": (
                    f"place:held:{held_id}:vacated:{owner_id}"
                ),
                "target_kind": "vacated_pose",
                "held_object_target_world_m": _round_list(target),
                "vacated_by_instance_id": owner_id,
                "support_instance_id": None,
                "support_evidence": "observed_prior_support_pose",
                "priority": 0.0 if owner_id == held_id else 1.0,
            }
        )

    for side_id, target in (
        _free_support_positions(
            instances,
            held_id=held_id,
            held_extent=held_extent,
            support_plane_z=support_plane_z,
        )
    ):
        if _near_existing_target(specs, target):
            continue
        specs.append(
            {
                "target_id": (
                    f"place:held:{held_id}:free-support:{side_id}"
                ),
                "target_kind": "free_support",
                "held_object_target_world_m": _round_list(target),
                "vacated_by_instance_id": None,
                "support_instance_id": None,
                "support_evidence": "coplanar_support_plane_inference",
                "priority": 2.0,
            }
        )

    for anchor in instances:
        anchor_id = str(anchor.get("instance_id", "") or "").strip()
        if not anchor_id or anchor_id == held_id:
            continue
        anchor_world = _instance_world(anchor)
        anchor_top = _instance_top_z(anchor)
        if (
            anchor_world is None
            or anchor_top is None
            or not _object_top_support_valid(
                anchor,
                held_extent=held_extent,
            )
        ):
            continue
        target = np.asarray(
            [anchor_world[0], anchor_world[1], anchor_top + held_half_z],
            dtype=np.float64,
        )
        if _occupants_at_object_target(
            instances,
            target=target,
            extent=held_extent,
            ignored_instance_ids={held_id, anchor_id},
        ):
            continue
        specs.append(
            {
                "target_id": (
                    f"place:held:{held_id}:object-top:{anchor_id}"
                ),
                "target_kind": "object_top",
                "held_object_target_world_m": _round_list(target),
                "vacated_by_instance_id": None,
                "support_instance_id": anchor_id,
                "support_evidence": "grounded_object_top_surface",
                "priority": 4.0,
            }
        )

    deduped: list[dict[str, Any]] = []
    for spec in sorted(
        specs,
        key=lambda item: (
            _finite_number(item.get("priority"), default=float("inf")),
            str(item.get("target_id", "")),
        ),
    ):
        target = _xyz(spec.get("held_object_target_world_m"))
        if target is None or _near_existing_target(deduped, target):
            continue
        deduped.append(spec)
    return deduped


def _free_support_positions(
    instances: list[dict[str, Any]],
    *,
    held_id: str,
    held_extent: np.ndarray,
    support_plane_z: float,
) -> list[tuple[str, np.ndarray]]:
    points: list[np.ndarray] = []
    radii: list[float] = []
    for instance in instances:
        if str(instance.get("instance_id", "") or "").strip() == held_id:
            continue
        world = _instance_world(instance)
        extent = _instance_extent(instance)
        if world is None or extent is None:
            continue
        points.append(np.asarray(world[:2], dtype=np.float64))
        radii.append(float(np.linalg.norm(extent[:2]) / 2.0))
    if not points:
        return []
    matrix = np.stack(points, axis=0)
    center = np.median(matrix, axis=0)
    if len(points) >= 2:
        centered = matrix - center
        covariance = centered.T @ centered
        values, vectors = np.linalg.eigh(covariance)
        principal = vectors[:, int(np.argmax(values))]
    else:
        principal = np.asarray([1.0, 0.0], dtype=np.float64)
    principal = _canonical_axis(principal)
    normal = np.asarray([-principal[1], principal[0]], dtype=np.float64)
    projections = (matrix - center) @ normal
    held_radius = float(np.linalg.norm(held_extent[:2]) / 2.0)
    scene_radius = max(radii) if radii else held_radius
    clearance = held_radius + scene_radius + _PLACE_HORIZONTAL_MARGIN_M
    result: list[tuple[str, np.ndarray]] = []
    for sign in (-1.0, 1.0):
        edge = float(np.min(projections)) if sign < 0.0 else float(np.max(projections))
        xy = center + normal * (edge + sign * clearance)
        target = np.asarray(
            [
                xy[0],
                xy[1],
                support_plane_z + abs(float(held_extent[2])) / 2.0,
            ],
            dtype=np.float64,
        )
        if not _occupants_at_object_target(
            instances,
            target=target,
            extent=held_extent,
            ignored_instance_ids={held_id},
        ):
            result.append(
                ("negative" if sign < 0.0 else "positive", target)
            )
    return result[:2]


def _place_candidate_for_arm(
    target: dict[str, Any],
    *,
    arm: str,
    held: dict[str, Any],
    held_extent: np.ndarray,
    current_object: np.ndarray,
    current_action_pose: np.ndarray,
    calibration: Any,
    attachment: Any,
    scene_instances: list[dict[str, Any]],
    source_index: int,
) -> dict[str, Any] | None:
    object_target = _xyz(target.get("held_object_target_world_m"))
    current_action = pose7_to_matrix(current_action_pose)
    if object_target is None or current_action is None:
        return None
    action_to_tcp = action_to_tcp_transform(calibration)
    current_tcp = current_action @ action_to_tcp
    attachment_local = (
        _xyz(
            attachment.get(
                "object_centroid_to_tcp_translation_tcp_m"
            )
        )
        if isinstance(attachment, dict)
        else None
    )
    if attachment_local is None:
        return None
    object_to_tcp_world = current_tcp[:3, :3] @ attachment_local
    current_object = current_tcp[:3, 3] - object_to_tcp_world
    attachment_source = "verifier_confirmed_tcp_local_attachment"
    clearance = max(
        _PLACE_CLEARANCE_MARGIN_M,
        abs(float(held_extent[2])) + _PLACE_CLEARANCE_MARGIN_M,
    )
    target_tcp = current_tcp.copy()
    target_tcp[:3, 3] = object_target + object_to_tcp_world
    approach_tcp = target_tcp.copy()
    approach_tcp[2, 3] += clearance
    try:
        tcp_to_action = np.linalg.inv(action_to_tcp)
    except np.linalg.LinAlgError:
        return None
    target_action_pose = matrix_to_pose7(target_tcp @ tcp_to_action)
    approach_action_pose = matrix_to_pose7(approach_tcp @ tcp_to_action)
    target_tcp_pose = matrix_to_pose7(target_tcp)
    if (
        target_action_pose is None
        or approach_action_pose is None
        or target_tcp_pose is None
    ):
        return None
    reach_distance = float(
        np.linalg.norm(
            np.asarray(target_action_pose[:3], dtype=np.float64)
            - current_action[:3, 3]
        )
    )
    scene_points = [
        point
        for item in scene_instances
        if (point := _instance_world(item)) is not None
    ]
    scene_diameter = 0.0
    for index, first in enumerate(scene_points):
        for second in scene_points[index + 1 :]:
            scene_diameter = max(
                scene_diameter,
                float(np.linalg.norm(first[:2] - second[:2])),
            )
    reach_bound = min(0.65, max(0.30, scene_diameter + 0.25))
    if reach_distance > reach_bound:
        return None
    target_id = str(target.get("target_id", "") or "").strip()
    held_id = str(held.get("instance_id", "") or "").strip()
    if not target_id or not held_id:
        return None
    return {
        "candidate_id": f"{target_id}:arm:{arm}",
        "target_id": target_id,
        "target_kind": target.get("target_kind"),
        "action_mode": "place",
        "arm": arm,
        "held_instance_id": held_id,
        "ee_target_pose": target_action_pose,
        "approach_pose": approach_action_pose,
        "tcp_pose": target_tcp_pose,
        "object_contact_pose": [
            *_round_list(object_target),
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        "held_object_target_world_m": _round_list(object_target),
        "held_extent_m": _round_list(held_extent),
        "held_object_to_tcp_translation_world_m": _round_list(
            object_to_tcp_world
        ),
        "attachment_transform_source": attachment_source,
        "orientation_policy": "preserve_current_rigid_attachment",
        "support_instance_id": target.get("support_instance_id"),
        "support_evidence": target.get("support_evidence"),
        "vacated_by_instance_id": target.get("vacated_by_instance_id"),
        "placement_relation": target.get("placement_relation"),
        "goal_target_role": target.get("goal_target_role"),
        "goal_target_relation": target.get("goal_target_relation"),
        "source_supported_world_m": target.get("source_supported_world_m"),
        "reference_region_id": target.get("reference_region_id"),
        "reference_object_id": target.get("reference_object_id"),
        "reference_instance_ids": list(
            target.get("reference_instance_ids", []) or []
        ),
        "occupancy_exempt_instance_ids": list(
            target.get("occupancy_exempt_instance_ids", []) or []
        ),
        "expected_count": target.get("expected_count"),
        "observed_member_count": target.get("observed_member_count"),
        "reference_evidence_status": target.get(
            "reference_evidence_status"
        ),
        "approach_clearance_m": round(clearance, 6),
        "reach_distance_m": round(reach_distance, 6),
        "reachable_estimate": True,
        "priority": _finite_number(target.get("priority"), default=0.0)
        + reach_distance,
        "source_candidate_index": int(source_index),
        "geometry_source": "runtime_dynamic_place_geometry",
    }


def _object_world_from_attachment(
    *,
    current_action_pose: np.ndarray,
    calibration: Any,
    attachment: Any,
) -> np.ndarray | None:
    if not isinstance(attachment, dict):
        return None
    local = _xyz(
        attachment.get("object_centroid_to_tcp_translation_tcp_m")
    )
    current_action = pose7_to_matrix(current_action_pose)
    if local is None or current_action is None:
        return None
    current_tcp = current_action @ action_to_tcp_transform(calibration)
    return current_tcp[:3, 3] - current_tcp[:3, :3] @ local


def _occupants_at_object_target(
    instances: list[dict[str, Any]],
    *,
    target: np.ndarray,
    extent: np.ndarray,
    ignored_instance_ids: set[str],
) -> list[str]:
    occupied: list[str] = []
    for instance in instances:
        instance_id = str(instance.get("instance_id", "") or "").strip()
        if not instance_id or instance_id in ignored_instance_ids:
            continue
        world = _instance_world(instance)
        other_extent = _instance_extent(instance)
        if world is None or other_extent is None:
            continue
        if not _horizontal_aabb_overlap(
            target,
            extent,
            world,
            other_extent,
            margin=_PLACE_HORIZONTAL_MARGIN_M,
        ):
            continue
        target_bottom = float(target[2] - abs(extent[2]) / 2.0)
        target_top = float(target[2] + abs(extent[2]) / 2.0)
        other_bottom = float(world[2] - abs(other_extent[2]) / 2.0)
        other_top = float(world[2] + abs(other_extent[2]) / 2.0)
        if target_bottom <= other_top + _PLACE_HORIZONTAL_MARGIN_M and other_bottom <= target_top + _PLACE_HORIZONTAL_MARGIN_M:
            occupied.append(instance_id)
    return sorted(set(occupied))


def _horizontal_aabb_overlap(
    first_world: np.ndarray,
    first_extent: np.ndarray,
    second_world: np.ndarray,
    second_extent: np.ndarray,
    *,
    margin: float,
) -> bool:
    for axis in (0, 1):
        separation = abs(float(first_world[axis] - second_world[axis]))
        required = (
            abs(float(first_extent[axis])) / 2.0
            + abs(float(second_extent[axis])) / 2.0
            + max(0.0, float(margin))
        )
        if separation >= required:
            return False
    return True


def _support_plane_z(instances: list[dict[str, Any]]) -> float | None:
    bottoms: list[float] = []
    for instance in instances:
        first = _xyz(instance.get("first_observed_world_m"))
        extent = _instance_extent(instance)
        if first is None or extent is None:
            continue
        bottoms.append(float(first[2] - abs(extent[2]) / 2.0))
    if not bottoms:
        return None
    return float(np.median(np.asarray(bottoms, dtype=np.float64)))


def _instance_world(instance: dict[str, Any]) -> np.ndarray | None:
    for key in ("latest_world_m", "world_m"):
        value = _xyz(instance.get(key))
        if value is not None:
            return value
    return None


def _instance_extent(instance: dict[str, Any]) -> np.ndarray | None:
    quality = instance.get("quality")
    extent = _xyz(
        quality.get("world_extent_m") if isinstance(quality, dict) else None
    )
    if extent is None or np.any(extent <= 1e-5):
        return None
    return np.abs(extent)


def _object_top_support_valid(
    instance: dict[str, Any],
    *,
    held_extent: np.ndarray,
) -> bool:
    support_extent = _instance_extent(instance)
    if support_extent is None:
        return False
    try:
        missing_steps = int(instance.get("missing_steps", 0) or 0)
    except (TypeError, ValueError):
        return False
    status = str(instance.get("status", "visible") or "").strip().lower()
    if missing_steps > 0 or status not in {"visible", "tracked"}:
        return False
    footprint_ratio = 0.75
    return all(
        float(support_extent[axis]) + _PLACE_HORIZONTAL_MARGIN_M
        >= float(abs(held_extent[axis])) * footprint_ratio
        for axis in (0, 1)
    )


def _instance_top_z(instance: dict[str, Any]) -> float | None:
    top = _xyz(instance.get("top_surface_world_m"))
    if top is not None:
        return float(top[2])
    quality = instance.get("quality")
    raw = quality.get("world_z_max_m") if isinstance(quality, dict) else None
    value = _finite_number(raw, default=float("nan"))
    if math.isfinite(value):
        return value
    world = _instance_world(instance)
    extent = _instance_extent(instance)
    if world is None or extent is None:
        return None
    return float(world[2] + extent[2] / 2.0)


def _resolve_instance(
    instances: list[dict[str, Any]],
    instance_ref: str,
) -> dict[str, Any] | None:
    normalized = str(instance_ref or "").strip().lower()
    if not normalized:
        return None
    matches = []
    for instance in instances:
        refs = {
            str(instance.get(key, "") or "").strip().lower()
            for key in (
                "instance_id",
                "track_id",
                "oracle_id",
                "oracle_source_path",
            )
            if str(instance.get(key, "") or "").strip()
        }
        if normalized in refs:
            matches.append(instance)
    return matches[0] if len(matches) == 1 else None


def _robot_action_pose(arm_state: Any) -> np.ndarray | None:
    if not isinstance(arm_state, dict):
        return None
    xyz = _xyz(arm_state.get("xyz"))
    quat = _quat(arm_state.get("quat_wxyz"))
    if xyz is None or quat is None:
        return None
    return np.concatenate([xyz, quat])


def _xyz(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if array.size != 3 or not np.all(np.isfinite(array)):
        return None
    return array


def _quat(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if array.size != 4 or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    return array / norm


def _near_existing_target(
    specs: list[dict[str, Any]],
    target: np.ndarray,
) -> bool:
    for spec in specs:
        existing = _xyz(spec.get("held_object_target_world_m"))
        if existing is not None and float(np.linalg.norm(existing - target)) <= _PLACE_POSITION_DEDUP_M:
            return True
    return False


def _canonical_axis(value: np.ndarray) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1e-8:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    axis = axis / norm
    if axis[0] < 0.0 or (abs(float(axis[0])) <= 1e-8 and axis[1] < 0.0):
        axis = -axis
    return axis


def _finite_number(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int, str]:
    try:
        priority = float(candidate.get("priority", 0.0))
    except (TypeError, ValueError):
        priority = 0.0
    if not math.isfinite(priority):
        priority = float("inf")
    try:
        source_index = int(candidate.get("source_candidate_index", 0))
    except (TypeError, ValueError):
        source_index = 0
    return priority, source_index, str(candidate.get("candidate_id", ""))


def _pose7(value: Any) -> list[float] | None:
    transform = pose7_to_matrix(value)
    return matrix_to_pose7(transform) if transform is not None else None


def _orthonormal_rotation(value: Any) -> np.ndarray:
    try:
        rotation = np.asarray(value, dtype=np.float64).reshape(3, 3)
    except Exception:
        return np.eye(3, dtype=np.float64)
    if not np.all(np.isfinite(rotation)):
        return np.eye(3, dtype=np.float64)
    try:
        u, _, vh = np.linalg.svd(rotation)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)
    result = u @ vh
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def _round_list(values: Any) -> list[float]:
    return [round(float(item), 6) for item in np.asarray(values).reshape(-1)]

"""ID-free semantic features derived from private operation candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from roboharn_evo.agent.hpk.schemas import (
    HPKUnresolved,
    CandidateGeometryFeaturesV1,
    contains_private_transfer_text,
)
from roboharn_evo.agent.hpk.policy_config import (
    LoadedHPKPolicy,
    load_default_geometry_policy,
    validate_geometry_policy_payload,
)


def _geometry_policy_payload(
    value: LoadedHPKPolicy | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return load_default_geometry_policy().payload
    if isinstance(value, LoadedHPKPolicy):
        if value.kind != "geometry":
            raise ValueError("candidate features require a geometry policy")
        return value.payload
    return validate_geometry_policy_payload(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _token(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector(value: Any, *, length: int = 3) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < length:
        return None
    result = tuple(_finite(item) for item in value[:length])
    if any(item is None for item in result):
        return None
    return tuple(float(item) for item in result if item is not None)


def _target_summary(
    scene_state: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    target_id = str(candidate.get("target_id", "") or "").strip()
    raw = scene_state.get("operation_targets")
    if not target_id or not isinstance(raw, list):
        return {}, False
    matches = [
        dict(item)
        for item in raw
        if isinstance(item, Mapping)
        and str(item.get("target_id", "") or "").strip() == target_id
    ]
    if len(matches) > 1:
        return {}, True
    return (matches[0], False) if matches else ({}, False)


def _relation(candidate: Mapping[str, Any], target: Mapping[str, Any]) -> str | None:
    for source in (target, candidate):
        for key in ("placement_relation", "target_relation"):
            if key not in source:
                continue
            value = source.get(key)
            if isinstance(value, Mapping):
                value = value.get("relation")
            return "center_of" if _token(value) == "center_of" else None
    return None


def _target_reference_role(
    candidate: Mapping[str, Any], target: Mapping[str, Any]
) -> str | None:
    for source in (target, candidate):
        for key in ("target_reference_role", "reference_role"):
            if key not in source:
                continue
            role = str(source.get(key, "") or "").strip()
            if not role or contains_private_transfer_text(role):
                return None
            return role
    return None


def _geometry_source_class(value: Any) -> str:
    source = _token(value)
    if not source:
        return "unknown"
    if "oracle" in source or "actor_contact_matrix" in source:
        return "oracle"
    if source.startswith("rgbd") or "observed_surface" in source:
        return "rgbd_observed"
    if source.startswith("runtime") or "relational" in source:
        return "runtime_relational"
    return "unknown"


def _combined_geometry_source_class(
    candidate: Mapping[str, Any], target: Mapping[str, Any]
) -> str:
    candidate_source = _geometry_source_class(
        candidate.get("geometry_source_class") or candidate.get("geometry_source")
    )
    target_source = _geometry_source_class(
        target.get("geometry_source_class") or target.get("geometry_source")
    )
    known = {value for value in (candidate_source, target_source) if value != "unknown"}
    if "oracle" in known:
        return "oracle"
    if len(known) == 1:
        return next(iter(known))
    return "unknown"


def _authoritative_boolean(
    candidate: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    target_keys: tuple[str, ...],
    candidate_keys: tuple[str, ...],
) -> bool | None:
    for source, keys in ((target, target_keys), (candidate, candidate_keys)):
        for key in keys:
            if key not in source:
                continue
            value = source.get(key)
            return value if isinstance(value, bool) else None
    return None


def _valid_place_attachment_geometry(candidate: Mapping[str, Any]) -> bool:
    if _token(candidate.get("action_mode")) != "place":
        return False
    if (
        _token(candidate.get("attachment_transform_source"))
        != "verifier_confirmed_tcp_local_attachment"
    ):
        return False
    holding_status = _token(candidate.get("holding_status"))
    transport_policy = _token(candidate.get("grasp_transport_policy"))
    return bool(
        (
            holding_status == "verified"
            and transport_policy in {"strict", "evidence_only"}
        )
        or (
            holding_status == "provisional_evidence_only"
            and transport_policy == "evidence_only"
        )
    )


def _approach_family(candidate: Mapping[str, Any], source: str) -> str:
    explicit = _token(candidate.get("approach_family"))
    if explicit in {
        "clearance_first",
        "surface_normal",
        "principal_axis_relative",
        "unconstrained",
    }:
        return explicit
    if source == "runtime_relational" and (
        _finite(candidate.get("approach_clearance_m")) is not None
        or _token(candidate.get("action_mode")) == "place"
    ):
        return "clearance_first"
    raw_source = _token(candidate.get("geometry_source"))
    if "surface_normal" in raw_source:
        return "surface_normal"
    if "principal_ax" in raw_source:
        return "principal_axis_relative"
    return "unconstrained"


def _direction_bucket(
    candidate: Mapping[str, Any],
    geometry_policy: Mapping[str, Any],
) -> str:
    explicit = _token(candidate.get("approach_direction_bucket"))
    if explicit in {"above", "below", "lateral", "oblique", "unknown"}:
        return explicit

    approach = _vector(candidate.get("approach_pose"))
    target = _vector(candidate.get("ee_target_pose"))
    if approach is not None and target is not None:
        direction = tuple(first - second for first, second in zip(approach, target))
    else:
        raw = _vector(candidate.get("approach_direction"))
        if raw is None:
            return "unknown"
        # Runtime approach directions point from clearance toward contact.  A
        # negative vertical direction therefore realizes an above approach.
        direction = (-raw[0], -raw[1], -raw[2])

    x_value, y_value, z_value = direction
    horizontal = math.hypot(x_value, y_value)
    vertical = abs(z_value)
    direction = geometry_policy["direction_classification"]
    epsilon = float(direction["epsilon"])
    vertical_ratio = float(direction["vertical_to_horizontal_ratio"])
    horizontal_ratio = float(direction["horizontal_to_vertical_ratio"])
    if horizontal <= epsilon and vertical <= epsilon:
        return "unknown"
    if vertical >= vertical_ratio * max(horizontal, epsilon):
        return "above" if z_value > 0.0 else "below"
    if horizontal >= horizontal_ratio * max(vertical, epsilon):
        return "lateral"
    return "oblique"


def _orientation_relation(candidate: Mapping[str, Any]) -> str:
    explicit = _token(candidate.get("orientation_relation"))
    if explicit in {
        "preserve_current_attachment",
        "align_principal_axis_0",
        "align_principal_axis_1",
        "unconstrained",
        "unknown",
    }:
        if (
            explicit == "preserve_current_attachment"
            and _token(candidate.get("action_mode")) == "place"
            and not _valid_place_attachment_geometry(candidate)
        ):
            return "unknown"
        return explicit
    policy = _token(candidate.get("orientation_policy"))
    if policy in {
        "preserve_current_attachment",
        "preserve_current_rigid_attachment",
    }:
        if _token(candidate.get("action_mode")) == "place" and not (
            _valid_place_attachment_geometry(candidate)
        ):
            return "unknown"
        return "preserve_current_attachment"
    axis_index = candidate.get("principal_axis_index")
    if axis_index in {0, 1}:
        return f"align_principal_axis_{axis_index}"
    geometry_source = _token(
        candidate.get("geometry_source_class") or candidate.get("geometry_source")
    )
    source_index = candidate.get("source_candidate_index")
    if (
        "principal_ax" in geometry_source
        and not isinstance(source_index, bool)
        and source_index in {0, 1}
    ):
        # RGB-D grounding names its two observed principal-axis alternatives
        # with source_candidate_index.  This projection is deliberately gated
        # by the trusted geometry-source family so an unrelated source index
        # cannot masquerade as an orientation relation.
        return f"align_principal_axis_{source_index}"
    return "unknown"


def _grasp_projection(
    candidate: Mapping[str, Any], *, geometry_source_class: str
) -> tuple[str, str | None]:
    explicit_region = _token(candidate.get("grasp_region"))
    semantic_part = str(candidate.get("semantic_part", "") or "").strip()
    semantic_observed = candidate.get("semantic_part_observed") is True
    if (
        explicit_region == "semantic_part"
        and semantic_observed
        and semantic_part
        and not contains_private_transfer_text(semantic_part)
    ):
        return "semantic_part", semantic_part
    if explicit_region in {"observed_surface", "object_body"} and (
        candidate.get("grasp_region_observed") is True
    ):
        return explicit_region, None
    if (
        _token(candidate.get("action_mode")) == "grasp"
        and geometry_source_class == "rgbd_observed"
    ):
        return "observed_surface", None
    return "unknown", None


def _grasp_width_bucket(
    candidate: Mapping[str, Any],
    geometry_policy: Mapping[str, Any],
) -> str:
    explicit = _token(candidate.get("grasp_width_bucket"))
    if explicit in {"narrow", "medium", "wide", "unknown"}:
        return explicit
    width = _finite(
        candidate.get("observed_grasp_width_m", candidate.get("grasp_width_m"))
    )
    if width is None or width < 0.0:
        return "unknown"
    thresholds = geometry_policy["grasp_width_buckets_m"]
    if width <= float(thresholds["narrow_max"]):
        return "narrow"
    if width <= float(thresholds["medium_max"]):
        return "medium"
    return "wide"


def _reach_bucket(
    candidate: Mapping[str, Any],
    geometry_policy: Mapping[str, Any],
) -> str:
    explicit = _token(candidate.get("reach_distance_bucket"))
    if explicit in {"near", "medium", "far", "unknown"}:
        return explicit
    distance = _finite(candidate.get("reach_distance_m"))
    if distance is None or distance < 0.0:
        return "unknown"
    thresholds = geometry_policy["reach_distance_buckets_m"]
    if distance <= float(thresholds["near_max"]):
        return "near"
    if distance <= float(thresholds["medium_max"]):
        return "medium"
    return "far"


def candidate_semantic_features(
    scene_state: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    *,
    geometry_policy: LoadedHPKPolicy | Mapping[str, Any] | None = None,
) -> CandidateGeometryFeaturesV1 | HPKUnresolved:
    """Project one candidate without returning its private association key."""

    scene = _mapping(scene_state)
    value = _mapping(candidate)
    policy = _geometry_policy_payload(geometry_policy)
    action_mode = _token(value.get("action_mode"))
    if action_mode not in {"contact", "grasp", "place"}:
        return HPKUnresolved(
            component="candidate_features",
            reason="action_mode_unresolved",
            missing_fields=("action_mode",),
        )
    target, target_ambiguous = _target_summary(scene, value)
    if target_ambiguous:
        return HPKUnresolved(
            component="candidate_features",
            reason="target_id_ambiguous_in_current_operation_targets",
            missing_fields=("unique_operation_target",),
        )
    source = _combined_geometry_source_class(value, target)
    support_valid = _authoritative_boolean(
        value,
        target,
        target_keys=("support_valid",),
        candidate_keys=("support_valid",),
    )
    target_free = _authoritative_boolean(
        value,
        target,
        target_keys=("target_region_free", "free"),
        candidate_keys=("target_region_free", "free"),
    )
    grasp_region, semantic_part = _grasp_projection(
        value,
        geometry_source_class=source,
    )
    payload = {
        "action_mode": action_mode,
        "target_relation": _relation(value, target),
        "target_reference_role": _target_reference_role(value, target),
        "support_valid": support_valid,
        "target_region_free": target_free,
        "approach_family": _approach_family(value, source),
        "approach_direction_bucket": _direction_bucket(value, policy),
        "orientation_relation": _orientation_relation(value),
        "grasp_region": grasp_region,
        "semantic_part": semantic_part,
        "grasp_width_bucket": _grasp_width_bucket(value, policy),
        "reach_distance_bucket": _reach_bucket(value, policy),
        "geometry_source_class": source,
    }
    try:
        return CandidateGeometryFeaturesV1.from_dict(payload)
    except ValueError as exc:
        return HPKUnresolved(
            component="candidate_features",
            reason=f"feature_validation_failed:{type(exc).__name__}",
        )


extract_candidate_features = candidate_semantic_features


__all__ = [
    "candidate_semantic_features",
    "extract_candidate_features",
]

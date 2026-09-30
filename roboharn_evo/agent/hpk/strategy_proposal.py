from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.candidate_features import candidate_semantic_features
from roboharn_evo.agent.hpk.geometry_policy import rank_operation_pose_candidates
from roboharn_evo.agent.hpk.schemas import (
    GEOMETRIC_STRATEGY_SCHEMA,
    HPKUnresolved,
    AbstractEffectV1,
    ConditionV1,
    GeometricStrategyV1,
    TaskStrategyV1,
    canonical_json_bytes,
    stable_content_id,
)

STRATEGY_DELTA_SCHEMA = "roboharn_evo/hpk/strategy_delta/v1"
STRATEGY_PROPOSAL_SCHEMA = "roboharn_evo/hpk/strategy_proposal/v1"
PROPOSAL_EVIDENCE_PACKET_SCHEMA = "roboharn_evo/hpk/proposal_input/v1"
CONDITION_V2_SCHEMA = "roboharn_evo/hpk/condition/v2"
TASK_STRATEGY_V2_SCHEMA = "roboharn_evo/hpk/task_strategy/v2"
GEOMETRIC_STRATEGY_V2_SCHEMA = "roboharn_evo/hpk/geometric_strategy/v2"
ABSTRACT_EFFECT_V2_SCHEMA = "roboharn_evo/hpk/abstract_effect/v2"
PROPOSAL_RANK_TRACE_SCHEMA = "roboharn_evo/hpk/proposal_rank_change_trace/v1"
VLM_GEOMETRY_PROPOSER_VERSION = "vlm_geometry_strategy_proposer/v1"

_OPERATIONS = frozenset({"contact", "grasp", "place"})
_ARMS = frozenset({"left", "right", "either"})
_APPROACH_FAMILIES = frozenset(
    {"clearance_first", "surface_normal", "principal_axis_relative", "unconstrained"}
)
_APPROACH_DIRECTIONS = frozenset({"above", "below", "lateral", "oblique"})
_ORIENTATION_RELATIONS = frozenset(
    {
        "preserve_current_attachment",
        "align_principal_axis_0",
        "align_principal_axis_1",
        "unconstrained",
    }
)
_GRASP_REGIONS = frozenset({"observed_surface", "object_body", "semantic_part"})
_PLACEMENT_RELATIONS = frozenset({"center_of"})
_HARD_CONSTRAINTS = frozenset({"support_valid", "target_region_free"})
_SOFT_PREFERENCES = frozenset({"lower_reach_distance"})
_AVOID = frozenset({"repeat_equivalent_failed_candidate"})
_STRATEGY_FAMILIES = frozenset(
    {"placement_relation", "observed_grasp_geometry", "contact_relation"}
)
_REFERENCE_FRAMES = frozenset(
    {
        "support_normal",
        "object_principal_axes",
        "current_attachment",
        "world_gravity",
        "unknown",
    }
)
_GEOMETRY_SOURCE_CLASSES = frozenset(
    {"rgbd_observed", "runtime_relational", "oracle", "unknown"}
)
_EFFECT_TYPES = frozenset({"grasp", "place", "contact", "no_effect", "unknown"})
_EFFECT_PREDICATES = frozenset(
    {
        "object_attached",
        "object_released",
        "gripper_empty",
        "object_supported_by_target",
        "target_relation_satisfied",
        "placement_stable",
    }
)
_TASK_DELTA_FIELDS = {
    "operation",
    "target_role",
    "target_relation",
    "preferred_arm",
    "subgoal_purpose",
}
_GEOMETRY_DELTA_FIELDS = {
    "approach_family",
    "approach_direction",
    "orientation_relation",
    "grasp_region",
    "semantic_part",
    "placement_relation",
    "add_hard_constraints",
    "add_avoid",
}
_FORBIDDEN_GEOMETRY_KEYS = frozenset(
    {
        "candidate_id",
        "selected_candidate_id",
        "selected_candidate_private_ref",
        "xyz",
        "position",
        "absolute_position",
        "pose",
        "absolute_pose",
        "ee_target_pose",
        "approach_pose",
        "quaternion",
        "joint_command",
        "joint_positions",
        "joint_trajectory",
    }
)
_FORBIDDEN_TEXT_RE = re.compile(
    r"(?:\bcandidate\s*(?:id|[_:#-])\s*[A-Za-z0-9]|"
    r"\b(?:absolute\s+)?(?:xyz|pose|se\s*\(\s*3\s*\))\b|"
    r"\bjoint\s+(?:command|positions?|trajectory)\b|"
    r"(?:^|\s)/(?:mnt|tmp|root|home|workspace)(?:/|\s|$))",
    re.IGNORECASE,
)


class StrategyProposalValidationError(ValueError):
    """A v2 proposal record violates the frozen minimal contract."""


def _fail(path: str, message: str) -> None:
    raise StrategyProposalValidationError(f"{path}: {message}")


def _canonical_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    try:
        return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except Exception as exc:
        raise StrategyProposalValidationError(
            f"{path}: must be finite strict JSON ({type(exc).__name__})"
        ) from exc


def _exact(value: Mapping[str, Any], fields: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != fields:
        _fail(
            path,
            f"fields mismatch: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}",
        )


def _text(value: Any, *, path: str, max_chars: int = 1200) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    normalized = " ".join(value.strip().split())
    if not normalized or len(normalized) > max_chars:
        _fail(path, f"must contain 1..{max_chars} characters")
    return normalized


def _optional_text(value: Any, *, path: str, max_chars: int = 1200) -> str | None:
    if value is None:
        return None
    return _text(value, path=path, max_chars=max_chars)


def _enum(value: Any, allowed: frozenset[str], *, path: str) -> str:
    token = _text(value, path=path, max_chars=120)
    if token not in allowed:
        _fail(path, f"unsupported value {token!r}")
    return token


def _optional_enum(
    value: Any,
    allowed: frozenset[str],
    *,
    path: str,
) -> str | None:
    if value is None:
        return None
    return _enum(value, allowed, path=path)


def _string_list(
    value: Any,
    *,
    path: str,
    allowed: frozenset[str] | None = None,
    max_items: int = 32,
) -> list[str]:
    if not isinstance(value, list) or len(value) > max_items:
        _fail(path, f"must be a list of at most {max_items} strings")
    result = [
        _text(item, path=f"{path}[{index}]", max_chars=240)
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        _fail(path, "must not contain duplicates")
    if allowed is not None:
        unsupported = sorted(set(result) - allowed)
        if unsupported:
            _fail(path, f"contains unsupported values: {unsupported}")
    return result


def _reject_private_geometry(value: Any, *, path: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_GEOMETRY_KEYS:
                _fail(f"{path}.{key}", "absolute/private geometry field is forbidden")
            _reject_private_geometry(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        if (
            len(value) in {3, 4, 6, 7, 14}
            and value
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            )
        ):
            _fail(path, "pose/joint-like numeric vector is forbidden")
        for index, child in enumerate(value):
            _reject_private_geometry(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _FORBIDDEN_TEXT_RE.search(value):
        _fail(path, "candidate identity, pose, path, or joint command is forbidden")


def _validate_condition_v2(value: Any, *, path: str) -> dict[str, Any]:
    payload = _canonical_mapping(value, path=path)
    _exact(
        payload,
        {
            "schema",
            "task_family",
            "operation",
            "manipulation_phase",
            "manipulated_role",
            "target_role",
            "held_state",
            "scene_predicates",
            "abstraction_version",
        },
        path=path,
    )
    if payload["schema"] not in {CONDITION_V2_SCHEMA, "tcm/afk/condition/v2"}:
        _fail(f"{path}.schema", f"must be {CONDITION_V2_SCHEMA}")
    result = {
        "schema": payload["schema"],
        "task_family": _text(payload["task_family"], path=f"{path}.task_family"),
        "operation": _enum(payload["operation"], _OPERATIONS, path=f"{path}.operation"),
        "manipulation_phase": _text(
            payload["manipulation_phase"], path=f"{path}.manipulation_phase"
        ),
        "manipulated_role": _text(
            payload["manipulated_role"], path=f"{path}.manipulated_role"
        ),
        "target_role": _optional_text(
            payload["target_role"], path=f"{path}.target_role"
        ),
        "held_state": _enum(
            payload["held_state"],
            frozenset({"held", "not_held", "unknown"}),
            path=f"{path}.held_state",
        ),
        "scene_predicates": _string_list(
            payload["scene_predicates"], path=f"{path}.scene_predicates"
        ),
        "abstraction_version": _text(
            payload["abstraction_version"], path=f"{path}.abstraction_version"
        ),
    }
    _reject_private_geometry(result, path=path)
    return result


def _validate_task_strategy_v2(value: Any, *, path: str) -> dict[str, Any]:
    payload = _canonical_mapping(value, path=path)
    _exact(
        payload,
        {
            "schema",
            "operation",
            "manipulated_role",
            "target_role",
            "target_relation",
            "manipulation_phase",
            "subgoal_purpose",
            "preferred_arm",
        },
        path=path,
    )
    if payload["schema"] not in {TASK_STRATEGY_V2_SCHEMA, "tcm/afk/task_strategy/v2"}:
        _fail(f"{path}.schema", f"must be {TASK_STRATEGY_V2_SCHEMA}")
    result = {
        "schema": payload["schema"],
        "operation": _enum(payload["operation"], _OPERATIONS, path=f"{path}.operation"),
        "manipulated_role": _text(
            payload["manipulated_role"], path=f"{path}.manipulated_role"
        ),
        "target_role": _optional_text(
            payload["target_role"], path=f"{path}.target_role"
        ),
        "target_relation": _optional_text(
            payload["target_relation"], path=f"{path}.target_relation"
        ),
        "manipulation_phase": _text(
            payload["manipulation_phase"], path=f"{path}.manipulation_phase"
        ),
        "subgoal_purpose": _text(
            payload["subgoal_purpose"], path=f"{path}.subgoal_purpose"
        ),
        "preferred_arm": _enum(
            payload["preferred_arm"], _ARMS, path=f"{path}.preferred_arm"
        ),
    }
    _reject_private_geometry(result, path=path)
    return result


def _validate_geometric_strategy_v2(value: Any, *, path: str) -> dict[str, Any]:
    payload = _canonical_mapping(value, path=path)
    _exact(
        payload,
        {
            "schema",
            "strategy_family",
            "reference_frame",
            "approach_family",
            "approach_direction",
            "orientation_relation",
            "grasp_region",
            "semantic_part",
            "placement_relation",
            "hard_constraints",
            "soft_preferences",
            "avoid",
            "geometry_source_class",
        },
        path=path,
    )
    if payload["schema"] not in {GEOMETRIC_STRATEGY_V2_SCHEMA, "tcm/afk/geometric_strategy/v2"}:
        _fail(f"{path}.schema", f"must be {GEOMETRIC_STRATEGY_V2_SCHEMA}")
    result = {
        "schema": payload["schema"],
        "strategy_family": _enum(
            payload["strategy_family"],
            _STRATEGY_FAMILIES,
            path=f"{path}.strategy_family",
        ),
        "reference_frame": _enum(
            payload["reference_frame"],
            _REFERENCE_FRAMES,
            path=f"{path}.reference_frame",
        ),
        "approach_family": _enum(
            payload["approach_family"],
            _APPROACH_FAMILIES,
            path=f"{path}.approach_family",
        ),
        "approach_direction": _enum(
            payload["approach_direction"],
            _APPROACH_DIRECTIONS | {"unknown"},
            path=f"{path}.approach_direction",
        ),
        "orientation_relation": _enum(
            payload["orientation_relation"],
            _ORIENTATION_RELATIONS | {"unknown"},
            path=f"{path}.orientation_relation",
        ),
        "grasp_region": _enum(
            payload["grasp_region"],
            _GRASP_REGIONS | {"unknown"},
            path=f"{path}.grasp_region",
        ),
        "semantic_part": _optional_text(
            payload["semantic_part"], path=f"{path}.semantic_part"
        ),
        "placement_relation": _optional_enum(
            payload["placement_relation"],
            _PLACEMENT_RELATIONS,
            path=f"{path}.placement_relation",
        ),
        "hard_constraints": _string_list(
            payload["hard_constraints"],
            path=f"{path}.hard_constraints",
            allowed=_HARD_CONSTRAINTS,
        ),
        "soft_preferences": _string_list(
            payload["soft_preferences"],
            path=f"{path}.soft_preferences",
            allowed=_SOFT_PREFERENCES,
        ),
        "avoid": _string_list(payload["avoid"], path=f"{path}.avoid", allowed=_AVOID),
        "geometry_source_class": _enum(
            payload["geometry_source_class"],
            _GEOMETRY_SOURCE_CLASSES,
            path=f"{path}.geometry_source_class",
        ),
    }
    if result["grasp_region"] == "semantic_part" and result["semantic_part"] is None:
        _fail(f"{path}.semantic_part", "is required for semantic_part grasp_region")
    _reject_private_geometry(result, path=path)
    return result


def _validate_effect_v2(value: Any, *, path: str) -> dict[str, Any]:
    payload = _canonical_mapping(value, path=path)
    _exact(payload, {"schema", "effect_type", "expected_predicates"}, path=path)
    if payload["schema"] not in {ABSTRACT_EFFECT_V2_SCHEMA, "tcm/afk/abstract_effect/v2"}:
        _fail(f"{path}.schema", f"must be {ABSTRACT_EFFECT_V2_SCHEMA}")
    result = {
        "schema": payload["schema"],
        "effect_type": _enum(
            payload["effect_type"], _EFFECT_TYPES, path=f"{path}.effect_type"
        ),
        "expected_predicates": _string_list(
            payload["expected_predicates"],
            path=f"{path}.expected_predicates",
            allowed=_EFFECT_PREDICATES,
        ),
    }
    _reject_private_geometry(result, path=path)
    return result


class _StrictRecord(Mapping[str, Any]):
    def __init__(self, value: Mapping[str, Any]) -> None:
        self._data = self._validate(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> _StrictRecord:
        return cls(value)

    def _validate(self, value: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def canonical_json(self) -> str:
        return canonical_json_bytes(self._data).decode("utf-8")


class StrategyDeltaV1(_StrictRecord):
    """At most two typed changes relative to one current strategy."""

    def _validate(self, value: Mapping[str, Any]) -> dict[str, Any]:
        payload = _canonical_mapping(value, path="StrategyDeltaV1")
        _exact(
            payload,
            {"schema", "scope", "task_delta", "geometry_delta"},
            path="StrategyDeltaV1",
        )
        if payload["schema"] not in {STRATEGY_DELTA_SCHEMA, "tcm/afk/strategy_delta/v1"}:
            _fail("StrategyDeltaV1.schema", f"must be {STRATEGY_DELTA_SCHEMA}")
        scope = _enum(
            payload["scope"],
            frozenset({"task", "geometry"}),
            path="StrategyDeltaV1.scope",
        )
        if scope == "geometry":
            if payload["task_delta"] is not None:
                _fail("StrategyDeltaV1.task_delta", "must be null for geometry scope")
            geometry = _canonical_mapping(
                payload["geometry_delta"], path="StrategyDeltaV1.geometry_delta"
            )
            _exact(
                geometry, _GEOMETRY_DELTA_FIELDS, path="StrategyDeltaV1.geometry_delta"
            )
            normalized_geometry = {
                "approach_family": _optional_enum(
                    geometry["approach_family"],
                    _APPROACH_FAMILIES,
                    path="geometry_delta.approach_family",
                ),
                "approach_direction": _optional_enum(
                    geometry["approach_direction"],
                    _APPROACH_DIRECTIONS,
                    path="geometry_delta.approach_direction",
                ),
                "orientation_relation": _optional_enum(
                    geometry["orientation_relation"],
                    _ORIENTATION_RELATIONS,
                    path="geometry_delta.orientation_relation",
                ),
                "grasp_region": _optional_enum(
                    geometry["grasp_region"],
                    _GRASP_REGIONS,
                    path="geometry_delta.grasp_region",
                ),
                "semantic_part": _optional_text(
                    geometry["semantic_part"], path="geometry_delta.semantic_part"
                ),
                "placement_relation": _optional_enum(
                    geometry["placement_relation"],
                    _PLACEMENT_RELATIONS,
                    path="geometry_delta.placement_relation",
                ),
                "add_hard_constraints": _string_list(
                    geometry["add_hard_constraints"],
                    path="geometry_delta.add_hard_constraints",
                    allowed=_HARD_CONSTRAINTS,
                ),
                "add_avoid": _string_list(
                    geometry["add_avoid"],
                    path="geometry_delta.add_avoid",
                    allowed=_AVOID,
                ),
            }
            task = None
            changed = sum(
                value is not None
                for key, value in normalized_geometry.items()
                if key not in {"add_hard_constraints", "add_avoid"}
            )
            changed += int(bool(normalized_geometry["add_hard_constraints"]))
            changed += int(bool(normalized_geometry["add_avoid"]))
        else:
            if payload["geometry_delta"] is not None:
                _fail("StrategyDeltaV1.geometry_delta", "must be null for task scope")
            task_payload = _canonical_mapping(
                payload["task_delta"], path="StrategyDeltaV1.task_delta"
            )
            _exact(task_payload, _TASK_DELTA_FIELDS, path="StrategyDeltaV1.task_delta")
            task = {
                "operation": _optional_enum(
                    task_payload["operation"], _OPERATIONS, path="task_delta.operation"
                ),
                "target_role": _optional_text(
                    task_payload["target_role"], path="task_delta.target_role"
                ),
                "target_relation": _optional_text(
                    task_payload["target_relation"], path="task_delta.target_relation"
                ),
                "preferred_arm": _optional_enum(
                    task_payload["preferred_arm"],
                    _ARMS,
                    path="task_delta.preferred_arm",
                ),
                "subgoal_purpose": _optional_text(
                    task_payload["subgoal_purpose"], path="task_delta.subgoal_purpose"
                ),
            }
            normalized_geometry = None
            changed = sum(value is not None for value in task.values())
        if changed < 1 or changed > 2:
            _fail("StrategyDeltaV1", "must change one or two typed fields")
        normalized = {
            "schema": payload["schema"],
            "scope": scope,
            "task_delta": task,
            "geometry_delta": normalized_geometry,
        }
        _reject_private_geometry(normalized, path="StrategyDeltaV1")
        return normalized


class ProposalEvidencePacketV1(_StrictRecord):
    """Compact verified evidence presented to the proposal model."""

    def _validate(self, value: Mapping[str, Any]) -> dict[str, Any]:
        payload = _canonical_mapping(value, path="ProposalEvidencePacketV1")
        _exact(
            payload,
            {
                "schema",
                "condition",
                "current_task_strategy",
                "current_geometric_strategy",
                "expected_effect",
                "observed_effect",
                "verdict",
                "recent_evidence",
                "available_capabilities",
                "retrieved_hpk",
                "optional_observations",
            },
            path="ProposalEvidencePacketV1",
        )
        if payload["schema"] not in {PROPOSAL_EVIDENCE_PACKET_SCHEMA, "tcm/afk/proposal_input/v1"}:
            _fail(
                "ProposalEvidencePacketV1.schema",
                f"must be {PROPOSAL_EVIDENCE_PACKET_SCHEMA}",
            )
        observed = _canonical_mapping(
            payload["observed_effect"], path="ProposalEvidencePacketV1.observed_effect"
        )
        _exact(
            observed, {"predicates"}, path="ProposalEvidencePacketV1.observed_effect"
        )
        recent_raw = payload["recent_evidence"]
        if not isinstance(recent_raw, list) or len(recent_raw) > 16:
            _fail(
                "ProposalEvidencePacketV1.recent_evidence",
                "must contain at most 16 items",
            )
        recent: list[dict[str, Any]] = []
        for index, item in enumerate(recent_raw):
            record = _canonical_mapping(item, path=f"recent_evidence[{index}]")
            _exact(
                record,
                {"strategy_summary", "verdict"},
                path=f"recent_evidence[{index}]",
            )
            summary = _canonical_mapping(
                record["strategy_summary"],
                path=f"recent_evidence[{index}].strategy_summary",
            )
            evidence_ref = _text(
                summary.get("evidence_ref"),
                path=f"recent_evidence[{index}].strategy_summary.evidence_ref",
                max_chars=240,
            )
            summary["evidence_ref"] = evidence_ref
            recent.append(
                {
                    "strategy_summary": summary,
                    "verdict": _enum(
                        record["verdict"],
                        frozenset({"support", "oppose"}),
                        path=f"recent_evidence[{index}].verdict",
                    ),
                }
            )
        capabilities = _canonical_mapping(
            payload["available_capabilities"],
            path="ProposalEvidencePacketV1.available_capabilities",
        )
        _exact(
            capabilities,
            {"task_fields", "geometry_fields"},
            path="ProposalEvidencePacketV1.available_capabilities",
        )
        normalized_capabilities = {
            "task_fields": _validate_capability_fields(
                capabilities["task_fields"],
                supported=_TASK_DELTA_FIELDS,
                path="available_capabilities.task_fields",
            ),
            "geometry_fields": _validate_capability_fields(
                capabilities["geometry_fields"],
                supported=_GEOMETRY_DELTA_FIELDS,
                path="available_capabilities.geometry_fields",
            ),
        }
        retrieved = payload["retrieved_hpk"]
        if not isinstance(retrieved, list) or len(retrieved) > 8:
            _fail(
                "ProposalEvidencePacketV1.retrieved_hpk",
                "must contain at most 8 entries",
            )
        normalized_retrieved = [
            _canonical_mapping(item, path=f"retrieved_hpk[{index}]")
            for index, item in enumerate(retrieved)
        ]
        observations = _canonical_mapping(
            payload["optional_observations"],
            path="ProposalEvidencePacketV1.optional_observations",
        )
        _exact(
            observations,
            {"before_image_ref", "after_image_ref"},
            path="ProposalEvidencePacketV1.optional_observations",
        )
        normalized = {
            "schema": payload["schema"],
            "condition": _validate_condition_v2(payload["condition"], path="condition"),
            "current_task_strategy": _validate_task_strategy_v2(
                payload["current_task_strategy"], path="current_task_strategy"
            ),
            "current_geometric_strategy": _validate_geometric_strategy_v2(
                payload["current_geometric_strategy"], path="current_geometric_strategy"
            ),
            "expected_effect": _validate_effect_v2(
                payload["expected_effect"], path="expected_effect"
            ),
            "observed_effect": {
                "predicates": _string_list(
                    observed["predicates"],
                    path="observed_effect.predicates",
                    allowed=_EFFECT_PREDICATES,
                )
            },
            "verdict": _enum(
                payload["verdict"],
                frozenset({"oppose", "support"}),
                path="verdict",
            ),
            "recent_evidence": recent,
            "available_capabilities": normalized_capabilities,
            "retrieved_hpk": normalized_retrieved,
            "optional_observations": {
                "before_image_ref": _optional_text(
                    observations["before_image_ref"],
                    path="optional_observations.before_image_ref",
                ),
                "after_image_ref": _optional_text(
                    observations["after_image_ref"],
                    path="optional_observations.after_image_ref",
                ),
            },
        }
        if (
            normalized["condition"]["operation"]
            != normalized["current_task_strategy"]["operation"]
        ):
            _fail("ProposalEvidencePacketV1", "condition and task operation differ")
        if (
            normalized["expected_effect"]["effect_type"]
            != normalized["condition"]["operation"]
        ):
            _fail(
                "ProposalEvidencePacketV1", "expected effect does not match operation"
            )
        _reject_private_geometry(normalized, path="ProposalEvidencePacketV1")
        return normalized

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return tuple(
            item["strategy_summary"]["evidence_ref"]
            for item in self._data["recent_evidence"]
        )

    @property
    def stable_id(self) -> str:
        return stable_content_id("afkproposalinput", self._data)


def _validate_capability_fields(
    value: Any,
    *,
    supported: set[str],
    path: str,
) -> dict[str, list[str]]:
    payload = _canonical_mapping(value, path=path)
    unknown = sorted(set(payload) - supported)
    if unknown:
        _fail(path, f"contains unsupported fields: {unknown}")
    result: dict[str, list[str]] = {}
    for key in sorted(payload):
        allowed = _capability_vocabulary(key)
        result[key] = _string_list(
            payload[key], path=f"{path}.{key}", allowed=allowed, max_items=64
        )
    return result


def _capability_vocabulary(field: str) -> frozenset[str] | None:
    return {
        "operation": _OPERATIONS,
        "preferred_arm": _ARMS,
        "approach_family": _APPROACH_FAMILIES,
        "approach_direction": _APPROACH_DIRECTIONS,
        "orientation_relation": _ORIENTATION_RELATIONS,
        "grasp_region": _GRASP_REGIONS,
        "placement_relation": _PLACEMENT_RELATIONS,
        "add_hard_constraints": _HARD_CONSTRAINTS,
        "add_avoid": _AVOID,
    }.get(field)


def proposal_id_for(
    *,
    condition: Mapping[str, Any],
    base_task_strategy: Mapping[str, Any],
    base_geometric_strategy: Mapping[str, Any],
    delta: Mapping[str, Any],
    expected_effect: Mapping[str, Any],
    proposer_version: str = VLM_GEOMETRY_PROPOSER_VERSION,
) -> str:
    return stable_content_id(
        "afkproposal",
        {
            "condition": _validate_condition_v2(condition, path="condition"),
            "base_task_strategy": _validate_task_strategy_v2(
                base_task_strategy, path="base_task_strategy"
            ),
            "base_geometric_strategy": _validate_geometric_strategy_v2(
                base_geometric_strategy, path="base_geometric_strategy"
            ),
            "delta": StrategyDeltaV1(delta).to_dict(),
            "expected_effect": _validate_effect_v2(
                expected_effect, path="expected_effect"
            ),
            "proposer_version": _text(
                proposer_version, path="proposer_version", max_chars=160
            ),
        },
    )


class StrategyProposalV1(_StrictRecord):
    """需要通过执行验证的 VLM 假设。"""

    def _validate(self, value: Mapping[str, Any]) -> dict[str, Any]:
        payload = _canonical_mapping(value, path="StrategyProposalV1")
        _exact(
            payload,
            {
                "schema",
                "proposal_id",
                "condition",
                "base_task_strategy",
                "base_geometric_strategy",
                "delta",
                "expected_effect",
                "evidence_refs",
                "rationale",
                "status",
                "resolution",
            },
            path="StrategyProposalV1",
        )
        if payload["schema"] not in {STRATEGY_PROPOSAL_SCHEMA, "tcm/afk/strategy_proposal/v1"}:
            _fail("StrategyProposalV1.schema", f"must be {STRATEGY_PROPOSAL_SCHEMA}")
        condition = _validate_condition_v2(payload["condition"], path="condition")
        task = _validate_task_strategy_v2(
            payload["base_task_strategy"], path="base_task_strategy"
        )
        geometry = _validate_geometric_strategy_v2(
            payload["base_geometric_strategy"], path="base_geometric_strategy"
        )
        delta = StrategyDeltaV1(payload["delta"])
        expected = _validate_effect_v2(
            payload["expected_effect"], path="expected_effect"
        )
        evidence_refs = _string_list(
            payload["evidence_refs"], path="evidence_refs", max_items=16
        )
        if not evidence_refs:
            _fail("evidence_refs", "must cite at least one compact evidence item")
        resolution = _canonical_mapping(payload["resolution"], path="resolution")
        _exact(
            resolution,
            {
                "verdict",
                "evidence_ref",
                "realized_task_strategy_id",
                "realized_geometric_strategy_id",
            },
            path="resolution",
        )
        normalized_resolution = {
            "verdict": _optional_enum(
                resolution["verdict"],
                frozenset({"support", "oppose", "unverified", "unrealizable"}),
                path="resolution.verdict",
            ),
            "evidence_ref": _optional_text(
                resolution["evidence_ref"], path="resolution.evidence_ref"
            ),
            "realized_task_strategy_id": _optional_text(
                resolution["realized_task_strategy_id"],
                path="resolution.realized_task_strategy_id",
            ),
            "realized_geometric_strategy_id": _optional_text(
                resolution["realized_geometric_strategy_id"],
                path="resolution.realized_geometric_strategy_id",
            ),
        }
        status = _enum(
            payload["status"],
            frozenset({"proposed", "scheduled", "executed", "resolved"}),
            path="status",
        )
        if status == "proposed" and any(
            item is not None for item in normalized_resolution.values()
        ):
            _fail("resolution", "must be empty while status is proposed")
        normalized = {
            "schema": payload["schema"],
            "proposal_id": _text(
                payload["proposal_id"], path="proposal_id", max_chars=160
            ),
            "condition": condition,
            "base_task_strategy": task,
            "base_geometric_strategy": geometry,
            "delta": delta.to_dict(),
            "expected_effect": expected,
            "evidence_refs": evidence_refs,
            "rationale": _text(payload["rationale"], path="rationale"),
            "status": status,
            "resolution": normalized_resolution,
        }
        expected_id = proposal_id_for(
            condition=condition,
            base_task_strategy=task,
            base_geometric_strategy=geometry,
            delta=delta,
            expected_effect=expected,
        )
        if normalized["proposal_id"] != expected_id:
            _fail("proposal_id", "does not match canonical proposal content")
        if (
            condition["operation"] != task["operation"]
            or expected["effect_type"] != task["operation"]
        ):
            _fail("StrategyProposalV1", "condition, task, and effect operations differ")
        _reject_private_geometry(normalized, path="StrategyProposalV1")
        return normalized


def validate_delta_capability_and_non_equivalence(
    packet: ProposalEvidencePacketV1,
    delta: StrategyDeltaV1,
) -> dict[str, Any]:
    """Perform exactly the P0 vocabulary and non-equivalence checks."""

    if delta["scope"] != "geometry":
        _fail("delta.scope", "offline P0 supports geometry proposals only")
    capabilities = packet["available_capabilities"]["geometry_fields"]
    geometry_delta = delta["geometry_delta"]
    for field, value in geometry_delta.items():
        values = value if isinstance(value, list) else [] if value is None else [value]
        if not values:
            continue
        allowed = capabilities.get(field)
        if not isinstance(allowed, list) or any(item not in allowed for item in values):
            _fail(
                f"delta.geometry_delta.{field}",
                "is outside current capability vocabulary",
            )
    return apply_geometry_delta(packet["current_geometric_strategy"], delta)


def apply_geometry_delta(
    base_geometric_strategy: Mapping[str, Any],
    delta: StrategyDeltaV1 | Mapping[str, Any],
) -> dict[str, Any]:
    base = _validate_geometric_strategy_v2(
        base_geometric_strategy, path="base_geometric_strategy"
    )
    typed_delta = (
        delta if isinstance(delta, StrategyDeltaV1) else StrategyDeltaV1(delta)
    )
    if typed_delta["scope"] != "geometry":
        _fail("delta.scope", "must be geometry")
    changes = typed_delta["geometry_delta"]
    result = copy.deepcopy(base)
    for field in (
        "approach_family",
        "approach_direction",
        "orientation_relation",
        "grasp_region",
        "semantic_part",
        "placement_relation",
    ):
        if changes[field] is not None:
            result[field] = changes[field]
    result["hard_constraints"] = list(
        dict.fromkeys([*result["hard_constraints"], *changes["add_hard_constraints"]])
    )
    result["avoid"] = list(dict.fromkeys([*result["avoid"], *changes["add_avoid"]]))
    normalized = _validate_geometric_strategy_v2(
        result, path="applied_geometric_strategy"
    )
    if normalized == base:
        _fail("delta", "is equivalent to the base geometric strategy")
    return normalized


def condition_v1_to_v2(
    value: ConditionV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Project the existing factual condition into the proposal vocabulary."""

    typed = value if isinstance(value, ConditionV1) else ConditionV1.from_dict(value)
    manipulated = typed["manipulated_object"]
    target = typed["target"]
    return _validate_condition_v2(
        {
            "schema": CONDITION_V2_SCHEMA,
            "task_family": typed["task_family"],
            "operation": typed["operation"],
            "manipulation_phase": typed["manipulation_phase"],
            "manipulated_role": manipulated["role"],
            "target_role": target["role"],
            "held_state": manipulated["held_state"],
            "scene_predicates": typed["scene_predicates"],
            "abstraction_version": "hpk_condition_builder/v2",
        },
        path="condition",
    )


def task_strategy_v1_to_v2(
    value: TaskStrategyV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Remove private source metadata while preserving task semantics."""

    typed = (
        value if isinstance(value, TaskStrategyV1) else TaskStrategyV1.from_dict(value)
    )
    return _validate_task_strategy_v2(
        {
            "schema": TASK_STRATEGY_V2_SCHEMA,
            "operation": typed["operation"],
            "manipulated_role": typed["manipulated_role"],
            "target_role": typed["target_role"],
            "target_relation": typed["target_relation"],
            "manipulation_phase": typed["manipulation_phase"],
            "subgoal_purpose": typed["subgoal_purpose"],
            "preferred_arm": typed["preferred_arm"],
        },
        path="task_strategy",
    )


def geometric_strategy_v1_to_v2(
    value: GeometricStrategyV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """将 HPK 几何策略转换为 VLM 提议输入。"""

    typed = (
        value
        if isinstance(value, GeometricStrategyV1)
        else GeometricStrategyV1.from_dict(value)
    )
    return _validate_geometric_strategy_v2(
        {
            "schema": GEOMETRIC_STRATEGY_V2_SCHEMA,
            "strategy_family": typed["strategy_family"],
            "reference_frame": typed["reference_frame"],
            "approach_family": typed["approach"]["family"],
            "approach_direction": typed["approach"]["direction_bucket"],
            "orientation_relation": typed["orientation"]["relation"],
            "grasp_region": typed["grasp"]["region"],
            "semantic_part": typed["grasp"]["semantic_part"],
            "placement_relation": typed["target_relation"]["relation"],
            "hard_constraints": typed["hard_constraints"],
            "soft_preferences": typed["soft_preferences"],
            "avoid": typed["avoid"],
            "geometry_source_class": typed["capability_evidence"][
                "geometry_source_class"
            ],
        },
        path="geometric_strategy",
    )


def abstract_effect_v1_to_v2(
    value: AbstractEffectV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Project only the expected physical effect, never a model verdict."""

    typed = (
        value
        if isinstance(value, AbstractEffectV1)
        else AbstractEffectV1.from_dict(value)
    )
    return _validate_effect_v2(
        {
            "schema": ABSTRACT_EFFECT_V2_SCHEMA,
            "effect_type": typed["effect_type"],
            "expected_predicates": typed["expected_predicates"],
        },
        path="expected_effect",
    )


def geometric_strategy_v2_to_v1(
    geometry: Mapping[str, Any],
    task_strategy: Mapping[str, Any],
) -> GeometricStrategyV1:
    """将 v2 几何描述转换为 HPK 排序器输入。"""

    value = _validate_geometric_strategy_v2(geometry, path="geometry")
    task = _validate_task_strategy_v2(task_strategy, path="task_strategy")
    return GeometricStrategyV1.from_dict(
        {
            "schema": GEOMETRIC_STRATEGY_SCHEMA,
            "strategy_family": value["strategy_family"],
            "reference_frame": value["reference_frame"],
            "target_relation": {
                "relation": value["placement_relation"],
                "reference_role": task["target_role"],
            },
            "approach": {
                "family": value["approach_family"],
                "direction_bucket": value["approach_direction"],
            },
            "orientation": {"relation": value["orientation_relation"]},
            "grasp": {
                "region": value["grasp_region"],
                "semantic_part": value["semantic_part"],
            },
            "hard_constraints": value["hard_constraints"],
            "soft_preferences": value["soft_preferences"],
            "avoid": value["avoid"],
            "capability_evidence": {
                "semantic_part_observed": value["semantic_part"] is not None,
                "geometry_source_class": value["geometry_source_class"],
            },
        }
    )


@dataclass(frozen=True, slots=True)
class ProposalRankChangeTrace:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


def rank_frozen_candidates_with_proposal(
    *,
    proposal: StrategyProposalV1 | Mapping[str, Any],
    frozen_candidates: Sequence[Mapping[str, Any]],
    scene_state: Mapping[str, Any] | None = None,
) -> ProposalRankChangeTrace:
    """Apply one proposal to an already legal, baseline-ranked candidate list."""

    typed = (
        proposal
        if isinstance(proposal, StrategyProposalV1)
        else StrategyProposalV1(proposal)
    )
    proposed_geometry = apply_geometry_delta(
        typed["base_geometric_strategy"], typed["delta"]
    )
    ranker_strategy = geometric_strategy_v2_to_v1(
        proposed_geometry, typed["base_task_strategy"]
    )
    baseline = tuple(copy.deepcopy(dict(candidate)) for candidate in frozen_candidates)
    ranking = rank_operation_pose_candidates(
        baseline,
        geometric_strategy=ranker_strategy,
        scene_state=scene_state,
    )
    selected = ranking.select()
    if selected is None:
        _fail("frozen_candidates", "proposal maps to no compliant candidate")

    def projection(candidate: Mapping[str, Any], rank: int) -> dict[str, Any]:
        features = candidate_semantic_features(scene_state, candidate)
        if isinstance(features, HPKUnresolved):
            _fail(
                "frozen_candidates",
                f"candidate feature projection failed: {features.reason}",
            )
        return {"rank": rank, "geometry": features.to_dict()}

    before = [projection(candidate, index) for index, candidate in enumerate(baseline)]
    after = [
        projection(candidate, index)
        for index, candidate in enumerate(ranking.ranked_candidates)
    ]
    selected_projection = projection(selected, 0)["geometry"]
    selected_id = str(selected.get("candidate_id", "") or "")
    selected_evaluation = next(
        (item for item in ranking.evaluations if item.candidate_id == selected_id),
        None,
    )
    compliant = bool(
        selected_evaluation is not None
        and selected_evaluation.geometric_compliance is True
    )
    payload = {
        "schema": PROPOSAL_RANK_TRACE_SCHEMA,
        "proposal_id": typed["proposal_id"],
        "rank_before": before,
        "rank_after": after,
        "selected_candidate_features": selected_projection,
        "rank_changed": before != after,
        "selection_changed": bool(
            before and before[0]["geometry"] != selected_projection
        ),
        "selected_candidate_satisfies_proposal": compliant,
        "motion_executed": False,
        "hpk_store_write_performed": False,
    }
    _reject_private_geometry(payload, path="ProposalRankChangeTrace")
    return ProposalRankChangeTrace(payload)


def strategy_proposal_output_json_schema() -> dict[str, Any]:
    def nullable(values: frozenset[str]) -> dict[str, Any]:
        return {
            "anyOf": [
                {"type": "string", "enum": sorted(values)},
                {"type": "null"},
            ]
        }

    geometry_delta = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_GEOMETRY_DELTA_FIELDS),
        "properties": {
            "approach_family": nullable(_APPROACH_FAMILIES),
            "approach_direction": nullable(_APPROACH_DIRECTIONS),
            "orientation_relation": nullable(_ORIENTATION_RELATIONS),
            "grasp_region": nullable(_GRASP_REGIONS),
            "semantic_part": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "placement_relation": nullable(_PLACEMENT_RELATIONS),
            "add_hard_constraints": {
                "type": "array",
                "maxItems": 2,
                "items": {"type": "string", "enum": sorted(_HARD_CONSTRAINTS)},
            },
            "add_avoid": {
                "type": "array",
                "maxItems": 1,
                "items": {"type": "string", "enum": sorted(_AVOID)},
            },
        },
    }
    delta = {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "scope", "task_delta", "geometry_delta"],
        "properties": {
            "schema": {"type": "string", "enum": [STRATEGY_DELTA_SCHEMA]},
            "scope": {"type": "string", "enum": ["geometry"]},
            "task_delta": {"type": "null"},
            "geometry_delta": geometry_delta,
        },
    }
    effect = {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "effect_type", "expected_predicates"],
        "properties": {
            "schema": {"type": "string", "enum": [ABSTRACT_EFFECT_V2_SCHEMA]},
            "effect_type": {"type": "string", "enum": sorted(_EFFECT_TYPES)},
            "expected_predicates": {
                "type": "array",
                "maxItems": len(_EFFECT_PREDICATES),
                "items": {"type": "string", "enum": sorted(_EFFECT_PREDICATES)},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["proposals"],
        "properties": {
            "proposals": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "delta",
                        "expected_effect",
                        "evidence_refs",
                        "rationale",
                    ],
                    "properties": {
                        "delta": delta,
                        "expected_effect": effect,
                        "evidence_refs": {
                            "type": "array",
                            "maxItems": 16,
                            "items": {"type": "string"},
                        },
                        "rationale": {"type": "string"},
                    },
                },
            }
        },
    }


__all__ = [
    "abstract_effect_v1_to_v2",
    "ABSTRACT_EFFECT_V2_SCHEMA",
    "CONDITION_V2_SCHEMA",
    "GEOMETRIC_STRATEGY_V2_SCHEMA",
    "PROPOSAL_EVIDENCE_PACKET_SCHEMA",
    "PROPOSAL_RANK_TRACE_SCHEMA",
    "STRATEGY_DELTA_SCHEMA",
    "STRATEGY_PROPOSAL_SCHEMA",
    "ProposalEvidencePacketV1",
    "ProposalRankChangeTrace",
    "StrategyDeltaV1",
    "StrategyProposalV1",
    "StrategyProposalValidationError",
    "apply_geometry_delta",
    "condition_v1_to_v2",
    "geometric_strategy_v1_to_v2",
    "geometric_strategy_v2_to_v1",
    "proposal_id_for",
    "rank_frozen_candidates_with_proposal",
    "task_strategy_v1_to_v2",
    "strategy_proposal_output_json_schema",
    "validate_delta_capability_and_non_equivalence",
]

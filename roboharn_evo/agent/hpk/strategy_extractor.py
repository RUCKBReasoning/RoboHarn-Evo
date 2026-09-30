from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from roboharn_evo.agent.hpk.schemas import (
    ABSTRACT_EFFECT_SCHEMA,
    GEOMETRIC_STRATEGY_SCHEMA,
    AbstractEffectV1,
    HPKUnresolved,
    CandidateGeometryFeaturesV1,
    ConditionV1,
    GeometricStrategyV1,
    TaskStrategyV1,
    stable_content_id,
)


EXTRACTION_REASON_ORDER = (
    "selected_hpk_already_present",
    "identity_mismatch",
    "unsupported_geometry_source",
    "oracle_geometry_forbidden",
    "target_relation_unresolved",
    "place_safety_unverified",
    "grasp_geometry_unresolved",
    "fixed_arm_not_transferable",
    "effect_identity_mismatch",
    "effect_predicates_unresolved",
)


@dataclass(frozen=True, slots=True)
class ExtractedHPKStrategy:
    """A transferable strategy candidate; it contains no runtime identity."""

    condition: ConditionV1
    task_strategy: TaskStrategyV1
    geometric_strategy: GeometricStrategyV1
    expected_effect: AbstractEffectV1
    strategy_key_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition.to_dict(),
            "task_strategy": self.task_strategy.to_dict(),
            "geometric_strategy": self.geometric_strategy.to_dict(),
            "expected_effect": self.expected_effect.expected_projection(),
            "strategy_key_id": self.strategy_key_id,
        }


def strategy_key_payload(
    *,
    condition: ConditionV1,
    task_strategy: TaskStrategyV1,
    geometric_strategy: GeometricStrategyV1,
    expected_effect: AbstractEffectV1,
) -> dict[str, Any]:
    """Return the exact transferable projection used to group entry updates."""

    return {
        "condition": condition.to_dict(),
        "task_strategy": task_strategy.transferable_dict(),
        "geometric_strategy": geometric_strategy.to_dict(),
        "expected_effect": expected_effect.expected_projection(),
    }


def strategy_key_id_for(
    *,
    condition: ConditionV1,
    task_strategy: TaskStrategyV1,
    geometric_strategy: GeometricStrategyV1,
    expected_effect: AbstractEffectV1,
) -> str:
    return stable_content_id(
        "afkstrategy",
        strategy_key_payload(
            condition=condition,
            task_strategy=task_strategy,
            geometric_strategy=geometric_strategy,
            expected_effect=expected_effect,
        ),
    )


def _typed_condition(value: ConditionV1 | Mapping[str, Any]) -> ConditionV1:
    return value if isinstance(value, ConditionV1) else ConditionV1.from_dict(value)


def _typed_task_strategy(
    value: TaskStrategyV1 | Mapping[str, Any],
) -> TaskStrategyV1:
    return (
        value if isinstance(value, TaskStrategyV1) else TaskStrategyV1.from_dict(value)
    )


def _typed_features(
    value: CandidateGeometryFeaturesV1 | Mapping[str, Any],
) -> CandidateGeometryFeaturesV1:
    return (
        value
        if isinstance(value, CandidateGeometryFeaturesV1)
        else CandidateGeometryFeaturesV1.from_dict(value)
    )


def _typed_effect(
    value: AbstractEffectV1 | Mapping[str, Any],
) -> AbstractEffectV1:
    return (
        value
        if isinstance(value, AbstractEffectV1)
        else AbstractEffectV1.from_dict(value)
    )


def _unresolved(reason: str, *missing_fields: str) -> HPKUnresolved:
    return HPKUnresolved(
        component="strategy_extractor",
        reason=reason,
        missing_fields=tuple(missing_fields),
    )


def _reference_frame(features: CandidateGeometryFeaturesV1) -> str:
    orientation = str(features["orientation_relation"])
    approach = str(features["approach_family"])
    operation = str(features["action_mode"])
    if orientation in {"align_principal_axis_0", "align_principal_axis_1"}:
        return "object_principal_axes"
    if orientation == "preserve_current_attachment":
        return "current_attachment"
    if approach == "surface_normal":
        return "support_normal"
    if operation == "place" and approach == "clearance_first":
        return "world_gravity"
    return "unknown"


def _strategy_family(operation: str) -> str:
    return {
        "grasp": "observed_grasp_geometry",
        "place": "placement_relation",
        "contact": "contact_relation",
    }[operation]


def expected_effect_for_operation(
    operation: str,
) -> AbstractEffectV1 | HPKUnresolved:
    """Return only operation-level effects that the current schema can verify.

    This is a generic HPK state-machine mapping, not a task rule.  Contact has
    no task-neutral predicate in the frozen v1 effect vocabulary, so it remains
    unresolved rather than inventing a benchmark-specific success signal.
    """

    normalized = str(operation or "").strip().lower()
    predicates = {
        "grasp": ["object_attached"],
        "place": [
            "gripper_empty",
            "object_released",
            "object_supported_by_target",
            "placement_stable",
            "target_relation_satisfied",
        ],
    }.get(normalized)
    if predicates is None:
        return _unresolved("effect_predicates_unresolved", "expected_predicates")
    return AbstractEffectV1.from_dict(
        {
            "schema": ABSTRACT_EFFECT_SCHEMA,
            "effect_type": normalized,
            "expected_predicates": sorted(predicates),
            "verifiability": "unverified",
        }
    )


def extract_baseline_strategy(
    *,
    condition: ConditionV1 | Mapping[str, Any],
    task_strategy: TaskStrategyV1 | Mapping[str, Any],
    candidate_features: CandidateGeometryFeaturesV1 | Mapping[str, Any],
    expected_effect: AbstractEffectV1 | Mapping[str, Any],
    selected_hpk_entry_id: str | None = None,
    allow_oracle_geometry: bool = False,
) -> ExtractedHPKStrategy | HPKUnresolved:
    """Extract a typed attempted strategy from a baseline physical action.

    A call with a selected HPK entry is not a baseline extraction; its existing
    typed strategy must be updated instead.  The returned strategy describes
    the attempted geometry only.  It never invents a corrective strategy from
    an observed failure.
    """

    if selected_hpk_entry_id is not None:
        return _unresolved("selected_hpk_already_present", "selected_hpk_entry_id")

    current_condition = _typed_condition(condition)
    current_task = _typed_task_strategy(task_strategy)
    features = _typed_features(candidate_features)
    effect = _typed_effect(expected_effect)

    operation = str(current_condition["operation"])
    task_operation = str(current_task["operation"])
    if operation != task_operation or operation != features["action_mode"]:
        return _unresolved("identity_mismatch", "operation")
    if current_condition["manipulation_phase"] != current_task["manipulation_phase"]:
        return _unresolved("identity_mismatch", "manipulation_phase")
    if (
        current_condition["manipulated_object"]["role"]
        != current_task["manipulated_role"]
    ):
        return _unresolved("identity_mismatch", "manipulated_role")
    if current_condition["target"]["role"] != current_task["target_role"]:
        return _unresolved("identity_mismatch", "target_role")
    if current_condition["target"]["relation"] != current_task["target_relation"]:
        return _unresolved("identity_mismatch", "target_relation")
    if current_task["preferred_arm"] != "either":
        return _unresolved("fixed_arm_not_transferable", "preferred_arm")

    geometry_source = str(features["geometry_source_class"])
    if geometry_source == "oracle" and not allow_oracle_geometry:
        return _unresolved("oracle_geometry_forbidden", "geometry_source_class")
    if geometry_source == "unknown":
        return _unresolved("unsupported_geometry_source", "geometry_source_class")

    target_relation = current_task["target_relation"]
    target_role = current_task["target_role"]
    if operation == "place":
        if target_relation is None or target_role is None:
            return _unresolved(
                "target_relation_unresolved", "target_relation", "target_role"
            )
        if (
            features["target_relation"] != target_relation
            or features["target_reference_role"] != target_role
        ):
            return _unresolved("identity_mismatch", "candidate_target_relation")
        if (
            features["support_valid"] is not True
            or features["target_region_free"] is not True
        ):
            return _unresolved(
                "place_safety_unverified", "support_valid", "target_region_free"
            )
    elif (
        target_relation is not None
        or target_role is not None
        or features["target_relation"] is not None
        or features["target_reference_role"] is not None
    ):
        return _unresolved("identity_mismatch", "unexpected_target_relation")

    if operation == "grasp" and (
        features["orientation_relation"] in {"unknown", "unconstrained"}
        or features["grasp_region"] == "unknown"
    ):
        return _unresolved(
            "grasp_geometry_unresolved", "orientation_relation", "grasp_region"
        )

    if effect["effect_type"] != operation:
        return _unresolved("effect_identity_mismatch", "effect_type")
    if not effect["expected_predicates"]:
        return _unresolved("effect_predicates_unresolved", "expected_predicates")

    grasp_region = str(features["grasp_region"]) if operation == "grasp" else "unknown"
    semantic_part = features["semantic_part"] if operation == "grasp" else None
    hard_constraints = (
        ["support_valid", "target_region_free"] if operation == "place" else []
    )
    payload = {
        "schema": GEOMETRIC_STRATEGY_SCHEMA,
        "strategy_family": _strategy_family(operation),
        "reference_frame": _reference_frame(features),
        "target_relation": {
            "relation": target_relation,
            "reference_role": target_role,
        },
        "approach": {
            "family": features["approach_family"],
            "direction_bucket": features["approach_direction_bucket"],
        },
        "orientation": {"relation": features["orientation_relation"]},
        "grasp": {"region": grasp_region, "semantic_part": semantic_part},
        "hard_constraints": hard_constraints,
        "soft_preferences": [],
        "avoid": [],
        "capability_evidence": {
            "semantic_part_observed": semantic_part is not None,
            "geometry_source_class": geometry_source,
        },
    }
    geometric_strategy = GeometricStrategyV1.from_dict(payload)
    expected_projection = AbstractEffectV1.from_dict(
        {
            "schema": ABSTRACT_EFFECT_SCHEMA,
            **{
                key: value
                for key, value in effect.expected_projection().items()
                if key != "schema"
            },
        }
    )
    return ExtractedHPKStrategy(
        condition=current_condition,
        task_strategy=current_task,
        geometric_strategy=geometric_strategy,
        expected_effect=expected_projection,
        strategy_key_id=strategy_key_id_for(
            condition=current_condition,
            task_strategy=current_task,
            geometric_strategy=geometric_strategy,
            expected_effect=expected_projection,
        ),
    )


__all__ = [
    "EXTRACTION_REASON_ORDER",
    "ExtractedHPKStrategy",
    "expected_effect_for_operation",
    "extract_baseline_strategy",
    "strategy_key_id_for",
    "strategy_key_payload",
]

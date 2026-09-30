"""Extract an abstract action effect from structured before/after evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from roboharn_evo.agent.hpk.schemas import (
    ABSTRACT_EFFECT_SCHEMA,
    EFFECT_PREDICATES,
    AbstractEffectV1,
)


_SOURCE_ORDER = (
    "scene_memory_delta",
    "robot_state",
    "attachment_state",
    "runtime_grasp_validation",
    "runtime_place_validation",
    "vlm_action_effect_verifier",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _status(value: Any) -> str:
    if value is True:
        return "verified"
    if value is False:
        return "contradicted"
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"true", "verified", "support", "supported", "satisfied"}:
        return "verified"
    if token in {"false", "contradicted", "oppose", "opposed", "violated"}:
        return "contradicted"
    return "unverified"


def _predicate_map(value: Mapping[str, Any]) -> dict[str, bool | None]:
    result: dict[str, bool | None] = {}
    raw = value.get("predicates")
    if isinstance(raw, Mapping):
        for key, item in raw.items():
            if key in EFFECT_PREDICATES and (isinstance(item, bool) or item is None):
                result[key] = item
    elif isinstance(raw, list):
        for item in raw:
            if item in EFFECT_PREDICATES:
                result[item] = True
    for key in EFFECT_PREDICATES:
        direct = value.get(key)
        if isinstance(direct, bool) or direct is None and key in value:
            result[key] = direct
    for list_key, truth in (
        ("observed_predicates", True),
        ("satisfied_predicates", True),
        ("contradicted_predicates", False),
        ("opposed_predicates", False),
    ):
        items = value.get(list_key)
        if isinstance(items, list):
            for item in items:
                if item in EFFECT_PREDICATES:
                    result[item] = truth
    return result


def _runtime_components(
    runtime_validation: Mapping[str, Any],
    *,
    effect_type: str,
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for key, source in (
        ("grasp_validation", "runtime_grasp_validation"),
        ("runtime_grasp_validation", "runtime_grasp_validation"),
        ("place_validation", "runtime_place_validation"),
        ("runtime_place_validation", "runtime_place_validation"),
        ("attachment_state", "attachment_state"),
        ("robot_state", "robot_state"),
        ("scene_memory_delta", "scene_memory_delta"),
    ):
        nested = _mapping(runtime_validation.get(key))
        if nested:
            result.append((source, nested))
    del effect_type
    return result


def _overall_status(value: Mapping[str, Any]) -> str:
    for key in (
        "effect_verified",
        "verifiability",
        "validation_status",
        "effect_status",
    ):
        if key in value:
            status = _status(value.get(key))
            if status != "unverified":
                return status
    # Runtime validators use ``verified=False`` for both explicit failure and
    # incomplete evidence.  Only True is an overall proof; concrete negative
    # predicates below decide contradiction.
    if value.get("verified") is True:
        return "verified"
    return "unverified"


def _runtime_predicate_map(
    source: str,
    value: Mapping[str, Any],
) -> dict[str, bool | None]:
    result = _predicate_map(value)
    if source == "runtime_grasp_validation":
        if value.get("verified") is True or value.get("attachment_verified") is True:
            result["object_attached"] = True
        elif value.get("verified") is False and (
            value.get("negative_attachment_evidence") is True
            or value.get("fresh_visible_negative_evidence") is True
        ):
            result["object_attached"] = False
    elif source == "runtime_place_validation":
        released = bool(
            value.get("release_executed") is True
            and value.get("detachment_verified") is True
        )
        if released:
            result["object_released"] = True
            if value.get("gripper_open") is True:
                # Open alone is not an empty-gripper proof.  Verified physical
                # detachment plus the executed release is.
                result["gripper_empty"] = True
        fresh_target_observation = bool(
            value.get("object_position_observed") is True
            and value.get("object_position_fresh") is True
        )
        if fresh_target_observation and isinstance(value.get("object_at_target"), bool):
            at_target = bool(value["object_at_target"])
            result["target_relation_satisfied"] = at_target
            result["target_region_occupied_by_manipulated_object"] = at_target
        if value.get("stable_across_fresh_observations") is True:
            result["placement_stable"] = True
        if value.get("object_supported_by_target") is True and (
            value.get("support_valid") is True
            or value.get("support_validation_verified") is True
        ):
            result["object_supported_by_target"] = True
        elif value.get("object_supported_by_target") is False and (
            value.get("support_validation_verified") is True
        ):
            result["object_supported_by_target"] = False
    return result


def _merge_deterministic(
    components: list[tuple[str, Mapping[str, Any]]],
) -> tuple[dict[str, bool | None], str, set[str], bool]:
    predicates: dict[str, bool | None] = {}
    overall_values: list[str] = []
    sources: set[str] = set()
    conflict = False
    for source, component in components:
        current = _runtime_predicate_map(source, component)
        status = _overall_status(component)
        # A populated deterministic validator is provenance even when its
        # conservative outcome is unverified and it yields no positive fact.
        if component:
            sources.add(source)
        if status != "unverified":
            overall_values.append(status)
        for key, value in current.items():
            previous = predicates.get(key)
            if previous is not None and value is not None and previous != value:
                conflict = True
                predicates[key] = None
            elif key not in predicates or previous is None:
                predicates[key] = value
    distinct = set(overall_values)
    if len(distinct) > 1:
        conflict = True
        overall = "unverified"
    elif overall_values:
        overall = overall_values[0]
    else:
        overall = "unverified"
    return predicates, overall, sources, conflict


def _state_delta(
    pre_effect_state: Mapping[str, Any],
    post_effect_state: Mapping[str, Any],
) -> tuple[dict[str, bool | None], set[str]]:
    pre = _predicate_map(pre_effect_state)
    post = _predicate_map(post_effect_state)
    result: dict[str, bool | None] = {}
    for key, value in post.items():
        if value is True and pre.get(key) is not True:
            result[key] = True
        elif value is False:
            result[key] = False
    sources: set[str] = set()
    if result:
        sources.add("scene_memory_delta")
    return result, sources


def _expected_payload(
    expected_effect: AbstractEffectV1 | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(expected_effect, AbstractEffectV1):
        return expected_effect.to_dict()
    return AbstractEffectV1.from_dict(expected_effect).to_dict()


def _ordered_sources(values: set[str]) -> list[str]:
    return [source for source in _SOURCE_ORDER if source in values]


class AbstractEffectExtractor:
    """Conservative extractor with deterministic Runtime evidence priority."""

    def extract(
        self,
        expected_effect: AbstractEffectV1 | Mapping[str, Any],
        *,
        pre_effect_state: Mapping[str, Any] | None = None,
        post_effect_state: Mapping[str, Any] | None = None,
        runtime_validation: Mapping[str, Any] | None = None,
        verifier_result: Mapping[str, Any] | None = None,
        motion_status: str = "unknown",
    ) -> AbstractEffectV1:
        expected = _expected_payload(expected_effect)
        expected_predicates = list(expected["expected_predicates"])
        pre = _mapping(pre_effect_state)
        post = _mapping(post_effect_state)
        runtime = _mapping(runtime_validation)
        verifier = _mapping(verifier_result)

        delta_map, delta_sources = _state_delta(pre, post)
        deterministic_map, runtime_status, runtime_sources, runtime_conflict = (
            _merge_deterministic(
                [
                    ("scene_memory_delta", delta_map),
                    *_runtime_components(
                        runtime,
                        effect_type=str(expected["effect_type"]),
                    ),
                ]
            )
        )
        sources = set(delta_sources) | runtime_sources

        verifier_map = _predicate_map(verifier)
        verifier_status = _overall_status(verifier)
        if verifier:
            sources.add("vlm_action_effect_verifier")
        cross_conflict = runtime_conflict
        for key, runtime_value in deterministic_map.items():
            verifier_value = verifier_map.get(key)
            if (
                runtime_value is not None
                and verifier_value is not None
                and runtime_value != verifier_value
            ):
                cross_conflict = True
        if (
            runtime_status != "unverified"
            and verifier_status != "unverified"
            and runtime_status != verifier_status
        ):
            cross_conflict = True

        combined = dict(verifier_map)
        # Deterministic sources are authoritative whenever they have a value.
        combined.update(
            {
                key: value
                for key, value in deterministic_map.items()
                if value is not None
            }
        )
        observed = sorted(key for key, value in combined.items() if value is True)
        completed = str(motion_status or "").strip().lower() == "completed"

        if not completed or cross_conflict:
            verifiability = "unverified"
        elif any(combined.get(key) is False for key in expected_predicates):
            verifiability = "contradicted"
        elif expected_predicates and all(
            combined.get(key) is True for key in expected_predicates
        ):
            verifiability = "verified"
        elif runtime_status == "contradicted":
            # Only an explicit deterministic validator may oppose without a
            # predicate-level projection.  A model-only overall negative is a
            # claim, not physical evidence.
            verifiability = "contradicted"
        else:
            verifiability = "unverified"

        return AbstractEffectV1.from_dict(
            {
                "schema": ABSTRACT_EFFECT_SCHEMA,
                "effect_type": expected["effect_type"],
                "expected_predicates": expected_predicates,
                "observed_predicates": observed,
                "verifiability": verifiability,
                "verifier_sources": _ordered_sources(sources),
            }
        )


def extract_abstract_effect(
    expected_effect: AbstractEffectV1 | Mapping[str, Any],
    **kwargs: Any,
) -> AbstractEffectV1:
    return AbstractEffectExtractor().extract(expected_effect, **kwargs)


__all__ = ["AbstractEffectExtractor", "extract_abstract_effect"]

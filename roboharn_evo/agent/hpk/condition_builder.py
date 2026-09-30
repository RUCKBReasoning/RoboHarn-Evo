from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from roboharn_evo.agent.hpk.schemas import (
    CONDITION_ABSTRACTION_VERSION,
    CONDITION_SCHEMA,
    HPKUnresolved,
    ConditionV1,
    TaskStrategyV1,
    contains_private_transfer_text,
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _instances(scene_memory: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = scene_memory.get("instances")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _instance_private_id(instance: Mapping[str, Any]) -> str:
    return _first_text(instance.get("instance_id"), instance.get("track_id"))


def _instance_role(instance: Mapping[str, Any]) -> str:
    return _first_text(
        instance.get("semantic_role"),
        instance.get("task_role"),
        instance.get("role"),
    )


def _instance_for_role(
    scene_memory: Mapping[str, Any],
    *,
    role: str,
    private_ref: str = "",
) -> dict[str, Any]:
    exact_ref_matches: list[dict[str, Any]] = []
    role_matches: list[dict[str, Any]] = []
    for instance in _instances(scene_memory):
        if private_ref and _instance_private_id(instance) == private_ref:
            exact_ref_matches.append(instance)
        if role and _instance_role(instance) == role:
            role_matches.append(instance)
    if len(exact_ref_matches) == 1:
        return exact_ref_matches[0]
    if len(role_matches) == 1:
        return role_matches[0]
    return {}


def _semantic_class(value: Mapping[str, Any]) -> str:
    return _first_text(
        value.get("semantic_class"),
        value.get("object_class"),
        value.get("class_name"),
        value.get("category"),
    )


def _geometry_class(value: Mapping[str, Any]) -> str:
    # Geometry classes are accepted only when a prior component explicitly
    # authored one.  Dimensions/coordinates are never converted into semantic
    # classes here because that would silently invent an abstraction policy.
    return _first_text(value.get("geometry_class"), value.get("shape_class"))


def _held_private_refs(runtime_state: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    manipulation = _mapping(runtime_state.get("manipulation_state"))
    for arm in ("left", "right"):
        arm_state = _mapping(manipulation.get(arm))
        ref = _text(arm_state.get("held_instance_id"))
        if ref:
            refs.add(ref)
    direct = _text(runtime_state.get("held_instance_id"))
    if direct:
        refs.add(direct)
    return refs


def _held_state(
    descriptor: Mapping[str, Any],
    *,
    instance: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> str:
    explicit = _text(descriptor.get("held_state"))
    if explicit in {"held", "not_held", "unknown"}:
        return explicit
    private_ref = _instance_private_id(instance)
    held_refs = _held_private_refs(runtime_state)
    if private_ref and private_ref in held_refs:
        return "held"
    if private_ref and held_refs:
        return "not_held"
    return "unknown"


def _structured_descriptor(
    *sources: Mapping[str, Any],
) -> dict[str, Any]:
    for source in sources:
        if source:
            return dict(source)
    return {}


def _target_descriptor(
    scene_memory: Mapping[str, Any],
    *,
    runtime_state: Mapping[str, Any],
    bound_operation_target: Mapping[str, Any],
    target_role: str,
) -> dict[str, Any]:
    nested = _structured_descriptor(
        _mapping(bound_operation_target.get("target")),
        _mapping(runtime_state.get("target")),
        _mapping(scene_memory.get("target")),
    )
    if nested:
        return nested

    private_ref = _first_text(
        bound_operation_target.get("target_instance_id"),
        bound_operation_target.get("reference_object_id"),
        bound_operation_target.get("support_instance_id"),
    )
    instance = _instance_for_role(
        scene_memory,
        role=target_role,
        private_ref=private_ref,
    )
    if instance:
        return instance

    raw_targets = scene_memory.get("operation_targets")
    if isinstance(raw_targets, list):
        matches = [
            dict(item)
            for item in raw_targets
            if isinstance(item, Mapping)
            and _first_text(
                item.get("target_role"),
                item.get("reference_role"),
                item.get("role"),
            )
            == target_role
        ]
        if len(matches) == 1:
            return matches[0]
    return {}


def _scene_predicates(
    *sources: Mapping[str, Any],
) -> list[str]:
    result: set[str] = set()
    for source in sources:
        if source.get("support_valid") is True:
            result.add("support_valid")
        if source.get("target_region_free") is True or source.get("free") is True:
            result.add("target_region_free")
        explicit = source.get("scene_predicates")
        if isinstance(explicit, list):
            result.update(
                value
                for value in explicit
                if value in {"support_valid", "target_region_free"}
            )
    return sorted(result)


def _preconditions(runtime_state: Mapping[str, Any]) -> tuple[list[str], bool]:
    raw = runtime_state.get("preconditions")
    values = raw if isinstance(raw, list) else []
    texts = [_text(value) for value in values if _text(value)]
    private_input = any(contains_private_transfer_text(value) for value in texts)
    result = {value for value in texts if not contains_private_transfer_text(value)}
    if len(_held_private_refs(runtime_state)) == 1:
        result.add("exactly_one_object_held")
    return sorted(result), private_input


def _unique_structured(values: list[str]) -> tuple[str, bool]:
    nonempty = list(dict.fromkeys(value for value in values if value))
    if len(nonempty) == 1:
        return nonempty[0], False
    return "", len(nonempty) > 1


def _runtime_condition_fields(
    scene: Mapping[str, Any],
    runtime: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> tuple[dict[str, str], tuple[str, ...]]:
    focus = _mapping(scene.get("task_focus"))
    manipulation = _mapping(runtime.get("manipulation_state"))
    bound_arm = _text(bound.get("arm") or bound.get("preferred_arm")).lower()
    phase_arms = (bound_arm,) if bound_arm in {"left", "right"} else ("left", "right")
    manipulation_phases = [
        _text(_mapping(manipulation.get(arm)).get("phase"))
        for arm in phase_arms
        if _mapping(manipulation.get(arm))
    ]
    operation, operation_conflict = _unique_structured(
        [
            _text(bound.get("operation")),
            _text(bound.get("action_mode")),
            _text(runtime.get("operation")),
            _text(runtime.get("operation_action_mode")),
            _text(scene.get("operation")),
        ]
    )
    operation = operation.lower().replace("-", "_").replace(" ", "_")
    phase, phase_conflict = _unique_structured(
        [
            _text(bound.get("manipulation_phase")),
            _text(runtime.get("manipulation_phase")),
            _text(runtime.get("phase")),
            _text(manipulation.get("phase")),
            _text(focus.get("manipulation_phase")),
            *manipulation_phases,
        ]
    )
    manipulated_role, manipulated_conflict = _unique_structured(
        [
            _text(bound.get("manipulated_role")),
            _text(bound.get("role")) if operation != "place" else "",
            _text(runtime.get("manipulated_role")),
            _text(runtime.get("held_role")),
            _text(focus.get("manipulated_role")),
        ]
    )
    target_role, target_conflict = _unique_structured(
        [
            _text(bound.get("target_role")),
            _text(bound.get("reference_role")),
            _text(bound.get("role")) if operation == "place" else "",
            _text(runtime.get("target_role")),
            _text(focus.get("target_role")),
        ]
    )
    relation_values: list[str] = []
    for raw in (
        bound.get("target_relation"),
        bound.get("placement_relation"),
        runtime.get("target_relation"),
        focus.get("target_relation"),
    ):
        if isinstance(raw, Mapping):
            raw = raw.get("relation")
        value = _text(raw).lower().replace("-", "_").replace(" ", "_")
        if value:
            relation_values.append(value)
    relation, relation_conflict = _unique_structured(relation_values)
    conflicts = tuple(
        name
        for name, conflict in (
            ("operation", operation_conflict),
            ("manipulation_phase", phase_conflict),
            ("manipulated_role", manipulated_conflict),
            ("target_role", target_conflict),
            ("target_relation", relation_conflict),
        )
        if conflict
    )
    return {
        "operation": operation,
        "manipulation_phase": phase,
        "manipulated_role": manipulated_role,
        "target_role": target_role,
        "target_relation": relation,
    }, conflicts


class HPKConditionBuilder:
    """Deterministic Scene Memory/runtime projection into ``ConditionV1``."""

    def build(
        self,
        scene_memory: Mapping[str, Any],
        task_strategy: TaskStrategyV1 | Mapping[str, Any] | None = None,
        runtime_state: Mapping[str, Any] | None = None,
        bound_operation_target: Mapping[str, Any] | None = None,
        *,
        task_family: str | None = None,
    ) -> ConditionV1 | HPKUnresolved:
        scene = _mapping(scene_memory)
        runtime = _mapping(runtime_state)
        bound = _mapping(bound_operation_target)
        strategy: TaskStrategyV1 | None = None
        if task_strategy is not None:
            try:
                strategy = (
                    task_strategy
                    if isinstance(task_strategy, TaskStrategyV1)
                    else TaskStrategyV1.from_dict(task_strategy)
                )
            except (TypeError, ValueError) as exc:
                return HPKUnresolved(
                    component="condition_builder",
                    reason=f"invalid_task_strategy:{type(exc).__name__}",
                    missing_fields=("task_strategy",),
                )

        authoritative, conflicts = _runtime_condition_fields(scene, runtime, bound)
        if conflicts:
            return HPKUnresolved(
                component="condition_builder",
                reason="conflicting_runtime_condition_sources",
                missing_fields=conflicts,
            )
        missing_authoritative = [
            key
            for key in ("operation", "manipulation_phase", "manipulated_role")
            if not authoritative[key]
        ]
        if authoritative["operation"] == "place":
            for key in ("target_role", "target_relation"):
                if not authoritative[key]:
                    missing_authoritative.append(key)
        if missing_authoritative:
            return HPKUnresolved(
                component="condition_builder",
                reason="runtime_condition_unresolved",
                missing_fields=tuple(missing_authoritative),
            )
        if strategy is not None:
            strategy_data = strategy.to_dict()
            mismatch = [
                key
                for key in (
                    "operation",
                    "manipulation_phase",
                    "manipulated_role",
                    "target_role",
                    "target_relation",
                )
                if (strategy_data.get(key) or "") != (authoritative.get(key) or "")
            ]
            if mismatch:
                return HPKUnresolved(
                    component="condition_builder",
                    reason="task_strategy_runtime_mismatch",
                    missing_fields=tuple(mismatch),
                )

        family = _first_text(
            task_family,
            scene.get("task_family"),
            _mapping(scene.get("task_focus")).get("task_family"),
            runtime.get("task_family"),
        )
        manipulated_role = authoritative["manipulated_role"]
        target_role = authoritative["target_role"]

        manipulated_descriptor = _structured_descriptor(
            _mapping(bound.get("manipulated_object")),
            _mapping(runtime.get("manipulated_object")),
            _mapping(scene.get("manipulated_object")),
        )
        manipulated_private_ref = _first_text(
            bound.get("manipulated_instance_id"),
            runtime.get("manipulated_instance_id"),
            manipulated_descriptor.get("instance_id"),
            manipulated_descriptor.get("track_id"),
        )
        manipulated_instance = _instance_for_role(
            scene,
            role=manipulated_role,
            private_ref=manipulated_private_ref,
        )
        manipulated = manipulated_descriptor or manipulated_instance
        semantic_class = _semantic_class(manipulated)
        geometry_class = _geometry_class(manipulated)

        missing: list[str] = []
        if not family:
            missing.append("task_family")
        if not semantic_class:
            missing.append("manipulated_object.semantic_class")
        if not geometry_class:
            missing.append("manipulated_object.geometry_class")
        if missing:
            return HPKUnresolved(
                component="condition_builder",
                reason="required_structured_state_missing",
                missing_fields=tuple(missing),
            )

        target_descriptor = _target_descriptor(
            scene,
            runtime_state=runtime,
            bound_operation_target=bound,
            target_role=target_role,
        )
        target_semantic = _semantic_class(target_descriptor) or None
        target_geometry = _geometry_class(target_descriptor) or None
        if authoritative["operation"] == "place" and not target_role:
            return HPKUnresolved(
                component="condition_builder",
                reason="place_target_unresolved",
                missing_fields=("target.role",),
            )

        preconditions, private_precondition = _preconditions(runtime)
        if private_precondition:
            return HPKUnresolved(
                component="condition_builder",
                reason="precondition_contains_private_runtime_data",
                missing_fields=("preconditions",),
            )
        payload = {
            "schema": CONDITION_SCHEMA,
            "task_family": family,
            "manipulation_phase": authoritative["manipulation_phase"],
            "operation": authoritative["operation"],
            "manipulated_object": {
                "semantic_class": semantic_class,
                "geometry_class": geometry_class,
                "role": manipulated_role,
                "held_state": _held_state(
                    manipulated,
                    instance=manipulated_instance,
                    runtime_state=runtime,
                ),
            },
            "target": {
                "semantic_class": target_semantic,
                "geometry_class": target_geometry,
                "role": target_role or None,
                "relation": authoritative["target_relation"] or None,
            },
            "scene_predicates": _scene_predicates(
                scene,
                runtime,
                bound,
                target_descriptor,
            ),
            "preconditions": preconditions,
            "abstraction_version": CONDITION_ABSTRACTION_VERSION,
        }
        try:
            return ConditionV1.from_dict(payload)
        except ValueError as exc:
            return HPKUnresolved(
                component="condition_builder",
                reason=f"condition_validation_failed:{type(exc).__name__}",
            )


def build_condition(
    scene_memory: Mapping[str, Any],
    task_strategy: TaskStrategyV1 | Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, Any] | None = None,
    bound_operation_target: Mapping[str, Any] | None = None,
    *,
    task_family: str | None = None,
) -> ConditionV1 | HPKUnresolved:
    return HPKConditionBuilder().build(
        scene_memory,
        task_strategy,
        runtime_state,
        bound_operation_target,
        task_family=task_family,
    )


ConditionBuilder = HPKConditionBuilder


__all__ = ["HPKConditionBuilder", "ConditionBuilder", "build_condition"]

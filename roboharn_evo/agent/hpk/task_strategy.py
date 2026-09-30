from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from roboharn_evo.agent.hpk.schemas import (
    TASK_STRATEGY_NORMALIZATION_VERSION,
    TASK_STRATEGY_SCHEMA,
    HPKUnresolved,
    OPERATIONS,
    TaskStrategyV1,
    contains_private_transfer_text,
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _normalized_token(value: Any) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _safe_text(value: Any) -> str:
    text = _text(value)
    return "" if contains_private_transfer_text(text) else text


def _unique(values: list[str]) -> tuple[str, bool]:
    nonempty = list(dict.fromkeys(value for value in values if value))
    if len(nonempty) == 1:
        return nonempty[0], False
    return "", len(nonempty) > 1


def _explicit_operation(
    planner: Mapping[str, Any],
    bound: Mapping[str, Any],
    active: Mapping[str, Any],
) -> tuple[str, bool]:
    tool_call = _mapping(planner.get("tool_call"))
    tool_args = _mapping(tool_call.get("args"))
    values = [
        _normalized_token(planner.get("operation")),
        _normalized_token(planner.get("operation_action_mode")),
        _normalized_token(planner.get("manipulation_operation")),
        _normalized_token(tool_args.get("operation")),
        _normalized_token(tool_args.get("action_mode")),
        _normalized_token(bound.get("operation")),
        _normalized_token(bound.get("action_mode")),
        _normalized_token(active.get("operation")),
        _normalized_token(active.get("action_mode")),
    ]
    # Planner ``action_mode`` is normally start/switch/continue.  It is used
    # only when it already is one of the frozen physical operation enums.
    planner_action = _normalized_token(planner.get("action_mode"))
    if planner_action in OPERATIONS:
        values.append(planner_action)
    values = [value for value in values if value in OPERATIONS]
    return _unique(values)


def _scene_instance_role(scene: Mapping[str, Any], private_ref: str) -> str:
    if not private_ref:
        return ""
    raw = scene.get("instances")
    if not isinstance(raw, list):
        return ""
    matches: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        item_ref = _text(item.get("instance_id")) or _text(item.get("track_id"))
        if item_ref != private_ref:
            continue
        role = _safe_text(
            item.get("semantic_role") or item.get("task_role") or item.get("role")
        )
        if role:
            matches.append(role)
    role, conflict = _unique(matches)
    return "" if conflict else role


def _manipulated_role(
    *,
    operation: str,
    planner: Mapping[str, Any],
    scene: Mapping[str, Any],
    bound: Mapping[str, Any],
    active: Mapping[str, Any],
) -> tuple[str, bool]:
    focus = _mapping(scene.get("task_focus"))
    values = [
        _safe_text(planner.get("manipulated_role")),
        _safe_text(bound.get("manipulated_role")),
        _safe_text(active.get("manipulated_role")),
        _safe_text(focus.get("manipulated_role")),
    ]
    if operation != "place":
        values.append(_safe_text(bound.get("role")))
    private_ref = _text(
        bound.get("manipulated_instance_id")
        or bound.get("held_instance_id")
        or active.get("manipulated_instance_id")
    )
    mapped_role = _scene_instance_role(scene, private_ref)
    if mapped_role:
        values.append(mapped_role)
    return _unique(values)


def _target_role(
    *,
    planner: Mapping[str, Any],
    scene: Mapping[str, Any],
    bound: Mapping[str, Any],
    active: Mapping[str, Any],
) -> tuple[str, bool]:
    focus = _mapping(scene.get("task_focus"))
    values = [
        _safe_text(planner.get("target_role")),
        _safe_text(bound.get("target_role")),
        _safe_text(bound.get("reference_role")),
        _safe_text(bound.get("role")),
        _safe_text(active.get("target_role")),
        _safe_text(focus.get("target_role")),
    ]
    private_ref = _text(
        bound.get("target_instance_id")
        or bound.get("reference_object_id")
        or bound.get("support_instance_id")
    )
    mapped_role = _scene_instance_role(scene, private_ref)
    if mapped_role:
        values.append(mapped_role)
    return _unique(values)


def _target_relation(
    planner: Mapping[str, Any],
    bound: Mapping[str, Any],
    active: Mapping[str, Any],
) -> tuple[str, bool]:
    raw_values = [
        planner.get("target_relation"),
        bound.get("target_relation"),
        bound.get("placement_relation"),
        active.get("target_relation"),
    ]
    values: list[str] = []
    for raw in raw_values:
        if isinstance(raw, Mapping):
            raw = raw.get("relation")
        token = _normalized_token(raw)
        if token:
            values.append(token)
    value, conflict = _unique(values)
    if conflict:
        return "", True
    if value and value != "center_of":
        return "", True
    return value, False


def _phase(
    planner: Mapping[str, Any],
    scene: Mapping[str, Any],
    active: Mapping[str, Any],
) -> tuple[str, bool]:
    tags = _mapping(planner.get("semantic_tags"))
    focus = _mapping(scene.get("task_focus"))
    values = [
        _safe_text(planner.get("manipulation_phase")),
        _safe_text(tags.get("manipulation_phase")),
        _safe_text(tags.get("phase")),
        _safe_text(active.get("manipulation_phase")),
        _safe_text(active.get("phase")),
        _safe_text(focus.get("manipulation_phase")),
    ]
    return _unique(values)


def _preferred_arm(
    planner: Mapping[str, Any],
    bound: Mapping[str, Any],
    active: Mapping[str, Any],
) -> tuple[str, bool]:
    preferences = [
        _normalized_token(planner.get("preferred_arm")),
        _normalized_token(bound.get("preferred_arm")),
        _normalized_token(active.get("preferred_arm")),
    ]
    preferences = [
        value for value in preferences if value in {"left", "right", "either"}
    ]
    preference, preference_conflict = _unique(preferences)
    execution_arms = [
        _normalized_token(bound.get("arm")),
        _normalized_token(active.get("arm")),
    ]
    execution_arms = [value for value in execution_arms if value in {"left", "right"}]
    execution_arm, execution_conflict = _unique(execution_arms)
    if preference_conflict or execution_conflict:
        return "", True
    if preference:
        if (
            preference in {"left", "right"}
            and execution_arm
            and preference != execution_arm
        ):
            return "", True
        return preference, False
    return execution_arm, False


class TaskStrategyNormalizer:
    """Normalize structured current-state fields without keyword inference."""

    def normalize(
        self,
        planner_prediction: Mapping[str, Any],
        current_scene_memory: Mapping[str, Any] | None = None,
        bound_operation_target: Mapping[str, Any] | None = None,
        active_skill: Mapping[str, Any] | str | None = None,
    ) -> TaskStrategyV1 | HPKUnresolved:
        planner = _mapping(planner_prediction)
        scene = _mapping(current_scene_memory)
        bound = _mapping(bound_operation_target)
        active = _mapping(active_skill)
        if isinstance(active_skill, str):
            active = {"selected_skill": active_skill}

        subtask_text = _text(
            planner.get("subtask_text")
            or planner.get("planner_subtask_text")
            or active.get("subtask_text")
            or active.get("instruction")
        )
        if subtask_text and contains_private_transfer_text(subtask_text):
            return HPKUnresolved(
                component="task_strategy_normalizer",
                reason="planner_subtask_contains_private_runtime_data",
                missing_fields=("source.planner_subtask_text",),
            )

        selected_skill = _safe_text(
            planner.get("selected_skill")
            or active.get("selected_skill")
            or active.get("skill_name")
            or active.get("name")
        )
        operation, operation_conflict = _explicit_operation(
            planner,
            bound,
            active,
        )
        manipulated_role, manipulated_conflict = _manipulated_role(
            operation=operation,
            planner=planner,
            scene=scene,
            bound=bound,
            active=active,
        )
        target_role, target_conflict = _target_role(
            planner=planner,
            scene=scene,
            bound=bound,
            active=active,
        )
        target_relation, relation_conflict = _target_relation(
            planner,
            bound,
            active,
        )
        phase, phase_conflict = _phase(planner, scene, active)
        arm, arm_conflict = _preferred_arm(planner, bound, active)
        purpose = _safe_text(
            planner.get("subgoal_purpose")
            or active.get("subgoal_purpose")
            or subtask_text
        )

        conflicts = [
            name
            for name, conflict in (
                ("operation", operation_conflict),
                ("manipulated_role", manipulated_conflict),
                ("target_role", target_conflict),
                ("target_relation", relation_conflict),
                ("manipulation_phase", phase_conflict),
                ("preferred_arm", arm_conflict),
            )
            if conflict
        ]
        if conflicts:
            return HPKUnresolved(
                component="task_strategy_normalizer",
                reason="conflicting_structured_sources",
                missing_fields=tuple(conflicts),
            )

        missing = [
            name
            for name, value in (
                ("operation", operation),
                ("manipulated_role", manipulated_role),
                ("manipulation_phase", phase),
                ("subgoal_purpose", purpose),
                ("preferred_arm", arm),
                ("source.planner_subtask_text", subtask_text),
                ("source.selected_skill", selected_skill),
            )
            if not value
        ]
        if operation == "place":
            if not target_role:
                missing.append("target_role")
            if not target_relation:
                missing.append("target_relation")
        if missing:
            return HPKUnresolved(
                component="task_strategy_normalizer",
                reason="required_structured_field_unresolved",
                missing_fields=tuple(missing),
            )

        payload = {
            "schema": TASK_STRATEGY_SCHEMA,
            "operation": operation,
            "manipulated_role": manipulated_role,
            "target_role": target_role or None,
            "target_relation": target_relation or None,
            "manipulation_phase": phase,
            "subgoal_purpose": purpose,
            "preferred_arm": arm,
            "source": {
                "planner_subtask_text": subtask_text,
                "selected_skill": selected_skill,
                "action_mode": operation,
                "normalization_version": TASK_STRATEGY_NORMALIZATION_VERSION,
            },
        }
        try:
            return TaskStrategyV1.from_dict(payload)
        except ValueError as exc:
            return HPKUnresolved(
                component="task_strategy_normalizer",
                reason=f"strategy_validation_failed:{type(exc).__name__}",
            )


def normalize_task_strategy(
    planner_prediction: Mapping[str, Any],
    current_scene_memory: Mapping[str, Any] | None = None,
    bound_operation_target: Mapping[str, Any] | None = None,
    active_skill: Mapping[str, Any] | str | None = None,
) -> TaskStrategyV1 | HPKUnresolved:
    return TaskStrategyNormalizer().normalize(
        planner_prediction,
        current_scene_memory,
        bound_operation_target,
        active_skill,
    )


__all__ = ["TaskStrategyNormalizer", "normalize_task_strategy"]

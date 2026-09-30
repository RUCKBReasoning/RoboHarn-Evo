from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.goal_consistency import (
    RuntimeGoalBindingV31,
    StagedCarryRealizationV31,
    build_subtask_goal_contract_v31,
    runtime_goal_semantics_v31,
)
from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    HPKV3ValidationError,
    ActionKnowledgeV3,
    SubtaskKnowledgeV3,
    SubtaskGoalContractV31,
)
from roboharn_evo.agent.core.img_agent import ImgAgent
from roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall


RMBENCH_FROZEN_GOAL_CONTEXT_SCHEMA = "roboharn_evo/rmbench/v31/frozen_goal_context"


def _finite_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    try:
        result = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON") from exc
    if not isinstance(result, dict):
        raise TypeError(f"{label} must be an object")
    return result


def _private_reference(value: Any, *, label: str, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    result = value.strip()
    if not result and not required:
        return None
    if not result or len(result) > 1000 or any(char in result for char in "\r\n"):
        raise ValueError(f"{label} must be one bounded private reference")
    return result


def _normalized_semantic(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().replace("_", " ").split())


@dataclass(frozen=True, slots=True)
class RMBenchFrozenGoalContextV31:
    """One preregistered semantic goal and its private RMBench entity binding."""

    goal_contract: SubtaskGoalContractV31
    manipulated_object_ref: str
    required_target_scene_ref: str | None = None
    local_trace_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "goal_contract",
            (
                self.goal_contract
                if isinstance(self.goal_contract, SubtaskGoalContractV31)
                else SubtaskGoalContractV31(self.goal_contract)
            ),
        )
        object.__setattr__(
            self,
            "manipulated_object_ref",
            _private_reference(
                self.manipulated_object_ref,
                label="runtime_binding.manipulated_object_ref",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "required_target_scene_ref",
            _private_reference(
                self.required_target_scene_ref,
                label="runtime_binding.required_target_scene_ref",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "local_trace_ref",
            _private_reference(
                self.local_trace_ref,
                label="runtime_binding.local_trace_ref",
                required=False,
            ),
        )
        target_required = bool(
            {"required_target_role", "required_target_relation"}.intersection(
                self.goal_contract
            )
        )
        if target_required != (self.required_target_scene_ref is not None):
            raise HPKV3ValidationError(
                "RMBench target binding must be present exactly when the Goal "
                "Contract requires a target"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RMBenchFrozenGoalContextV31":
        payload = _finite_mapping(value, label="rmbench_frozen_goal_context")
        if set(payload) != {"schema", "goal_contract", "runtime_binding"}:
            raise ValueError(
                "rmbench_frozen_goal_context must contain exactly schema, "
                "goal_contract, and runtime_binding"
            )
        if payload["schema"] not in {
            RMBENCH_FROZEN_GOAL_CONTEXT_SCHEMA, "tcm/rmbench/v31/frozen_goal_context"
        }:
            raise ValueError("unsupported RMBench frozen Goal Context schema")
        binding = _finite_mapping(payload["runtime_binding"], label="runtime_binding")
        allowed = {
            "manipulated_object_ref",
            "required_target_scene_ref",
            "local_trace_ref",
        }
        if set(binding) - allowed or "manipulated_object_ref" not in binding:
            raise ValueError("runtime_binding has unknown or missing fields")
        return cls(
            goal_contract=SubtaskGoalContractV31(payload["goal_contract"]),
            manipulated_object_ref=binding["manipulated_object_ref"],
            required_target_scene_ref=binding.get("required_target_scene_ref"),
            local_trace_ref=binding.get("local_trace_ref"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RMBenchFrozenGoalContextV31":
        source = Path(path).expanduser().resolve(strict=True)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read RMBench frozen Goal Context") from exc
        return cls.from_mapping(payload)

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema": RMBENCH_FROZEN_GOAL_CONTEXT_SCHEMA,
            "goal_contract": self.goal_contract.to_dict(),
            "runtime_binding": {
                "manipulated_object_ref": self.manipulated_object_ref,
                "required_target_scene_ref": self.required_target_scene_ref,
                "local_trace_ref": self.local_trace_ref,
            },
        }


@dataclass(frozen=True, slots=True)
class PreparedRMBenchGoalContextV31:
    scene_memory: dict[str, Any]
    runtime_binding: RuntimeGoalBindingV31
    installed: bool
    binding_status: str

    def private_audit(self) -> dict[str, Any]:
        return {
            "schema": "roboharn_evo/rmbench/v31/goal_binding_audit",
            "installed": self.installed,
            "binding_status": self.binding_status,
            "runtime_binding": self.runtime_binding.to_private_dict(),
        }


def _record_target_references(record: Mapping[str, Any]) -> set[str]:
    result = {
        str(record.get(key, "") or "").strip()
        for key in (
            "target_id",
            "support_instance_id",
            "reference_object_id",
            "reference_region_id",
        )
        if str(record.get(key, "") or "").strip()
    }
    references = record.get("reference_instance_ids")
    if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
        result.update(str(item).strip() for item in references if str(item).strip())
    return result


def build_rmbench_v31_goal_context_from_knowledge(
    *,
    task_knowledge: SubtaskKnowledgeV3 | Mapping[str, Any],
    action_knowledge: ActionKnowledgeV3 | Mapping[str, Any],
    scene_memory: Mapping[str, Any],
    manipulated_object_ref: str,
    required_target_scene_ref: str,
    expected_runtime_target_ref: str,
    local_trace_ref: str,
) -> RMBenchFrozenGoalContextV31:
    """Combine selected Store knowledge with one current Runtime target.

    The selection is preregistered by the caller.  This validator contains no
    task name, object class, spatial side, support type, or seed rule.
    """

    task = (
        task_knowledge
        if isinstance(task_knowledge, SubtaskKnowledgeV3)
        else SubtaskKnowledgeV3(task_knowledge)
    )
    action = (
        action_knowledge
        if isinstance(action_knowledge, ActionKnowledgeV3)
        else ActionKnowledgeV3(action_knowledge)
    )
    if task["status"] != "supported":
        raise HPKV3ValidationError("selected Task Knowledge is not supported")
    if action["status"] != "supported" or action["condition"]["action"] != "place":
        raise HPKV3ValidationError(
            "a supported place Action Knowledge record is required"
        )
    manipulated_ref = _private_reference(
        manipulated_object_ref,
        label="manipulated_object_ref",
        required=True,
    )
    semantic_target_ref = _private_reference(
        required_target_scene_ref,
        label="required_target_scene_ref",
        required=True,
    )
    runtime_target_ref = _private_reference(
        expected_runtime_target_ref,
        label="expected_runtime_target_ref",
        required=True,
    )
    trace_ref = _private_reference(
        local_trace_ref,
        label="local_trace_ref",
        required=True,
    )
    scene = _finite_mapping(scene_memory, label="scene_memory")
    raw_targets = scene.get("operation_targets")
    target_records = tuple(
        value
        for value in (
            raw_targets
            if isinstance(raw_targets, Sequence)
            and not isinstance(raw_targets, (str, bytes))
            else ()
        )
        if isinstance(value, Mapping)
    )
    matches = [
        value
        for value in target_records
        if str(value.get("target_id", "") or "").strip() == runtime_target_ref
        and str(value.get("action_mode", "") or "").strip().casefold() == "place"
        and str(value.get("held_instance_id", "") or "").strip() == manipulated_ref
        and semantic_target_ref in _record_target_references(value)
    ]
    if len(matches) != 1:
        raise HPKV3ValidationError(
            "the selected semantic target does not resolve to one current "
            "place target for the held object"
        )
    target = matches[0]
    if target.get("support_valid") is not True or target.get("free") is not True:
        raise HPKV3ValidationError(
            "the preregistered Runtime target must be valid and free"
        )
    runtime_semantics = runtime_goal_semantics_v31(
        instance={
            "instance_id": manipulated_ref,
            "manipulated_role": "currently held object",
        },
        operation="place",
        required_target_ref=runtime_target_ref,
        target_records=target_records,
        candidates=(target,),
        expected_effect=action["expected_effect"],
    )
    contract = build_subtask_goal_contract_v31(
        task.to_dict(),
        runtime_semantics=runtime_semantics,
        operation="place",
        expected_effect=action["expected_effect"],
    )
    context = RMBenchFrozenGoalContextV31(
        goal_contract=contract,
        manipulated_object_ref=manipulated_ref,
        required_target_scene_ref=semantic_target_ref,
        local_trace_ref=trace_ref,
    )
    prepared = RMBenchGoalContextBridgeV31(context).prepare(
        scene_memory=scene,
        active_skill_ref=trace_ref,
    )
    if (
        not prepared.installed
        or prepared.runtime_binding.required_target_ref != runtime_target_ref
    ):
        raise HPKV3ValidationError(
            "the Store-derived Goal Context does not reproduce the selected "
            "Runtime target"
        )
    return context


class RMBenchGoalContextBridgeV31:
    """Resolve one semantic destination through current RMBench target records."""

    def __init__(self, context: RMBenchFrozenGoalContextV31) -> None:
        if not isinstance(context, RMBenchFrozenGoalContextV31):
            raise TypeError("context must be RMBenchFrozenGoalContextV31")
        self.context = context
        self.last_prepared: PreparedRMBenchGoalContextV31 | None = None

    def prepare(
        self,
        *,
        scene_memory: Mapping[str, Any],
        active_skill_ref: str | None,
    ) -> PreparedRMBenchGoalContextV31:
        scene = copy.deepcopy(_finite_mapping(scene_memory, label="scene_memory"))
        scoped_ref = self.context.local_trace_ref
        current_skill_ref = str(active_skill_ref or "").strip() or None
        if scoped_ref is not None and current_skill_ref != scoped_ref:
            result = PreparedRMBenchGoalContextV31(
                scene,
                RuntimeGoalBindingV31(),
                False,
                "active skill is outside the frozen Goal Context attempt",
            )
            self.last_prepared = result
            return result

        required_scene_ref = self.context.required_target_scene_ref
        runtime_target_ref: str | None = None
        binding_status = "no target is required by the Goal Contract"
        if required_scene_ref is not None:
            raw_targets = scene.get("operation_targets")
            targets = raw_targets if isinstance(raw_targets, list) else []
            matching_indices: list[int] = []
            matching_target_refs: list[str] = []
            for index, record in enumerate(targets):
                if not isinstance(record, Mapping):
                    continue
                if required_scene_ref not in _record_target_references(record):
                    continue
                held_ref = str(record.get("held_instance_id", "") or "").strip()
                if held_ref and held_ref != self.context.manipulated_object_ref:
                    continue
                target_ref = str(record.get("target_id", "") or "").strip()
                if not target_ref:
                    continue
                matching_indices.append(index)
                if target_ref not in matching_target_refs:
                    matching_target_refs.append(target_ref)
            if len(matching_target_refs) == 1:
                runtime_target_ref = matching_target_refs[0]
                target_role = self.context.goal_contract.get("required_target_role")
                target_relation = self.context.goal_contract.get(
                    "required_target_relation"
                )
                for index in matching_indices:
                    record = dict(targets[index])
                    if target_role is not None and not record.get("target_role"):
                        record["target_role"] = target_role
                    if (
                        target_relation is not None
                        and not record.get("placement_relation")
                        and str(record.get("target_kind", "") or "").strip()
                        == "object_top"
                        and _normalized_semantic(target_relation) == "center of"
                    ):
                        # RMBench's object-top target is constructed at the
                        # support centroid.  This projection names that existing
                        # geometric fact; it does not invent or move the target.
                        record["placement_relation"] = "center of"
                    targets[index] = record
                scene["operation_targets"] = targets
                binding_status = "semantic target resolved to one Runtime target group"
            elif not matching_target_refs:
                # Keep the semantic entity ref as a deliberately non-matching
                # private target.  The shared filter then returns unresolved;
                # it can never fall back to another legal support.
                runtime_target_ref = required_scene_ref
                binding_status = "required semantic target has no Runtime target group"
            else:
                runtime_target_ref = required_scene_ref
                binding_status = "required semantic target maps ambiguously"

        result = PreparedRMBenchGoalContextV31(
            scene_memory=scene,
            runtime_binding=RuntimeGoalBindingV31(
                manipulated_object_ref=self.context.manipulated_object_ref,
                required_target_ref=runtime_target_ref,
                local_trace_ref=self.context.local_trace_ref,
            ),
            installed=True,
            binding_status=binding_status,
        )
        self.last_prepared = result
        return result

    def install(self, runtime: Any, prepared: PreparedRMBenchGoalContextV31) -> None:
        setter = getattr(runtime, "set_current_task_strategy", None)
        if not callable(setter):
            raise TypeError("HPK runtime does not expose set_current_task_strategy")
        if not prepared.installed:
            setter(None)
            return
        setter(
            self.context.goal_contract,
            runtime_binding=prepared.runtime_binding,
        )


class RMBenchV31ImgAgent(ImgAgent):
    """ImgAgent variant that installs a frozen goal at the existing seam."""

    def __init__(
        self,
        *,
        agent_card: Any,
        frozen_goal_context: RMBenchFrozenGoalContextV31,
    ) -> None:
        super().__init__(agent_card=agent_card)
        self._rmbench_v31_goal_bridge = RMBenchGoalContextBridgeV31(frozen_goal_context)

    def _select_operation_candidate_with_hierarchical_hpk(
        self,
        *,
        runtime: Any,
        scene_memory: dict[str, Any],
        instance: dict[str, Any],
        arm: str,
        action_mode: str,
        blocked_candidate_ids: list[str],
        requested_target_id: Any,
        requested_candidate_id: Any = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        active = self.memory_store.state.active_skill
        prepared = self._rmbench_v31_goal_bridge.prepare(
            scene_memory=scene_memory,
            active_skill_ref=(None if active is None else str(active.skill_id)),
        )
        self._rmbench_v31_goal_bridge.install(runtime, prepared)
        goal_consistency_enabled = bool(
            getattr(runtime, "hpk_goal_consistency_enabled", False) is True
        )
        self._record_trace_and_rollout_event(
            "hpk_v31_goal_context_prepared",
            {
                "schema": "roboharn_evo/rmbench/v31/goal_context_usage",
                "goal_contract": self._rmbench_v31_goal_bridge.context.goal_contract.to_dict(),
                "installed": prepared.installed,
                "binding_status": prepared.binding_status,
                "goal_consistency_enabled": goal_consistency_enabled,
            },
        )
        self._emit_hpk_private_trace_record_strict(
            "hpk_v31_runtime_goal_binding",
            {
                **prepared.private_audit(),
                "goal_consistency_enabled": goal_consistency_enabled,
            },
        )
        result = super()._select_operation_candidate_with_hierarchical_hpk(
            runtime=runtime,
            scene_memory=prepared.scene_memory,
            instance=instance,
            arm=arm,
            action_mode=action_mode,
            blocked_candidate_ids=blocked_candidate_ids,
            requested_target_id=requested_target_id,
            requested_candidate_id=requested_candidate_id,
        )
        if goal_consistency_enabled:
            report = getattr(runtime, "last_runtime_feasibility_report", None)
            self._record_trace_and_rollout_event(
                "hpk_v31_goal_filter_result",
                {
                    "schema": "roboharn_evo/rmbench/v31/goal_filter_result",
                    "goal_contract": self._rmbench_v31_goal_bridge.context.goal_contract.to_dict(),
                    "goal_consistent_candidate_selected": result[0] is not None,
                    "runtime_feasibility": (
                        None if report is None else report.to_dict()
                    ),
                },
            )
        return result


def install_rmbench_v31_agent(
    model: Any,
    context: RMBenchFrozenGoalContextV31,
) -> RMBenchV31ImgAgent:
    """Replace the not-yet-started default session Agent with the local bridge."""

    session = getattr(model, "session", None)
    if session is None or getattr(session, "running", False):
        raise RuntimeError(
            "RMBench v3.1 Agent must be installed before the session starts"
        )
    agent_card = getattr(model, "agent_card", None)
    if agent_card is None:
        raise TypeError("RMBench v3.1 requires a RoboHarnAgentRuntime")
    agent = RMBenchV31ImgAgent(
        agent_card=agent_card,
        frozen_goal_context=context,
    )
    session.agent = agent
    configure = getattr(model, "_configure_recovery_tool_ablations", None)
    if callable(configure):
        configure()
    return agent


def map_staged_carry_to_rmbench_calls_v31(
    realization: StagedCarryRealizationV31,
    *,
    arm: str,
    validated_lift_segments_m: Sequence[float],
    bounded_translation_steps: int,
    max_translation_m: float,
) -> tuple[RecoveryToolCall, ...]:
    """Expand shared staged carry into existing guarded RMBench primitives.

    The caller supplies Runtime-validated motion bounds.  These calls still go
    through ImgAgent's normal sanitization, candidate selection, attachment,
    collision, occupancy, release, and effect-verification path.
    """

    if not isinstance(realization, StagedCarryRealizationV31):
        raise TypeError("realization must be StagedCarryRealizationV31")
    normalized_arm = str(arm or "").strip().casefold()
    if normalized_arm not in {"left", "right"}:
        raise ValueError("arm must be left or right")
    if realization.contract["operation"] != "place":
        raise ValueError("RMBench staged carry requires a place Goal Contract")
    held_ref = realization.runtime_binding.manipulated_object_ref
    target_ref = realization.runtime_binding.required_target_ref
    if held_ref is None or target_ref is None:
        raise ValueError("RMBench staged carry requires exact private bindings")
    if (
        isinstance(bounded_translation_steps, bool)
        or not isinstance(bounded_translation_steps, int)
        or bounded_translation_steps <= 0
    ):
        raise ValueError("bounded_translation_steps must be a positive integer")
    try:
        max_translation = float(max_translation_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_translation_m must be finite and positive") from exc
    if not (0.0 < max_translation <= 0.12):
        raise ValueError("max_translation_m must use the existing RMBench bound")

    calls: list[RecoveryToolCall] = []
    for raw_distance in validated_lift_segments_m:
        try:
            distance = float(raw_distance)
        except (TypeError, ValueError) as exc:
            raise ValueError("lift segments must be finite positive values") from exc
        if not (0.0 < distance <= 0.05):
            raise ValueError("lift segments must use the existing RMBench lift bound")
        calls.append(
            RecoveryToolCall(
                "lift_ee",
                {"arm": normalized_arm, "distance": distance, "steps": 1},
            )
        )
    if not calls:
        raise ValueError("staged carry requires at least one validated lift segment")

    common = {
        "arm": normalized_arm,
        "action_mode": "place",
        "instance_id": held_ref,
        "target_id": target_ref,
        "max_translation": max_translation,
        "steps": min(bounded_translation_steps, 20),
    }
    calls.extend(
        (
            RecoveryToolCall(
                "move_ee_to_grounded_instance",
                {**common, "point_key": "approach_world_m"},
            ),
            RecoveryToolCall(
                "move_ee_to_grounded_instance",
                {**common, "point_key": "place_world_m"},
            ),
            RecoveryToolCall(
                "open_gripper",
                {
                    "arm": normalized_arm,
                    "release_held_instance_id": held_ref,
                    "release_target_id": target_ref,
                },
            ),
            RecoveryToolCall("reobserve_scene", {}),
        )
    )
    return tuple(calls)


__all__ = [
    "PreparedRMBenchGoalContextV31",
    "RMBENCH_FROZEN_GOAL_CONTEXT_SCHEMA",
    "RMBenchFrozenGoalContextV31",
    "RMBenchGoalContextBridgeV31",
    "RMBenchV31ImgAgent",
    "build_rmbench_v31_goal_context_from_knowledge",
    "install_rmbench_v31_agent",
    "map_staged_carry_to_rmbench_calls_v31",
]

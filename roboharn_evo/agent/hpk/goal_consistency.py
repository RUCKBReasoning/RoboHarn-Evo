from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    HPKV3ValidationError,
    RealizationHypothesisV31,
    RuntimeFeasibilityReportV31,
    SubtaskGoalContractV31,
)

_ACTIONS = frozenset({"grasp", "place", "contact"})
_REALIZATION_CHANGE_FIELDS = frozenset(
    {
        "approach_direction",
        "approach_reference",
        "interaction_region",
        "orientation_relation",
        "placement_relation",
        "contact_relation",
        "transport_realization",
        "avoid",
    }
)


def _finite_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    try:
        parsed = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must be an object")
    return parsed


def _natural(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().replace("_", " ").split())
    return text or None


def _reference(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or None")
    result = value.strip()
    if not result or len(result) > 1000 or any(char in result for char in "\r\n"):
        raise ValueError(f"{label} must be one bounded opaque reference")
    return result


def _semantic_equal(first: Any, second: Any) -> bool:
    left = _natural(first)
    right = _natural(second)
    return (
        left is not None and right is not None and left.casefold() == right.casefold()
    )


def _one_semantic_value(
    sources: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    *,
    label: str,
) -> str | None:
    values: list[str] = []
    for source in sources:
        for name in names:
            value = _natural(source.get(name))
            if value is not None and value.casefold() not in {
                item.casefold() for item in values
            }:
                values.append(value)
    if len(values) > 1:
        raise HPKV3ValidationError(f"{label}: conflicting structured values")
    return values[0] if values else None


def _first_semantic_value(
    sources: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> str | None:
    for source in sources:
        for name in names:
            value = _natural(source.get(name))
            if value is not None:
                return value
    return None


def _effect_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("physical_effect", "expected_effect"):
            result = _natural(value.get(key))
            if result is not None:
                return result
        return None
    return _natural(value)


@dataclass(frozen=True, slots=True)
class RuntimeGoalBindingV31:
    """Attempt-local private binding for a public semantic Goal Contract."""

    manipulated_object_ref: str | None = None
    required_target_ref: str | None = None
    local_trace_ref: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "manipulated_object_ref",
            "required_target_ref",
            "local_trace_ref",
        ):
            object.__setattr__(
                self,
                field,
                _reference(getattr(self, field), label=f"runtime_binding.{field}"),
            )

    @classmethod
    def from_value(
        cls, value: RuntimeGoalBindingV31 | Mapping[str, Any] | None
    ) -> RuntimeGoalBindingV31:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = _finite_mapping(value, label="runtime_binding")
        allowed = {
            "manipulated_object_ref",
            "required_target_ref",
            "local_trace_ref",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"runtime_binding has unknown fields: {sorted(unknown)}")
        return cls(**payload)

    def to_private_dict(self) -> dict[str, str | None]:
        return {
            "manipulated_object_ref": self.manipulated_object_ref,
            "required_target_ref": self.required_target_ref,
            "local_trace_ref": self.local_trace_ref,
        }


def merge_runtime_goal_bindings_v31(
    first: RuntimeGoalBindingV31 | Mapping[str, Any] | None,
    second: RuntimeGoalBindingV31 | Mapping[str, Any] | None,
) -> RuntimeGoalBindingV31:
    left = RuntimeGoalBindingV31.from_value(first)
    right = RuntimeGoalBindingV31.from_value(second)
    result: dict[str, str | None] = {}
    for field in (
        "manipulated_object_ref",
        "required_target_ref",
        "local_trace_ref",
    ):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if (
            left_value is not None
            and right_value is not None
            and left_value != right_value
        ):
            raise HPKV3ValidationError(
                f"runtime_binding.{field}: current binding conflicts with the Task binding"
            )
        result[field] = left_value or right_value
    return RuntimeGoalBindingV31(**result)


def build_subtask_goal_contract_v31(
    task_strategy: Mapping[str, Any],
    *,
    planner_output: Mapping[str, Any] | None = None,
    runtime_semantics: Mapping[str, Any] | None = None,
    operation: str | None = None,
    expected_effect: str | Mapping[str, Any] | None = None,
) -> SubtaskGoalContractV31:
    """Build one public contract without parsing free-form task-specific text."""

    raw_task = _finite_mapping(task_strategy, label="task_strategy")
    nested = raw_task.get("subtask_strategy")
    task = (
        _finite_mapping(nested, label="task_strategy.subtask_strategy")
        if isinstance(nested, Mapping)
        else raw_task
    )
    planner = (
        {}
        if planner_output is None
        else _finite_mapping(planner_output, label="planner_output")
    )
    runtime = (
        {}
        if runtime_semantics is None
        else _finite_mapping(runtime_semantics, label="runtime_semantics")
    )
    semantic_tags = planner.get("semantic_tags")
    tags = dict(semantic_tags) if isinstance(semantic_tags, Mapping) else {}

    subtask = _first_semantic_value((task, planner), ("subtask", "subtask_text"))
    purpose = _first_semantic_value((task, planner), ("purpose",))
    completion = _first_semantic_value(
        (task, planner),
        ("completion_condition",),
    )
    operation_values: list[Mapping[str, Any]] = [task, runtime, tags]
    if operation is not None:
        operation_values.insert(0, {"operation": operation})
    action = _one_semantic_value(
        operation_values,
        ("operation", "action", "action_mode", "subtask_type"),
        label="operation",
    )
    manipulated_role = _first_semantic_value(
        (task, runtime),
        ("manipulated_role", "object_role"),
    )
    target_role = _first_semantic_value(
        (task, runtime),
        ("required_target_role", "target_role"),
    )
    target_relation = _first_semantic_value(
        (task, runtime),
        ("required_target_relation", "target_relation", "placement_relation"),
    )
    # The Task strategy owns the target effect. Runtime's operation-level
    # effect is only a fallback because two natural-language phrases may
    # describe the same effect without being textually identical.
    effect = next(
        (
            value
            for value in (
                _effect_text(task.get("expected_effect")),
                _effect_text(runtime.get("expected_effect")),
                _effect_text(expected_effect),
            )
            if value is not None
        ),
        None,
    )
    missing = [
        name
        for name, value in (
            ("subtask", subtask),
            ("purpose", purpose),
            ("operation", action),
            ("manipulated_role", manipulated_role),
            ("expected_effect", effect),
            ("completion_condition", completion),
        )
        if value is None
    ]
    if missing:
        raise HPKV3ValidationError(
            "subtask_goal_contract: missing structured fields " + ", ".join(missing)
        )
    assert subtask is not None
    assert purpose is not None
    assert action is not None
    assert manipulated_role is not None
    assert effect is not None
    assert completion is not None
    payload: dict[str, Any] = {
        "subtask": subtask,
        "purpose": purpose,
        "operation": action.casefold(),
        "manipulated_role": manipulated_role,
        "expected_effect": effect,
        "completion_condition": completion,
    }
    if target_role is not None:
        payload["required_target_role"] = target_role
    if target_relation is not None:
        payload["required_target_relation"] = target_relation
    return SubtaskGoalContractV31(payload)


def _target_record(
    target_ref: str | None,
    target_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if target_ref is None:
        return None
    matches = [
        value
        for value in target_records
        if str(value.get("target_id", "") or "").strip() == target_ref
    ]
    return matches[0] if len(matches) == 1 else None


def _target_semantics(
    candidate: Mapping[str, Any],
    target_records: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    target_ref = str(candidate.get("target_id", "") or "").strip() or None
    record = _target_record(target_ref, target_records)
    sources = [candidate]
    if record is not None:
        sources.append(record)
    try:
        goal_role = _one_semantic_value(
            sources, ("goal_target_role",), label="candidate.goal_target_role"
        )
        goal_relation = _one_semantic_value(
            sources, ("goal_target_relation",), label="candidate.goal_target_relation"
        )
        if goal_role is not None and goal_relation is not None:
            return goal_role, goal_relation
        role = _one_semantic_value(
            sources,
            ("target_role", "reference_role", "semantic_role", "query_role", "role"),
            label="candidate.target_role",
        )
        relation = _one_semantic_value(
            sources,
            ("placement_relation", "target_relation"),
            label="candidate.target_relation",
        )
    except HPKV3ValidationError:
        return None, None
    target_kind = _one_semantic_value(
        sources, ("target_kind",), label="candidate.target_kind"
    )
    if relation is None and target_kind is not None:
        relation = target_kind
    return role, relation


def runtime_goal_semantics_v31(
    *,
    instance: Mapping[str, Any],
    operation: str,
    required_target_ref: str | None,
    target_records: Sequence[Mapping[str, Any]] = (),
    candidates: Sequence[Mapping[str, Any]] = (),
    expected_effect: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project only structured current Runtime facts into public semantics."""

    action = _natural(operation)
    if action is None or action.casefold() not in _ACTIONS:
        raise HPKV3ValidationError("runtime_semantics.operation: unsupported action")
    manipulated_role = _one_semantic_value(
        (instance,),
        ("manipulated_role", "task_role", "semantic_role", "query_role", "role"),
        label="runtime_semantics.manipulated_role",
    )
    if manipulated_role is None:
        manipulated_role = "current manipulated object"
    result: dict[str, Any] = {
        "operation": action.casefold(),
        "manipulated_role": manipulated_role,
    }
    effect = _effect_text(expected_effect)
    if effect is not None:
        result["expected_effect"] = effect
    if required_target_ref is None:
        return result
    target_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("target_id", "") or "").strip() == required_target_ref
    ]
    semantic_source: Mapping[str, Any] = (
        target_candidates[0]
        if target_candidates
        else _target_record(required_target_ref, target_records) or {}
    )
    target_role, target_relation = _target_semantics(semantic_source, target_records)
    if target_role is None:
        target_role = "required target"
    if target_relation is not None:
        result["required_target_relation"] = target_relation
    result["required_target_role"] = target_role
    return result


def _realization_family(candidate: Mapping[str, Any]) -> str | None:
    value = _one_semantic_value(
        (candidate,),
        ("realization_family", "geometry_source_class", "geometry_source"),
        label="candidate.realization_family",
    )
    if value is None:
        action = _natural(candidate.get("action_mode"))
        return None if action is None else f"{action} geometry"
    # Runtime source strings can contain opaque suffixes.  The hierarchical
    # schema is the final public-data gate; unsafe values are simply omitted.
    try:
        RuntimeFeasibilityReportV31(
            {
                "goal_consistent_candidate_available": True,
                "failed_stage": None,
                "missing_prerequisite": None,
                "available_target_roles": [],
                "available_realization_families": [value],
                "repairable_variables": [],
                "non_repairable_reason": None,
            }
        )
    except HPKV3ValidationError:
        return None
    return value


def runtime_candidate_missing_prerequisite_v31(
    candidates: Sequence[Mapping[str, Any]],
    *,
    operation: str,
    arm: str,
    required_target_ref: str | None = None,
    blocked_candidate_refs: Sequence[str] = (),
) -> str | None:
    """Summarize explicit existing Guard facts without exposing a candidate."""

    action = str(operation or "").strip().casefold()
    selected_arm = str(arm or "").strip().casefold()
    relevant = [
        candidate
        for candidate in candidates
        if str(candidate.get("action_mode", "") or "").strip().casefold() == action
        and str(candidate.get("arm", "") or "").strip().casefold() == selected_arm
        and (
            required_target_ref is None
            or str(candidate.get("target_id", "") or "").strip() == required_target_ref
        )
    ]
    if not relevant:
        return (
            "the Runtime generated no candidate for the required target"
            if required_target_ref is not None
            else "the Runtime generated no candidate for the current action"
        )
    blocked = {
        str(value).strip() for value in blocked_candidate_refs if str(value).strip()
    }
    if all(
        str(value.get("candidate_id", "") or "").strip() in blocked
        for value in relevant
    ):
        return "all current candidates are blocked by the existing Runtime guard"
    if all(
        value.get("free") is False or value.get("occupied") is True
        for value in relevant
    ):
        return "the required target region is currently occupied"
    if all(
        value.get("support_valid") is False or value.get("support_invalid") is True
        for value in relevant
    ):
        return "the required target support is not currently valid"
    if all(
        value.get("reachable") is False
        or value.get("reachable_estimate") is False
        or value.get("unreachable") is True
        for value in relevant
    ):
        return "the required realization is outside the current reachable set"
    if all(
        value.get("valid") is False
        or value.get("legal") is False
        or value.get("eligible") is False
        or value.get("feasible") is False
        for value in relevant
    ):
        return "the current candidates fail an existing Runtime feasibility guard"
    return None


@dataclass(frozen=True, slots=True)
class GoalConsistentCandidateSetV31:
    candidates: tuple[dict[str, Any], ...]
    contract: SubtaskGoalContractV31 | None
    runtime_binding: RuntimeGoalBindingV31
    feasibility_report: RuntimeFeasibilityReportV31
    rejection_reasons: tuple[str, ...] = ()

    @property
    def unresolved(self) -> bool:
        return not self.feasibility_report["goal_consistent_candidate_available"]


def unresolved_feasibility_report_v31(
    *,
    failed_stage: str,
    missing_prerequisite: str | None,
    available_target_roles: Sequence[str] = (),
    available_realization_families: Sequence[str] = (),
    repairable_variables: Sequence[str] = (),
    non_repairable_reason: str | None = None,
) -> RuntimeFeasibilityReportV31:
    return RuntimeFeasibilityReportV31(
        {
            "goal_consistent_candidate_available": False,
            "failed_stage": failed_stage,
            "missing_prerequisite": missing_prerequisite,
            "available_target_roles": list(dict.fromkeys(available_target_roles)),
            "available_realization_families": list(
                dict.fromkeys(available_realization_families)
            ),
            "repairable_variables": list(dict.fromkeys(repairable_variables)),
            "non_repairable_reason": non_repairable_reason,
        }
    )


def filter_goal_consistent_candidates_v31(
    candidates: Sequence[Mapping[str, Any]],
    *,
    contract: SubtaskGoalContractV31 | Mapping[str, Any],
    runtime_binding: RuntimeGoalBindingV31 | Mapping[str, Any],
    target_records: Sequence[Mapping[str, Any]] = (),
    additional_realization_families: Sequence[str] = (),
    repairable_variables: Sequence[str] = (),
    runtime_missing_prerequisite: str | None = None,
) -> GoalConsistentCandidateSetV31:
    """Filter an already Guard-legal set before any Action Knowledge ranking."""

    typed_contract = (
        contract
        if isinstance(contract, SubtaskGoalContractV31)
        else SubtaskGoalContractV31(contract)
    )
    binding = RuntimeGoalBindingV31.from_value(runtime_binding)
    copied = tuple(
        _finite_mapping(value, label=f"candidates[{index}]")
        for index, value in enumerate(candidates)
    )
    roles: list[str] = []
    families: list[str] = []
    for candidate in copied:
        role, _relation = _target_semantics(candidate, target_records)
        if role is not None and role not in roles:
            roles.append(role)
        family = _realization_family(candidate)
        if family is not None and family not in families:
            families.append(family)
    for family in additional_realization_families:
        normalized = _natural(family)
        if normalized is not None and normalized not in families:
            families.append(normalized)

    target_required = bool(
        {"required_target_role", "required_target_relation"}.intersection(
            typed_contract
        )
    )
    if binding.manipulated_object_ref is None:
        report = unresolved_feasibility_report_v31(
            failed_stage="goal contract binding",
            missing_prerequisite="the manipulated object is not bound to the current Runtime state",
            available_target_roles=roles,
            available_realization_families=families,
            repairable_variables=("object re-observation",),
        )
        return GoalConsistentCandidateSetV31((), typed_contract, binding, report)
    if target_required and binding.required_target_ref is None:
        report = unresolved_feasibility_report_v31(
            failed_stage="goal contract binding",
            missing_prerequisite="the required target is not bound to the current Runtime state",
            available_target_roles=roles,
            available_realization_families=families,
            repairable_variables=("target re-observation",),
        )
        return GoalConsistentCandidateSetV31((), typed_contract, binding, report)

    selected: list[dict[str, Any]] = []
    rejected: list[str] = []
    for candidate in copied:
        action = str(candidate.get("action_mode", "") or "").strip().casefold()
        if action != typed_contract["operation"]:
            rejected.append("operation mismatch")
            continue
        candidate_object_ref = str(
            candidate.get(
                "held_instance_id",
                candidate.get("manipulated_object_ref", ""),
            )
            or ""
        ).strip()
        if (
            candidate_object_ref
            and candidate_object_ref != binding.manipulated_object_ref
        ):
            rejected.append("manipulated object mismatch")
            continue
        if target_required:
            candidate_target_ref = str(candidate.get("target_id", "") or "").strip()
            if candidate_target_ref != binding.required_target_ref:
                rejected.append("required target mismatch")
                continue
            role, relation = _target_semantics(candidate, target_records)
            if (
                "required_target_role" in typed_contract
                and role is not None
                and not _semantic_equal(role, typed_contract["required_target_role"])
            ):
                rejected.append("required target role mismatch")
                continue
            if "required_target_relation" in typed_contract and not _semantic_equal(
                relation,
                typed_contract["required_target_relation"],
            ):
                rejected.append("required target relation mismatch")
                continue
        selected.append(copy.deepcopy(candidate))

    if selected:
        report = RuntimeFeasibilityReportV31(
            {
                "goal_consistent_candidate_available": True,
                "failed_stage": None,
                "missing_prerequisite": None,
                "available_target_roles": roles,
                "available_realization_families": families,
                "repairable_variables": [],
                "non_repairable_reason": None,
            }
        )
        return GoalConsistentCandidateSetV31(
            tuple(selected), typed_contract, binding, report, tuple(rejected)
        )

    failed_stage = "runtime guards" if not copied else "goal contract consistency"
    if not copied:
        missing = runtime_missing_prerequisite or (
            "no candidate passed the existing Runtime feasibility guards"
        )
    elif "required target mismatch" in rejected:
        missing = "the Runtime-legal candidates do not preserve the required target"
    elif "required target relation mismatch" in rejected:
        missing = (
            "the Runtime-legal candidates do not preserve the required target relation"
        )
    elif "manipulated object mismatch" in rejected:
        missing = "the Runtime-legal candidates do not preserve the manipulated object"
    else:
        missing = "no Runtime-legal candidate satisfies the current Goal Contract"
    repairs = list(repairable_variables)
    if not repairs:
        repairs.append(
            "candidate re-observation"
            if failed_stage == "runtime guards"
            else "goal-consistent candidate generation"
        )
    report = unresolved_feasibility_report_v31(
        failed_stage=failed_stage,
        missing_prerequisite=missing,
        available_target_roles=roles,
        available_realization_families=families,
        repairable_variables=repairs,
    )
    return GoalConsistentCandidateSetV31(
        (), typed_contract, binding, report, tuple(rejected)
    )


def build_realization_hypothesis_v31(
    *,
    contract: SubtaskGoalContractV31 | Mapping[str, Any],
    feasibility_report: RuntimeFeasibilityReportV31 | Mapping[str, Any],
    observed_problem: str,
    change: Mapping[str, Any],
    runtime_realization: str,
    support_condition: str,
    oppose_condition: str,
    rationale: str,
    preserve: Mapping[str, Any] | None = None,
    status: str = "pending hypothesis",
) -> RealizationHypothesisV31:
    """Validate an Action-level proposal against the frozen semantic goal."""

    typed_contract = (
        contract
        if isinstance(contract, SubtaskGoalContractV31)
        else SubtaskGoalContractV31(contract)
    )
    report = (
        feasibility_report
        if isinstance(feasibility_report, RuntimeFeasibilityReportV31)
        else RuntimeFeasibilityReportV31(feasibility_report)
    )
    preserved = typed_contract.to_dict() if preserve is None else dict(preserve)
    if SubtaskGoalContractV31(preserved).to_dict() != typed_contract.to_dict():
        raise HPKV3ValidationError(
            "realization_hypothesis.preserve: must exactly equal the current Goal Contract"
        )
    if status != "pending hypothesis":
        raise HPKV3ValidationError(
            "a proposal remains a pending hypothesis until real execution and verification"
        )
    raw_change = _finite_mapping(change, label="change")
    unknown = set(raw_change) - _REALIZATION_CHANGE_FIELDS
    if unknown:
        raise HPKV3ValidationError(
            f"change: unsupported realization variables {sorted(unknown)}"
        )
    normalized_realization = _natural(runtime_realization)
    if normalized_realization is None:
        raise HPKV3ValidationError("runtime_realization: must be non-empty")
    available = {value.casefold() for value in report["available_realization_families"]}
    if normalized_realization.casefold() not in available:
        raise HPKV3ValidationError(
            "runtime_realization: is not supported by the current Runtime report"
        )
    repairable = {value.casefold() for value in report["repairable_variables"]}
    changed_variables = {key.replace("_", " ").casefold() for key in raw_change}
    if not changed_variables.intersection(repairable):
        raise HPKV3ValidationError(
            "change: does not modify a Runtime-reported repairable variable"
        )
    return RealizationHypothesisV31(
        {
            "scope": "action realization",
            "observed_problem": observed_problem,
            "preserve": typed_contract.to_dict(),
            "change": raw_change,
            "runtime_realization": normalized_realization,
            "support_condition": support_condition,
            "oppose_condition": oppose_condition,
            "rationale": rationale,
            "status": status,
        }
    )


@dataclass(frozen=True, slots=True)
class StagedCarryRealizationV31:
    """A target-preserving mapping onto existing guarded motion primitives."""

    contract: SubtaskGoalContractV31
    runtime_binding: RuntimeGoalBindingV31
    stages: tuple[str, str, str] = (
        "validated lift",
        "bounded translation",
        "target-relative lower",
    )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "realization_family": "staged carry",
            "goal_contract": self.contract.to_dict(),
            "stages": list(self.stages),
            "guard_policy": "reuse the existing attachment and motion guards at every stage",
        }

    def stage_failure_report(
        self, *, stage: str, missing_prerequisite: str
    ) -> RuntimeFeasibilityReportV31:
        if stage not in self.stages:
            raise ValueError("stage must belong to this staged carry realization")
        return unresolved_feasibility_report_v31(
            failed_stage=stage,
            missing_prerequisite=missing_prerequisite,
            available_realization_families=("staged carry",),
            repairable_variables=("transport realization",),
        )


def map_staged_carry_realization_v31(
    *,
    hypothesis: RealizationHypothesisV31 | Mapping[str, Any],
    contract: SubtaskGoalContractV31 | Mapping[str, Any],
    runtime_binding: RuntimeGoalBindingV31 | Mapping[str, Any],
    verified_attached: bool,
) -> StagedCarryRealizationV31:
    typed_hypothesis = (
        hypothesis
        if isinstance(hypothesis, RealizationHypothesisV31)
        else RealizationHypothesisV31(hypothesis)
    )
    typed_contract = (
        contract
        if isinstance(contract, SubtaskGoalContractV31)
        else SubtaskGoalContractV31(contract)
    )
    binding = RuntimeGoalBindingV31.from_value(runtime_binding)
    if typed_hypothesis["preserve"] != typed_contract.to_dict():
        raise HPKV3ValidationError("staged carry changed the current Goal Contract")
    if typed_hypothesis["runtime_realization"].casefold() != "staged carry":
        raise HPKV3ValidationError("hypothesis does not request staged carry")
    if typed_contract["operation"] != "place":
        raise HPKV3ValidationError("staged carry is only a place realization")
    if not verified_attached:
        raise HPKV3ValidationError(
            "staged carry requires a deterministically verified attached object"
        )
    if binding.manipulated_object_ref is None or binding.required_target_ref is None:
        raise HPKV3ValidationError(
            "staged carry requires exact manipulated-object and target bindings"
        )
    return StagedCarryRealizationV31(typed_contract, binding)


__all__ = [
    "GoalConsistentCandidateSetV31",
    "RuntimeGoalBindingV31",
    "StagedCarryRealizationV31",
    "build_realization_hypothesis_v31",
    "build_subtask_goal_contract_v31",
    "filter_goal_consistent_candidates_v31",
    "map_staged_carry_realization_v31",
    "merge_runtime_goal_bindings_v31",
    "runtime_candidate_missing_prerequisite_v31",
    "runtime_goal_semantics_v31",
    "unresolved_feasibility_report_v31",
]

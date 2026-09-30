from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar

_ACTIONS = frozenset({"grasp", "place", "contact"})
_VERDICTS = frozenset({"support", "oppose", "unverified"})
_EXECUTION_STATUSES = frozenset({"completed", "not completed", "interrupted"})
_EVIDENCE_TIMINGS = frozenset({"immediate", "delayed"})
_KNOWLEDGE_STATUSES = frozenset({"candidate", "supported", "contested"})
_RATIONALE_STATUSES = frozenset({"observed pattern", "inferred hypothesis"})
_REALIZATION_STATUSES = frozenset(
    {
        "pending hypothesis",
        "executed",
        "supported",
        "opposed",
        "unverified",
        "unrealizable",
    }
)
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
_REALIZATION_RESULTS = frozenset({"goal consistent", "goal inconsistent", "unresolved"})
_DYNAMIC_OBJECT_FIELDS = frozenset(
    {"held_state", "support_relation", "attachment_state", "current_position"}
)
_OPAQUE_VALUE_RE = re.compile(
    r"(?:\b(?:knowledge|proposal|episode|track|candidate)[ _-]?id\b|"
    r"\b(?:track|candidate)_[A-Za-z0-9_-]+\b|"
    r"\b(?:hpk|afk)[a-z]*_[0-9a-f]{12,}\b|\b[0-9a-f]{64}\b)",
    re.IGNORECASE,
)
_PRIVATE_GEOMETRY_RE = re.compile(
    r"(?:\babsolute (?:pose|position|coordinate)s?\b|"
    r"\b(?:x|y|z)\s*=\s*-?\d|"
    r"\b(?:joint|motion) trajector(?:y|ies)\b|"
    r"\bcandidate (?:name|identifier|id)\b|"
    r"(?:^|\s)/(?:mnt|tmp|root|home)(?:/|\s|$))",
    re.IGNORECASE,
)
_IDENTIFIER_STYLE_VALUE_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b", re.IGNORECASE)


class HPKV3ValidationError(ValueError):
    """层次化 HPK 数据违反 v3 语义约定。"""


def _fail(path: str, message: str) -> None:
    raise HPKV3ValidationError(f"{path}: {message}")


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        parsed = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise HPKV3ValidationError(f"{path}: must be finite JSON") from exc
    if not isinstance(parsed, dict):
        _fail(path, "must be an object")
    return parsed


def _fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    path: str,
) -> None:
    actual = set(value)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        _fail(
            path,
            f"fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}",
        )


def _text(
    value: Any,
    *,
    path: str,
    max_chars: int = 1200,
    natural_language: bool = True,
) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    normalized = " ".join(value.strip().split())
    if not normalized or len(normalized) > max_chars:
        _fail(path, f"must contain 1..{max_chars} characters")
    if natural_language and _IDENTIFIER_STYLE_VALUE_RE.search(normalized):
        _fail(path, "must use natural-language words instead of underscore enums")
    if _OPAQUE_VALUE_RE.search(normalized):
        _fail(path, "must not contain an opaque runtime identifier")
    if _PRIVATE_GEOMETRY_RE.search(normalized):
        _fail(path, "must not contain private or absolute motion geometry")
    return normalized


def _optional_text(value: Any, *, path: str, max_chars: int = 1200) -> str | None:
    if value is None:
        return None
    return _text(value, path=path, max_chars=max_chars)


def _enum(value: Any, allowed: frozenset[str], *, path: str) -> str:
    normalized = _text(value, path=path, max_chars=80)
    if normalized not in allowed:
        _fail(path, f"unsupported value {normalized!r}")
    return normalized


def _strings(
    value: Any,
    *,
    path: str,
    max_items: int = 32,
    unique: bool = True,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(path, "must be an array")
    if len(value) > max_items:
        _fail(path, f"must contain at most {max_items} values")
    result = [
        _text(item, path=f"{path}[{index}]", max_chars=600)
        for index, item in enumerate(value)
    ]
    if unique and len(result) != len(set(result)):
        _fail(path, "must not contain duplicate values")
    return result


def _nonnegative_integer(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, "must be a non-negative integer")
    return value


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


class _Record(Mapping[str, Any]):
    _validator: ClassVar[Any]

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._data = self._validator(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> _Record:
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


def _stable_object(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={"description"},
        optional={"shape", "category", "role", "color"},
        path=path,
    )
    dynamic = set(payload).intersection(_DYNAMIC_OBJECT_FIELDS)
    if dynamic:
        _fail(
            path,
            f"dynamic state belongs in before_state/after_state: {sorted(dynamic)}",
        )
    result = {"description": _text(payload["description"], path=f"{path}.description")}
    for key in ("shape", "category", "role", "color"):
        if key in payload and payload[key] is not None:
            result[key] = _text(payload[key], path=f"{path}.{key}", max_chars=240)
    return result


def _temporal_state(
    value: Any,
    *,
    path: str,
    subtask: bool = False,
) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    scalar_fields = {
        "object",
        "held_state",
        "support_relation",
        "target_relation",
    }
    if subtask:
        scalar_fields.add("task_progress")
        scalar_fields.discard("target_relation")
    _fields(
        payload,
        required={"relevant_relations"},
        optional=scalar_fields,
        path=path,
    )
    result = {
        key: _optional_text(payload.get(key), path=f"{path}.{key}", max_chars=300)
        for key in sorted(scalar_fields)
    }
    result["relevant_relations"] = _strings(
        payload["relevant_relations"], path=f"{path}.relevant_relations"
    )
    return _drop_none(result)


def _applicability_condition(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    fields = {
        "object",
        "held_state",
        "support_relation",
        "target_relation",
        "manipulation_state",
    }
    _fields(payload, required=set(), optional=fields, path=path)
    result = {
        key: _optional_text(payload.get(key), path=f"{path}.{key}", max_chars=300)
        for key in sorted(fields)
    }
    return _drop_none(result)


def _selection_basis(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={"observed_facts", "inferred_rationale", "source"},
        path=path,
    )
    facts = _strings(payload["observed_facts"], path=f"{path}.observed_facts")
    if not facts:
        _fail(f"{path}.observed_facts", "must contain at least one fact")
    return {
        "observed_facts": facts,
        "inferred_rationale": _text(
            payload["inferred_rationale"], path=f"{path}.inferred_rationale"
        ),
        "source": _text(payload["source"], path=f"{path}.source", max_chars=300),
    }


def _geometry(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    scalar_fields = {
        "approach_direction",
        "approach_reference",
        "interaction_region",
        "orientation_relation",
        "placement_relation",
        "contact_relation",
        "clearance_or_support_constraint",
    }
    _fields(payload, required={"avoid"}, optional=scalar_fields, path=path)
    result = {
        key: _optional_text(payload.get(key), path=f"{path}.{key}", max_chars=500)
        for key in sorted(scalar_fields)
    }
    result["avoid"] = _strings(payload["avoid"], path=f"{path}.avoid")
    return _drop_none(result)


def _strategy_rationale(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    _fields(payload, required={"rationale", "source", "status"}, path=path)
    return {
        "rationale": _text(payload["rationale"], path=f"{path}.rationale"),
        "source": _text(payload["source"], path=f"{path}.source", max_chars=300),
        "status": _enum(payload["status"], _RATIONALE_STATUSES, path=f"{path}.status"),
    }


def _expected_effect(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={"physical_effect", "verification_observation"},
        path=path,
    )
    return {
        "physical_effect": _text(
            payload["physical_effect"], path=f"{path}.physical_effect"
        ),
        "verification_observation": _text(
            payload["verification_observation"],
            path=f"{path}.verification_observation",
        ),
    }


def _executed_action(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={"action", "observed_strategy"},
        optional={"executed_arm"},
        path=path,
    )
    result = {
        "action": _enum(payload["action"], _ACTIONS, path=f"{path}.action"),
        "executed_arm": _optional_text(
            payload.get("executed_arm"), path=f"{path}.executed_arm", max_chars=80
        ),
        "observed_strategy": _text(
            payload["observed_strategy"], path=f"{path}.observed_strategy"
        ),
    }
    return _drop_none(result)


def _validate_action_evidence(value: Any) -> dict[str, Any]:
    path = "action_evidence"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={
            "before_state",
            "executed_action",
            "execution_status",
            "after_state",
            "observed_result",
            "verdict",
            "evidence_timing",
            "verifier_source",
        },
        optional={"missing_evidence"},
        path=path,
    )
    verdict = _enum(payload["verdict"], _VERDICTS, path=f"{path}.verdict")
    missing = _optional_text(
        payload.get("missing_evidence"), path=f"{path}.missing_evidence"
    )
    if verdict == "unverified" and missing is None:
        _fail(f"{path}.missing_evidence", "is required for unverified evidence")
    result = {
        "before_state": _temporal_state(
            payload["before_state"], path=f"{path}.before_state"
        ),
        "executed_action": _executed_action(
            payload["executed_action"], path=f"{path}.executed_action"
        ),
        "execution_status": _enum(
            payload["execution_status"],
            _EXECUTION_STATUSES,
            path=f"{path}.execution_status",
        ),
        "after_state": _temporal_state(
            payload["after_state"], path=f"{path}.after_state"
        ),
        "observed_result": _text(
            payload["observed_result"], path=f"{path}.observed_result"
        ),
        "verdict": verdict,
        "missing_evidence": missing,
        "evidence_timing": _enum(
            payload["evidence_timing"],
            _EVIDENCE_TIMINGS,
            path=f"{path}.evidence_timing",
        ),
        "verifier_source": _text(
            payload["verifier_source"], path=f"{path}.verifier_source"
        ),
    }
    return _drop_none(result)


class ActionEvidenceV3(_Record):
    _validator = staticmethod(_validate_action_evidence)


def _validate_subtask_goal_contract(value: Any) -> dict[str, Any]:
    path = "subtask_goal_contract"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={
            "subtask",
            "purpose",
            "operation",
            "manipulated_role",
            "expected_effect",
            "completion_condition",
        },
        optional={"required_target_role", "required_target_relation"},
        path=path,
    )
    operation = _enum(payload["operation"], _ACTIONS, path=f"{path}.operation")
    target_role = _optional_text(
        payload.get("required_target_role"),
        path=f"{path}.required_target_role",
        max_chars=300,
    )
    target_relation = _optional_text(
        payload.get("required_target_relation"),
        path=f"{path}.required_target_relation",
        max_chars=300,
    )
    if operation == "place" and (target_role is None or target_relation is None):
        _fail(path, "place requires a target role and target relation")
    result = {
        "subtask": _text(payload["subtask"], path=f"{path}.subtask"),
        "purpose": _text(payload["purpose"], path=f"{path}.purpose"),
        "operation": operation,
        "manipulated_role": _text(
            payload["manipulated_role"],
            path=f"{path}.manipulated_role",
            max_chars=300,
        ),
        "required_target_role": target_role,
        "required_target_relation": target_relation,
        "expected_effect": _text(
            payload["expected_effect"], path=f"{path}.expected_effect"
        ),
        "completion_condition": _text(
            payload["completion_condition"],
            path=f"{path}.completion_condition",
        ),
    }
    return _drop_none(result)


class SubtaskGoalContractV31(_Record):
    """One execution attempt's semantic goal; never a Store knowledge unit."""

    _validator = staticmethod(_validate_subtask_goal_contract)


def subtask_goal_contract_json_schema() -> dict[str, Any]:
    """Model output for the current semantic goal, with no Runtime bindings."""

    text = {"type": "string", "minLength": 1, "pattern": r"^[^_]+$"}
    properties = {
        key: dict(text)
        for key in (
            "subtask",
            "purpose",
            "manipulated_role",
            "expected_effect",
            "completion_condition",
        )
    }
    properties["operation"] = {"type": "string", "enum": sorted(_ACTIONS)}
    for key in ("required_target_role", "required_target_relation"):
        properties[key] = {"anyOf": [dict(text), {"type": "null"}]}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _validate_runtime_feasibility_report(value: Any) -> dict[str, Any]:
    path = "runtime_feasibility_report"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={
            "goal_consistent_candidate_available",
            "failed_stage",
            "missing_prerequisite",
            "available_target_roles",
            "available_realization_families",
            "repairable_variables",
            "non_repairable_reason",
        },
        path=path,
    )
    available = payload["goal_consistent_candidate_available"]
    if not isinstance(available, bool):
        _fail(f"{path}.goal_consistent_candidate_available", "must be a boolean")
    failed_stage = _optional_text(
        payload["failed_stage"], path=f"{path}.failed_stage", max_chars=300
    )
    missing = _optional_text(
        payload["missing_prerequisite"],
        path=f"{path}.missing_prerequisite",
    )
    non_repairable = _optional_text(
        payload["non_repairable_reason"],
        path=f"{path}.non_repairable_reason",
    )
    if available and any(
        value is not None for value in (failed_stage, missing, non_repairable)
    ):
        _fail(path, "an available realization cannot report a failed stage")
    if not available and failed_stage is None:
        _fail(f"{path}.failed_stage", "is required when no realization is available")
    if not available and missing is None and non_repairable is None:
        _fail(
            path,
            "an unavailable realization requires a missing prerequisite or non-repairable reason",
        )
    return {
        "goal_consistent_candidate_available": available,
        "failed_stage": failed_stage,
        "missing_prerequisite": missing,
        "available_target_roles": _strings(
            payload["available_target_roles"],
            path=f"{path}.available_target_roles",
        ),
        "available_realization_families": _strings(
            payload["available_realization_families"],
            path=f"{path}.available_realization_families",
        ),
        "repairable_variables": _strings(
            payload["repairable_variables"],
            path=f"{path}.repairable_variables",
        ),
        "non_repairable_reason": non_repairable,
    }


class RuntimeFeasibilityReportV31(_Record):
    """Public, Runtime-authored explanation of one unresolved realization."""

    _validator = staticmethod(_validate_runtime_feasibility_report)


def _validate_realization_hypothesis(value: Any) -> dict[str, Any]:
    path = "realization_hypothesis"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={
            "scope",
            "observed_problem",
            "preserve",
            "change",
            "runtime_realization",
            "support_condition",
            "oppose_condition",
            "rationale",
            "status",
        },
        path=path,
    )
    scope = _enum(
        payload["scope"], frozenset({"action realization"}), path=f"{path}.scope"
    )
    raw_change = _mapping(payload["change"], path=f"{path}.change")
    unknown = set(raw_change) - _REALIZATION_CHANGE_FIELDS
    if unknown:
        _fail(f"{path}.change", f"unsupported realization variables: {sorted(unknown)}")
    if not raw_change:
        _fail(f"{path}.change", "must change at least one realization variable")
    change: dict[str, Any] = {}
    for key, item in raw_change.items():
        if key == "avoid":
            change[key] = _strings(item, path=f"{path}.change.avoid")
        else:
            change[key] = _text(item, path=f"{path}.change.{key}")
    return {
        "scope": scope,
        "observed_problem": _text(
            payload["observed_problem"], path=f"{path}.observed_problem"
        ),
        "preserve": SubtaskGoalContractV31(payload["preserve"]).to_dict(),
        "change": change,
        "runtime_realization": _text(
            payload["runtime_realization"], path=f"{path}.runtime_realization"
        ),
        "support_condition": _text(
            payload["support_condition"], path=f"{path}.support_condition"
        ),
        "oppose_condition": _text(
            payload["oppose_condition"], path=f"{path}.oppose_condition"
        ),
        "rationale": _text(payload["rationale"], path=f"{path}.rationale"),
        "status": _enum(
            payload["status"], _REALIZATION_STATUSES, path=f"{path}.status"
        ),
    }


class RealizationHypothesisV31(_Record):
    """A goal-preserving realization to test, not persistent knowledge."""

    _validator = staticmethod(_validate_realization_hypothesis)


def _validate_action_evidence_v31(value: Any) -> dict[str, Any]:
    path = "action_evidence_v31"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={
            "goal_contract",
            "realization_status",
            "before_state",
            "executed_action",
            "execution_status",
            "after_state",
            "observed_result",
            "verdict",
            "evidence_timing",
            "verifier_source",
        },
        optional={"missing_evidence"},
        path=path,
    )
    contract = SubtaskGoalContractV31(payload["goal_contract"])
    base = _validate_action_evidence(
        {
            key: item
            for key, item in payload.items()
            if key not in {"goal_contract", "realization_status"}
        }
    )
    realization_status = _enum(
        payload["realization_status"],
        _REALIZATION_RESULTS,
        path=f"{path}.realization_status",
    )
    if base["executed_action"]["action"] != contract["operation"]:
        _fail(path, "executed action must match the Goal Contract operation")
    if realization_status != "goal consistent" and base["verdict"] != "unverified":
        _fail(
            path,
            "goal-inconsistent or unresolved realization can only be unverified",
        )
    return {
        "goal_contract": contract.to_dict(),
        **base,
        "realization_status": realization_status,
    }


class ActionEvidenceV31(_Record):
    _validator = staticmethod(_validate_action_evidence_v31)


def _statistics(value: Any, *, path: str) -> dict[str, int]:
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={"support", "oppose", "unverified", "independent_verified_trials"},
        path=path,
    )
    return {
        key: _nonnegative_integer(payload[key], path=f"{path}.{key}")
        for key in ("support", "oppose", "unverified", "independent_verified_trials")
    }


def _status_from_counts(counts: Mapping[str, int]) -> str:
    if counts["oppose"]:
        return "contested"
    if counts["support"]:
        return "supported"
    return "candidate"


def _validate_action_candidate(value: Any) -> dict[str, Any]:
    path = "action_knowledge_candidate"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={
            "action",
            "applicability_condition",
            "geometric_strategy",
            "strategy_rationale",
            "expected_effect",
            "evidence",
            "statistics",
            "status",
        },
        path=path,
    )
    action = _enum(payload["action"], _ACTIONS, path=f"{path}.action")
    evidence_raw = payload["evidence"]
    if isinstance(evidence_raw, (str, bytes)) or not isinstance(evidence_raw, Sequence):
        _fail(f"{path}.evidence", "must be an array")
    if not evidence_raw:
        _fail(f"{path}.evidence", "must contain at least one execution")
    evidence = [ActionEvidenceV3(item).to_dict() for item in evidence_raw]
    for index, item in enumerate(evidence):
        if item["executed_action"]["action"] != action:
            _fail(
                f"{path}.evidence[{index}].executed_action.action",
                "must match the action knowledge type",
            )
    counts = _statistics(payload["statistics"], path=f"{path}.statistics")
    actual = {
        verdict: sum(item["verdict"] == verdict for item in evidence)
        for verdict in _VERDICTS
    }
    if any(counts[key] != actual[key] for key in _VERDICTS):
        _fail(f"{path}.statistics", "must exactly count the evidence verdicts")
    if counts["independent_verified_trials"] != counts["support"] + counts["oppose"]:
        _fail(
            f"{path}.statistics.independent_verified_trials",
            "must count support and oppose executions exactly once",
        )
    status = _enum(payload["status"], _KNOWLEDGE_STATUSES, path=f"{path}.status")
    if status != _status_from_counts(counts):
        _fail(f"{path}.status", "does not match the evidence statistics")
    return {
        "action": action,
        "applicability_condition": _applicability_condition(
            payload["applicability_condition"],
            path=f"{path}.applicability_condition",
        ),
        "geometric_strategy": _geometry(
            payload["geometric_strategy"], path=f"{path}.geometric_strategy"
        ),
        "strategy_rationale": _strategy_rationale(
            payload["strategy_rationale"], path=f"{path}.strategy_rationale"
        ),
        "expected_effect": _expected_effect(
            payload["expected_effect"], path=f"{path}.expected_effect"
        ),
        "evidence": evidence,
        "statistics": counts,
        "status": status,
    }


class ActionKnowledgeCandidateV3(_Record):
    _validator = staticmethod(_validate_action_candidate)


def _validate_subtask_package(value: Any) -> dict[str, Any]:
    path = "subtask_package"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={
            "subtask",
            "purpose",
            "before_state",
            "selection_basis",
            "completion_condition",
            "action_knowledge",
            "planned_next_subtask",
        },
        path=path,
    )
    actions_raw = payload["action_knowledge"]
    if isinstance(actions_raw, (str, bytes)) or not isinstance(actions_raw, Sequence):
        _fail(f"{path}.action_knowledge", "must be an array")
    if not actions_raw:
        _fail(f"{path}.action_knowledge", "must contain at least one action")
    return {
        "subtask": _text(payload["subtask"], path=f"{path}.subtask"),
        "purpose": _text(payload["purpose"], path=f"{path}.purpose"),
        "before_state": _temporal_state(
            payload["before_state"], path=f"{path}.before_state", subtask=True
        ),
        "selection_basis": _selection_basis(
            payload["selection_basis"], path=f"{path}.selection_basis"
        ),
        "completion_condition": _text(
            payload["completion_condition"], path=f"{path}.completion_condition"
        ),
        "action_knowledge": [
            ActionKnowledgeCandidateV3(item).to_dict() for item in actions_raw
        ],
        "planned_next_subtask": _optional_text(
            payload["planned_next_subtask"], path=f"{path}.planned_next_subtask"
        ),
    }


class SubtaskPackageV3(_Record):
    _validator = staticmethod(_validate_subtask_package)


def _validate_trajectory_package(value: Any) -> dict[str, Any]:
    path = "trajectory_knowledge_package"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={"trajectory_context", "task_strategy", "trajectory_outcome"},
        path=path,
    )
    context = _mapping(payload["trajectory_context"], path=f"{path}.trajectory_context")
    _fields(context, required={"task", "objects"}, path=f"{path}.trajectory_context")
    objects_raw = context["objects"]
    if isinstance(objects_raw, (str, bytes)) or not isinstance(objects_raw, Sequence):
        _fail(f"{path}.trajectory_context.objects", "must be an array")
    if not objects_raw:
        _fail(f"{path}.trajectory_context.objects", "must contain at least one object")
    strategy = _mapping(payload["task_strategy"], path=f"{path}.task_strategy")
    _fields(
        strategy,
        required={"overall_goal", "plan_summary", "subtasks"},
        path=f"{path}.task_strategy",
    )
    subtasks_raw = strategy["subtasks"]
    if isinstance(subtasks_raw, (str, bytes)) or not isinstance(subtasks_raw, Sequence):
        _fail(f"{path}.task_strategy.subtasks", "must be an array")
    if not subtasks_raw:
        _fail(f"{path}.task_strategy.subtasks", "must contain at least one subtask")
    outcome = _mapping(payload["trajectory_outcome"], path=f"{path}.trajectory_outcome")
    _fields(outcome, required={"status", "reason"}, path=f"{path}.trajectory_outcome")
    return {
        "trajectory_context": {
            "task": _text(context["task"], path=f"{path}.trajectory_context.task"),
            "objects": [
                _stable_object(item, path=f"{path}.trajectory_context.objects[{index}]")
                for index, item in enumerate(objects_raw)
            ],
        },
        "task_strategy": {
            "overall_goal": _text(
                strategy["overall_goal"], path=f"{path}.task_strategy.overall_goal"
            ),
            "plan_summary": _text(
                strategy["plan_summary"], path=f"{path}.task_strategy.plan_summary"
            ),
            "subtasks": [SubtaskPackageV3(item).to_dict() for item in subtasks_raw],
        },
        "trajectory_outcome": {
            "status": _text(
                outcome["status"], path=f"{path}.trajectory_outcome.status"
            ),
            "reason": _text(
                outcome["reason"], path=f"{path}.trajectory_outcome.reason"
            ),
        },
    }


class TrajectoryKnowledgePackageV3(_Record):
    _validator = staticmethod(_validate_trajectory_package)


def _validate_subtask_knowledge(value: Any) -> dict[str, Any]:
    path = "subtask_knowledge"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={"condition", "subtask_strategy", "evidence_summary", "status"},
        path=path,
    )
    condition = _mapping(payload["condition"], path=f"{path}.condition")
    _fields(
        condition,
        required={"overall_goal", "task_state", "relevant_relations"},
        path=f"{path}.condition",
    )
    strategy = _mapping(payload["subtask_strategy"], path=f"{path}.subtask_strategy")
    _fields(
        strategy,
        required={
            "subtask",
            "purpose",
            "selection_basis_summary",
            "completion_condition",
            "planned_next_subtask",
        },
        path=f"{path}.subtask_strategy",
    )
    summary = _mapping(payload["evidence_summary"], path=f"{path}.evidence_summary")
    _fields(
        summary,
        required={"support", "oppose", "unverified"},
        path=f"{path}.evidence_summary",
    )
    counts = {
        key: _nonnegative_integer(summary[key], path=f"{path}.evidence_summary.{key}")
        for key in ("support", "oppose", "unverified")
    }
    status = _enum(payload["status"], _KNOWLEDGE_STATUSES, path=f"{path}.status")
    if status != _status_from_counts({**counts, "independent_verified_trials": 0}):
        _fail(f"{path}.status", "does not match the evidence summary")
    return {
        "condition": {
            "overall_goal": _text(
                condition["overall_goal"], path=f"{path}.condition.overall_goal"
            ),
            "task_state": _text(
                condition["task_state"], path=f"{path}.condition.task_state"
            ),
            "relevant_relations": _strings(
                condition["relevant_relations"],
                path=f"{path}.condition.relevant_relations",
            ),
        },
        "subtask_strategy": {
            "subtask": _text(
                strategy["subtask"], path=f"{path}.subtask_strategy.subtask"
            ),
            "purpose": _text(
                strategy["purpose"], path=f"{path}.subtask_strategy.purpose"
            ),
            "selection_basis_summary": _text(
                strategy["selection_basis_summary"],
                path=f"{path}.subtask_strategy.selection_basis_summary",
            ),
            "completion_condition": _text(
                strategy["completion_condition"],
                path=f"{path}.subtask_strategy.completion_condition",
            ),
            "planned_next_subtask": _optional_text(
                strategy["planned_next_subtask"],
                path=f"{path}.subtask_strategy.planned_next_subtask",
            ),
        },
        "evidence_summary": counts,
        "status": status,
    }


class SubtaskKnowledgeV3(_Record):
    _validator = staticmethod(_validate_subtask_knowledge)


def _validate_action_knowledge(value: Any) -> dict[str, Any]:
    path = "action_knowledge"
    payload = _mapping(value, path=path)
    _fields(
        payload,
        required={
            "condition",
            "geometric_strategy",
            "expected_effect",
            "evidence_summary",
            "status",
        },
        path=path,
    )
    condition = _mapping(payload["condition"], path=f"{path}.condition")
    _fields(
        condition,
        required={"action", "object_description"},
        optional={"held_state", "support_relation", "target_relation"},
        path=f"{path}.condition",
    )
    counts = _statistics(payload["evidence_summary"], path=f"{path}.evidence_summary")
    status = _enum(payload["status"], _KNOWLEDGE_STATUSES, path=f"{path}.status")
    if status != _status_from_counts(counts):
        _fail(f"{path}.status", "does not match the evidence summary")
    condition_result = {
        "action": _enum(condition["action"], _ACTIONS, path=f"{path}.condition.action"),
        "object_description": _text(
            condition["object_description"], path=f"{path}.condition.object_description"
        ),
    }
    for key in ("held_state", "support_relation", "target_relation"):
        if key in condition and condition[key] is not None:
            condition_result[key] = _text(
                condition[key], path=f"{path}.condition.{key}"
            )
    return {
        "condition": condition_result,
        "geometric_strategy": _geometry(
            payload["geometric_strategy"], path=f"{path}.geometric_strategy"
        ),
        "expected_effect": _expected_effect(
            payload["expected_effect"], path=f"{path}.expected_effect"
        ),
        "evidence_summary": counts,
        "status": status,
    }


class ActionKnowledgeV3(_Record):
    _validator = staticmethod(_validate_action_knowledge)


def _state_summary(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    object_text = state.get("object")
    if isinstance(object_text, str):
        parts.append(object_text)
    for key in ("held_state", "support_relation", "task_progress"):
        value = state.get(key)
        if isinstance(value, str):
            parts.append(value)
    relations = state.get("relevant_relations", [])
    if isinstance(relations, list):
        parts.extend(str(item) for item in relations)
    return "; ".join(parts) if parts else "the relevant task state is observed"


def atomize_package(
    package: TrajectoryKnowledgePackageV3 | Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Split one trajectory package into reusable task and action knowledge."""

    typed = (
        package
        if isinstance(package, TrajectoryKnowledgePackageV3)
        else TrajectoryKnowledgePackageV3(package)
    )
    payload = typed.to_dict()
    overall_goal = payload["task_strategy"]["overall_goal"]
    task_units: list[dict[str, Any]] = []
    action_units: list[dict[str, Any]] = []
    for subtask in payload["task_strategy"]["subtasks"]:
        actions = subtask["action_knowledge"]
        summary = {
            verdict: sum(action["statistics"][verdict] for action in actions)
            for verdict in ("support", "oppose", "unverified")
        }
        basis = subtask["selection_basis"]
        basis_summary = (
            "observed facts: "
            + "; ".join(basis["observed_facts"])
            + "; inferred rationale: "
            + basis["inferred_rationale"]
        )
        task_units.append(
            SubtaskKnowledgeV3(
                {
                    "condition": {
                        "overall_goal": overall_goal,
                        "task_state": _state_summary(subtask["before_state"]),
                        "relevant_relations": subtask["before_state"].get(
                            "relevant_relations", []
                        ),
                    },
                    "subtask_strategy": {
                        "subtask": subtask["subtask"],
                        "purpose": subtask["purpose"],
                        "selection_basis_summary": basis_summary,
                        "completion_condition": subtask["completion_condition"],
                        "planned_next_subtask": subtask["planned_next_subtask"],
                    },
                    "evidence_summary": summary,
                    "status": _status_from_counts(summary),
                }
            ).to_dict()
        )
        for action in actions:
            condition = {
                "action": action["action"],
                "object_description": action["applicability_condition"].get(
                    "object", "the relevant object"
                ),
            }
            for key in ("held_state", "support_relation", "target_relation"):
                if key in action["applicability_condition"]:
                    condition[key] = action["applicability_condition"][key]
            action_units.append(
                ActionKnowledgeV3(
                    {
                        "condition": condition,
                        "geometric_strategy": action["geometric_strategy"],
                        "expected_effect": action["expected_effect"],
                        "evidence_summary": action["statistics"],
                        "status": action["status"],
                    }
                ).to_dict()
            )
    return {"task_knowledge": task_units, "action_knowledge": action_units}


def _nullable_string() -> dict[str, Any]:
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _object_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _string_array_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "maxItems": 32}


def trajectory_knowledge_package_json_schema() -> dict[str, Any]:
    """Provider-compatible strict schema for one hierarchical reflector call."""

    temporal_state = _object_schema(
        {
            "held_state": _nullable_string(),
            "support_relation": _nullable_string(),
            "target_relation": _nullable_string(),
            "relevant_relations": _string_array_schema(),
        }
    )
    subtask_state = _object_schema(
        {
            "object": _nullable_string(),
            "held_state": _nullable_string(),
            "support_relation": _nullable_string(),
            "task_progress": _nullable_string(),
            "relevant_relations": _string_array_schema(),
        }
    )
    applicability = _object_schema(
        {
            "object": _nullable_string(),
            "held_state": _nullable_string(),
            "support_relation": _nullable_string(),
            "target_relation": _nullable_string(),
            "manipulation_state": _nullable_string(),
        }
    )
    geometry = _object_schema(
        {
            "approach_direction": _nullable_string(),
            "approach_reference": _nullable_string(),
            "interaction_region": _nullable_string(),
            "orientation_relation": _nullable_string(),
            "placement_relation": _nullable_string(),
            "contact_relation": _nullable_string(),
            "clearance_or_support_constraint": _nullable_string(),
            "avoid": _string_array_schema(),
        }
    )
    expected_effect = _object_schema(
        {
            "physical_effect": {"type": "string"},
            "verification_observation": {"type": "string"},
        }
    )
    evidence = _object_schema(
        {
            "before_state": temporal_state,
            "executed_action": _object_schema(
                {
                    "action": {"type": "string", "enum": sorted(_ACTIONS)},
                    "executed_arm": _nullable_string(),
                    "observed_strategy": {"type": "string"},
                }
            ),
            "execution_status": {
                "type": "string",
                "enum": sorted(_EXECUTION_STATUSES),
            },
            "after_state": temporal_state,
            "observed_result": {"type": "string"},
            "verdict": {"type": "string", "enum": sorted(_VERDICTS)},
            "missing_evidence": _nullable_string(),
            "evidence_timing": {
                "type": "string",
                "enum": sorted(_EVIDENCE_TIMINGS),
            },
            "verifier_source": {"type": "string"},
        }
    )
    action = _object_schema(
        {
            "action": {"type": "string", "enum": sorted(_ACTIONS)},
            "applicability_condition": applicability,
            "geometric_strategy": geometry,
            "strategy_rationale": _object_schema(
                {
                    "rationale": {"type": "string"},
                    "source": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": sorted(_RATIONALE_STATUSES),
                    },
                }
            ),
            "expected_effect": expected_effect,
            "evidence": {"type": "array", "items": evidence, "minItems": 1},
            "statistics": _object_schema(
                {
                    "support": {"type": "integer", "minimum": 0},
                    "oppose": {"type": "integer", "minimum": 0},
                    "unverified": {"type": "integer", "minimum": 0},
                    "independent_verified_trials": {
                        "type": "integer",
                        "minimum": 0,
                    },
                }
            ),
            "status": {"type": "string", "enum": sorted(_KNOWLEDGE_STATUSES)},
        }
    )
    subtask = _object_schema(
        {
            "subtask": {"type": "string"},
            "purpose": {"type": "string"},
            "before_state": subtask_state,
            "selection_basis": _object_schema(
                {
                    "observed_facts": {**_string_array_schema(), "minItems": 1},
                    "inferred_rationale": {"type": "string"},
                    "source": {"type": "string"},
                }
            ),
            "completion_condition": {"type": "string"},
            "action_knowledge": {
                "type": "array",
                "items": action,
                "minItems": 1,
            },
            "planned_next_subtask": _nullable_string(),
        }
    )
    stable_object = _object_schema(
        {
            "description": {"type": "string"},
            "shape": _nullable_string(),
            "category": _nullable_string(),
            "role": _nullable_string(),
            "color": _nullable_string(),
        }
    )
    schema = _object_schema(
        {
            "trajectory_context": _object_schema(
                {
                    "task": {"type": "string"},
                    "objects": {
                        "type": "array",
                        "items": stable_object,
                        "minItems": 1,
                    },
                }
            ),
            "task_strategy": _object_schema(
                {
                    "overall_goal": {"type": "string"},
                    "plan_summary": {"type": "string"},
                    "subtasks": {
                        "type": "array",
                        "items": subtask,
                        "minItems": 1,
                    },
                }
            ),
            "trajectory_outcome": _object_schema(
                {
                    "status": {"type": "string"},
                    "reason": {"type": "string"},
                }
            ),
        }
    )

    def require_natural_values(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "string" and "enum" not in value:
                value["pattern"] = r"^[^_]+$"
            for child in value.values():
                require_natural_values(child)
        elif isinstance(value, list):
            for child in value:
                require_natural_values(child)

    require_natural_values(schema)
    return schema


__all__ = [
    "HPKV3ValidationError",
    "ActionEvidenceV31",
    "ActionEvidenceV3",
    "ActionKnowledgeCandidateV3",
    "ActionKnowledgeV3",
    "RealizationHypothesisV31",
    "RuntimeFeasibilityReportV31",
    "SubtaskGoalContractV31",
    "SubtaskKnowledgeV3",
    "SubtaskPackageV3",
    "TrajectoryKnowledgePackageV3",
    "atomize_package",
    "subtask_goal_contract_json_schema",
    "trajectory_knowledge_package_json_schema",
]

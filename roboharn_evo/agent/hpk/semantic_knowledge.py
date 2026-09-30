from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from roboharn_evo.agent.hpk.schemas import (
    GEOMETRIC_STRATEGY_SCHEMA,
    AbstractEffectV1,
    HPKValidationError,
    ConditionV1,
    EntryV1,
    GeometricStrategyV1,
    TaskStrategyV1,
)

_KNOWLEDGE_FIELDS = {
    "object",
    "condition",
    "task_strategy",
    "geometric_strategy",
    "reasoning",
    "expected_effect",
    "evidence",
    "statistics",
    "status",
}
_OBJECT_FIELDS = {"category", "color", "shape", "role", "held_state"}
_REASONING_FIELDS = {
    "observed_problem",
    "failure_analysis",
    "strategy_rationale",
    "causal_hypothesis",
    "expected_observation",
    "failure_condition",
    "source",
    "confidence",
}
_OUTCOME_REASONING_FIELDS = {
    "verdict",
    "reason",
    "missing_evidence",
    "source",
}
_VERDICTS = {"support", "oppose", "unverified"}
_STATUSES = {"candidate", "accepted", "deprecated"}
_OPERATIONS = {"grasp", "place", "contact"}
_HELD_STATES = {"held", "not held", "unknown"}
_OPAQUE_TEXT = re.compile(
    r"(?:\b(?:track|candidate)_\d+\b|\b(?:hpk|afk)[a-z]*_[0-9a-f]{16,}\b|\b[0-9a-f]{64}\b)",
    re.IGNORECASE,
)
SEMANTIC_CONTEXT_SCHEMA = "roboharn_evo/hpk/semantic_context/v1"
SEMANTIC_USAGE_SCHEMA = "roboharn_evo/hpk/semantic_usage/v1"


def _fail(path: str, message: str) -> None:
    raise HPKValidationError(f"{path}: {message}")


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return copy.deepcopy(dict(value))


def _fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    path: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        _fail(
            path, f"fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _text(value: Any, *, path: str, natural: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    result = " ".join(value.strip().split())
    if natural and "_" in result:
        _fail(path, "must use natural words with spaces, not a code enum")
    if _OPAQUE_TEXT.search(result):
        _fail(path, "must not contain an opaque runtime or content identifier")
    return result


def _number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        _fail(path, "must be between 0 and 1")
    return result


def _reject_identity_fields(value: Any, *, path: str = "knowledge") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if (
                key_text == "id"
                or key_text.endswith(("_id", "_ids"))
                or "sha256" in key_text
                or "hash" in key_text
                or "fingerprint" in key_text
            ):
                _fail(f"{path}.{key}", "identity and hash fields do not belong in HPK")
            _reject_identity_fields(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_identity_fields(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _OPAQUE_TEXT.search(value):
        _fail(path, "opaque runtime and content identifiers do not belong in HPK")


def _natural(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _first_text(value: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return _natural(candidate)
    return ""


def semantic_object_from_mapping(value: Mapping[str, Any]) -> dict[str, str]:
    """Project reliable structured object descriptors without preserving handles."""

    payload = _mapping(value, path="object_source")
    attributes = payload.get("attributes")
    attributes = dict(attributes) if isinstance(attributes, Mapping) else {}
    category = (
        _first_text(
            payload,
            "semantic_class",
            "category",
            "object_class",
            "class_name",
            "class",
            "source_object_id",
        )
        or "unknown"
    )
    color = _first_text(payload, "color") or _first_text(attributes, "color")
    shape = _first_text(
        payload, "shape", "geometry_class", "shape_class"
    ) or _first_text(attributes, "shape")
    role = _first_text(payload, "semantic_role", "query_role", "task_role", "role")
    held_state = _first_text(payload, "held_state") or "unknown"

    if not color:
        description = _first_text(
            payload,
            "text_prompt",
            "source_text_prompt",
            "object_description",
            "semantic_label",
            "query_text",
        )
        category_tokens = category.split()
        description_tokens = description.split()
        if (
            category != "unknown"
            and len(description_tokens) == len(category_tokens) + 1
            and description_tokens[-len(category_tokens) :] == category_tokens
        ):
            color = description_tokens[0]

    result = {
        "category": category,
        "color": color or "unknown",
        "shape": shape or "unknown",
        "role": role or "unknown",
        "held_state": held_state if held_state in _HELD_STATES else "unknown",
    }
    return validate_semantic_object(result)


def validate_semantic_object(value: Mapping[str, Any]) -> dict[str, str]:
    payload = _mapping(value, path="object")
    _fields(payload, required=_OBJECT_FIELDS, path="object")
    result = {
        key: _text(payload[key], path=f"object.{key}", natural=True).lower()
        for key in _OBJECT_FIELDS
    }
    if result["held_state"] not in _HELD_STATES:
        _fail("object.held_state", f"must be one of {sorted(_HELD_STATES)}")
    return result


def validate_semantic_reasoning(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(value, path="reasoning")
    _fields(payload, required=_REASONING_FIELDS, path="reasoning")
    result = {
        key: _text(payload[key], path=f"reasoning.{key}", natural=True)
        for key in _REASONING_FIELDS - {"confidence"}
    }
    result["confidence"] = _number(payload["confidence"], path="reasoning.confidence")
    return result


def _validate_condition(value: Mapping[str, Any]) -> dict[str, str]:
    payload = _mapping(value, path="condition")
    _fields(payload, required={"task", "phase"}, path="condition")
    return {
        "task": _text(payload["task"], path="condition.task", natural=True).lower(),
        "phase": _text(payload["phase"], path="condition.phase", natural=True).lower(),
    }


def _validate_task_strategy(value: Mapping[str, Any]) -> dict[str, str]:
    payload = _mapping(value, path="task_strategy")
    _fields(
        payload,
        required={"operation", "preferred_arm"},
        path="task_strategy",
    )
    operation = _text(
        payload["operation"], path="task_strategy.operation", natural=True
    ).lower()
    if operation not in _OPERATIONS:
        _fail("task_strategy.operation", f"must be one of {sorted(_OPERATIONS)}")
    return {
        "operation": operation,
        "preferred_arm": _text(
            payload["preferred_arm"],
            path="task_strategy.preferred_arm",
            natural=True,
        ).lower(),
    }


def _validate_expected_effect(value: Mapping[str, Any]) -> dict[str, bool]:
    payload = _mapping(value, path="expected_effect")
    if not payload or any(not isinstance(key, str) for key in payload):
        _fail("expected_effect", "must contain at least one named effect")
    if any(not isinstance(item, bool) for item in payload.values()):
        _fail("expected_effect", "effect values must be booleans")
    return dict(payload)


def validate_semantic_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(value, path="evidence")
    _fields(
        payload,
        required={
            "object",
            "action",
            "observed_result",
            "verdict",
            "outcome_reasoning",
        },
        path="evidence",
    )
    verdict = _text(payload["verdict"], path="evidence.verdict", natural=True).lower()
    if verdict not in _VERDICTS:
        _fail("evidence.verdict", f"must be one of {sorted(_VERDICTS)}")
    outcome = _mapping(payload["outcome_reasoning"], path="evidence.outcome_reasoning")
    _fields(
        outcome,
        required=_OUTCOME_REASONING_FIELDS,
        path="evidence.outcome_reasoning",
    )
    result = {
        "object": validate_semantic_object(payload["object"]),
        "action": _text(payload["action"], path="evidence.action", natural=True),
        "observed_result": _text(
            payload["observed_result"],
            path="evidence.observed_result",
            natural=True,
        ),
        "verdict": verdict,
        "outcome_reasoning": {
            key: _text(
                outcome[key],
                path=f"evidence.outcome_reasoning.{key}",
                natural=True,
            )
            for key in _OUTCOME_REASONING_FIELDS
        },
    }
    if result["outcome_reasoning"]["verdict"].lower() != verdict:
        _fail(
            "evidence.outcome_reasoning.verdict",
            "must agree with evidence.verdict",
        )
    result["outcome_reasoning"]["verdict"] = verdict
    _reject_identity_fields(result, path="evidence")
    return result


def validate_semantic_knowledge(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(value, path="knowledge")
    _fields(payload, required=_KNOWLEDGE_FIELDS, path="knowledge")
    _reject_identity_fields(payload)

    condition = _validate_condition(payload["condition"])
    task_strategy = _validate_task_strategy(payload["task_strategy"])
    operation = task_strategy["operation"]

    geometry = _mapping(payload["geometric_strategy"], path="geometric_strategy")
    _fields(
        geometry,
        required={"approach", "orientation"},
        optional={"grasp_region", "contact_region", "target_relation"},
        path="geometric_strategy",
    )
    required_region = {
        "grasp": "grasp_region",
        "contact": "contact_region",
        "place": "target_relation",
    }[operation]
    if required_region not in geometry:
        _fail("geometric_strategy", f"{operation} requires {required_region}")
    approach = _mapping(geometry["approach"], path="geometric_strategy.approach")
    _fields(
        approach,
        required={"direction", "reference"},
        path="geometric_strategy.approach",
    )
    orientation = _mapping(
        geometry["orientation"], path="geometric_strategy.orientation"
    )
    _fields(
        orientation,
        required={"relation", "reference", "order"},
        path="geometric_strategy.orientation",
    )
    natural_geometry: dict[str, Any] = {
        "approach": {
            key: _text(
                approach[key],
                path=f"geometric_strategy.approach.{key}",
                natural=True,
            ).lower()
            for key in ("direction", "reference")
        },
        "orientation": {
            key: _text(
                orientation[key],
                path=f"geometric_strategy.orientation.{key}",
                natural=True,
            ).lower()
            for key in ("relation", "reference", "order")
        },
    }
    for key in ("grasp_region", "contact_region", "target_relation"):
        if key in geometry:
            natural_geometry[key] = _text(
                geometry[key], path=f"geometric_strategy.{key}", natural=True
            ).lower()

    expected = _validate_expected_effect(payload["expected_effect"])

    evidence_source = payload["evidence"]
    if not isinstance(evidence_source, list):
        _fail("evidence", "must be an array")
    evidence = [validate_semantic_evidence(item) for item in evidence_source]
    statistics = _mapping(payload["statistics"], path="statistics")
    _fields(
        statistics,
        required={"support", "oppose", "unverified"},
        path="statistics",
    )
    normalized_statistics: dict[str, int] = {}
    for verdict in _VERDICTS:
        count = statistics[verdict]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail(f"statistics.{verdict}", "must be a non-negative integer")
        actual = sum(item["verdict"] == verdict for item in evidence)
        if count != actual:
            _fail(f"statistics.{verdict}", f"must equal evidence count {actual}")
        normalized_statistics[verdict] = count

    status = _text(payload["status"], path="status", natural=True).lower()
    if status not in _STATUSES:
        _fail("status", f"must be one of {sorted(_STATUSES)}")

    result = {
        "object": validate_semantic_object(payload["object"]),
        "condition": condition,
        "task_strategy": task_strategy,
        "geometric_strategy": natural_geometry,
        "reasoning": validate_semantic_reasoning(payload["reasoning"]),
        "expected_effect": dict(expected),
        "evidence": evidence,
        "statistics": normalized_statistics,
        "status": status,
    }
    _reject_identity_fields(result)
    return result


def validate_semantic_query(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the semantic fields available at retrieval time."""

    payload = _mapping(value, path="query")
    _fields(
        payload,
        required={"object", "condition", "task_strategy", "expected_effect"},
        path="query",
    )
    _reject_identity_fields(payload, path="query")
    return {
        "object": validate_semantic_object(payload["object"]),
        "condition": _validate_condition(payload["condition"]),
        "task_strategy": _validate_task_strategy(payload["task_strategy"]),
        "expected_effect": _validate_expected_effect(payload["expected_effect"]),
    }


def validate_semantic_context(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(value, path="semantic_context")
    _fields(
        payload,
        required={"schema", "knowledge"},
        path="semantic_context",
    )
    if payload["schema"] not in {SEMANTIC_CONTEXT_SCHEMA, "tcm/afk/semantic_context/v1"}:
        _fail("semantic_context.schema", f"must be {SEMANTIC_CONTEXT_SCHEMA}")
    records = payload["knowledge"]
    if not isinstance(records, list) or not records:
        _fail("semantic_context.knowledge", "must be a non-empty array")
    return {
        "schema": SEMANTIC_CONTEXT_SCHEMA,
        "knowledge": [validate_semantic_knowledge(value) for value in records],
    }


def render_semantic_context(value: Mapping[str, Any]) -> str:
    context = validate_semantic_context(value)
    body = json.dumps(context["knowledge"], ensure_ascii=False, separators=(",", ":"))
    return f"<hierarchical_physical_knowledge>\n{body}\n</hierarchical_physical_knowledge>"


def semantic_usage_receipt(
    value: Mapping[str, Any], *, expected_context: Mapping[str, Any]
) -> dict[str, Any]:
    context = validate_semantic_context(expected_context)
    payload = _mapping(value, path="semantic_usage")
    _fields(
        payload,
        required={"schema", "rendered", "knowledge_count", "usage"},
        path="semantic_usage",
    )
    expected = {
        "schema": SEMANTIC_USAGE_SCHEMA,
        "rendered": True,
        "knowledge_count": len(context["knowledge"]),
        "usage": "injected; behavioral effect unverified",
    }
    if payload["schema"] == "tcm/afk/semantic_usage/v1":
        payload["schema"] = SEMANTIC_USAGE_SCHEMA
    if payload != expected:
        _fail("semantic_usage", "does not describe the supplied semantic context")
    return expected


def _natural_approach_direction(value: str) -> str:
    return {
        "above": "from above",
        "below": "from below",
        "lateral": "from the side",
        "oblique": "at an angle",
    }.get(value, "unknown")


def _natural_reference_frame(value: str) -> str:
    return {
        "object_principal_axes": "object frame",
        "support_normal": "support surface",
        "current_attachment": "current attachment",
        "world_gravity": "gravity",
    }.get(value, "unknown")


def _natural_orientation(value: str) -> dict[str, str]:
    if value == "align_principal_axis_0":
        return {"relation": "align", "reference": "principal axis", "order": "first"}
    if value == "align_principal_axis_1":
        return {
            "relation": "align",
            "reference": "principal axis",
            "order": "second",
        }
    if value == "preserve_current_attachment":
        return {
            "relation": "preserve",
            "reference": "current attachment",
            "order": "not applicable",
        }
    return {"relation": "unconstrained", "reference": "unknown", "order": "unknown"}


def _typed_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return copy.deepcopy(dict(payload))
    _fail(path, "must be a structured record")


def _semantic_geometry(
    value: GeometricStrategyV1 | Mapping[str, Any], *, operation: str
) -> dict[str, Any]:
    geometry = _typed_mapping(value, path="geometric_strategy")
    approach = _mapping(geometry.get("approach"), path="geometric_strategy.approach")
    result: dict[str, Any] = {
        "approach": {
            "direction": _natural_approach_direction(
                str(approach.get("direction_bucket") or "")
            ),
            "reference": _natural_reference_frame(
                str(geometry.get("reference_frame") or "")
            ),
        },
        "orientation": _natural_orientation(
            str(
                _mapping(geometry.get("orientation"), path="orientation").get(
                    "relation"
                )
                or ""
            )
        ),
    }
    grasp = _mapping(geometry.get("grasp"), path="geometric_strategy.grasp")
    target = _mapping(
        geometry.get("target_relation"), path="geometric_strategy.target_relation"
    )
    if operation == "grasp":
        result["grasp_region"] = _natural(grasp.get("region") or "unknown")
    elif operation == "contact":
        result["contact_region"] = _natural(grasp.get("region") or "unknown")
    else:
        result["target_relation"] = _natural(target.get("relation") or "unknown")
    return result


def semantic_evidence_from_transition(
    transition: Mapping[str, Any],
    *,
    object_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe one action outcome without carrying rollout identity."""

    payload = _mapping(transition, path="transition")
    verdict = _natural(payload.get("evidence_verdict") or "unverified")
    if verdict not in _VERDICTS:
        verdict = "unverified"
    operation = _natural(payload.get("operation") or "action")
    arm = _natural(payload.get("arm") or "unknown")
    object_value = validate_semantic_object(object_semantics)
    object_words = [
        value
        for value in (object_value["color"], object_value["category"])
        if value != "unknown"
    ]
    object_name = " ".join(object_words) or "object"
    action = f"{operation} the {object_name} with the {arm} arm"

    observed = payload.get("observed_effect")
    observed_payload = (
        _typed_mapping(observed, path="transition.observed_effect")
        if observed is not None
        else {}
    )
    predicates = observed_payload.get("observed_predicates")
    if isinstance(predicates, list) and predicates:
        result_words = ", ".join(_natural(value) for value in predicates)
        observed_result = f"observed {result_words}"
    else:
        observed_result = "no deterministic effect was confirmed"

    reasons = payload.get("attribution_reasons")
    reason_words = (
        [_natural(value) for value in reasons] if isinstance(reasons, list) else []
    )
    if verdict == "support":
        reason = "deterministic post action evidence confirmed the expected effect"
        missing = "none"
    elif verdict == "oppose":
        reason = "deterministic post action evidence contradicted the expected effect"
        missing = "none"
    else:
        reason = (
            "; ".join(reason_words)
            if reason_words
            else "the available observation did not prove the action effect"
        )
        missing = (
            "; ".join(reason_words)
            if reason_words
            else "fresh deterministic post action evidence"
        )
    sources = payload.get("verifier_sources")
    source_words = (
        [_natural(value) for value in sources] if isinstance(sources, list) else []
    )
    source = ", ".join(source_words) or "runtime verifier"
    return validate_semantic_evidence(
        {
            "object": object_value,
            "action": action,
            "observed_result": observed_result,
            "verdict": verdict,
            "outcome_reasoning": {
                "verdict": verdict,
                "reason": reason,
                "missing_evidence": missing,
                "source": source,
            },
        }
    )


def semantic_evidence_from_runtime_verification(
    verification: Mapping[str, Any],
    *,
    object_semantics: Mapping[str, Any],
    operation: str,
    arm: str,
) -> dict[str, Any]:
    """Project a delayed deterministic verifier result back to its action."""

    payload = _mapping(verification, path="runtime_verification")
    operation_value = _natural(operation)
    arm_value = _natural(arm)
    if operation_value == "grasp":
        runtime = _mapping(
            payload.get("runtime_grasp_validation"),
            path="runtime_grasp_validation",
        )
        verified = runtime.get("verified")
        negative = bool(
            runtime.get("negative_attachment_evidence") is True
            or runtime.get("fresh_visible_negative_evidence") is True
        )
        camera_count = runtime.get("supporting_camera_count")
        if verified is True:
            verdict = "support"
            observed_result = "the object moved with the lifted gripper"
            reason = (
                f"object motion matched gripper motion in {camera_count} calibrated views"
                if isinstance(camera_count, int) and camera_count > 0
                else "deterministic attachment validation confirmed coupled motion"
            )
            missing = "none"
        elif verified is False and negative:
            verdict = "oppose"
            observed_result = "the object did not move with the lifted gripper"
            reason = "fresh deterministic observation contradicted attachment"
            missing = "none"
        else:
            verdict = "unverified"
            observed_result = "attachment could not be confirmed"
            reason = "the deterministic attachment check was inconclusive"
            missing = "independent views that clearly show object motion"
        source = "deterministic runtime grasp verifier"
    elif operation_value == "place":
        runtime = _mapping(
            payload.get("runtime_place_validation"),
            path="runtime_place_validation",
        )
        if runtime.get("verified") is True:
            verdict = "support"
            observed_result = "the object was released and remained at the target"
            reason = "fresh runtime state confirmed release and target placement"
            missing = "none"
        elif runtime.get("verified") is False and (
            runtime.get("placement_recovery_required") is True
            or runtime.get("object_at_target") is False
        ):
            verdict = "oppose"
            observed_result = "the released object was outside the target"
            reason = "fresh runtime state contradicted the requested placement"
            missing = "none"
        else:
            verdict = "unverified"
            observed_result = "placement could not be confirmed"
            reason = "the deterministic placement check was inconclusive"
            missing = "fresh object position and release confirmation"
        source = "deterministic runtime place verifier"
    else:
        _fail("runtime_verification.operation", "must be grasp or place")

    object_value = validate_semantic_object(object_semantics)
    object_words = [
        value
        for value in (object_value["color"], object_value["category"])
        if value != "unknown"
    ]
    object_name = " ".join(object_words) or "object"
    return validate_semantic_evidence(
        {
            "object": object_value,
            "action": f"{operation_value} the {object_name} with the {arm_value} arm",
            "observed_result": observed_result,
            "verdict": verdict,
            "outcome_reasoning": {
                "verdict": verdict,
                "reason": reason,
                "missing_evidence": missing,
                "source": source,
            },
        }
    )


def semantic_knowledge_from_strategy(
    *,
    condition: ConditionV1 | Mapping[str, Any],
    task_strategy: TaskStrategyV1 | Mapping[str, Any],
    geometric_strategy: GeometricStrategyV1 | Mapping[str, Any],
    expected_effect: AbstractEffectV1 | Mapping[str, Any],
    object_semantics: Mapping[str, Any],
    reasoning: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]] = (),
    status: str = "candidate",
) -> dict[str, Any]:
    """Build the complete semantic knowledge record from typed strategy data."""

    condition_value = _typed_mapping(condition, path="condition")
    task_value = _typed_mapping(task_strategy, path="task_strategy")
    effect_value = _typed_mapping(expected_effect, path="expected_effect")
    operation = _natural(task_value.get("operation"))
    expected = {
        str(predicate): True
        for predicate in effect_value.get("expected_predicates", [])
    }
    evidence_values: list[dict[str, Any]] = []
    for value in evidence:
        typed_evidence = validate_semantic_evidence(value)
        if typed_evidence not in evidence_values:
            evidence_values.append(typed_evidence)
    return validate_semantic_knowledge(
        {
            "object": validate_semantic_object(object_semantics),
            "condition": {
                "task": _natural(condition_value.get("task_family")),
                "phase": _natural(condition_value.get("manipulation_phase")),
            },
            "task_strategy": {
                "operation": operation,
                "preferred_arm": (
                    "either arm"
                    if task_value.get("preferred_arm") == "either"
                    else f"{_natural(task_value.get('preferred_arm'))} arm"
                ),
            },
            "geometric_strategy": _semantic_geometry(
                geometric_strategy, operation=operation
            ),
            "reasoning": validate_semantic_reasoning(reasoning),
            "expected_effect": expected,
            "evidence": evidence_values,
            "statistics": {
                verdict: sum(item["verdict"] == verdict for item in evidence_values)
                for verdict in _VERDICTS
            },
            "status": status,
        }
    )


def semantic_geometry_to_v1(
    knowledge: Mapping[str, Any],
) -> GeometricStrategyV1:
    """Translate natural semantic geometry only at the robot ranking boundary."""

    typed = validate_semantic_knowledge(knowledge)
    geometry = typed["geometric_strategy"]
    operation = typed["task_strategy"]["operation"]
    orientation = geometry["orientation"]
    orientation_relation = "unconstrained"
    if (
        orientation["relation"] == "align"
        and orientation["reference"] == "principal axis"
        and orientation["order"] in {"first", "second"}
    ):
        orientation_relation = (
            "align_principal_axis_0"
            if orientation["order"] == "first"
            else "align_principal_axis_1"
        )
    elif (
        orientation["relation"] == "preserve"
        and orientation["reference"] == "current attachment"
    ):
        orientation_relation = "preserve_current_attachment"
    direction = {
        "from above": "above",
        "from below": "below",
        "from the side": "lateral",
        "at an angle": "oblique",
    }.get(geometry["approach"]["direction"], "unknown")
    reference = {
        "object frame": "object_principal_axes",
        "support surface": "support_normal",
        "current attachment": "current_attachment",
        "gravity": "world_gravity",
    }.get(geometry["approach"]["reference"], "unknown")
    strategy_family = {
        "grasp": "observed_grasp_geometry",
        "place": "placement_relation",
        "contact": "contact_relation",
    }[operation]
    approach_family = (
        "principal_axis_relative"
        if orientation_relation.startswith("align_principal_axis_")
        else "clearance_first"
    )
    region_key = {
        "grasp": "grasp_region",
        "contact": "contact_region",
    }.get(operation)
    natural_region = (
        str(geometry.get(region_key, "unknown")) if region_key else "unknown"
    )
    region = {
        "visible surface": "observed_surface",
        "observed surface": "observed_surface",
    }.get(natural_region, natural_region.replace(" ", "_"))
    target_relation = (
        str(geometry.get("target_relation", "unknown")).replace(" ", "_")
        if operation == "place"
        else None
    )
    return GeometricStrategyV1.from_dict(
        {
            "schema": GEOMETRIC_STRATEGY_SCHEMA,
            "strategy_family": strategy_family,
            "reference_frame": reference,
            "target_relation": {
                "relation": target_relation,
                "reference_role": None,
            },
            "approach": {
                "family": approach_family,
                "direction_bucket": direction,
            },
            "orientation": {"relation": orientation_relation},
            "grasp": {"region": region, "semantic_part": None},
            "hard_constraints": [],
            "soft_preferences": [],
            "avoid": [],
            "capability_evidence": {
                "geometry_source_class": (
                    "rgbd_observed"
                    if operation in {"grasp", "contact"}
                    else "runtime_relational"
                ),
                "semantic_part_observed": False,
            },
        }
    )


def migrate_entry_v1_to_semantic(
    entry: EntryV1 | Mapping[str, Any],
    *,
    object_semantics: Mapping[str, Any],
    reasoning: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """One-way legacy migration; opaque legacy identity is deliberately discarded."""

    typed = entry if isinstance(entry, EntryV1) else EntryV1.from_dict(entry)
    task = typed["task_strategy"]
    geometry = typed["geometric_strategy"]
    operation = str(task["operation"])
    approach = geometry["approach"]
    migrated_geometry: dict[str, Any] = {
        "approach": {
            "direction": _natural_approach_direction(str(approach["direction_bucket"])),
            "reference": _natural_reference_frame(str(geometry["reference_frame"])),
        },
        "orientation": _natural_orientation(str(geometry["orientation"]["relation"])),
    }
    if operation == "grasp":
        migrated_geometry["grasp_region"] = _natural(geometry["grasp"]["region"])
    elif operation == "contact":
        migrated_geometry["contact_region"] = _natural(geometry["grasp"]["region"])
    else:
        migrated_geometry["target_relation"] = _natural(
            geometry["target_relation"]["relation"] or "unknown"
        )

    migrated_evidence = [validate_semantic_evidence(item) for item in evidence]
    statistics = {
        verdict: sum(item["verdict"] == verdict for item in migrated_evidence)
        for verdict in _VERDICTS
    }
    expected = {
        str(predicate): True
        for predicate in typed["expected_effect"]["expected_predicates"]
    }
    return validate_semantic_knowledge(
        {
            "object": validate_semantic_object(object_semantics),
            "condition": {
                "task": _natural(typed["condition"]["task_family"]),
                "phase": _natural(typed["condition"]["manipulation_phase"]),
            },
            "task_strategy": {
                "operation": operation,
                "preferred_arm": (
                    "either arm"
                    if task["preferred_arm"] == "either"
                    else f"{task['preferred_arm']} arm"
                ),
            },
            "geometric_strategy": migrated_geometry,
            "reasoning": validate_semantic_reasoning(reasoning),
            "expected_effect": expected,
            "evidence": migrated_evidence,
            "statistics": statistics,
            "status": typed["status"],
        }
    )


__all__ = [
    "migrate_entry_v1_to_semantic",
    "render_semantic_context",
    "semantic_evidence_from_runtime_verification",
    "semantic_evidence_from_transition",
    "semantic_geometry_to_v1",
    "semantic_knowledge_from_strategy",
    "semantic_object_from_mapping",
    "semantic_usage_receipt",
    "validate_semantic_context",
    "validate_semantic_evidence",
    "validate_semantic_knowledge",
    "validate_semantic_object",
    "validate_semantic_query",
    "validate_semantic_reasoning",
]

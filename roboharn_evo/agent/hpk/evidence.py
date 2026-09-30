from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from roboharn_evo.agent.hpk.schemas import (
    EVIDENCE_SCHEMA,
    AbstractEffectV1,
    ConditionV1,
    EvidenceV1,
    GeometricStrategyV1,
    TaskStrategyV1,
    evidence_id_for,
    validate_content_id,
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
    if value.get("verified") is True:
        return "verified"
    return "unverified"


def _validation_conflict(
    runtime_validation: Mapping[str, Any] | None,
    verifier_result: Mapping[str, Any] | None,
) -> bool:
    runtime = _mapping(runtime_validation)
    components = [runtime]
    for key in (
        "runtime_grasp_validation",
        "grasp_validation",
        "runtime_place_validation",
        "place_validation",
    ):
        nested = _mapping(runtime.get(key))
        if nested:
            components.append(nested)
    statuses: list[str] = []
    for component in components:
        status = _overall_status(component)
        if status != "unverified":
            statuses.append(status)
        if (
            component.get("object_position_observed") is True
            and component.get("object_position_fresh") is True
            and component.get("object_at_target") is False
        ):
            statuses.append("contradicted")
        if component.get("verified") is False and (
            component.get("negative_attachment_evidence") is True
            or component.get("fresh_visible_negative_evidence") is True
        ):
            statuses.append("contradicted")
    distinct_runtime = set(statuses)
    if len(distinct_runtime) > 1:
        return True
    runtime_status = statuses[0] if statuses else "unverified"
    verifier_status = _overall_status(_mapping(verifier_result))
    return bool(
        runtime_status != "unverified"
        and verifier_status != "unverified"
        and runtime_status != verifier_status
    )


def determine_evidence_verdict(
    expected_effect: AbstractEffectV1 | Mapping[str, Any],
    observed_effect: AbstractEffectV1 | Mapping[str, Any],
    *,
    realization_status: str,
    motion_status: str,
    runtime_validation: Mapping[str, Any] | None = None,
    verifier_result: Mapping[str, Any] | None = None,
) -> str:
    """Return support/oppose/unverified under the frozen evidence rules."""

    expected = (
        expected_effect
        if isinstance(expected_effect, AbstractEffectV1)
        else AbstractEffectV1.from_dict(expected_effect)
    )
    observed = (
        observed_effect
        if isinstance(observed_effect, AbstractEffectV1)
        else AbstractEffectV1.from_dict(observed_effect)
    )
    if (
        str(motion_status or "").strip().lower() != "completed"
        or str(realization_status or "").strip().lower() != "satisfied"
        or _validation_conflict(runtime_validation, verifier_result)
    ):
        return "unverified"
    expected_predicates = set(expected["expected_predicates"])
    observed_predicates = set(observed.to_dict().get("observed_predicates", []))
    verifiability = observed["verifiability"]
    if verifiability == "contradicted":
        return "oppose"
    if verifiability == "verified" and expected_predicates <= observed_predicates:
        return "support"
    return "unverified"


def _stable_ref(value: Any, *, kind: str) -> str:
    if kind == "condition" and isinstance(value, ConditionV1):
        return value.stable_id
    if kind == "task_strategy" and isinstance(value, TaskStrategyV1):
        return value.stable_id
    if kind == "geometric_strategy" and isinstance(value, GeometricStrategyV1):
        return value.stable_id
    if kind == "expected_effect" and isinstance(value, AbstractEffectV1):
        return value.stable_id
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{kind}_id must be provided")
    prefixes = {
        "condition": "afkc",
        "task_strategy": "afku",
        "geometric_strategy": "afkz",
        "expected_effect": "afkfx",
    }
    return validate_content_id(
        text,
        prefix=prefixes[kind],
        path=f"{kind}_id",
    )


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def build_action_effect_evidence(
    *,
    episode_ref: str,
    step_or_segment_ref: str,
    condition: ConditionV1 | str,
    task_strategy: TaskStrategyV1 | str,
    geometric_strategy: GeometricStrategyV1 | str,
    expected_effect: AbstractEffectV1,
    observed_effect: AbstractEffectV1,
    realization_status: str,
    motion_status: str,
    geometry_source_class: str,
    selected_candidate_private_ref: str | None = None,
    trace_event_refs: list[str] | tuple[str, ...] = (),
    verifier_summary: str = "",
    runtime_validation_summary: Mapping[str, Any] | None = None,
    verifier_result: Mapping[str, Any] | None = None,
    confidence: float | None = None,
    created_at: str | None = None,
) -> EvidenceV1:
    runtime_summary = _mapping(runtime_validation_summary)
    verdict = determine_evidence_verdict(
        expected_effect,
        observed_effect,
        realization_status=realization_status,
        motion_status=motion_status,
        runtime_validation=runtime_summary,
        verifier_result=verifier_result,
    )
    if confidence is None:
        confidence = 1.0 if verdict in {"support", "oppose"} else 0.0
    observed_payload = observed_effect.to_dict()
    payload: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "evidence_id": "pending",
        "episode_ref": str(episode_ref),
        "step_or_segment_ref": str(step_or_segment_ref),
        "condition_id": _stable_ref(condition, kind="condition"),
        "task_strategy_id": _stable_ref(task_strategy, kind="task_strategy"),
        "geometric_strategy_id": _stable_ref(
            geometric_strategy,
            kind="geometric_strategy",
        ),
        "expected_effect_id": _stable_ref(
            expected_effect,
            kind="expected_effect",
        ),
        "observed_effect": {
            "effect_type": observed_payload["effect_type"],
            "predicates": list(observed_payload.get("observed_predicates", [])),
        },
        "verdict": verdict,
        "confidence": confidence,
        "realization_status": str(realization_status),
        "motion_status": str(motion_status),
        "provenance": {
            "geometry_source_class": str(geometry_source_class),
            "selected_candidate_private_ref": selected_candidate_private_ref,
            "trace_event_refs": list(trace_event_refs),
            "verifier_summary": str(verifier_summary),
            "runtime_validation_summary": copy.deepcopy(runtime_summary),
        },
        "created_at": created_at or _utc_now(),
    }
    payload["evidence_id"] = evidence_id_for(payload)
    return EvidenceV1.from_dict(payload)


build_evidence = build_action_effect_evidence
evidence_verdict = determine_evidence_verdict


__all__ = [
    "build_action_effect_evidence",
    "build_evidence",
    "determine_evidence_verdict",
    "evidence_verdict",
]

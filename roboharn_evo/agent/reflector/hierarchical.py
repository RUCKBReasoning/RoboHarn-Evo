"""Pure contracts and orchestration for hierarchical trajectory reflection.

Phase A2.1 separates semantic reflection into four independently validated
model calls:

``action chunk -> subtask summary -> procedure draft -> attribution``.

This module owns no transport, image decoding, filesystem access, or model
provider logic.  Every backend response is untrusted JSON.  Locally admitted
facts are always episode-private, and no draft becomes planner-visible until an
independent attribution pass and the existing :class:`ReflectionQualityGate`
both accept it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .quality import (
    DeterministicLeakageCritic,
    ReflectionQualityGate,
)
from .schemas import ProvenanceV1, SchemaValidationError
from .whole_trajectory import (
    UntrustedReflectionOutput,
    WholeTrajectoryReflectionRequest,
    WholeTrajectoryReflectionResult,
    annotation_evidence_ref,
)


_ACTION_REQUEST_SCHEMA = "roboharn_evo/action_chunk_reflection_request/v1"
_ACTION_OUTPUT_SCHEMA = "roboharn_evo/action_chunk_reflection_output/v1"
_ACTION_BATCH_REQUEST_SCHEMA = "roboharn_evo/action_chunk_batch_reflection_request/v1"
_ACTION_BATCH_OUTPUT_SCHEMA = "roboharn_evo/action_chunk_batch_reflection_output/v1"
_SUBTASK_REQUEST_SCHEMA = "roboharn_evo/subtask_summary_request/v1"
_SUBTASK_OUTPUT_SCHEMA = "roboharn_evo/subtask_summary_output/v1"
_DRAFT_REQUEST_SCHEMA = "roboharn_evo/procedure_draft_request/v1"
_DRAFT_OUTPUT_SCHEMA = "roboharn_evo/procedure_draft_output/v1"
_ATTRIBUTION_REQUEST_SCHEMA = "roboharn_evo/evidence_attribution_request/v1"
_ATTRIBUTION_OUTPUT_SCHEMA = "roboharn_evo/evidence_attribution_output/v1"
_LAYER_ABSTENTION_SCHEMA = "roboharn_evo/hierarchical_layer_abstention/v1"
_SCHEMA_VERSION = 1
_CONFIDENCE = frozenset({"low", "medium", "high"})
_TEMPORAL_SCOPES = frozenset({"single_frame", "before_after", "cross_segment"})
_SOURCE_TYPES = frozenset(
    {
        "annotation",
        "observed_robot_state",
        "observed_visual",
        "cross_segment_inference",
    }
)
_FACT_FIELDS = frozenset(
    {
        "fact_id",
        "claim",
        "source_type",
        "supporting_evidence_refs",
        "supporting_segment_refs",
        "confidence",
        "uncertainty",
        "temporal_scope",
        "transferable_allowed",
        "structured_value",
    }
)
_ATTRIBUTION_FIELDS = frozenset(
    {
        "attribution_id",
        "target_path",
        "claim",
        "source_type",
        "supporting_evidence_refs",
        "supporting_segment_refs",
        "confidence",
        "uncertainty",
        "temporal_scope",
        "prediction_status",
    }
)
_CAPABILITY_FIELDS = frozenset(
    {
        "image_input_supported",
        "image_input_acknowledged",
        "structured_output_supported",
        "text_only_fallback",
        "scripted_backend",
    }
)


def hierarchical_source_semantics() -> dict[str, str]:
    """Return the task-neutral epistemic contract for every model prompt."""

    return {
        "annotation": (
            "describes the action or variable change intended for the segment; "
            "it does not prove physical execution, stability, or task feedback"
        ),
        "observed_robot_state": (
            "describes observed arm motion, gripper state, and timing; it is not "
            "a commanded action or an exact high-level tool call"
        ),
        "observed_visual": (
            "supports only visible object state, release, and task-feedback claims"
        ),
        "cross_segment_inference": (
            "combines admitted facts from multiple ordered segments and must cite "
            "evidence spanning every declared segment"
        ),
        "trajectory_outcome": (
            "is unavailable unless an explicit source-artifact evidence reference "
            "is present; a runner-side status is never evidence"
        ),
    }


def _source_semantics_schema() -> dict[str, Any]:
    keys = tuple(hierarchical_source_semantics())
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(keys),
        "properties": {key: {"type": "string"} for key in keys},
    }


def _validate_source_semantics(value: Any, *, label: str) -> None:
    if value != hierarchical_source_semantics():
        raise SchemaValidationError(
            f"{label}: must match the immutable hierarchical source contract"
        )


def semantic_leakage_audit_contract() -> dict[str, str]:
    """Return the task-neutral contract for the independent semantic audit."""

    return {
        "scope": (
            "compare every planner-facing claim against all episode-private "
            "claims and structured values for semantic equivalence"
        ),
        "prohibited_transfer": (
            "episode-specific successful assignments, poses, identifiers, or "
            "answers must not appear in transferable guidance"
        ),
        "method_boundary": (
            "do not treat lexical overlap or any fixed phrase table as a "
            "complete semantic leakage test"
        ),
        "uncertainty_policy": (
            "if a confident negative verdict cannot be made, abstain rather "
            "than approving the draft"
        ),
    }


def _semantic_leakage_contract_schema() -> dict[str, Any]:
    keys = tuple(semantic_leakage_audit_contract())
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(keys),
        "properties": {key: {"type": "string"} for key in keys},
    }


def _validate_semantic_leakage_contract(value: Any, *, label: str) -> None:
    if value != semantic_leakage_audit_contract():
        raise SchemaValidationError(
            f"{label}: must match the immutable semantic leakage audit contract"
        )


def _strict_json(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{label}: must be strict JSON ({exc})") from exc


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{label}: must be an object")
    return _strict_json(dict(value), label=label)


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{label}: must be a non-empty string")
    return value


def _string_tuple(
    value: Any,
    *,
    label: str,
    nonempty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError(f"{label}: must be an array")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _nonempty(item, label=f"{label}[{index}]")
        if text in seen:
            raise SchemaValidationError(f"{label}[{index}]: duplicate {text!r}")
        seen.add(text)
        result.append(text)
    if nonempty and not result:
        raise SchemaValidationError(f"{label}: must not be empty")
    return tuple(result)


def _mapping_tuple(value: Any, *, label: str) -> tuple[dict[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError(f"{label}: must be an array")
    return tuple(
        _mapping(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )


def _exact_fields(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(expected) - set(payload))
    unknown = sorted(set(payload) - set(expected))
    if missing:
        raise SchemaValidationError(
            f"{label}: missing required field(s): {', '.join(missing)}"
        )
    if unknown:
        raise SchemaValidationError(f"{label}: unknown field(s): {', '.join(unknown)}")


def _schema_header(payload: Mapping[str, Any], expected: str, *, label: str) -> None:
    legacy = "tcm/" + expected.removeprefix("roboharn_evo/")
    if payload.get("schema") not in {expected, legacy} or payload.get("schema_version") != 1:
        raise SchemaValidationError(f"{label}: unsupported schema or version")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def project_trajectory_outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Project only artifact-explicit outcome evidence into model context.

    A runner-side status is not evidence.  Unless ``explicit_in_artifact`` is
    exactly true and an evidence ref is present, the status and any hidden
    details are removed rather than sent in an internally contradictory form.
    """

    data = _mapping(outcome, label="trajectory_outcome")
    if data.get("availability") == "verified":
        evidence_ref = data.get("evidence_ref")
        status = data.get("status")
        if (
            isinstance(evidence_ref, str)
            and evidence_ref.strip()
            and isinstance(status, str)
            and status.strip()
        ):
            return {
                "availability": "verified",
                "status": status,
                "evidence_ref": evidence_ref,
            }
    if data.get("explicit_in_artifact") is True:
        evidence_ref = data.get("evidence_ref")
        status = data.get("status")
        if (
            isinstance(evidence_ref, str)
            and evidence_ref.strip()
            and isinstance(status, str)
            and status.strip()
        ):
            return {
                "availability": "verified",
                "status": status,
                "evidence_ref": evidence_ref,
            }
    return {"availability": "unverified"}


def _source_fact_schema(*, source_types: Sequence[str] | None = None) -> dict[str, Any]:
    allowed_sources = list(source_types or sorted(_SOURCE_TYPES))
    properties = {
        "fact_id": {"type": "string", "minLength": 1},
        "claim": {"type": "string", "minLength": 1},
        "source_type": {"type": "string", "enum": allowed_sources},
        "supporting_evidence_refs": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "supporting_segment_refs": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "confidence": {"type": "string", "enum": sorted(_CONFIDENCE)},
        "uncertainty": {"type": "string"},
        "temporal_scope": {"type": "string", "enum": sorted(_TEMPORAL_SCOPES)},
        "transferable_allowed": {"type": "boolean", "enum": [False]},
        "structured_value": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _attribution_schema() -> dict[str, Any]:
    properties = {
        "attribution_id": {"type": "string", "minLength": 1},
        "target_path": {"type": "string", "pattern": r"^/"},
        "claim": {"type": "string", "minLength": 1},
        "source_type": {"type": "string", "enum": sorted(_SOURCE_TYPES)},
        "supporting_evidence_refs": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "supporting_segment_refs": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "confidence": {"type": "string", "enum": sorted(_CONFIDENCE)},
        "uncertainty": {"type": "string"},
        "temporal_scope": {"type": "string", "enum": sorted(_TEMPORAL_SCOPES)},
        "prediction_status": {
            "type": "string",
            "enum": ["evidence_supported", "pending_validation"],
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _request_ref_schema() -> dict[str, Any]:
    properties = {
        "evidence_ref": {"type": "string", "minLength": 1},
        "payload": {"type": "object"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def action_chunk_request_json_schema() -> dict[str, Any]:
    properties = {
        "schema": {"const": _ACTION_REQUEST_SCHEMA},
        "schema_version": {"const": 1},
        "trajectory_id": {"type": "string", "minLength": 1},
        "segment_id": {"type": "string", "minLength": 1},
        "subtask_annotation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["evidence_ref", "text"],
            "properties": {
                "evidence_ref": {"type": "string", "minLength": 1},
                "text": {"type": "string"},
            },
        },
        "action_chunk": {
            "type": "object",
            "required": [
                "action_chunk_id",
                "trajectory_id",
                "segment_id",
                "frame_range",
                "evidence_frame_refs",
            ],
            "properties": {
                "frame_range": {
                    "type": "object",
                    "required": ["start_inclusive", "end_inclusive"],
                    "properties": {
                        "start_inclusive": {"type": "integer", "minimum": 0},
                        "end_inclusive": {"type": "integer", "minimum": 0},
                    },
                },
                "evidence_frame_refs": {
                    "type": "object",
                    "required": ["before", "during", "after", "settled"],
                    "properties": {
                        role: {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                        for role in ("before", "during", "after", "settled")
                    },
                },
            },
        },
        "observed_robot_state": {
            "type": "array",
            "items": _request_ref_schema(),
        },
        "visual_evidence": {
            "type": "array",
            "items": _request_ref_schema(),
        },
        "trajectory_outcome": {"type": "object"},
        "provenance": {"type": "object"},
        "source_semantics": _source_semantics_schema(),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def action_chunk_output_json_schema() -> dict[str, Any]:
    properties = {
        "schema": {"const": _ACTION_OUTPUT_SCHEMA},
        "schema_version": {"const": 1},
        "action_chunk_id": {"type": "string", "minLength": 1},
        "status": {"enum": ["admitted", "partial", "abstained"]},
        "abstention_reason": {"type": "string"},
        "chunk_completion": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "supporting_fact_ids", "uncertainty"],
            "properties": {
                "status": {"enum": ["completed", "incomplete", "unverified"]},
                "supporting_fact_ids": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "uncertainty": {"type": "string"},
            },
        },
        "annotation_facts": {
            "type": "array",
            "items": _source_fact_schema(source_types=["annotation"]),
        },
        "robot_state_facts": {
            "type": "array",
            "items": _source_fact_schema(source_types=["observed_robot_state"]),
        },
        "visual_facts": {
            "type": "array",
            "items": _source_fact_schema(source_types=["observed_visual"]),
        },
        "uncertainties": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def action_chunk_batch_request_json_schema() -> dict[str, Any]:
    """Return the one-call-per-subtask Layer-1 request schema."""

    chunk_properties = {
        "action_chunk": action_chunk_request_json_schema()["properties"][
            "action_chunk"
        ],
        "observed_robot_state": {
            "type": "array",
            "items": _request_ref_schema(),
        },
        "visual_evidence": {
            "type": "array",
            "items": _request_ref_schema(),
        },
    }
    properties = {
        "schema": {"const": _ACTION_BATCH_REQUEST_SCHEMA},
        "schema_version": {"const": 1},
        "trajectory_id": {"type": "string", "minLength": 1},
        "segment_id": {"type": "string", "minLength": 1},
        "subtask_annotation": action_chunk_request_json_schema()["properties"][
            "subtask_annotation"
        ],
        "chunks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(chunk_properties),
                "properties": chunk_properties,
            },
        },
        "segment_visual_evidence": {
            "type": "array",
            "minItems": 1,
            "items": _request_ref_schema(),
        },
        "trajectory_outcome": {"type": "object"},
        "provenance": {"type": "object"},
        "source_semantics": _source_semantics_schema(),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def action_chunk_batch_output_json_schema() -> dict[str, Any]:
    properties = {
        "schema": {"const": _ACTION_BATCH_OUTPUT_SCHEMA},
        "schema_version": {"const": 1},
        "chunk_outputs": {
            "type": "array",
            "minItems": 1,
            "items": action_chunk_output_json_schema(),
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def subtask_summary_request_json_schema() -> dict[str, Any]:
    properties = {
        "schema": {"const": _SUBTASK_REQUEST_SCHEMA},
        "schema_version": {"const": 1},
        "trajectory_id": {"type": "string", "minLength": 1},
        "segment_id": {"type": "string", "minLength": 1},
        "segment_index": {"type": "integer", "minimum": 0},
        "subtask_annotation": {"type": "object"},
        "ordered_action_chunks": {"type": "array", "items": {"type": "object"}},
        "admitted_chunk_facts": {"type": "array", "items": _source_fact_schema()},
        "verification_visual_evidence": {
            "type": "array",
            "items": _request_ref_schema(),
        },
        "trajectory_outcome": {"type": "object"},
        "provenance": {"type": "object"},
        "source_semantics": _source_semantics_schema(),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def subtask_summary_output_json_schema() -> dict[str, Any]:
    summary_properties = {
        "summary_id": {"type": "string", "minLength": 1},
        "segment_id": {"type": "string", "minLength": 1},
        "attempted_action": _source_fact_schema(source_types=["annotation"]),
        "observed_execution": {
            "type": "array",
            "items": _source_fact_schema(
                source_types=["observed_robot_state", "observed_visual"]
            ),
        },
        "state_changes": {
            "type": "array",
            "items": _source_fact_schema(
                source_types=["observed_robot_state", "observed_visual"]
            ),
        },
        "unchanged_results": {
            "type": "array",
            "items": _source_fact_schema(source_types=["observed_visual"]),
        },
        "release_observations": {
            "type": "array",
            "items": _source_fact_schema(
                source_types=["observed_robot_state", "observed_visual"]
            ),
        },
        "feedback_observations": {
            "type": "array",
            "items": _source_fact_schema(source_types=["observed_visual"]),
        },
        "supporting_chunk_fact_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "confidence": {"type": "string", "enum": sorted(_CONFIDENCE)},
        "uncertainties": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    }
    properties = {
        "schema": {"const": _SUBTASK_OUTPUT_SCHEMA},
        "schema_version": {"const": 1},
        "status": {"enum": ["admitted", "partial", "abstained"]},
        "abstention_reason": {"type": "string"},
        "summary": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(summary_properties),
                    "properties": summary_properties,
                },
                {"type": "null"},
            ]
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _transferable_guidance_schema() -> dict[str, Any]:
    string_array = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "uniqueItems": True,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["condition", "guidance", "predicted_effects"],
        "properties": {
            "condition": {
                "type": "object",
                "additionalProperties": False,
                "required": ["task_family", "subtask_type", "preconditions"],
                "properties": {
                    "task_family": {"type": "string", "minLength": 1},
                    "subtask_type": {"type": "string", "minLength": 1},
                    "preconditions": string_array,
                },
            },
            "guidance": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ordered_steps", "feedback_policy", "avoid"],
                "properties": {
                    "ordered_steps": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["action_pattern", "instruction"],
                            "properties": {
                                "action_pattern": {"type": "string", "minLength": 1},
                                "instruction": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                    "feedback_policy": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "state_variables",
                            "attempt_tracking",
                            "observation_rules",
                            "failure_branch",
                            "success_branch",
                            "termination_conditions",
                        ],
                        "properties": {
                            "state_variables": string_array,
                            "attempt_tracking": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "state_variable",
                                    "record_after_attempt",
                                    "exclude_previously_recorded",
                                ],
                                "properties": {
                                    "state_variable": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "record_after_attempt": {"const": True},
                                    "exclude_previously_recorded": {"const": True},
                                },
                            },
                            "observation_rules": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "after_action_pattern",
                                        "observe_signal",
                                    ],
                                    "properties": {
                                        "after_action_pattern": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                        "observe_signal": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                    },
                                },
                            },
                            "failure_branch": string_array,
                            "success_branch": string_array,
                            "termination_conditions": string_array,
                        },
                    },
                    "avoid": string_array,
                },
            },
            "predicted_effects": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["effect", "validation_status"],
                    "properties": {
                        "effect": {"type": "string", "minLength": 1},
                        "validation_status": {
                            "enum": ["evidence_supported", "pending_validation"]
                        },
                    },
                },
            },
        },
    }


def procedure_draft_request_json_schema() -> dict[str, Any]:
    properties = {
        "schema": {"const": _DRAFT_REQUEST_SCHEMA},
        "schema_version": {"const": 1},
        "trajectory_id": {"type": "string", "minLength": 1},
        "instruction": {"type": "string", "minLength": 1},
        "ordered_subtask_summaries": {
            "type": "array",
            "items": subtask_summary_output_json_schema()["properties"]["summary"][
                "anyOf"
            ][0],
        },
        "episode_specific_facts": {"type": "array", "items": _source_fact_schema()},
        "cross_segment_visual_evidence": {
            "type": "array",
            "items": _request_ref_schema(),
        },
        "trajectory_outcome": {"type": "object"},
        "provenance": {"type": "object"},
        "source_semantics": _source_semantics_schema(),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def procedure_draft_output_json_schema() -> dict[str, Any]:
    properties = {
        "schema": {"const": _DRAFT_OUTPUT_SCHEMA},
        "schema_version": {"const": 1},
        "status": {"enum": ["candidate", "abstained"]},
        "abstention_reason": {"type": "string"},
        "summary": {"type": "string", "minLength": 1},
        "confidence": {"type": "string", "enum": sorted(_CONFIDENCE)},
        "cross_segment_facts": {
            "type": "array",
            "items": _source_fact_schema(source_types=["cross_segment_inference"]),
        },
        "transferable_guidance": {
            "anyOf": [_transferable_guidance_schema(), {"type": "null"}]
        },
        "supporting_fact_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "supporting_evidence_refs": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "supporting_segment_refs": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "uncertainties": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def attribution_request_json_schema() -> dict[str, Any]:
    properties = {
        "schema": {"const": _ATTRIBUTION_REQUEST_SCHEMA},
        "schema_version": {"const": 1},
        "trajectory_id": {"type": "string", "minLength": 1},
        "procedure_draft": _transferable_guidance_schema(),
        "episode_specific_facts": {"type": "array", "items": _source_fact_schema()},
        "valid_evidence_refs": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "valid_segment_refs": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "evidence_registry": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_ref", "source_types", "segment_id"],
                "properties": {
                    "evidence_ref": {"type": "string", "minLength": 1},
                    "source_types": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "enum": [
                                "annotation",
                                "observed_robot_state",
                                "observed_visual",
                            ]
                        },
                    },
                    "segment_id": {"type": "string", "minLength": 1},
                },
            },
        },
        "provenance": {"type": "object"},
        "source_semantics": _source_semantics_schema(),
        "semantic_leakage_audit": _semantic_leakage_contract_schema(),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def attribution_output_json_schema() -> dict[str, Any]:
    properties = {
        "schema": {"const": _ATTRIBUTION_OUTPUT_SCHEMA},
        "schema_version": {"const": 1},
        "status": {"enum": ["attributed", "abstained"]},
        "abstention_reason": {"type": "string"},
        "attributions": {"type": "array", "items": _attribution_schema()},
        "episode_specific_leakage": {"type": "boolean"},
        "leakage_reasons": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "leakage_confidence": {"type": "string", "enum": sorted(_CONFIDENCE)},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


@dataclass(frozen=True, slots=True)
class EvidenceInput:
    """One request-local evidence ref plus JSON metadata for a transport."""

    evidence_ref: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.evidence_ref, label="evidence.evidence_ref")
        object.__setattr__(
            self, "payload", _mapping(self.payload, label="evidence.payload")
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceInput":
        data = _mapping(payload, label="evidence")
        _exact_fields(data, {"evidence_ref", "payload"}, label="evidence")
        return cls(evidence_ref=data["evidence_ref"], payload=data["payload"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_ref": self.evidence_ref,
            "payload": deepcopy(dict(self.payload)),
        }


def _evidence_tuple(value: Any, *, label: str) -> tuple[EvidenceInput, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError(f"{label}: must be an array")
    result: list[EvidenceInput] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        evidence = (
            item if isinstance(item, EvidenceInput) else EvidenceInput.from_dict(item)
        )
        if evidence.evidence_ref in seen:
            raise SchemaValidationError(
                f"{label}[{index}].evidence_ref: duplicate {evidence.evidence_ref!r}"
            )
        seen.add(evidence.evidence_ref)
        result.append(evidence)
    return tuple(result)


def _action_chunk_bounds(
    chunk: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, int]:
    frame_range = chunk.get("frame_range")
    if not isinstance(frame_range, Mapping):
        raise SchemaValidationError(f"{label}.frame_range: must be an object")
    start = frame_range.get("start_inclusive")
    end = frame_range.get("end_inclusive")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end < start
    ):
        raise SchemaValidationError(f"{label}.frame_range: invalid inclusive range")
    duration = frame_range.get("duration_frames")
    if duration is not None and duration != end - start + 1:
        raise SchemaValidationError(f"{label}.frame_range: inconsistent duration")
    return start, end


def _evidence_frame_refs(value: EvidenceInput) -> frozenset[str]:
    refs: set[str] = set()
    direct = value.payload.get("frame_ref")
    if isinstance(direct, str) and direct:
        refs.add(direct)
    frame = value.payload.get("frame")
    if isinstance(frame, Mapping):
        nested = frame.get("evidence_ref")
        if isinstance(nested, str) and nested:
            refs.add(nested)
    sources = value.payload.get("source_frame_refs")
    if isinstance(sources, list):
        refs.update(item for item in sources if isinstance(item, str) and item)
    return frozenset(refs)


def _chunk_role_frame_refs(
    chunk: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, tuple[str, ...]]:
    raw = chunk.get("evidence_frame_refs")
    roles = {"before", "during", "after", "settled"}
    if not isinstance(raw, Mapping) or set(raw) != roles:
        raise SchemaValidationError(
            f"{label}.evidence_frame_refs: must define before/during/after/settled"
        )
    result: dict[str, tuple[str, ...]] = {}
    for role in ("before", "during", "after", "settled"):
        result[role] = _string_tuple(
            raw[role],
            label=f"{label}.evidence_frame_refs.{role}",
        )
    if not result["before"] or not result["after"]:
        raise SchemaValidationError(
            f"{label}.evidence_frame_refs: before and after are required"
        )
    return result


@dataclass(frozen=True, slots=True)
class LayerModelOutput:
    """Hash-bound, untrusted structured response from one hierarchy layer."""

    raw_text: str = field(repr=False)
    backend: str
    prompt_template_hash: str
    output_sha256: str
    parsed: Mapping[str, Any] | None = field(repr=False)
    parse_error: str | None

    @classmethod
    def from_raw(
        cls,
        raw: str | bytes | Mapping[str, Any],
        *,
        backend: str,
        prompt_template_hash: str,
    ) -> "LayerModelOutput":
        _nonempty(backend, label="backend")
        if (
            not isinstance(prompt_template_hash, str)
            or len(prompt_template_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in prompt_template_hash
            )
        ):
            raise SchemaValidationError(
                "prompt_template_hash: must be lowercase SHA-256"
            )
        parsed: dict[str, Any] | None = None
        parse_error: str | None = None
        if isinstance(raw, Mapping):
            parsed = _mapping(raw, label="layer_output")
            raw_text = _canonical(parsed)
        else:
            if isinstance(raw, bytes):
                try:
                    raw_text = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raw_text = raw.decode("utf-8", errors="replace")
                    parse_error = f"invalid_utf8:{exc.start}"
            elif isinstance(raw, str):
                raw_text = raw
            else:
                raise SchemaValidationError(
                    "layer_output: must be JSON text, bytes, or object"
                )
            if parse_error is None:
                try:

                    def reject_duplicates(
                        pairs: list[tuple[str, Any]],
                    ) -> dict[str, Any]:
                        value: dict[str, Any] = {}
                        for key, item in pairs:
                            if key in value:
                                raise ValueError(f"duplicate JSON key {key!r}")
                            value[key] = item
                        return value

                    decoded = json.loads(
                        raw_text,
                        object_pairs_hook=reject_duplicates,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"non-finite value {value}")
                        ),
                    )
                    parsed = _mapping(decoded, label="layer_output")
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    SchemaValidationError,
                ) as exc:
                    parse_error = f"invalid_json:{exc}"
        return cls(
            raw_text=raw_text,
            backend=backend,
            prompt_template_hash=prompt_template_hash,
            output_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            parsed=parsed,
            parse_error=parse_error,
        )

    def parsed_dict(self) -> dict[str, Any] | None:
        return None if self.parsed is None else deepcopy(dict(self.parsed))


@dataclass(frozen=True, slots=True)
class LayerIssue:
    code: str
    offending_field: str
    detail: str
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "offending_field": self.offending_field,
            "detail": self.detail,
            "evidence_refs": list(self.evidence_refs),
        }


def _layer_abstention(
    *,
    layer: str,
    scope_id: str,
    output: LayerModelOutput,
    issues: Sequence[LayerIssue],
) -> dict[str, Any]:
    return {
        "schema": _LAYER_ABSTENTION_SCHEMA,
        "schema_version": 1,
        "layer": layer,
        "scope_id": scope_id,
        "status": "abstained",
        "reason": issues[0].code if issues else "unknown_rejection",
        "issues": [issue.to_dict() for issue in issues],
        "backend": output.backend,
        "prompt_template_hash": output.prompt_template_hash,
        "output_sha256": output.output_sha256,
    }


def _dedupe_issues(issues: Sequence[LayerIssue]) -> tuple[LayerIssue, ...]:
    result: list[LayerIssue] = []
    seen: set[tuple[Any, ...]] = set()
    for issue in issues:
        key = (issue.code, issue.offending_field, issue.detail, issue.evidence_refs)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return tuple(result)


def _parse_output(
    output: LayerModelOutput,
    *,
    expected_schema: str,
    expected_fields: set[str],
) -> tuple[dict[str, Any] | None, list[LayerIssue]]:
    if not isinstance(output, LayerModelOutput):
        raise TypeError("backend must return LayerModelOutput")
    if output.parsed is None:
        return None, [
            LayerIssue(
                code="malformed_output",
                offending_field="backend_output",
                detail=output.parse_error or "backend output is not a JSON object",
            )
        ]
    payload = output.parsed_dict()
    assert payload is not None
    if set(payload) != expected_fields:
        return None, [
            LayerIssue(
                code="malformed_output",
                offending_field="backend_output",
                detail="top-level fields do not match the strict layer schema",
            )
        ]
    legacy_schema = "tcm/" + expected_schema.removeprefix("roboharn_evo/")
    if payload.get("schema") not in {expected_schema, legacy_schema} or payload.get("schema_version") != 1:
        return None, [
            LayerIssue(
                code="malformed_output",
                offending_field="backend_output.schema",
                detail="unknown layer output schema",
            )
        ]
    return payload, []


def _validate_fact(
    value: Any,
    *,
    path: str,
    allowed_sources: frozenset[str],
    source_refs: Mapping[str, frozenset[str]],
    evidence_segments: Mapping[str, str],
    allowed_segments: frozenset[str],
) -> tuple[dict[str, Any] | None, list[LayerIssue]]:
    issues: list[LayerIssue] = []
    if not isinstance(value, Mapping) or set(value) != _FACT_FIELDS:
        return None, [
            LayerIssue("malformed_output", path, "fact fields do not match schema")
        ]
    fact = _strict_json(dict(value), label=path)
    fact_id = fact.get("fact_id")
    source_type = fact.get("source_type")
    refs = fact.get("supporting_evidence_refs")
    segments = fact.get("supporting_segment_refs")
    if not isinstance(fact_id, str) or not fact_id.strip():
        issues.append(
            LayerIssue("malformed_output", f"{path}.fact_id", "invalid fact id")
        )
    if not isinstance(fact.get("claim"), str) or not fact["claim"].strip():
        issues.append(LayerIssue("malformed_output", f"{path}.claim", "invalid claim"))
    if source_type not in allowed_sources:
        issues.append(
            LayerIssue(
                "source_type_confusion",
                f"{path}.source_type",
                "source is not allowed in this field",
            )
        )
    if fact.get("confidence") not in _CONFIDENCE:
        issues.append(
            LayerIssue("malformed_output", f"{path}.confidence", "invalid confidence")
        )
    if not isinstance(fact.get("uncertainty"), str):
        issues.append(
            LayerIssue("malformed_output", f"{path}.uncertainty", "must be a string")
        )
    if fact.get("temporal_scope") not in _TEMPORAL_SCOPES:
        issues.append(
            LayerIssue(
                "malformed_output", f"{path}.temporal_scope", "invalid temporal scope"
            )
        )
    if fact.get("transferable_allowed") is not False:
        issues.append(
            LayerIssue(
                "episode_fact_not_private",
                f"{path}.transferable_allowed",
                "hierarchical facts must remain episode-private",
            )
        )
    if not isinstance(fact.get("structured_value"), str):
        issues.append(
            LayerIssue(
                "malformed_output", f"{path}.structured_value", "must be a string"
            )
        )
    try:
        ref_values = _string_tuple(
            refs, label=f"{path}.supporting_evidence_refs", nonempty=True
        )
        segment_values = _string_tuple(
            segments, label=f"{path}.supporting_segment_refs", nonempty=True
        )
    except SchemaValidationError as exc:
        issues.append(LayerIssue("malformed_output", path, str(exc)))
        return None, issues
    unknown_segments = set(segment_values) - set(allowed_segments)
    if unknown_segments:
        issues.append(
            LayerIssue(
                "unknown_segment_ref",
                f"{path}.supporting_segment_refs",
                "fact cites an unknown segment",
            )
        )
    if source_type in _SOURCE_TYPES:
        valid_for_source = source_refs.get(str(source_type), frozenset())
        if not set(ref_values).issubset(valid_for_source):
            issues.append(
                LayerIssue(
                    "source_type_confusion",
                    f"{path}.supporting_evidence_refs",
                    "fact cites evidence from another source class",
                    ref_values,
                )
            )
    if source_type == "annotation":
        if fact.get("temporal_scope") != "single_frame" or fact.get("structured_value"):
            issues.append(
                LayerIssue(
                    "source_type_confusion",
                    path,
                    "annotation facts describe intent only",
                )
            )
    elif source_type == "observed_robot_state":
        if fact.get("temporal_scope") == "cross_segment" or not fact.get(
            "structured_value"
        ):
            issues.append(
                LayerIssue(
                    "source_type_confusion",
                    path,
                    "robot-state facts require a structured observed value and local scope",
                )
            )
    elif source_type == "observed_visual":
        if fact.get("temporal_scope") == "cross_segment" or fact.get(
            "structured_value"
        ):
            issues.append(
                LayerIssue(
                    "source_type_confusion",
                    path,
                    "visual facts must remain visual and local",
                )
            )
    elif source_type == "cross_segment_inference":
        cited_segments = {
            evidence_segments[ref] for ref in ref_values if ref in evidence_segments
        }
        if (
            fact.get("temporal_scope") != "cross_segment"
            or len(set(segment_values)) < 2
            or cited_segments != set(segment_values)
            or fact.get("structured_value")
        ):
            issues.append(
                LayerIssue(
                    "cross_segment_evidence_mismatch",
                    path,
                    "cross-segment fact must cite evidence spanning exactly its segments",
                    ref_values,
                )
            )
    return (None if issues else fact), issues


@dataclass(frozen=True, slots=True)
class ActionChunkReflectionRequest:
    """Layer-1 request with structurally separated evidence sources."""

    trajectory_id: str
    segment_id: str
    subtask_annotation: Mapping[str, Any]
    action_chunk: Mapping[str, Any]
    observed_robot_state: tuple[EvidenceInput, ...]
    visual_evidence: tuple[EvidenceInput, ...]
    trajectory_outcome: Mapping[str, Any]
    provenance: Mapping[str, Any]
    schema: str = _ACTION_REQUEST_SCHEMA
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.trajectory_id, label="action_request.trajectory_id")
        _nonempty(self.segment_id, label="action_request.segment_id")
        if self.schema not in {_ACTION_REQUEST_SCHEMA, "tcm/action_chunk_reflection_request/v1"} or self.schema_version != 1:
            raise SchemaValidationError("action_request: unsupported schema or version")
        annotation = _mapping(
            self.subtask_annotation,
            label="action_request.subtask_annotation",
        )
        _exact_fields(
            annotation,
            {"evidence_ref", "text"},
            label="action_request.subtask_annotation",
        )
        _nonempty(
            annotation["evidence_ref"],
            label="action_request.subtask_annotation.evidence_ref",
        )
        if not isinstance(annotation["text"], str):
            raise SchemaValidationError(
                "action_request.subtask_annotation.text: must be a string"
            )
        expected_annotation_ref = annotation_evidence_ref(self.segment_id)
        if annotation["evidence_ref"] != expected_annotation_ref:
            raise SchemaValidationError(
                "action_request.subtask_annotation.evidence_ref: must match segment"
            )
        chunk = _mapping(self.action_chunk, label="action_request.action_chunk")
        required_chunk = {
            "action_chunk_id",
            "trajectory_id",
            "segment_id",
            "frame_range",
            "evidence_frame_refs",
        }
        missing = sorted(required_chunk - set(chunk))
        if missing:
            raise SchemaValidationError(
                "action_request.action_chunk: missing required field(s): "
                + ", ".join(missing)
            )
        _nonempty(chunk["action_chunk_id"], label="action_chunk.action_chunk_id")
        if chunk["trajectory_id"] != self.trajectory_id:
            raise SchemaValidationError("action_chunk.trajectory_id: request mismatch")
        if chunk["segment_id"] != self.segment_id:
            raise SchemaValidationError("action_chunk.segment_id: request mismatch")
        _action_chunk_bounds(chunk, label="action_chunk")
        role_refs = _chunk_role_frame_refs(chunk, label="action_chunk")
        robot = _evidence_tuple(
            self.observed_robot_state,
            label="action_request.observed_robot_state",
        )
        visual = _evidence_tuple(
            self.visual_evidence,
            label="action_request.visual_evidence",
        )
        overlap = {item.evidence_ref for item in robot} & {
            item.evidence_ref for item in visual
        }
        # A synchronized visual item may also carry observed gripper state.  It
        # remains explicitly declared in both namespaces rather than inferred.
        if any(not ref.strip() for ref in overlap):
            raise SchemaValidationError("action_request: invalid shared evidence ref")
        allowed_visual_frames = {
            frame_ref for refs in role_refs.values() for frame_ref in refs
        }
        covered_visual_frames: set[str] = set()
        for index, evidence in enumerate(visual):
            bound_frames = _evidence_frame_refs(evidence)
            if not bound_frames or not bound_frames.issubset(allowed_visual_frames):
                raise SchemaValidationError(
                    f"action_request.visual_evidence[{index}]: not bound to this action chunk"
                )
            covered_visual_frames.update(bound_frames)
        missing_visual_frames = allowed_visual_frames - covered_visual_frames
        if missing_visual_frames:
            raise SchemaValidationError(
                "action_request.visual_evidence: declared chunk frame roles are not covered"
            )
        outcome = project_trajectory_outcome(self.trajectory_outcome)
        provenance = ProvenanceV1.from_dict(
            _mapping(self.provenance, label="action_request.provenance")
        ).to_dict()
        object.__setattr__(self, "subtask_annotation", annotation)
        object.__setattr__(self, "action_chunk", chunk)
        object.__setattr__(self, "observed_robot_state", robot)
        object.__setattr__(self, "visual_evidence", visual)
        object.__setattr__(self, "trajectory_outcome", outcome)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionChunkReflectionRequest":
        data = _mapping(payload, label="action_request")
        expected = {
            "schema",
            "schema_version",
            "trajectory_id",
            "segment_id",
            "subtask_annotation",
            "action_chunk",
            "observed_robot_state",
            "visual_evidence",
            "trajectory_outcome",
            "provenance",
            "source_semantics",
        }
        _exact_fields(data, expected, label="action_request")
        _schema_header(data, _ACTION_REQUEST_SCHEMA, label="action_request")
        _validate_source_semantics(
            data["source_semantics"],
            label="action_request.source_semantics",
        )
        return cls(
            schema=data["schema"],
            schema_version=data["schema_version"],
            trajectory_id=data["trajectory_id"],
            segment_id=data["segment_id"],
            subtask_annotation=data["subtask_annotation"],
            action_chunk=data["action_chunk"],
            observed_robot_state=tuple(data["observed_robot_state"]),
            visual_evidence=tuple(data["visual_evidence"]),
            trajectory_outcome=data["trajectory_outcome"],
            provenance=data["provenance"],
        )

    @property
    def action_chunk_id(self) -> str:
        return str(self.action_chunk["action_chunk_id"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "segment_id": self.segment_id,
            "subtask_annotation": deepcopy(dict(self.subtask_annotation)),
            "action_chunk": deepcopy(dict(self.action_chunk)),
            "observed_robot_state": [
                item.to_dict() for item in self.observed_robot_state
            ],
            "visual_evidence": [item.to_dict() for item in self.visual_evidence],
            "trajectory_outcome": deepcopy(dict(self.trajectory_outcome)),
            "provenance": deepcopy(dict(self.provenance)),
            "source_semantics": hierarchical_source_semantics(),
        }


@dataclass(frozen=True, slots=True)
class ActionChunkBatchReflectionRequest:
    """Layer-1 batch: all chunks in one subtask share one model call."""

    chunks: tuple[ActionChunkReflectionRequest, ...]
    segment_visual_evidence: tuple[EvidenceInput, ...] = ()
    schema: str = _ACTION_BATCH_REQUEST_SCHEMA
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema not in {_ACTION_BATCH_REQUEST_SCHEMA, "tcm/action_chunk_batch_reflection_request/v1"} or self.schema_version != 1:
            raise SchemaValidationError(
                "action_batch_request: unsupported schema or version"
            )
        if not self.chunks:
            raise SchemaValidationError(
                "action_batch_request.chunks: must not be empty"
            )
        normalized: list[ActionChunkReflectionRequest] = []
        for index, item in enumerate(self.chunks):
            if not isinstance(item, ActionChunkReflectionRequest):
                raise SchemaValidationError(
                    f"action_batch_request.chunks[{index}]: invalid request"
                )
            normalized.append(item)
        first = normalized[0]
        scope = {
            (
                item.trajectory_id,
                item.segment_id,
                _canonical(item.subtask_annotation),
                _canonical(item.trajectory_outcome),
                _canonical(item.provenance),
            )
            for item in normalized
        }
        if len(scope) != 1:
            raise SchemaValidationError(
                "action_batch_request.chunks: all chunks must share one subtask context"
            )
        ids = [item.action_chunk_id for item in normalized]
        if len(set(ids)) != len(ids):
            raise SchemaValidationError(
                "action_batch_request.chunks: duplicate chunk id"
            )
        normalized.sort(
            key=lambda item: (
                *_action_chunk_bounds(item.action_chunk, label="action_chunk"),
                item.action_chunk_id,
            )
        )
        segment_visual = tuple(self.segment_visual_evidence)
        if segment_visual:
            segment_visual = _evidence_tuple(
                segment_visual,
                label="action_batch_request.segment_visual_evidence",
            )
        else:
            by_ref: dict[str, EvidenceInput] = {}
            for item in normalized:
                for evidence in item.visual_evidence:
                    previous = by_ref.get(evidence.evidence_ref)
                    if previous is not None and previous != evidence:
                        raise SchemaValidationError(
                            "action_batch_request.segment_visual_evidence: "
                            "conflicting duplicate evidence ref"
                        )
                    by_ref[evidence.evidence_ref] = evidence
            segment_visual = tuple(by_ref.values())
        registry = {value.evidence_ref: value for value in segment_visual}
        for index, evidence in enumerate(segment_visual):
            if evidence.payload.get("segment_id") != normalized[0].segment_id:
                raise SchemaValidationError(
                    "action_batch_request.segment_visual_evidence"
                    f"[{index}].payload.segment_id: batch segment mismatch"
                )
        for item in normalized:
            for evidence in item.visual_evidence:
                registered = registry.get(evidence.evidence_ref)
                if registered is None or registered != evidence:
                    raise SchemaValidationError(
                        "action_batch_request.segment_visual_evidence: every "
                        "chunk visual ref must have an identical registry entry"
                    )
        object.__setattr__(self, "chunks", tuple(normalized))
        object.__setattr__(self, "segment_visual_evidence", segment_visual)
        # Keep the variable live as an explicit invariant witness for readers.
        del first

    @property
    def trajectory_id(self) -> str:
        return self.chunks[0].trajectory_id

    @property
    def segment_id(self) -> str:
        return self.chunks[0].segment_id

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ActionChunkBatchReflectionRequest":
        data = _mapping(payload, label="action_batch_request")
        expected = {
            "schema",
            "schema_version",
            "trajectory_id",
            "segment_id",
            "subtask_annotation",
            "chunks",
            "segment_visual_evidence",
            "trajectory_outcome",
            "provenance",
            "source_semantics",
        }
        _exact_fields(data, expected, label="action_batch_request")
        _schema_header(
            data,
            _ACTION_BATCH_REQUEST_SCHEMA,
            label="action_batch_request",
        )
        _validate_source_semantics(
            data["source_semantics"],
            label="action_batch_request.source_semantics",
        )
        raw_chunks = data["chunks"]
        if not isinstance(raw_chunks, list):
            raise SchemaValidationError("action_batch_request.chunks: must be an array")
        requests: list[ActionChunkReflectionRequest] = []
        for index, item in enumerate(raw_chunks):
            chunk = _mapping(item, label=f"action_batch_request.chunks[{index}]")
            _exact_fields(
                chunk,
                {"action_chunk", "observed_robot_state", "visual_evidence"},
                label=f"action_batch_request.chunks[{index}]",
            )
            requests.append(
                ActionChunkReflectionRequest(
                    trajectory_id=data["trajectory_id"],
                    segment_id=data["segment_id"],
                    subtask_annotation=data["subtask_annotation"],
                    action_chunk=chunk["action_chunk"],
                    observed_robot_state=tuple(chunk["observed_robot_state"]),
                    visual_evidence=tuple(chunk["visual_evidence"]),
                    trajectory_outcome=data["trajectory_outcome"],
                    provenance=data["provenance"],
                )
            )
        return cls(
            chunks=tuple(requests),
            segment_visual_evidence=tuple(data["segment_visual_evidence"]),
            schema=data["schema"],
            schema_version=data["schema_version"],
        )

    def to_dict(self) -> dict[str, Any]:
        first = self.chunks[0]
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "trajectory_id": first.trajectory_id,
            "segment_id": first.segment_id,
            "subtask_annotation": deepcopy(dict(first.subtask_annotation)),
            "chunks": [
                {
                    "action_chunk": deepcopy(dict(item.action_chunk)),
                    "observed_robot_state": [
                        evidence.to_dict() for evidence in item.observed_robot_state
                    ],
                    "visual_evidence": [
                        evidence.to_dict() for evidence in item.visual_evidence
                    ],
                }
                for item in self.chunks
            ],
            "segment_visual_evidence": [
                value.to_dict() for value in self.segment_visual_evidence
            ],
            "trajectory_outcome": deepcopy(dict(first.trajectory_outcome)),
            "provenance": deepcopy(dict(first.provenance)),
            "source_semantics": hierarchical_source_semantics(),
        }


@dataclass(frozen=True, slots=True)
class ActionChunkReflectionResult:
    action_chunk_id: str
    admitted_facts: tuple[Mapping[str, Any], ...]
    rejected_facts: tuple[Mapping[str, Any], ...]
    chunk_completion: Mapping[str, Any] | None
    abstention: Mapping[str, Any] | None
    output_sha256: str

    @property
    def admitted(self) -> bool:
        return self.chunk_completion is not None and self.abstention is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_chunk_id": self.action_chunk_id,
            "admitted_facts": [deepcopy(dict(item)) for item in self.admitted_facts],
            "rejected_facts": [deepcopy(dict(item)) for item in self.rejected_facts],
            "chunk_completion": (
                None
                if self.chunk_completion is None
                else deepcopy(dict(self.chunk_completion))
            ),
            "abstention": (
                None if self.abstention is None else deepcopy(dict(self.abstention))
            ),
            "output_sha256": self.output_sha256,
        }


class ActionChunkQualityGate:
    """Admit source-bound Layer-1 facts without promoting any guidance."""

    _FIELDS = {
        "schema",
        "schema_version",
        "action_chunk_id",
        "status",
        "abstention_reason",
        "chunk_completion",
        "annotation_facts",
        "robot_state_facts",
        "visual_facts",
        "uncertainties",
    }

    def evaluate(
        self,
        request: ActionChunkReflectionRequest,
        output: LayerModelOutput,
    ) -> ActionChunkReflectionResult:
        payload, issues = _parse_output(
            output,
            expected_schema=_ACTION_OUTPUT_SCHEMA,
            expected_fields=self._FIELDS,
        )
        if payload is None:
            return self._result(request, output, (), (), None, issues)
        status = payload.get("status")
        if payload.get("action_chunk_id") != request.action_chunk_id:
            issues.append(
                LayerIssue(
                    "scope_mismatch",
                    "action_chunk_id",
                    "output chunk id does not match its request",
                )
            )
        reason = payload.get("abstention_reason")
        if status not in {"admitted", "partial", "abstained"}:
            issues.append(LayerIssue("malformed_output", "status", "invalid status"))
        if not isinstance(reason, str) or (
            status == "abstained" and not reason.strip()
        ):
            issues.append(
                LayerIssue(
                    "malformed_output", "abstention_reason", "invalid abstention reason"
                )
            )
        if status != "abstained" and reason:
            issues.append(
                LayerIssue(
                    "malformed_output",
                    "abstention_reason",
                    "non-abstained output must use an empty reason",
                )
            )
        if not isinstance(payload.get("uncertainties"), list):
            issues.append(
                LayerIssue("malformed_output", "uncertainties", "must be an array")
            )
        else:
            try:
                _string_tuple(payload["uncertainties"], label="uncertainties")
            except SchemaValidationError as exc:
                issues.append(LayerIssue("malformed_output", "uncertainties", str(exc)))

        annotation_ref = str(request.subtask_annotation["evidence_ref"])
        robot_refs = frozenset(
            item.evidence_ref for item in request.observed_robot_state
        )
        visual_refs = frozenset(item.evidence_ref for item in request.visual_evidence)
        source_refs = {
            "annotation": frozenset({annotation_ref}),
            "observed_robot_state": robot_refs,
            "observed_visual": visual_refs,
            "cross_segment_inference": frozenset(),
        }
        evidence_segments = {
            ref: request.segment_id
            for ref in {annotation_ref, *robot_refs, *visual_refs}
        }
        admitted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        groups = (
            ("annotation_facts", frozenset({"annotation"})),
            ("robot_state_facts", frozenset({"observed_robot_state"})),
            ("visual_facts", frozenset({"observed_visual"})),
        )
        for field_name, allowed_sources in groups:
            values = payload.get(field_name)
            if not isinstance(values, list):
                issues.append(
                    LayerIssue("malformed_output", field_name, "must be an array")
                )
                continue
            for index, value in enumerate(values):
                fact, fact_issues = _validate_fact(
                    value,
                    path=f"{field_name}[{index}]",
                    allowed_sources=allowed_sources,
                    source_refs=source_refs,
                    evidence_segments=evidence_segments,
                    allowed_segments=frozenset({request.segment_id}),
                )
                if fact is None:
                    rejected.append(
                        {
                            "fact": _strict_json(value, label=f"{field_name}[{index}]"),
                            "issues": [issue.to_dict() for issue in fact_issues],
                        }
                    )
                    issues.extend(fact_issues)
                    continue
                if fact["fact_id"] in seen_ids:
                    duplicate = LayerIssue(
                        "duplicate_fact_id",
                        f"{field_name}[{index}].fact_id",
                        "fact id must be unique within the layer output",
                    )
                    rejected.append({"fact": fact, "issues": [duplicate.to_dict()]})
                    issues.append(duplicate)
                    continue
                seen_ids.add(str(fact["fact_id"]))
                admitted.append(fact)

        completion = payload.get("chunk_completion")
        completion_value: dict[str, Any] | None = None
        if not isinstance(completion, Mapping) or set(completion) != {
            "status",
            "supporting_fact_ids",
            "uncertainty",
        }:
            issues.append(
                LayerIssue(
                    "malformed_output", "chunk_completion", "fields do not match schema"
                )
            )
        else:
            completion_value = _strict_json(dict(completion), label="chunk_completion")
            if completion_value.get("status") not in {
                "completed",
                "incomplete",
                "unverified",
            } or not isinstance(completion_value.get("uncertainty"), str):
                issues.append(
                    LayerIssue(
                        "malformed_output", "chunk_completion", "invalid typed values"
                    )
                )
                completion_value = None
            else:
                try:
                    support_ids = _string_tuple(
                        completion_value.get("supporting_fact_ids"),
                        label="chunk_completion.supporting_fact_ids",
                    )
                except SchemaValidationError as exc:
                    issues.append(
                        LayerIssue("malformed_output", "chunk_completion", str(exc))
                    )
                    completion_value = None
                else:
                    if not set(support_ids).issubset(seen_ids):
                        issues.append(
                            LayerIssue(
                                "unknown_fact_ref",
                                "chunk_completion.supporting_fact_ids",
                                "completion cites a fact that was not admitted",
                            )
                        )
                        completion_value = None
                    elif completion_value["status"] == "completed" and not {
                        fact["source_type"] for fact in admitted
                    }.intersection({"observed_robot_state", "observed_visual"}):
                        issues.append(
                            LayerIssue(
                                "unverified_chunk_completion",
                                "chunk_completion.status",
                                "annotation alone cannot prove physical completion",
                            )
                        )
                        completion_value = None

        fatal = any(
            issue.offending_field
            in {
                "backend_output",
                "backend_output.schema",
                "status",
                "abstention_reason",
                "action_chunk_id",
                "chunk_completion",
            }
            for issue in issues
        )
        if status == "abstained":
            issues.append(
                LayerIssue("backend_abstained", "abstention_reason", str(reason))
            )
            completion_value = None
        elif fatal:
            completion_value = None
        return self._result(
            request,
            output,
            admitted,
            rejected,
            completion_value,
            issues,
        )

    @staticmethod
    def _result(
        request: ActionChunkReflectionRequest,
        output: LayerModelOutput,
        admitted: Sequence[Mapping[str, Any]],
        rejected: Sequence[Mapping[str, Any]],
        completion: Mapping[str, Any] | None,
        issues: Sequence[LayerIssue],
    ) -> ActionChunkReflectionResult:
        unique_issues = _dedupe_issues(issues)
        abstention = (
            None
            if completion is not None
            else _layer_abstention(
                layer="action_chunk",
                scope_id=request.action_chunk_id,
                output=output,
                issues=unique_issues
                or (
                    LayerIssue(
                        "layer_not_admitted",
                        "chunk_completion",
                        "chunk completion was not admitted",
                    ),
                ),
            )
        )
        return ActionChunkReflectionResult(
            action_chunk_id=request.action_chunk_id,
            admitted_facts=tuple(deepcopy(list(admitted))),
            rejected_facts=tuple(deepcopy(list(rejected))),
            chunk_completion=(
                None if completion is None else deepcopy(dict(completion))
            ),
            abstention=abstention,
            output_sha256=output.output_sha256,
        )


class ActionChunkBatchQualityGate:
    """Split one Layer-1 call and apply the chunk gate independently."""

    _FIELDS = {"schema", "schema_version", "chunk_outputs"}

    def __init__(self, *, chunk_gate: ActionChunkQualityGate | None = None) -> None:
        self._chunk_gate = chunk_gate or ActionChunkQualityGate()

    def evaluate(
        self,
        request: ActionChunkBatchReflectionRequest,
        output: LayerModelOutput,
    ) -> tuple[ActionChunkReflectionResult, ...]:
        payload, issues = _parse_output(
            output,
            expected_schema=_ACTION_BATCH_OUTPUT_SCHEMA,
            expected_fields=self._FIELDS,
        )
        if payload is None or not isinstance(payload.get("chunk_outputs"), list):
            if payload is not None:
                issues.append(
                    LayerIssue("malformed_output", "chunk_outputs", "must be an array")
                )
            return tuple(
                self._missing_result(item, output, issues) for item in request.chunks
            )
        by_id: dict[str, Any] = {}
        duplicate_ids: set[str] = set()
        for value in payload["chunk_outputs"]:
            if not isinstance(value, Mapping):
                continue
            chunk_id = value.get("action_chunk_id")
            if not isinstance(chunk_id, str):
                continue
            if chunk_id in by_id:
                duplicate_ids.add(chunk_id)
            else:
                by_id[chunk_id] = value
        request_ids = {item.action_chunk_id for item in request.chunks}
        unknown_ids = set(by_id) - request_ids
        results: list[ActionChunkReflectionResult] = []
        for item in request.chunks:
            if (
                item.action_chunk_id not in by_id
                or item.action_chunk_id in duplicate_ids
            ):
                local_issues = [
                    LayerIssue(
                        "missing_or_duplicate_chunk_output",
                        "chunk_outputs",
                        "each requested chunk requires exactly one output",
                    )
                ]
                results.append(self._missing_result(item, output, local_issues))
                continue
            child_output = LayerModelOutput.from_raw(
                by_id[item.action_chunk_id],
                backend=output.backend,
                prompt_template_hash=output.prompt_template_hash,
            )
            results.append(self._chunk_gate.evaluate(item, child_output))
        if unknown_ids:
            strict_results: list[ActionChunkReflectionResult] = []
            for item, result in zip(request.chunks, results, strict=True):
                issue = LayerIssue(
                    "unexpected_chunk_output",
                    "chunk_outputs",
                    "batch contains output for an unrequested chunk",
                )
                strict_results.append(
                    ActionChunkReflectionResult(
                        action_chunk_id=result.action_chunk_id,
                        admitted_facts=result.admitted_facts,
                        rejected_facts=result.rejected_facts,
                        chunk_completion=None,
                        abstention=_layer_abstention(
                            layer="action_chunk",
                            scope_id=item.action_chunk_id,
                            output=output,
                            issues=(issue,),
                        ),
                        output_sha256=result.output_sha256,
                    )
                )
            results = strict_results
        return tuple(results)

    @staticmethod
    def _missing_result(
        request: ActionChunkReflectionRequest,
        output: LayerModelOutput,
        issues: Sequence[LayerIssue],
    ) -> ActionChunkReflectionResult:
        abstention = _layer_abstention(
            layer="action_chunk",
            scope_id=request.action_chunk_id,
            output=output,
            issues=_dedupe_issues(issues),
        )
        return ActionChunkReflectionResult(
            action_chunk_id=request.action_chunk_id,
            admitted_facts=(),
            rejected_facts=(),
            chunk_completion=None,
            abstention=abstention,
            output_sha256=output.output_sha256,
        )


@dataclass(frozen=True, slots=True)
class SubtaskSummaryRequest:
    """Layer-2 request built from ordered chunks and admitted Layer-1 facts."""

    trajectory_id: str
    segment_id: str
    segment_index: int
    subtask_annotation: Mapping[str, Any]
    ordered_action_chunks: tuple[Mapping[str, Any], ...]
    admitted_chunk_facts: tuple[Mapping[str, Any], ...]
    verification_visual_evidence: tuple[EvidenceInput, ...]
    trajectory_outcome: Mapping[str, Any]
    provenance: Mapping[str, Any]
    schema: str = _SUBTASK_REQUEST_SCHEMA
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.trajectory_id, label="subtask_request.trajectory_id")
        _nonempty(self.segment_id, label="subtask_request.segment_id")
        if (
            not isinstance(self.segment_index, int)
            or isinstance(self.segment_index, bool)
            or self.segment_index < 0
        ):
            raise SchemaValidationError("subtask_request.segment_index: invalid index")
        if self.schema not in {_SUBTASK_REQUEST_SCHEMA, "tcm/subtask_summary_request/v1"} or self.schema_version != 1:
            raise SchemaValidationError(
                "subtask_request: unsupported schema or version"
            )
        annotation = _mapping(
            self.subtask_annotation,
            label="subtask_request.subtask_annotation",
        )
        _exact_fields(
            annotation,
            {"evidence_ref", "text"},
            label="subtask_request.subtask_annotation",
        )
        if annotation.get("evidence_ref") != annotation_evidence_ref(self.segment_id):
            raise SchemaValidationError(
                "subtask_request.subtask_annotation.evidence_ref: segment mismatch"
            )
        if not isinstance(annotation.get("text"), str):
            raise SchemaValidationError(
                "subtask_request.subtask_annotation.text: must be a string"
            )
        chunks = _mapping_tuple(
            self.ordered_action_chunks,
            label="subtask_request.ordered_action_chunks",
        )
        last_end = -1
        chunk_ids: set[str] = set()
        for index, chunk in enumerate(chunks):
            required = {
                "action_chunk_id",
                "trajectory_id",
                "segment_id",
                "frame_range",
                "evidence_frame_refs",
            }
            if required - set(chunk):
                raise SchemaValidationError(
                    f"subtask_request.ordered_action_chunks[{index}]: incomplete chunk identity"
                )
            if (
                chunk["trajectory_id"] != self.trajectory_id
                or chunk["segment_id"] != self.segment_id
            ):
                raise SchemaValidationError(
                    f"subtask_request.ordered_action_chunks[{index}]: scope mismatch"
                )
            chunk_id = _nonempty(
                chunk["action_chunk_id"],
                label=f"subtask_request.ordered_action_chunks[{index}].action_chunk_id",
            )
            if chunk_id in chunk_ids:
                raise SchemaValidationError(
                    f"subtask_request.ordered_action_chunks[{index}]: duplicate chunk id"
                )
            chunk_ids.add(chunk_id)
            start, end = _action_chunk_bounds(
                chunk,
                label=f"subtask_request.ordered_action_chunks[{index}]",
            )
            _chunk_role_frame_refs(
                chunk,
                label=f"subtask_request.ordered_action_chunks[{index}]",
            )
            if start < last_end:
                raise SchemaValidationError(
                    "subtask_request.ordered_action_chunks: must be time ordered"
                )
            last_end = end
        facts = _mapping_tuple(
            self.admitted_chunk_facts,
            label="subtask_request.admitted_chunk_facts",
        )
        if len({fact.get("fact_id") for fact in facts}) != len(facts):
            raise SchemaValidationError(
                "subtask_request.admitted_chunk_facts: duplicate fact id"
            )
        for index, fact in enumerate(facts):
            if (
                set(fact) != _FACT_FIELDS
                or fact.get("transferable_allowed") is not False
            ):
                raise SchemaValidationError(
                    f"subtask_request.admitted_chunk_facts[{index}]: not an admitted private fact"
                )
            if fact.get("supporting_segment_refs") != [self.segment_id]:
                raise SchemaValidationError(
                    f"subtask_request.admitted_chunk_facts[{index}]: segment mismatch"
                )
        evidence = _evidence_tuple(
            self.verification_visual_evidence,
            label="subtask_request.verification_visual_evidence",
        )
        outcome = project_trajectory_outcome(self.trajectory_outcome)
        provenance = ProvenanceV1.from_dict(
            _mapping(self.provenance, label="subtask_request.provenance")
        ).to_dict()
        object.__setattr__(self, "subtask_annotation", annotation)
        object.__setattr__(self, "ordered_action_chunks", chunks)
        object.__setattr__(self, "admitted_chunk_facts", facts)
        object.__setattr__(self, "verification_visual_evidence", evidence)
        object.__setattr__(self, "trajectory_outcome", outcome)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubtaskSummaryRequest":
        data = _mapping(payload, label="subtask_request")
        expected = {
            "schema",
            "schema_version",
            "trajectory_id",
            "segment_id",
            "segment_index",
            "subtask_annotation",
            "ordered_action_chunks",
            "admitted_chunk_facts",
            "verification_visual_evidence",
            "trajectory_outcome",
            "provenance",
            "source_semantics",
        }
        _exact_fields(data, expected, label="subtask_request")
        _schema_header(data, _SUBTASK_REQUEST_SCHEMA, label="subtask_request")
        _validate_source_semantics(
            data["source_semantics"],
            label="subtask_request.source_semantics",
        )
        return cls(
            schema=data["schema"],
            schema_version=data["schema_version"],
            trajectory_id=data["trajectory_id"],
            segment_id=data["segment_id"],
            segment_index=data["segment_index"],
            subtask_annotation=data["subtask_annotation"],
            ordered_action_chunks=tuple(data["ordered_action_chunks"]),
            admitted_chunk_facts=tuple(data["admitted_chunk_facts"]),
            verification_visual_evidence=tuple(data["verification_visual_evidence"]),
            trajectory_outcome=data["trajectory_outcome"],
            provenance=data["provenance"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "segment_id": self.segment_id,
            "segment_index": self.segment_index,
            "subtask_annotation": deepcopy(dict(self.subtask_annotation)),
            "ordered_action_chunks": [
                deepcopy(dict(item)) for item in self.ordered_action_chunks
            ],
            "admitted_chunk_facts": [
                deepcopy(dict(item)) for item in self.admitted_chunk_facts
            ],
            "verification_visual_evidence": [
                item.to_dict() for item in self.verification_visual_evidence
            ],
            "trajectory_outcome": deepcopy(dict(self.trajectory_outcome)),
            "provenance": deepcopy(dict(self.provenance)),
            "source_semantics": hierarchical_source_semantics(),
        }


@dataclass(frozen=True, slots=True)
class SubtaskSummaryResult:
    segment_id: str
    summary: Mapping[str, Any] | None
    admitted_episode_facts: tuple[Mapping[str, Any], ...]
    rejected_facts: tuple[Mapping[str, Any], ...]
    abstention: Mapping[str, Any] | None
    output_sha256: str

    @property
    def admitted(self) -> bool:
        return self.summary is not None and self.abstention is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "summary": None if self.summary is None else deepcopy(dict(self.summary)),
            "admitted_episode_facts": [
                deepcopy(dict(item)) for item in self.admitted_episode_facts
            ],
            "rejected_facts": [deepcopy(dict(item)) for item in self.rejected_facts],
            "abstention": (
                None if self.abstention is None else deepcopy(dict(self.abstention))
            ),
            "output_sha256": self.output_sha256,
        }


class SubtaskSummaryQualityGate:
    """Validate Layer-2 semantic roles and preserve independently valid facts."""

    _FIELDS = {
        "schema",
        "schema_version",
        "status",
        "abstention_reason",
        "summary",
    }
    _SUMMARY_FIELDS = {
        "summary_id",
        "segment_id",
        "attempted_action",
        "observed_execution",
        "state_changes",
        "unchanged_results",
        "release_observations",
        "feedback_observations",
        "supporting_chunk_fact_ids",
        "confidence",
        "uncertainties",
    }

    def evaluate(
        self,
        request: SubtaskSummaryRequest,
        output: LayerModelOutput,
    ) -> SubtaskSummaryResult:
        payload, issues = _parse_output(
            output,
            expected_schema=_SUBTASK_OUTPUT_SCHEMA,
            expected_fields=self._FIELDS,
        )
        if payload is None:
            return self._result(request, output, None, (), (), issues)
        status = payload.get("status")
        reason = payload.get("abstention_reason")
        if status not in {"admitted", "partial", "abstained"}:
            issues.append(LayerIssue("malformed_output", "status", "invalid status"))
        if not isinstance(reason, str) or (
            status == "abstained" and not reason.strip()
        ):
            issues.append(
                LayerIssue("malformed_output", "abstention_reason", "invalid reason")
            )
        if status != "abstained" and reason:
            issues.append(
                LayerIssue(
                    "malformed_output",
                    "abstention_reason",
                    "non-abstained output must use an empty reason",
                )
            )
        raw_summary = payload.get("summary")
        if status == "abstained":
            if raw_summary is not None:
                issues.append(
                    LayerIssue(
                        "malformed_output",
                        "summary",
                        "abstained output must use null summary",
                    )
                )
            issues.append(
                LayerIssue("backend_abstained", "abstention_reason", str(reason))
            )
            return self._result(request, output, None, (), (), issues)
        if (
            not isinstance(raw_summary, Mapping)
            or set(raw_summary) != self._SUMMARY_FIELDS
        ):
            issues.append(
                LayerIssue("malformed_output", "summary", "fields do not match schema")
            )
            return self._result(request, output, None, (), (), issues)
        summary = _strict_json(dict(raw_summary), label="summary")
        if (
            not isinstance(summary.get("summary_id"), str)
            or not summary["summary_id"].strip()
        ):
            issues.append(
                LayerIssue(
                    "malformed_output", "summary.summary_id", "invalid summary id"
                )
            )
        if summary.get("segment_id") != request.segment_id:
            issues.append(
                LayerIssue("scope_mismatch", "summary.segment_id", "segment mismatch")
            )
        if summary.get("confidence") not in _CONFIDENCE:
            issues.append(
                LayerIssue(
                    "malformed_output", "summary.confidence", "invalid confidence"
                )
            )
        try:
            uncertainty_values = _string_tuple(
                summary.get("uncertainties"),
                label="summary.uncertainties",
            )
            support_ids = _string_tuple(
                summary.get("supporting_chunk_fact_ids"),
                label="summary.supporting_chunk_fact_ids",
            )
        except SchemaValidationError as exc:
            issues.append(LayerIssue("malformed_output", "summary", str(exc)))
            uncertainty_values = ()
            support_ids = ()
        known_chunk_facts = {
            str(fact["fact_id"]) for fact in request.admitted_chunk_facts
        }
        if not set(support_ids).issubset(known_chunk_facts):
            issues.append(
                LayerIssue(
                    "unknown_fact_ref",
                    "summary.supporting_chunk_fact_ids",
                    "summary cites an unadmitted chunk fact",
                )
            )

        annotation_ref = str(request.subtask_annotation["evidence_ref"])
        robot_refs = {
            str(ref)
            for fact in request.admitted_chunk_facts
            if fact.get("source_type") == "observed_robot_state"
            for ref in fact.get("supporting_evidence_refs", [])
        }
        visual_refs = {
            str(ref)
            for fact in request.admitted_chunk_facts
            if fact.get("source_type") == "observed_visual"
            for ref in fact.get("supporting_evidence_refs", [])
        }
        visual_refs.update(
            item.evidence_ref for item in request.verification_visual_evidence
        )
        source_refs = {
            "annotation": frozenset({annotation_ref}),
            "observed_robot_state": frozenset(robot_refs),
            "observed_visual": frozenset(visual_refs),
            "cross_segment_inference": frozenset(),
        }
        evidence_segments = {
            ref: request.segment_id
            for ref in {annotation_ref, *robot_refs, *visual_refs}
        }
        specifications = (
            ("attempted_action", frozenset({"annotation"}), False, False),
            (
                "observed_execution",
                frozenset({"observed_robot_state", "observed_visual"}),
                True,
                False,
            ),
            (
                "state_changes",
                frozenset({"observed_robot_state", "observed_visual"}),
                True,
                True,
            ),
            ("unchanged_results", frozenset({"observed_visual"}), True, True),
            (
                "release_observations",
                frozenset({"observed_robot_state", "observed_visual"}),
                True,
                False,
            ),
            ("feedback_observations", frozenset({"observed_visual"}), True, False),
        )
        admitted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_ids: set[str] = set(known_chunk_facts)
        admitted_by_field: dict[str, Any] = {
            "attempted_action": None,
            "observed_execution": [],
            "state_changes": [],
            "unchanged_results": [],
            "release_observations": [],
            "feedback_observations": [],
        }
        for field_name, sources, is_array, requires_change in specifications:
            values = summary.get(field_name)
            if is_array:
                if not isinstance(values, list):
                    issues.append(
                        LayerIssue(
                            "malformed_output",
                            f"summary.{field_name}",
                            "must be an array",
                        )
                    )
                    continue
                candidates = values
            else:
                candidates = [values]
            for index, value in enumerate(candidates):
                path = f"summary.{field_name}" + (f"[{index}]" if is_array else "")
                fact, fact_issues = _validate_fact(
                    value,
                    path=path,
                    allowed_sources=sources,
                    source_refs=source_refs,
                    evidence_segments=evidence_segments,
                    allowed_segments=frozenset({request.segment_id}),
                )
                if (
                    fact is not None
                    and requires_change
                    and fact["temporal_scope"] != "before_after"
                ):
                    fact_issues.append(
                        LayerIssue(
                            "missing_before_after_evidence",
                            path,
                            "state-change claims require before/after evidence",
                            tuple(fact["supporting_evidence_refs"]),
                        )
                    )
                    fact = None
                if fact is None:
                    rejected.append(
                        {
                            "fact": _strict_json(value, label=path),
                            "issues": [issue.to_dict() for issue in fact_issues],
                        }
                    )
                    issues.extend(fact_issues)
                    continue
                if fact["fact_id"] in seen_ids:
                    duplicate = LayerIssue(
                        "duplicate_fact_id",
                        f"{path}.fact_id",
                        "fact id must be unique within summary",
                    )
                    rejected.append({"fact": fact, "issues": [duplicate.to_dict()]})
                    issues.append(duplicate)
                    continue
                seen_ids.add(str(fact["fact_id"]))
                admitted.append(fact)
                if is_array:
                    admitted_by_field[field_name].append(fact)
                else:
                    admitted_by_field[field_name] = fact

        summary_admitted = (
            summary
            if not any(
                issue.offending_field
                in {
                    "status",
                    "abstention_reason",
                    "summary",
                    "summary.summary_id",
                    "summary.segment_id",
                    "summary.confidence",
                    "summary.supporting_chunk_fact_ids",
                }
                for issue in issues
            )
            and any(fact["source_type"] == "annotation" for fact in admitted)
            and any(
                fact["source_type"] in {"observed_robot_state", "observed_visual"}
                for fact in admitted
            )
            else None
        )
        if summary_admitted is None:
            issues.append(
                LayerIssue(
                    "subtask_summary_not_grounded",
                    "summary",
                    "summary needs distinct annotation intent and observed execution/result facts",
                )
            )
        else:
            summary_admitted["uncertainties"] = list(uncertainty_values)
            for field_name, value in admitted_by_field.items():
                summary_admitted[field_name] = deepcopy(value)
        return self._result(
            request,
            output,
            summary_admitted,
            admitted,
            rejected,
            issues,
        )

    @staticmethod
    def _result(
        request: SubtaskSummaryRequest,
        output: LayerModelOutput,
        summary: Mapping[str, Any] | None,
        admitted: Sequence[Mapping[str, Any]],
        rejected: Sequence[Mapping[str, Any]],
        issues: Sequence[LayerIssue],
    ) -> SubtaskSummaryResult:
        abstention = (
            None
            if summary is not None
            else _layer_abstention(
                layer="subtask_summary",
                scope_id=request.segment_id,
                output=output,
                issues=_dedupe_issues(issues),
            )
        )
        return SubtaskSummaryResult(
            segment_id=request.segment_id,
            summary=None if summary is None else deepcopy(dict(summary)),
            admitted_episode_facts=tuple(deepcopy(list(admitted))),
            rejected_facts=tuple(deepcopy(list(rejected))),
            abstention=abstention,
            output_sha256=output.output_sha256,
        )


@dataclass(frozen=True, slots=True)
class ProcedureDraftRequest:
    """Layer-3 request; its draft is private until final candidate admission."""

    trajectory_id: str
    instruction: str
    ordered_subtask_summaries: tuple[Mapping[str, Any], ...]
    episode_specific_facts: tuple[Mapping[str, Any], ...]
    cross_segment_visual_evidence: tuple[EvidenceInput, ...]
    trajectory_outcome: Mapping[str, Any]
    provenance: Mapping[str, Any]
    schema: str = _DRAFT_REQUEST_SCHEMA
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.trajectory_id, label="draft_request.trajectory_id")
        _nonempty(self.instruction, label="draft_request.instruction")
        if self.schema not in {_DRAFT_REQUEST_SCHEMA, "tcm/procedure_draft_request/v1"} or self.schema_version != 1:
            raise SchemaValidationError("draft_request: unsupported schema or version")
        summaries = _mapping_tuple(
            self.ordered_subtask_summaries,
            label="draft_request.ordered_subtask_summaries",
        )
        seen_segments: set[str] = set()
        for index, summary in enumerate(summaries):
            if set(summary) != SubtaskSummaryQualityGate._SUMMARY_FIELDS:
                raise SchemaValidationError(
                    f"draft_request.ordered_subtask_summaries[{index}]: fields do not match admitted summary schema"
                )
            segment_id = _nonempty(
                summary.get("segment_id"),
                label=f"draft_request.ordered_subtask_summaries[{index}].segment_id",
            )
            if segment_id in seen_segments:
                raise SchemaValidationError(
                    "draft_request.ordered_subtask_summaries: duplicate segment"
                )
            seen_segments.add(segment_id)
            for field_name in (
                "attempted_action",
                "observed_execution",
                "state_changes",
                "unchanged_results",
                "release_observations",
                "feedback_observations",
            ):
                raw_facts = (
                    [summary[field_name]]
                    if field_name == "attempted_action"
                    else summary[field_name]
                )
                if not isinstance(raw_facts, list):
                    raise SchemaValidationError(
                        f"draft_request.ordered_subtask_summaries[{index}].{field_name}: invalid fact container"
                    )
                for fact in raw_facts:
                    if (
                        not isinstance(fact, Mapping)
                        or set(fact) != _FACT_FIELDS
                        or fact.get("transferable_allowed") is not False
                        or fact.get("supporting_segment_refs") != [segment_id]
                    ):
                        raise SchemaValidationError(
                            f"draft_request.ordered_subtask_summaries[{index}].{field_name}: contains a non-admitted fact"
                        )
        facts = _mapping_tuple(
            self.episode_specific_facts,
            label="draft_request.episode_specific_facts",
        )
        seen_facts: set[str] = set()
        for index, fact in enumerate(facts):
            if (
                set(fact) != _FACT_FIELDS
                or fact.get("transferable_allowed") is not False
            ):
                raise SchemaValidationError(
                    f"draft_request.episode_specific_facts[{index}]: invalid private fact"
                )
            fact_id = _nonempty(
                fact.get("fact_id"),
                label=f"draft_request.episode_specific_facts[{index}].fact_id",
            )
            if fact_id in seen_facts:
                raise SchemaValidationError(
                    "draft_request.episode_specific_facts: duplicate fact id"
                )
            seen_facts.add(fact_id)
        evidence = _evidence_tuple(
            self.cross_segment_visual_evidence,
            label="draft_request.cross_segment_visual_evidence",
        )
        for index, item in enumerate(evidence):
            segment_id = item.payload.get("segment_id")
            if not isinstance(segment_id, str) or segment_id not in seen_segments:
                raise SchemaValidationError(
                    f"draft_request.cross_segment_visual_evidence[{index}].payload.segment_id: unknown segment"
                )
        outcome = project_trajectory_outcome(self.trajectory_outcome)
        provenance = ProvenanceV1.from_dict(
            _mapping(self.provenance, label="draft_request.provenance")
        ).to_dict()
        object.__setattr__(self, "ordered_subtask_summaries", summaries)
        object.__setattr__(self, "episode_specific_facts", facts)
        object.__setattr__(self, "cross_segment_visual_evidence", evidence)
        object.__setattr__(self, "trajectory_outcome", outcome)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcedureDraftRequest":
        data = _mapping(payload, label="draft_request")
        expected = {
            "schema",
            "schema_version",
            "trajectory_id",
            "instruction",
            "ordered_subtask_summaries",
            "episode_specific_facts",
            "cross_segment_visual_evidence",
            "trajectory_outcome",
            "provenance",
            "source_semantics",
        }
        _exact_fields(data, expected, label="draft_request")
        _schema_header(data, _DRAFT_REQUEST_SCHEMA, label="draft_request")
        _validate_source_semantics(
            data["source_semantics"],
            label="draft_request.source_semantics",
        )
        return cls(
            schema=data["schema"],
            schema_version=data["schema_version"],
            trajectory_id=data["trajectory_id"],
            instruction=data["instruction"],
            ordered_subtask_summaries=tuple(data["ordered_subtask_summaries"]),
            episode_specific_facts=tuple(data["episode_specific_facts"]),
            cross_segment_visual_evidence=tuple(data["cross_segment_visual_evidence"]),
            trajectory_outcome=data["trajectory_outcome"],
            provenance=data["provenance"],
        )

    @property
    def valid_segment_refs(self) -> tuple[str, ...]:
        return tuple(
            str(summary["segment_id"]) for summary in self.ordered_subtask_summaries
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "instruction": self.instruction,
            "ordered_subtask_summaries": [
                deepcopy(dict(item)) for item in self.ordered_subtask_summaries
            ],
            "episode_specific_facts": [
                deepcopy(dict(item)) for item in self.episode_specific_facts
            ],
            "cross_segment_visual_evidence": [
                item.to_dict() for item in self.cross_segment_visual_evidence
            ],
            "trajectory_outcome": deepcopy(dict(self.trajectory_outcome)),
            "provenance": deepcopy(dict(self.provenance)),
            "source_semantics": hierarchical_source_semantics(),
        }


@dataclass(frozen=True, slots=True)
class ProcedureDraftResult:
    private_procedure_draft: Mapping[str, Any] | None
    summary: str
    confidence: str
    admitted_episode_facts: tuple[Mapping[str, Any], ...]
    rejected_facts: tuple[Mapping[str, Any], ...]
    supporting_evidence_refs: tuple[str, ...]
    supporting_segment_refs: tuple[str, ...]
    abstention: Mapping[str, Any] | None
    output_sha256: str

    @property
    def admitted(self) -> bool:
        return self.private_procedure_draft is not None and self.abstention is None

    @property
    def planner_visible_guidance(self) -> None:
        """Drafts are intentionally never planner-visible."""

        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "private_procedure_draft": (
                None
                if self.private_procedure_draft is None
                else deepcopy(dict(self.private_procedure_draft))
            ),
            "summary": self.summary,
            "confidence": self.confidence,
            "admitted_episode_facts": [
                deepcopy(dict(item)) for item in self.admitted_episode_facts
            ],
            "rejected_facts": [deepcopy(dict(item)) for item in self.rejected_facts],
            "supporting_evidence_refs": list(self.supporting_evidence_refs),
            "supporting_segment_refs": list(self.supporting_segment_refs),
            "abstention": (
                None if self.abstention is None else deepcopy(dict(self.abstention))
            ),
            "output_sha256": self.output_sha256,
        }


def _valid_unique_strings(value: Any, *, nonempty: bool) -> bool:
    if not isinstance(value, list) or (nonempty and not value):
        return False
    return all(isinstance(item, str) and item.strip() for item in value) and len(
        value
    ) == len(set(value))


def _validate_transferable_guidance(value: Any) -> list[LayerIssue]:
    """Early draft shape gate; the existing final quality gate stays authoritative."""

    issues: list[LayerIssue] = []
    if not isinstance(value, Mapping) or set(value) != {
        "condition",
        "guidance",
        "predicted_effects",
    }:
        return [
            LayerIssue(
                "malformed_procedure_draft",
                "transferable_guidance",
                "expected condition, guidance, and predicted_effects",
            )
        ]
    condition = value.get("condition")
    if (
        not isinstance(condition, Mapping)
        or set(condition) != {"task_family", "subtask_type", "preconditions"}
        or not isinstance(condition.get("task_family"), str)
        or not condition["task_family"].strip()
        or not isinstance(condition.get("subtask_type"), str)
        or not condition["subtask_type"].strip()
        or not _valid_unique_strings(condition.get("preconditions"), nonempty=False)
    ):
        issues.append(
            LayerIssue(
                "malformed_procedure_draft",
                "transferable_guidance.condition",
                "invalid condition",
            )
        )
    guidance = value.get("guidance")
    if not isinstance(guidance, Mapping) or set(guidance) != {
        "ordered_steps",
        "feedback_policy",
        "avoid",
    }:
        issues.append(
            LayerIssue(
                "malformed_procedure_draft",
                "transferable_guidance.guidance",
                "invalid guidance fields",
            )
        )
        return issues
    steps = guidance.get("ordered_steps")
    if (
        not isinstance(steps, list)
        or not steps
        or any(
            not isinstance(step, Mapping)
            or set(step) != {"action_pattern", "instruction"}
            or not isinstance(step.get("action_pattern"), str)
            or not step["action_pattern"].strip()
            or not isinstance(step.get("instruction"), str)
            or not step["instruction"].strip()
            for step in steps
        )
    ):
        issues.append(
            LayerIssue(
                "malformed_procedure_draft",
                "transferable_guidance.guidance.ordered_steps",
                "invalid ordered steps",
            )
        )
    if not _valid_unique_strings(guidance.get("avoid"), nonempty=True):
        issues.append(
            LayerIssue(
                "missing_repeat_prevention",
                "transferable_guidance.guidance.avoid",
                "avoid must be non-empty",
            )
        )
    policy = guidance.get("feedback_policy")
    expected_policy = {
        "state_variables",
        "attempt_tracking",
        "observation_rules",
        "failure_branch",
        "success_branch",
        "termination_conditions",
    }
    if not isinstance(policy, Mapping) or set(policy) != expected_policy:
        issues.append(
            LayerIssue(
                "malformed_procedure_draft",
                "transferable_guidance.guidance.feedback_policy",
                "invalid feedback policy",
            )
        )
    else:
        states = policy.get("state_variables")
        tracking = policy.get("attempt_tracking")
        observations = policy.get("observation_rules")
        if not _valid_unique_strings(states, nonempty=True):
            issues.append(
                LayerIssue(
                    "missing_state_tracking",
                    "transferable_guidance.guidance.feedback_policy.state_variables",
                    "state variables required",
                )
            )
        if (
            not isinstance(tracking, Mapping)
            or set(tracking)
            != {"state_variable", "record_after_attempt", "exclude_previously_recorded"}
            or tracking.get("state_variable") not in (states or [])
            or tracking.get("record_after_attempt") is not True
            or tracking.get("exclude_previously_recorded") is not True
        ):
            issues.append(
                LayerIssue(
                    "missing_repeat_prevention",
                    "transferable_guidance.guidance.feedback_policy.attempt_tracking",
                    "attempt tracking must record and exclude prior attempts",
                )
            )
        if (
            not isinstance(observations, list)
            or not observations
            or any(
                not isinstance(rule, Mapping)
                or set(rule) != {"after_action_pattern", "observe_signal"}
                or not isinstance(rule.get("after_action_pattern"), str)
                or not rule["after_action_pattern"].strip()
                or not isinstance(rule.get("observe_signal"), str)
                or not rule["observe_signal"].strip()
                for rule in observations
            )
        ):
            issues.append(
                LayerIssue(
                    "missing_observation_rule",
                    "transferable_guidance.guidance.feedback_policy.observation_rules",
                    "post-action observation required",
                )
            )
        for key, code in (
            ("failure_branch", "missing_failure_branch"),
            ("success_branch", "missing_success_branch"),
            ("termination_conditions", "missing_termination_condition"),
        ):
            if not _valid_unique_strings(policy.get(key), nonempty=True):
                issues.append(
                    LayerIssue(
                        code,
                        f"transferable_guidance.guidance.feedback_policy.{key}",
                        f"{key} must be non-empty",
                    )
                )
    effects = value.get("predicted_effects")
    if (
        not isinstance(effects, list)
        or not effects
        or any(
            not isinstance(effect, Mapping)
            or set(effect) != {"effect", "validation_status"}
            or not isinstance(effect.get("effect"), str)
            or not effect["effect"].strip()
            or effect.get("validation_status")
            not in {"evidence_supported", "pending_validation"}
            for effect in effects
        )
    ):
        issues.append(
            LayerIssue(
                "malformed_procedure_draft",
                "transferable_guidance.predicted_effects",
                "invalid predicted effects",
            )
        )
    return issues


class ProcedureDraftQualityGate:
    """Validate Layer-3 draft while retaining independently grounded facts."""

    _FIELDS = {
        "schema",
        "schema_version",
        "status",
        "abstention_reason",
        "summary",
        "confidence",
        "cross_segment_facts",
        "transferable_guidance",
        "supporting_fact_ids",
        "supporting_evidence_refs",
        "supporting_segment_refs",
        "uncertainties",
    }

    def __init__(self) -> None:
        self._leakage_critic = DeterministicLeakageCritic()

    def evaluate(
        self,
        request: ProcedureDraftRequest,
        output: LayerModelOutput,
    ) -> ProcedureDraftResult:
        payload, issues = _parse_output(
            output,
            expected_schema=_DRAFT_OUTPUT_SCHEMA,
            expected_fields=self._FIELDS,
        )
        if payload is None:
            return self._result(
                request, output, None, "", "low", (), (), (), (), issues
            )
        status = payload.get("status")
        reason = payload.get("abstention_reason")
        summary = payload.get("summary")
        confidence = payload.get("confidence")
        if status not in {"candidate", "abstained"}:
            issues.append(LayerIssue("malformed_output", "status", "invalid status"))
        if not isinstance(reason, str) or (
            status == "abstained" and not reason.strip()
        ):
            issues.append(
                LayerIssue("malformed_output", "abstention_reason", "invalid reason")
            )
        if status == "candidate" and reason:
            issues.append(
                LayerIssue(
                    "malformed_output",
                    "abstention_reason",
                    "candidate must use an empty reason",
                )
            )
        if not isinstance(summary, str) or not summary.strip():
            issues.append(
                LayerIssue("malformed_output", "summary", "summary must be non-empty")
            )
            summary = ""
        if confidence not in _CONFIDENCE:
            issues.append(
                LayerIssue("malformed_output", "confidence", "invalid confidence")
            )
            confidence = "low"

        segment_ids = request.valid_segment_refs
        evidence_segments: dict[str, str] = {}
        source_refs: dict[str, set[str]] = {
            "annotation": set(),
            "observed_robot_state": set(),
            "observed_visual": set(),
            "cross_segment_inference": set(),
        }
        for fact in request.episode_specific_facts:
            source_type = str(fact["source_type"])
            fact_segments = fact["supporting_segment_refs"]
            for ref in fact["supporting_evidence_refs"]:
                source_refs[source_type].add(str(ref))
                source_refs["cross_segment_inference"].add(str(ref))
                if len(fact_segments) == 1:
                    evidence_segments[str(ref)] = str(fact_segments[0])
        for item in request.cross_segment_visual_evidence:
            segment_id = str(item.payload["segment_id"])
            source_refs["observed_visual"].add(item.evidence_ref)
            source_refs["cross_segment_inference"].add(item.evidence_ref)
            evidence_segments[item.evidence_ref] = segment_id
        frozen_refs = {key: frozenset(value) for key, value in source_refs.items()}
        admitted_cross: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        raw_cross = payload.get("cross_segment_facts")
        if not isinstance(raw_cross, list):
            issues.append(
                LayerIssue(
                    "malformed_output", "cross_segment_facts", "must be an array"
                )
            )
            raw_cross = []
        seen_ids = {str(fact["fact_id"]) for fact in request.episode_specific_facts}
        for index, value in enumerate(raw_cross):
            fact, fact_issues = _validate_fact(
                value,
                path=f"cross_segment_facts[{index}]",
                allowed_sources=frozenset({"cross_segment_inference"}),
                source_refs=frozen_refs,
                evidence_segments=evidence_segments,
                allowed_segments=frozenset(segment_ids),
            )
            if fact is not None and fact["fact_id"] in seen_ids:
                fact_issues.append(
                    LayerIssue(
                        "duplicate_fact_id",
                        f"cross_segment_facts[{index}].fact_id",
                        "fact id already exists",
                    )
                )
                fact = None
            if fact is None:
                rejected.append(
                    {
                        "fact": _strict_json(
                            value, label=f"cross_segment_facts[{index}]"
                        ),
                        "issues": [issue.to_dict() for issue in fact_issues],
                    }
                )
                issues.extend(fact_issues)
            else:
                seen_ids.add(str(fact["fact_id"]))
                admitted_cross.append(fact)

        try:
            support_fact_ids = _string_tuple(
                payload.get("supporting_fact_ids"),
                label="supporting_fact_ids",
                nonempty=status == "candidate",
            )
            support_refs = _string_tuple(
                payload.get("supporting_evidence_refs"),
                label="supporting_evidence_refs",
                nonempty=status == "candidate",
            )
            support_segments = _string_tuple(
                payload.get("supporting_segment_refs"),
                label="supporting_segment_refs",
                nonempty=status == "candidate",
            )
            _string_tuple(payload.get("uncertainties"), label="uncertainties")
        except SchemaValidationError as exc:
            issues.append(LayerIssue("malformed_output", "backend_output", str(exc)))
            support_fact_ids = ()
            support_refs = ()
            support_segments = ()
        if not set(support_fact_ids).issubset(seen_ids):
            issues.append(
                LayerIssue(
                    "unknown_fact_ref",
                    "supporting_fact_ids",
                    "draft cites unadmitted facts",
                )
            )
        if not set(support_refs).issubset(frozen_refs["cross_segment_inference"]):
            issues.append(
                LayerIssue(
                    "unknown_evidence_ref",
                    "supporting_evidence_refs",
                    "draft cites unknown evidence",
                )
            )
        if not set(support_segments).issubset(set(segment_ids)):
            issues.append(
                LayerIssue(
                    "unknown_segment_ref",
                    "supporting_segment_refs",
                    "draft cites unknown segments",
                )
            )
        if status == "candidate" and len(set(support_segments)) < 2:
            issues.append(
                LayerIssue(
                    "missing_cross_segment_support",
                    "supporting_segment_refs",
                    "procedure needs at least two ordered segments",
                )
            )

        guidance = payload.get("transferable_guidance")
        draft: dict[str, Any] | None = None
        if status == "abstained":
            if guidance is not None:
                issues.append(
                    LayerIssue(
                        "malformed_output",
                        "transferable_guidance",
                        "abstained output must use null guidance",
                    )
                )
            issues.append(
                LayerIssue("backend_abstained", "abstention_reason", str(reason))
            )
        else:
            issues.extend(_validate_transferable_guidance(guidance))
            if isinstance(guidance, Mapping):
                critic = self._leakage_critic.audit(
                    guidance,
                    (*request.episode_specific_facts, *admitted_cross),
                )
                for issue in critic.issues:
                    issues.append(
                        LayerIssue(
                            issue.code,
                            issue.offending_field,
                            issue.detail,
                            issue.evidence_refs,
                        )
                    )
            fatal_codes = {
                "malformed_output",
                "malformed_procedure_draft",
                "missing_repeat_prevention",
                "missing_state_tracking",
                "missing_observation_rule",
                "missing_failure_branch",
                "missing_success_branch",
                "missing_termination_condition",
                "missing_cross_segment_support",
                "unknown_fact_ref",
                "unknown_evidence_ref",
                "unknown_segment_ref",
                "episode_specific_value_leakage",
            }
            if isinstance(guidance, Mapping) and not any(
                issue.code in fatal_codes for issue in issues
            ):
                draft = _strict_json(dict(guidance), label="transferable_guidance")
        return self._result(
            request,
            output,
            draft,
            str(summary),
            str(confidence),
            admitted_cross,
            rejected,
            support_refs,
            support_segments,
            issues,
        )

    @staticmethod
    def _result(
        request: ProcedureDraftRequest,
        output: LayerModelOutput,
        draft: Mapping[str, Any] | None,
        summary: str,
        confidence: str,
        facts: Sequence[Mapping[str, Any]],
        rejected: Sequence[Mapping[str, Any]],
        evidence_refs: Sequence[str],
        segment_refs: Sequence[str],
        issues: Sequence[LayerIssue],
    ) -> ProcedureDraftResult:
        abstention = (
            None
            if draft is not None
            else _layer_abstention(
                layer="procedure_draft",
                scope_id=request.trajectory_id,
                output=output,
                issues=_dedupe_issues(issues),
            )
        )
        return ProcedureDraftResult(
            private_procedure_draft=None if draft is None else deepcopy(dict(draft)),
            summary=summary,
            confidence=confidence,
            admitted_episode_facts=tuple(deepcopy(list(facts))),
            rejected_facts=tuple(deepcopy(list(rejected))),
            supporting_evidence_refs=tuple(evidence_refs),
            supporting_segment_refs=tuple(segment_refs),
            abstention=abstention,
            output_sha256=output.output_sha256,
        )


def _whole_request_evidence_registry(
    request: WholeTrajectoryReflectionRequest,
) -> tuple[dict[str, Any], ...]:
    """Mirror the final quality gate's registry in a serializable form."""

    _, _, source_types, _, evidence_segments = ReflectionQualityGate._evidence_registry(
        request
    )
    result: list[dict[str, Any]] = []
    for evidence_ref in sorted(source_types):
        sources = sorted(source_types[evidence_ref] & set(_SOURCE_TYPES))
        segment_id = evidence_segments.get(evidence_ref)
        if sources and isinstance(segment_id, str) and segment_id:
            result.append(
                {
                    "evidence_ref": evidence_ref,
                    "source_types": sources,
                    "segment_id": segment_id,
                }
            )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AttributionPassRequest:
    """Independent attribution input for one private procedure draft."""

    trajectory_id: str
    procedure_draft: Mapping[str, Any]
    episode_specific_facts: tuple[Mapping[str, Any], ...]
    valid_evidence_refs: tuple[str, ...]
    valid_segment_refs: tuple[str, ...]
    evidence_registry: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    schema: str = _ATTRIBUTION_REQUEST_SCHEMA
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.trajectory_id, label="attribution_request.trajectory_id")
        if self.schema not in {_ATTRIBUTION_REQUEST_SCHEMA, "tcm/evidence_attribution_request/v1"} or self.schema_version != 1:
            raise SchemaValidationError(
                "attribution_request: unsupported schema or version"
            )
        draft = _mapping(
            self.procedure_draft, label="attribution_request.procedure_draft"
        )
        draft_issues = _validate_transferable_guidance(draft)
        if draft_issues:
            raise SchemaValidationError(
                "attribution_request.procedure_draft: draft failed structural admission"
            )
        facts = _mapping_tuple(
            self.episode_specific_facts,
            label="attribution_request.episode_specific_facts",
        )
        seen_facts: set[str] = set()
        for index, fact in enumerate(facts):
            if (
                set(fact) != _FACT_FIELDS
                or fact.get("transferable_allowed") is not False
            ):
                raise SchemaValidationError(
                    f"attribution_request.episode_specific_facts[{index}]: invalid private fact"
                )
            fact_id = _nonempty(
                fact.get("fact_id"),
                label=f"attribution_request.episode_specific_facts[{index}].fact_id",
            )
            if fact_id in seen_facts:
                raise SchemaValidationError(
                    "attribution_request.episode_specific_facts: duplicate fact id"
                )
            seen_facts.add(fact_id)
        evidence_refs = _string_tuple(
            self.valid_evidence_refs,
            label="attribution_request.valid_evidence_refs",
        )
        segments = _string_tuple(
            self.valid_segment_refs,
            label="attribution_request.valid_segment_refs",
            nonempty=True,
        )
        registry = _mapping_tuple(
            self.evidence_registry,
            label="attribution_request.evidence_registry",
        )
        seen_refs: set[str] = set()
        for index, item in enumerate(registry):
            _exact_fields(
                item,
                {"evidence_ref", "source_types", "segment_id"},
                label=f"attribution_request.evidence_registry[{index}]",
            )
            ref = _nonempty(
                item["evidence_ref"],
                label=f"attribution_request.evidence_registry[{index}].evidence_ref",
            )
            if ref in seen_refs:
                raise SchemaValidationError(
                    "attribution_request.evidence_registry: duplicate ref"
                )
            seen_refs.add(ref)
            source_values = _string_tuple(
                item["source_types"],
                label=f"attribution_request.evidence_registry[{index}].source_types",
                nonempty=True,
            )
            if not set(source_values).issubset(
                {"annotation", "observed_robot_state", "observed_visual"}
            ):
                raise SchemaValidationError(
                    f"attribution_request.evidence_registry[{index}]: invalid source type"
                )
            if item["segment_id"] not in segments:
                raise SchemaValidationError(
                    f"attribution_request.evidence_registry[{index}]: unknown segment"
                )
        if set(evidence_refs) != seen_refs:
            raise SchemaValidationError(
                "attribution_request.valid_evidence_refs: must equal registry refs"
            )
        provenance = ProvenanceV1.from_dict(
            _mapping(self.provenance, label="attribution_request.provenance")
        ).to_dict()
        object.__setattr__(self, "procedure_draft", draft)
        object.__setattr__(self, "episode_specific_facts", facts)
        object.__setattr__(self, "valid_evidence_refs", evidence_refs)
        object.__setattr__(self, "valid_segment_refs", segments)
        object.__setattr__(self, "evidence_registry", registry)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_whole_request(
        cls,
        request: WholeTrajectoryReflectionRequest,
        *,
        procedure_draft: Mapping[str, Any],
        episode_specific_facts: Sequence[Mapping[str, Any]],
    ) -> "AttributionPassRequest":
        registry = _whole_request_evidence_registry(request)
        return cls(
            trajectory_id=request.trajectory_id,
            procedure_draft=procedure_draft,
            episode_specific_facts=tuple(episode_specific_facts),
            valid_evidence_refs=tuple(str(item["evidence_ref"]) for item in registry),
            valid_segment_refs=tuple(
                str(segment["segment_id"]) for segment in request.segments
            ),
            evidence_registry=registry,
            provenance=request.provenance,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttributionPassRequest":
        data = _mapping(payload, label="attribution_request")
        expected = {
            "schema",
            "schema_version",
            "trajectory_id",
            "procedure_draft",
            "episode_specific_facts",
            "valid_evidence_refs",
            "valid_segment_refs",
            "evidence_registry",
            "provenance",
            "source_semantics",
            "semantic_leakage_audit",
        }
        _exact_fields(data, expected, label="attribution_request")
        _schema_header(data, _ATTRIBUTION_REQUEST_SCHEMA, label="attribution_request")
        _validate_source_semantics(
            data["source_semantics"],
            label="attribution_request.source_semantics",
        )
        _validate_semantic_leakage_contract(
            data["semantic_leakage_audit"],
            label="attribution_request.semantic_leakage_audit",
        )
        return cls(
            schema=data["schema"],
            schema_version=data["schema_version"],
            trajectory_id=data["trajectory_id"],
            procedure_draft=data["procedure_draft"],
            episode_specific_facts=tuple(data["episode_specific_facts"]),
            valid_evidence_refs=tuple(data["valid_evidence_refs"]),
            valid_segment_refs=tuple(data["valid_segment_refs"]),
            evidence_registry=tuple(data["evidence_registry"]),
            provenance=data["provenance"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "procedure_draft": deepcopy(dict(self.procedure_draft)),
            "episode_specific_facts": [
                deepcopy(dict(item)) for item in self.episode_specific_facts
            ],
            "valid_evidence_refs": list(self.valid_evidence_refs),
            "valid_segment_refs": list(self.valid_segment_refs),
            "evidence_registry": [
                deepcopy(dict(item)) for item in self.evidence_registry
            ],
            "provenance": deepcopy(dict(self.provenance)),
            "source_semantics": hierarchical_source_semantics(),
            "semantic_leakage_audit": semantic_leakage_audit_contract(),
        }


@dataclass(frozen=True, slots=True)
class AttributionPassResult:
    attributions: tuple[Mapping[str, Any], ...]
    leakage_audit: Mapping[str, Any]
    abstention: Mapping[str, Any] | None
    output_sha256: str

    @property
    def admitted(self) -> bool:
        return bool(self.attributions) and self.abstention is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributions": [deepcopy(dict(item)) for item in self.attributions],
            "leakage_audit": deepcopy(dict(self.leakage_audit)),
            "abstention": (
                None if self.abstention is None else deepcopy(dict(self.abstention))
            ),
            "output_sha256": self.output_sha256,
        }


def required_attribution_paths(draft: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every planner-facing field that requires independent evidence."""

    guidance = draft.get("guidance", {})
    effects = draft.get("predicted_effects", [])
    paths: list[str] = ["/condition"]
    if isinstance(guidance, Mapping):
        steps = guidance.get("ordered_steps", [])
        if isinstance(steps, list):
            paths.extend(
                f"/guidance/ordered_steps/{index}" for index in range(len(steps))
            )
        if isinstance(guidance.get("feedback_policy"), Mapping):
            paths.extend(
                (
                    "/guidance/feedback_policy/state_variables",
                    "/guidance/feedback_policy/attempt_tracking",
                    "/guidance/feedback_policy/observation_rules",
                    "/guidance/feedback_policy/failure_branch",
                    "/guidance/feedback_policy/success_branch",
                    "/guidance/feedback_policy/termination_conditions",
                )
            )
    paths.append("/guidance/avoid")
    if isinstance(effects, list):
        paths.extend(f"/predicted_effects/{index}" for index in range(len(effects)))
    return tuple(paths)


class AttributionQualityGate:
    """Require complete, source-correct attribution before final admission."""

    _FIELDS = {
        "schema",
        "schema_version",
        "status",
        "abstention_reason",
        "attributions",
        "episode_specific_leakage",
        "leakage_reasons",
        "leakage_confidence",
    }

    def evaluate(
        self,
        request: AttributionPassRequest,
        output: LayerModelOutput,
    ) -> AttributionPassResult:
        payload, issues = _parse_output(
            output,
            expected_schema=_ATTRIBUTION_OUTPUT_SCHEMA,
            expected_fields=self._FIELDS,
        )
        if payload is None:
            return self._result(request, output, (), issues)
        status = payload.get("status")
        reason = payload.get("abstention_reason")
        if status not in {"attributed", "abstained"}:
            issues.append(LayerIssue("malformed_output", "status", "invalid status"))
        if not isinstance(reason, str) or (
            status == "abstained" and not reason.strip()
        ):
            issues.append(
                LayerIssue("malformed_output", "abstention_reason", "invalid reason")
            )
        if status == "attributed" and reason:
            issues.append(
                LayerIssue(
                    "malformed_output",
                    "abstention_reason",
                    "attributed output must use an empty reason",
                )
            )
        leakage = payload.get("episode_specific_leakage")
        leakage_confidence = payload.get("leakage_confidence")
        try:
            leakage_reasons = _string_tuple(
                payload.get("leakage_reasons"),
                label="leakage_reasons",
            )
        except SchemaValidationError as exc:
            issues.append(LayerIssue("malformed_output", "leakage_reasons", str(exc)))
            leakage_reasons = ()
        if not isinstance(leakage, bool):
            issues.append(
                LayerIssue(
                    "malformed_output",
                    "episode_specific_leakage",
                    "leakage verdict must be boolean",
                )
            )
        if leakage_confidence not in _CONFIDENCE:
            issues.append(
                LayerIssue(
                    "malformed_output",
                    "leakage_confidence",
                    "invalid leakage confidence",
                )
            )
        if leakage is True and not leakage_reasons:
            issues.append(
                LayerIssue(
                    "malformed_output",
                    "leakage_reasons",
                    "positive leakage verdict requires reasons",
                )
            )
        if (
            leakage is False
            and leakage_confidence in {"medium", "high"}
            and leakage_reasons
        ):
            issues.append(
                LayerIssue(
                    "malformed_output",
                    "leakage_reasons",
                    "confident negative leakage verdict must not invent reasons",
                )
            )
        if status == "attributed" and (
            leakage is not False or leakage_confidence not in {"medium", "high"}
        ):
            issues.append(
                LayerIssue(
                    "episode_specific_leakage"
                    if leakage is True
                    else "leakage_audit_uncertain",
                    "episode_specific_leakage",
                    "attribution requires a confident negative semantic leakage audit",
                )
            )
        raw_attributions = payload.get("attributions")
        if not isinstance(raw_attributions, list):
            issues.append(
                LayerIssue("malformed_output", "attributions", "must be an array")
            )
            raw_attributions = []
        if status == "abstained":
            if raw_attributions:
                issues.append(
                    LayerIssue(
                        "malformed_output",
                        "attributions",
                        "abstained output must not contain attributions",
                    )
                )
            issues.append(
                LayerIssue("backend_abstained", "abstention_reason", str(reason))
            )
            return self._result(request, output, (), issues)

        registry = {
            str(item["evidence_ref"]): {
                "source_types": set(item["source_types"]),
                "segment_id": str(item["segment_id"]),
            }
            for item in request.evidence_registry
        }
        admitted: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        required_paths = set(required_attribution_paths(request.procedure_draft))
        for index, value in enumerate(raw_attributions):
            path = f"attributions[{index}]"
            if not isinstance(value, Mapping) or set(value) != _ATTRIBUTION_FIELDS:
                issues.append(
                    LayerIssue(
                        "malformed_output",
                        path,
                        "attribution fields do not match schema",
                    )
                )
                continue
            item = _strict_json(dict(value), label=path)
            attribution_id = item.get("attribution_id")
            target_path = item.get("target_path")
            source_type = item.get("source_type")
            refs = item.get("supporting_evidence_refs")
            segments = item.get("supporting_segment_refs")
            prediction = item.get("prediction_status")
            local: list[LayerIssue] = []
            if (
                not isinstance(attribution_id, str)
                or not attribution_id.strip()
                or attribution_id in seen_ids
            ):
                local.append(
                    LayerIssue(
                        "malformed_output",
                        f"{path}.attribution_id",
                        "invalid or duplicate id",
                    )
                )
            if target_path not in required_paths or target_path in seen_paths:
                local.append(
                    LayerIssue(
                        "invalid_attribution_target",
                        f"{path}.target_path",
                        "target is unknown or duplicated",
                    )
                )
            if not isinstance(item.get("claim"), str) or not item["claim"].strip():
                local.append(
                    LayerIssue("malformed_output", f"{path}.claim", "invalid claim")
                )
            if source_type not in _SOURCE_TYPES:
                local.append(
                    LayerIssue(
                        "malformed_output", f"{path}.source_type", "invalid source type"
                    )
                )
            if item.get("confidence") not in _CONFIDENCE:
                local.append(
                    LayerIssue(
                        "malformed_output", f"{path}.confidence", "invalid confidence"
                    )
                )
            if not isinstance(item.get("uncertainty"), str):
                local.append(
                    LayerIssue(
                        "malformed_output", f"{path}.uncertainty", "must be a string"
                    )
                )
            if item.get("temporal_scope") not in _TEMPORAL_SCOPES:
                local.append(
                    LayerIssue(
                        "malformed_output",
                        f"{path}.temporal_scope",
                        "invalid temporal scope",
                    )
                )
            if prediction not in {"evidence_supported", "pending_validation"}:
                local.append(
                    LayerIssue(
                        "malformed_output",
                        f"{path}.prediction_status",
                        "invalid prediction status",
                    )
                )
            try:
                ref_values = _string_tuple(
                    refs,
                    label=f"{path}.supporting_evidence_refs",
                    nonempty=prediction != "pending_validation",
                )
                segment_values = _string_tuple(
                    segments,
                    label=f"{path}.supporting_segment_refs",
                    nonempty=True,
                )
            except SchemaValidationError as exc:
                local.append(LayerIssue("malformed_output", path, str(exc)))
                ref_values = ()
                segment_values = ()
            if prediction == "pending_validation" and not (
                isinstance(target_path, str)
                and target_path.startswith("/predicted_effects/")
            ):
                local.append(
                    LayerIssue(
                        "unsupported_pending_validation",
                        f"{path}.prediction_status",
                        "only predicted effects may be pending",
                    )
                )
            unknown_refs = set(ref_values) - set(registry)
            if unknown_refs:
                local.append(
                    LayerIssue(
                        "unknown_evidence_ref",
                        f"{path}.supporting_evidence_refs",
                        "unknown evidence ref",
                        ref_values,
                    )
                )
            if not set(segment_values).issubset(set(request.valid_segment_refs)):
                local.append(
                    LayerIssue(
                        "unknown_segment_ref",
                        f"{path}.supporting_segment_refs",
                        "unknown segment ref",
                    )
                )
            if source_type in {"annotation", "observed_robot_state", "observed_visual"}:
                if any(
                    source_type not in registry[ref]["source_types"]
                    for ref in ref_values
                    if ref in registry
                ):
                    local.append(
                        LayerIssue(
                            "source_type_confusion",
                            path,
                            "attribution cites another source class",
                            ref_values,
                        )
                    )
                cited_segments = {
                    registry[ref]["segment_id"] for ref in ref_values if ref in registry
                }
                if cited_segments and cited_segments != set(segment_values):
                    local.append(
                        LayerIssue(
                            "source_segment_mismatch",
                            path,
                            "evidence and declared segments differ",
                            ref_values,
                        )
                    )
                if item.get("temporal_scope") == "cross_segment":
                    local.append(
                        LayerIssue(
                            "source_type_confusion",
                            path,
                            "local source attribution cannot claim cross-segment scope",
                        )
                    )
            elif source_type == "cross_segment_inference":
                cited_segments = {
                    registry[ref]["segment_id"] for ref in ref_values if ref in registry
                }
                if (
                    item.get("temporal_scope") != "cross_segment"
                    or len(set(segment_values)) < 2
                    or cited_segments != set(segment_values)
                ):
                    local.append(
                        LayerIssue(
                            "cross_segment_evidence_mismatch",
                            path,
                            "cross-segment attribution must span exactly its segments",
                            ref_values,
                        )
                    )
            if local:
                issues.extend(local)
                continue
            seen_ids.add(str(attribution_id))
            seen_paths.add(str(target_path))
            admitted.append(item)
        missing = sorted(required_paths - seen_paths)
        for target_path in missing:
            issues.append(
                LayerIssue(
                    "missing_evidence_attribution",
                    target_path,
                    "planner-facing field lacks independent attribution",
                )
            )
        if issues:
            return self._result(request, output, (), issues)
        return self._result(request, output, admitted, ())

    @staticmethod
    def _result(
        request: AttributionPassRequest,
        output: LayerModelOutput,
        attributions: Sequence[Mapping[str, Any]],
        issues: Sequence[LayerIssue],
    ) -> AttributionPassResult:
        abstention = (
            None
            if attributions
            else _layer_abstention(
                layer="evidence_attribution",
                scope_id=request.trajectory_id,
                output=output,
                issues=_dedupe_issues(issues),
            )
        )
        return AttributionPassResult(
            attributions=tuple(deepcopy(list(attributions))),
            leakage_audit={
                "auditor": "combined_attribution_semantic_leakage_pass",
                "backend": output.backend,
                "prompt_template_hash": output.prompt_template_hash,
                "output_sha256": output.output_sha256,
                "episode_specific_leakage": output.parsed.get(
                    "episode_specific_leakage"
                )
                if isinstance(output.parsed, Mapping)
                else None,
                "leakage_reasons": deepcopy(output.parsed.get("leakage_reasons", []))
                if isinstance(output.parsed, Mapping)
                else [],
                "leakage_confidence": output.parsed.get("leakage_confidence", "")
                if isinstance(output.parsed, Mapping)
                else "",
            },
            abstention=abstention,
            output_sha256=output.output_sha256,
        )


@runtime_checkable
class HierarchicalReflectionBackend(Protocol):
    """Provider-neutral transport seam for the bounded eight-call pipeline."""

    @property
    def backend_name(self) -> str: ...

    @property
    def capabilities(self) -> Mapping[str, bool]: ...

    def reflect_action_chunks(
        self,
        request: ActionChunkBatchReflectionRequest,
    ) -> LayerModelOutput: ...

    def summarize_subtask(
        self,
        request: SubtaskSummaryRequest,
    ) -> LayerModelOutput: ...

    def draft_procedure(
        self,
        request: ProcedureDraftRequest,
    ) -> LayerModelOutput: ...

    def attribute_procedure(
        self,
        request: AttributionPassRequest,
    ) -> LayerModelOutput: ...


@dataclass(frozen=True, slots=True)
class HierarchicalTrajectoryReflectionResult:
    """Full private audit result; only an accepted candidate is planner-visible."""

    action_chunk_results: tuple[ActionChunkReflectionResult, ...]
    subtask_summary_results: tuple[SubtaskSummaryResult, ...]
    procedure_draft_result: ProcedureDraftResult | None
    attribution_result: AttributionPassResult | None
    final_admission: WholeTrajectoryReflectionResult | None
    episode_specific_facts: tuple[Mapping[str, Any], ...]
    abstentions: tuple[Mapping[str, Any], ...]
    model_call_audit: tuple[Mapping[str, Any], ...]
    model_outputs: tuple[Mapping[str, Any], ...] = ()

    @property
    def candidate_experience(self) -> Mapping[str, Any] | None:
        if self.final_admission is None or not self.final_admission.accepted:
            return None
        candidate = self.final_admission.candidate_experience
        return None if candidate is None else deepcopy(dict(candidate))

    @property
    def planner_visible_guidance(self) -> Mapping[str, Any] | None:
        candidate = self.candidate_experience
        if candidate is None:
            return None
        guidance = candidate.get("guidance")
        return deepcopy(dict(guidance)) if isinstance(guidance, Mapping) else None

    @property
    def accepted(self) -> bool:
        return self.candidate_experience is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_chunk_results": [
                item.to_dict() for item in self.action_chunk_results
            ],
            "subtask_summary_results": [
                item.to_dict() for item in self.subtask_summary_results
            ],
            "procedure_draft_result": (
                None
                if self.procedure_draft_result is None
                else self.procedure_draft_result.to_dict()
            ),
            "attribution_result": (
                None
                if self.attribution_result is None
                else self.attribution_result.to_dict()
            ),
            "final_admission": (
                None if self.final_admission is None else self.final_admission.to_dict()
            ),
            "episode_specific_facts": [
                deepcopy(dict(item)) for item in self.episode_specific_facts
            ],
            "candidate_experience": self.candidate_experience,
            "planner_visible_guidance": self.planner_visible_guidance,
            "abstentions": [deepcopy(dict(item)) for item in self.abstentions],
            "model_call_audit": [
                deepcopy(dict(item)) for item in self.model_call_audit
            ],
            "model_outputs": [deepcopy(dict(item)) for item in self.model_outputs],
        }


def _unique_facts(
    *groups: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for group in groups:
        for fact in group:
            fact_id = str(fact.get("fact_id", ""))
            encoded = _canonical(fact)
            if fact_id not in seen:
                seen[fact_id] = encoded
                result.append(deepcopy(dict(fact)))
            elif seen[fact_id] == encoded:
                continue
            # Conflicting duplicate IDs are never silently renamed because the
            # original evidence attribution would no longer be trustworthy.
    return tuple(result)


def _duplicate_fact_ids(
    *groups: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for group in groups:
        for fact in group:
            fact_id = str(fact.get("fact_id", ""))
            counts[fact_id] = counts.get(fact_id, 0) + 1
    return tuple(sorted(fact_id for fact_id, count in counts.items() if count > 1))


def _pipeline_fact_id_abstention(
    *,
    trajectory_id: str,
    backend_name: str,
    duplicate_ids: Sequence[str],
) -> dict[str, Any]:
    output = _synthetic_output(
        {"duplicate_fact_ids": list(duplicate_ids)},
        backend=backend_name,
        marker="hierarchical-duplicate-fact-id",
    )
    return _layer_abstention(
        layer="hierarchical_orchestration",
        scope_id=trajectory_id,
        output=output,
        issues=(
            LayerIssue(
                "duplicate_fact_id",
                "episode_specific_facts",
                "independently generated facts must have globally unique ids",
            ),
        ),
    )


def _call_audit(
    *,
    layer: str,
    scope_id: str,
    output: LayerModelOutput,
    image_count: int,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "scope_id": scope_id,
        "backend": output.backend,
        "prompt_template_hash": output.prompt_template_hash,
        "output_sha256": output.output_sha256,
        "parse_error": output.parse_error or "",
        "representation": (
            "local_reference_restored"
            if output.backend == "openai_compatible_hierarchical_multimodal"
            else "backend_layer_output"
        ),
        "reference_restoration_applied": (
            output.backend == "openai_compatible_hierarchical_multimodal"
        ),
        "image_count": image_count,
    }


def _model_output_record(
    *,
    layer: str,
    scope_id: str,
    output: LayerModelOutput,
) -> dict[str, Any] | None:
    """Preserve one provider response without any request or image payload."""

    parsed = output.parsed_dict()
    if (
        isinstance(parsed, Mapping)
        and "schema" not in parsed
        and parsed.get("error") in {"backend_exception", "malformed_backend_envelope"}
    ):
        # No provider output exists for a pre-I/O exception or malformed return
        # envelope.  The transport audit records that local failure instead.
        return None
    payload_text = output.raw_text
    lowered = payload_text.lower()
    unsafe = any(
        marker in lowered
        for marker in (
            "data:image/",
            "authorization: bearer ",
            '"api_key"',
            '"access_token"',
            '"secret"',
        )
    )
    record: dict[str, Any] = {
        "schema": "roboharn_evo/hierarchical_model_output/v1",
        "schema_version": 1,
        "layer": layer,
        "scope_id": scope_id,
        "backend": output.backend,
        "prompt_template_hash": output.prompt_template_hash,
        "output_sha256": output.output_sha256,
        "parse_error": output.parse_error or "",
        "request_included": False,
        "image_payload_included": False,
        "secret_included": False,
    }
    if unsafe:
        record.update(
            {
                "payload_kind": "suppressed_unsafe_output",
                "payload_suppressed": True,
            }
        )
    elif parsed is not None:
        record.update(
            {
                "payload_kind": "parsed_json",
                "payload_suppressed": False,
                "parsed_json": parsed,
            }
        )
    else:
        record.update(
            {
                "payload_kind": "raw_text",
                "payload_suppressed": False,
                "raw_text": output.raw_text,
            }
        )
    return record


def _synthetic_output(
    payload: Mapping[str, Any],
    *,
    backend: str,
    marker: str,
) -> LayerModelOutput:
    return LayerModelOutput.from_raw(
        payload,
        backend=backend,
        prompt_template_hash=hashlib.sha256(marker.encode("utf-8")).hexdigest(),
    )


class HierarchicalTrajectoryReflector:
    """Pure bounded-call orchestrator for Phase A2.1.

    With three subtasks this performs at most eight model calls: one batched
    Layer-1 and one Layer-2 call per subtask, one Layer-3 call, and one
    attribution call.  It does not retry a semantic response.
    """

    def __init__(
        self,
        backend: HierarchicalReflectionBackend,
        *,
        action_chunk_gate: ActionChunkBatchQualityGate | None = None,
        subtask_gate: SubtaskSummaryQualityGate | None = None,
        draft_gate: ProcedureDraftQualityGate | None = None,
        attribution_gate: AttributionQualityGate | None = None,
        final_quality_gate: ReflectionQualityGate | None = None,
        max_model_calls: int = 8,
        producer_name: str = "hierarchical_trajectory_reflector",
        producer_version: str = "1",
    ) -> None:
        if max_model_calls <= 0:
            raise ValueError("max_model_calls must be positive")
        self._backend = backend
        self._action_gate = action_chunk_gate or ActionChunkBatchQualityGate()
        self._subtask_gate = subtask_gate or SubtaskSummaryQualityGate()
        self._draft_gate = draft_gate or ProcedureDraftQualityGate()
        self._attribution_gate = attribution_gate or AttributionQualityGate()
        self._final_gate = final_quality_gate or ReflectionQualityGate()
        self._max_model_calls = max_model_calls
        self._producer_name = _nonempty(producer_name, label="producer_name")
        self._producer_version = _nonempty(producer_version, label="producer_version")
        self._last_preflight_audit: dict[str, Any] | None = None

    @property
    def last_preflight_audit(self) -> Mapping[str, Any] | None:
        """Return the most recent local call-budget decision."""

        return (
            None
            if self._last_preflight_audit is None
            else deepcopy(self._last_preflight_audit)
        )

    def reflect(
        self,
        request: WholeTrajectoryReflectionRequest,
        *,
        action_chunk_requests: Sequence[ActionChunkReflectionRequest],
        cross_segment_visual_evidence: Sequence[EvidenceInput | Mapping[str, Any]],
        segment_visual_evidence_by_segment: Mapping[
            str,
            Sequence[EvidenceInput | Mapping[str, Any]],
        ]
        | None = None,
    ) -> HierarchicalTrajectoryReflectionResult:
        if not isinstance(request, WholeTrajectoryReflectionRequest):
            raise TypeError("request must be WholeTrajectoryReflectionRequest")
        self._last_preflight_audit = None
        chunks = tuple(action_chunk_requests)
        if any(not isinstance(item, ActionChunkReflectionRequest) for item in chunks):
            raise TypeError("action_chunk_requests contain an invalid request")
        valid_segments = {
            str(segment["segment_id"]): segment for segment in request.segments
        }
        grouped: dict[str, list[ActionChunkReflectionRequest]] = {
            segment_id: [] for segment_id in valid_segments
        }
        expected_outcome = project_trajectory_outcome(request.trajectory_outcome)
        for item in chunks:
            if (
                item.trajectory_id != request.trajectory_id
                or item.segment_id not in grouped
            ):
                raise SchemaValidationError(
                    "action_chunk_requests: whole-request scope mismatch"
                )
            segment = valid_segments[item.segment_id]
            expected_annotation = {
                "evidence_ref": annotation_evidence_ref(item.segment_id),
                "text": str(segment["context"]["subtask_instruction"]),
            }
            if dict(item.subtask_annotation) != expected_annotation:
                raise SchemaValidationError(
                    "action_chunk_requests: annotation does not match the whole request"
                )
            if dict(item.trajectory_outcome) != expected_outcome:
                raise SchemaValidationError(
                    "action_chunk_requests: outcome projection does not match the whole request"
                )
            if _canonical(item.provenance) != _canonical(request.provenance):
                raise SchemaValidationError(
                    "action_chunk_requests: provenance does not match the whole request"
                )
            grouped[item.segment_id].append(item)
        cross_visual = _evidence_tuple(
            cross_segment_visual_evidence,
            label="cross_segment_visual_evidence",
        )
        segment_visual_by_segment: dict[str, list[EvidenceInput]] = {
            segment_id: [] for segment_id in valid_segments
        }
        batch_bindings_by_segment: dict[str, tuple[Mapping[str, Any], ...] | None] = {
            segment_id: None for segment_id in valid_segments
        }
        supplied_segment_visual = (
            {}
            if segment_visual_evidence_by_segment is None
            else segment_visual_evidence_by_segment
        )
        if not isinstance(supplied_segment_visual, Mapping):
            raise SchemaValidationError(
                "segment_visual_evidence_by_segment: must be an object"
            )
        unknown_segment_keys = set(supplied_segment_visual) - set(valid_segments)
        if unknown_segment_keys:
            raise SchemaValidationError(
                "segment_visual_evidence_by_segment: contains an unknown segment"
            )
        if segment_visual_evidence_by_segment is not None and set(
            supplied_segment_visual
        ) != set(valid_segments):
            raise SchemaValidationError(
                "segment_visual_evidence_by_segment: must cover every segment"
            )
        for segment_id, values in supplied_segment_visual.items():
            raw_bindings = getattr(values, "batch_bindings", None)
            if raw_bindings is not None:
                if isinstance(raw_bindings, (str, bytes)) or not isinstance(
                    raw_bindings, Sequence
                ):
                    raise SchemaValidationError(
                        "segment_visual_evidence_by_segment: invalid batch bindings"
                    )
                batch_bindings_by_segment[segment_id] = tuple(raw_bindings)
            evidence_values = _evidence_tuple(
                values,
                label=f"segment_visual_evidence_by_segment[{segment_id!r}]",
            )
            if any(
                evidence.payload.get("segment_id") != segment_id
                for evidence in evidence_values
            ):
                raise SchemaValidationError(
                    "segment_visual_evidence_by_segment: evidence has an "
                    "incorrect batch segment"
                )
            segment_visual_by_segment[segment_id].extend(evidence_values)

        layer1_batches_by_segment: dict[
            str, list[ActionChunkBatchReflectionRequest]
        ] = {segment_id: [] for segment_id in valid_segments}
        for segment_id, segment_chunks in grouped.items():
            if not segment_chunks:
                continue
            bindings = batch_bindings_by_segment[segment_id]
            if bindings is None:
                layer1_batches_by_segment[segment_id].append(
                    ActionChunkBatchReflectionRequest(
                        tuple(segment_chunks),
                        segment_visual_evidence=tuple(
                            segment_visual_by_segment[segment_id]
                        ),
                    )
                )
                continue
            chunks_by_id = {item.action_chunk_id: item for item in segment_chunks}
            evidence_by_ref = {
                item.evidence_ref: item
                for item in segment_visual_by_segment[segment_id]
            }
            seen_chunk_ids: set[str] = set()
            seen_request_ids: set[str] = set()
            for index, raw_binding in enumerate(bindings):
                if not isinstance(raw_binding, Mapping):
                    raise SchemaValidationError(
                        "segment_visual_evidence_by_segment: malformed batch binding"
                    )
                binding = dict(raw_binding)
                request_id = binding.get("request_id")
                bound_segment = binding.get("segment_id")
                chunk_ids = binding.get("action_chunk_ids")
                evidence_refs = binding.get("evidence_refs")
                if (
                    not isinstance(request_id, str)
                    or not request_id
                    or request_id in seen_request_ids
                    or bound_segment != segment_id
                    or not isinstance(chunk_ids, list)
                    or not chunk_ids
                    or not isinstance(evidence_refs, list)
                    or not evidence_refs
                ):
                    raise SchemaValidationError(
                        "segment_visual_evidence_by_segment: malformed batch binding"
                    )
                normalized_chunk_ids = [str(value) for value in chunk_ids]
                normalized_evidence_refs = [str(value) for value in evidence_refs]
                if (
                    len(set(normalized_chunk_ids)) != len(normalized_chunk_ids)
                    or len(set(normalized_evidence_refs))
                    != len(normalized_evidence_refs)
                    or seen_chunk_ids.intersection(normalized_chunk_ids)
                    or not set(normalized_chunk_ids).issubset(chunks_by_id)
                    or not set(normalized_evidence_refs).issubset(evidence_by_ref)
                ):
                    raise SchemaValidationError(
                        "segment_visual_evidence_by_segment: a Layer-1 batch "
                        "repeats a chunk or cites an unknown registry entry"
                    )
                seen_request_ids.add(request_id)
                seen_chunk_ids.update(normalized_chunk_ids)
                try:
                    batch = ActionChunkBatchReflectionRequest(
                        tuple(chunks_by_id[value] for value in normalized_chunk_ids),
                        segment_visual_evidence=tuple(
                            evidence_by_ref[value] for value in normalized_evidence_refs
                        ),
                    )
                except SchemaValidationError as exc:
                    raise SchemaValidationError(
                        "segment_visual_evidence_by_segment: batch registry does "
                        f"not completely cover its chunks at part {index}"
                    ) from exc
                layer1_batches_by_segment[segment_id].append(batch)
            if seen_chunk_ids != set(chunks_by_id):
                raise SchemaValidationError(
                    "segment_visual_evidence_by_segment: Layer-1 batch bindings "
                    "must cover every chunk exactly once"
                )

        expected_calls = (
            sum(len(values) for values in layer1_batches_by_segment.values())
            + len(request.segments)
            + 2
        )
        self._last_preflight_audit = {
            "schema": "roboharn_evo/hierarchical_call_budget_preflight/v1",
            "schema_version": 1,
            "status": (
                "passed" if expected_calls <= self._max_model_calls else "local_failed"
            ),
            "delivery_status": "not_sent",
            "required_model_calls": expected_calls,
            "maximum_model_calls": self._max_model_calls,
            "layer1_batch_calls": sum(
                len(values) for values in layer1_batches_by_segment.values()
            ),
            "layer2_subtask_calls": len(request.segments),
            "layer3_calls": 1,
            "layer4_calls": 1,
        }
        if expected_calls > self._max_model_calls:
            raise SchemaValidationError(
                "hierarchical reflection call budget cannot cover all split "
                "Layer-1 batches and downstream layers; no model request was sent"
            )
        capabilities = _mapping(
            self._backend.capabilities,
            label="backend.capabilities",
        )
        if set(capabilities) != _CAPABILITY_FIELDS or any(
            not isinstance(value, bool) for value in capabilities.values()
        ):
            raise SchemaValidationError(
                "backend.capabilities: invalid capability contract"
            )
        backend_name = _nonempty(
            self._backend.backend_name, label="backend.backend_name"
        )
        if not (
            capabilities["image_input_supported"]
            and capabilities["image_input_acknowledged"]
            and capabilities["structured_output_supported"]
            and not capabilities["text_only_fallback"]
            and not capabilities["scripted_backend"]
        ):
            output = _synthetic_output(
                {"capabilities": capabilities},
                backend=backend_name,
                marker="hierarchical-backend-not-semantic-multimodal",
            )
            abstention = _layer_abstention(
                layer="hierarchical_orchestration",
                scope_id=request.trajectory_id,
                output=output,
                issues=(
                    LayerIssue(
                        "backend_not_semantic_multimodal",
                        "backend.capabilities",
                        "hierarchical reflection requires acknowledged images and structured output",
                    ),
                ),
            )
            return HierarchicalTrajectoryReflectionResult(
                action_chunk_results=(),
                subtask_summary_results=(),
                procedure_draft_result=None,
                attribution_result=None,
                final_admission=None,
                episode_specific_facts=(),
                abstentions=(abstention,),
                model_call_audit=(),
            )

        call_count = 0
        call_audit: list[dict[str, Any]] = []
        model_outputs: list[dict[str, Any]] = []
        chunk_results: list[ActionChunkReflectionResult] = []
        subtask_results: list[SubtaskSummaryResult] = []
        abstentions: list[dict[str, Any]] = []
        layer1_by_segment: dict[str, list[ActionChunkReflectionResult]] = {}
        for segment in request.segments:
            segment_id = str(segment["segment_id"])
            segment_chunks = grouped[segment_id]
            if not segment_chunks:
                missing_output = _synthetic_output(
                    {"error": "missing_action_chunks"},
                    backend=backend_name,
                    marker="missing-action-chunks",
                )
                missing = _layer_abstention(
                    layer="action_chunk_batch",
                    scope_id=segment_id,
                    output=missing_output,
                    issues=(
                        LayerIssue(
                            "missing_action_chunks",
                            "action_chunk_requests",
                            "subtask has no action chunks",
                        ),
                    ),
                )
                abstentions.append(missing)
                layer1_by_segment[segment_id] = []
                continue
            segment_results: list[ActionChunkReflectionResult] = []
            batches = layer1_batches_by_segment[segment_id]
            for batch_index, batch in enumerate(batches):
                scope_id = (
                    segment_id
                    if len(batches) == 1
                    else f"{segment_id}#batch_{batch_index + 1}_of_{len(batches)}"
                )
                output = self._safe_backend_call(
                    "action_chunk_batch",
                    scope_id,
                    lambda batch=batch: self._backend.reflect_action_chunks(batch),
                    backend_name=backend_name,
                )
                output_record = _model_output_record(
                    layer="action_chunk_batch",
                    scope_id=scope_id,
                    output=output,
                )
                if output_record is not None:
                    model_outputs.append(output_record)
                call_count += 1
                call_audit.append(
                    _call_audit(
                        layer="action_chunk_batch",
                        scope_id=scope_id,
                        output=output,
                        image_count=len(batch.segment_visual_evidence),
                    )
                )
                evaluated = list(self._action_gate.evaluate(batch, output))
                segment_results.extend(evaluated)
                chunk_results.extend(evaluated)
                abstentions.extend(
                    deepcopy(dict(result.abstention))
                    for result in evaluated
                    if result.abstention is not None
                )
            layer1_by_segment[segment_id] = segment_results

        layer1_fact_groups = tuple(result.admitted_facts for result in chunk_results)
        all_layer1_facts = _unique_facts(*layer1_fact_groups)
        duplicate_ids = _duplicate_fact_ids(*layer1_fact_groups)
        if duplicate_ids:
            abstentions.append(
                _pipeline_fact_id_abstention(
                    trajectory_id=request.trajectory_id,
                    backend_name=backend_name,
                    duplicate_ids=duplicate_ids,
                )
            )
            return HierarchicalTrajectoryReflectionResult(
                action_chunk_results=tuple(chunk_results),
                subtask_summary_results=(),
                procedure_draft_result=None,
                attribution_result=None,
                final_admission=None,
                episode_specific_facts=all_layer1_facts,
                abstentions=tuple(abstentions),
                model_call_audit=tuple(call_audit),
                model_outputs=tuple(model_outputs),
            )
        if not all_layer1_facts:
            empty_output = _synthetic_output(
                {"error": "no_admitted_layer1_facts"},
                backend=backend_name,
                marker="no-admitted-layer1-facts",
            )
            abstentions.append(
                _layer_abstention(
                    layer="hierarchical_orchestration",
                    scope_id=request.trajectory_id,
                    output=empty_output,
                    issues=(
                        LayerIssue(
                            "no_admitted_layer1_facts",
                            "action_chunk_results",
                            "Layer-2 was not invoked because Layer-1 admitted no facts",
                        ),
                    ),
                )
            )
            return HierarchicalTrajectoryReflectionResult(
                action_chunk_results=tuple(chunk_results),
                subtask_summary_results=(),
                procedure_draft_result=None,
                attribution_result=None,
                final_admission=None,
                episode_specific_facts=(),
                abstentions=tuple(abstentions),
                model_call_audit=tuple(call_audit),
                model_outputs=tuple(model_outputs),
            )
        summary_by_segment: dict[str, dict[str, Any]] = {}
        all_layer2_facts: list[Mapping[str, Any]] = []
        for segment in request.segments:
            segment_id = str(segment["segment_id"])
            segment_chunks = grouped[segment_id]
            if not segment_chunks:
                continue
            segment_results = layer1_by_segment[segment_id]
            segment_facts = _unique_facts(
                *(result.admitted_facts for result in segment_results)
            )
            verification_visual: list[EvidenceInput] = []
            seen_visual: set[str] = set()
            for item in segment_chunks:
                for evidence in item.visual_evidence:
                    if evidence.evidence_ref not in seen_visual:
                        seen_visual.add(evidence.evidence_ref)
                        verification_visual.append(evidence)
            subtask_request = SubtaskSummaryRequest(
                trajectory_id=request.trajectory_id,
                segment_id=segment_id,
                segment_index=int(segment["segment_index"]),
                subtask_annotation={
                    "evidence_ref": annotation_evidence_ref(segment_id),
                    "text": str(segment["context"]["subtask_instruction"]),
                },
                ordered_action_chunks=tuple(
                    item.action_chunk for item in segment_chunks
                ),
                admitted_chunk_facts=segment_facts,
                verification_visual_evidence=tuple(verification_visual),
                trajectory_outcome=request.trajectory_outcome,
                provenance=request.provenance,
            )
            output = self._safe_backend_call(
                "subtask_summary",
                segment_id,
                lambda subtask_request=subtask_request: self._backend.summarize_subtask(
                    subtask_request
                ),
                backend_name=backend_name,
            )
            output_record = _model_output_record(
                layer="subtask_summary",
                scope_id=segment_id,
                output=output,
            )
            if output_record is not None:
                model_outputs.append(output_record)
            call_count += 1
            call_audit.append(
                _call_audit(
                    layer="subtask_summary",
                    scope_id=segment_id,
                    output=output,
                    image_count=len(verification_visual),
                )
            )
            result = self._subtask_gate.evaluate(subtask_request, output)
            subtask_results.append(result)
            all_layer2_facts.extend(result.admitted_episode_facts)
            if result.summary is not None:
                summary_by_segment[segment_id] = deepcopy(dict(result.summary))
            if result.abstention is not None:
                abstentions.append(deepcopy(dict(result.abstention)))

        duplicate_ids = _duplicate_fact_ids(all_layer1_facts, all_layer2_facts)
        episode_facts = _unique_facts(all_layer1_facts, all_layer2_facts)
        if duplicate_ids:
            abstentions.append(
                _pipeline_fact_id_abstention(
                    trajectory_id=request.trajectory_id,
                    backend_name=backend_name,
                    duplicate_ids=duplicate_ids,
                )
            )
            return HierarchicalTrajectoryReflectionResult(
                action_chunk_results=tuple(chunk_results),
                subtask_summary_results=tuple(subtask_results),
                procedure_draft_result=None,
                attribution_result=None,
                final_admission=None,
                episode_specific_facts=episode_facts,
                abstentions=tuple(abstentions),
                model_call_audit=tuple(call_audit),
                model_outputs=tuple(model_outputs),
            )
        if set(summary_by_segment) != set(valid_segments):
            return HierarchicalTrajectoryReflectionResult(
                action_chunk_results=tuple(chunk_results),
                subtask_summary_results=tuple(subtask_results),
                procedure_draft_result=None,
                attribution_result=None,
                final_admission=None,
                episode_specific_facts=episode_facts,
                abstentions=tuple(abstentions),
                model_call_audit=tuple(call_audit),
                model_outputs=tuple(model_outputs),
            )

        ordered_summaries = tuple(
            summary_by_segment[str(segment["segment_id"])]
            for segment in request.segments
        )
        draft_request = ProcedureDraftRequest(
            trajectory_id=request.trajectory_id,
            instruction=request.instruction,
            ordered_subtask_summaries=ordered_summaries,
            episode_specific_facts=episode_facts,
            cross_segment_visual_evidence=cross_visual,
            trajectory_outcome=request.trajectory_outcome,
            provenance=request.provenance,
        )
        draft_output = self._safe_backend_call(
            "procedure_draft",
            request.trajectory_id,
            lambda: self._backend.draft_procedure(draft_request),
            backend_name=backend_name,
        )
        output_record = _model_output_record(
            layer="procedure_draft",
            scope_id=request.trajectory_id,
            output=draft_output,
        )
        if output_record is not None:
            model_outputs.append(output_record)
        call_count += 1
        call_audit.append(
            _call_audit(
                layer="procedure_draft",
                scope_id=request.trajectory_id,
                output=draft_output,
                image_count=len(cross_visual),
            )
        )
        draft_result = self._draft_gate.evaluate(draft_request, draft_output)
        episode_facts = _unique_facts(
            episode_facts,
            draft_result.admitted_episode_facts,
        )
        if draft_result.abstention is not None:
            abstentions.append(deepcopy(dict(draft_result.abstention)))
        if draft_result.private_procedure_draft is None:
            return HierarchicalTrajectoryReflectionResult(
                action_chunk_results=tuple(chunk_results),
                subtask_summary_results=tuple(subtask_results),
                procedure_draft_result=draft_result,
                attribution_result=None,
                final_admission=None,
                episode_specific_facts=episode_facts,
                abstentions=tuple(abstentions),
                model_call_audit=tuple(call_audit),
                model_outputs=tuple(model_outputs),
            )

        attribution_request = AttributionPassRequest.from_whole_request(
            request,
            procedure_draft=draft_result.private_procedure_draft,
            episode_specific_facts=episode_facts,
        )
        attribution_output = self._safe_backend_call(
            "evidence_attribution",
            request.trajectory_id,
            lambda: self._backend.attribute_procedure(attribution_request),
            backend_name=backend_name,
        )
        output_record = _model_output_record(
            layer="evidence_attribution",
            scope_id=request.trajectory_id,
            output=attribution_output,
        )
        if output_record is not None:
            model_outputs.append(output_record)
        call_count += 1
        call_audit.append(
            _call_audit(
                layer="evidence_attribution",
                scope_id=request.trajectory_id,
                output=attribution_output,
                image_count=0,
            )
        )
        if call_count > self._max_model_calls:
            raise RuntimeError("hierarchical model-call budget invariant violated")
        attribution_result = self._attribution_gate.evaluate(
            attribution_request,
            attribution_output,
        )
        if attribution_result.abstention is not None:
            abstentions.append(deepcopy(dict(attribution_result.abstention)))
            return HierarchicalTrajectoryReflectionResult(
                action_chunk_results=tuple(chunk_results),
                subtask_summary_results=tuple(subtask_results),
                procedure_draft_result=draft_result,
                attribution_result=attribution_result,
                final_admission=None,
                episode_specific_facts=episode_facts,
                abstentions=tuple(abstentions),
                model_call_audit=tuple(call_audit),
                model_outputs=tuple(model_outputs),
            )

        final_payload = {
            "schema": "roboharn_evo/whole_trajectory_reflection_output/v1",
            "schema_version": 1,
            "status": "candidate",
            "abstention_reason": "",
            "summary": draft_result.summary,
            "confidence": draft_result.confidence,
            "episode_specific_facts": [deepcopy(dict(item)) for item in episode_facts],
            "transferable_guidance": deepcopy(
                dict(draft_result.private_procedure_draft)
            ),
            "attributions": [
                deepcopy(dict(item)) for item in attribution_result.attributions
            ],
            "supporting_evidence_refs": list(
                dict.fromkeys(
                    (
                        *draft_result.supporting_evidence_refs,
                        *(
                            ref
                            for item in attribution_result.attributions
                            for ref in item["supporting_evidence_refs"]
                        ),
                    )
                )
            ),
            "supporting_segment_refs": list(
                dict.fromkeys(
                    (
                        *draft_result.supporting_segment_refs,
                        *(
                            ref
                            for item in attribution_result.attributions
                            for ref in item["supporting_segment_refs"]
                        ),
                    )
                )
            ),
        }
        combined_prompt_hash = hashlib.sha256(
            (
                draft_output.prompt_template_hash
                + attribution_output.prompt_template_hash
                + "hierarchical-final-admission-v1"
            ).encode("utf-8")
        ).hexdigest()
        final_output = UntrustedReflectionOutput.from_raw(
            final_payload,
            backend=backend_name,
            prompt_template_hash=combined_prompt_hash,
            capabilities=capabilities,
        )
        final_admission = self._final_gate.evaluate(
            request,
            final_output,
            producer_name=self._producer_name,
            producer_version=self._producer_version,
        )
        if final_admission.abstention is not None:
            abstentions.append(deepcopy(dict(final_admission.abstention)))
        return HierarchicalTrajectoryReflectionResult(
            action_chunk_results=tuple(chunk_results),
            subtask_summary_results=tuple(subtask_results),
            procedure_draft_result=draft_result,
            attribution_result=attribution_result,
            final_admission=final_admission,
            episode_specific_facts=episode_facts,
            abstentions=tuple(abstentions),
            model_call_audit=tuple(call_audit),
            model_outputs=tuple(model_outputs),
        )

    @staticmethod
    def _safe_backend_call(
        layer: str,
        scope_id: str,
        operation: Any,
        *,
        backend_name: str,
    ) -> LayerModelOutput:
        try:
            output = operation()
        except Exception as exc:
            return _synthetic_output(
                {
                    "error": "backend_exception",
                    "exception_type": type(exc).__name__,
                    "layer": layer,
                    "scope_id": scope_id,
                },
                backend=backend_name,
                marker=f"{layer}-backend-exception",
            )
        if isinstance(output, LayerModelOutput):
            return output
        return _synthetic_output(
            {
                "error": "malformed_backend_envelope",
                "layer": layer,
                "scope_id": scope_id,
            },
            backend=backend_name,
            marker=f"{layer}-malformed-backend-envelope",
        )


__all__ = [
    "ActionChunkBatchQualityGate",
    "ActionChunkBatchReflectionRequest",
    "ActionChunkQualityGate",
    "ActionChunkReflectionRequest",
    "ActionChunkReflectionResult",
    "AttributionPassRequest",
    "AttributionPassResult",
    "AttributionQualityGate",
    "EvidenceInput",
    "HierarchicalReflectionBackend",
    "HierarchicalTrajectoryReflectionResult",
    "HierarchicalTrajectoryReflector",
    "LayerIssue",
    "LayerModelOutput",
    "ProcedureDraftQualityGate",
    "ProcedureDraftRequest",
    "ProcedureDraftResult",
    "SubtaskSummaryQualityGate",
    "SubtaskSummaryRequest",
    "SubtaskSummaryResult",
    "action_chunk_batch_output_json_schema",
    "action_chunk_batch_request_json_schema",
    "action_chunk_output_json_schema",
    "action_chunk_request_json_schema",
    "attribution_output_json_schema",
    "attribution_request_json_schema",
    "hierarchical_source_semantics",
    "procedure_draft_output_json_schema",
    "procedure_draft_request_json_schema",
    "project_trajectory_outcome",
    "required_attribution_paths",
    "semantic_leakage_audit_contract",
    "subtask_summary_output_json_schema",
    "subtask_summary_request_json_schema",
]

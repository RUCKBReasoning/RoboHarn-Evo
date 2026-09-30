"""Pure contracts for multimodal, whole-trajectory procedure reflection.

This module deliberately owns no transport and performs no file or network
I/O.  A caller supplies a validated request and an implementation of
``MultimodalReflectionBackend``.  Backend output remains untrusted until it has
passed :class:`roboharn_evo.agent.reflector.quality.ReflectionQualityGate`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .schemas import ProvenanceV1, SchemaValidationError, SubtaskSegmentV1


_REQUEST_SCHEMA = "roboharn_evo/whole_trajectory_reflection_request/v1"
_REQUEST_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def annotation_evidence_ref(segment_id: str) -> str:
    """Return the request-local ref for a segment's benchmark annotation."""

    value = _nonempty_string(segment_id, label="segment_id")
    return f"{value}#annotation"


def reflection_output_json_schema() -> dict[str, Any]:
    """Return the provider-neutral structured-output schema expected by A2b.

    The quality gate remains authoritative.  This schema is an early transport
    constraint and is returned as a fresh value so callers cannot mutate a
    process-global contract.
    """

    string_array = {
        "type": "array",
        "items": {"type": "string"},
    }
    source_type = {
        "type": "string",
        "enum": [
            "observed_visual",
            "observed_robot_state",
            "annotation",
            "cross_segment_inference",
        ],
    }
    attribution_target_pattern = (
        r"^/(?:condition|guidance/ordered_steps/[0-9]+|"
        r"guidance/feedback_policy/(?:state_variables|attempt_tracking|"
        r"observation_rules|failure_branch|success_branch|termination_conditions)|"
        r"guidance/avoid|predicted_effects/[0-9]+)$"
    )
    fact_properties = {
        "fact_id": {"type": "string"},
        "claim": {"type": "string"},
        "source_type": source_type,
        "supporting_evidence_refs": string_array,
        "supporting_segment_refs": string_array,
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "uncertainty": {"type": "string"},
        "temporal_scope": {
            "type": "string",
            "enum": ["single_frame", "before_after", "cross_segment"],
            "description": (
                "Declare whether the fact is supported by one observation, a "
                "before/after comparison, or evidence spanning segments."
            ),
        },
        "transferable_allowed": {"type": "boolean"},
        # A canonical string is sufficient for structural equality checks and
        # avoids an unconstrained nested object in strict Structured Outputs.
        "structured_value": {
            "type": "string",
            "description": (
                "For observed_robot_state only, encode one cited configured "
                "state exactly as <channel_id>=<state>; use an empty string for "
                "all other source types."
            ),
        },
    }
    fact = {
        "type": "object",
        "additionalProperties": False,
        "required": list(fact_properties),
        "properties": fact_properties,
    }
    attribution_properties = {
        "attribution_id": {"type": "string"},
        "target_path": {
            "type": "string",
            "pattern": attribution_target_pattern,
            "description": (
                "Path relative to transferable_guidance; never prefix it with "
                "/transferable_guidance. Attribute /condition, each "
                "/guidance/ordered_steps/<index>, each feedback_policy field, "
                "/guidance/avoid, and every /predicted_effects/<index>."
            ),
        },
        "claim": {"type": "string"},
        "source_type": source_type,
        "supporting_evidence_refs": string_array,
        "supporting_segment_refs": string_array,
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "uncertainty": {"type": "string"},
        "temporal_scope": {
            "type": "string",
            "enum": ["single_frame", "before_after", "cross_segment"],
        },
        "prediction_status": {
            "type": "string",
            "enum": ["evidence_supported", "pending_validation"],
        },
    }
    attribution = {
        "type": "object",
        "additionalProperties": False,
        "required": list(attribution_properties),
        "properties": attribution_properties,
    }
    attempt_tracking_properties = {
        "state_variable": {"type": "string"},
        "record_after_attempt": {"type": "boolean"},
        "exclude_previously_recorded": {"type": "boolean"},
    }
    observation_properties = {
        "after_action_pattern": {"type": "string"},
        "observe_signal": {"type": "string"},
    }
    feedback_properties = {
        "state_variables": string_array,
        "attempt_tracking": {
            "type": "object",
            "additionalProperties": False,
            "required": list(attempt_tracking_properties),
            "properties": attempt_tracking_properties,
        },
        "observation_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(observation_properties),
                "properties": observation_properties,
            },
        },
        "failure_branch": string_array,
        "success_branch": string_array,
        "termination_conditions": string_array,
    }
    feedback_policy = {
        "type": "object",
        "additionalProperties": False,
        "required": list(feedback_properties),
        "properties": feedback_properties,
    }
    condition_properties = {
        "task_family": {"type": "string"},
        "subtask_type": {"type": "string"},
        "preconditions": string_array,
    }
    step_properties = {
        "action_pattern": {"type": "string"},
        "instruction": {"type": "string"},
    }
    effect_properties = {
        "effect": {"type": "string"},
        "validation_status": {
            "type": "string",
            "enum": ["evidence_supported", "pending_validation"],
        },
    }
    transferable = {
        "type": "object",
        "additionalProperties": False,
        "required": ["condition", "guidance", "predicted_effects"],
        "properties": {
            "condition": {
                "type": "object",
                "additionalProperties": False,
                "required": list(condition_properties),
                "properties": condition_properties,
            },
            "guidance": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ordered_steps", "feedback_policy", "avoid"],
                "properties": {
                    "ordered_steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(step_properties),
                            "properties": step_properties,
                        },
                    },
                    "feedback_policy": feedback_policy,
                    "avoid": string_array,
                },
            },
            "predicted_effects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(effect_properties),
                    "properties": effect_properties,
                },
            },
        },
    }
    properties = {
        "schema": {
            "type": "string",
            "enum": ["roboharn_evo/whole_trajectory_reflection_output/v1"],
        },
        "schema_version": {"type": "integer", "enum": [1]},
        "status": {"type": "string", "enum": ["candidate", "abstained"]},
        "abstention_reason": {"type": "string"},
        "summary": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "episode_specific_facts": {"type": "array", "items": fact},
        "transferable_guidance": {
            "anyOf": [transferable, {"type": "null"}],
        },
        "attributions": {"type": "array", "items": attribution},
        "supporting_evidence_refs": string_array,
        "supporting_segment_refs": string_array,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RoboHarn-Evo Whole-Trajectory Reflection Output V1",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _json_copy(value: Any, *, label: str) -> Any:
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


def _mapping_copy(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_copy(dict(value), label=label)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return _json_copy(dict(payload), label=label)
    raise SchemaValidationError(f"{label}: must be an object or expose to_dict()")


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{label}: must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class WholeTrajectoryReflectionRequest:
    """JSON-safe input for one complete trajectory.

    ``visual_evidence`` is the serialized A2a bundle.  Keeping that bundle as
    one opaque-but-JSON value avoids coupling this pure boundary to an RMBench
    adapter while preserving every stable evidence identifier and review-image
    locator for a transport implementation.
    """

    trajectory_id: str
    instruction: str
    trajectory_outcome: Mapping[str, Any]
    segments: tuple[Mapping[str, Any], ...]
    visual_evidence: Mapping[str, Any]
    provenance: Mapping[str, Any]
    uncertainties: tuple[str, ...] = ()
    schema: str = _REQUEST_SCHEMA
    schema_version: int = _REQUEST_VERSION

    def __post_init__(self) -> None:
        _nonempty_string(self.trajectory_id, label="request.trajectory_id")
        _nonempty_string(self.instruction, label="request.instruction")
        if self.schema not in {
            _REQUEST_SCHEMA, "tcm/whole_trajectory_reflection_request/v1"
        } or self.schema_version != _REQUEST_VERSION:
            raise SchemaValidationError("request: unsupported schema or version")

        outcome = _mapping_copy(
            self.trajectory_outcome,
            label="request.trajectory_outcome",
        )
        visual = _mapping_copy(self.visual_evidence, label="request.visual_evidence")
        provenance = ProvenanceV1.from_dict(
            _mapping_copy(self.provenance, label="request.provenance")
        ).to_dict()

        if isinstance(self.segments, (str, bytes)) or not isinstance(
            self.segments, Sequence
        ):
            raise SchemaValidationError("request.segments: must be an array")
        normalized_segments: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_indices: set[int] = set()
        for number, raw in enumerate(self.segments):
            segment = SubtaskSegmentV1.from_dict(
                _mapping_copy(raw, label=f"request.segments[{number}]")
            ).to_dict()
            if segment["trajectory_id"] != self.trajectory_id:
                raise SchemaValidationError(
                    f"request.segments[{number}].trajectory_id: expected "
                    f"{self.trajectory_id!r}"
                )
            segment_id = segment["segment_id"]
            segment_index = segment["segment_index"]
            if segment_id in seen_ids:
                raise SchemaValidationError(
                    f"request.segments[{number}].segment_id: duplicate {segment_id!r}"
                )
            if segment_index in seen_indices:
                raise SchemaValidationError(
                    f"request.segments[{number}].segment_index: duplicate "
                    f"{segment_index!r}"
                )
            seen_ids.add(segment_id)
            seen_indices.add(segment_index)
            normalized_segments.append(segment)
        normalized_segments.sort(key=lambda value: value["segment_index"])

        if isinstance(self.uncertainties, (str, bytes)) or not isinstance(
            self.uncertainties, Sequence
        ):
            raise SchemaValidationError("request.uncertainties: must be an array")
        uncertainty_values: list[str] = []
        seen_uncertainties: set[str] = set()
        for number, value in enumerate(self.uncertainties):
            text = _nonempty_string(
                value,
                label=f"request.uncertainties[{number}]",
            )
            if text in seen_uncertainties:
                raise SchemaValidationError(
                    f"request.uncertainties[{number}]: duplicate value {text!r}"
                )
            seen_uncertainties.add(text)
            uncertainty_values.append(text)

        object.__setattr__(self, "trajectory_outcome", outcome)
        object.__setattr__(self, "segments", tuple(normalized_segments))
        object.__setattr__(self, "visual_evidence", visual)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "uncertainties", tuple(uncertainty_values))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WholeTrajectoryReflectionRequest":
        data = _mapping_copy(payload, label="request")
        allowed = {
            "schema",
            "schema_version",
            "trajectory_id",
            "instruction",
            "trajectory_outcome",
            "segments",
            "visual_evidence",
            "provenance",
            "uncertainties",
            "annotation_evidence_refs",
        }
        unknown = sorted(set(data) - allowed)
        required = allowed - {"uncertainties", "annotation_evidence_refs"}
        missing = sorted(required - set(data))
        if missing:
            raise SchemaValidationError(
                "request: missing required field(s): " + ", ".join(missing)
            )
        if unknown:
            raise SchemaValidationError(
                "request: unknown field(s): " + ", ".join(unknown)
            )
        request = cls(
            schema=data["schema"],
            schema_version=data["schema_version"],
            trajectory_id=data["trajectory_id"],
            instruction=data["instruction"],
            trajectory_outcome=data["trajectory_outcome"],
            segments=tuple(data["segments"]),
            visual_evidence=data["visual_evidence"],
            provenance=data["provenance"],
            uncertainties=tuple(data.get("uncertainties", ())),
        )
        supplied_annotation_refs = data.get("annotation_evidence_refs")
        if supplied_annotation_refs is not None and supplied_annotation_refs != (
            request.annotation_evidence_refs
        ):
            raise SchemaValidationError(
                "request.annotation_evidence_refs: must match derived segment refs"
            )
        return request

    @property
    def annotation_evidence_refs(self) -> list[dict[str, str]]:
        return [
            {
                "segment_id": str(segment["segment_id"]),
                "evidence_ref": annotation_evidence_ref(str(segment["segment_id"])),
            }
            for segment in self.segments
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "instruction": self.instruction,
            "trajectory_outcome": deepcopy(dict(self.trajectory_outcome)),
            "segments": [deepcopy(dict(value)) for value in self.segments],
            "visual_evidence": deepcopy(dict(self.visual_evidence)),
            "provenance": deepcopy(dict(self.provenance)),
            "uncertainties": list(self.uncertainties),
            "annotation_evidence_refs": self.annotation_evidence_refs,
        }


@dataclass(frozen=True, slots=True)
class UntrustedReflectionOutput:
    """A hash-bound backend response which has not passed semantic gates."""

    raw_text: str = field(repr=False)
    backend: str
    prompt_template_hash: str
    capabilities: Mapping[str, bool]
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
        capabilities: Mapping[str, bool],
    ) -> "UntrustedReflectionOutput":
        backend_name = _nonempty_string(backend, label="backend")
        if not isinstance(prompt_template_hash, str) or not _SHA256_RE.fullmatch(
            prompt_template_hash
        ):
            raise SchemaValidationError(
                "prompt_template_hash: must be a lowercase SHA-256 digest"
            )
        required_capabilities = {
            "image_input_supported",
            "image_input_acknowledged",
            "structured_output_supported",
            "text_only_fallback",
            "scripted_backend",
        }
        capability_data = _mapping_copy(capabilities, label="capabilities")
        if set(capability_data) != required_capabilities:
            raise SchemaValidationError(
                "capabilities: expected exactly "
                + ", ".join(sorted(required_capabilities))
            )
        if any(not isinstance(value, bool) for value in capability_data.values()):
            raise SchemaValidationError("capabilities: every value must be boolean")

        parsed: dict[str, Any] | None
        parse_error: str | None = None
        if isinstance(raw, Mapping):
            parsed = _mapping_copy(raw, label="backend output")
            raw_text = json.dumps(
                parsed,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            if isinstance(raw, bytes):
                try:
                    raw_text = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raw_text = raw.decode("utf-8", errors="replace")
                    parsed = None
                    parse_error = f"invalid_utf8:{exc.start}"
                else:
                    parsed = None
            elif isinstance(raw, str):
                raw_text = raw
                parsed = None
            else:
                raise SchemaValidationError(
                    "backend output: must be JSON text, bytes, or an object"
                )
            if parse_error is None:
                try:
                    def reject_duplicate_keys(
                        pairs: list[tuple[str, Any]],
                    ) -> dict[str, Any]:
                        value: dict[str, Any] = {}
                        for key, item in pairs:
                            if key in value:
                                raise ValueError(f"duplicate JSON key {key!r}")
                            value[key] = item
                        return value

                    def reject_constant(value: str) -> None:
                        raise ValueError(f"non-finite JSON value {value}")

                    decoded = json.loads(
                        raw_text,
                        object_pairs_hook=reject_duplicate_keys,
                        parse_constant=reject_constant,
                    )
                    if not isinstance(decoded, Mapping):
                        raise TypeError("top-level JSON must be an object")
                    parsed = _mapping_copy(decoded, label="backend output")
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    SchemaValidationError,
                ) as exc:
                    parsed = None
                    parse_error = f"invalid_json:{exc}"

        output_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        return cls(
            raw_text=raw_text,
            backend=backend_name,
            prompt_template_hash=prompt_template_hash,
            capabilities=capability_data,
            output_sha256=output_sha256,
            parsed=parsed,
            parse_error=parse_error,
        )

    def parsed_dict(self) -> dict[str, Any] | None:
        return None if self.parsed is None else deepcopy(dict(self.parsed))


@runtime_checkable
class MultimodalReflectionBackend(Protocol):
    """Model/provider-neutral backend seam used by the whole-trajectory layer."""

    @property
    def backend_name(self) -> str: ...

    def reflect(
        self,
        request: WholeTrajectoryReflectionRequest,
    ) -> UntrustedReflectionOutput: ...


@dataclass(frozen=True, slots=True)
class WholeTrajectoryReflectionResult:
    """Either one candidate plus private audit sidecars, or one abstention."""

    candidate_experience: Mapping[str, Any] | None
    episode_specific_facts: tuple[Mapping[str, Any], ...]
    attributions: tuple[Mapping[str, Any], ...]
    abstention: Mapping[str, Any] | None
    output_sha256: str

    @property
    def accepted(self) -> bool:
        return self.candidate_experience is not None and self.abstention is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_experience": None
            if self.candidate_experience is None
            else deepcopy(dict(self.candidate_experience)),
            "episode_specific_facts": [
                deepcopy(dict(value)) for value in self.episode_specific_facts
            ],
            "attributions": [deepcopy(dict(value)) for value in self.attributions],
            "abstention": None
            if self.abstention is None
            else deepcopy(dict(self.abstention)),
            "output_sha256": self.output_sha256,
        }


class WholeTrajectoryReflector:
    """Invoke one backend and admit output only through the strict quality gate."""

    def __init__(
        self,
        backend: MultimodalReflectionBackend,
        *,
        quality_gate: Any | None = None,
        producer_name: str = "whole_trajectory_reflector",
        producer_version: str = "1",
    ) -> None:
        self._backend = backend
        self._producer_name = _nonempty_string(
            producer_name, label="producer_name"
        )
        self._producer_version = _nonempty_string(
            producer_version, label="producer_version"
        )
        if quality_gate is None:
            from .quality import ReflectionQualityGate

            quality_gate = ReflectionQualityGate()
        self._quality_gate = quality_gate

    def reflect(
        self,
        request: WholeTrajectoryReflectionRequest,
    ) -> WholeTrajectoryReflectionResult:
        if not isinstance(request, WholeTrajectoryReflectionRequest):
            raise TypeError("request must be WholeTrajectoryReflectionRequest")

        # Preserve the legacy backend for its old per-segment API while making
        # it impossible to label that scripted path as a semantic A2 result.
        from .candidate import ScriptedProcedureBackend

        if isinstance(self._backend, ScriptedProcedureBackend):
            return self._quality_gate.backend_abstention(
                request,
                backend=self._backend.__class__.__name__,
                reason="scripted_backend_not_semantic",
            )
        try:
            output = self._backend.reflect(request)
        except Exception as exc:
            return self._quality_gate.backend_abstention(
                request,
                backend=getattr(
                    self._backend,
                    "backend_name",
                    self._backend.__class__.__name__,
                ),
                reason="backend_exception",
                detail=type(exc).__name__,
            )
        if not isinstance(output, UntrustedReflectionOutput):
            return self._quality_gate.backend_abstention(
                request,
                backend=getattr(
                    self._backend,
                    "backend_name",
                    self._backend.__class__.__name__,
                ),
                reason="malformed_backend_envelope",
            )
        return self._quality_gate.evaluate(
            request,
            output,
            producer_name=self._producer_name,
            producer_version=self._producer_version,
        )


__all__ = [
    "MultimodalReflectionBackend",
    "UntrustedReflectionOutput",
    "WholeTrajectoryReflectionRequest",
    "WholeTrajectoryReflectionResult",
    "WholeTrajectoryReflector",
    "annotation_evidence_ref",
    "reflection_output_json_schema",
]

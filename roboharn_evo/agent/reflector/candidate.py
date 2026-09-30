"""Candidate-only procedure reflection over caller-supplied offline evidence.

This module is deliberately pure: it does not discover trajectories, read or
write files, call services, execute actions, persist experience, or promote a
candidate.  The caller supplies a normalized ``SubtaskSegmentV1``, an
``EvidenceIndex``, and provenance through ``ReflectorInput.experience_context``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import ReflectorInput, ReflectorOutput
from .evidence import EvidenceIndex
from .schemas import (
    ProcedureExperienceV1,
    ProvenanceV1,
    SchemaValidationError,
    SubtaskSegmentV1,
    stable_experience_id,
    validate_reflector_candidate,
)


_TRAJECTORY_SCHEMA_VERSION = 1
_SEGMENT_SCHEMA = "roboharn_evo/subtask_segment/v1"
_SEGMENT_SCHEMA_VERSION = 1
_PROCEDURE_SCHEMA = "roboharn_evo/experience/procedure/v1"
_ABSTENTION_SCOPE = "procedure_candidate"
_CONFIDENCE_VALUES = frozenset({"low", "medium", "high"})
# The Phase A scripted expert importer does not emit a grounded relation.  The
# default backend treats the instruction as opaque and therefore has no
# relation semantics it can safely transfer.  Relation-aware backends can be
# supplied explicitly through ``CandidateProcedureReflector(backend=...)``.
_SCRIPTED_SUPPORTED_RELATIONS: frozenset[str] = frozenset()


class ProcedureBackendAbstention(ValueError):
    """A conservative backend declined to derive a supported procedure."""

    def __init__(self, reason: str) -> None:
        normalized = str(reason or "backend_abstained").strip() or "backend_abstained"
        self.reason = normalized
        super().__init__(normalized)


@dataclass(frozen=True, slots=True)
class ReflectorAbstention:
    """Additive, module-local abstention representation.

    ``ReflectorOutput`` has no abstention field yet.  Until that public contract
    is extended, ``to_summary`` provides a small machine-readable value while
    the candidate and evidence fields remain empty.
    """

    reason: str
    scope: str = _ABSTENTION_SCOPE
    status: str = "abstained"

    def to_dict(self) -> dict[str, str]:
        return {
            "reason": self.reason,
            "scope": self.scope,
            "status": self.status,
        }

    def to_summary(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@runtime_checkable
class ProcedureCandidateBackend(Protocol):
    """Pure backend boundary for deriving one procedure proposal.

    Every argument is an isolated deep copy.  Implementations return only an
    untrusted proposal mapping; ``CandidateProcedureReflector`` owns evidence
    resolution, lifecycle enforcement, provenance construction, and schema
    validation.
    """

    def propose(
        self,
        *,
        segment: Mapping[str, Any],
        evidence: Mapping[str, Mapping[str, Any]],
        trajectory: Mapping[str, Any],
        scene_memory: Mapping[str, Any],
        experience_context: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


# A concise alias is useful to callers without adding anything to the existing
# public reflector package facade.
ProcedureBackend = ProcedureCandidateBackend


def _as_mapping_copy(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return deepcopy(dict(payload))
    raise TypeError(f"{label} must be a mapping or expose to_dict()")


def _string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of evidence refs")
    refs: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"{label} must contain non-empty string refs")
        refs.append(item)
    return tuple(refs)


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _entry_to_mapping(entry: Any) -> dict[str, Any]:
    return _as_mapping_copy(entry, label="resolved evidence entry")


def _frame_index(entry: Mapping[str, Any]) -> int | None:
    locator = entry.get("locator")
    if not isinstance(locator, Mapping):
        return None
    kind = str(locator.get("kind", "")).strip()
    if "frame" not in kind:
        return None
    value = locator.get("frame_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class ScriptedProcedureBackend:
    """Conservative CPU-only backend for a successful scripted expert segment.

    The backend treats ``subtask_instruction`` as opaque data.  It never parses
    the text to infer entities, relations, poses, state deltas, or recovery.  A
    generic procedure merely says to follow that exact scripted subtask.  An
    optional explicit ``context.procedure_guidance`` may supply structured
    steps, avoid items, and predicted effects after strict shape checks.
    """

    def propose(
        self,
        *,
        segment: Mapping[str, Any],
        evidence: Mapping[str, Mapping[str, Any]],
        trajectory: Mapping[str, Any],
        scene_memory: Mapping[str, Any],
        experience_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del scene_memory, experience_context

        trajectory_outcome = trajectory.get("outcome")
        if not isinstance(trajectory_outcome, Mapping):
            raise ProcedureBackendAbstention("missing_collection_contract_success")
        if (
            trajectory_outcome.get("status") != "success"
            or trajectory_outcome.get("evidence_source")
            != "benchmark_collection_contract"
            or trajectory_outcome.get("explicit_in_artifact") is not False
        ):
            raise ProcedureBackendAbstention("missing_collection_contract_success")

        segment_outcome = segment.get("outcome")
        if not isinstance(segment_outcome, Mapping):
            raise ProcedureBackendAbstention("malformed_scripted_segment")
        # RMBench's stored ``joint_action`` is not proven to be a commanded
        # action.  Its normalized scripted segment therefore legitimately has
        # no action refs and status=unknown even though the *trajectory* is
        # covered by the collection success contract.  Do not rewrite that
        # local uncertainty into a segment-level success claim.
        if segment_outcome.get("status") not in {"success", "unknown"}:
            raise ProcedureBackendAbstention("segment_is_not_procedure_source")

        context = segment.get("context")
        transition = segment.get("transition")
        derivation = segment.get("derivation")
        if not isinstance(context, Mapping) or not isinstance(transition, Mapping):
            raise ProcedureBackendAbstention("malformed_scripted_segment")
        if not isinstance(derivation, Mapping):
            raise ProcedureBackendAbstention("malformed_scripted_segment")
        derivation_producer = derivation.get("producer")
        if (
            not isinstance(derivation_producer, Mapping)
            or derivation_producer.get("kind") != "benchmark"
            or derivation_producer.get("name") != "benchmark_scripted_annotation"
        ):
            raise ProcedureBackendAbstention("unsupported_segment_derivation")

        instruction = context.get("subtask_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ProcedureBackendAbstention("missing_subtask_instruction")

        before_refs = _string_sequence(
            transition.get("before_event_refs", ()),
            label="transition.before_event_refs",
        )
        after_refs = _string_sequence(
            transition.get("after_event_refs", ()),
            label="transition.after_event_refs",
        )
        before_frames = [
            index
            for ref in before_refs
            if ref in evidence
            for index in [_frame_index(evidence[ref])]
            if index is not None
        ]
        after_frames = [
            index
            for ref in after_refs
            if ref in evidence
            for index in [_frame_index(evidence[ref])]
            if index is not None
        ]
        if not before_frames or not after_frames:
            raise ProcedureBackendAbstention("insufficient_frame_evidence")
        if max(after_frames) <= min(before_frames):
            raise ProcedureBackendAbstention("segment_span_too_short")

        relation = context.get("relation")
        if relation is not None:
            if not isinstance(relation, str) or not relation.strip():
                raise ProcedureBackendAbstention("malformed_scripted_condition")
            if relation not in _SCRIPTED_SUPPORTED_RELATIONS:
                raise ProcedureBackendAbstention("unsupported_relation")

        condition: dict[str, Any] = {"subtask_type": "scripted_subtask"}
        for key in ("task_family", "manipulation_phase"):
            value = context.get(key)
            if value is None or value == "":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ProcedureBackendAbstention("malformed_scripted_condition")
            condition[key] = value

        explicit_guidance = context.get("procedure_guidance")
        if explicit_guidance is None:
            guidance: dict[str, Any] = {
                "ordered_steps": [
                    {
                        "action_pattern": "follow_scripted_subtask",
                        "instruction": instruction,
                    }
                ],
                "avoid": [],
            }
            predicted_effects: list[dict[str, Any]] = []
        else:
            guidance, predicted_effects = self._validate_explicit_guidance(
                explicit_guidance
            )

        confidence = derivation.get("confidence", "low")
        if confidence not in _CONFIDENCE_VALUES:
            raise ProcedureBackendAbstention("malformed_confidence")

        evidence_refs = _deduplicate(
            (
                *_string_sequence(
                    derivation.get("evidence_event_refs", ()),
                    label="derivation.evidence_event_refs",
                ),
                *before_refs,
                *_string_sequence(
                    transition.get("action_event_refs", ()),
                    label="transition.action_event_refs",
                ),
                *after_refs,
            )
        )
        return {
            "summary": instruction,
            "kind": "procedure",
            "status": "candidate",
            "confidence": confidence,
            "condition": condition,
            "guidance": guidance,
            "predicted_effects": predicted_effects,
            "evidence_event_refs": list(evidence_refs),
        }

    @staticmethod
    def _validate_explicit_guidance(
        raw: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(raw, Mapping):
            raise ProcedureBackendAbstention("malformed_procedure_guidance")
        allowed = {"ordered_steps", "avoid", "predicted_effects"}
        if set(raw) - allowed:
            raise ProcedureBackendAbstention("malformed_procedure_guidance")

        ordered_steps = raw.get("ordered_steps")
        if (
            isinstance(ordered_steps, (str, bytes))
            or not isinstance(ordered_steps, Sequence)
            or not ordered_steps
        ):
            raise ProcedureBackendAbstention("malformed_procedure_guidance")
        normalized_steps: list[dict[str, Any]] = []
        step_keys = {"action_pattern", "manipulation_phase", "instruction"}
        for step in ordered_steps:
            if not isinstance(step, Mapping) or set(step) - step_keys:
                raise ProcedureBackendAbstention("malformed_procedure_guidance")
            action_pattern = step.get("action_pattern")
            if not isinstance(action_pattern, str) or not action_pattern.strip():
                raise ProcedureBackendAbstention("malformed_procedure_guidance")
            for key in ("manipulation_phase", "instruction"):
                value = step.get(key)
                if value is not None and not isinstance(value, str):
                    raise ProcedureBackendAbstention("malformed_procedure_guidance")
            normalized_steps.append(deepcopy(dict(step)))

        avoid = raw.get("avoid", [])
        if (
            isinstance(avoid, (str, bytes))
            or not isinstance(avoid, Sequence)
            or any(not isinstance(item, str) or not item.strip() for item in avoid)
        ):
            raise ProcedureBackendAbstention("malformed_procedure_guidance")

        predicted = raw.get("predicted_effects", [])
        if (
            isinstance(predicted, (str, bytes))
            or not isinstance(predicted, Sequence)
            or any(not isinstance(item, Mapping) for item in predicted)
        ):
            raise ProcedureBackendAbstention("malformed_procedure_guidance")
        guidance = {
            "ordered_steps": normalized_steps,
            "avoid": list(avoid),
        }
        return guidance, [deepcopy(dict(item)) for item in predicted]


class CandidateProcedureReflector:
    """Validate and return one procedure candidate, or explicitly abstain."""

    def __init__(
        self,
        backend: ProcedureCandidateBackend | None = None,
        *,
        producer_name: str = "expert_procedure_reflector",
        producer_version: str = "1",
        created_at: str | None = None,
    ) -> None:
        if not str(producer_name).strip() or not str(producer_version).strip():
            raise ValueError("reflector producer name/version must be non-empty")
        self._backend = backend if backend is not None else ScriptedProcedureBackend()
        self._producer_name = str(producer_name)
        self._producer_version = str(producer_version)
        self._created_at = deepcopy(created_at)

    @staticmethod
    def _abstain(reason: str) -> ReflectorOutput:
        abstention = ReflectorAbstention(reason=str(reason))
        return ReflectorOutput(
            summary=abstention.to_summary(),
            candidate_experience={},
            evidence_event_ids=(),
        )

    def reflect(self, request: ReflectorInput) -> ReflectorOutput:
        try:
            trajectory_version = request.trajectory.schema_version
        except Exception:
            return self._abstain("malformed_reflector_input")
        if trajectory_version != _TRAJECTORY_SCHEMA_VERSION:
            return self._abstain("unsupported_trajectory_schema")

        try:
            trajectory = {
                "trajectory_id": deepcopy(request.trajectory.trajectory_id),
                "instruction": deepcopy(request.trajectory.instruction),
                "trace_events": deepcopy(list(request.trajectory.trace_events)),
                "outcome": deepcopy(dict(request.trajectory.outcome)),
                "schema_version": trajectory_version,
            }
            scene_memory = deepcopy(dict(request.scene_memory))
            context = deepcopy(dict(request.experience_context))
        except Exception:
            return self._abstain("malformed_reflector_input")

        if not str(trajectory.get("trajectory_id", "")).strip():
            return self._abstain("malformed_reflector_input")
        missing = [
            key
            for key in ("normalized_segment", "evidence_index", "provenance")
            if key not in context
        ]
        if missing:
            return self._abstain("missing_experience_context")

        try:
            raw_segment = _as_mapping_copy(
                context["normalized_segment"],
                label="normalized_segment",
            )
        except Exception:
            return self._abstain("malformed_normalized_segment")
        if (
            raw_segment.get("schema") not in {_SEGMENT_SCHEMA, "tcm/subtask_segment/v1"}
            or raw_segment.get("schema_version") != _SEGMENT_SCHEMA_VERSION
        ):
            return self._abstain("unsupported_segment_schema")

        evidence_index = context["evidence_index"]
        if not isinstance(evidence_index, EvidenceIndex):
            return self._abstain("missing_evidence_index")

        try:
            segment_record = SubtaskSegmentV1.from_dict(
                deepcopy(raw_segment),
            )
            segment = segment_record.to_dict()
        except SchemaValidationError:
            return self._abstain("malformed_normalized_segment")
        except Exception:
            return self._abstain("malformed_normalized_segment")

        trajectory_id = str(trajectory["trajectory_id"])
        if str(segment.get("trajectory_id", "")) != trajectory_id:
            return self._abstain("segment_trajectory_mismatch")

        try:
            provenance_payload = _as_mapping_copy(
                context["provenance"],
                label="provenance",
            )
            provenance = ProvenanceV1.from_dict(provenance_payload).to_dict()
        except SchemaValidationError:
            return self._abstain("malformed_provenance")
        except Exception:
            return self._abstain("malformed_provenance")

        if self._expert_scene_memory_is_masquerading(scene_memory, provenance):
            return self._abstain("expert_scene_memory_masquerades_as_available")
        if isinstance(self._backend, ScriptedProcedureBackend) and not (
            provenance.get("source_kind") == "benchmark_expert"
            and provenance.get("source_subtype") == "benchmark_scripted_expert"
            and provenance.get("oracle_derived") is True
            and provenance.get("expert_derived") is True
        ):
            return self._abstain("unsupported_scripted_expert_provenance")

        transition = segment.get("transition")
        derivation = segment.get("derivation")
        if not isinstance(transition, Mapping) or not isinstance(derivation, Mapping):
            return self._abstain("malformed_normalized_segment")
        try:
            before_refs = _string_sequence(
                transition.get("before_event_refs", ()),
                label="transition.before_event_refs",
            )
            action_refs = _string_sequence(
                transition.get("action_event_refs", ()),
                label="transition.action_event_refs",
            )
            after_refs = _string_sequence(
                transition.get("after_event_refs", ()),
                label="transition.after_event_refs",
            )
            derivation_refs = _string_sequence(
                derivation.get("evidence_event_refs", ()),
                label="derivation.evidence_event_refs",
            )
            effect_refs: list[str] = []
            outcome = segment.get("outcome", {})
            if isinstance(outcome, Mapping):
                for effect_number, effect in enumerate(
                    outcome.get("observed_effects", [])
                ):
                    if not isinstance(effect, Mapping):
                        raise TypeError("observed effect must be a mapping")
                    effect_refs.extend(
                        _string_sequence(
                            effect.get("evidence_event_refs", ()),
                            label=(
                                "outcome.observed_effects"
                                f"[{effect_number}].evidence_event_refs"
                            ),
                        )
                    )
        except (TypeError, ValueError):
            return self._abstain("malformed_evidence_refs")
        if not before_refs or not after_refs or not derivation_refs:
            return self._abstain("insufficient_evidence_refs")
        declared_refs = _deduplicate(
            (
                *derivation_refs,
                *before_refs,
                *action_refs,
                *after_refs,
                *effect_refs,
            )
        )

        try:
            validation = evidence_index.validate_refs(
                declared_refs,
                trajectory_id=trajectory_id,
            )
            if validation is False:
                return self._abstain("unresolved_evidence_ref")
            resolved_evidence: dict[str, dict[str, Any]] = {}
            provenance_source_ids = {
                str(source_ref.get("source_ref_id", ""))
                for source_ref in provenance.get("source_refs", [])
                if isinstance(source_ref, Mapping)
            }
            for ref in declared_refs:
                entry = evidence_index.resolve(ref)
                if entry is None:
                    return self._abstain("unresolved_evidence_ref")
                entry_payload = _entry_to_mapping(entry)
                if entry_payload.get("source_ref_id") not in provenance_source_ids:
                    return self._abstain("evidence_provenance_mismatch")
                resolved_evidence[ref] = entry_payload
        except Exception:
            return self._abstain("unresolved_evidence_ref")

        backend_context = {
            key: deepcopy(value)
            for key, value in context.items()
            if key not in {"normalized_segment", "evidence_index", "provenance"}
        }
        try:
            proposal = self._backend.propose(
                segment=deepcopy(segment),
                evidence=deepcopy(resolved_evidence),
                trajectory=deepcopy(trajectory),
                scene_memory=deepcopy(scene_memory),
                experience_context=deepcopy(backend_context),
            )
        except ProcedureBackendAbstention as exc:
            return self._abstain(exc.reason)
        except Exception:
            return self._abstain("backend_exception")

        proposal_error = self._proposal_error(proposal)
        if proposal_error is not None:
            return self._abstain(proposal_error)
        assert isinstance(proposal, Mapping)

        try:
            proposed_refs = _string_sequence(
                proposal["evidence_event_refs"],
                label="backend evidence_event_refs",
            )
        except (KeyError, TypeError, ValueError):
            return self._abstain("malformed_backend_output")
        if not proposed_refs:
            return self._abstain("insufficient_evidence_refs")
        if any(ref not in declared_refs for ref in proposed_refs):
            return self._abstain("invented_evidence_ref")
        try:
            validation = evidence_index.validate_refs(
                proposed_refs,
                trajectory_id=trajectory_id,
            )
            if validation is False:
                return self._abstain("invented_evidence_ref")
            if any(evidence_index.resolve(ref) is None for ref in proposed_refs):
                return self._abstain("invented_evidence_ref")
        except Exception:
            return self._abstain("invented_evidence_ref")

        candidate_provenance = deepcopy(provenance)
        candidate_provenance["producer"] = {
            "kind": "reflector",
            "name": self._producer_name,
            "version": self._producer_version,
        }
        derivation_created_at = derivation.get("created_at")
        created_at = (
            self._created_at or derivation_created_at or provenance.get("created_at")
        )
        if not isinstance(created_at, str) or not created_at.strip():
            return self._abstain("missing_candidate_created_at")
        candidate_provenance["created_at"] = created_at

        condition = deepcopy(dict(proposal["condition"]))
        guidance = deepcopy(dict(proposal["guidance"]))
        predicted_effects = [
            deepcopy(dict(effect)) for effect in proposal["predicted_effects"]
        ]
        candidate = {
            "schema": _PROCEDURE_SCHEMA,
            "schema_version": 1,
            "experience_id": stable_experience_id(
                "procedure",
                condition,
                guidance,
                predicted_effects,
                prefix="exp",
            ),
            "kind": "procedure",
            "status": "candidate",
            "confidence": proposal["confidence"],
            "condition": condition,
            "guidance": guidance,
            "predicted_effects": predicted_effects,
            "evidence": {
                "supporting_segment_refs": [str(segment["segment_id"])],
                "opposing_segment_refs": [],
                "support_count": 1,
                "opposing_count": 0,
                "evaluation_status": "not_evaluated",
                "evaluation_ref": None,
            },
            "last_validated_at": None,
            "provenance": candidate_provenance,
        }
        try:
            candidate_record = ProcedureExperienceV1.from_dict(candidate)
            validate_reflector_candidate(candidate_record)
            candidate_payload = candidate_record.to_dict()
        except SchemaValidationError:
            return self._abstain("invalid_procedure_candidate")
        except Exception:
            return self._abstain("invalid_procedure_candidate")

        return ReflectorOutput(
            summary=str(proposal["summary"]),
            candidate_experience=deepcopy(candidate_payload),
            evidence_event_ids=tuple(proposed_refs),
        )

    @staticmethod
    def _expert_scene_memory_is_masquerading(
        scene_memory: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> bool:
        if provenance.get("source_kind") != "benchmark_expert":
            return False
        if not scene_memory:
            return False
        if scene_memory.get("available") is not False:
            return True
        snapshot = scene_memory.get("snapshot", {})
        if snapshot not in ({}, None):
            return True
        source = scene_memory.get("source", "unavailable")
        return source != "unavailable"

    @staticmethod
    def _proposal_error(proposal: Any) -> str | None:
        if not isinstance(proposal, Mapping):
            return "malformed_backend_output"
        allowed = {
            "summary",
            "kind",
            "status",
            "confidence",
            "condition",
            "guidance",
            "predicted_effects",
            "evidence_event_refs",
        }
        if set(proposal) != allowed:
            return "malformed_backend_output"
        if proposal.get("kind") != "procedure":
            return "forbidden_candidate_kind"
        if proposal.get("status") != "candidate":
            return "forbidden_candidate_status"
        if proposal.get("confidence") not in _CONFIDENCE_VALUES:
            return "malformed_backend_output"
        summary = proposal.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return "malformed_backend_output"
        if not isinstance(proposal.get("condition"), Mapping):
            return "malformed_backend_output"
        if not isinstance(proposal.get("guidance"), Mapping):
            return "malformed_backend_output"
        effects = proposal.get("predicted_effects")
        if (
            isinstance(effects, (str, bytes))
            or not isinstance(effects, Sequence)
            or any(not isinstance(item, Mapping) for item in effects)
        ):
            return "malformed_backend_output"
        return None


__all__ = [
    "CandidateProcedureReflector",
    "ProcedureBackend",
    "ProcedureBackendAbstention",
    "ProcedureCandidateBackend",
    "ReflectorAbstention",
    "ScriptedProcedureBackend",
]

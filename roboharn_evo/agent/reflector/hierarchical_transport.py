"""Audited OpenAI-compatible transport for hierarchical reflection.

This module connects the provider-neutral contracts in :mod:`hierarchical` to
the already fail-closed Responses transport.  It deliberately keeps image
planning, semantic admission, and publication outside the network boundary.

Every outbound request uses request-local aliases.  Filesystem locations,
content digests, raw records, trajectory identifiers, and seed or episode
identifiers are never projected into model text.  Derived images are labelled
as presentations of their original evidence and can never be cited as new
physical evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Protocol, runtime_checkable

from .hierarchical import (
    ActionChunkBatchReflectionRequest,
    AttributionPassRequest,
    LayerModelOutput,
    ProcedureDraftRequest,
    SubtaskSummaryRequest,
    action_chunk_batch_output_json_schema,
    attribution_output_json_schema,
    hierarchical_source_semantics,
    procedure_draft_output_json_schema,
    required_attribution_paths,
    semantic_leakage_audit_contract,
    subtask_summary_output_json_schema,
)
from .multimodal_transport import (
    CapabilityReport,
    ImageEgressAuthorization,
    ModelInputProjectionError,
    MultimodalImage,
    OpenAICompatibleMultimodalTransport,
    validate_model_projection_text,
)
from .visual_presentations import VisualPresentationResult


_LAYER_ACTION = "action_chunk_batch"
_LAYER_SUBTASK = "subtask_summary"
_LAYER_DRAFT = "procedure_draft"
_LAYER_ATTRIBUTION = "evidence_attribution"
_PRIVATE_KEY_RE = re.compile(
    r"(?:^|_)(?:"
    r"path|paths|uri|url|file|filename|hash|sha|sha256|digest|raw|seed|episode|"
    r"trajectory|image_ref|source_image|checkpoint|manifest"
    r")(?:$|_)",
    re.IGNORECASE,
)
_IDENTIFIER_KEY_RE = re.compile(r"(?:^|_)(?:id|ids|ref|refs)$", re.IGNORECASE)
_PRIVATE_RUN_IDENTIFIER_RE = re.compile(
    r"\b(?:seed|episode)[_\s:#=-]*(?:id[_\s:#=-]*)?"
    r"[A-Za-z0-9]*[0-9][A-Za-z0-9_.-]*\b",
    re.IGNORECASE,
)
_DROP = object()


class HierarchicalTransportError(RuntimeError):
    """Base error for the local hierarchical transport boundary."""


class HierarchicalImagePlanError(HierarchicalTransportError):
    """A local image plan is incomplete, inconsistent, or inadmissible."""


class HierarchicalBudgetError(HierarchicalTransportError):
    """A model call would exceed a configured trajectory budget."""


@dataclass(frozen=True, slots=True)
class HierarchicalTransportLimits:
    """Configurable request and trajectory limits for one backend instance."""

    max_model_calls_per_trajectory: int = 8
    max_images_per_request: int = 16
    max_images_per_trajectory: int = 64
    max_image_bytes: int = 8 * 1024 * 1024
    max_total_image_bytes_per_request: int = 8 * 1024 * 1024
    max_total_image_bytes_per_trajectory: int = 8 * 1024 * 1024
    layer2_verification_images: int = 0

    def __post_init__(self) -> None:
        positive = (
            "max_model_calls_per_trajectory",
            "max_images_per_request",
            "max_images_per_trajectory",
            "max_image_bytes",
            "max_total_image_bytes_per_request",
            "max_total_image_bytes_per_trajectory",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        value = self.layer2_verification_images
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("layer2_verification_images must be non-negative")
        if value > self.max_images_per_request:
            raise ValueError(
                "layer2_verification_images exceeds max_images_per_request"
            )


@dataclass(frozen=True, slots=True)
class PlannedPresentation:
    """One local image payload and the original evidence it presents."""

    presentation_id: str
    mime_type: str
    content: bytes = field(repr=False)
    underlying_evidence_refs: tuple[str, ...]
    presentation_kind: str = "source_frame"

    def __post_init__(self) -> None:
        if not isinstance(self.presentation_id, str) or not self.presentation_id:
            raise ValueError("presentation_id must be non-empty")
        if self.mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("unsupported presentation MIME type")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("presentation content must be non-empty bytes")
        refs = tuple(self.underlying_evidence_refs)
        if not refs or any(not isinstance(value, str) or not value for value in refs):
            raise ValueError("underlying_evidence_refs must be non-empty strings")
        if len(set(refs)) != len(refs):
            raise ValueError("underlying_evidence_refs must be unique")
        if not isinstance(self.presentation_kind, str) or not self.presentation_kind:
            raise ValueError("presentation_kind must be non-empty")
        object.__setattr__(self, "underlying_evidence_refs", refs)


@dataclass(frozen=True, slots=True)
class HierarchicalImagePlanQuery:
    """Local-only query supplied to a generic image-plan provider."""

    layer: str
    trajectory_id: str
    segment_id: str | None
    evidence_refs: tuple[str, ...]
    max_images: int
    max_total_bytes: int
    action_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HierarchicalImagePlan:
    """A provider-selected ordered image plan for one model request."""

    presentations: tuple[PlannedPresentation, ...]
    coverage_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(self.presentations)
        if any(not isinstance(value, PlannedPresentation) for value in values):
            raise TypeError("presentations must contain PlannedPresentation values")
        ids = [value.presentation_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("image plan contains duplicate presentation IDs")
        object.__setattr__(self, "presentations", values)
        object.__setattr__(self, "coverage_notes", tuple(self.coverage_notes))


@runtime_checkable
class HierarchicalImagePlanProvider(Protocol):
    """Provider-neutral, local-only image planning seam."""

    def plan_images(
        self, query: HierarchicalImagePlanQuery
    ) -> HierarchicalImagePlan: ...


class VisualPresentationImagePlanProvider:
    """Adapt a :class:`VisualPresentationResult` to the generic plan seam."""

    def __init__(self, result: VisualPresentationResult) -> None:
        if not isinstance(result, VisualPresentationResult):
            raise TypeError("result must be VisualPresentationResult")
        if not result.admissible:
            raise HierarchicalImagePlanError(
                "visual presentation result is not admissible"
            )
        payloads = {value.presentation_id: value for value in result.outbound_payloads}
        records = {
            str(value.get("presentation_id")): dict(value)
            for value in result.presentation_records
        }
        if set(payloads) != set(records):
            raise HierarchicalImagePlanError(
                "presentation records and outbound payloads do not match"
            )
        self._result = result
        self._presentations: dict[str, PlannedPresentation] = {}
        for presentation_id, payload in payloads.items():
            record = records[presentation_id]
            refs = record.get("underlying_evidence_refs")
            if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
                raise HierarchicalImagePlanError(
                    "presentation is missing underlying evidence refs"
                )
            self._presentations[presentation_id] = PlannedPresentation(
                presentation_id=presentation_id,
                mime_type=payload.media_type,
                content=payload.data,
                underlying_evidence_refs=tuple(str(value) for value in refs),
                presentation_kind=str(record.get("presentation_kind", "source_frame")),
            )

    @staticmethod
    def _ids_from_plans(plans: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for plan in plans:
            values = plan.get("presentation_ids", ())
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise HierarchicalImagePlanError(
                    "request plan presentation_ids must be an array"
                )
            for value in values:
                presentation_id = str(value)
                if presentation_id not in seen:
                    seen.add(presentation_id)
                    result.append(presentation_id)
        return tuple(result)

    def plan_images(self, query: HierarchicalImagePlanQuery) -> HierarchicalImagePlan:
        known_refs = set(query.evidence_refs)
        if query.layer == _LAYER_ACTION:
            plans = tuple(
                value
                for value in self._result.layer1_request_plans
                if value.get("segment_id") == query.segment_id
            )
            if not plans:
                raise HierarchicalImagePlanError(
                    "no Layer-1 request plan exists for the segment"
                )
            if len(plans) > 1:
                requested_chunks = set(query.action_chunk_ids)
                exact = tuple(
                    value
                    for value in plans
                    if isinstance(value.get("underlying_evidence_refs"), list)
                    and set(str(item) for item in value["underlying_evidence_refs"])
                    == known_refs
                    and (
                        not requested_chunks
                        or (
                            isinstance(value.get("action_chunk_ids"), list)
                            and set(str(item) for item in value["action_chunk_ids"])
                            == requested_chunks
                        )
                    )
                )
                if len(exact) != 1:
                    raise HierarchicalImagePlanError(
                        "split Layer-1 plan has no unique exact registry match"
                    )
                plans = exact
            ids = self._ids_from_plans(plans)
        elif query.layer == _LAYER_DRAFT:
            raw = self._result.cross_segment_feedback_plan.get("requests", ())
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise HierarchicalImagePlanError(
                    "cross-segment request plan must be an array"
                )
            plans = tuple(value for value in raw if isinstance(value, Mapping))
            if len(plans) != 1:
                raise HierarchicalImagePlanError(
                    "Layer 3 requires one lossless cross-segment request plan; "
                    "multiple plans cannot be silently merged"
                )
            ids = self._ids_from_plans(plans)
        elif query.layer == _LAYER_SUBTASK:
            matching = [
                value.presentation_id
                for value in self._presentations.values()
                if set(value.underlying_evidence_refs).issubset(known_refs)
                and set(value.underlying_evidence_refs) & known_refs
            ]
            ids = tuple(matching[: query.max_images])
        else:
            ids = ()
        missing = [value for value in ids if value not in self._presentations]
        if missing:
            raise HierarchicalImagePlanError(
                "request plan references an unknown presentation"
            )
        selected: list[PlannedPresentation] = []
        for presentation_id in ids:
            value = self._presentations[presentation_id]
            if not set(value.underlying_evidence_refs).issubset(known_refs):
                raise HierarchicalImagePlanError(
                    "presentation includes evidence outside the layer request"
                )
            selected.append(value)
        return HierarchicalImagePlan(tuple(selected))


@dataclass(slots=True)
class _Aliases:
    evidence: dict[str, str] = field(default_factory=dict)
    segments: dict[str, str] = field(default_factory=dict)
    chunks: dict[str, str] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)
    targets: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _add(values: dict[str, str], real: str, prefix: str) -> str:
        if real not in values:
            values[real] = f"{prefix}_ref_{len(values):03d}"
        return values[real]

    def evidence_ref(self, real: str) -> str:
        return self._add(self.evidence, real, "evidence")

    def segment_ref(self, real: str) -> str:
        return self._add(self.segments, real, "segment")

    def chunk_ref(self, real: str) -> str:
        return self._add(self.chunks, real, "chunk")

    def fact_ref(self, real: str) -> str:
        return self._add(self.facts, real, "fact")

    def target_ref(self, real: str) -> str:
        return self._add(self.targets, real, "target")

    def reverse(self, kind: str) -> dict[str, str]:
        source = {
            "evidence": self.evidence,
            "segment": self.segments,
            "chunk": self.chunks,
            "fact": self.facts,
            "target": self.targets,
        }[kind]
        return {alias: real for real, alias in source.items()}

    def audit(self) -> dict[str, Any]:
        """Return counts and a binding digest, never the private mapping."""

        payload = {
            key: sorted(value.items())
            for key, value in {
                "evidence": self.evidence,
                "segment": self.segments,
                "chunk": self.chunks,
                "fact": self.facts,
                "target": self.targets,
            }.items()
        }
        return {
            "mapping_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
            "evidence_count": len(self.evidence),
            "segment_count": len(self.segments),
            "chunk_count": len(self.chunks),
            "fact_count": len(self.facts),
            "target_count": len(self.targets),
            "real_identifiers_logged": False,
        }


@dataclass(slots=True)
class _TrajectoryUsage:
    calls: int = 0
    image_count: int = 0
    image_bytes: int = 0


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_outbound_projection(value: str, aliases: _Aliases) -> None:
    validate_model_projection_text(value)
    if _PRIVATE_RUN_IDENTIFIER_RE.search(value):
        raise ModelInputProjectionError(
            "model projection contains a seed or episode identifier"
        )
    private_values = (
        *aliases.evidence,
        *aliases.segments,
        *aliases.chunks,
        *aliases.facts,
        *aliases.targets,
    )
    if any(
        len(real) >= 4
        and re.search(
            rf"(?<![A-Za-z0-9_.-]){re.escape(real)}(?![A-Za-z0-9_.-])",
            value,
        )
        for real in private_values
    ):
        raise ModelInputProjectionError(
            "model projection contains a non-aliased local identifier"
        )


def _safe_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ModelInputProjectionError(f"{label} must be text")
    validate_model_projection_text(value)
    return value


def _safe_optional_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ModelInputProjectionError(f"{label} must be text")
    if value:
        validate_model_projection_text(value)
    return value


def _safe_metadata(value: Any, aliases: _Aliases, *, key: str = "") -> Any:
    """Project useful low-dimensional metadata through a strict deny list."""

    if _PRIVATE_KEY_RE.search(key):
        return _DROP
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if not isinstance(child_key, str) or _IDENTIFIER_KEY_RE.search(child_key):
                continue
            projected = _safe_metadata(child_value, aliases, key=child_key)
            if projected is not _DROP:
                result[child_key] = projected
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for child in value:
            projected = _safe_metadata(child, aliases, key=key)
            if projected is not _DROP:
                result.append(projected)
        return result
    if isinstance(value, str):
        if value in aliases.evidence:
            return aliases.evidence[value]
        if value in aliases.segments:
            return aliases.segments[value]
        if value in aliases.chunks:
            return aliases.chunks[value]
        if value in aliases.facts:
            return aliases.facts[value]
        try:
            validate_model_projection_text(value)
        except ModelInputProjectionError:
            return _DROP
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _DROP


def _project_outcome(value: Mapping[str, Any], aliases: _Aliases) -> dict[str, Any]:
    if value.get("availability") != "verified":
        return {"availability": "unverified"}
    evidence_ref = value.get("evidence_ref")
    status = value.get("status")
    if not isinstance(evidence_ref, str) or not isinstance(status, str):
        return {"availability": "unverified"}
    return {
        "availability": "verified",
        "status": _safe_text(status, label="trajectory outcome status"),
        "evidence_ref": aliases.evidence_ref(evidence_ref),
    }


def _project_evidence(value: Any, aliases: _Aliases) -> dict[str, Any]:
    evidence_ref = str(value.evidence_ref)
    payload = _safe_metadata(value.payload, aliases)
    return {
        "evidence_ref": aliases.evidence_ref(evidence_ref),
        "metadata": payload if isinstance(payload, Mapping) else {},
    }


def _project_fact(value: Mapping[str, Any], aliases: _Aliases) -> dict[str, Any]:
    return {
        "fact_id": aliases.fact_ref(str(value["fact_id"])),
        "claim": _safe_text(value["claim"], label="fact claim"),
        "source_type": value["source_type"],
        "supporting_evidence_refs": [
            aliases.evidence_ref(str(ref)) for ref in value["supporting_evidence_refs"]
        ],
        "supporting_segment_refs": [
            aliases.segment_ref(str(ref)) for ref in value["supporting_segment_refs"]
        ],
        "confidence": value["confidence"],
        "uncertainty": _safe_optional_text(
            value["uncertainty"], label="fact uncertainty"
        ),
        "temporal_scope": value["temporal_scope"],
        "transferable_allowed": False,
        "structured_value": _safe_optional_text(
            value["structured_value"], label="fact structured value"
        ),
    }


def _register_chunk_aliases(
    request: ActionChunkBatchReflectionRequest,
    aliases: _Aliases,
) -> None:
    aliases.segment_ref(request.segment_id)
    for value in request.segment_visual_evidence:
        aliases.evidence_ref(value.evidence_ref)
        source_segment = value.payload.get("segment_id")
        if isinstance(source_segment, str):
            aliases.segment_ref(source_segment)
    for chunk in request.chunks:
        aliases.chunk_ref(chunk.action_chunk_id)
        aliases.evidence_ref(str(chunk.subtask_annotation["evidence_ref"]))
        for value in (*chunk.observed_robot_state, *chunk.visual_evidence):
            aliases.evidence_ref(value.evidence_ref)
        outcome_ref = chunk.trajectory_outcome.get("evidence_ref")
        if isinstance(outcome_ref, str):
            aliases.evidence_ref(outcome_ref)


def _project_chunk(
    value: Any,
    aliases: _Aliases,
    *,
    ordinal: int,
) -> dict[str, Any]:
    chunk = value.action_chunk
    frame_range = chunk.get("frame_range", {})
    start = frame_range.get("start_inclusive")
    end = frame_range.get("end_inclusive")
    duration = end - start + 1 if isinstance(start, int) and isinstance(end, int) else 0
    projected: dict[str, Any] = {
        "action_chunk_ref": aliases.chunk_ref(value.action_chunk_id),
        "temporal_ordinal": ordinal,
        "duration_in_observed_frames": duration,
        "observed_robot_state": [
            _project_evidence(item, aliases) for item in value.observed_robot_state
        ],
        "visual_evidence": [
            _project_evidence(item, aliases) for item in value.visual_evidence
        ],
    }
    for field_name in (
        "active_arms",
        "observed_motion_state",
        "gripper_transitions",
        "inferred_phase",
        "confidence",
        "uncertainty",
    ):
        if field_name in chunk:
            safe = _safe_metadata(chunk[field_name], aliases, key=field_name)
            if safe is not _DROP:
                projected[field_name] = safe
    return projected


def _presentation_projection(
    plan: HierarchicalImagePlan,
    aliases: _Aliases,
) -> tuple[tuple[MultimodalImage, ...], list[dict[str, Any]]]:
    images: list[MultimodalImage] = []
    bindings: list[dict[str, Any]] = []
    for index, value in enumerate(plan.presentations):
        original_refs = [
            aliases.evidence_ref(ref) for ref in value.underlying_evidence_refs
        ]
        image_alias = f"model_image_{index:03d}"
        images.append(
            MultimodalImage(
                evidence_id=image_alias,
                mime_type=value.mime_type,
                content=value.content,
                detail="high",
            )
        )
        bindings.append(
            {
                "model_image_alias": image_alias,
                "presentation_kind": _safe_text(
                    value.presentation_kind,
                    label="presentation kind",
                ),
                "presentation_only": True,
                "underlying_original_evidence_refs": original_refs,
                "citation_rule": "cite_underlying_original_evidence_only",
            }
        )
    return tuple(images), bindings


def _local_presentation_bindings(
    plan: HierarchicalImagePlan,
) -> list[dict[str, Any]]:
    """Return local-only image bindings for the persisted call audit."""

    return [
        {
            "model_image_alias": f"model_image_{index:03d}",
            "presentation_id": value.presentation_id,
            "presentation_kind": value.presentation_kind,
            "underlying_evidence_refs": list(value.underlying_evidence_refs),
            "presentation_only": True,
        }
        for index, value in enumerate(plan.presentations)
    ]


def _action_projection(
    request: ActionChunkBatchReflectionRequest,
    aliases: _Aliases,
) -> dict[str, Any]:
    first = request.chunks[0]
    return {
        "layer": "action_chunk_evidence",
        "segment_ref": aliases.segment_ref(request.segment_id),
        "subtask_annotation": {
            "evidence_ref": aliases.evidence_ref(
                str(first.subtask_annotation["evidence_ref"])
            ),
            "text": _safe_text(
                first.subtask_annotation["text"], label="subtask annotation"
            ),
            "epistemic_role": "intended_action_description_only",
        },
        "ordered_action_chunks": [
            _project_chunk(value, aliases, ordinal=index)
            for index, value in enumerate(request.chunks)
        ],
        "segment_visual_evidence": [
            _project_evidence(value, aliases)
            for value in request.segment_visual_evidence
        ],
        "segment_evidence_rule": (
            "segment-level images provide context; action-chunk facts may cite only "
            "original refs explicitly bound inside that action chunk"
        ),
        "trajectory_outcome": _project_outcome(first.trajectory_outcome, aliases),
        "source_semantics": hierarchical_source_semantics(),
    }


def _subtask_projection(
    request: SubtaskSummaryRequest,
    aliases: _Aliases,
) -> dict[str, Any]:
    aliases.segment_ref(request.segment_id)
    aliases.evidence_ref(str(request.subtask_annotation["evidence_ref"]))
    for chunk in request.ordered_action_chunks:
        aliases.chunk_ref(str(chunk["action_chunk_id"]))
    for fact in request.admitted_chunk_facts:
        aliases.fact_ref(str(fact["fact_id"]))
        for ref in fact["supporting_evidence_refs"]:
            aliases.evidence_ref(str(ref))
    for evidence in request.verification_visual_evidence:
        aliases.evidence_ref(evidence.evidence_ref)
    return {
        "layer": "subtask_summary",
        "segment_ref": aliases.segment_ref(request.segment_id),
        "segment_ordinal": request.segment_index,
        "subtask_annotation": {
            "evidence_ref": aliases.evidence_ref(
                str(request.subtask_annotation["evidence_ref"])
            ),
            "text": _safe_text(
                request.subtask_annotation["text"], label="subtask annotation"
            ),
            "epistemic_role": "intended_action_description_only",
        },
        "ordered_action_chunk_refs": [
            aliases.chunk_ref(str(value["action_chunk_id"]))
            for value in request.ordered_action_chunks
        ],
        "admitted_private_chunk_facts": [
            _project_fact(value, aliases) for value in request.admitted_chunk_facts
        ],
        "verification_visual_evidence": [
            _project_evidence(value, aliases)
            for value in request.verification_visual_evidence
        ],
        "trajectory_outcome": _project_outcome(request.trajectory_outcome, aliases),
        "source_semantics": hierarchical_source_semantics(),
    }


def _project_summary(value: Mapping[str, Any], aliases: _Aliases) -> dict[str, Any]:
    result: dict[str, Any] = {
        "summary_id": "local_summary",
        "segment_id": aliases.segment_ref(str(value["segment_id"])),
        "attempted_action": _project_fact(value["attempted_action"], aliases),
    }
    for field_name in (
        "observed_execution",
        "state_changes",
        "unchanged_results",
        "release_observations",
        "feedback_observations",
    ):
        result[field_name] = [
            _project_fact(fact, aliases) for fact in value[field_name]
        ]
    result["supporting_chunk_fact_ids"] = [
        aliases.fact_ref(str(fact_id)) for fact_id in value["supporting_chunk_fact_ids"]
    ]
    result["confidence"] = value["confidence"]
    result["uncertainties"] = [
        _safe_text(item, label="subtask uncertainty") for item in value["uncertainties"]
    ]
    return result


def _draft_projection(
    request: ProcedureDraftRequest,
    aliases: _Aliases,
) -> dict[str, Any]:
    for summary in request.ordered_subtask_summaries:
        aliases.segment_ref(str(summary["segment_id"]))
    for fact in request.episode_specific_facts:
        aliases.fact_ref(str(fact["fact_id"]))
        for ref in fact["supporting_evidence_refs"]:
            aliases.evidence_ref(str(ref))
    for evidence in request.cross_segment_visual_evidence:
        aliases.evidence_ref(evidence.evidence_ref)
        segment = evidence.payload.get("segment_id")
        if isinstance(segment, str):
            aliases.segment_ref(segment)
    return {
        "layer": "whole_trajectory_procedure_draft",
        "instruction": _safe_text(request.instruction, label="instruction"),
        "ordered_subtask_summaries": [
            _project_summary(value, aliases)
            for value in request.ordered_subtask_summaries
        ],
        "episode_private_facts": [
            _project_fact(value, aliases) for value in request.episode_specific_facts
        ],
        "cross_segment_visual_evidence": [
            _project_evidence(value, aliases)
            for value in request.cross_segment_visual_evidence
        ],
        "trajectory_outcome": _project_outcome(request.trajectory_outcome, aliases),
        "source_semantics": hierarchical_source_semantics(),
    }


def _attribution_projection(
    request: AttributionPassRequest,
    aliases: _Aliases,
) -> dict[str, Any]:
    for segment in request.valid_segment_refs:
        aliases.segment_ref(segment)
    for evidence in request.valid_evidence_refs:
        aliases.evidence_ref(evidence)
    for fact in request.episode_specific_facts:
        aliases.fact_ref(str(fact["fact_id"]))
    target_paths = required_attribution_paths(request.procedure_draft)
    return {
        "layer": "evidence_attribution_and_semantic_leakage_audit",
        "procedure_draft": _safe_metadata(request.procedure_draft, aliases),
        "episode_private_facts": [
            _project_fact(value, aliases) for value in request.episode_specific_facts
        ],
        "valid_evidence_registry": [
            {
                "evidence_ref": aliases.evidence_ref(str(value["evidence_ref"])),
                "source_types": list(value["source_types"]),
                "segment_ref": aliases.segment_ref(str(value["segment_id"])),
            }
            for value in request.evidence_registry
        ],
        "required_target_refs": [
            {
                "target_ref": aliases.target_ref(path),
                "claim": _safe_metadata(
                    _value_at_path(request.procedure_draft, path), aliases
                ),
            }
            for path in target_paths
        ],
        "source_semantics": hierarchical_source_semantics(),
        "semantic_leakage_audit": semantic_leakage_audit_contract(),
    }


def _value_at_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for token in path.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            current = current[token]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            current = current[int(token)]
        else:  # pragma: no cover - required paths are generated from the draft.
            raise KeyError(path)
    return current


def _invalid_ref(kind: str) -> str:
    return f"invalid_external_{kind}_ref"


def _restore_output_refs(value: Mapping[str, Any], aliases: _Aliases) -> dict[str, Any]:
    """Restore known structured aliases; unknown refs remain invalid."""

    generated_fact_ids = {
        str(child)
        for child in _walk_field_values(value, "fact_id")
        if isinstance(child, str) and child not in aliases.reverse("fact")
    }
    reverse = {
        kind: aliases.reverse(kind)
        for kind in ("evidence", "segment", "chunk", "fact", "target")
    }

    def restore_scalar(item: Any, kind: str, *, generated_ok: bool = False) -> Any:
        if not isinstance(item, str):
            return item
        if item in reverse[kind]:
            return reverse[kind][item]
        if generated_ok and item in generated_fact_ids:
            return item
        return _invalid_ref(kind)

    def visit(item: Any, key: str = "") -> Any:
        if isinstance(item, Mapping):
            return {
                str(child_key): visit(child, str(child_key))
                for child_key, child in item.items()
            }
        if isinstance(item, list):
            if key in {"supporting_evidence_refs"}:
                return [restore_scalar(child, "evidence") for child in item]
            if key in {"supporting_segment_refs"}:
                return [restore_scalar(child, "segment") for child in item]
            if key in {"supporting_fact_ids", "supporting_chunk_fact_ids"}:
                return [
                    restore_scalar(child, "fact", generated_ok=True) for child in item
                ]
            return [visit(child, key) for child in item]
        if key == "action_chunk_id":
            return restore_scalar(item, "chunk")
        if key == "segment_id":
            return restore_scalar(item, "segment")
        if key == "target_path":
            return restore_scalar(item, "target")
        if key == "fact_id" and isinstance(item, str) and item in reverse["fact"]:
            return reverse["fact"][item]
        return item

    restored = visit(value)
    assert isinstance(restored, dict)
    return restored


def _walk_field_values(value: Any, field_name: str) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == field_name:
                result.append(child)
            result.extend(_walk_field_values(child, field_name))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            result.extend(_walk_field_values(child, field_name))
    return result


def generic_action_chunk_system_prompt() -> str:
    """Return the task-neutral Layer-1 prompt."""

    return (
        "Analyze every ordered action chunk in this one annotated subtask. "
        "Annotation describes intended action only. Observed robot state describes "
        "motion, gripper state, and timing, never a commanded high-level action. "
        "Visual evidence supports only visible execution, object state, release, "
        "and feedback. A model image is presentation-only; cite only its bound "
        "underlying original evidence refs. Emit one strict chunk output per chunk. "
        "Segment-level context images that are not bound inside a chunk cannot support "
        "that chunk's facts. "
        "Use partial status and explicit uncertainty when only some facts are "
        "supported; abstain only for the unsupported chunk, without discarding "
        "independently grounded facts. Never infer hidden answers or overall success."
    )


def generic_subtask_summary_system_prompt() -> str:
    """Return the task-neutral Layer-2 prompt."""

    return (
        "Summarize one ordered subtask from admitted private chunk facts. "
        "Keep annotation intent separate from observed robot timing and visible "
        "results. Report changed and unchanged observations, release, feedback, "
        "and uncertainty with exact admitted evidence refs. Verification images, "
        "when present, are presentation-only and bind to original evidence refs. "
        "Do not require visually proving an action parameter already stated as "
        "annotation intent. Use partial status when a useful grounded summary is "
        "possible and abstain only when the summary itself cannot be grounded."
    )


def generic_procedure_draft_system_prompt() -> str:
    """Return the task-neutral Layer-3 prompt."""

    return (
        "Compare the ordered subtask summaries and cross-segment feedback images. "
        "Cross-segment inference must cite original evidence spanning every declared "
        "segment. Infer a feedback-driven procedure: track attempted states, observe "
        "results after actions, change a bounded variable after failure, avoid "
        "repeating recorded attempts, and stop on visible success evidence. Keep all "
        "episode-private assignments, poses, identifiers, and answers out of "
        "transferable guidance. Model images and derived panels are presentation-only; "
        "cite their bound original evidence refs. Preserve uncertainties and abstain "
        "if a transferable procedure lacks grounded cross-segment support."
    )


def generic_attribution_system_prompt() -> str:
    """Return the task-neutral Layer-4 prompt."""

    return (
        "Independently attribute every supplied planner-facing target ref to admitted "
        "evidence and audit semantic leakage. Return each target ref exactly once. "
        "Use annotation only for intent, observed robot state only for motion and "
        "timing, visual evidence only for visible outcomes, and cross-segment "
        "inference only when cited evidence spans all declared segments. Compare the "
        "meaning of every draft claim against all episode-private facts; do not rely "
        "on lexical overlap or a synonym list. If a confident negative leakage "
        "verdict or complete attribution is unavailable, abstain."
    )


class OpenAICompatibleHierarchicalBackend:
    """Real bounded backend implementing ``HierarchicalReflectionBackend``."""

    backend_name = "openai_compatible_hierarchical_multimodal"

    def __init__(
        self,
        transport: OpenAICompatibleMultimodalTransport,
        *,
        authorization: ImageEgressAuthorization,
        image_plan_provider: (
            HierarchicalImagePlanProvider
            | Callable[[HierarchicalImagePlanQuery], HierarchicalImagePlan]
            | VisualPresentationResult
        ),
        limits: HierarchicalTransportLimits | None = None,
    ) -> None:
        if not isinstance(transport, OpenAICompatibleMultimodalTransport):
            raise TypeError("transport must be OpenAICompatibleMultimodalTransport")
        if not isinstance(authorization, ImageEgressAuthorization):
            raise TypeError("authorization must be ImageEgressAuthorization")
        if isinstance(image_plan_provider, VisualPresentationResult):
            provider: Any = VisualPresentationImagePlanProvider(image_plan_provider)
        else:
            provider = image_plan_provider
        if not callable(provider) and not callable(
            getattr(provider, "plan_images", None)
        ):
            raise TypeError(
                "image_plan_provider must be callable or expose plan_images"
            )
        self._transport = transport
        self._authorization = authorization
        self._image_plan_provider = provider
        self._limits = limits or HierarchicalTransportLimits()
        self._usage: dict[str, _TrajectoryUsage] = {}
        self._call_audits: list[dict[str, Any]] = []
        self._last_call_audit: dict[str, Any] | None = None

    @property
    def capabilities(self) -> Mapping[str, bool]:
        report = self._transport.capability_report
        return {
            "image_input_supported": bool(report and report.supports_images),
            "image_input_acknowledged": bool(
                report and report.image_roundtrip_verified
            ),
            "structured_output_supported": bool(
                report and report.structured_output_verified
            ),
            "text_only_fallback": bool(report and report.text_only_fallback_detected),
            "scripted_backend": False,
        }

    @property
    def last_call_audit(self) -> Mapping[str, Any] | None:
        return None if self._last_call_audit is None else dict(self._last_call_audit)

    @property
    def call_audits(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(value) for value in self._call_audits)

    def capability_preflight(self) -> CapabilityReport:
        """Run the transport's active image and structured-output challenge."""

        return self._transport.capability_preflight(authorization=self._authorization)

    def trajectory_usage(self, trajectory_id: str) -> Mapping[str, int]:
        value = self._usage.get(trajectory_id, _TrajectoryUsage())
        return {
            "model_calls": value.calls,
            "image_count": value.image_count,
            "image_bytes": value.image_bytes,
        }

    def _plan(
        self,
        *,
        layer: str,
        trajectory_id: str,
        segment_id: str | None,
        evidence_refs: Sequence[str],
        max_images: int | None = None,
        action_chunk_ids: Sequence[str] = (),
    ) -> HierarchicalImagePlan:
        query = HierarchicalImagePlanQuery(
            layer=layer,
            trajectory_id=trajectory_id,
            segment_id=segment_id,
            evidence_refs=tuple(dict.fromkeys(evidence_refs)),
            max_images=(
                self._limits.max_images_per_request
                if max_images is None
                else max_images
            ),
            max_total_bytes=self._limits.max_total_image_bytes_per_request,
            action_chunk_ids=tuple(dict.fromkeys(action_chunk_ids)),
        )
        method = getattr(self._image_plan_provider, "plan_images", None)
        raw = method(query) if callable(method) else self._image_plan_provider(query)
        if not isinstance(raw, HierarchicalImagePlan):
            raise TypeError("image plan provider must return HierarchicalImagePlan")
        requested_refs = set(query.evidence_refs)
        for value in raw.presentations:
            if not set(value.underlying_evidence_refs).issubset(requested_refs):
                raise HierarchicalImagePlanError(
                    "image plan includes evidence outside the layer request"
                )
        return raw

    def _reserve(
        self,
        *,
        trajectory_id: str,
        layer: str,
        scope_id: str,
        images: Sequence[MultimodalImage],
        aliases: _Aliases,
        bindings: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        usage = self._usage.setdefault(trajectory_id, _TrajectoryUsage())
        call_index = usage.calls + 1
        image_count = len(images)
        image_bytes = sum(len(value.content) for value in images)
        base = {
            "layer": layer,
            "scope_id": scope_id,
            "status": "budget_check",
            "model_call_index": call_index,
            "image_count": image_count,
            "image_total_bytes": image_bytes,
            "request_local_aliases": aliases.audit(),
            "presentation_bindings": [dict(value) for value in bindings],
            "no_retry": True,
        }
        if call_index > self._limits.max_model_calls_per_trajectory:
            self._record_local_failed(base, "trajectory_call_budget_exhausted")
            raise HierarchicalBudgetError(
                "hierarchical trajectory model-call budget exhausted"
            )
        if image_count > self._limits.max_images_per_request:
            self._record_local_failed(base, "request_image_count_budget_exhausted")
            raise HierarchicalBudgetError("request image-count budget exhausted")
        if any(len(value.content) > self._limits.max_image_bytes for value in images):
            self._record_local_failed(base, "per_image_byte_budget_exhausted")
            raise HierarchicalBudgetError("one presentation exceeds its byte budget")
        if image_bytes > self._limits.max_total_image_bytes_per_request:
            self._record_local_failed(base, "request_image_byte_budget_exhausted")
            raise HierarchicalBudgetError("request image-byte budget exhausted")
        if usage.image_count + image_count > self._limits.max_images_per_trajectory:
            self._record_local_failed(base, "trajectory_image_count_budget_exhausted")
            raise HierarchicalBudgetError("trajectory image-count budget exhausted")
        if (
            usage.image_bytes + image_bytes
            > self._limits.max_total_image_bytes_per_trajectory
        ):
            self._record_local_failed(base, "trajectory_image_byte_budget_exhausted")
            raise HierarchicalBudgetError("trajectory image-byte budget exhausted")
        # Reserve before I/O.  A failed request may already have crossed the
        # process boundary, so its images remain charged to the trajectory.
        usage.calls = call_index
        usage.image_count += image_count
        usage.image_bytes += image_bytes
        self._last_call_audit = base
        return base

    def _record_local_failed(
        self,
        base: Mapping[str, Any],
        reason: str,
        *,
        stage: str = "budget_check",
    ) -> None:
        audit = {
            **dict(base),
            "status": "local_failed",
            "delivery_status": "not_sent",
            "failure": reason,
            "failure_stage": stage,
            "images_sent": 0,
            "image_bytes_sent": 0,
        }
        self._last_call_audit = audit
        self._call_audits.append(audit)

    def _record_unhandled_local_failure(
        self,
        *,
        audit_count_before: int,
        trajectory_id: str,
        layer: str,
        scope_id: str,
        aliases: _Aliases,
        stage: str,
        failure: Exception,
        plan: HierarchicalImagePlan | None = None,
    ) -> None:
        if len(self._call_audits) != audit_count_before:
            return
        presentations = () if plan is None else plan.presentations
        base = {
            "layer": layer,
            "scope_id": scope_id,
            "model_call_index": self._usage.get(trajectory_id, _TrajectoryUsage()).calls
            + 1,
            "image_count": len(presentations),
            "image_total_bytes": sum(len(value.content) for value in presentations),
            "request_local_aliases": aliases.audit(),
            "presentation_bindings": (
                [] if plan is None else _local_presentation_bindings(plan)
            ),
            "no_retry": True,
        }
        self._record_local_failed(
            base,
            type(failure).__name__,
            stage=stage,
        )

    def _complete(
        self,
        *,
        trajectory_id: str,
        layer: str,
        scope_id: str,
        prompt: str,
        projection: Mapping[str, Any],
        aliases: _Aliases,
        output_schema: Mapping[str, Any],
        schema_name: str,
        plan: HierarchicalImagePlan | None,
    ) -> LayerModelOutput:
        audit_count_before = len(self._call_audits)
        images: tuple[MultimodalImage, ...] = ()
        model_bindings: list[dict[str, Any]] = []
        audit_bindings: list[dict[str, Any]] = []
        try:
            outbound = dict(projection)
            if plan is not None:
                images, model_bindings = _presentation_projection(plan, aliases)
                audit_bindings = _local_presentation_bindings(plan)
                outbound["model_image_bindings"] = model_bindings
                outbound["image_citation_rule"] = (
                    "model images are presentation only; cite bound original "
                    "evidence refs"
                )
            encoded = _canonical_bytes(outbound).decode("utf-8")
            _validate_outbound_projection(encoded, aliases)
        except Exception as exc:
            self._record_unhandled_local_failure(
                audit_count_before=audit_count_before,
                trajectory_id=trajectory_id,
                layer=layer,
                scope_id=scope_id,
                aliases=aliases,
                stage="projection",
                failure=exc,
                plan=plan,
            )
            raise
        base = self._reserve(
            trajectory_id=trajectory_id,
            layer=layer,
            scope_id=scope_id,
            images=images,
            aliases=aliases,
            bindings=audit_bindings,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        try:
            if images:
                completion = self._transport.complete(
                    instructions=prompt,
                    input_text=encoded,
                    images=images,
                    output_schema=output_schema,
                    schema_name=schema_name,
                    authorization=self._authorization,
                )
                transport_audit = self._transport.last_multimodal_completion_audit
            else:
                completion = self._transport.complete_text(
                    instructions=prompt,
                    input_text=encoded,
                    output_schema=output_schema,
                    schema_name=schema_name,
                    authorization=self._authorization,
                    purpose=layer,
                )
                transport_audit = self._transport.last_text_completion_audit
        except Exception as exc:
            transport_audit = (
                self._transport.last_multimodal_completion_audit
                if images
                else self._transport.last_text_completion_audit
            )
            transport_delivery = (
                transport_audit.get("delivery_status")
                if isinstance(transport_audit, Mapping)
                else None
            )
            delivery_status = (
                str(transport_delivery)
                if transport_delivery in {"not_sent", "sent", "sent_or_attempted"}
                else "sent_or_attempted"
            )
            images_sent = len(images) if delivery_status != "not_sent" else 0
            image_bytes_sent = (
                sum(len(value.content) for value in images)
                if delivery_status != "not_sent"
                else 0
            )
            audit = {
                **base,
                **({} if transport_audit is None else dict(transport_audit)),
                "layer": layer,
                "status": "failed",
                "delivery_status": delivery_status,
                "failure": type(exc).__name__,
                "image_count": len(images),
                "image_total_bytes": sum(len(value.content) for value in images),
                "request_local_aliases": aliases.audit(),
                "presentation_bindings": audit_bindings,
                "images_sent": images_sent,
                "image_bytes_sent": image_bytes_sent,
                "no_retry": True,
            }
            self._last_call_audit = audit
            self._call_audits.append(audit)
            raise
        provider_parsed_json = json.loads(_canonical_bytes(completion.output))
        provider_canonical_json_sha256 = hashlib.sha256(
            _canonical_bytes(provider_parsed_json)
        ).hexdigest()
        restored = _restore_output_refs(completion.output, aliases)
        layer_output = LayerModelOutput.from_raw(
            restored,
            backend=self.backend_name,
            prompt_template_hash=prompt_hash,
        )
        audit = {
            **base,
            **dict(completion.audit),
            "layer": layer,
            "status": "completed",
            "delivery_status": "sent",
            "image_count": len(images),
            "image_total_bytes": sum(len(value.content) for value in images),
            "request_local_aliases": aliases.audit(),
            "presentation_bindings": audit_bindings,
            "images_sent": len(images),
            "image_bytes_sent": sum(len(value.content) for value in images),
            "provider_parsed_json": provider_parsed_json,
            "provider_canonical_json_sha256": provider_canonical_json_sha256,
            "restored_output_sha256": layer_output.output_sha256,
            "reference_restoration_applied": True,
            "provider_references_are_request_local_aliases": True,
            "no_retry": True,
        }
        self._last_call_audit = audit
        self._call_audits.append(audit)
        return layer_output

    @staticmethod
    def _action_evidence_refs(
        request: ActionChunkBatchReflectionRequest,
    ) -> tuple[str, ...]:
        return tuple(value.evidence_ref for value in request.segment_visual_evidence)

    def reflect_action_chunks(
        self,
        request: ActionChunkBatchReflectionRequest,
    ) -> LayerModelOutput:
        if not isinstance(request, ActionChunkBatchReflectionRequest):
            raise TypeError("request must be ActionChunkBatchReflectionRequest")
        self._last_call_audit = None
        audit_count_before = len(self._call_audits)
        aliases = _Aliases()
        plan: HierarchicalImagePlan | None = None
        stage = "image_plan"
        try:
            _register_chunk_aliases(request, aliases)
            refs = self._action_evidence_refs(request)
            plan = self._plan(
                layer=_LAYER_ACTION,
                trajectory_id=request.trajectory_id,
                segment_id=request.segment_id,
                evidence_refs=refs,
                action_chunk_ids=tuple(
                    value.action_chunk_id for value in request.chunks
                ),
            )
            if not plan.presentations:
                raise HierarchicalImagePlanError(
                    "Layer 1 requires action-chunk visual presentations"
                )
            stage = "projection"
            projection = _action_projection(request, aliases)
            stage = "transport"
            return self._complete(
                trajectory_id=request.trajectory_id,
                layer=_LAYER_ACTION,
                scope_id=request.segment_id,
                prompt=generic_action_chunk_system_prompt(),
                projection=projection,
                aliases=aliases,
                output_schema=action_chunk_batch_output_json_schema(),
                schema_name="phase_a21_action_chunk_batch",
                plan=plan,
            )
        except Exception as exc:
            self._record_unhandled_local_failure(
                audit_count_before=audit_count_before,
                trajectory_id=request.trajectory_id,
                layer=_LAYER_ACTION,
                scope_id=request.segment_id,
                aliases=aliases,
                stage=stage,
                failure=exc,
                plan=plan,
            )
            raise

    def summarize_subtask(self, request: SubtaskSummaryRequest) -> LayerModelOutput:
        if not isinstance(request, SubtaskSummaryRequest):
            raise TypeError("request must be SubtaskSummaryRequest")
        self._last_call_audit = None
        audit_count_before = len(self._call_audits)
        aliases = _Aliases()
        plan: HierarchicalImagePlan | None = None
        stage = "projection"
        try:
            projection = _subtask_projection(request, aliases)
            if self._limits.layer2_verification_images:
                stage = "image_plan"
                refs = tuple(
                    value.evidence_ref for value in request.verification_visual_evidence
                )
                candidate = self._plan(
                    layer=_LAYER_SUBTASK,
                    trajectory_id=request.trajectory_id,
                    segment_id=request.segment_id,
                    evidence_refs=refs,
                    max_images=self._limits.layer2_verification_images,
                )
                if candidate.presentations:
                    plan = candidate
            stage = "transport"
            return self._complete(
                trajectory_id=request.trajectory_id,
                layer=_LAYER_SUBTASK,
                scope_id=request.segment_id,
                prompt=generic_subtask_summary_system_prompt(),
                projection=projection,
                aliases=aliases,
                output_schema=subtask_summary_output_json_schema(),
                schema_name="phase_a21_subtask_summary",
                plan=plan,
            )
        except Exception as exc:
            self._record_unhandled_local_failure(
                audit_count_before=audit_count_before,
                trajectory_id=request.trajectory_id,
                layer=_LAYER_SUBTASK,
                scope_id=request.segment_id,
                aliases=aliases,
                stage=stage,
                failure=exc,
                plan=plan,
            )
            raise

    def draft_procedure(self, request: ProcedureDraftRequest) -> LayerModelOutput:
        if not isinstance(request, ProcedureDraftRequest):
            raise TypeError("request must be ProcedureDraftRequest")
        self._last_call_audit = None
        audit_count_before = len(self._call_audits)
        aliases = _Aliases()
        plan: HierarchicalImagePlan | None = None
        stage = "projection"
        try:
            projection = _draft_projection(request, aliases)
            refs = tuple(
                value.evidence_ref for value in request.cross_segment_visual_evidence
            )
            stage = "image_plan"
            plan = self._plan(
                layer=_LAYER_DRAFT,
                trajectory_id=request.trajectory_id,
                segment_id=None,
                evidence_refs=refs,
            )
            if not plan.presentations:
                raise HierarchicalImagePlanError(
                    "Layer 3 requires cross-segment feedback presentations"
                )
            stage = "transport"
            return self._complete(
                trajectory_id=request.trajectory_id,
                layer=_LAYER_DRAFT,
                scope_id=request.trajectory_id,
                prompt=generic_procedure_draft_system_prompt(),
                projection=projection,
                aliases=aliases,
                output_schema=procedure_draft_output_json_schema(),
                schema_name="phase_a21_procedure_draft",
                plan=plan,
            )
        except Exception as exc:
            self._record_unhandled_local_failure(
                audit_count_before=audit_count_before,
                trajectory_id=request.trajectory_id,
                layer=_LAYER_DRAFT,
                scope_id=request.trajectory_id,
                aliases=aliases,
                stage=stage,
                failure=exc,
                plan=plan,
            )
            raise

    def attribute_procedure(self, request: AttributionPassRequest) -> LayerModelOutput:
        if not isinstance(request, AttributionPassRequest):
            raise TypeError("request must be AttributionPassRequest")
        self._last_call_audit = None
        audit_count_before = len(self._call_audits)
        aliases = _Aliases()
        stage = "projection"
        try:
            projection = _attribution_projection(request, aliases)
            stage = "transport"
            return self._complete(
                trajectory_id=request.trajectory_id,
                layer=_LAYER_ATTRIBUTION,
                scope_id=request.trajectory_id,
                prompt=generic_attribution_system_prompt(),
                projection=projection,
                aliases=aliases,
                output_schema=attribution_output_json_schema(),
                schema_name="phase_a21_evidence_attribution",
                plan=None,
            )
        except Exception as exc:
            self._record_unhandled_local_failure(
                audit_count_before=audit_count_before,
                trajectory_id=request.trajectory_id,
                layer=_LAYER_ATTRIBUTION,
                scope_id=request.trajectory_id,
                aliases=aliases,
                stage=stage,
                failure=exc,
                plan=None,
            )
            raise


__all__ = [
    "HierarchicalBudgetError",
    "HierarchicalImagePlan",
    "HierarchicalImagePlanError",
    "HierarchicalImagePlanProvider",
    "HierarchicalImagePlanQuery",
    "HierarchicalTransportError",
    "HierarchicalTransportLimits",
    "OpenAICompatibleHierarchicalBackend",
    "PlannedPresentation",
    "VisualPresentationImagePlanProvider",
    "generic_action_chunk_system_prompt",
    "generic_attribution_system_prompt",
    "generic_procedure_draft_system_prompt",
    "generic_subtask_summary_system_prompt",
]

"""Bind action chunks, original visual refs, and presentation request plans."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, overload

from .hierarchical import ActionChunkReflectionRequest, EvidenceInput
from .schemas import SchemaValidationError
from .visual_presentations import VisualPresentationResult
from .whole_trajectory import (
    WholeTrajectoryReflectionRequest,
    annotation_evidence_ref,
)


@dataclass(frozen=True, slots=True)
class HierarchicalPreparedInputs:
    """Provider-neutral requests plus the exact audited image plans."""

    action_chunk_requests: tuple[ActionChunkReflectionRequest, ...]
    segment_visual_evidence_by_segment: Mapping[str, "Layer1SegmentVisualRegistry"]
    cross_segment_visual_evidence: tuple[EvidenceInput, ...]
    layer1_request_plans: tuple[Mapping[str, Any], ...]
    cross_segment_feedback_requests: tuple[Mapping[str, Any], ...]

    @property
    def planned_hierarchical_model_calls(self) -> int:
        """Layer-1 batches + one Layer-2/subtask + Layer-3 + Layer-4."""

        return (
            len(self.layer1_request_plans)
            + len(self.segment_visual_evidence_by_segment)
            + 2
        )

    @property
    def planned_external_calls_with_preflight(self) -> int:
        """Total service calls when the active capability preflight is enabled."""

        return self.planned_hierarchical_model_calls + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_chunk_requests": [
                value.to_dict() for value in self.action_chunk_requests
            ],
            "segment_visual_evidence_by_segment": {
                segment_id: [value.to_dict() for value in values]
                for segment_id, values in self.segment_visual_evidence_by_segment.items()
            },
            "layer1_batch_bindings_by_segment": {
                segment_id: [deepcopy(dict(value)) for value in values.batch_bindings]
                for segment_id, values in self.segment_visual_evidence_by_segment.items()
            },
            "cross_segment_visual_evidence": [
                value.to_dict() for value in self.cross_segment_visual_evidence
            ],
            "layer1_request_plans": [
                deepcopy(dict(value)) for value in self.layer1_request_plans
            ],
            "cross_segment_feedback_requests": [
                deepcopy(dict(value)) for value in self.cross_segment_feedback_requests
            ],
            "planned_hierarchical_model_calls": self.planned_hierarchical_model_calls,
            "planned_external_calls_with_preflight": (
                self.planned_external_calls_with_preflight
            ),
        }


@dataclass(frozen=True, slots=True)
class Layer1SegmentVisualRegistry(Sequence[EvidenceInput]):
    """Flattened segment registry plus its exact chunk-batch partitions.

    It remains a normal ``Sequence[EvidenceInput]`` for existing callers.  The
    reflector consumes ``batch_bindings`` to issue one Layer-1 call per visual
    request plan while retaining one Layer-2 summary for the whole subtask.
    """

    evidence: tuple[EvidenceInput, ...]
    batch_bindings: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        if not evidence or any(
            not isinstance(value, EvidenceInput) for value in evidence
        ):
            raise SchemaValidationError(
                "layer1_segment_visual_registry.evidence: invalid evidence"
            )
        evidence_refs = {value.evidence_ref for value in evidence}
        if len(evidence_refs) != len(evidence):
            raise SchemaValidationError(
                "layer1_segment_visual_registry.evidence: duplicate evidence ref"
            )
        bindings: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()
        seen_requests: set[str] = set()
        for index, raw in enumerate(self.batch_bindings):
            if not isinstance(raw, Mapping):
                raise SchemaValidationError(
                    f"layer1_segment_visual_registry.batch_bindings[{index}]: invalid binding"
                )
            value = deepcopy(dict(raw))
            if set(value) != {
                "request_id",
                "segment_id",
                "action_chunk_ids",
                "evidence_refs",
            }:
                raise SchemaValidationError(
                    "layer1_segment_visual_registry.batch_bindings: invalid fields"
                )
            chunk_ids = value["action_chunk_ids"]
            refs = value["evidence_refs"]
            if (
                not isinstance(value["request_id"], str)
                or not value["request_id"]
                or value["request_id"] in seen_requests
                or not isinstance(value["segment_id"], str)
                or not value["segment_id"]
                or not isinstance(chunk_ids, list)
                or not chunk_ids
                or any(not isinstance(item, str) or not item for item in chunk_ids)
                or len(set(chunk_ids)) != len(chunk_ids)
                or not isinstance(refs, list)
                or not refs
                or any(not isinstance(item, str) or not item for item in refs)
                or len(set(refs)) != len(refs)
            ):
                raise SchemaValidationError(
                    "layer1_segment_visual_registry.batch_bindings: malformed binding"
                )
            if seen_chunks.intersection(chunk_ids):
                raise SchemaValidationError(
                    "layer1_segment_visual_registry.batch_bindings: chunk repeated across batches"
                )
            seen_chunks.update(chunk_ids)
            seen_requests.add(value["request_id"])
            if not set(refs).issubset(evidence_refs):
                raise SchemaValidationError(
                    "layer1_segment_visual_registry.batch_bindings: unknown evidence ref"
                )
            bindings.append(value)
        if not bindings:
            raise SchemaValidationError(
                "layer1_segment_visual_registry.batch_bindings: empty"
            )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "batch_bindings", tuple(bindings))

    @overload
    def __getitem__(self, index: int) -> EvidenceInput: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[EvidenceInput, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> EvidenceInput | tuple[EvidenceInput, ...]:
        return self.evidence[index]

    def __len__(self) -> int:
        return len(self.evidence)

    def __iter__(self) -> Iterator[EvidenceInput]:
        return iter(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [value.to_dict() for value in self.evidence],
            "batch_bindings": [deepcopy(dict(value)) for value in self.batch_bindings],
        }


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{label}: must be an object or expose to_dict()")
    return deepcopy(dict(value))


def _visual_item_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    frame = item.get("frame")
    frame_payload = (
        {key: frame[key] for key in ("index", "evidence_ref") if key in frame}
        if isinstance(frame, Mapping)
        else {}
    )
    return {
        "source_type": "observed_visual",
        "segment_id": item.get("segment_id"),
        "segment_index": item.get("segment_index"),
        "frame": frame_payload,
        "camera": item.get("camera"),
        "temporal_role": item.get("temporal_role"),
        "confidence": item.get("confidence"),
        "limitations": [
            "rgb_does_not_establish_physical_contact_or_causality",
            "derived_presentations_are_not_independent_physical_evidence",
        ],
    }


def _robot_item_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    frame = item.get("frame")
    frame_payload = (
        {key: frame[key] for key in ("index", "evidence_ref") if key in frame}
        if isinstance(frame, Mapping)
        else {}
    )
    action = item.get("action_range")
    action_payload = (
        {
            key: action[key]
            for key in (
                "source_type",
                "start_inclusive",
                "end_inclusive",
                "semantics",
                "sample_frame_index",
            )
            if key in action
        }
        if isinstance(action, Mapping)
        else {}
    )
    grippers: list[dict[str, Any]] = []
    raw_grippers = item.get("gripper_states")
    if isinstance(raw_grippers, Sequence) and not isinstance(
        raw_grippers, (str, bytes)
    ):
        for raw in raw_grippers:
            if not isinstance(raw, Mapping):
                continue
            grippers.append(
                {
                    key: raw[key]
                    for key in (
                        "channel_id",
                        "source_type",
                        "state",
                        "confidence",
                    )
                    if key in raw
                }
            )
    return {
        "source_type": "observed_robot_state",
        "segment_id": item.get("segment_id"),
        "segment_index": item.get("segment_index"),
        "frame": frame_payload,
        "action_range": action_payload,
        "gripper_states": grippers,
        "semantics": "observed_robot_state_not_commanded_action",
    }


def build_hierarchical_inputs(
    *,
    whole_request: WholeTrajectoryReflectionRequest,
    action_chunk_bundle: Mapping[str, Any] | Any,
    visual_evidence_bundle: Mapping[str, Any] | Any,
    presentation_result: VisualPresentationResult,
) -> HierarchicalPreparedInputs:
    """Create requests only from original refs actually bound to model images."""

    if not isinstance(whole_request, WholeTrajectoryReflectionRequest):
        raise TypeError("whole_request must be WholeTrajectoryReflectionRequest")
    if not isinstance(presentation_result, VisualPresentationResult):
        raise TypeError("presentation_result must be VisualPresentationResult")
    if not presentation_result.admissible:
        raise SchemaValidationError("visual presentation coverage is not admissible")
    action = _mapping(action_chunk_bundle, label="action_chunk_bundle")
    visual = _mapping(visual_evidence_bundle, label="visual_evidence_bundle")
    if (
        action.get("trajectory_id") != whole_request.trajectory_id
        or visual.get("trajectory_id") != whole_request.trajectory_id
    ):
        raise SchemaValidationError("hierarchical inputs cross trajectory scope")

    segment_ids = [str(value["segment_id"]) for value in whole_request.segments]
    plans_by_segment: dict[str, list[dict[str, Any]]] = {
        segment_id: [] for segment_id in segment_ids
    }
    for index, raw in enumerate(presentation_result.layer1_request_plans):
        if not isinstance(raw, Mapping):
            raise SchemaValidationError(
                f"layer1_request_plans[{index}]: must be an object"
            )
        plan = dict(raw)
        segment_id = plan.get("segment_id")
        if not isinstance(segment_id, str) or segment_id not in segment_ids:
            raise SchemaValidationError(
                f"layer1_request_plans[{index}]: unknown segment"
            )
        plans_by_segment[segment_id].append(plan)
    if any(not plans_by_segment[segment_id] for segment_id in segment_ids):
        raise SchemaValidationError("Layer-1 image plans do not cover every subtask")
    for segment_id, plans in plans_by_segment.items():
        ordered = sorted(plans, key=lambda value: int(value.get("part_index", 0)))
        expected_count = len(ordered)
        if any(
            plan.get("part_index", index) != index
            or plan.get("part_count", expected_count) != expected_count
            or bool(plan.get("is_split_request", expected_count > 1))
            != (expected_count > 1)
            for index, plan in enumerate(ordered)
        ):
            raise SchemaValidationError(
                f"Layer-1 plans for {segment_id!r} have inconsistent split metadata"
            )
        plans_by_segment[segment_id] = ordered

    raw_items = visual.get("items")
    if not isinstance(raw_items, list):
        raise SchemaValidationError("visual evidence bundle has no item array")
    item_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            raise SchemaValidationError(
                f"visual_evidence.items[{index}]: must be an object"
            )
        evidence_id = raw.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise SchemaValidationError(
                f"visual_evidence.items[{index}]: invalid evidence ID"
            )
        if evidence_id in item_by_id:
            raise SchemaValidationError("visual evidence ID is duplicated")
        item_by_id[evidence_id] = dict(raw)

    refs_by_request: dict[str, tuple[str, ...]] = {}
    plan_by_chunk: dict[str, dict[str, Any]] = {}
    for segment_id, plans in plans_by_segment.items():
        for plan in plans:
            request_id = plan.get("request_id")
            refs = plan.get("underlying_evidence_refs")
            chunk_ids = plan.get("action_chunk_ids")
            if (
                not isinstance(request_id, str)
                or not request_id
                or request_id in refs_by_request
                or not isinstance(refs, list)
                or not refs
                or not isinstance(chunk_ids, list)
                or not chunk_ids
            ):
                raise SchemaValidationError(
                    f"Layer-1 plan for {segment_id!r} is missing batch bindings"
                )
            ordered_refs = tuple(dict.fromkeys(str(value) for value in refs))
            if len(ordered_refs) != len(refs) or not set(ordered_refs).issubset(
                item_by_id
            ):
                raise SchemaValidationError(
                    "Layer-1 plan cites duplicate or unknown visual evidence"
                )
            if any(
                item_by_id[evidence_id].get("segment_id") != segment_id
                for evidence_id in ordered_refs
            ):
                raise SchemaValidationError(
                    "Layer-1 plan cites visual evidence from another segment"
                )
            normalized_chunk_ids = [str(value) for value in chunk_ids]
            if any(not value for value in normalized_chunk_ids) or len(
                set(normalized_chunk_ids)
            ) != len(normalized_chunk_ids):
                raise SchemaValidationError(
                    "Layer-1 plan has invalid action chunk bindings"
                )
            for chunk_id in normalized_chunk_ids:
                if chunk_id in plan_by_chunk:
                    raise SchemaValidationError(
                        "an action chunk is repeated across Layer-1 batches"
                    )
                plan_by_chunk[chunk_id] = plan
            refs_by_request[request_id] = ordered_refs

    segments = {str(value["segment_id"]): value for value in whole_request.segments}
    raw_chunks = action.get("action_chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise SchemaValidationError("action chunk bundle has no chunks")
    action_requests: list[ActionChunkReflectionRequest] = []
    chunk_ids_by_segment: dict[str, list[str]] = {
        segment_id: [] for segment_id in segment_ids
    }
    for index, raw in enumerate(raw_chunks):
        if not isinstance(raw, Mapping):
            raise SchemaValidationError(f"action_chunks[{index}]: must be an object")
        chunk = deepcopy(dict(raw))
        segment_id = chunk.get("segment_id")
        if not isinstance(segment_id, str) or segment_id not in segments:
            raise SchemaValidationError(f"action_chunks[{index}]: unknown segment")
        chunk_id = chunk.get("action_chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise SchemaValidationError(f"action_chunks[{index}]: invalid chunk ID")
        plan = plan_by_chunk.get(chunk_id)
        if plan is None or plan.get("segment_id") != segment_id:
            raise SchemaValidationError(
                f"action_chunks[{index}]: not bound to exactly one Layer-1 batch"
            )
        chunk_ids_by_segment[segment_id].append(chunk_id)
        request_id = str(plan["request_id"])
        allowed_for_chunk_batch = set(refs_by_request[request_id])
        role_refs = chunk.get("evidence_frame_refs")
        if not isinstance(role_refs, Mapping):
            raise SchemaValidationError(
                f"action_chunks[{index}]: evidence roles are missing"
            )
        expected_frame_refs = {
            str(frame_ref)
            for values in role_refs.values()
            if isinstance(values, list)
            for frame_ref in values
        }
        selected_items = [
            item
            for evidence_id, item in item_by_id.items()
            if evidence_id in allowed_for_chunk_batch
            and item.get("segment_id") == segment_id
            and isinstance(item.get("frame"), Mapping)
            and item["frame"].get("evidence_ref") in expected_frame_refs
        ]
        covered_frames = {str(item["frame"]["evidence_ref"]) for item in selected_items}
        if covered_frames != expected_frame_refs:
            raise SchemaValidationError(
                f"action_chunks[{index}]: image plan does not cover every declared frame role"
            )
        visual_inputs = tuple(
            EvidenceInput(
                evidence_ref=str(item["evidence_id"]),
                payload=_visual_item_payload(item),
            )
            for item in selected_items
        )
        robot_inputs = tuple(
            EvidenceInput(
                evidence_ref=str(item["evidence_id"]),
                payload=_robot_item_payload(item),
            )
            for item in selected_items
        )
        segment = segments[segment_id]
        action_requests.append(
            ActionChunkReflectionRequest(
                trajectory_id=whole_request.trajectory_id,
                segment_id=segment_id,
                subtask_annotation={
                    "evidence_ref": annotation_evidence_ref(segment_id),
                    "text": str(segment["context"]["subtask_instruction"]),
                },
                action_chunk=chunk,
                observed_robot_state=robot_inputs,
                visual_evidence=visual_inputs,
                trajectory_outcome=whole_request.trajectory_outcome,
                provenance=whole_request.provenance,
            )
        )

    if set(plan_by_chunk) != {
        chunk_id for values in chunk_ids_by_segment.values() for chunk_id in values
    }:
        raise SchemaValidationError("Layer-1 plans cite an unknown action chunk")

    segment_visual_inputs: dict[str, Layer1SegmentVisualRegistry] = {}
    for segment_id in segment_ids:
        plans = plans_by_segment[segment_id]
        ordered_refs = tuple(
            dict.fromkeys(
                evidence_ref
                for plan in plans
                for evidence_ref in refs_by_request[str(plan["request_id"])]
            )
        )
        bindings = tuple(
            {
                "request_id": str(plan["request_id"]),
                "segment_id": segment_id,
                "action_chunk_ids": [str(value) for value in plan["action_chunk_ids"]],
                "evidence_refs": list(refs_by_request[str(plan["request_id"])]),
            }
            for plan in plans
        )
        segment_visual_inputs[segment_id] = Layer1SegmentVisualRegistry(
            evidence=tuple(
                EvidenceInput(
                    evidence_ref=evidence_id,
                    payload=_visual_item_payload(item_by_id[evidence_id]),
                )
                for evidence_id in ordered_refs
            ),
            batch_bindings=bindings,
        )

    cross_plan = presentation_result.cross_segment_feedback_plan
    requests = cross_plan.get("requests") if isinstance(cross_plan, Mapping) else None
    if not isinstance(requests, list) or len(requests) != 1:
        raise SchemaValidationError(
            "cross-segment feedback must fit one lossless Layer-3 image request; "
            "required evidence is never silently merged or dropped"
        )
    cross_refs = requests[0].get("underlying_evidence_refs")
    if not isinstance(cross_refs, list) or not cross_refs:
        raise SchemaValidationError("cross-segment feedback has no underlying refs")
    cross_ids = {str(value) for value in cross_refs}
    if not cross_ids.issubset(item_by_id):
        raise SchemaValidationError("cross-segment plan cites unknown evidence")
    cross_inputs = tuple(
        EvidenceInput(
            evidence_ref=evidence_id,
            payload=_visual_item_payload(item_by_id[evidence_id]),
        )
        for evidence_id in sorted(cross_ids)
    )
    if len({str(item.payload.get("segment_id")) for item in cross_inputs}) < 2:
        raise SchemaValidationError(
            "cross-segment feedback evidence must span multiple subtasks"
        )
    return HierarchicalPreparedInputs(
        action_chunk_requests=tuple(action_requests),
        segment_visual_evidence_by_segment=segment_visual_inputs,
        cross_segment_visual_evidence=cross_inputs,
        layer1_request_plans=tuple(
            deepcopy(dict(value)) for value in presentation_result.layer1_request_plans
        ),
        cross_segment_feedback_requests=tuple(
            deepcopy(dict(value)) for value in requests
        ),
    )


__all__ = [
    "HierarchicalPreparedInputs",
    "Layer1SegmentVisualRegistry",
    "build_hierarchical_inputs",
]

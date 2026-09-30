"""Task-neutral visual presentation planning for hierarchical reflection.

This module is deliberately model-free.  It turns an ActionChunkBundle and a
VisualEvidenceBundle into bounded image presentations and request plans.  A
presentation is never a new physical fact: every source, crop, comparison, and
difference image resolves to the original visual evidence references.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
import math
import re
from typing import Any, Protocol

import numpy as np
from PIL import Image


_ROLE_PRIORITY = {
    "subtask_before": 0,
    "subtask_after": 0,
    "action_chunk_before": 0,
    "action_chunk_after": 0,
    "gripper_transition_before": 1,
    "gripper_transition_after": 1,
    "action_chunk_during": 1,
    "release_after_settle": 2,
    "action_chunk_settled": 2,
    "task_feedback": 2,
    "context": 5,
}
_REQUIRED_ROLES = frozenset(_ROLE_PRIORITY) - {"context"}
_ROLE_TO_TEMPORAL_ROLE = {
    "subtask_before": "before",
    "subtask_after": "after",
    "action_chunk_before": "before",
    "action_chunk_after": "after",
    "gripper_transition_before": "before",
    "gripper_transition_after": "after",
    "action_chunk_during": "during",
    "release_after_settle": "settled",
    "action_chunk_settled": "settled",
    "task_feedback": "task_feedback",
    "context": "context",
}
_FRAME_REF_RE = re.compile(r"#frame/(?P<index>[0-9]+)$")
_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class VisualPresentationError(ValueError):
    """Invalid evidence, payload, or presentation configuration."""


class VisualPayloadResolver(Protocol):
    """Resolve a bundle ``image_ref_id`` without exposing filesystem paths."""

    def __call__(self, image_ref_id: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class VisualPresentationLimits:
    """Configurable outbound limits; no fixed eight-image assumption exists."""

    max_images_per_request: int = 16
    max_unique_source_frames_per_trajectory: int = 40
    max_total_image_bytes_per_request: int = 8 * 1024 * 1024
    max_total_image_bytes_per_trajectory: int = 8 * 1024 * 1024
    max_model_calls_per_trajectory: int = 8
    max_pixels_per_image: int = 1_500_000
    max_total_pixels_per_request: int = 8_000_000
    max_total_pixels_per_trajectory: int = 32_000_000
    max_derived_presentations_per_source_frame: int = 12
    max_change_rois_per_pair: int = 3
    roi_tile_size: int = 16
    roi_padding_pixels: int = 8
    roi_min_output_side: int = 96
    roi_min_normalized_change: float = 0.035
    roi_mad_multiplier: float = 4.0
    secondary_view_min_gain: float = 0.08
    secondary_change_relative_gain_min: float = 0.15
    secondary_change_retention_ratio_min: float = 0.8
    secondary_component_gain_weight: float = 0.1

    def __post_init__(self) -> None:
        integer_fields = (
            "max_images_per_request",
            "max_unique_source_frames_per_trajectory",
            "max_total_image_bytes_per_request",
            "max_total_image_bytes_per_trajectory",
            "max_model_calls_per_trajectory",
            "max_pixels_per_image",
            "max_total_pixels_per_request",
            "max_total_pixels_per_trajectory",
            "max_derived_presentations_per_source_frame",
            "max_change_rois_per_pair",
            "roi_tile_size",
            "roi_min_output_side",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VisualPresentationError(f"{name} must be a positive integer")
        if (
            isinstance(self.roi_padding_pixels, bool)
            or not isinstance(self.roi_padding_pixels, int)
            or self.roi_padding_pixels < 0
        ):
            raise VisualPresentationError("roi_padding_pixels must be non-negative")
        for name in (
            "roi_min_normalized_change",
            "roi_mad_multiplier",
            "secondary_view_min_gain",
            "secondary_change_relative_gain_min",
            "secondary_change_retention_ratio_min",
            "secondary_component_gain_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise VisualPresentationError(f"{name} must be finite")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise VisualPresentationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PresentationPayload:
    """Outbound bytes kept separate from serializable audit records."""

    presentation_id: str
    media_type: str
    data: bytes = field(repr=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True, slots=True)
class VisualPresentationResult:
    """Serializable plans plus payloads; requests are empty when fail-closed."""

    presentation_records: tuple[Mapping[str, Any], ...]
    derived_visual_presentations: tuple[Mapping[str, Any], ...]
    outbound_payloads: tuple[PresentationPayload, ...]
    layer1_request_plans: tuple[Mapping[str, Any], ...]
    cross_segment_feedback_plan: Mapping[str, Any]
    coverage_audit: Mapping[str, Any]
    admissible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "roboharn_evo/visual_presentation_result/v1",
            "schema_version": 1,
            "admissible": self.admissible,
            "presentation_records": [
                dict(value) for value in self.presentation_records
            ],
            "derived_visual_presentations": [
                dict(value) for value in self.derived_visual_presentations
            ],
            "layer1_request_plans": [
                dict(value) for value in self.layer1_request_plans
            ],
            "cross_segment_feedback_plan": dict(self.cross_segment_feedback_plan),
            "coverage_audit": dict(self.coverage_audit),
        }


@dataclass(slots=True)
class _Source:
    evidence_id: str
    segment_id: str
    frame_index: int
    frame_ref: str
    camera_id: str
    temporal_role: str
    image_ref_id: str
    media_type: str
    declared_sha256: str
    data: bytes = field(repr=False)
    image: Image.Image = field(repr=False)
    selection_reasons: tuple[str, ...] = ()
    explicit_gain: float | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def pixels(self) -> int:
        return self.image.width * self.image.height


@dataclass(slots=True)
class _Asset:
    presentation_id: str
    kind: str
    media_type: str
    data: bytes = field(repr=False)
    width: int
    height: int
    underlying_refs: list[str]
    logical_bindings: list[dict[str, Any]]
    segment_ids: set[str]
    source_hashes: list[str]
    priority: int
    required: bool
    provenance: dict[str, Any]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def record(self) -> dict[str, Any]:
        return {
            "schema": "roboharn_evo/derived_visual_presentation/v1",
            "schema_version": 1,
            "presentation_id": self.presentation_id,
            "presentation_kind": self.kind,
            "is_derived_presentation": self.kind != "source_frame",
            "independent_physical_evidence": False,
            "underlying_evidence_refs": sorted(set(self.underlying_refs)),
            "logical_bindings": sorted(
                self.logical_bindings,
                key=lambda value: (
                    str(value.get("segment_id", "")),
                    str(value.get("action_chunk_id", "")),
                    str(value.get("role", "")),
                    str(value.get("evidence_id", "")),
                ),
            ),
            "segments": sorted(self.segment_ids),
            "content": {
                "sha256": self.sha256,
                "byte_length": len(self.data),
                "media_type": self.media_type,
                "width": self.width,
                "height": self.height,
            },
            "source_content_sha256": sorted(set(self.source_hashes)),
            "presentation_provenance": self.provenance,
        }


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise VisualPresentationError(f"{label} must be a mapping or expose to_dict()")
    return dict(value)


def _frame_index_from_ref(value: str) -> int:
    match = _FRAME_REF_RE.search(value)
    if match is None:
        raise VisualPresentationError(f"invalid frame reference: {value!r}")
    return int(match.group("index"))


def _resolved_bytes(resolver: VisualPayloadResolver, image_ref_id: str) -> bytes:
    value = resolver(image_ref_id)
    if isinstance(value, bytes):
        return value
    data = getattr(value, "data", None)
    if isinstance(data, bytes):
        return data
    raise VisualPresentationError(
        "payload resolver must return bytes or an object with bytes .data"
    )


def _load_sources(
    visual_bundle: Mapping[str, Any], resolver: VisualPayloadResolver
) -> tuple[list[_Source], list[str]]:
    raw_items = visual_bundle.get("items", [])
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise VisualPresentationError("visual evidence bundle items must be an array")
    sources: list[_Source] = []
    camera_order: list[str] = []
    seen_ids: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise VisualPresentationError("visual evidence item must be a mapping")
        evidence_id = raw.get("evidence_id")
        segment_id = raw.get("segment_id")
        camera_id = raw.get("camera", raw.get("camera_id"))
        frame = raw.get("frame", {})
        image_meta = raw.get("image", {})
        if not all(
            isinstance(value, str) and value
            for value in (evidence_id, segment_id, camera_id)
        ):
            raise VisualPresentationError(
                "visual evidence item has invalid id/segment/camera"
            )
        if evidence_id in seen_ids:
            raise VisualPresentationError(f"duplicate evidence_id {evidence_id!r}")
        seen_ids.add(evidence_id)
        if not isinstance(frame, Mapping) or not isinstance(image_meta, Mapping):
            raise VisualPresentationError(
                "visual evidence frame/image metadata is invalid"
            )
        frame_index = frame.get("index")
        frame_ref = frame.get("evidence_ref", frame.get("ref", frame.get("frame_ref")))
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise VisualPresentationError("visual evidence frame index is invalid")
        if not isinstance(frame_ref, str) or not frame_ref:
            frame_ref = f"#frame/{frame_index:06d}"
        image_ref_id = image_meta.get("image_ref_id")
        media_type = image_meta.get("media_type")
        declared_sha = image_meta.get("sha256")
        if not isinstance(image_ref_id, str) or not image_ref_id:
            raise VisualPresentationError("visual evidence image_ref_id is invalid")
        if media_type not in _MEDIA_TYPES:
            raise VisualPresentationError("visual evidence media_type is unsupported")
        if not isinstance(declared_sha, str) or len(declared_sha) != 64:
            raise VisualPresentationError("visual evidence sha256 is invalid")
        data = _resolved_bytes(resolver, image_ref_id)
        if hashlib.sha256(data).hexdigest() != declared_sha:
            raise VisualPresentationError(
                "resolved payload hash does not match evidence"
            )
        declared_length = image_meta.get("byte_length")
        if isinstance(declared_length, int) and declared_length != len(data):
            raise VisualPresentationError(
                "resolved payload length does not match evidence"
            )
        try:
            with Image.open(BytesIO(data)) as decoded:
                decoded.load()
                image = decoded.convert("RGB")
        except Exception as exc:
            raise VisualPresentationError(
                "resolved payload is not a decodable image"
            ) from exc
        gain = raw.get("view_information_gain")
        extraction_provenance = raw.get("extraction_provenance")
        selection_reasons = (
            extraction_provenance.get("selection_reasons", [])
            if isinstance(extraction_provenance, Mapping)
            else []
        )
        explicit_gain = (
            float(gain)
            if isinstance(gain, (int, float))
            and not isinstance(gain, bool)
            and math.isfinite(float(gain))
            else None
        )
        sources.append(
            _Source(
                evidence_id=evidence_id,
                segment_id=segment_id,
                frame_index=frame_index,
                frame_ref=frame_ref,
                camera_id=camera_id,
                temporal_role=str(raw.get("temporal_role", "")),
                image_ref_id=image_ref_id,
                media_type=media_type,
                declared_sha256=declared_sha,
                data=data,
                image=image,
                selection_reasons=tuple(
                    value
                    for value in selection_reasons
                    if isinstance(value, str) and value
                ),
                explicit_gain=explicit_gain,
            )
        )
        if camera_id not in camera_order:
            camera_order.append(camera_id)
    capabilities = visual_bundle.get("data_capabilities", {})
    if isinstance(capabilities, Mapping):
        rgb = capabilities.get("rgb", {})
        if isinstance(rgb, Mapping):
            declared = rgb.get("cameras", [])
            if isinstance(declared, Sequence) and not isinstance(
                declared, (str, bytes)
            ):
                ordered = [value for value in declared if value in camera_order]
                camera_order = ordered + [
                    value for value in camera_order if value not in ordered
                ]
    return sources, camera_order


def _selected_frames(action_bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    declared_values = action_bundle.get("selected_frames")
    chunks = action_bundle.get("action_chunks", action_bundle.get("chunks", []))
    transition_values = action_bundle.get("gripper_transitions", [])
    transitions = (
        {
            str(value["transition_id"]): value
            for value in transition_values
            if isinstance(value, Mapping)
            and isinstance(value.get("transition_id"), str)
        }
        if isinstance(transition_values, Sequence)
        and not isinstance(transition_values, (str, bytes))
        else {}
    )
    result_by_index: dict[int, dict[str, Any]] = {}
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise VisualPresentationError("action chunk bundle has no action_chunks array")
    action_role_names = {
        "before": "action_chunk_before",
        "during": "action_chunk_during",
        "after": "action_chunk_after",
        "settled": "release_after_settle",
    }
    for raw in chunks:
        if not isinstance(raw, Mapping):
            continue
        refs = raw.get("evidence_frame_refs", {})
        if not isinstance(refs, Mapping):
            continue
        segment_id = raw.get("segment_id")
        chunk_id = raw.get("action_chunk_id")
        for key, role in action_role_names.items():
            values = refs.get(key, [])
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for frame_ref in values:
                if not isinstance(frame_ref, str):
                    continue
                index = _frame_index_from_ref(frame_ref)
                entry = result_by_index.setdefault(
                    index,
                    {
                        "frame_index": index,
                        "frame_ref": frame_ref,
                        "roles": [],
                        "action_chunk_ids": [],
                        "segment_ids": [],
                        "role_chunk_ids": defaultdict(set),
                    },
                )
                entry["roles"].append(role)
                if isinstance(chunk_id, str):
                    entry["action_chunk_ids"].append(chunk_id)
                    entry["role_chunk_ids"][role].add(chunk_id)
                if isinstance(segment_id, str):
                    entry["segment_ids"].append(segment_id)
        transition_ids = raw.get("gripper_transition_refs", [])
        if isinstance(transition_ids, Sequence) and not isinstance(
            transition_ids, (str, bytes)
        ):
            for transition_id in transition_ids:
                transition = transitions.get(str(transition_id))
                if transition is None:
                    continue
                for endpoint, role in (
                    ("before_frame_index", "gripper_transition_before"),
                    ("after_frame_index", "gripper_transition_after"),
                ):
                    index = transition.get(endpoint)
                    if isinstance(index, bool) or not isinstance(index, int):
                        continue
                    entry = result_by_index.setdefault(
                        index,
                        {
                            "frame_index": index,
                            "frame_ref": f"#frame/{index:06d}",
                            "roles": [],
                            "action_chunk_ids": [],
                            "segment_ids": [],
                            "role_chunk_ids": defaultdict(set),
                        },
                    )
                    entry["roles"].append(role)
                    if isinstance(chunk_id, str):
                        entry["action_chunk_ids"].append(chunk_id)
                        entry["role_chunk_ids"][role].add(chunk_id)
                    if isinstance(segment_id, str):
                        entry["segment_ids"].append(segment_id)

    # Subtask boundary and feedback roles do not belong to one action chunk.
    # Read only those declarations from the flattened selected-frame index;
    # chunk-local roles above are reconstructed from each chunk's own refs so a
    # roles×chunk_ids Cartesian product cannot corrupt provenance.
    if isinstance(declared_values, Sequence) and not isinstance(
        declared_values, (str, bytes)
    ):
        for raw in declared_values:
            if not isinstance(raw, Mapping):
                continue
            frame_index = raw.get("frame_index")
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                continue
            roles = raw.get("roles", [])
            segment_ids = raw.get("segment_ids", [])
            if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
                continue
            entry = result_by_index.setdefault(
                frame_index,
                {
                    "frame_index": frame_index,
                    "frame_ref": raw.get("frame_ref", f"#frame/{frame_index:06d}"),
                    "roles": [],
                    "action_chunk_ids": [],
                    "segment_ids": [],
                    "role_chunk_ids": defaultdict(set),
                },
            )
            entry["roles"].extend(
                role
                for role in roles
                if role
                in {"subtask_before", "subtask_after", "task_feedback", "context"}
            )
            if isinstance(segment_ids, Sequence) and not isinstance(
                segment_ids, (str, bytes)
            ):
                entry["segment_ids"].extend(
                    value for value in segment_ids if isinstance(value, str)
                )
    for entry in result_by_index.values():
        entry["roles"] = sorted(set(entry["roles"]))
        entry["action_chunk_ids"] = sorted(set(entry["action_chunk_ids"]))
        entry["segment_ids"] = sorted(set(entry["segment_ids"]))
        entry["role_chunk_ids"] = {
            role: sorted(chunk_ids)
            for role, chunk_ids in sorted(entry["role_chunk_ids"].items())
        }
    return sorted(result_by_index.values(), key=lambda value: value["frame_index"])


def _entropy_score(image: Image.Image) -> float:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    counts = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) / 8.0
    horizontal = float(np.abs(np.diff(array.astype(np.int16), axis=1)).mean()) / 255.0
    vertical = float(np.abs(np.diff(array.astype(np.int16), axis=0)).mean()) / 255.0
    return 0.7 * entropy + 0.15 * horizontal + 0.15 * vertical


def _normalized_change_map(
    before: Image.Image, after: Image.Image
) -> np.ndarray | None:
    if before.size != after.size:
        return None
    first = np.asarray(before.convert("RGB"), dtype=np.int16)
    second = np.asarray(after.convert("RGB"), dtype=np.int16)
    delta = second - first
    illumination = np.median(delta.reshape(-1, 3), axis=0)
    return np.mean(np.abs(delta - illumination), axis=2) / 255.0


def _normalized_change_score(before: Image.Image, after: Image.Image) -> float:
    residual = _normalized_change_map(before, after)
    if residual is None:
        return 0.0
    # A high quantile captures small, salient changes without rewarding a
    # globally different/bright camera view.
    return float(np.quantile(residual, 0.95))


def _find_change_rois(
    before: Image.Image, after: Image.Image, limits: VisualPresentationLimits
) -> tuple[tuple[int, int, int, int], ...]:
    residual = _normalized_change_map(before, after)
    if residual is None:
        return ()
    height, width = residual.shape
    tile = limits.roi_tile_size
    rows = math.ceil(height / tile)
    columns = math.ceil(width / tile)
    scores = np.zeros((rows, columns), dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            patch = residual[
                row * tile : min(height, (row + 1) * tile),
                column * tile : min(width, (column + 1) * tile),
            ]
            scores[row, column] = float(patch.mean())
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    threshold = max(
        limits.roi_min_normalized_change,
        median + limits.roi_mad_multiplier * 1.4826 * mad,
    )
    active = scores >= threshold
    if not bool(active.any()):
        return ()
    # Rank spatially distinct connected components.  No task label, learned
    # object category, preselected timestep, or preselected image coordinate
    # enters this step.
    visited: set[tuple[int, int]] = set()
    components: list[tuple[float, list[tuple[int, int]]]] = []
    for start_row, start_column in zip(*np.nonzero(active), strict=True):
        start = (int(start_row), int(start_column))
        if start in visited:
            continue
        stack = [start]
        component: list[tuple[int, int]] = []
        score = 0.0
        while stack:
            row, column = stack.pop()
            if (row, column) in visited or not active[row, column]:
                continue
            visited.add((row, column))
            component.append((row, column))
            score += float(scores[row, column])
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if 0 <= next_row < rows and 0 <= next_column < columns:
                    stack.append((next_row, next_column))
        if component:
            components.append((score, component))
    boxes: list[tuple[int, int, int, int]] = []
    padding = limits.roi_padding_pixels
    for _score, component in sorted(components, key=lambda value: -value[0])[
        : limits.max_change_rois_per_pair
    ]:
        left = min(column for _, column in component) * tile
        top = min(row for row, _ in component) * tile
        right = min(width, (max(column for _, column in component) + 1) * tile)
        bottom = min(height, (max(row for row, _ in component) + 1) * tile)
        boxes.append(
            (
                max(0, left - padding),
                max(0, top - padding),
                min(width, right + padding),
                min(height, bottom + padding),
            )
        )
    return tuple(boxes)


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _resize_crop(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    limits: VisualPresentationLimits,
) -> tuple[Image.Image, dict[str, Any]]:
    cropped = image.crop(crop_box)
    original_size = cropped.size
    scale = max(1, math.ceil(limits.roi_min_output_side / min(cropped.size)))
    target = (cropped.width * scale, cropped.height * scale)
    if target[0] * target[1] > limits.max_pixels_per_image:
        shrink = math.sqrt(limits.max_pixels_per_image / (target[0] * target[1]))
        target = (max(1, int(target[0] * shrink)), max(1, int(target[1] * shrink)))
    if target != cropped.size:
        cropped = cropped.resize(target, resample=Image.Resampling.NEAREST)
    return cropped, {
        "from_size": [original_size[0], original_size[1]],
        "to_size": [cropped.width, cropped.height],
        "interpolation": "nearest",
    }


def build_visual_presentation_plan(
    *,
    action_chunk_bundle: Mapping[str, Any] | Any,
    visual_evidence_bundle: Mapping[str, Any] | Any,
    payload_resolver: VisualPayloadResolver | Callable[[str], bytes],
    limits: VisualPresentationLimits | None = None,
    primary_camera_id: str | None = None,
) -> VisualPresentationResult:
    """Build bounded Layer-1 and cross-segment visual request plans.

    The function performs no network I/O.  All generated records contain only
    presentation metadata; byte payloads are returned separately.  Any missing
    required role or exceeded hard budget makes the result inadmissible and
    suppresses every request plan.
    """

    limits = limits or VisualPresentationLimits()
    action = _as_mapping(action_chunk_bundle, label="action_chunk_bundle")
    visual = _as_mapping(visual_evidence_bundle, label="visual_evidence_bundle")
    sources, camera_order = _load_sources(visual, payload_resolver)
    if not camera_order:
        raise VisualPresentationError("visual evidence has no cameras")
    primary_camera = primary_camera_id or camera_order[0]
    if primary_camera not in camera_order:
        raise VisualPresentationError(
            "primary_camera_id is absent from visual evidence"
        )
    by_frame: dict[int, list[_Source]] = defaultdict(list)
    for source in sources:
        by_frame[source.frame_index].append(source)
    selected_frames = _selected_frames(action)
    chunks_value = action.get("action_chunks", action.get("chunks", []))
    chunk_records = (
        [dict(value) for value in chunks_value if isinstance(value, Mapping)]
        if isinstance(chunks_value, Sequence)
        and not isinstance(chunks_value, (str, bytes))
        else []
    )
    transition_values = action.get("gripper_transitions", [])
    transitions_by_id = (
        {
            str(value["transition_id"]): dict(value)
            for value in transition_values
            if isinstance(value, Mapping)
            and isinstance(value.get("transition_id"), str)
        }
        if isinstance(transition_values, Sequence)
        and not isinstance(transition_values, (str, bytes))
        else {}
    )
    chunk_frame_roles: dict[str, dict[str, set[int]]] = {}
    chunks_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunk_records:
        chunk_id = chunk.get("action_chunk_id")
        segment_id = chunk.get("segment_id")
        refs = chunk.get("evidence_frame_refs", {})
        if (
            not isinstance(chunk_id, str)
            or not isinstance(segment_id, str)
            or not isinstance(refs, Mapping)
        ):
            continue
        frames_by_role: dict[str, set[int]] = {}
        for role in ("before", "during", "after", "settled"):
            role_refs = refs.get(role, [])
            frames_by_role[role] = (
                {
                    _frame_index_from_ref(value)
                    for value in role_refs
                    if isinstance(value, str)
                }
                if isinstance(role_refs, Sequence)
                and not isinstance(role_refs, (str, bytes))
                else set()
            )
        chunk_frame_roles[chunk_id] = frames_by_role
        chunks_by_segment[segment_id].append(chunk)

    def bound_chunk_ids(segment_id: str, frame_index: int, role: str) -> list[str]:
        role_key = {
            "action_chunk_before": "before",
            "action_chunk_during": "during",
            "action_chunk_after": "after",
            "action_chunk_settled": "settled",
        }.get(role)
        result: list[str] = []
        for chunk in chunks_by_segment.get(segment_id, []):
            chunk_id = str(chunk.get("action_chunk_id", ""))
            if role_key is not None:
                if frame_index in chunk_frame_roles.get(chunk_id, {}).get(
                    role_key, set()
                ):
                    result.append(chunk_id)
                continue
            if role == "release_after_settle":
                if chunk.get("inferred_phase") in {
                    "release",
                    "release_and_settle",
                } and frame_index in chunk_frame_roles.get(chunk_id, {}).get(
                    "settled", set()
                ):
                    result.append(chunk_id)
                continue
            if role in {"gripper_transition_before", "gripper_transition_after"}:
                endpoint = (
                    "before_frame_index"
                    if role.endswith("before")
                    else "after_frame_index"
                )
                transition_ids = chunk.get("gripper_transition_refs", [])
                if (
                    isinstance(transition_ids, Sequence)
                    and not isinstance(transition_ids, (str, bytes))
                    and any(
                        transitions_by_id.get(str(transition_id), {}).get(endpoint)
                        == frame_index
                        for transition_id in transition_ids
                    )
                ):
                    result.append(chunk_id)
        return sorted(set(result))

    gaps: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    assets_by_sha: dict[str, _Asset] = {}
    selected_frame_keys: set[tuple[str, int]] = set()
    role_sources: dict[tuple[str, str, str], _Source] = {}

    def gap(code: str, *, blocking: bool, **details: Any) -> None:
        gaps.append({"code": code, "blocking": blocking, **details})

    def add_source(
        source: _Source,
        *,
        role: str,
        segment_id: str,
        chunk_id: str,
        required: bool,
        priority: int,
        reason: str,
    ) -> bool:
        frame_key = (source.camera_id, source.frame_index)
        new_frame = frame_key not in selected_frame_keys
        existing = assets_by_sha.get(source.sha256)
        if source.pixels > limits.max_pixels_per_image:
            gap(
                "source_exceeds_pixel_budget",
                blocking=required,
                evidence_id=source.evidence_id,
                pixels=source.pixels,
            )
            decisions.append(
                {
                    "evidence_id": source.evidence_id,
                    "role": role,
                    "priority": priority,
                    "decision": "rejected",
                    "reason": "source_exceeds_pixel_budget",
                }
            )
            return False
        if (
            new_frame
            and len(selected_frame_keys)
            >= limits.max_unique_source_frames_per_trajectory
        ):
            gap(
                "unique_source_frame_budget_exhausted",
                blocking=required,
                evidence_id=source.evidence_id,
                role=role,
            )
            decisions.append(
                {
                    "evidence_id": source.evidence_id,
                    "role": role,
                    "priority": priority,
                    "decision": "rejected",
                    "reason": "unique_source_frame_budget_exhausted",
                }
            )
            return False
        binding = {
            "evidence_id": source.evidence_id,
            "segment_id": segment_id,
            "action_chunk_id": chunk_id,
            "role": role,
            "frame_index": source.frame_index,
            "camera_id": source.camera_id,
        }
        if existing is None:
            existing = _Asset(
                presentation_id=f"vp_src_{source.sha256[:24]}",
                kind="source_frame",
                media_type=source.media_type,
                data=source.data,
                width=source.image.width,
                height=source.image.height,
                underlying_refs=[source.evidence_id],
                logical_bindings=[binding],
                segment_ids={segment_id},
                source_hashes=[source.sha256],
                priority=priority,
                required=required,
                provenance={
                    "generation_method": "unaltered_source_payload/v1",
                    "source_image_refs": [source.image_ref_id],
                    "content_hash_deduplication": True,
                    "presentation_only": True,
                },
            )
            assets_by_sha[source.sha256] = existing
        else:
            if source.evidence_id not in existing.underlying_refs:
                existing.underlying_refs.append(source.evidence_id)
            if binding not in existing.logical_bindings:
                existing.logical_bindings.append(binding)
            existing.segment_ids.add(segment_id)
            existing.priority = min(existing.priority, priority)
            existing.required = existing.required or required
        selected_frame_keys.add(frame_key)
        role_sources[(segment_id, chunk_id, role)] = source
        decisions.append(
            {
                "evidence_id": source.evidence_id,
                "role": role,
                "priority": priority,
                "decision": "selected"
                if len(existing.logical_bindings) == 1
                else "content_deduplicated_logical_ref_retained",
                "reason": reason,
            }
        )
        return True

    normalized_frames: list[dict[str, Any]] = []
    for raw in selected_frames:
        frame_index = raw.get("frame_index")
        roles = raw.get("roles", [])
        segment_ids = raw.get("segment_ids", [])
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            continue
        if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
            continue
        valid_roles = [role for role in roles if role in _ROLE_PRIORITY]
        if not valid_roles:
            continue
        segments = (
            [value for value in segment_ids if isinstance(value, str)]
            if isinstance(segment_ids, Sequence)
            and not isinstance(segment_ids, (str, bytes))
            else []
        )
        normalized_frames.append(
            {
                "frame_index": frame_index,
                "roles": valid_roles,
                "segment_ids": segments,
            }
        )
    normalized_frames.sort(
        key=lambda value: (
            min(_ROLE_PRIORITY[role] for role in value["roles"]),
            value["frame_index"],
        )
    )
    for frame in normalized_frames:
        candidates = by_frame.get(frame["frame_index"], [])
        for role in sorted(frame["roles"], key=lambda value: _ROLE_PRIORITY[value]):
            required = role in _REQUIRED_ROLES
            segments = frame["segment_ids"] or (
                [candidates[0].segment_id] if candidates else [""]
            )
            for segment_id in segments:
                segment_candidates = [
                    item for item in candidates if item.segment_id == segment_id
                ] or candidates
                expected_temporal_role = _ROLE_TO_TEMPORAL_ROLE[role]
                role_candidates = [
                    item
                    for item in segment_candidates
                    if item.temporal_role == expected_temporal_role
                ]
                if role_candidates:
                    segment_candidates = role_candidates
                chunks = bound_chunk_ids(segment_id, frame["frame_index"], role)
                if role in {
                    "subtask_before",
                    "subtask_after",
                    "task_feedback",
                    "context",
                }:
                    chunks = [""]
                elif not chunks:
                    gap(
                        "role_has_no_exact_action_chunk_binding",
                        blocking=required,
                        frame_index=frame["frame_index"],
                        role=role,
                        segment_id=segment_id,
                    )
                    continue
                for chunk_id in chunks:
                    chunk_candidates = segment_candidates
                    if chunk_id:
                        exact = [
                            item
                            for item in segment_candidates
                            if f"action_chunk_ref:{chunk_id}" in item.selection_reasons
                        ]
                        if exact:
                            chunk_candidates = exact
                    primary_candidates = [
                        item
                        for item in chunk_candidates
                        if item.camera_id == primary_camera
                    ]
                    source = (
                        primary_candidates[0]
                        if primary_candidates
                        else (chunk_candidates[0] if chunk_candidates else None)
                    )
                    if source is None:
                        gap(
                            "required_role_has_no_visual_frame",
                            blocking=required,
                            frame_index=frame["frame_index"],
                            role=role,
                            segment_id=segment_id,
                            action_chunk_id=chunk_id,
                        )
                        continue
                    add_source(
                        source,
                        role=role,
                        segment_id=segment_id,
                        chunk_id=chunk_id,
                        required=required,
                        priority=_ROLE_PRIORITY[role],
                        reason="role_priority_primary_view",
                    )

    derived_counts: dict[str, int] = defaultdict(int)
    derived_assets: list[_Asset] = []
    derived_keys: set[str] = set()

    def add_derived(
        *,
        kind: str,
        image: Image.Image,
        before: _Source,
        after: _Source,
        segment_id: str,
        chunk_id: str,
        crop_box: tuple[int, int, int, int],
        resize: Mapping[str, Any],
        required: bool,
        method: str,
        additional_sources: Sequence[_Source] = (),
    ) -> bool:
        source_values = (before, *tuple(additional_sources), after)
        source_hashes = sorted({source.sha256 for source in source_values})
        source_frame_keys = sorted({source.image_ref_id for source in source_values})
        if any(
            derived_counts[source_key]
            >= limits.max_derived_presentations_per_source_frame
            for source_key in source_frame_keys
        ):
            gap(
                "derived_per_source_budget_exhausted",
                blocking=required,
                segment_id=segment_id,
                action_chunk_id=chunk_id,
                presentation_kind=kind,
            )
            return False
        data = _encode_png(image)
        if image.width * image.height > limits.max_pixels_per_image:
            gap(
                "derived_pixel_budget_exhausted",
                blocking=required,
                presentation_kind=kind,
            )
            return False
        key_payload = {
            "kind": kind,
            "source_hashes": source_hashes,
            "crop_box": list(crop_box),
            "resize": dict(resize),
            "output_sha256": hashlib.sha256(data).hexdigest(),
        }
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if key in derived_keys:
            return True
        derived_keys.add(key)
        refs = sorted({source.evidence_id for source in source_values})
        asset = _Asset(
            presentation_id=f"vp_der_{key[:24]}",
            kind=kind,
            media_type="image/png",
            data=data,
            width=image.width,
            height=image.height,
            underlying_refs=refs,
            logical_bindings=[
                {
                    "evidence_id": reference,
                    "segment_id": segment_id,
                    "action_chunk_id": chunk_id,
                    "role": "derived_visual_presentation",
                }
                for reference in refs
            ],
            segment_ids={segment_id},
            source_hashes=source_hashes,
            priority=3,
            required=required,
            provenance={
                "generation_method": method,
                "source_evidence_refs": refs,
                "source_image_sha256": source_hashes,
                "source_cameras": sorted(
                    {source.camera_id for source in source_values}
                ),
                "crop_box_xyxy_exclusive": list(crop_box),
                "resize_transform": dict(resize),
                "semantic_interpretation": "none",
                "causal_attribution": False,
                "presentation_only": True,
                "output_sha256": hashlib.sha256(data).hexdigest(),
            },
        )
        derived_assets.append(asset)
        for source_key in source_frame_keys:
            derived_counts[source_key] += 1
        return True

    pair_specs: list[tuple[str, str, _Source, _Source, bool]] = []
    chunks_raw = action.get("action_chunks", action.get("chunks", []))
    if isinstance(chunks_raw, Sequence) and not isinstance(chunks_raw, (str, bytes)):
        for raw in chunks_raw:
            if not isinstance(raw, Mapping):
                continue
            segment_id = raw.get("segment_id")
            chunk_id = raw.get("action_chunk_id")
            if not isinstance(segment_id, str) or not isinstance(chunk_id, str):
                continue
            before = role_sources.get((segment_id, chunk_id, "action_chunk_before"))
            after = role_sources.get((segment_id, chunk_id, "action_chunk_after"))
            if before is not None and after is not None:
                pair_specs.append((segment_id, chunk_id, before, after, False))
    segments = sorted(
        {
            segment_id
            for asset in assets_by_sha.values()
            for segment_id in asset.segment_ids
            if segment_id
        }
    )
    for segment_id in segments:
        before_values = [
            source
            for (segment, _chunk, role), source in role_sources.items()
            if segment == segment_id and role == "subtask_before"
        ]
        after_values = [
            source
            for (segment, _chunk, role), source in role_sources.items()
            if segment == segment_id and role in {"task_feedback", "subtask_after"}
        ]
        if before_values and after_values:
            pair_specs.insert(
                0, (segment_id, "", before_values[0], after_values[-1], True)
            )
    seen_pairs: set[tuple[str, str, str]] = set()

    def derive_pair_presentations(
        segment_id: str,
        chunk_id: str,
        before: _Source,
        after: _Source,
        *,
        feedback_pair: bool,
        view_role: str,
    ) -> None:
        pair_key = (segment_id, before.sha256, after.sha256)
        if pair_key in seen_pairs:
            return
        seen_pairs.add(pair_key)
        if before.image.size == after.image.size:
            full_width, full_height = before.image.size
            during = (
                role_sources.get((segment_id, chunk_id, "action_chunk_during"))
                if chunk_id
                else None
            )
            temporal_sources = [before]
            if during is not None and during.image.size == before.image.size:
                temporal_sources.append(during)
            temporal_sources.append(after)
            full_panel = Image.new(
                "RGB", (full_width * len(temporal_sources), full_height)
            )
            for index, source in enumerate(temporal_sources):
                full_panel.paste(source.image, (full_width * index, 0))
            if full_panel.width * full_panel.height > limits.max_pixels_per_image:
                scale = math.sqrt(
                    limits.max_pixels_per_image / (full_panel.width * full_panel.height)
                )
                full_panel = full_panel.resize(
                    (
                        max(2, int(full_panel.width * scale)),
                        max(1, int(full_panel.height * scale)),
                    ),
                    Image.Resampling.NEAREST,
                )
            add_derived(
                kind=(
                    "full_before_during_after_panel"
                    if during is not None and len(temporal_sources) == 3
                    else "full_before_after_panel"
                ),
                image=full_panel,
                before=before,
                after=after,
                segment_id=segment_id,
                chunk_id=chunk_id,
                crop_box=(0, 0, full_width, full_height),
                resize={
                    "from_size": [full_width, full_height],
                    "to_size": [full_panel.width, full_panel.height],
                    "interpolation": "nearest",
                    "layout": (
                        "full_before_left_full_during_center_full_after_right"
                        if len(temporal_sources) == 3
                        else "full_before_left_full_after_right"
                    ),
                },
                required=True,
                method="task_neutral_full_before_after_panel/v1",
                additional_sources=(
                    (during,)
                    if during is not None and len(temporal_sources) == 3
                    else ()
                ),
            )
        else:
            gap(
                "full_panel_source_shapes_do_not_match",
                blocking=True,
                segment_id=segment_id,
                action_chunk_id=chunk_id,
            )
        if not feedback_pair:
            return
        rois = _find_change_rois(before.image, after.image, limits)
        if not rois:
            gap(
                "local_change_roi_not_detected",
                blocking=False,
                segment_id=segment_id,
                action_chunk_id=chunk_id,
                camera_id=before.camera_id,
            )
            return
        for roi_rank, roi in enumerate(rois):
            before_crop, resize = _resize_crop(before.image, roi, limits)
            after_crop, _ = _resize_crop(after.image, roi, limits)
            if before_crop.size != after_crop.size:
                gap(
                    "roi_source_shapes_do_not_match",
                    blocking=feedback_pair,
                    segment_id=segment_id,
                    camera_id=before.camera_id,
                )
                continue
            difference = Image.fromarray(
                np.abs(
                    np.asarray(after_crop, dtype=np.int16)
                    - np.asarray(before_crop, dtype=np.int16)
                ).astype(np.uint8),
                mode="RGB",
            )
            crop_width, crop_height = before_crop.size
            panel_width = crop_width * 3
            context_height = max(1, min(crop_height, panel_width // 4))
            context_width = panel_width // 2
            before_context = before.image.copy()
            before_context.thumbnail(
                (context_width, context_height), Image.Resampling.BILINEAR
            )
            after_context = after.image.copy()
            after_context.thumbnail(
                (context_width, context_height), Image.Resampling.BILINEAR
            )
            panel = Image.new(
                "RGB", (panel_width, context_height + crop_height), (0, 0, 0)
            )
            panel.paste(before_context, (0, 0))
            panel.paste(after_context, (context_width, 0))
            panel.paste(before_crop, (0, context_height))
            panel.paste(after_crop, (crop_width, context_height))
            panel.paste(difference, (crop_width * 2, context_height))
            if panel.width * panel.height > limits.max_pixels_per_image:
                scale = math.sqrt(
                    limits.max_pixels_per_image / (panel.width * panel.height)
                )
                panel = panel.resize(
                    (
                        max(3, int(panel.width * scale)),
                        max(1, int(panel.height * scale)),
                    ),
                    Image.Resampling.NEAREST,
                )
            ranked_resize = {
                **resize,
                "roi_rank": roi_rank,
                "view_role": view_role,
            }
            add_derived(
                kind="roi_crop_before",
                image=before_crop,
                before=before,
                after=after,
                segment_id=segment_id,
                chunk_id=chunk_id,
                crop_box=roi,
                resize=ranked_resize,
                required=False,
                method="task_neutral_topk_local_change_crop/v1",
            )
            add_derived(
                kind="roi_crop_after",
                image=after_crop,
                before=before,
                after=after,
                segment_id=segment_id,
                chunk_id=chunk_id,
                crop_box=roi,
                resize=ranked_resize,
                required=False,
                method="task_neutral_topk_local_change_crop/v1",
            )
            add_derived(
                kind="multiscale_change_panel",
                image=panel,
                before=before,
                after=after,
                segment_id=segment_id,
                chunk_id=chunk_id,
                crop_box=roi,
                resize={
                    **ranked_resize,
                    "layout": (
                        "full_before_after_top;"
                        "roi_before_after_absolute_difference_bottom"
                    ),
                    "difference_semantics": "non_semantic_absolute_pixel_delta",
                },
                required=feedback_pair and roi_rank == 0,
                method="task_neutral_multiscale_before_after_difference_panel/v1",
            )

    for segment_id, chunk_id, before, after, feedback_pair in pair_specs:
        derive_pair_presentations(
            segment_id,
            chunk_id,
            before,
            after,
            feedback_pair=feedback_pair,
            view_role="primary",
        )

    # Secondary views are considered only after endpoints, transitions,
    # release/feedback, and ROI presentations have consumed their budgets.
    selected_secondary_pairs: list[tuple[str, str, _Source, _Source, bool]] = []
    for segment_id, chunk_id, before, after, feedback_pair in pair_specs:
        primary_change = _normalized_change_score(before.image, after.image)
        primary_roi_count = len(_find_change_rois(before.image, after.image, limits))
        candidates: list[tuple[float, str, _Source, _Source, dict[str, Any]]] = []
        for camera_id in camera_order:
            if camera_id == primary_camera:
                continue
            before_candidates = [
                source
                for source in by_frame.get(before.frame_index, [])
                if source.segment_id == segment_id
                and source.camera_id == camera_id
                and source.temporal_role == before.temporal_role
            ]
            after_candidates = [
                source
                for source in by_frame.get(after.frame_index, [])
                if source.segment_id == segment_id
                and source.camera_id == camera_id
                and source.temporal_role == after.temporal_role
            ]
            if not before_candidates or not after_candidates:
                continue
            secondary_before = before_candidates[0]
            secondary_after = after_candidates[0]
            secondary_change = _normalized_change_score(
                secondary_before.image, secondary_after.image
            )
            relative_gain = (
                (secondary_change - primary_change) / max(primary_change, 1e-6)
                if primary_change > 0.0
                else (1.0 if secondary_change > 0.0 else 0.0)
            )
            secondary_roi_count = len(
                _find_change_rois(secondary_before.image, secondary_after.image, limits)
            )
            roi_component_gain = secondary_roi_count - primary_roi_count
            declared_gain = max(
                value
                for value in (
                    secondary_before.explicit_gain,
                    secondary_after.explicit_gain,
                    0.0,
                )
                if value is not None
            )
            complementary = (
                relative_gain >= limits.secondary_change_relative_gain_min
                or (
                    roi_component_gain > 0
                    and secondary_change
                    >= primary_change
                    * limits.secondary_change_retention_ratio_min
                )
                or declared_gain >= limits.secondary_view_min_gain
            )
            metrics = {
                "primary_normalized_change_score": primary_change,
                "secondary_normalized_change_score": secondary_change,
                "relative_change_gain": relative_gain,
                "primary_change_component_count": primary_roi_count,
                "secondary_change_component_count": secondary_roi_count,
                "component_count_gain": roi_component_gain,
                "declared_task_neutral_gain": declared_gain,
            }
            if not complementary:
                decisions.append(
                    {
                        "evidence_id": secondary_after.evidence_id,
                        "role": "secondary_view_pair",
                        "priority": 4,
                        "decision": "rejected",
                        "reason": "no_complementary_before_after_change_gain",
                        "metrics": metrics,
                    }
                )
                continue
            score = max(relative_gain, declared_gain) + (
                limits.secondary_component_gain_weight
                * max(0, roi_component_gain)
            )
            candidates.append(
                (
                    score,
                    camera_id,
                    secondary_before,
                    secondary_after,
                    metrics,
                )
            )
        if not candidates:
            continue
        _score, camera_id, secondary_before, secondary_after, metrics = max(
            candidates, key=lambda value: (value[0], value[1])
        )
        new_frame_keys = {
            (source.camera_id, source.frame_index)
            for source in (secondary_before, secondary_after)
            if (source.camera_id, source.frame_index) not in selected_frame_keys
        }
        if (
            len(selected_frame_keys) + len(new_frame_keys)
            > limits.max_unique_source_frames_per_trajectory
        ):
            decisions.append(
                {
                    "evidence_id": secondary_after.evidence_id,
                    "role": "secondary_view_pair",
                    "priority": 4,
                    "decision": "rejected",
                    "reason": "secondary_pair_source_frame_budget_exhausted",
                    "metrics": metrics,
                }
            )
            continue
        selected_before = add_source(
            secondary_before,
            role="secondary_view",
            segment_id=segment_id,
            chunk_id=chunk_id,
            required=False,
            priority=4,
            reason="complementary_before_after_change_gain",
        )
        selected_after = add_source(
            secondary_after,
            role="secondary_view",
            segment_id=segment_id,
            chunk_id=chunk_id,
            required=False,
            priority=4,
            reason="complementary_before_after_change_gain",
        )
        if selected_before and selected_after:
            decisions.append(
                {
                    "evidence_id": secondary_after.evidence_id,
                    "role": "secondary_view_pair",
                    "priority": 4,
                    "decision": "selected",
                    "reason": "complementary_before_after_change_gain",
                    "camera_id": camera_id,
                    "metrics": metrics,
                }
            )
            selected_secondary_pairs.append(
                (
                    segment_id,
                    chunk_id,
                    secondary_before,
                    secondary_after,
                    feedback_pair,
                )
            )
    for segment_id, chunk_id, before, after, feedback_pair in selected_secondary_pairs:
        derive_pair_presentations(
            segment_id,
            chunk_id,
            before,
            after,
            feedback_pair=feedback_pair,
            view_role=f"secondary:{before.camera_id}",
        )

    all_assets = list(assets_by_sha.values()) + derived_assets
    all_assets.sort(
        key=lambda asset: (
            asset.priority,
            min(asset.segment_ids or {""}),
            asset.presentation_id,
        )
    )

    def _unique_assets(values: Sequence[_Asset]) -> list[_Asset]:
        result: list[_Asset] = []
        seen: set[str] = set()
        for asset in values:
            if asset.presentation_id in seen:
                continue
            seen.add(asset.presentation_id)
            result.append(asset)
        return result

    def _fits_one_request(values: Sequence[_Asset]) -> bool:
        unique = _unique_assets(values)
        return (
            len(unique) <= limits.max_images_per_request
            and sum(len(asset.data) for asset in unique)
            <= limits.max_total_image_bytes_per_request
            and sum(asset.pixels for asset in unique)
            <= limits.max_total_pixels_per_request
        )

    def pack(
        scope: str,
        segment_id: str,
        values: Sequence[_Asset],
        *,
        action_chunk_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        current: list[_Asset] = []
        current_bytes = 0
        current_pixels = 0

        def flush() -> None:
            nonlocal current, current_bytes, current_pixels
            if not current:
                return
            request_id = f"vp_req_{len(requests):03d}_{hashlib.sha256((scope + segment_id + ''.join(asset.presentation_id for asset in current)).encode()).hexdigest()[:12]}"
            requests.append(
                {
                    "request_id": request_id,
                    "scope": scope,
                    "segment_id": segment_id,
                    "presentation_ids": [asset.presentation_id for asset in current],
                    "underlying_evidence_refs": sorted(
                        {
                            reference
                            for asset in current
                            for reference in asset.underlying_refs
                        }
                    ),
                    "image_count": len(current),
                    "total_image_bytes": current_bytes,
                    "total_pixels": current_pixels,
                    "action_chunk_ids": list(action_chunk_ids),
                }
            )
            current = []
            current_bytes = 0
            current_pixels = 0

        for asset in values:
            if (
                len(asset.data) > limits.max_total_image_bytes_per_request
                or asset.pixels > limits.max_total_pixels_per_request
            ):
                gap(
                    "single_presentation_exceeds_request_budget",
                    blocking=asset.required,
                    presentation_id=asset.presentation_id,
                )
                continue
            if current and (
                len(current) >= limits.max_images_per_request
                or current_bytes + len(asset.data)
                > limits.max_total_image_bytes_per_request
                or current_pixels + asset.pixels > limits.max_total_pixels_per_request
            ):
                flush()
            current.append(asset)
            current_bytes += len(asset.data)
            current_pixels += asset.pixels
        flush()
        for part_index, request in enumerate(requests):
            request["part_index"] = part_index
            request["part_count"] = len(requests)
            request["is_split_request"] = len(requests) > 1
        return requests

    def has_binding(
        asset: _Asset,
        *,
        segment_id: str,
        chunk_id: str | None = None,
        roles: frozenset[str] | None = None,
    ) -> bool:
        return any(
            binding.get("segment_id") == segment_id
            and (
                chunk_id is None or str(binding.get("action_chunk_id", "")) == chunk_id
            )
            and (roles is None or binding.get("role") in roles)
            for binding in asset.logical_bindings
        )

    def pack_layer1_chunk_groups(
        *,
        segment_id: str,
        groups: Sequence[tuple[str, Sequence[_Asset]]],
        optional_assets: Sequence[_Asset],
    ) -> list[dict[str, Any]]:
        """Pack complete chunk groups without dropping or splitting a chunk.

        A presentation shared by two chunks is deduplicated inside one request,
        but may intentionally be repeated in two requests when it is required
        to keep both chunk-local batches self-contained.
        """

        packed: list[tuple[list[str], list[_Asset]]] = []
        current_ids: list[str] = []
        current_assets: list[_Asset] = []

        def flush() -> None:
            nonlocal current_ids, current_assets
            if current_ids:
                packed.append((current_ids, _unique_assets(current_assets)))
            current_ids = []
            current_assets = []

        for chunk_id, raw_assets in groups:
            chunk_assets = _unique_assets(raw_assets)
            if not chunk_assets:
                gap(
                    "chunk_required_presentation_group_empty",
                    blocking=True,
                    scope=f"layer1:{segment_id}",
                    segment_id=segment_id,
                    action_chunk_id=chunk_id,
                )
                continue
            if not _fits_one_request(chunk_assets):
                gap(
                    "atomic_chunk_group_exceeds_request_budget",
                    blocking=True,
                    scope=f"layer1:{segment_id}",
                    segment_id=segment_id,
                    action_chunk_id=chunk_id,
                    required_images=len(chunk_assets),
                    required_bytes=sum(len(asset.data) for asset in chunk_assets),
                    required_pixels=sum(asset.pixels for asset in chunk_assets),
                )
                continue
            candidate = _unique_assets((*current_assets, *chunk_assets))
            if current_ids and not _fits_one_request(candidate):
                flush()
                candidate = chunk_assets
            current_ids.append(chunk_id)
            current_assets = candidate
        flush()

        # Optional context never creates another model call.  It is attached
        # only to an already complete batch whose chunk bindings it matches.
        for asset in _unique_assets(optional_assets):
            bound_ids = {
                str(binding.get("action_chunk_id", ""))
                for binding in asset.logical_bindings
                if binding.get("segment_id") == segment_id
                and str(binding.get("action_chunk_id", ""))
            }
            candidate_indexes = [
                index
                for index, (chunk_ids, _assets) in enumerate(packed)
                if not bound_ids or bound_ids.intersection(chunk_ids)
            ]
            for index in candidate_indexes:
                chunk_ids, batch_assets = packed[index]
                candidate = _unique_assets((*batch_assets, asset))
                if _fits_one_request(candidate):
                    packed[index] = (chunk_ids, candidate)
                    break

        requests: list[dict[str, Any]] = []
        for chunk_ids, batch_assets in packed:
            parts = pack(
                "layer1_segment",
                segment_id,
                batch_assets,
                action_chunk_ids=chunk_ids,
            )
            if len(parts) != 1:
                raise AssertionError("an atomic Layer-1 group was split")
            requests.extend(parts)
        for part_index, request in enumerate(requests):
            request["part_index"] = part_index
            request["part_count"] = len(requests)
            request["is_split_request"] = len(requests) > 1
        return requests

    raw_action_chunks = action.get("action_chunks", action.get("chunks", []))
    chunks_by_segment: dict[str, list[str]] = defaultdict(list)
    if isinstance(raw_action_chunks, Sequence) and not isinstance(
        raw_action_chunks, (str, bytes)
    ):
        for raw in raw_action_chunks:
            if not isinstance(raw, Mapping):
                continue
            segment_id = raw.get("segment_id")
            chunk_id = raw.get("action_chunk_id")
            if isinstance(segment_id, str) and isinstance(chunk_id, str):
                chunks_by_segment[segment_id].append(chunk_id)

    layer_requests: list[dict[str, Any]] = []
    for segment_id in segments:
        chunk_groups: list[tuple[str, list[_Asset]]] = []
        for chunk_id in chunks_by_segment.get(segment_id, []):
            required_assets: list[_Asset] = []
            panels = [
                asset
                for asset in derived_assets
                if asset.kind
                in {
                    "full_before_after_panel",
                    "full_before_during_after_panel",
                }
                and has_binding(
                    asset,
                    segment_id=segment_id,
                    chunk_id=chunk_id,
                )
            ]
            if panels:
                required_assets.append(panels[0])
            else:
                endpoints = [
                    asset
                    for asset in assets_by_sha.values()
                    if has_binding(
                        asset,
                        segment_id=segment_id,
                        chunk_id=chunk_id,
                        roles=frozenset({"action_chunk_before", "action_chunk_after"}),
                    )
                ]
                if len(endpoints) < 2:
                    gap(
                        "chunk_before_after_presentation_missing",
                        blocking=True,
                        segment_id=segment_id,
                        action_chunk_id=chunk_id,
                    )
                required_assets.extend(endpoints)
            required_assets.extend(
                asset
                for asset in assets_by_sha.values()
                if has_binding(
                    asset,
                    segment_id=segment_id,
                    chunk_id=chunk_id,
                    roles=frozenset({"release_after_settle"}),
                )
            )
            chunk_groups.append((chunk_id, _unique_assets(required_assets)))
        optional_assets = [
            asset
            for asset in assets_by_sha.values()
            if has_binding(
                asset,
                segment_id=segment_id,
                chunk_id=None,
                roles=frozenset({"secondary_view", "context"}),
            )
            and any(
                str(binding.get("action_chunk_id", ""))
                for binding in asset.logical_bindings
                if binding.get("segment_id") == segment_id
            )
        ]
        layer_requests.extend(
            pack_layer1_chunk_groups(
                segment_id=segment_id,
                groups=chunk_groups,
                optional_assets=optional_assets,
            )
        )

    cross_required: list[_Asset] = []
    cross_roi_required: list[_Asset] = []
    cross_roi_optional: list[_Asset] = []
    for segment_id in segments:
        feedback_derived = [
            asset
            for asset in derived_assets
            if segment_id in asset.segment_ids
            and any(
                str(binding.get("action_chunk_id", "")) == ""
                for binding in asset.logical_bindings
            )
        ]
        primary_full = [
            asset
            for asset in feedback_derived
            if asset.kind
            in {"full_before_after_panel", "full_before_during_after_panel"}
            and asset.provenance.get("source_cameras") == [primary_camera]
        ]
        primary_rois = sorted(
            (
                asset
                for asset in feedback_derived
                if asset.kind == "multiscale_change_panel"
                and asset.provenance.get("source_cameras") == [primary_camera]
            ),
            key=lambda asset: int(
                asset.provenance.get("resize_transform", {}).get("roi_rank", 0)
            ),
        )
        # A multiscale panel already contains the full before/after context.
        # Prefer two spatially distinct local-change panels and avoid sending
        # a redundant separate full panel; fall back to the full panel when no
        # local component is available.
        if primary_rois:
            cross_roi_required.extend(primary_rois[:2])
        elif primary_full:
            cross_required.append(primary_full[0])
        for camera_id in camera_order:
            if camera_id == primary_camera:
                continue
            secondary_rois = sorted(
                (
                    asset
                    for asset in feedback_derived
                    if asset.kind == "multiscale_change_panel"
                    and asset.provenance.get("source_cameras") == [camera_id]
                ),
                key=lambda asset: int(
                    asset.provenance.get("resize_transform", {}).get("roi_rank", 0)
                ),
            )
            if secondary_rois:
                cross_roi_optional.append(secondary_rois[0])
    cross_covered_refs = {
        reference for asset in cross_required for reference in asset.underlying_refs
    }
    cross_required.extend(
        asset
        for asset in assets_by_sha.values()
        if any(
            binding.get("role") in {"task_feedback", "release_after_settle"}
            for binding in asset.logical_bindings
        )
        and not set(asset.underlying_refs).issubset(cross_covered_refs)
    )
    cross_required.extend(cross_roi_required)
    # Individual ROI crops remain in the local review artifact.  The
    # before/after ROI panel already carries both crops into Layer 3, so sending
    # the two individual images again would add bytes without new evidence.
    cross_required = _unique_assets(cross_required)
    feedback_assets = list(cross_required)
    if not _fits_one_request(cross_required):
        gap(
            "cross_segment_required_evidence_exceeds_single_request_budget",
            blocking=True,
            scope="cross_segment_feedback",
            required_images=len(cross_required),
            required_bytes=sum(len(asset.data) for asset in cross_required),
            required_pixels=sum(asset.pixels for asset in cross_required),
            detail=(
                "Layer 3 cannot silently discard or independently summarize "
                "required cross-segment evidence under the current quality contract"
            ),
        )
        feedback_assets = []
    else:
        for asset in _unique_assets(cross_roi_optional):
            candidate = _unique_assets((*feedback_assets, asset))
            if _fits_one_request(candidate):
                feedback_assets = candidate
            else:
                gap(
                    "optional_cross_segment_presentation_omitted",
                    blocking=False,
                    scope="cross_segment_feedback",
                    presentation_id=asset.presentation_id,
                )
    cross_requests = (
        pack("cross_segment_feedback", "cross_segment", feedback_assets)
        if feedback_assets
        else []
    )
    if len(cross_requests) > 1:
        raise AssertionError("cross-segment evidence unexpectedly split")
    total_requests = layer_requests + cross_requests
    transmitted_bytes = sum(request["total_image_bytes"] for request in total_requests)
    transmitted_pixels = sum(request["total_pixels"] for request in total_requests)
    required_hierarchy_calls = len(layer_requests) + len(segments) + 2
    if required_hierarchy_calls > limits.max_model_calls_per_trajectory:
        gap(
            "model_call_budget_exhausted",
            blocking=True,
            required_calls=required_hierarchy_calls,
            maximum=limits.max_model_calls_per_trajectory,
            layer1_calls=len(layer_requests),
            layer2_calls=len(segments),
            layer3_calls=1,
            layer4_calls=1,
        )
    if transmitted_bytes > limits.max_total_image_bytes_per_trajectory:
        gap(
            "transmitted_trajectory_byte_budget_exhausted",
            blocking=True,
            required_bytes=transmitted_bytes,
            maximum=limits.max_total_image_bytes_per_trajectory,
        )
    if transmitted_pixels > limits.max_total_pixels_per_trajectory:
        gap(
            "transmitted_trajectory_pixel_budget_exhausted",
            blocking=True,
            required_pixels=transmitted_pixels,
            maximum=limits.max_total_pixels_per_trajectory,
        )
    selected_transition_ids: list[str] = []
    selected_role_frames = {
        (binding.get("role"), binding.get("frame_index"))
        for asset in assets_by_sha.values()
        for binding in asset.logical_bindings
    }
    for transition_id, transition in sorted(transitions_by_id.items()):
        before_index = transition.get("before_frame_index")
        after_index = transition.get("after_frame_index")
        if ("gripper_transition_before", before_index) in selected_role_frames and (
            "gripper_transition_after",
            after_index,
        ) in selected_role_frames:
            selected_transition_ids.append(transition_id)
        else:
            gap(
                "gripper_transition_visual_pair_missing",
                blocking=True,
                transition_id=transition_id,
                before_frame_index=before_index,
                after_frame_index=after_index,
            )
    blocking_gaps = [value for value in gaps if value["blocking"]]
    admissible = not blocking_gaps
    if not admissible:
        layer_requests = []
        cross_requests = []
    records = tuple(asset.record() for asset in all_assets)
    derived_records = tuple(
        record for record in records if record["is_derived_presentation"]
    )
    payloads = tuple(
        PresentationPayload(asset.presentation_id, asset.media_type, asset.data)
        for asset in all_assets
    )
    audit = {
        "schema": "roboharn_evo/visual_presentation_coverage_audit/v1",
        "schema_version": 1,
        "primary_camera_id": primary_camera,
        "role_priority": [
            "required_endpoints",
            "gripper_or_action_transitions",
            "release_or_feedback_settled",
            "task_neutral_roi_presentations",
            "incremental_secondary_view",
            "optional_context",
        ],
        "selected_unique_source_timepoints": len(
            {
                source.frame_index
                for source in sources
                if any(
                    source.evidence_id in asset.underlying_refs
                    for asset in assets_by_sha.values()
                )
            }
        ),
        "selected_unique_source_contents": len(assets_by_sha),
        "derived_presentation_count": len(derived_assets),
        "selected_gripper_transitions": selected_transition_ids,
        "selected_secondary_views": sorted(
            {
                binding["camera_id"]
                for asset in assets_by_sha.values()
                for binding in asset.logical_bindings
                if binding["role"] == "secondary_view"
            }
        ),
        "roi_presentations": [
            {
                "presentation_id": asset.presentation_id,
                "kind": asset.kind,
                "source_cameras": asset.provenance.get("source_cameras", []),
                "crop_box_xyxy_exclusive": asset.provenance.get(
                    "crop_box_xyxy_exclusive"
                ),
                "roi_rank": asset.provenance.get("resize_transform", {}).get(
                    "roi_rank"
                ),
            }
            for asset in derived_assets
            if asset.kind
            in {"roi_crop_before", "roi_crop_after", "multiscale_change_panel"}
        ],
        "coverage_gaps": gaps,
        "blocking_gap_count": len(blocking_gaps),
        "selection_decisions": decisions,
        "limits": {name: getattr(limits, name) for name in limits.__dataclass_fields__},
        "proposed_model_calls": required_hierarchy_calls,
        "proposed_image_requests": len(total_requests),
        "proposed_transmitted_bytes": transmitted_bytes,
        "proposed_transmitted_pixels": transmitted_pixels,
        "requests_suppressed": not admissible,
    }
    return VisualPresentationResult(
        presentation_records=records,
        derived_visual_presentations=derived_records,
        outbound_payloads=payloads,
        layer1_request_plans=tuple(layer_requests),
        cross_segment_feedback_plan={
            "schema": "roboharn_evo/cross_segment_visual_request_plan/v1",
            "requests": cross_requests,
        },
        coverage_audit=audit,
        admissible=admissible,
    )


__all__ = [
    "PresentationPayload",
    "VisualPayloadResolver",
    "VisualPresentationError",
    "VisualPresentationLimits",
    "VisualPresentationResult",
    "build_visual_presentation_plan",
]

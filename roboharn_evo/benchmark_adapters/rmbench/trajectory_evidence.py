"""Sparse, read-only visual evidence extraction for RMBench trajectories.

The extractor is deliberately task agnostic.  Segment boundaries come from a
normalized trajectory, cameras and robot-state semantics come from an adapter
configuration, and every HDF5 access is a scalar frame read.  Image bytes are
kept in a separate in-memory payload object so persisted JSON never contains
base64 or a copy of the expert video.

The manifest facade at the bottom of this module reuses the authorization,
content-hash, and path-substitution checks of :mod:`expert_trajectory`; it does
not discover dataset files or reopen a caller-provided dataset path.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Literal

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_FRAME_REF_RE = re.compile(r"^(?P<trajectory>[^#]+)#frame/(?P<index>\d{6})$")
_EVIDENCE_ID_RE = re.compile(r"^ve_[0-9a-f]{24}$")
_IMAGE_REF_ID_RE = re.compile(r"^img_[0-9a-f]{24}$")
_CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
_TEMPORAL_ROLES = frozenset(
    {"before", "during", "after", "settled", "task_feedback", "context"}
)
_GRIPPER_STATES = frozenset({"open", "closed", "indeterminate"})
_OPEN_DIRECTIONS = frozenset({"at_or_above", "at_or_below"})
_SELECTION_POLICY = "segment_boundaries_with_context/v1"
_GRIPPER_SCAN_MODE = "chunked_low_dimensional_state_scan/v1"
_EXTRACTOR_NAME = "rmbench_trajectory_evidence_extractor"
_EXTRACTOR_VERSION = "1"


class TrajectoryEvidenceError(ValueError):
    """A malformed input, unsupported HDF5 layout, or invalid evidence record."""


class TrajectoryEvidenceBudgetExceeded(TrajectoryEvidenceError):
    """Sparse extraction would exceed an explicit resource limit."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrajectoryEvidenceError("value is not strict JSON") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryEvidenceError(f"{path}: must be a non-empty string")
    return value


def _require_safe_id(value: Any, *, path: str) -> str:
    text = _require_string(value, path=path)
    if _SAFE_ID_RE.fullmatch(text) is None:
        raise TrajectoryEvidenceError(
            f"{path}: must contain only letters, digits, '.', '_', or '-'"
        )
    return text


def _require_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrajectoryEvidenceError(
            f"{path}: must be an integer no smaller than {minimum}"
        )
    return value


def _require_finite_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryEvidenceError(f"{path}: must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TrajectoryEvidenceError(f"{path}: must be a finite number")
    return result


def _require_sha256(value: Any, *, path: str) -> str:
    text = _require_string(value, path=path)
    if _SHA256_RE.fullmatch(text) is None:
        raise TrajectoryEvidenceError(f"{path}: must be lowercase SHA-256")
    return text


def _require_exact_keys(
    value: Any,
    keys: Sequence[str],
    *,
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrajectoryEvidenceError(f"{path}: must be an object")
    expected = set(keys)
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise TrajectoryEvidenceError(
            f"{path}: missing required field(s): {', '.join(missing)}"
        )
    if unknown:
        raise TrajectoryEvidenceError(f"{path}: unknown field(s): {', '.join(unknown)}")
    return value


def _require_hdf5_dataset_path(value: Any, *, path: str) -> str:
    text = _require_string(value, path=path)
    if text.startswith("/") or "\\" in text:
        raise TrajectoryEvidenceError(f"{path}: must be a relative POSIX HDF5 path")
    parts = PurePosixPath(text).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise TrajectoryEvidenceError(f"{path}: contains an unsafe path component")
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class CameraChannelConfig:
    """One RGB camera supplied by the benchmark adapter."""

    camera_id: str
    dataset_path: str

    def __post_init__(self) -> None:
        _require_safe_id(self.camera_id, path="camera.camera_id")
        _require_hdf5_dataset_path(self.dataset_path, path="camera.dataset_path")

    def to_dict(self) -> dict[str, str]:
        return {"camera_id": self.camera_id, "dataset_path": self.dataset_path}


@dataclass(frozen=True, slots=True)
class ActionChannelConfig:
    """Dataset and epistemic meaning of the per-frame action/state vector."""

    channel_id: str
    dataset_path: str
    semantics: str

    def __post_init__(self) -> None:
        _require_safe_id(self.channel_id, path="action.channel_id")
        _require_hdf5_dataset_path(self.dataset_path, path="action.dataset_path")
        _require_string(self.semantics, path="action.semantics")

    def to_dict(self) -> dict[str, str]:
        return {
            "channel_id": self.channel_id,
            "dataset_path": self.dataset_path,
            "semantics": self.semantics,
        }


@dataclass(frozen=True, slots=True)
class GripperChannelConfig:
    """Embodiment-specific scalar mapping; no global numeric convention exists."""

    channel_id: str
    dataset_path: str
    open_when: Literal["at_or_above", "at_or_below"]
    open_threshold: float
    closed_threshold: float
    semantics_source: str

    def __post_init__(self) -> None:
        _require_safe_id(self.channel_id, path="gripper.channel_id")
        _require_hdf5_dataset_path(self.dataset_path, path="gripper.dataset_path")
        if self.open_when not in _OPEN_DIRECTIONS:
            raise TrajectoryEvidenceError(
                "gripper.open_when: expected 'at_or_above' or 'at_or_below'"
            )
        open_threshold = _require_finite_number(
            self.open_threshold, path="gripper.open_threshold"
        )
        closed_threshold = _require_finite_number(
            self.closed_threshold, path="gripper.closed_threshold"
        )
        if self.open_when == "at_or_above" and not (closed_threshold < open_threshold):
            raise TrajectoryEvidenceError(
                "gripper thresholds require closed_threshold < open_threshold "
                "when open_when='at_or_above'"
            )
        if self.open_when == "at_or_below" and not (open_threshold < closed_threshold):
            raise TrajectoryEvidenceError(
                "gripper thresholds require open_threshold < closed_threshold "
                "when open_when='at_or_below'"
            )
        _require_string(self.semantics_source, path="gripper.semantics_source")

    def classify(self, value: float) -> str:
        value = _require_finite_number(value, path=f"gripper.{self.channel_id}.value")
        if self.open_when == "at_or_above":
            if value >= self.open_threshold:
                return "open"
            if value <= self.closed_threshold:
                return "closed"
        else:
            if value <= self.open_threshold:
                return "open"
            if value >= self.closed_threshold:
                return "closed"
        return "indeterminate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "dataset_path": self.dataset_path,
            "open_when": self.open_when,
            "open_threshold": float(self.open_threshold),
            "closed_threshold": float(self.closed_threshold),
            "semantics_source": self.semantics_source,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryEvidenceAdapterConfig:
    """All benchmark/embodiment meaning needed by the generic extractor."""

    adapter_id: str
    adapter_version: str
    embodiment_id: str
    cameras: tuple[CameraChannelConfig, ...]
    action: ActionChannelConfig
    grippers: tuple[GripperChannelConfig, ...]

    def __post_init__(self) -> None:
        _require_safe_id(self.adapter_id, path="adapter.adapter_id")
        _require_safe_id(self.adapter_version, path="adapter.adapter_version")
        _require_safe_id(self.embodiment_id, path="adapter.embodiment_id")
        if not self.cameras:
            raise TrajectoryEvidenceError("adapter.cameras: must not be empty")
        if not self.grippers:
            raise TrajectoryEvidenceError("adapter.grippers: must not be empty")
        if not all(isinstance(item, CameraChannelConfig) for item in self.cameras):
            raise TrajectoryEvidenceError(
                "adapter.cameras: all values must be CameraChannelConfig"
            )
        if not isinstance(self.action, ActionChannelConfig):
            raise TrajectoryEvidenceError("adapter.action: must be ActionChannelConfig")
        if not all(isinstance(item, GripperChannelConfig) for item in self.grippers):
            raise TrajectoryEvidenceError(
                "adapter.grippers: all values must be GripperChannelConfig"
            )
        camera_ids = [item.camera_id for item in self.cameras]
        camera_paths = [item.dataset_path for item in self.cameras]
        gripper_ids = [item.channel_id for item in self.grippers]
        gripper_paths = [item.dataset_path for item in self.grippers]
        for label, values in (
            ("camera_id", camera_ids),
            ("camera dataset_path", camera_paths),
            ("gripper channel_id", gripper_ids),
            ("gripper dataset_path", gripper_paths),
        ):
            if len(values) != len(set(values)):
                raise TrajectoryEvidenceError(f"adapter: duplicate {label}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "embodiment_id": self.embodiment_id,
            "cameras": [item.to_dict() for item in self.cameras],
            "action": self.action.to_dict(),
            "grippers": [item.to_dict() for item in self.grippers],
        }

    @property
    def configuration_sha256(self) -> str:
        return _sha256_json(self.to_dict())


def rmbench_dual_arm_trajectory_evidence_config(
    *, camera_ids: Sequence[str] | None = None
) -> TrajectoryEvidenceAdapterConfig:
    """Return the copied RMBench dual-arm HDF5/robot-state contract.

    The threshold meaning is owned by this adapter configuration and is
    traceable to ``benchmarks/rmbench/envs/robot/robot.py``.  Callers may select
    any subset of the five public RGB channels without repeating dataset paths
    or gripper semantics in a CLI.  The default keeps the two review cameras
    with the broadest scene coverage so sparse evidence remains bounded.
    """

    available_cameras = {
        "head_camera": "observation/head_camera/rgb",
        "third_view": "third_view_rgb",
        "front_camera": "observation/front_camera/rgb",
        "left_camera": "observation/left_camera/rgb",
        "right_camera": "observation/right_camera/rgb",
    }
    selected = (
        ("head_camera", "third_view") if camera_ids is None else tuple(camera_ids)
    )
    if not selected:
        raise TrajectoryEvidenceError("camera_ids: must not be empty")
    if len(selected) != len(set(selected)):
        raise TrajectoryEvidenceError("camera_ids: duplicate camera identifier")
    unknown = sorted(set(selected) - set(available_cameras))
    if unknown:
        raise TrajectoryEvidenceError(
            "camera_ids: unknown RMBench camera(s): " + ", ".join(unknown)
        )
    semantics_source = (
        "benchmarks/rmbench/envs/robot/robot.py:"
        "is_left_gripper_open,is_right_gripper_open,"
        "is_left_gripper_close,is_right_gripper_close"
    )
    return TrajectoryEvidenceAdapterConfig(
        adapter_id="rmbench_dual_arm_hdf5",
        adapter_version="1",
        embodiment_id="rmbench_normalized_dual_arm",
        cameras=tuple(
            CameraChannelConfig(camera_id, available_cameras[camera_id])
            for camera_id in selected
        ),
        action=ActionChannelConfig(
            channel_id="joint_state_vector",
            dataset_path="joint_action/vector",
            semantics="observed_robot_state_not_commanded_action",
        ),
        grippers=(
            GripperChannelConfig(
                channel_id="left_gripper",
                dataset_path="joint_action/left_gripper",
                open_when="at_or_above",
                open_threshold=0.8,
                closed_threshold=0.2,
                semantics_source=semantics_source,
            ),
            GripperChannelConfig(
                channel_id="right_gripper",
                dataset_path="joint_action/right_gripper",
                open_when="at_or_above",
                open_threshold=0.8,
                closed_threshold=0.2,
                semantics_source=semantics_source,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class TrajectoryEvidenceLimits:
    """Hard allocation/read ceilings for one trajectory."""

    max_segments: int = 256
    max_cameras: int = 8
    max_gripper_channels: int = 8
    max_context_radius: int = 4
    max_evidence_items: int = 2_048
    max_unique_images: int = 512
    max_image_bytes: int = 8 * 1024 * 1024
    max_total_image_bytes: int = 128 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    max_comparison_pixels: int = 4_000_000
    max_action_elements_per_frame: int = 4_096
    gripper_scan_chunk_frames: int = 256
    max_gripper_scan_samples: int = 200_000
    max_gripper_transitions: int = 2_048
    max_hdf5_objects: int = 4_096
    max_hdf5_metadata_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class TrajectoryEvidenceBatchLimits:
    """Manifest-wide ceilings in addition to per-trajectory limits."""

    max_entries: int = 32
    max_total_evidence_items: int = 16_384
    max_total_comparisons: int = 4_096
    max_total_images: int = 4_096
    max_total_image_bytes: int = 512 * 1024 * 1024
    max_total_gripper_scan_samples: int = 2_000_000
    max_total_gripper_transitions: int = 16_384

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SegmentBoundary:
    """Validated inclusive frame span recovered from a normalized segment."""

    segment_id: str
    segment_index: int
    start_inclusive: int
    end_inclusive: int


@dataclass(frozen=True, slots=True)
class _KeyframeSelection:
    segment_id: str
    segment_index: int
    frame_index: int
    temporal_role: str
    action_start: int
    action_end: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DetectedGripperTransition:
    channel_id: str
    dataset_path: str
    from_state: str
    to_state: str
    before_frame_index: int
    after_frame_index: int


@dataclass(frozen=True, slots=True)
class VisualImagePayload:
    """Ephemeral image bytes; deliberately absent from JSON serialization."""

    image_ref_id: str
    sha256: str
    media_type: str
    data: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if _IMAGE_REF_ID_RE.fullmatch(self.image_ref_id) is None:
            raise TrajectoryEvidenceError("image payload has an invalid image_ref_id")
        _require_sha256(self.sha256, path="image_payload.sha256")
        _require_string(self.media_type, path="image_payload.media_type")
        if not isinstance(self.data, bytes) or not self.data:
            raise TrajectoryEvidenceError("image_payload.data: must be non-empty bytes")
        if hashlib.sha256(self.data).hexdigest() != self.sha256:
            raise TrajectoryEvidenceError("image payload hash does not match its bytes")


class VisualEvidenceItemV1:
    """Strict JSON boundary for one camera/segment/temporal-role item."""

    SCHEMA = "roboharn_evo/visual_evidence_item/v1"
    SCHEMA_VERSION = 1
    __slots__ = ("_payload",)

    _KEYS = (
        "schema",
        "schema_version",
        "evidence_id",
        "trajectory_id",
        "segment_id",
        "segment_index",
        "frame",
        "camera",
        "temporal_role",
        "action_range",
        "gripper_states",
        "image",
        "evidence_source",
        "modalities",
        "confidence",
        "extraction_provenance",
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        copied = copy.deepcopy(dict(payload))
        _canonical_json(copied)
        self._validate(copied)
        self._payload = copied

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VisualEvidenceItemV1:
        return cls(payload)

    @property
    def evidence_id(self) -> str:
        return str(self._payload["evidence_id"])

    @property
    def image_ref_id(self) -> str:
        return str(self._payload["image"]["image_ref_id"])

    @property
    def trajectory_id(self) -> str:
        return str(self._payload["trajectory_id"])

    @property
    def segment_id(self) -> str:
        return str(self._payload["segment_id"])

    @property
    def camera(self) -> str:
        return str(self._payload["camera"])

    @property
    def temporal_role(self) -> str:
        return str(self._payload["temporal_role"])

    @property
    def frame_index(self) -> int:
        return int(self._payload["frame"]["index"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        _require_exact_keys(payload, VisualEvidenceItemV1._KEYS, path="visual_item")
        if payload["schema"] not in {VisualEvidenceItemV1.SCHEMA, "tcm/visual_evidence_item/v1"}:
            raise TrajectoryEvidenceError("visual_item.schema: unsupported schema")
        if payload["schema_version"] != 1 or isinstance(
            payload["schema_version"], bool
        ):
            raise TrajectoryEvidenceError(
                "visual_item.schema_version: unsupported major version"
            )
        evidence_id = _require_string(
            payload["evidence_id"], path="visual_item.evidence_id"
        )
        if _EVIDENCE_ID_RE.fullmatch(evidence_id) is None:
            raise TrajectoryEvidenceError("visual_item.evidence_id: invalid stable ID")
        trajectory_id = _require_safe_id(
            payload["trajectory_id"], path="visual_item.trajectory_id"
        )
        _require_string(payload["segment_id"], path="visual_item.segment_id")
        _require_int(payload["segment_index"], path="visual_item.segment_index")
        frame = _require_exact_keys(
            payload["frame"], ("index", "evidence_ref"), path="visual_item.frame"
        )
        frame_index = _require_int(frame["index"], path="visual_item.frame.index")
        frame_ref = _require_string(
            frame["evidence_ref"], path="visual_item.frame.evidence_ref"
        )
        match = _FRAME_REF_RE.fullmatch(frame_ref)
        if (
            match is None
            or match.group("trajectory") != trajectory_id
            or int(match.group("index")) != frame_index
        ):
            raise TrajectoryEvidenceError(
                "visual_item.frame.evidence_ref: does not match trajectory/frame"
            )
        _require_safe_id(payload["camera"], path="visual_item.camera")
        if payload["temporal_role"] not in _TEMPORAL_ROLES:
            raise TrajectoryEvidenceError("visual_item.temporal_role: unsupported role")
        VisualEvidenceItemV1._validate_action(payload["action_range"], frame_index)
        grippers = payload["gripper_states"]
        if not isinstance(grippers, list) or not grippers:
            raise TrajectoryEvidenceError(
                "visual_item.gripper_states: must be a non-empty array"
            )
        gripper_ids: set[str] = set()
        for index, gripper in enumerate(grippers):
            channel_id = VisualEvidenceItemV1._validate_gripper(
                gripper, frame_index, path=f"visual_item.gripper_states[{index}]"
            )
            if channel_id in gripper_ids:
                raise TrajectoryEvidenceError(
                    "visual_item.gripper_states: duplicate channel_id"
                )
            gripper_ids.add(channel_id)
        VisualEvidenceItemV1._validate_image(payload["image"], frame_index)
        if payload["evidence_source"] != "authorized_hdf5_expert_trajectory":
            raise TrajectoryEvidenceError(
                "visual_item.evidence_source: unsupported evidence source"
            )
        if payload["modalities"] != ["visual", "robot_state"]:
            raise TrajectoryEvidenceError(
                "visual_item.modalities: expected ['visual', 'robot_state']"
            )
        if payload["confidence"] not in _CONFIDENCE_LEVELS:
            raise TrajectoryEvidenceError("visual_item.confidence: unsupported value")
        provenance = _require_exact_keys(
            payload["extraction_provenance"],
            (
                "extractor",
                "version",
                "selection_policy",
                "selection_reasons",
                "adapter_configuration_sha256",
            ),
            path="visual_item.extraction_provenance",
        )
        if provenance["extractor"] != _EXTRACTOR_NAME:
            raise TrajectoryEvidenceError(
                "visual_item.extraction_provenance.extractor: unsupported extractor"
            )
        if provenance["version"] != _EXTRACTOR_VERSION:
            raise TrajectoryEvidenceError(
                "visual_item.extraction_provenance.version: unsupported version"
            )
        if provenance["selection_policy"] != _SELECTION_POLICY:
            raise TrajectoryEvidenceError(
                "visual_item.extraction_provenance.selection_policy: unsupported policy"
            )
        reasons = provenance["selection_reasons"]
        if (
            not isinstance(reasons, list)
            or not reasons
            or not all(isinstance(item, str) and item for item in reasons)
            or len(reasons) != len(set(reasons))
        ):
            raise TrajectoryEvidenceError(
                "visual_item.extraction_provenance.selection_reasons: "
                "must be unique non-empty strings"
            )
        _require_sha256(
            provenance["adapter_configuration_sha256"],
            path="visual_item.extraction_provenance.adapter_configuration_sha256",
        )

    @staticmethod
    def _validate_action(value: Any, frame_index: int) -> None:
        path = "visual_item.action_range"
        action = _require_exact_keys(
            value,
            (
                "source_type",
                "source_ref_id",
                "dataset_path",
                "start_inclusive",
                "end_inclusive",
                "semantics",
                "sample_frame_index",
                "sample_sha256",
                "sample_shape",
                "sample_dtype",
            ),
            path=path,
        )
        if action["source_type"] != "observed_robot_state":
            raise TrajectoryEvidenceError(
                f"{path}.source_type: must be observed_robot_state"
            )
        _require_string(action["source_ref_id"], path=f"{path}.source_ref_id")
        _require_hdf5_dataset_path(action["dataset_path"], path=f"{path}.dataset_path")
        start = _require_int(action["start_inclusive"], path=f"{path}.start_inclusive")
        end = _require_int(action["end_inclusive"], path=f"{path}.end_inclusive")
        if end < start:
            raise TrajectoryEvidenceError(f"{path}: invalid inclusive range")
        _require_string(action["semantics"], path=f"{path}.semantics")
        sample = _require_int(
            action["sample_frame_index"], path=f"{path}.sample_frame_index"
        )
        if sample != frame_index:
            raise TrajectoryEvidenceError(
                f"{path}.sample_frame_index: does not match visual frame"
            )
        _require_sha256(action["sample_sha256"], path=f"{path}.sample_sha256")
        shape = action["sample_shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        ):
            raise TrajectoryEvidenceError(f"{path}.sample_shape: invalid shape")
        _require_string(action["sample_dtype"], path=f"{path}.sample_dtype")

    @staticmethod
    def _validate_gripper(value: Any, frame_index: int, *, path: str) -> str:
        gripper = _require_exact_keys(
            value,
            (
                "channel_id",
                "source_type",
                "state",
                "raw_value",
                "state_ref",
                "semantics_ref",
                "confidence",
            ),
            path=path,
        )
        channel_id = _require_safe_id(gripper["channel_id"], path=f"{path}.channel_id")
        if gripper["source_type"] != "observed_robot_state":
            raise TrajectoryEvidenceError(
                f"{path}.source_type: must be observed_robot_state"
            )
        if gripper["state"] not in _GRIPPER_STATES:
            raise TrajectoryEvidenceError(f"{path}.state: unsupported state")
        _require_finite_number(gripper["raw_value"], path=f"{path}.raw_value")
        state_ref = _require_exact_keys(
            gripper["state_ref"],
            ("source_ref_id", "dataset_path", "frame_index"),
            path=f"{path}.state_ref",
        )
        _require_string(
            state_ref["source_ref_id"], path=f"{path}.state_ref.source_ref_id"
        )
        _require_hdf5_dataset_path(
            state_ref["dataset_path"], path=f"{path}.state_ref.dataset_path"
        )
        if (
            _require_int(state_ref["frame_index"], path=f"{path}.state_ref.frame_index")
            != frame_index
        ):
            raise TrajectoryEvidenceError(
                f"{path}.state_ref.frame_index: does not match visual frame"
            )
        semantics_ref = _require_exact_keys(
            gripper["semantics_ref"],
            ("adapter_configuration_sha256", "channel_id", "semantics_source"),
            path=f"{path}.semantics_ref",
        )
        _require_sha256(
            semantics_ref["adapter_configuration_sha256"],
            path=f"{path}.semantics_ref.adapter_configuration_sha256",
        )
        if (
            _require_safe_id(
                semantics_ref["channel_id"],
                path=f"{path}.semantics_ref.channel_id",
            )
            != channel_id
        ):
            raise TrajectoryEvidenceError(
                f"{path}.semantics_ref.channel_id: does not match channel_id"
            )
        _require_string(
            semantics_ref["semantics_source"],
            path=f"{path}.semantics_ref.semantics_source",
        )
        if gripper["confidence"] not in _CONFIDENCE_LEVELS:
            raise TrajectoryEvidenceError(f"{path}.confidence: unsupported value")
        return channel_id

    @staticmethod
    def _validate_image(value: Any, frame_index: int) -> None:
        path = "visual_item.image"
        image = _require_exact_keys(
            value,
            (
                "source_type",
                "image_ref_id",
                "sha256",
                "byte_length",
                "media_type",
                "stable_ref",
            ),
            path=path,
        )
        if image["source_type"] != "observed_visual":
            raise TrajectoryEvidenceError(
                f"{path}.source_type: must be observed_visual"
            )
        image_ref_id = _require_string(
            image["image_ref_id"], path=f"{path}.image_ref_id"
        )
        if _IMAGE_REF_ID_RE.fullmatch(image_ref_id) is None:
            raise TrajectoryEvidenceError(f"{path}.image_ref_id: invalid stable ID")
        _require_sha256(image["sha256"], path=f"{path}.sha256")
        _require_int(image["byte_length"], path=f"{path}.byte_length", minimum=1)
        if image["media_type"] not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            raise TrajectoryEvidenceError(f"{path}.media_type: unsupported image type")
        stable_ref = _require_exact_keys(
            image["stable_ref"],
            ("source_ref_id", "dataset_path", "frame_index"),
            path=f"{path}.stable_ref",
        )
        _require_string(
            stable_ref["source_ref_id"], path=f"{path}.stable_ref.source_ref_id"
        )
        _require_hdf5_dataset_path(
            stable_ref["dataset_path"], path=f"{path}.stable_ref.dataset_path"
        )
        if (
            _require_int(
                stable_ref["frame_index"], path=f"{path}.stable_ref.frame_index"
            )
            != frame_index
        ):
            raise TrajectoryEvidenceError(
                f"{path}.stable_ref.frame_index: does not match visual frame"
            )


class VisualEvidenceComparisonV1:
    """Task-neutral before/after pixels and robot-state delta for one segment."""

    SCHEMA = "roboharn_evo/visual_evidence_comparison/v1"
    SCHEMA_VERSION = 1
    __slots__ = ("_payload",)

    _KEYS = (
        "schema",
        "schema_version",
        "comparison_id",
        "trajectory_id",
        "segment_id",
        "segment_index",
        "camera",
        "before_evidence_id",
        "after_evidence_id",
        "modalities",
        "visual_delta",
        "gripper_deltas",
        "confidence",
        "limitations",
    )
    _LIMITATIONS = frozenset(
        {
            "pixel_difference_is_non_semantic",
            "camera_or_scene_motion_may_contribute",
            "comparison_does_not_establish_causality",
        }
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        copied = copy.deepcopy(dict(payload))
        _canonical_json(copied)
        self._validate(copied)
        self._payload = copied

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VisualEvidenceComparisonV1:
        return cls(payload)

    @property
    def comparison_id(self) -> str:
        return str(self._payload["comparison_id"])

    @property
    def before_evidence_id(self) -> str:
        return str(self._payload["before_evidence_id"])

    @property
    def after_evidence_id(self) -> str:
        return str(self._payload["after_evidence_id"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        path = "visual_comparison"
        _require_exact_keys(payload, VisualEvidenceComparisonV1._KEYS, path=path)
        if payload["schema"] not in {VisualEvidenceComparisonV1.SCHEMA, "tcm/visual_evidence_comparison/v1"}:
            raise TrajectoryEvidenceError(f"{path}.schema: unsupported schema")
        if payload["schema_version"] != 1 or isinstance(
            payload["schema_version"], bool
        ):
            raise TrajectoryEvidenceError(
                f"{path}.schema_version: unsupported major version"
            )
        comparison_id = _require_string(
            payload["comparison_id"], path=f"{path}.comparison_id"
        )
        if re.fullmatch(r"vc_[0-9a-f]{24}", comparison_id) is None:
            raise TrajectoryEvidenceError(f"{path}.comparison_id: invalid stable ID")
        _require_safe_id(payload["trajectory_id"], path=f"{path}.trajectory_id")
        _require_string(payload["segment_id"], path=f"{path}.segment_id")
        _require_int(payload["segment_index"], path=f"{path}.segment_index")
        _require_safe_id(payload["camera"], path=f"{path}.camera")
        for key in ("before_evidence_id", "after_evidence_id"):
            evidence_id = _require_string(payload[key], path=f"{path}.{key}")
            if _EVIDENCE_ID_RE.fullmatch(evidence_id) is None:
                raise TrajectoryEvidenceError(f"{path}.{key}: invalid evidence ID")
        if payload["modalities"] != ["visual", "robot_state"]:
            raise TrajectoryEvidenceError(
                f"{path}.modalities: expected ['visual', 'robot_state']"
            )
        visual = _require_exact_keys(
            payload["visual_delta"],
            (
                "source_type",
                "metric",
                "before_shape",
                "after_shape",
                "shape_equal",
                "hash_changed",
                "normalized_mean_absolute_difference",
                "semantic_interpretation",
                "causal_attribution",
            ),
            path=f"{path}.visual_delta",
        )
        if visual["source_type"] != "observed_visual":
            raise TrajectoryEvidenceError(
                f"{path}.visual_delta.source_type: must be observed_visual"
            )
        if visual["metric"] != "normalized_rgb_mean_absolute_difference":
            raise TrajectoryEvidenceError(
                f"{path}.visual_delta.metric: unsupported metric"
            )
        for key in ("before_shape", "after_shape"):
            shape = visual[key]
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in shape
                )
            ):
                raise TrajectoryEvidenceError(
                    f"{path}.visual_delta.{key}: expected positive [H,W,C]"
                )
        if not isinstance(visual["shape_equal"], bool) or not isinstance(
            visual["hash_changed"], bool
        ):
            raise TrajectoryEvidenceError(
                f"{path}.visual_delta: shape_equal/hash_changed must be bool"
            )
        if visual["shape_equal"] != (visual["before_shape"] == visual["after_shape"]):
            raise TrajectoryEvidenceError(
                f"{path}.visual_delta.shape_equal: does not match shapes"
            )
        mad = visual["normalized_mean_absolute_difference"]
        if visual["shape_equal"]:
            value = _require_finite_number(
                mad,
                path=(f"{path}.visual_delta.normalized_mean_absolute_difference"),
            )
            if value < 0.0 or value > 1.0:
                raise TrajectoryEvidenceError(
                    f"{path}.visual_delta: normalized MAD must be in [0,1]"
                )
        elif mad is not None:
            raise TrajectoryEvidenceError(
                f"{path}.visual_delta: shape mismatch requires null MAD"
            )
        if visual["semantic_interpretation"] != "none":
            raise TrajectoryEvidenceError(
                f"{path}.visual_delta.semantic_interpretation: must be none"
            )
        if visual["causal_attribution"] is not False:
            raise TrajectoryEvidenceError(
                f"{path}.visual_delta.causal_attribution: must be false"
            )
        deltas = payload["gripper_deltas"]
        if not isinstance(deltas, list) or not deltas:
            raise TrajectoryEvidenceError(
                f"{path}.gripper_deltas: must be a non-empty array"
            )
        channels: set[str] = set()
        for index, raw_delta in enumerate(deltas):
            delta_path = f"{path}.gripper_deltas[{index}]"
            delta = _require_exact_keys(
                raw_delta,
                (
                    "channel_id",
                    "source_type",
                    "before_state",
                    "after_state",
                    "before_raw_value",
                    "after_raw_value",
                    "raw_delta",
                    "state_changed",
                    "supporting_evidence_refs",
                ),
                path=delta_path,
            )
            channel_id = _require_safe_id(
                delta["channel_id"], path=f"{delta_path}.channel_id"
            )
            if channel_id in channels:
                raise TrajectoryEvidenceError(
                    f"{path}.gripper_deltas: duplicate channel_id"
                )
            channels.add(channel_id)
            if delta["source_type"] != "observed_robot_state":
                raise TrajectoryEvidenceError(
                    f"{delta_path}.source_type: must be observed_robot_state"
                )
            if (
                delta["before_state"] not in _GRIPPER_STATES
                or delta["after_state"] not in _GRIPPER_STATES
            ):
                raise TrajectoryEvidenceError(
                    f"{delta_path}: unsupported gripper state"
                )
            before_value = _require_finite_number(
                delta["before_raw_value"], path=f"{delta_path}.before_raw_value"
            )
            after_value = _require_finite_number(
                delta["after_raw_value"], path=f"{delta_path}.after_raw_value"
            )
            raw_change = _require_finite_number(
                delta["raw_delta"], path=f"{delta_path}.raw_delta"
            )
            if not math.isclose(
                raw_change, after_value - before_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise TrajectoryEvidenceError(
                    f"{delta_path}.raw_delta: inconsistent with before/after"
                )
            if delta["state_changed"] is not (
                delta["before_state"] != delta["after_state"]
            ):
                raise TrajectoryEvidenceError(
                    f"{delta_path}.state_changed: inconsistent with states"
                )
            if delta["supporting_evidence_refs"] != [
                payload["before_evidence_id"],
                payload["after_evidence_id"],
            ]:
                raise TrajectoryEvidenceError(
                    f"{delta_path}.supporting_evidence_refs: invalid refs"
                )
        if payload["confidence"] not in _CONFIDENCE_LEVELS:
            raise TrajectoryEvidenceError(f"{path}.confidence: unsupported value")
        limitations = payload["limitations"]
        if (
            not isinstance(limitations, list)
            or set(limitations) != VisualEvidenceComparisonV1._LIMITATIONS
            or len(limitations) != len(VisualEvidenceComparisonV1._LIMITATIONS)
        ):
            raise TrajectoryEvidenceError(
                f"{path}.limitations: required non-semantic/non-causal limits missing"
            )


class GripperStateTransitionV1:
    """A stable open/closed transition found by a bounded low-dimensional scan."""

    SCHEMA = "roboharn_evo/gripper_state_transition/v1"
    SCHEMA_VERSION = 1
    __slots__ = ("_payload",)

    _KEYS = (
        "schema",
        "schema_version",
        "transition_id",
        "trajectory_id",
        "channel_id",
        "source_type",
        "from_state",
        "to_state",
        "before_segment_id",
        "after_segment_id",
        "before_state_ref",
        "after_state_ref",
        "supporting_evidence_refs",
        "confidence",
        "scan_provenance",
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        copied = copy.deepcopy(dict(payload))
        _canonical_json(copied)
        self._validate(copied)
        self._payload = copied

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GripperStateTransitionV1:
        return cls(payload)

    @property
    def transition_id(self) -> str:
        return str(self._payload["transition_id"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        path = "gripper_transition"
        _require_exact_keys(payload, GripperStateTransitionV1._KEYS, path=path)
        if payload["schema"] not in {GripperStateTransitionV1.SCHEMA, "tcm/gripper_state_transition/v1"}:
            raise TrajectoryEvidenceError(f"{path}.schema: unsupported schema")
        if payload["schema_version"] != 1 or isinstance(
            payload["schema_version"], bool
        ):
            raise TrajectoryEvidenceError(
                f"{path}.schema_version: unsupported major version"
            )
        transition_id = _require_string(
            payload["transition_id"], path=f"{path}.transition_id"
        )
        if re.fullmatch(r"gst_[0-9a-f]{24}", transition_id) is None:
            raise TrajectoryEvidenceError(f"{path}.transition_id: invalid stable ID")
        _require_safe_id(payload["trajectory_id"], path=f"{path}.trajectory_id")
        channel_id = _require_safe_id(payload["channel_id"], path=f"{path}.channel_id")
        if payload["source_type"] != "observed_robot_state":
            raise TrajectoryEvidenceError(
                f"{path}.source_type: must be observed_robot_state"
            )
        if payload["from_state"] not in {"open", "closed"} or payload[
            "to_state"
        ] not in {"open", "closed"}:
            raise TrajectoryEvidenceError(
                f"{path}: transitions require determinate open/closed states"
            )
        if payload["from_state"] == payload["to_state"]:
            raise TrajectoryEvidenceError(f"{path}: transition states must differ")
        _require_string(payload["before_segment_id"], path=f"{path}.before_segment_id")
        _require_string(payload["after_segment_id"], path=f"{path}.after_segment_id")
        before = _validate_transition_state_ref(
            payload["before_state_ref"], path=f"{path}.before_state_ref"
        )
        after = _validate_transition_state_ref(
            payload["after_state_ref"], path=f"{path}.after_state_ref"
        )
        if before["channel_id"] != channel_id or after["channel_id"] != channel_id:
            raise TrajectoryEvidenceError(
                f"{path}: state refs do not match transition channel"
            )
        if (
            before["state"] != payload["from_state"]
            or after["state"] != payload["to_state"]
        ):
            raise TrajectoryEvidenceError(
                f"{path}: state refs do not match transition states"
            )
        if before["frame_index"] >= after["frame_index"]:
            raise TrajectoryEvidenceError(
                f"{path}: before frame must precede after frame"
            )
        refs = _require_exact_keys(
            payload["supporting_evidence_refs"],
            ("before", "after"),
            path=f"{path}.supporting_evidence_refs",
        )
        for role in ("before", "after"):
            values = refs[role]
            if (
                not isinstance(values, list)
                or not values
                or not all(
                    isinstance(value, str)
                    and _EVIDENCE_ID_RE.fullmatch(value) is not None
                    for value in values
                )
                or len(values) != len(set(values))
            ):
                raise TrajectoryEvidenceError(
                    f"{path}.supporting_evidence_refs.{role}: invalid refs"
                )
        if payload["confidence"] not in _CONFIDENCE_LEVELS:
            raise TrajectoryEvidenceError(f"{path}.confidence: unsupported value")
        provenance = _require_exact_keys(
            payload["scan_provenance"],
            (
                "mode",
                "chunk_frames",
                "adapter_configuration_sha256",
            ),
            path=f"{path}.scan_provenance",
        )
        if provenance["mode"] != _GRIPPER_SCAN_MODE:
            raise TrajectoryEvidenceError(
                f"{path}.scan_provenance.mode: unsupported mode"
            )
        _require_int(
            provenance["chunk_frames"],
            path=f"{path}.scan_provenance.chunk_frames",
            minimum=1,
        )
        _require_sha256(
            provenance["adapter_configuration_sha256"],
            path=f"{path}.scan_provenance.adapter_configuration_sha256",
        )


def _validate_transition_state_ref(value: Any, *, path: str) -> dict[str, Any]:
    result = _require_exact_keys(
        value,
        (
            "source_ref_id",
            "dataset_path",
            "channel_id",
            "frame_index",
            "state",
        ),
        path=path,
    )
    _require_string(result["source_ref_id"], path=f"{path}.source_ref_id")
    _require_hdf5_dataset_path(result["dataset_path"], path=f"{path}.dataset_path")
    _require_safe_id(result["channel_id"], path=f"{path}.channel_id")
    _require_int(result["frame_index"], path=f"{path}.frame_index")
    if result["state"] not in {"open", "closed"}:
        raise TrajectoryEvidenceError(f"{path}.state: must be open or closed")
    return result


def _validate_gripper_state_scan(value: Any) -> dict[str, Any]:
    path = "visual_bundle.gripper_state_scan"
    scan = _require_exact_keys(
        value,
        (
            "schema",
            "schema_version",
            "source_type",
            "source_ref_id",
            "mode",
            "chunk_frames",
            "frame_count",
            "channel_count",
            "sample_count",
            "channels",
            "adapter_configuration_sha256",
        ),
        path=path,
    )
    if scan["schema"] not in {"roboharn_evo/gripper_state_scan/v1", "tcm/gripper_state_scan/v1"}:
        raise TrajectoryEvidenceError(f"{path}.schema: unsupported schema")
    if scan["schema_version"] != 1 or isinstance(scan["schema_version"], bool):
        raise TrajectoryEvidenceError(
            f"{path}.schema_version: unsupported major version"
        )
    if scan["source_type"] != "observed_robot_state":
        raise TrajectoryEvidenceError(
            f"{path}.source_type: must be observed_robot_state"
        )
    _require_string(scan["source_ref_id"], path=f"{path}.source_ref_id")
    if scan["mode"] != _GRIPPER_SCAN_MODE:
        raise TrajectoryEvidenceError(f"{path}.mode: unsupported scan mode")
    _require_int(scan["chunk_frames"], path=f"{path}.chunk_frames", minimum=1)
    frame_count = _require_int(
        scan["frame_count"], path=f"{path}.frame_count", minimum=1
    )
    channel_count = _require_int(
        scan["channel_count"], path=f"{path}.channel_count", minimum=1
    )
    sample_count = _require_int(
        scan["sample_count"], path=f"{path}.sample_count", minimum=1
    )
    channels = scan["channels"]
    if not isinstance(channels, list) or len(channels) != channel_count:
        raise TrajectoryEvidenceError(f"{path}.channels: channel count mismatch")
    seen: set[str] = set()
    for index, raw_channel in enumerate(channels):
        channel_path = f"{path}.channels[{index}]"
        channel = _require_exact_keys(
            raw_channel,
            ("channel_id", "dataset_path", "state_counts", "transition_count"),
            path=channel_path,
        )
        channel_id = _require_safe_id(
            channel["channel_id"], path=f"{channel_path}.channel_id"
        )
        if channel_id in seen:
            raise TrajectoryEvidenceError(f"{path}.channels: duplicate channel_id")
        seen.add(channel_id)
        _require_hdf5_dataset_path(
            channel["dataset_path"], path=f"{channel_path}.dataset_path"
        )
        counts = _require_exact_keys(
            channel["state_counts"],
            ("open", "closed", "indeterminate"),
            path=f"{channel_path}.state_counts",
        )
        for state in ("open", "closed", "indeterminate"):
            _require_int(counts[state], path=f"{channel_path}.state_counts.{state}")
        if sum(counts.values()) != frame_count:
            raise TrajectoryEvidenceError(
                f"{channel_path}.state_counts: does not cover every frame"
            )
        _require_int(
            channel["transition_count"],
            path=f"{channel_path}.transition_count",
        )
    if sample_count != frame_count * channel_count:
        raise TrajectoryEvidenceError(f"{path}.sample_count: count mismatch")
    _require_sha256(
        scan["adapter_configuration_sha256"],
        path=f"{path}.adapter_configuration_sha256",
    )
    return scan


def _resolve_transition_support(
    evidence_ids: Sequence[str],
    *,
    by_id: Mapping[str, VisualEvidenceItemV1],
    expected_frame: int,
    expected_channel: str,
    expected_state: str,
    expected_segment: str,
    expected_dataset_path: str,
    expected_selection_reason: str,
) -> tuple[VisualEvidenceItemV1, ...]:
    result: list[VisualEvidenceItemV1] = []
    for evidence_id in evidence_ids:
        try:
            item = by_id[evidence_id]
        except KeyError as exc:
            raise TrajectoryEvidenceError(
                "visual_bundle.gripper_transitions: unresolved evidence ref"
            ) from exc
        payload = item.to_dict()
        states = {state["channel_id"]: state for state in payload["gripper_states"]}
        state = states.get(expected_channel)
        if (
            item.frame_index != expected_frame
            or item.segment_id != expected_segment
            or state is None
            or state["state"] != expected_state
            or state["state_ref"]["dataset_path"] != expected_dataset_path
            or expected_selection_reason
            not in payload["extraction_provenance"]["selection_reasons"]
        ):
            raise TrajectoryEvidenceError(
                "visual_bundle.gripper_transitions: evidence does not support state"
            )
        result.append(item)
    return tuple(result)


class VisualEvidenceBundleV1:
    """Strict, resolvable collection of sparse visual evidence metadata."""

    SCHEMA = "roboharn_evo/visual_evidence_bundle/v1"
    SCHEMA_VERSION = 1
    __slots__ = (
        "_by_comparison_id",
        "_by_gripper_transition_id",
        "_by_id",
        "_payload",
    )

    _KEYS = (
        "schema",
        "schema_version",
        "trajectory_id",
        "source",
        "frame_count",
        "segment_order",
        "adapter",
        "selection",
        "data_capabilities",
        "gripper_state_scan",
        "items",
        "comparisons",
        "gripper_transitions",
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        copied = copy.deepcopy(dict(payload))
        _canonical_json(copied)
        by_id, by_comparison_id, by_gripper_transition_id = self._validate(copied)
        self._payload = copied
        self._by_id = by_id
        self._by_comparison_id = by_comparison_id
        self._by_gripper_transition_id = by_gripper_transition_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VisualEvidenceBundleV1:
        return cls(payload)

    @property
    def trajectory_id(self) -> str:
        return str(self._payload["trajectory_id"])

    @property
    def items(self) -> tuple[VisualEvidenceItemV1, ...]:
        return tuple(
            self._by_id[item["evidence_id"]] for item in self._payload["items"]
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item["evidence_id"] for item in self._payload["items"])

    @property
    def comparisons(self) -> tuple[VisualEvidenceComparisonV1, ...]:
        return tuple(
            self._by_comparison_id[item["comparison_id"]]
            for item in self._payload["comparisons"]
        )

    @property
    def gripper_transitions(self) -> tuple[GripperStateTransitionV1, ...]:
        return tuple(
            self._by_gripper_transition_id[item["transition_id"]]
            for item in self._payload["gripper_transitions"]
        )

    def resolve(self, evidence_id: str) -> VisualEvidenceItemV1:
        try:
            return self._by_id[evidence_id]
        except KeyError as exc:
            raise TrajectoryEvidenceError(
                f"unknown visual evidence reference {evidence_id!r}"
            ) from exc

    def validate_refs(self, evidence_ids: Sequence[str]) -> None:
        if isinstance(evidence_ids, (str, bytes, bytearray)):
            raise TrajectoryEvidenceError("visual evidence refs must be an array")
        seen: set[str] = set()
        for value in evidence_ids:
            if not isinstance(value, str) or value not in self._by_id:
                raise TrajectoryEvidenceError(
                    f"unknown visual evidence reference {value!r}"
                )
            if value in seen:
                raise TrajectoryEvidenceError(
                    f"duplicate visual evidence reference {value!r}"
                )
            seen.add(value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    @staticmethod
    def _validate(
        payload: dict[str, Any],
    ) -> tuple[
        dict[str, VisualEvidenceItemV1],
        dict[str, VisualEvidenceComparisonV1],
        dict[str, GripperStateTransitionV1],
    ]:
        _require_exact_keys(payload, VisualEvidenceBundleV1._KEYS, path="visual_bundle")
        if payload["schema"] not in {VisualEvidenceBundleV1.SCHEMA, "tcm/visual_evidence_bundle/v1"}:
            raise TrajectoryEvidenceError("visual_bundle.schema: unsupported schema")
        if payload["schema_version"] != 1 or isinstance(
            payload["schema_version"], bool
        ):
            raise TrajectoryEvidenceError(
                "visual_bundle.schema_version: unsupported major version"
            )
        trajectory_id = _require_safe_id(
            payload["trajectory_id"], path="visual_bundle.trajectory_id"
        )
        source = _require_exact_keys(
            payload["source"],
            ("kind", "source_ref_id", "content_sha256"),
            path="visual_bundle.source",
        )
        if source["kind"] != "hdf5_expert_trajectory":
            raise TrajectoryEvidenceError("visual_bundle.source.kind: unsupported kind")
        source_ref_id = _require_string(
            source["source_ref_id"], path="visual_bundle.source.source_ref_id"
        )
        _require_sha256(
            source["content_sha256"], path="visual_bundle.source.content_sha256"
        )
        frame_count = _require_int(
            payload["frame_count"], path="visual_bundle.frame_count", minimum=1
        )
        segment_order = payload["segment_order"]
        if (
            not isinstance(segment_order, list)
            or not segment_order
            or not all(isinstance(item, str) and item for item in segment_order)
            or len(segment_order) != len(set(segment_order))
        ):
            raise TrajectoryEvidenceError(
                "visual_bundle.segment_order: must be unique non-empty strings"
            )
        adapter = _require_exact_keys(
            payload["adapter"],
            (
                "adapter_id",
                "adapter_version",
                "embodiment_id",
                "configuration_sha256",
            ),
            path="visual_bundle.adapter",
        )
        for key in ("adapter_id", "adapter_version", "embodiment_id"):
            _require_safe_id(adapter[key], path=f"visual_bundle.adapter.{key}")
        config_sha = _require_sha256(
            adapter["configuration_sha256"],
            path="visual_bundle.adapter.configuration_sha256",
        )
        selection = _require_exact_keys(
            payload["selection"],
            (
                "policy",
                "context_radius",
                "selected_frame_count",
                "selected_image_count",
                "evidence_item_count",
            ),
            path="visual_bundle.selection",
        )
        if selection["policy"] != _SELECTION_POLICY:
            raise TrajectoryEvidenceError("visual_bundle.selection.policy: unsupported")
        _require_int(
            selection["context_radius"], path="visual_bundle.selection.context_radius"
        )
        for key in (
            "selected_frame_count",
            "selected_image_count",
            "evidence_item_count",
        ):
            _require_int(
                selection[key], path=f"visual_bundle.selection.{key}", minimum=1
            )
        capabilities = _validate_capabilities(payload["data_capabilities"])
        gripper_scan = _validate_gripper_state_scan(payload["gripper_state_scan"])
        if gripper_scan["frame_count"] != frame_count:
            raise TrajectoryEvidenceError(
                "visual_bundle.gripper_state_scan.frame_count: count mismatch"
            )
        if gripper_scan["adapter_configuration_sha256"] != config_sha:
            raise TrajectoryEvidenceError(
                "visual_bundle.gripper_state_scan: adapter config mismatch"
            )
        if gripper_scan["source_ref_id"] != source_ref_id:
            raise TrajectoryEvidenceError(
                "visual_bundle.gripper_state_scan: source mismatch"
            )
        items = payload["items"]
        if not isinstance(items, list) or not items:
            raise TrajectoryEvidenceError("visual_bundle.items: must be non-empty")
        by_id: dict[str, VisualEvidenceItemV1] = {}
        image_bindings: dict[str, tuple[Any, ...]] = {}
        observed_roles: dict[tuple[str, str], set[str]] = {}
        frames: set[int] = set()
        for raw_item in items:
            item = VisualEvidenceItemV1.from_dict(raw_item)
            normalized = item.to_dict()
            if item.evidence_id in by_id:
                raise TrajectoryEvidenceError(
                    f"visual_bundle.items: duplicate evidence_id {item.evidence_id!r}"
                )
            if item.trajectory_id != trajectory_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: cross-trajectory evidence is forbidden"
                )
            if item.segment_id not in segment_order:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: item references an unknown segment"
                )
            if item.frame_index >= frame_count:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: item frame is outside the trajectory"
                )
            if item.camera not in capabilities["rgb"]["cameras"]:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: item camera is not a declared capability"
                )
            action = normalized["action_range"]
            if action["source_ref_id"] != source_ref_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: action source does not match bundle source"
                )
            for gripper in normalized["gripper_states"]:
                if gripper["state_ref"]["source_ref_id"] != source_ref_id:
                    raise TrajectoryEvidenceError(
                        "visual_bundle.items: gripper source does not match bundle source"
                    )
                if (
                    gripper["semantics_ref"]["adapter_configuration_sha256"]
                    != config_sha
                ):
                    raise TrajectoryEvidenceError(
                        "visual_bundle.items: gripper semantics config mismatch"
                    )
            image = normalized["image"]
            if image["stable_ref"]["source_ref_id"] != source_ref_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: image source does not match bundle source"
                )
            binding = (
                image["sha256"],
                image["byte_length"],
                image["media_type"],
                image["stable_ref"]["dataset_path"],
                image["stable_ref"]["frame_index"],
            )
            previous = image_bindings.setdefault(item.image_ref_id, binding)
            if previous != binding:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: image_ref_id resolves ambiguously"
                )
            if (
                normalized["extraction_provenance"]["adapter_configuration_sha256"]
                != config_sha
            ):
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: extraction config mismatch"
                )
            expected_evidence_id = _make_evidence_id(
                trajectory_id=trajectory_id,
                segment_id=item.segment_id,
                frame_index=item.frame_index,
                camera=item.camera,
                temporal_role=item.temporal_role,
                adapter_configuration_sha256=config_sha,
            )
            if item.evidence_id != expected_evidence_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: evidence_id is not content-stable"
                )
            expected_image_ref_id = _make_image_ref_id(
                source_ref_id=source_ref_id,
                dataset_path=image["stable_ref"]["dataset_path"],
                frame_index=item.frame_index,
            )
            if item.image_ref_id != expected_image_ref_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.items: image_ref_id is not locator-stable"
                )
            by_id[item.evidence_id] = item
            frames.add(item.frame_index)
            observed_roles.setdefault((item.segment_id, item.camera), set()).add(
                item.temporal_role
            )
        if selection["evidence_item_count"] != len(items):
            raise TrajectoryEvidenceError(
                "visual_bundle.selection.evidence_item_count: count mismatch"
            )
        if selection["selected_frame_count"] != len(frames):
            raise TrajectoryEvidenceError(
                "visual_bundle.selection.selected_frame_count: count mismatch"
            )
        if selection["selected_image_count"] != len(image_bindings):
            raise TrajectoryEvidenceError(
                "visual_bundle.selection.selected_image_count: count mismatch"
            )
        cameras = capabilities["rgb"]["cameras"]
        for segment_id in segment_order:
            for camera in cameras:
                roles = observed_roles.get((segment_id, camera), set())
                if not {"before", "after"}.issubset(roles):
                    raise TrajectoryEvidenceError(
                        "visual_bundle.items: every segment/camera requires before and after"
                    )
        comparisons = payload["comparisons"]
        if not isinstance(comparisons, list) or not comparisons:
            raise TrajectoryEvidenceError(
                "visual_bundle.comparisons: must be a non-empty array"
            )
        by_comparison_id: dict[str, VisualEvidenceComparisonV1] = {}
        covered_pairs: set[tuple[str, str]] = set()
        for raw_comparison in comparisons:
            comparison = VisualEvidenceComparisonV1.from_dict(raw_comparison)
            normalized = comparison.to_dict()
            if comparison.comparison_id in by_comparison_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.comparisons: duplicate comparison_id"
                )
            try:
                before = by_id[comparison.before_evidence_id]
                after = by_id[comparison.after_evidence_id]
            except KeyError as exc:
                raise TrajectoryEvidenceError(
                    "visual_bundle.comparisons: unresolved evidence ref"
                ) from exc
            expected = (
                normalized["trajectory_id"],
                normalized["segment_id"],
                normalized["segment_index"],
                normalized["camera"],
            )
            before_values = (
                before.trajectory_id,
                before.segment_id,
                before.to_dict()["segment_index"],
                before.camera,
            )
            after_values = (
                after.trajectory_id,
                after.segment_id,
                after.to_dict()["segment_index"],
                after.camera,
            )
            if (
                expected != before_values
                or expected != after_values
                or before.temporal_role != "before"
                or after.temporal_role != "after"
            ):
                raise TrajectoryEvidenceError(
                    "visual_bundle.comparisons: refs do not bind matching before/after items"
                )
            before_image = before.to_dict()["image"]
            after_image = after.to_dict()["image"]
            if normalized["visual_delta"]["hash_changed"] != (
                before_image["sha256"] != after_image["sha256"]
            ):
                raise TrajectoryEvidenceError(
                    "visual_bundle.comparisons: hash_changed does not match item hashes"
                )
            pair = (normalized["segment_id"], normalized["camera"])
            if pair in covered_pairs:
                raise TrajectoryEvidenceError(
                    "visual_bundle.comparisons: duplicate segment/camera pair"
                )
            expected_comparison_id = _make_comparison_id(
                trajectory_id=trajectory_id,
                segment_id=normalized["segment_id"],
                camera=normalized["camera"],
                before_evidence_id=comparison.before_evidence_id,
                after_evidence_id=comparison.after_evidence_id,
            )
            if comparison.comparison_id != expected_comparison_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.comparisons: comparison_id is not content-stable"
                )
            covered_pairs.add(pair)
            by_comparison_id[comparison.comparison_id] = comparison
        required_pairs = {
            (segment_id, camera) for segment_id in segment_order for camera in cameras
        }
        if covered_pairs != required_pairs:
            raise TrajectoryEvidenceError(
                "visual_bundle.comparisons: must cover every segment/camera pair"
            )
        transition_records = payload["gripper_transitions"]
        if not isinstance(transition_records, list):
            raise TrajectoryEvidenceError(
                "visual_bundle.gripper_transitions: must be an array"
            )
        by_gripper_transition_id: dict[str, GripperStateTransitionV1] = {}
        transition_counts: dict[str, int] = {
            channel["channel_id"]: 0 for channel in gripper_scan["channels"]
        }
        scan_channels = {
            channel["channel_id"]: channel for channel in gripper_scan["channels"]
        }
        for raw_transition in transition_records:
            transition = GripperStateTransitionV1.from_dict(raw_transition)
            normalized = transition.to_dict()
            transition_id = transition.transition_id
            if transition_id in by_gripper_transition_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.gripper_transitions: duplicate transition_id"
                )
            if normalized["trajectory_id"] != trajectory_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.gripper_transitions: cross-trajectory record"
                )
            channel_id = normalized["channel_id"]
            if channel_id not in transition_counts:
                raise TrajectoryEvidenceError(
                    "visual_bundle.gripper_transitions: unknown gripper channel"
                )
            before_ref = normalized["before_state_ref"]
            after_ref = normalized["after_state_ref"]
            for state_ref in (before_ref, after_ref):
                if state_ref["source_ref_id"] != source_ref_id:
                    raise TrajectoryEvidenceError(
                        "visual_bundle.gripper_transitions: source mismatch"
                    )
                if state_ref["frame_index"] >= frame_count:
                    raise TrajectoryEvidenceError(
                        "visual_bundle.gripper_transitions: frame out of range"
                    )
                if (
                    state_ref["dataset_path"]
                    != scan_channels[channel_id]["dataset_path"]
                ):
                    raise TrajectoryEvidenceError(
                        "visual_bundle.gripper_transitions: dataset path mismatch"
                    )
            if (
                normalized["scan_provenance"]["adapter_configuration_sha256"]
                != config_sha
            ):
                raise TrajectoryEvidenceError(
                    "visual_bundle.gripper_transitions: adapter config mismatch"
                )
            before_items = _resolve_transition_support(
                normalized["supporting_evidence_refs"]["before"],
                by_id=by_id,
                expected_frame=before_ref["frame_index"],
                expected_channel=channel_id,
                expected_state=normalized["from_state"],
                expected_segment=normalized["before_segment_id"],
                expected_dataset_path=before_ref["dataset_path"],
                expected_selection_reason=_gripper_transition_reason_from_values(
                    channel_id=channel_id,
                    from_state=normalized["from_state"],
                    to_state=normalized["to_state"],
                    before_frame_index=before_ref["frame_index"],
                    after_frame_index=after_ref["frame_index"],
                    phase="before",
                ),
            )
            after_items = _resolve_transition_support(
                normalized["supporting_evidence_refs"]["after"],
                by_id=by_id,
                expected_frame=after_ref["frame_index"],
                expected_channel=channel_id,
                expected_state=normalized["to_state"],
                expected_segment=normalized["after_segment_id"],
                expected_dataset_path=after_ref["dataset_path"],
                expected_selection_reason=_gripper_transition_reason_from_values(
                    channel_id=channel_id,
                    from_state=normalized["from_state"],
                    to_state=normalized["to_state"],
                    before_frame_index=before_ref["frame_index"],
                    after_frame_index=after_ref["frame_index"],
                    phase="after",
                ),
            )
            if {item.camera for item in before_items} != set(cameras) or {
                item.camera for item in after_items
            } != set(cameras):
                raise TrajectoryEvidenceError(
                    "visual_bundle.gripper_transitions: support must cover all cameras"
                )
            expected_transition_id = _make_gripper_transition_id(
                trajectory_id=trajectory_id,
                channel_id=channel_id,
                from_state=normalized["from_state"],
                to_state=normalized["to_state"],
                before_frame_index=before_ref["frame_index"],
                after_frame_index=after_ref["frame_index"],
                adapter_configuration_sha256=config_sha,
            )
            if transition_id != expected_transition_id:
                raise TrajectoryEvidenceError(
                    "visual_bundle.gripper_transitions: transition_id is not stable"
                )
            transition_counts[channel_id] += 1
            by_gripper_transition_id[transition_id] = transition
        declared_counts = {
            channel["channel_id"]: channel["transition_count"]
            for channel in gripper_scan["channels"]
        }
        if transition_counts != declared_counts:
            raise TrajectoryEvidenceError(
                "visual_bundle.gripper_state_scan: transition count mismatch"
            )
        return by_id, by_comparison_id, by_gripper_transition_id


@dataclass(frozen=True, slots=True)
class ExtractedTrajectoryEvidence:
    """Serializable metadata plus bounded, non-serializable image payloads."""

    bundle: VisualEvidenceBundleV1
    image_payloads: tuple[VisualImagePayload, ...]

    def __post_init__(self) -> None:
        by_ref: dict[str, VisualImagePayload] = {}
        for payload in self.image_payloads:
            if payload.image_ref_id in by_ref:
                raise TrajectoryEvidenceError(
                    f"duplicate image payload {payload.image_ref_id!r}"
                )
            by_ref[payload.image_ref_id] = payload
        required = {item.image_ref_id for item in self.bundle.items}
        if set(by_ref) != required:
            raise TrajectoryEvidenceError(
                "image payloads do not exactly cover bundle image references"
            )
        for item in self.bundle.items:
            image = item.to_dict()["image"]
            payload = by_ref[item.image_ref_id]
            if (
                payload.sha256 != image["sha256"]
                or payload.media_type != image["media_type"]
                or len(payload.data) != image["byte_length"]
            ):
                raise TrajectoryEvidenceError(
                    "image payload metadata does not match bundle item"
                )

    def image_payload(self, image_ref_id: str) -> VisualImagePayload:
        for payload in self.image_payloads:
            if payload.image_ref_id == image_ref_id:
                return payload
        raise TrajectoryEvidenceError(f"unknown image payload {image_ref_id!r}")

    def image_bytes_for_evidence(self, evidence_id: str) -> bytes:
        item = self.bundle.resolve(evidence_id)
        return self.image_payload(item.image_ref_id).data


@dataclass(frozen=True, slots=True)
class RMBenchTrajectoryEvidenceEntry:
    """One normalized importer entry paired with its sparse visual evidence."""

    entry_index: int
    trajectory_id: str
    normalized_trajectory: dict[str, Any]
    segments: tuple[dict[str, Any], ...]
    extracted: ExtractedTrajectoryEvidence


@dataclass(frozen=True, slots=True)
class RMBenchTrajectoryEvidenceBatch:
    """Manifest-safe A2a result; image payloads remain inside each entry."""

    entries: tuple[RMBenchTrajectoryEvidenceEntry, ...]
    abstentions: tuple[dict[str, Any], ...]
    input_manifest: dict[str, Any]
    resource_usage: dict[str, int]


def _parse_frame_ref(value: Any, *, trajectory_id: str, path: str) -> int:
    text = _require_string(value, path=path)
    match = _FRAME_REF_RE.fullmatch(text)
    if match is None or match.group("trajectory") != trajectory_id:
        raise TrajectoryEvidenceError(f"{path}: invalid trajectory-local frame ref")
    return int(match.group("index"))


def recover_segment_boundaries(
    segments: Sequence[Mapping[str, Any]],
    *,
    trajectory_id: str,
    frame_count: int,
    max_segments: int = 256,
) -> tuple[SegmentBoundary, ...]:
    """Recover and cross-check complete ordered spans without task frame constants."""

    _require_safe_id(trajectory_id, path="trajectory_id")
    _require_int(frame_count, path="frame_count", minimum=1)
    _require_int(max_segments, path="max_segments", minimum=1)
    if isinstance(segments, (str, bytes, bytearray)) or not isinstance(
        segments, Sequence
    ):
        raise TrajectoryEvidenceError("segments: must be an array")
    if not segments:
        raise TrajectoryEvidenceError("segments: must not be empty")
    if len(segments) > max_segments:
        raise TrajectoryEvidenceBudgetExceeded(
            f"segment count exceeds max_segments={max_segments}"
        )
    result: list[SegmentBoundary] = []
    expected_start = 0
    for ordinal, raw in enumerate(segments):
        if not isinstance(raw, Mapping):
            raise TrajectoryEvidenceError(f"segments[{ordinal}]: must be an object")
        segment = dict(raw)
        segment_id = _require_string(
            segment.get("segment_id"), path=f"segments[{ordinal}].segment_id"
        )
        if segment.get("trajectory_id") != trajectory_id:
            raise TrajectoryEvidenceError(
                f"segments[{ordinal}].trajectory_id: does not match trajectory"
            )
        segment_index = _require_int(
            segment.get("segment_index"), path=f"segments[{ordinal}].segment_index"
        )
        if segment_index != ordinal:
            raise TrajectoryEvidenceError(
                "segments: segment_index values must be ordered and contiguous"
            )
        derivation = segment.get("derivation")
        if not isinstance(derivation, Mapping):
            raise TrajectoryEvidenceError(
                f"segments[{ordinal}].derivation: must be an object"
            )
        frame_range = derivation.get("frame_range")
        if not isinstance(frame_range, Mapping):
            raise TrajectoryEvidenceError(
                f"segments[{ordinal}].derivation.frame_range: must be an object"
            )
        start = _require_int(
            frame_range.get("start_inclusive"),
            path=f"segments[{ordinal}].derivation.frame_range.start_inclusive",
        )
        end = _require_int(
            frame_range.get("end_inclusive"),
            path=f"segments[{ordinal}].derivation.frame_range.end_inclusive",
        )
        if start != expected_start or end < start or end >= frame_count:
            raise TrajectoryEvidenceError(
                "segments: frame ranges must cover the trajectory contiguously"
            )
        transition = segment.get("transition")
        if not isinstance(transition, Mapping):
            raise TrajectoryEvidenceError(
                f"segments[{ordinal}].transition: must be an object"
            )
        before_refs = transition.get("before_event_refs")
        after_refs = transition.get("after_event_refs")
        if not isinstance(before_refs, list) or not before_refs:
            raise TrajectoryEvidenceError(
                f"segments[{ordinal}].transition.before_event_refs: must not be empty"
            )
        if not isinstance(after_refs, list) or not after_refs:
            raise TrajectoryEvidenceError(
                f"segments[{ordinal}].transition.after_event_refs: must not be empty"
            )
        if (
            _parse_frame_ref(
                before_refs[0],
                trajectory_id=trajectory_id,
                path=f"segments[{ordinal}].transition.before_event_refs[0]",
            )
            != start
            or _parse_frame_ref(
                after_refs[-1],
                trajectory_id=trajectory_id,
                path=f"segments[{ordinal}].transition.after_event_refs[-1]",
            )
            != end
        ):
            raise TrajectoryEvidenceError(
                "segments: transition frame refs disagree with frame_range"
            )
        result.append(
            SegmentBoundary(
                segment_id=segment_id,
                segment_index=segment_index,
                start_inclusive=start,
                end_inclusive=end,
            )
        )
        expected_start = end + 1
    if expected_start != frame_count:
        raise TrajectoryEvidenceError(
            "segments: frame ranges do not cover the final trajectory frame"
        )
    return tuple(result)


def _select_keyframes(
    boundaries: Sequence[SegmentBoundary],
    *,
    frame_count: int,
    context_radius: int,
    gripper_transitions: Sequence[_DetectedGripperTransition] = (),
) -> tuple[_KeyframeSelection, ...]:
    selected: list[_KeyframeSelection] = []
    final_index = frame_count - 1
    for boundary in boundaries:
        entries: dict[tuple[str, int], set[str]] = {}

        def add(role: str, frame_index: int, reason: str) -> None:
            if 0 <= frame_index < frame_count:
                entries.setdefault((role, frame_index), set()).add(reason)

        def add_transition_frame(frame_index: int, reason: str) -> None:
            if not (boundary.start_inclusive <= frame_index <= boundary.end_inclusive):
                return
            if frame_index == boundary.start_inclusive:
                role = "before"
            elif frame_index == boundary.end_inclusive:
                role = "after"
            else:
                role = "context"
            add(role, frame_index, reason)

        add("before", boundary.start_inclusive, "segment_start")
        add("after", boundary.end_inclusive, "segment_end")
        if boundary.start_inclusive == 0:
            add("before", 0, "trajectory_initial")
        if boundary.end_inclusive == final_index:
            add("after", final_index, "trajectory_final")
        for offset in range(1, context_radius + 1):
            for frame_index, reason in (
                (boundary.start_inclusive - offset, "context_before_segment_start"),
                (boundary.start_inclusive + offset, "context_after_segment_start"),
                (boundary.end_inclusive - offset, "context_before_segment_end"),
                (boundary.end_inclusive + offset, "context_after_segment_end"),
            ):
                if frame_index in {
                    boundary.start_inclusive,
                    boundary.end_inclusive,
                }:
                    continue
                add("context", frame_index, reason)
        for transition in gripper_transitions:
            endpoints = (
                (
                    transition.before_frame_index,
                    _gripper_transition_reason(transition, "before"),
                ),
                (
                    transition.after_frame_index,
                    _gripper_transition_reason(transition, "after"),
                ),
            )
            for endpoint, endpoint_reason in endpoints:
                add_transition_frame(endpoint, endpoint_reason)
        role_rank = {"before": 0, "context": 1, "after": 2}
        for (role, frame_index), reasons in sorted(
            entries.items(), key=lambda value: (value[0][1], role_rank[value[0][0]])
        ):
            selected.append(
                _KeyframeSelection(
                    segment_id=boundary.segment_id,
                    segment_index=boundary.segment_index,
                    frame_index=frame_index,
                    temporal_role=role,
                    action_start=boundary.start_inclusive,
                    action_end=boundary.end_inclusive,
                    reasons=tuple(sorted(reasons)),
                )
            )
    return tuple(selected)


_ACTION_CHUNK_ROLE_TO_TEMPORAL_ROLE = {
    "subtask_before": "before",
    "subtask_after": "after",
    "task_feedback": "task_feedback",
    "action_chunk_before": "before",
    "action_chunk_during": "during",
    "action_chunk_after": "after",
    "gripper_transition_before": "before",
    "gripper_transition_after": "after",
    "release_after_settle": "settled",
}


def _action_chunk_keyframes(
    boundaries: Sequence[SegmentBoundary],
    *,
    frame_count: int,
    trajectory_id: str,
    source_ref_id: str,
    source_content_sha256: str,
    visual_adapter_configuration_sha256: str,
    action_chunk_bundle: Mapping[str, Any] | None,
) -> tuple[_KeyframeSelection, ...]:
    """Convert a validated action-chunk bundle into sparse visual roles.

    This adapter never interprets annotation text.  It accepts only the frame
    roles and chunk ownership emitted by the low-dimensional detector, then
    cross-checks them against the independently recovered segment boundaries.
    """

    if action_chunk_bundle is None:
        return ()
    # Import lazily: the action-chunk module reuses this module's safe HDF5
    # primitives, while this integration seam must still revalidate any
    # serialized bundle supplied by a caller.
    from .action_chunk_evidence import (
        ActionChunkBundleV1,
        ActionChunkEvidenceError,
    )

    try:
        bundle = ActionChunkBundleV1.from_dict(action_chunk_bundle).to_dict()
    except ActionChunkEvidenceError as exc:
        raise TrajectoryEvidenceError(
            "supplemental action chunk bundle failed schema validation"
        ) from exc
    if bundle.get("trajectory_id") != trajectory_id:
        raise TrajectoryEvidenceError(
            "supplemental action chunk bundle is for another trajectory"
        )
    if bundle.get("frame_count") != frame_count:
        raise TrajectoryEvidenceError(
            "supplemental action chunk bundle frame count does not match HDF5"
        )
    source = bundle.get("source")
    if not isinstance(source, Mapping):
        raise TrajectoryEvidenceError(
            "supplemental action chunk bundle source is missing"
        )
    if source.get("content_sha256") != source_content_sha256:
        raise TrajectoryEvidenceError(
            "supplemental action chunk bundle source hash does not match HDF5"
        )
    if source.get("source_ref_id") != source_ref_id:
        raise TrajectoryEvidenceError(
            "supplemental action chunk bundle source ref does not match HDF5"
        )
    adapter = bundle.get("adapter")
    if (
        not isinstance(adapter, Mapping)
        or adapter.get("gripper_adapter_configuration_sha256")
        != visual_adapter_configuration_sha256
    ):
        raise TrajectoryEvidenceError(
            "supplemental action and visual adapter configurations disagree"
        )
    boundary_by_id = {item.segment_id: item for item in boundaries}
    if bundle.get("segment_order") != [item.segment_id for item in boundaries]:
        raise TrajectoryEvidenceError(
            "supplemental action chunk segment order does not match annotations"
        )
    raw_chunks = bundle.get("action_chunks")
    raw_selected = bundle.get("selected_frames")
    if not isinstance(raw_chunks, list) or not isinstance(raw_selected, list):
        raise TrajectoryEvidenceError(
            "supplemental action chunk bundle lacks chunks or selected frames"
        )
    chunks: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw_chunks):
        if not isinstance(value, Mapping):
            raise TrajectoryEvidenceError(
                f"supplemental action_chunks[{index}] must be an object"
            )
        chunk = dict(value)
        chunk_id = _require_string(
            chunk.get("action_chunk_id"),
            path=f"supplemental.action_chunks[{index}].action_chunk_id",
        )
        segment_id = _require_string(
            chunk.get("segment_id"),
            path=f"supplemental.action_chunks[{index}].segment_id",
        )
        frame_range = chunk.get("frame_range")
        if segment_id not in boundary_by_id or not isinstance(frame_range, Mapping):
            raise TrajectoryEvidenceError(
                "supplemental action chunk references an unknown segment"
            )
        start = _require_int(
            frame_range.get("start_inclusive"),
            path=f"supplemental.action_chunks[{index}].start",
        )
        end = _require_int(
            frame_range.get("end_inclusive"),
            path=f"supplemental.action_chunks[{index}].end",
        )
        boundary = boundary_by_id[segment_id]
        if (
            start < boundary.start_inclusive
            or end > boundary.end_inclusive
            or end < start
            or chunk_id in chunks
        ):
            raise TrajectoryEvidenceError(
                "supplemental action chunk range or identity is invalid"
            )
        evidence_frame_refs = chunk.get("evidence_frame_refs")
        if not isinstance(evidence_frame_refs, Mapping):
            raise TrajectoryEvidenceError(
                "supplemental action chunk evidence roles are missing"
            )
        evidence_frames: dict[str, frozenset[int]] = {}
        for role in ("before", "during", "after", "settled"):
            refs = evidence_frame_refs.get(role)
            if not isinstance(refs, list):
                raise TrajectoryEvidenceError(
                    "supplemental action chunk evidence role is invalid"
                )
            evidence_frames[role] = frozenset(
                _parse_frame_ref(
                    ref,
                    trajectory_id=trajectory_id,
                    path=f"supplemental.action_chunks[{index}].evidence_frame_refs.{role}",
                )
                for ref in refs
            )
        chunks[chunk_id] = {
            "segment_id": segment_id,
            "start": start,
            "end": end,
            "evidence_frames": evidence_frames,
            "has_gripper_transition": bool(chunk.get("gripper_transition_refs")),
        }

    aggregated: dict[tuple[str, int, str], dict[str, Any]] = {}
    for index, value in enumerate(raw_selected):
        if not isinstance(value, Mapping):
            raise TrajectoryEvidenceError(
                f"supplemental selected_frames[{index}] must be an object"
            )
        frame_index = _require_int(
            value.get("frame_index"),
            path=f"supplemental.selected_frames[{index}].frame_index",
        )
        if frame_index >= frame_count:
            raise TrajectoryEvidenceError(
                "supplemental selected frame lies outside the trajectory"
            )
        roles = value.get("roles")
        chunk_ids = value.get("action_chunk_ids")
        segment_ids = value.get("segment_ids")
        if (
            not isinstance(roles, list)
            or not roles
            or not isinstance(chunk_ids, list)
            or not isinstance(segment_ids, list)
            or not segment_ids
        ):
            raise TrajectoryEvidenceError(
                "supplemental selected frame roles/ownership are invalid"
            )
        if any(chunk_id not in chunks for chunk_id in chunk_ids):
            raise TrajectoryEvidenceError(
                "supplemental selected frame cites an unknown action chunk"
            )
        for segment_id in segment_ids:
            if segment_id not in boundary_by_id:
                raise TrajectoryEvidenceError(
                    "supplemental selected frame cites an unknown segment"
                )
            boundary = boundary_by_id[segment_id]
            if not boundary.start_inclusive <= frame_index <= boundary.end_inclusive:
                raise TrajectoryEvidenceError(
                    "supplemental selected frame lies outside its segment"
                )
            role_chunk_ids: dict[str, tuple[str, ...]] = {}
            expected_selected_chunk_ids: set[str] = set()
            for role in roles:
                temporal_role = _ACTION_CHUNK_ROLE_TO_TEMPORAL_ROLE.get(role)
                if temporal_role is None:
                    raise TrajectoryEvidenceError(
                        f"supplemental selected frame has unsupported role {role!r}"
                    )
                source_role = {
                    "action_chunk_before": "before",
                    "action_chunk_during": "during",
                    "action_chunk_after": "after",
                    "gripper_transition_before": "before",
                    "gripper_transition_after": "after",
                    "release_after_settle": "settled",
                }.get(role)
                if source_role is None:
                    bound_chunk_ids: tuple[str, ...] = ()
                else:
                    bound_chunk_ids = tuple(
                        chunk_id
                        for chunk_id, chunk_value in chunks.items()
                        if chunk_value["segment_id"] == segment_id
                        and frame_index in chunk_value["evidence_frames"][source_role]
                        and (
                            not role.startswith("gripper_transition_")
                            or chunk_value["has_gripper_transition"]
                        )
                    )
                    if not bound_chunk_ids:
                        raise TrajectoryEvidenceError(
                            "supplemental selected role is not bound by a chunk evidence ref"
                        )
                    expected_selected_chunk_ids.update(bound_chunk_ids)
                role_chunk_ids[role] = bound_chunk_ids
            if set(chunk_ids) != expected_selected_chunk_ids:
                raise TrajectoryEvidenceError(
                    "supplemental selected frame chunk refs do not match role bindings"
                )

            for role in roles:
                temporal_role = _ACTION_CHUNK_ROLE_TO_TEMPORAL_ROLE[role]
                bound_chunk_ids = role_chunk_ids[role]
                local_chunks = [chunks[chunk_id] for chunk_id in bound_chunk_ids]
                action_start = (
                    min(item["start"] for item in local_chunks)
                    if local_chunks
                    else boundary.start_inclusive
                )
                action_end = (
                    max(item["end"] for item in local_chunks)
                    if local_chunks
                    else boundary.end_inclusive
                )
                key = (segment_id, frame_index, temporal_role)
                entry = aggregated.setdefault(
                    key,
                    {
                        "boundary": boundary,
                        "action_start": action_start,
                        "action_end": action_end,
                        "reasons": set(),
                    },
                )
                entry["action_start"] = min(entry["action_start"], action_start)
                entry["action_end"] = max(entry["action_end"], action_end)
                entry["reasons"].add(f"action_chunk_role:{role}")
                entry["reasons"].update(
                    f"action_chunk_ref:{chunk_id}" for chunk_id in bound_chunk_ids
                )

    role_rank = {
        "before": 0,
        "during": 1,
        "after": 2,
        "settled": 3,
        "task_feedback": 4,
        "context": 5,
    }
    result: list[_KeyframeSelection] = []
    for (segment_id, frame_index, temporal_role), value in sorted(
        aggregated.items(),
        key=lambda item: (
            item[1]["boundary"].segment_index,
            item[0][1],
            role_rank[item[0][2]],
        ),
    ):
        boundary = value["boundary"]
        result.append(
            _KeyframeSelection(
                segment_id=segment_id,
                segment_index=boundary.segment_index,
                frame_index=frame_index,
                temporal_role=temporal_role,
                action_start=value["action_start"],
                action_end=value["action_end"],
                reasons=tuple(sorted(value["reasons"])),
            )
        )
    return tuple(result)


def _merge_keyframe_selections(
    *groups: Sequence[_KeyframeSelection],
) -> tuple[_KeyframeSelection, ...]:
    merged: dict[tuple[str, int, str], dict[str, Any]] = {}
    for group in groups:
        for item in group:
            key = (item.segment_id, item.frame_index, item.temporal_role)
            value = merged.setdefault(
                key,
                {
                    "segment_index": item.segment_index,
                    "action_start": item.action_start,
                    "action_end": item.action_end,
                    "reasons": set(),
                },
            )
            if value["segment_index"] != item.segment_index:
                raise TrajectoryEvidenceError(
                    "merged keyframe selection has inconsistent segment index"
                )
            value["action_start"] = min(value["action_start"], item.action_start)
            value["action_end"] = max(value["action_end"], item.action_end)
            value["reasons"].update(item.reasons)
    role_rank = {
        "before": 0,
        "during": 1,
        "after": 2,
        "settled": 3,
        "task_feedback": 4,
        "context": 5,
    }
    return tuple(
        _KeyframeSelection(
            segment_id=segment_id,
            segment_index=value["segment_index"],
            frame_index=frame_index,
            temporal_role=temporal_role,
            action_start=value["action_start"],
            action_end=value["action_end"],
            reasons=tuple(sorted(value["reasons"])),
        )
        for (segment_id, frame_index, temporal_role), value in sorted(
            merged.items(),
            key=lambda item: (
                item[1]["segment_index"],
                item[0][1],
                role_rank[item[0][2]],
            ),
        )
    )


def _gripper_transition_reason(
    transition: _DetectedGripperTransition, phase: str
) -> str:
    return _gripper_transition_reason_from_values(
        channel_id=transition.channel_id,
        from_state=transition.from_state,
        to_state=transition.to_state,
        before_frame_index=transition.before_frame_index,
        after_frame_index=transition.after_frame_index,
        phase=phase,
    )


def _gripper_transition_reason_from_values(
    *,
    channel_id: str,
    from_state: str,
    to_state: str,
    before_frame_index: int,
    after_frame_index: int,
    phase: str,
) -> str:
    return (
        f"gripper_state_transition_{phase}:"
        f"{channel_id}:{from_state}_to_{to_state}:"
        f"{before_frame_index:06d}_{after_frame_index:06d}"
    )


def _make_evidence_id(
    *,
    trajectory_id: str,
    segment_id: str,
    frame_index: int,
    camera: str,
    temporal_role: str,
    adapter_configuration_sha256: str,
) -> str:
    digest = _sha256_json(
        {
            "trajectory_id": trajectory_id,
            "segment_id": segment_id,
            "frame_index": frame_index,
            "camera": camera,
            "temporal_role": temporal_role,
            "adapter_configuration_sha256": adapter_configuration_sha256,
        }
    )
    return f"ve_{digest[:24]}"


def _make_image_ref_id(
    *, source_ref_id: str, dataset_path: str, frame_index: int
) -> str:
    digest = _sha256_json(
        {
            "source_ref_id": source_ref_id,
            "dataset_path": dataset_path,
            "frame_index": frame_index,
        }
    )
    return f"img_{digest[:24]}"


def _make_comparison_id(
    *,
    trajectory_id: str,
    segment_id: str,
    camera: str,
    before_evidence_id: str,
    after_evidence_id: str,
) -> str:
    digest = _sha256_json(
        {
            "trajectory_id": trajectory_id,
            "segment_id": segment_id,
            "camera": camera,
            "before_evidence_id": before_evidence_id,
            "after_evidence_id": after_evidence_id,
        }
    )
    return f"vc_{digest[:24]}"


def _make_gripper_transition_id(
    *,
    trajectory_id: str,
    channel_id: str,
    from_state: str,
    to_state: str,
    before_frame_index: int,
    after_frame_index: int,
    adapter_configuration_sha256: str,
) -> str:
    digest = _sha256_json(
        {
            "trajectory_id": trajectory_id,
            "channel_id": channel_id,
            "from_state": from_state,
            "to_state": to_state,
            "before_frame_index": before_frame_index,
            "after_frame_index": after_frame_index,
            "adapter_configuration_sha256": adapter_configuration_sha256,
        }
    )
    return f"gst_{digest[:24]}"


def _load_h5py() -> Any:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise TrajectoryEvidenceError(
            "visual evidence extraction requires h5py; install roboharn-evo[expert]"
        ) from exc
    return h5py


def _safe_dataset(handle: Any, dataset_path: str, h5py: Any) -> Any:
    current = handle
    parts = PurePosixPath(dataset_path).parts
    for index, part in enumerate(parts):
        link = current.get(part, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise TrajectoryEvidenceError(
                f"HDF5 dataset {dataset_path!r} uses a non-local link"
            )
        obj = current.get(part)
        if index < len(parts) - 1:
            if not isinstance(obj, h5py.Group):
                raise TrajectoryEvidenceError(
                    f"HDF5 path component in {dataset_path!r} is not a group"
                )
            current = obj
        else:
            if not isinstance(obj, h5py.Dataset):
                raise TrajectoryEvidenceError(
                    f"HDF5 path {dataset_path!r} is not a dataset"
                )
            if bool(getattr(obj, "is_virtual", False)):
                raise TrajectoryEvidenceError(
                    f"HDF5 dataset {dataset_path!r} must not be virtual"
                )
            if obj.id.get_create_plist().get_external_count() > 0:
                raise TrajectoryEvidenceError(
                    f"HDF5 dataset {dataset_path!r} uses external storage"
                )
            return obj
    raise TrajectoryEvidenceError(f"HDF5 dataset {dataset_path!r} is unavailable")


def _dataset_catalog(
    handle: Any, h5py: Any, limits: TrajectoryEvidenceLimits
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    object_count = 0
    text_bytes = 0

    def walk(group: Any, prefix: str = "") -> None:
        nonlocal object_count, text_bytes
        for name in group.keys():
            object_count += 1
            if object_count > limits.max_hdf5_objects:
                raise TrajectoryEvidenceBudgetExceeded(
                    "HDF5 object count exceeds max_hdf5_objects"
                )
            full_name = f"{prefix}/{name}" if prefix else str(name)
            text_bytes += len(full_name.encode("utf-8", "surrogatepass"))
            if text_bytes > limits.max_hdf5_metadata_bytes:
                raise TrajectoryEvidenceBudgetExceeded(
                    "HDF5 metadata exceeds max_hdf5_metadata_bytes"
                )
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise TrajectoryEvidenceError("HDF5 contains a non-local link")
            obj = group.get(name)
            if isinstance(obj, h5py.Group):
                walk(obj, full_name)
            elif isinstance(obj, h5py.Dataset):
                if bool(getattr(obj, "is_virtual", False)):
                    raise TrajectoryEvidenceError("HDF5 contains a virtual dataset")
                if obj.id.get_create_plist().get_external_count() > 0:
                    raise TrajectoryEvidenceError(
                        "HDF5 contains an externally stored dataset"
                    )
                shape = tuple(int(item) for item in obj.shape)
                result[full_name] = {
                    "shape": shape,
                    "dtype": str(obj.dtype),
                    "dtype_kind": obj.dtype.kind,
                    "dtype_itemsize": int(obj.dtype.itemsize),
                }
                text_bytes += len(str(obj.dtype).encode("utf-8", "surrogatepass"))
                if text_bytes > limits.max_hdf5_metadata_bytes:
                    raise TrajectoryEvidenceBudgetExceeded(
                        "HDF5 metadata exceeds max_hdf5_metadata_bytes"
                    )
            else:
                raise TrajectoryEvidenceError("HDF5 contains an unknown object type")

    walk(handle)
    return result


def _validate_numeric_timeseries(
    dataset: Any,
    *,
    dataset_path: str,
    frame_count: int | None,
    max_elements_per_frame: int,
    scalar: bool,
) -> int:
    if dataset.dtype.kind not in {"i", "u", "f"}:
        raise TrajectoryEvidenceError(
            f"HDF5 dataset {dataset_path!r} must have a numeric dtype"
        )
    shape = tuple(int(value) for value in dataset.shape)
    if not shape or shape[0] <= 0:
        raise TrajectoryEvidenceError(
            f"HDF5 dataset {dataset_path!r} must be a non-empty time series"
        )
    if frame_count is not None and shape[0] != frame_count:
        raise TrajectoryEvidenceError(
            f"HDF5 dataset {dataset_path!r} frame count mismatch"
        )
    elements = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    if elements <= 0 or elements > max_elements_per_frame:
        raise TrajectoryEvidenceBudgetExceeded(
            f"HDF5 dataset {dataset_path!r} exceeds per-frame element budget"
        )
    if scalar and elements != 1:
        raise TrajectoryEvidenceError(
            f"HDF5 gripper dataset {dataset_path!r} must be scalar per frame"
        )
    return shape[0]


def _validate_camera_timeseries(
    dataset: Any,
    *,
    dataset_path: str,
    frame_count: int,
    limits: TrajectoryEvidenceLimits,
) -> int:
    shape = tuple(int(value) for value in dataset.shape)
    if not shape or shape[0] != frame_count:
        raise TrajectoryEvidenceError(
            f"HDF5 camera dataset {dataset_path!r} frame count mismatch"
        )
    if dataset.dtype.kind == "S" and len(shape) == 1:
        source_bytes = int(dataset.dtype.itemsize)
    elif dataset.dtype.kind == "u" and dataset.dtype.itemsize == 1:
        per_frame_shape = shape[1:]
        if len(per_frame_shape) == 1:
            source_bytes = per_frame_shape[0]
        elif len(per_frame_shape) == 2:
            source_bytes = int(np.prod(per_frame_shape, dtype=np.int64))
        elif len(per_frame_shape) == 3 and per_frame_shape[-1] in {1, 3, 4}:
            source_bytes = int(np.prod(per_frame_shape, dtype=np.int64))
        else:
            raise TrajectoryEvidenceError(
                f"HDF5 camera dataset {dataset_path!r} has unsupported uint8 shape"
            )
    else:
        raise TrajectoryEvidenceError(
            f"HDF5 camera dataset {dataset_path!r} must contain fixed bytes or uint8 frames"
        )
    if source_bytes <= 0 or source_bytes > limits.max_image_bytes:
        raise TrajectoryEvidenceBudgetExceeded(
            f"HDF5 camera dataset {dataset_path!r} exceeds max_image_bytes"
        )
    return source_bytes


def _read_dataset_element(dataset: Any, frame_index: int) -> Any:
    """Single-frame seam retained for deterministic streaming tests."""

    return dataset[frame_index]


def _read_dataset_slice(dataset: Any, start: int, stop: int) -> Any:
    """Bounded low-dimensional slice seam retained for streaming tests."""

    return dataset[start:stop]


def _scan_gripper_channels(
    *,
    config: TrajectoryEvidenceAdapterConfig,
    gripper_datasets: Mapping[str, Any],
    frame_count: int,
    limits: TrajectoryEvidenceLimits,
) -> tuple[tuple[_DetectedGripperTransition, ...], list[dict[str, Any]]]:
    sample_count = frame_count * len(config.grippers)
    if sample_count > limits.max_gripper_scan_samples:
        raise TrajectoryEvidenceBudgetExceeded(
            "gripper scan exceeds max_gripper_scan_samples"
        )
    transitions: list[_DetectedGripperTransition] = []
    channel_summaries: list[dict[str, Any]] = []
    for spec in config.grippers:
        dataset = gripper_datasets[spec.channel_id]
        counts = {"open": 0, "closed": 0, "indeterminate": 0}
        last_stable_state: str | None = None
        last_stable_frame: int | None = None
        channel_transition_count = 0
        for start in range(0, frame_count, limits.gripper_scan_chunk_frames):
            stop = min(frame_count, start + limits.gripper_scan_chunk_frames)
            chunk = np.asarray(_read_dataset_slice(dataset, start, stop)).reshape(-1)
            if chunk.size != stop - start:
                raise TrajectoryEvidenceError(
                    f"gripper {spec.channel_id!r} returned an invalid scan chunk"
                )
            for offset, raw in enumerate(chunk):
                frame_index = start + offset
                value = _require_finite_number(
                    raw.item(), path=f"gripper.{spec.channel_id}.scan_value"
                )
                state = spec.classify(value)
                counts[state] += 1
                if state == "indeterminate":
                    continue
                if last_stable_state is None:
                    last_stable_state = state
                    last_stable_frame = frame_index
                    continue
                if state != last_stable_state:
                    assert last_stable_frame is not None
                    transitions.append(
                        _DetectedGripperTransition(
                            channel_id=spec.channel_id,
                            dataset_path=spec.dataset_path,
                            from_state=last_stable_state,
                            to_state=state,
                            before_frame_index=last_stable_frame,
                            after_frame_index=frame_index,
                        )
                    )
                    channel_transition_count += 1
                    if len(transitions) > limits.max_gripper_transitions:
                        raise TrajectoryEvidenceBudgetExceeded(
                            "gripper scan exceeds max_gripper_transitions"
                        )
                    last_stable_state = state
                last_stable_frame = frame_index
        channel_summaries.append(
            {
                "channel_id": spec.channel_id,
                "dataset_path": spec.dataset_path,
                "state_counts": counts,
                "transition_count": channel_transition_count,
            }
        )
    transitions.sort(
        key=lambda item: (
            item.before_frame_index,
            item.after_frame_index,
            item.channel_id,
        )
    )
    return tuple(transitions), channel_summaries


def _hash_array_value(value: Any) -> tuple[str, list[int], str]:
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u", "f"}:
        raise TrajectoryEvidenceError("robot state sample must be numeric")
    contiguous = np.ascontiguousarray(array)
    header = _canonical_json(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)}
    )
    digest = hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()
    return digest, list(contiguous.shape), str(contiguous.dtype)


def _image_media_type(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    raise TrajectoryEvidenceError("camera frame is not a supported encoded image")


def _validate_encoded_image(payload: bytes, limits: TrajectoryEvidenceLimits) -> str:
    media_type = _image_media_type(payload)
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise TrajectoryEvidenceError(
            "visual evidence extraction requires Pillow"
        ) from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                if (
                    width <= 0
                    or height <= 0
                    or width * height > limits.max_image_pixels
                ):
                    raise TrajectoryEvidenceBudgetExceeded(
                        "decoded image exceeds max_image_pixels"
                    )
                image.verify()
    except TrajectoryEvidenceError:
        raise
    except (OSError, ValueError, Warning) as exc:
        raise TrajectoryEvidenceError("camera frame is not a valid image") from exc
    return media_type


def _encode_image_element(
    value: Any,
    *,
    limits: TrajectoryEvidenceLimits,
) -> tuple[bytes, str]:
    if isinstance(value, (bytes, np.bytes_)):
        payload = bytes(value)
    else:
        array = np.asarray(value)
        if array.dtype != np.uint8:
            raise TrajectoryEvidenceError("camera frame array must have dtype uint8")
        if array.ndim == 1:
            payload = np.ascontiguousarray(array).tobytes()
        elif array.ndim in {2, 3}:
            if array.size > limits.max_image_pixels * 4:
                raise TrajectoryEvidenceBudgetExceeded(
                    "raw camera frame exceeds max_image_pixels"
                )
            try:
                from PIL import Image
            except ModuleNotFoundError as exc:
                raise TrajectoryEvidenceError(
                    "raw RGB evidence extraction requires Pillow"
                ) from exc
            normalized = np.ascontiguousarray(array)
            if normalized.ndim == 3 and normalized.shape[-1] == 1:
                normalized = normalized[..., 0]
            output = io.BytesIO()
            Image.fromarray(normalized).save(output, format="PNG", optimize=False)
            payload = output.getvalue()
        else:
            raise TrajectoryEvidenceError("camera frame array rank is unsupported")
    if not payload or len(payload) > limits.max_image_bytes:
        raise TrajectoryEvidenceBudgetExceeded(
            "encoded camera frame exceeds max_image_bytes"
        )
    return payload, _validate_encoded_image(payload, limits)


def _decode_rgb_for_comparison(
    payload: bytes, limits: TrajectoryEvidenceLimits
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise TrajectoryEvidenceError(
            "visual evidence comparison requires Pillow"
        ) from exc
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if width * height > limits.max_comparison_pixels:
                raise TrajectoryEvidenceBudgetExceeded(
                    "before/after comparison exceeds max_comparison_pixels"
                )
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except TrajectoryEvidenceError:
        raise
    except (OSError, ValueError) as exc:
        raise TrajectoryEvidenceError(
            "camera frame cannot be decoded for comparison"
        ) from exc


def _normalized_rgb_mad(
    before: bytes,
    after: bytes,
    limits: TrajectoryEvidenceLimits,
) -> tuple[list[int], list[int], float | None]:
    before_array = _decode_rgb_for_comparison(before, limits)
    after_array = _decode_rgb_for_comparison(after, limits)
    before_shape = list(before_array.shape)
    after_shape = list(after_array.shape)
    if before_shape != after_shape:
        return before_shape, after_shape, None
    # int16 avoids uint8 wraparound and bounds the working set to two sparse
    # keyframes.  The scalar is descriptive only; it carries no task semantics.
    difference = np.abs(before_array.astype(np.int16) - after_array.astype(np.int16))
    return before_shape, after_shape, float(difference.mean(dtype=np.float64) / 255.0)


def _validate_capabilities(value: Any) -> dict[str, Any]:
    capabilities = _require_exact_keys(
        value,
        ("rgb", "depth", "pointcloud", "object_pose"),
        path="visual_bundle.data_capabilities",
    )
    rgb = _require_exact_keys(
        capabilities["rgb"], ("available", "cameras"), path="capabilities.rgb"
    )
    if rgb["available"] is not True:
        raise TrajectoryEvidenceError("capabilities.rgb.available: must be true")
    cameras = rgb["cameras"]
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(item, str) and item for item in cameras)
        or len(cameras) != len(set(cameras))
    ):
        raise TrajectoryEvidenceError("capabilities.rgb.cameras: invalid camera list")
    depth = _require_exact_keys(
        capabilities["depth"],
        ("available", "dataset_paths"),
        path="capabilities.depth",
    )
    object_pose = _require_exact_keys(
        capabilities["object_pose"],
        ("available", "dataset_paths"),
        path="capabilities.object_pose",
    )
    for name, entry in (("depth", depth), ("object_pose", object_pose)):
        if not isinstance(entry["available"], bool):
            raise TrajectoryEvidenceError(
                f"capabilities.{name}.available: must be bool"
            )
        paths = entry["dataset_paths"]
        if not isinstance(paths, list) or not all(
            isinstance(item, str) for item in paths
        ):
            raise TrajectoryEvidenceError(
                f"capabilities.{name}.dataset_paths: must be an array of strings"
            )
        if bool(paths) != entry["available"]:
            raise TrajectoryEvidenceError(
                f"capabilities.{name}: availability does not match paths"
            )
    pointcloud = _require_exact_keys(
        capabilities["pointcloud"],
        ("available", "dataset_path", "shape", "reason"),
        path="capabilities.pointcloud",
    )
    if not isinstance(pointcloud["available"], bool):
        raise TrajectoryEvidenceError("capabilities.pointcloud.available: must be bool")
    if pointcloud["dataset_path"] is not None:
        _require_hdf5_dataset_path(
            pointcloud["dataset_path"], path="capabilities.pointcloud.dataset_path"
        )
    if pointcloud["shape"] is not None and (
        not isinstance(pointcloud["shape"], list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in pointcloud["shape"]
        )
    ):
        raise TrajectoryEvidenceError("capabilities.pointcloud.shape: invalid shape")
    if pointcloud["reason"] is not None:
        _require_string(pointcloud["reason"], path="capabilities.pointcloud.reason")
    if pointcloud["available"] and pointcloud["dataset_path"] is None:
        raise TrajectoryEvidenceError(
            "capabilities.pointcloud: available data requires a dataset_path"
        )
    return capabilities


def _capabilities_from_catalog(
    catalog: Mapping[str, Mapping[str, Any]],
    config: TrajectoryEvidenceAdapterConfig,
) -> dict[str, Any]:
    depth_paths = sorted(name for name in catalog if name.endswith("/depth"))
    object_pose_paths = sorted(
        name
        for name in catalog
        if "object" in name.casefold() and "pose" in name.casefold()
    )
    pointcloud_paths = sorted(
        name for name in catalog if PurePosixPath(name).name.casefold() == "pointcloud"
    )
    pointcloud_path = pointcloud_paths[0] if pointcloud_paths else None
    pointcloud_shape = (
        list(catalog[pointcloud_path]["shape"]) if pointcloud_path is not None else None
    )
    pointcloud_available = bool(
        pointcloud_shape and len(pointcloud_shape) >= 2 and pointcloud_shape[1] > 0
    )
    return {
        "rgb": {
            "available": True,
            "cameras": [camera.camera_id for camera in config.cameras],
        },
        "depth": {
            "available": bool(depth_paths),
            "dataset_paths": depth_paths,
        },
        "pointcloud": {
            "available": pointcloud_available,
            "dataset_path": pointcloud_path,
            "shape": pointcloud_shape,
            "reason": None
            if pointcloud_available
            else ("empty_width" if pointcloud_path is not None else "absent"),
        },
        "object_pose": {
            "available": bool(object_pose_paths),
            "dataset_paths": object_pose_paths,
        },
    }


class TrajectoryEvidenceExtractor:
    """Select and read only sparse boundary evidence from an authorized HDF5 FD."""

    def __init__(
        self,
        adapter_config: TrajectoryEvidenceAdapterConfig,
        *,
        context_radius: int = 1,
        limits: TrajectoryEvidenceLimits | None = None,
        _total_gripper_scan_sample_limit: int | None = None,
        _total_gripper_transition_limit: int | None = None,
    ) -> None:
        if not isinstance(adapter_config, TrajectoryEvidenceAdapterConfig):
            raise TypeError("adapter_config must be TrajectoryEvidenceAdapterConfig")
        self.adapter_config = adapter_config
        self.limits = limits or TrajectoryEvidenceLimits()
        if (
            isinstance(context_radius, bool)
            or not isinstance(context_radius, int)
            or context_radius < 0
        ):
            raise ValueError("context_radius must be a non-negative integer")
        if context_radius > self.limits.max_context_radius:
            raise TrajectoryEvidenceBudgetExceeded(
                "context_radius exceeds max_context_radius"
            )
        if len(adapter_config.cameras) > self.limits.max_cameras:
            raise TrajectoryEvidenceBudgetExceeded("camera count exceeds max_cameras")
        if len(adapter_config.grippers) > self.limits.max_gripper_channels:
            raise TrajectoryEvidenceBudgetExceeded(
                "gripper channel count exceeds max_gripper_channels"
            )
        self.context_radius = context_radius
        self._total_gripper_scan_sample_limit = (
            self.limits.max_gripper_scan_samples
            if _total_gripper_scan_sample_limit is None
            else _require_int(
                _total_gripper_scan_sample_limit,
                path="total_gripper_scan_sample_limit",
                minimum=1,
            )
        )
        self._total_gripper_transition_limit = (
            self.limits.max_gripper_transitions
            if _total_gripper_transition_limit is None
            else _require_int(
                _total_gripper_transition_limit,
                path="total_gripper_transition_limit",
                minimum=1,
            )
        )
        self._gripper_scan_samples_consumed = 0
        self._gripper_transitions_consumed = 0

    @property
    def gripper_scan_samples_consumed(self) -> int:
        return self._gripper_scan_samples_consumed

    @property
    def gripper_transitions_consumed(self) -> int:
        return self._gripper_transitions_consumed

    def _consume_gripper_scan_budget(self, sample_count: int) -> None:
        proposed = self._gripper_scan_samples_consumed + sample_count
        if proposed > self._total_gripper_scan_sample_limit:
            raise TrajectoryEvidenceBudgetExceeded(
                "manifest-wide gripper scan exceeds max_total_gripper_scan_samples"
            )
        self._gripper_scan_samples_consumed = proposed

    def _consume_gripper_transition_budget(self, transition_count: int) -> None:
        proposed = self._gripper_transitions_consumed + transition_count
        if proposed > self._total_gripper_transition_limit:
            raise TrajectoryEvidenceBudgetExceeded(
                "manifest-wide gripper transitions exceed max_total_gripper_transitions"
            )
        self._gripper_transitions_consumed = proposed

    def extract(
        self,
        *,
        trajectory_id: str,
        segments: Sequence[Mapping[str, Any]],
        hdf5_handle: BinaryIO,
        source_ref_id: str,
        source_content_sha256: str,
        action_chunk_bundle: Mapping[str, Any] | None = None,
    ) -> ExtractedTrajectoryEvidence:
        """Extract evidence using a borrowed, already-authorized file descriptor."""

        _require_safe_id(trajectory_id, path="trajectory_id")
        _require_string(source_ref_id, path="source_ref_id")
        _require_sha256(source_content_sha256, path="source_content_sha256")
        if not hasattr(hdf5_handle, "fileno"):
            raise TrajectoryEvidenceError("hdf5_handle must expose a file descriptor")
        try:
            borrowed_fd = hdf5_handle.fileno()
            duplicate_fd = os.dup(borrowed_fd)
        except (OSError, ValueError) as exc:
            raise TrajectoryEvidenceError(
                "hdf5_handle does not expose a live file descriptor"
            ) from exc
        h5py = _load_h5py()
        try:
            with os.fdopen(duplicate_fd, "rb", closefd=True) as raw_handle:
                duplicate_fd = -1
                with h5py.File(raw_handle, "r") as handle:
                    return self._extract_open_hdf5(
                        trajectory_id=trajectory_id,
                        segments=segments,
                        handle=handle,
                        h5py=h5py,
                        source_ref_id=source_ref_id,
                        source_content_sha256=source_content_sha256,
                        action_chunk_bundle=action_chunk_bundle,
                    )
        except (TrajectoryEvidenceError, TrajectoryEvidenceBudgetExceeded):
            raise
        except (MemoryError, OSError, RuntimeError, ValueError) as exc:
            raise TrajectoryEvidenceError(
                "HDF5 visual evidence extraction failed"
            ) from exc
        finally:
            if duplicate_fd >= 0:
                os.close(duplicate_fd)

    def _extract_open_hdf5(
        self,
        *,
        trajectory_id: str,
        segments: Sequence[Mapping[str, Any]],
        handle: Any,
        h5py: Any,
        source_ref_id: str,
        source_content_sha256: str,
        action_chunk_bundle: Mapping[str, Any] | None,
    ) -> ExtractedTrajectoryEvidence:
        config = self.adapter_config
        action_dataset = _safe_dataset(handle, config.action.dataset_path, h5py)
        frame_count = _validate_numeric_timeseries(
            action_dataset,
            dataset_path=config.action.dataset_path,
            frame_count=None,
            max_elements_per_frame=self.limits.max_action_elements_per_frame,
            scalar=False,
        )
        boundaries = recover_segment_boundaries(
            segments,
            trajectory_id=trajectory_id,
            frame_count=frame_count,
            max_segments=self.limits.max_segments,
        )
        catalog = _dataset_catalog(handle, h5py, self.limits)
        gripper_datasets: dict[str, Any] = {}
        for gripper in config.grippers:
            dataset = _safe_dataset(handle, gripper.dataset_path, h5py)
            _validate_numeric_timeseries(
                dataset,
                dataset_path=gripper.dataset_path,
                frame_count=frame_count,
                max_elements_per_frame=1,
                scalar=True,
            )
            gripper_datasets[gripper.channel_id] = dataset
        gripper_scan_sample_count = frame_count * len(config.grippers)
        if gripper_scan_sample_count > self.limits.max_gripper_scan_samples:
            raise TrajectoryEvidenceBudgetExceeded(
                "gripper scan exceeds max_gripper_scan_samples"
            )
        self._consume_gripper_scan_budget(gripper_scan_sample_count)
        gripper_transitions, gripper_channel_summaries = _scan_gripper_channels(
            config=config,
            gripper_datasets=gripper_datasets,
            frame_count=frame_count,
            limits=self.limits,
        )
        self._consume_gripper_transition_budget(len(gripper_transitions))
        selections = _merge_keyframe_selections(
            _select_keyframes(
                boundaries,
                frame_count=frame_count,
                context_radius=self.context_radius,
                gripper_transitions=gripper_transitions,
            ),
            _action_chunk_keyframes(
                boundaries,
                frame_count=frame_count,
                trajectory_id=trajectory_id,
                source_ref_id=source_ref_id,
                source_content_sha256=source_content_sha256,
                visual_adapter_configuration_sha256=config.configuration_sha256,
                action_chunk_bundle=action_chunk_bundle,
            ),
        )
        evidence_item_count = len(selections) * len(config.cameras)
        if evidence_item_count > self.limits.max_evidence_items:
            raise TrajectoryEvidenceBudgetExceeded(
                "selected evidence exceeds max_evidence_items"
            )
        unique_frames = sorted({item.frame_index for item in selections})
        unique_image_count = len(unique_frames) * len(config.cameras)
        if unique_image_count > self.limits.max_unique_images:
            raise TrajectoryEvidenceBudgetExceeded(
                "selected images exceed max_unique_images"
            )
        camera_datasets: dict[str, Any] = {}
        worst_case_image_bytes = 0
        for camera in config.cameras:
            dataset = _safe_dataset(handle, camera.dataset_path, h5py)
            per_frame_bytes = _validate_camera_timeseries(
                dataset,
                dataset_path=camera.dataset_path,
                frame_count=frame_count,
                limits=self.limits,
            )
            camera_datasets[camera.camera_id] = dataset
            worst_case_image_bytes += per_frame_bytes * len(unique_frames)
        if worst_case_image_bytes > self.limits.max_total_image_bytes:
            raise TrajectoryEvidenceBudgetExceeded(
                "selected source images exceed max_total_image_bytes"
            )
        image_cache: dict[
            tuple[str, int], tuple[dict[str, Any], VisualImagePayload]
        ] = {}
        action_cache: dict[int, tuple[str, list[int], str]] = {}
        gripper_cache: dict[int, list[dict[str, Any]]] = {}
        total_image_bytes = 0
        config_sha = config.configuration_sha256

        def action_sample(frame_index: int) -> tuple[str, list[int], str]:
            if frame_index not in action_cache:
                action_cache[frame_index] = _hash_array_value(
                    _read_dataset_element(action_dataset, frame_index)
                )
            return action_cache[frame_index]

        def gripper_states(frame_index: int) -> list[dict[str, Any]]:
            if frame_index not in gripper_cache:
                states: list[dict[str, Any]] = []
                for spec in config.grippers:
                    raw = np.asarray(
                        _read_dataset_element(
                            gripper_datasets[spec.channel_id], frame_index
                        )
                    )
                    if raw.size != 1:
                        raise TrajectoryEvidenceError(
                            f"gripper {spec.channel_id!r} did not yield one scalar"
                        )
                    raw_value = _require_finite_number(
                        raw.reshape(-1)[0].item(),
                        path=f"gripper.{spec.channel_id}.raw_value",
                    )
                    states.append(
                        {
                            "channel_id": spec.channel_id,
                            "source_type": "observed_robot_state",
                            "state": spec.classify(raw_value),
                            "raw_value": raw_value,
                            "state_ref": {
                                "source_ref_id": source_ref_id,
                                "dataset_path": spec.dataset_path,
                                "frame_index": frame_index,
                            },
                            "semantics_ref": {
                                "adapter_configuration_sha256": config_sha,
                                "channel_id": spec.channel_id,
                                "semantics_source": spec.semantics_source,
                            },
                            "confidence": "high",
                        }
                    )
                gripper_cache[frame_index] = states
            return copy.deepcopy(gripper_cache[frame_index])

        def image_metadata(
            camera: CameraChannelConfig, frame_index: int
        ) -> tuple[dict[str, Any], VisualImagePayload]:
            nonlocal total_image_bytes
            key = (camera.camera_id, frame_index)
            if key not in image_cache:
                payload, media_type = _encode_image_element(
                    _read_dataset_element(
                        camera_datasets[camera.camera_id], frame_index
                    ),
                    limits=self.limits,
                )
                total_image_bytes += len(payload)
                if total_image_bytes > self.limits.max_total_image_bytes:
                    raise TrajectoryEvidenceBudgetExceeded(
                        "encoded images exceed max_total_image_bytes"
                    )
                digest = hashlib.sha256(payload).hexdigest()
                image_ref_id = _make_image_ref_id(
                    source_ref_id=source_ref_id,
                    dataset_path=camera.dataset_path,
                    frame_index=frame_index,
                )
                metadata = {
                    "source_type": "observed_visual",
                    "image_ref_id": image_ref_id,
                    "sha256": digest,
                    "byte_length": len(payload),
                    "media_type": media_type,
                    "stable_ref": {
                        "source_ref_id": source_ref_id,
                        "dataset_path": camera.dataset_path,
                        "frame_index": frame_index,
                    },
                }
                image_cache[key] = (
                    metadata,
                    VisualImagePayload(
                        image_ref_id=image_ref_id,
                        sha256=digest,
                        media_type=media_type,
                        data=payload,
                    ),
                )
            metadata, payload = image_cache[key]
            return copy.deepcopy(metadata), payload

        items: list[dict[str, Any]] = []
        primary_items: dict[tuple[str, str, str], dict[str, Any]] = {}
        boundary_by_id = {item.segment_id: item for item in boundaries}
        for selection in selections:
            sample_sha, sample_shape, sample_dtype = action_sample(
                selection.frame_index
            )
            states = gripper_states(selection.frame_index)
            for camera in config.cameras:
                image, _payload = image_metadata(camera, selection.frame_index)
                evidence_id = _make_evidence_id(
                    trajectory_id=trajectory_id,
                    segment_id=selection.segment_id,
                    frame_index=selection.frame_index,
                    camera=camera.camera_id,
                    temporal_role=selection.temporal_role,
                    adapter_configuration_sha256=config_sha,
                )
                item = {
                    "schema": VisualEvidenceItemV1.SCHEMA,
                    "schema_version": VisualEvidenceItemV1.SCHEMA_VERSION,
                    "evidence_id": evidence_id,
                    "trajectory_id": trajectory_id,
                    "segment_id": selection.segment_id,
                    "segment_index": selection.segment_index,
                    "frame": {
                        "index": selection.frame_index,
                        "evidence_ref": (
                            f"{trajectory_id}#frame/{selection.frame_index:06d}"
                        ),
                    },
                    "camera": camera.camera_id,
                    "temporal_role": selection.temporal_role,
                    "action_range": {
                        "source_type": "observed_robot_state",
                        "source_ref_id": source_ref_id,
                        "dataset_path": config.action.dataset_path,
                        "start_inclusive": selection.action_start,
                        "end_inclusive": selection.action_end,
                        "semantics": config.action.semantics,
                        "sample_frame_index": selection.frame_index,
                        "sample_sha256": sample_sha,
                        "sample_shape": sample_shape,
                        "sample_dtype": sample_dtype,
                    },
                    "gripper_states": copy.deepcopy(states),
                    "image": image,
                    "evidence_source": "authorized_hdf5_expert_trajectory",
                    "modalities": ["visual", "robot_state"],
                    "confidence": "high",
                    "extraction_provenance": {
                        "extractor": _EXTRACTOR_NAME,
                        "version": _EXTRACTOR_VERSION,
                        "selection_policy": _SELECTION_POLICY,
                        "selection_reasons": list(selection.reasons),
                        "adapter_configuration_sha256": config_sha,
                    },
                }
                items.append(item)
                boundary = boundary_by_id[selection.segment_id]
                is_segment_boundary = (
                    selection.temporal_role == "before"
                    and selection.frame_index == boundary.start_inclusive
                ) or (
                    selection.temporal_role == "after"
                    and selection.frame_index == boundary.end_inclusive
                )
                if is_segment_boundary:
                    primary_items[
                        (
                            selection.segment_id,
                            camera.camera_id,
                            selection.temporal_role,
                        )
                    ] = item

        gripper_transition_records: list[dict[str, Any]] = []
        for transition in gripper_transitions:
            before_reason = _gripper_transition_reason(transition, "before")
            after_reason = _gripper_transition_reason(transition, "after")
            before_items = [
                item
                for camera in config.cameras
                for item in items
                if item["camera"] == camera.camera_id
                and before_reason in item["extraction_provenance"]["selection_reasons"]
            ]
            after_items = [
                item
                for camera in config.cameras
                for item in items
                if item["camera"] == camera.camera_id
                and after_reason in item["extraction_provenance"]["selection_reasons"]
            ]
            if len(before_items) != len(config.cameras) or len(after_items) != len(
                config.cameras
            ):
                raise TrajectoryEvidenceError(
                    "gripper transition keyframes do not cover every camera"
                )
            gripper_transition_records.append(
                {
                    "schema": GripperStateTransitionV1.SCHEMA,
                    "schema_version": GripperStateTransitionV1.SCHEMA_VERSION,
                    "transition_id": _make_gripper_transition_id(
                        trajectory_id=trajectory_id,
                        channel_id=transition.channel_id,
                        from_state=transition.from_state,
                        to_state=transition.to_state,
                        before_frame_index=transition.before_frame_index,
                        after_frame_index=transition.after_frame_index,
                        adapter_configuration_sha256=config_sha,
                    ),
                    "trajectory_id": trajectory_id,
                    "channel_id": transition.channel_id,
                    "source_type": "observed_robot_state",
                    "from_state": transition.from_state,
                    "to_state": transition.to_state,
                    "before_segment_id": before_items[0]["segment_id"],
                    "after_segment_id": after_items[0]["segment_id"],
                    "before_state_ref": {
                        "source_ref_id": source_ref_id,
                        "dataset_path": transition.dataset_path,
                        "channel_id": transition.channel_id,
                        "frame_index": transition.before_frame_index,
                        "state": transition.from_state,
                    },
                    "after_state_ref": {
                        "source_ref_id": source_ref_id,
                        "dataset_path": transition.dataset_path,
                        "channel_id": transition.channel_id,
                        "frame_index": transition.after_frame_index,
                        "state": transition.to_state,
                    },
                    "supporting_evidence_refs": {
                        "before": [item["evidence_id"] for item in before_items],
                        "after": [item["evidence_id"] for item in after_items],
                    },
                    "confidence": "high",
                    "scan_provenance": {
                        "mode": _GRIPPER_SCAN_MODE,
                        "chunk_frames": self.limits.gripper_scan_chunk_frames,
                        "adapter_configuration_sha256": config_sha,
                    },
                }
            )

        comparisons: list[dict[str, Any]] = []
        for boundary in boundaries:
            for camera in config.cameras:
                before_item = primary_items[
                    (boundary.segment_id, camera.camera_id, "before")
                ]
                after_item = primary_items[
                    (boundary.segment_id, camera.camera_id, "after")
                ]
                before_payload = image_cache[
                    (camera.camera_id, boundary.start_inclusive)
                ][1]
                after_payload = image_cache[(camera.camera_id, boundary.end_inclusive)][
                    1
                ]
                before_shape, after_shape, normalized_mad = _normalized_rgb_mad(
                    before_payload.data,
                    after_payload.data,
                    self.limits,
                )
                before_grippers = {
                    item["channel_id"]: item for item in before_item["gripper_states"]
                }
                after_grippers = {
                    item["channel_id"]: item for item in after_item["gripper_states"]
                }
                before_evidence_id = before_item["evidence_id"]
                after_evidence_id = after_item["evidence_id"]
                comparisons.append(
                    {
                        "schema": VisualEvidenceComparisonV1.SCHEMA,
                        "schema_version": VisualEvidenceComparisonV1.SCHEMA_VERSION,
                        "comparison_id": _make_comparison_id(
                            trajectory_id=trajectory_id,
                            segment_id=boundary.segment_id,
                            camera=camera.camera_id,
                            before_evidence_id=before_evidence_id,
                            after_evidence_id=after_evidence_id,
                        ),
                        "trajectory_id": trajectory_id,
                        "segment_id": boundary.segment_id,
                        "segment_index": boundary.segment_index,
                        "camera": camera.camera_id,
                        "before_evidence_id": before_evidence_id,
                        "after_evidence_id": after_evidence_id,
                        "modalities": ["visual", "robot_state"],
                        "visual_delta": {
                            "source_type": "observed_visual",
                            "metric": "normalized_rgb_mean_absolute_difference",
                            "before_shape": before_shape,
                            "after_shape": after_shape,
                            "shape_equal": before_shape == after_shape,
                            "hash_changed": (
                                before_payload.sha256 != after_payload.sha256
                            ),
                            "normalized_mean_absolute_difference": normalized_mad,
                            "semantic_interpretation": "none",
                            "causal_attribution": False,
                        },
                        "gripper_deltas": [
                            {
                                "channel_id": channel_id,
                                "source_type": "observed_robot_state",
                                "before_state": before_grippers[channel_id]["state"],
                                "after_state": after_grippers[channel_id]["state"],
                                "before_raw_value": before_grippers[channel_id][
                                    "raw_value"
                                ],
                                "after_raw_value": after_grippers[channel_id][
                                    "raw_value"
                                ],
                                "raw_delta": (
                                    after_grippers[channel_id]["raw_value"]
                                    - before_grippers[channel_id]["raw_value"]
                                ),
                                "state_changed": (
                                    before_grippers[channel_id]["state"]
                                    != after_grippers[channel_id]["state"]
                                ),
                                "supporting_evidence_refs": [
                                    before_evidence_id,
                                    after_evidence_id,
                                ],
                            }
                            for channel_id in (
                                spec.channel_id for spec in config.grippers
                            )
                        ],
                        "confidence": "high",
                        "limitations": sorted(VisualEvidenceComparisonV1._LIMITATIONS),
                    }
                )
        bundle = VisualEvidenceBundleV1.from_dict(
            {
                "schema": VisualEvidenceBundleV1.SCHEMA,
                "schema_version": VisualEvidenceBundleV1.SCHEMA_VERSION,
                "trajectory_id": trajectory_id,
                "source": {
                    "kind": "hdf5_expert_trajectory",
                    "source_ref_id": source_ref_id,
                    "content_sha256": source_content_sha256,
                },
                "frame_count": frame_count,
                "segment_order": [item.segment_id for item in boundaries],
                "adapter": {
                    "adapter_id": config.adapter_id,
                    "adapter_version": config.adapter_version,
                    "embodiment_id": config.embodiment_id,
                    "configuration_sha256": config_sha,
                },
                "selection": {
                    "policy": _SELECTION_POLICY,
                    "context_radius": self.context_radius,
                    "selected_frame_count": len(unique_frames),
                    "selected_image_count": len(image_cache),
                    "evidence_item_count": len(items),
                },
                "data_capabilities": _capabilities_from_catalog(catalog, config),
                "gripper_state_scan": {
                    "schema": "roboharn_evo/gripper_state_scan/v1",
                    "schema_version": 1,
                    "source_type": "observed_robot_state",
                    "source_ref_id": source_ref_id,
                    "mode": _GRIPPER_SCAN_MODE,
                    "chunk_frames": self.limits.gripper_scan_chunk_frames,
                    "frame_count": frame_count,
                    "channel_count": len(config.grippers),
                    "sample_count": gripper_scan_sample_count,
                    "channels": gripper_channel_summaries,
                    "adapter_configuration_sha256": config_sha,
                },
                "items": items,
                "comparisons": comparisons,
                "gripper_transitions": gripper_transition_records,
            }
        )
        payloads = tuple(
            image_cache[key][1]
            for key in sorted(image_cache, key=lambda value: (value[1], value[0]))
        )
        return ExtractedTrajectoryEvidence(bundle=bundle, image_payloads=payloads)


def extract_rmbench_visual_evidence_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    adapter_config: TrajectoryEvidenceAdapterConfig,
    context_radius: int = 1,
    extraction_limits: TrajectoryEvidenceLimits | None = None,
    batch_limits: TrajectoryEvidenceBatchLimits | None = None,
    action_chunk_bundles_by_trajectory: Mapping[str, Mapping[str, Any]] | None = None,
) -> RMBenchTrajectoryEvidenceBatch:
    """Import and extract one explicit manifest without opening raw paths anew."""

    from .expert_trajectory import RMBenchExpertTrajectoryImporter

    global_limits = batch_limits or TrajectoryEvidenceBatchLimits()
    if action_chunk_bundles_by_trajectory is not None and not isinstance(
        action_chunk_bundles_by_trajectory, Mapping
    ):
        raise TrajectoryEvidenceError(
            "action_chunk_bundles_by_trajectory must be an object"
        )
    supplemental_bundles: dict[str, dict[str, Any]] = {}
    for key, value in (action_chunk_bundles_by_trajectory or {}).items():
        trajectory_key = _require_safe_id(
            key, path="action_chunk_bundles_by_trajectory.key"
        )
        if not isinstance(value, Mapping):
            raise TrajectoryEvidenceError(
                "action_chunk_bundles_by_trajectory values must be objects"
            )
        supplemental_bundles[trajectory_key] = copy.deepcopy(dict(value))
    used_supplemental_bundles: set[str] = set()
    extractor = TrajectoryEvidenceExtractor(
        adapter_config,
        context_radius=context_radius,
        limits=extraction_limits,
        _total_gripper_scan_sample_limit=(global_limits.max_total_gripper_scan_samples),
        _total_gripper_transition_limit=(global_limits.max_total_gripper_transitions),
    )

    class _ExtractingImporter(RMBenchExpertTrajectoryImporter):
        def __init__(self, path: str | os.PathLike[str]) -> None:
            super().__init__(path)
            self.extracted_by_id: dict[str, ExtractedTrajectoryEvidence] = {}
            self.evidence_abstained_ids: set[str] = set()

        def _convert_entry(self, **kwargs: Any) -> Any:
            converted = super()._convert_entry(**kwargs)
            record, segments, _evidence, _abstentions = converted
            artifacts = kwargs["artifacts"]
            hdf5_artifact = artifacts["hdf5"]
            if kwargs["trajectory_id"] in supplemental_bundles:
                used_supplemental_bundles.add(kwargs["trajectory_id"])
            try:
                with hdf5_artifact.open_binary() as hdf5_handle:
                    extracted = extractor.extract(
                        trajectory_id=kwargs["trajectory_id"],
                        segments=segments,
                        hdf5_handle=hdf5_handle,
                        source_ref_id=hdf5_artifact.source_ref_id,
                        source_content_sha256=hdf5_artifact.sha256,
                        action_chunk_bundle=supplemental_bundles.get(
                            kwargs["trajectory_id"]
                        ),
                    )
            except TrajectoryEvidenceBudgetExceeded:
                reason = "visual_evidence_budget_exceeded"
            except TrajectoryEvidenceError:
                reason = "visual_evidence_extraction_failed"
            else:
                self.extracted_by_id[kwargs["trajectory_id"]] = extracted
                return record, segments, _evidence, _abstentions

            self.evidence_abstained_ids.add(kwargs["trajectory_id"])
            abstention = {
                "schema": "roboharn_evo/bootstrap_abstention/v1",
                "schema_version": 1,
                "status": "abstained",
                "stage": "evidence_extraction",
                "reason": reason,
                "scope": "visual_evidence",
                "entry_index": kwargs["entry_index"],
                "trajectory_id": kwargs["trajectory_id"],
                "artifact_role": "hdf5",
                "details": {
                    "candidate_generated": False,
                    "raw_error_persisted": False,
                },
            }
            return record, segments, _evidence, [*_abstentions, abstention]

    importer = _ExtractingImporter(manifest_path)
    entries: list[RMBenchTrajectoryEvidenceEntry] = []
    abstentions: list[dict[str, Any]] = []
    total_evidence_items = 0
    total_comparisons = 0
    total_images = 0
    total_image_bytes = 0
    with importer.stream() as stream:
        input_manifest = stream.input_manifest
        for converted in stream:
            abstentions.extend(copy.deepcopy(converted.abstentions))
            if converted.trajectory is None:
                continue
            if converted.trajectory_id in importer.evidence_abstained_ids:
                importer.evidence_abstained_ids.remove(converted.trajectory_id)
                continue
            try:
                extracted = importer.extracted_by_id.pop(converted.trajectory_id)
            except KeyError as exc:
                raise TrajectoryEvidenceError(
                    "authorized importer did not produce visual evidence"
                ) from exc
            next_entries = len(entries) + 1
            next_evidence_items = total_evidence_items + len(extracted.bundle.items)
            next_comparisons = total_comparisons + len(extracted.bundle.comparisons)
            next_images = total_images + len(extracted.image_payloads)
            next_image_bytes = total_image_bytes + sum(
                len(payload.data) for payload in extracted.image_payloads
            )
            observed = {
                "max_entries": next_entries,
                "max_total_evidence_items": next_evidence_items,
                "max_total_comparisons": next_comparisons,
                "max_total_images": next_images,
                "max_total_image_bytes": next_image_bytes,
            }
            exceeded = [
                name
                for name, value in observed.items()
                if value > getattr(global_limits, name)
            ]
            if exceeded:
                raise TrajectoryEvidenceBudgetExceeded(
                    "manifest-wide visual evidence budget exceeded: "
                    + ", ".join(sorted(exceeded))
                )
            total_evidence_items = next_evidence_items
            total_comparisons = next_comparisons
            total_images = next_images
            total_image_bytes = next_image_bytes
            entries.append(
                RMBenchTrajectoryEvidenceEntry(
                    entry_index=converted.entry_index,
                    trajectory_id=converted.trajectory_id,
                    normalized_trajectory=copy.deepcopy(converted.trajectory),
                    segments=tuple(copy.deepcopy(converted.subtask_segments)),
                    extracted=extracted,
                )
            )
    if importer.extracted_by_id:
        raise TrajectoryEvidenceError("unconsumed visual evidence entry")
    if importer.evidence_abstained_ids:
        raise TrajectoryEvidenceError("unconsumed visual evidence abstention")
    unused_supplemental = sorted(
        set(supplemental_bundles) - used_supplemental_bundles
    )
    if unused_supplemental:
        raise TrajectoryEvidenceError(
            "action chunk bundles did not match imported trajectories: "
            + ", ".join(unused_supplemental)
        )
    return RMBenchTrajectoryEvidenceBatch(
        entries=tuple(entries),
        abstentions=tuple(abstentions),
        input_manifest=copy.deepcopy(input_manifest),
        resource_usage={
            "entries": len(entries),
            "evidence_items": total_evidence_items,
            "comparisons": total_comparisons,
            "images": total_images,
            "image_bytes": total_image_bytes,
            "gripper_scan_samples": extractor.gripper_scan_samples_consumed,
            "gripper_transitions": extractor.gripper_transitions_consumed,
        },
    )


__all__ = [
    "ActionChannelConfig",
    "CameraChannelConfig",
    "ExtractedTrajectoryEvidence",
    "GripperStateTransitionV1",
    "GripperChannelConfig",
    "GripperStateTransitionV1",
    "RMBenchTrajectoryEvidenceBatch",
    "RMBenchTrajectoryEvidenceEntry",
    "SegmentBoundary",
    "TrajectoryEvidenceAdapterConfig",
    "TrajectoryEvidenceBudgetExceeded",
    "TrajectoryEvidenceBatchLimits",
    "TrajectoryEvidenceError",
    "TrajectoryEvidenceExtractor",
    "TrajectoryEvidenceLimits",
    "VisualEvidenceBundleV1",
    "VisualEvidenceComparisonV1",
    "VisualEvidenceItemV1",
    "VisualImagePayload",
    "extract_rmbench_visual_evidence_manifest",
    "recover_segment_boundaries",
    "rmbench_dual_arm_trajectory_evidence_config",
]

"""Task-neutral action-chunk detection from authorized observed robot state.

This module deliberately separates *observed state* from commanded actions.  It
uses benchmark adapter configuration to locate per-arm joint state and gripper
channels, then performs bounded low-dimensional scans.  No RGB data is read by
the detector; the returned frame roles tell the visual-evidence layer which
frames to read sparsely after chunk boundaries are known.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, BinaryIO, ClassVar

import numpy as np

from .trajectory_evidence import (
    GripperChannelConfig,
    SegmentBoundary,
    TrajectoryEvidenceBudgetExceeded,
    TrajectoryEvidenceError,
    TrajectoryEvidenceLimits,
    _load_h5py,
    _make_gripper_transition_id,
    _read_dataset_slice,
    _safe_dataset,
    _scan_gripper_channels,
    _validate_numeric_timeseries,
    recover_segment_boundaries,
    rmbench_dual_arm_trajectory_evidence_config,
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_ID_RE = re.compile(r"^ac_[0-9a-f]{24}$")
_TRANSITION_ID_RE = re.compile(r"^gst_[0-9a-f]{24}$")
_FRAME_REF_RE = re.compile(r"^[^#]+#frame/\d{6}$")
_PHASES = frozenset(
    {
        "observed_motion",
        "parallel_observed_motion",
        "gripper_transition",
        "grasp_or_close",
        "release",
        "release_and_settle",
        "mixed_gripper_transition",
    }
)
_CONFIDENCE = frozenset({"low", "medium", "high"})
_SCAN_MODE = "chunked_observed_robot_state/v1"
_DETECTOR_NAME = "task_neutral_action_chunk_detector"
_DETECTOR_VERSION = "1"


class ActionChunkEvidenceError(TrajectoryEvidenceError):
    """Malformed state, configuration, transition, or action-chunk record."""


class ActionChunkEvidenceBudgetExceeded(TrajectoryEvidenceBudgetExceeded):
    """Action-chunk extraction exceeded a declared low-dimensional budget."""


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
        raise ActionChunkEvidenceError("value is not strict JSON") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _safe_id(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ActionChunkEvidenceError(f"{path}: invalid identifier")
    return value


def _positive_int(value: Any, *, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ActionChunkEvidenceError(
            f"{path}: must be an integer no smaller than {minimum}"
        )
    return value


def _finite_nonnegative(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionChunkEvidenceError(f"{path}: must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ActionChunkEvidenceError(f"{path}: must be a finite non-negative number")
    return result


def _dataset_path(value: Any, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
    ):
        raise ActionChunkEvidenceError(f"{path}: invalid relative HDF5 dataset path")
    parts = PurePosixPath(value).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ActionChunkEvidenceError(f"{path}: unsafe HDF5 dataset path")
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class ArmStateChannelConfig:
    """Adapter-owned mapping from one arm to an observed state dataset."""

    arm_id: str
    dataset_path: str
    gripper_channel_id: str

    def __post_init__(self) -> None:
        _safe_id(self.arm_id, path="arm.arm_id")
        _dataset_path(self.dataset_path, path="arm.dataset_path")
        _safe_id(self.gripper_channel_id, path="arm.gripper_channel_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "arm_id": self.arm_id,
            "dataset_path": self.dataset_path,
            "gripper_channel_id": self.gripper_channel_id,
        }


@dataclass(frozen=True, slots=True)
class ActionChunkAdapterConfig:
    """Embodiment-specific state layout; detection itself remains generic."""

    adapter_id: str
    adapter_version: str
    embodiment_id: str
    state_semantics: str
    arms: tuple[ArmStateChannelConfig, ...]
    grippers: tuple[GripperChannelConfig, ...]
    gripper_adapter_configuration_sha256: str

    def __post_init__(self) -> None:
        _safe_id(self.adapter_id, path="adapter.adapter_id")
        _safe_id(self.adapter_version, path="adapter.adapter_version")
        _safe_id(self.embodiment_id, path="adapter.embodiment_id")
        if self.state_semantics != "observed_robot_state_not_commanded_action":
            raise ActionChunkEvidenceError(
                "adapter.state_semantics must preserve observed-state epistemics"
            )
        if not self.arms or not all(
            isinstance(item, ArmStateChannelConfig) for item in self.arms
        ):
            raise ActionChunkEvidenceError("adapter.arms must contain arm channels")
        if not self.grippers or not all(
            isinstance(item, GripperChannelConfig) for item in self.grippers
        ):
            raise ActionChunkEvidenceError(
                "adapter.grippers must contain gripper channels"
            )
        arm_ids = [item.arm_id for item in self.arms]
        arm_paths = [item.dataset_path for item in self.arms]
        gripper_ids = [item.channel_id for item in self.grippers]
        mapped_grippers = [item.gripper_channel_id for item in self.arms]
        if len(arm_ids) != len(set(arm_ids)) or len(arm_paths) != len(set(arm_paths)):
            raise ActionChunkEvidenceError(
                "adapter arm identifiers and paths must be unique"
            )
        if len(gripper_ids) != len(set(gripper_ids)):
            raise ActionChunkEvidenceError("adapter gripper identifiers must be unique")
        if set(mapped_grippers) != set(gripper_ids):
            raise ActionChunkEvidenceError(
                "adapter arms must map one-to-one onto configured grippers"
            )
        if _SHA256_RE.fullmatch(self.gripper_adapter_configuration_sha256) is None:
            raise ActionChunkEvidenceError(
                "adapter.gripper_adapter_configuration_sha256: invalid SHA-256"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "embodiment_id": self.embodiment_id,
            "state_semantics": self.state_semantics,
            "arms": [item.to_dict() for item in self.arms],
            "grippers": [item.to_dict() for item in self.grippers],
            "gripper_adapter_configuration_sha256": (
                self.gripper_adapter_configuration_sha256
            ),
        }

    @property
    def configuration_sha256(self) -> str:
        return _sha256_json(self.to_dict())


def rmbench_dual_arm_action_chunk_config() -> ActionChunkAdapterConfig:
    """Return the RMBench state-layout adapter without task-specific knowledge."""

    visual_adapter = rmbench_dual_arm_trajectory_evidence_config()
    return ActionChunkAdapterConfig(
        adapter_id="rmbench_dual_arm_action_chunks",
        adapter_version="1",
        embodiment_id=visual_adapter.embodiment_id,
        state_semantics="observed_robot_state_not_commanded_action",
        arms=(
            ArmStateChannelConfig(
                arm_id="left_arm",
                dataset_path="joint_action/left_arm",
                gripper_channel_id="left_gripper",
            ),
            ArmStateChannelConfig(
                arm_id="right_arm",
                dataset_path="joint_action/right_arm",
                gripper_channel_id="right_gripper",
            ),
        ),
        grippers=visual_adapter.grippers,
        gripper_adapter_configuration_sha256=visual_adapter.configuration_sha256,
    )


@dataclass(frozen=True, slots=True)
class ActionChunkDetectionConfig:
    """Configurable temporal thresholds for task-neutral change-point detection."""

    smoothing_window_frames: int = 3
    motion_start_threshold: float = 0.003
    motion_stop_threshold: float = 0.001
    start_hysteresis_frames: int = 2
    stop_hysteresis_frames: int = 2
    min_motion_frames: int = 3
    min_motion_path: float = 0.006
    merge_gap_frames: int = 3
    dual_arm_overlap_gap_frames: int = 0
    transition_attach_window_frames: int = 3
    settled_consecutive_frames: int = 4
    settled_search_frames: int = 24
    during_frame_min_duration: int = 24

    def __post_init__(self) -> None:
        for name in (
            "smoothing_window_frames",
            "start_hysteresis_frames",
            "stop_hysteresis_frames",
            "min_motion_frames",
            "settled_consecutive_frames",
            "settled_search_frames",
            "during_frame_min_duration",
        ):
            _positive_int(getattr(self, name), path=f"detection.{name}")
        for name in (
            "merge_gap_frames",
            "dual_arm_overlap_gap_frames",
            "transition_attach_window_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ActionChunkEvidenceError(
                    f"detection.{name}: must be a non-negative integer"
                )
        if self.smoothing_window_frames % 2 != 1:
            raise ActionChunkEvidenceError(
                "detection.smoothing_window_frames must be odd"
            )
        start = _finite_nonnegative(
            self.motion_start_threshold,
            path="detection.motion_start_threshold",
        )
        stop = _finite_nonnegative(
            self.motion_stop_threshold,
            path="detection.motion_stop_threshold",
        )
        _finite_nonnegative(self.min_motion_path, path="detection.min_motion_path")
        if stop >= start:
            raise ActionChunkEvidenceError(
                "detection.motion_stop_threshold must be below start threshold"
            )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def configuration_sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ActionChunkExtractionLimits:
    """Hard ceilings for one low-dimensional trajectory scan."""

    max_frames: int = 200_000
    max_arms: int = 8
    max_state_elements_per_frame: int = 4_096
    max_total_state_samples: int = 2_000_000
    max_action_chunks: int = 4_096
    max_gripper_transitions: int = 2_048
    scan_chunk_frames: int = 256

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _positive_int(getattr(self, name), path=f"limits.{name}")


@dataclass(slots=True)
class _MotionInterval:
    start: int
    end: int
    arms: set[str]
    motion_path_by_arm: dict[str, float]
    transition_ids: set[str]
    transition_records: list[dict[str, Any]]
    merged_gap_frames: int = 0


def _normalize_boundaries(
    boundaries: Sequence[SegmentBoundary | Mapping[str, Any]],
    *,
    trajectory_id: str,
    frame_count: int,
) -> tuple[SegmentBoundary, ...]:
    if not boundaries:
        raise ActionChunkEvidenceError("boundaries must not be empty")
    if all(isinstance(item, SegmentBoundary) for item in boundaries):
        normalized = tuple(
            item for item in boundaries if isinstance(item, SegmentBoundary)
        )
        expected = 0
        for index, boundary in enumerate(normalized):
            if (
                boundary.segment_index != index
                or boundary.start_inclusive != expected
                or boundary.end_inclusive < boundary.start_inclusive
                or boundary.end_inclusive >= frame_count
            ):
                raise ActionChunkEvidenceError(
                    "boundaries must be ordered and cover the trajectory contiguously"
                )
            expected = boundary.end_inclusive + 1
        if expected != frame_count:
            raise ActionChunkEvidenceError(
                "boundaries must cover the complete trajectory"
            )
        return normalized
    if not all(isinstance(item, Mapping) for item in boundaries):
        raise ActionChunkEvidenceError(
            "boundaries must be all SegmentBoundary or all segment records"
        )
    return recover_segment_boundaries(
        [dict(item) for item in boundaries if isinstance(item, Mapping)],
        trajectory_id=trajectory_id,
        frame_count=frame_count,
    )


def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    result = np.empty_like(values)
    for index in range(values.size):
        result[index] = float(np.median(padded[index : index + window]))
    return result


def _motion_signal(
    state: np.ndarray, config: ActionChunkDetectionConfig
) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.diff(state, axis=0, prepend=state[[0]])
    raw = np.sqrt(np.mean(np.square(deltas), axis=1))
    return raw, _median_smooth(raw, config.smoothing_window_frames)


def _arm_intervals(
    raw_speed: np.ndarray,
    smooth_speed: np.ndarray,
    boundary: SegmentBoundary,
    config: ActionChunkDetectionConfig,
) -> list[tuple[int, int, float, int]]:
    """Detect hysteretic motion runs and merge adjacent micro-pauses."""

    provisional: list[tuple[int, int]] = []
    active_start: int | None = None
    high_count = 0
    low_count = 0
    for frame in range(boundary.start_inclusive + 1, boundary.end_inclusive + 1):
        speed = float(smooth_speed[frame])
        if active_start is None:
            if speed >= config.motion_start_threshold:
                high_count += 1
                if high_count >= config.start_hysteresis_frames:
                    first_high = frame - high_count + 1
                    active_start = max(boundary.start_inclusive, first_high - 1)
                    low_count = 0
            else:
                high_count = 0
            continue
        if speed <= config.motion_stop_threshold:
            low_count += 1
            if low_count >= config.stop_hysteresis_frames:
                first_low = frame - low_count + 1
                provisional.append((active_start, max(active_start, first_low - 1)))
                active_start = None
                high_count = 0
                low_count = 0
        else:
            low_count = 0
    if active_start is not None:
        provisional.append((active_start, boundary.end_inclusive))

    measured: list[tuple[int, int, float, int]] = []
    for start, end in provisional:
        path = float(np.sum(raw_speed[start + 1 : end + 1]))
        measured.append((start, end, path, 0))

    merged: list[tuple[int, int, float, int]] = []
    for start, end, path, _gap in measured:
        if merged and start - merged[-1][1] - 1 <= config.merge_gap_frames:
            old_start, old_end, old_path, old_gaps = merged[-1]
            gap = max(0, start - old_end - 1)
            merged[-1] = (old_start, end, old_path + path, old_gaps + gap)
        else:
            merged.append((start, end, path, 0))
    return [
        item
        for item in merged
        if item[1] - item[0] + 1 >= config.min_motion_frames
        and item[2] >= config.min_motion_path
    ]


def _transition_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "transition_id",
        "channel_id",
        "from_state",
        "to_state",
        "before_frame_index",
        "after_frame_index",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ActionChunkEvidenceError(
            "gripper transition missing field(s): " + ", ".join(missing)
        )
    transition_id = value["transition_id"]
    if (
        not isinstance(transition_id, str)
        or _TRANSITION_ID_RE.fullmatch(transition_id) is None
    ):
        raise ActionChunkEvidenceError("gripper transition has invalid stable ID")
    channel_id = _safe_id(value["channel_id"], path="transition.channel_id")
    from_state = value["from_state"]
    to_state = value["to_state"]
    if from_state not in {"open", "closed"} or to_state not in {"open", "closed"}:
        raise ActionChunkEvidenceError("transition states must be stable open/closed")
    if from_state == to_state:
        raise ActionChunkEvidenceError("transition must change state")
    before = _positive_int(
        value["before_frame_index"],
        path="transition.before_frame_index",
        minimum=0,
    )
    after = _positive_int(
        value["after_frame_index"],
        path="transition.after_frame_index",
        minimum=0,
    )
    if after <= before:
        raise ActionChunkEvidenceError("transition frame order is invalid")
    return {
        "transition_id": transition_id,
        "channel_id": channel_id,
        "from_state": from_state,
        "to_state": to_state,
        "before_frame_index": before,
        "after_frame_index": after,
    }


def _combine_overlapping_intervals(
    intervals: list[_MotionInterval], *, gap: int
) -> list[_MotionInterval]:
    combined: list[_MotionInterval] = []
    for current in sorted(
        intervals, key=lambda item: (item.start, item.end, sorted(item.arms))
    ):
        if combined and current.start <= combined[-1].end + gap + 1:
            previous = combined[-1]
            actual_gap = max(0, current.start - previous.end - 1)
            previous.end = max(previous.end, current.end)
            previous.arms.update(current.arms)
            for arm_id, path in current.motion_path_by_arm.items():
                previous.motion_path_by_arm[arm_id] = (
                    previous.motion_path_by_arm.get(arm_id, 0.0) + path
                )
            previous.transition_ids.update(current.transition_ids)
            previous.transition_records.extend(current.transition_records)
            previous.merged_gap_frames += current.merged_gap_frames + actual_gap
        else:
            combined.append(
                _MotionInterval(
                    start=current.start,
                    end=current.end,
                    arms=set(current.arms),
                    motion_path_by_arm=dict(current.motion_path_by_arm),
                    transition_ids=set(current.transition_ids),
                    transition_records=[
                        copy.deepcopy(item) for item in current.transition_records
                    ],
                    merged_gap_frames=current.merged_gap_frames,
                )
            )
    return combined


def _frame_ref(trajectory_id: str, frame_index: int) -> str:
    return f"{trajectory_id}#frame/{frame_index:06d}"


def _phase(interval: _MotionInterval, settled_frame: int | None) -> str:
    releases = [
        item
        for item in interval.transition_records
        if item["from_state"] == "closed" and item["to_state"] == "open"
    ]
    closings = [
        item
        for item in interval.transition_records
        if item["from_state"] == "open" and item["to_state"] == "closed"
    ]
    if releases and closings:
        return "mixed_gripper_transition"
    if releases:
        return "release_and_settle" if settled_frame is not None else "release"
    if closings:
        return "grasp_or_close"
    if interval.transition_records and not interval.motion_path_by_arm:
        return "gripper_transition"
    if len(interval.arms) > 1:
        return "parallel_observed_motion"
    return "observed_motion"


def _settled_frame_after_release(
    interval: _MotionInterval,
    boundary: SegmentBoundary,
    smooth_speeds: Mapping[str, np.ndarray],
    config: ActionChunkDetectionConfig,
) -> int | None:
    releases = [
        item["after_frame_index"]
        for item in interval.transition_records
        if item["from_state"] == "closed" and item["to_state"] == "open"
    ]
    if not releases:
        return None
    search_start = max(releases) + 1
    search_end = min(
        boundary.end_inclusive,
        search_start + config.settled_search_frames - 1,
    )
    stable_count = 0
    for frame in range(search_start, search_end + 1):
        if all(
            float(speed[frame]) <= config.motion_stop_threshold
            for speed in smooth_speeds.values()
        ):
            stable_count += 1
            if stable_count >= config.settled_consecutive_frames:
                return frame
        else:
            stable_count = 0
    return None


def _make_action_chunk_id(
    *,
    trajectory_id: str,
    segment_id: str,
    start: int,
    end: int,
    active_arms: Sequence[str],
    transition_ids: Sequence[str],
    adapter_sha256: str,
    detection_sha256: str,
) -> str:
    digest = _sha256_json(
        {
            "trajectory_id": trajectory_id,
            "segment_id": segment_id,
            "start_inclusive": start,
            "end_inclusive": end,
            "active_arms": list(active_arms),
            "gripper_transition_refs": list(transition_ids),
            "adapter_configuration_sha256": adapter_sha256,
            "detection_configuration_sha256": detection_sha256,
        }
    )
    return f"ac_{digest[:24]}"


class ActionChunkV1:
    """Strict JSON record for one observed-state action chunk."""

    SCHEMA = "roboharn_evo/action_chunk/v1"
    SCHEMA_VERSION = 1
    __slots__ = ("_payload",)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "schema_version",
            "trajectory_id",
            "segment_id",
            "segment_index",
            "action_chunk_id",
            "frame_range",
            "active_arms",
            "observed_motion_state",
            "gripper_transition_refs",
            "inferred_phase",
            "confidence",
            "uncertainty",
            "evidence_frame_refs",
            "source_refs",
            "detector_provenance",
        }
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        copied = copy.deepcopy(dict(payload))
        _canonical_json(copied)
        self._validate(copied)
        self._payload = copied

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionChunkV1:
        return cls(payload)

    @property
    def action_chunk_id(self) -> str:
        return str(self._payload["action_chunk_id"])

    @property
    def start_inclusive(self) -> int:
        return int(self._payload["frame_range"]["start_inclusive"])

    @property
    def end_inclusive(self) -> int:
        return int(self._payload["frame_range"]["end_inclusive"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        if set(payload) != ActionChunkV1._KEYS:
            raise ActionChunkEvidenceError("action chunk fields do not match schema")
        if payload["schema"] not in {
            ActionChunkV1.SCHEMA, "tcm/action_chunk/v1"
        } or payload["schema_version"] != 1:
            raise ActionChunkEvidenceError("unsupported action chunk schema")
        trajectory_id = _safe_id(payload["trajectory_id"], path="chunk.trajectory_id")
        if not isinstance(payload["segment_id"], str) or not payload["segment_id"]:
            raise ActionChunkEvidenceError("chunk.segment_id must be non-empty")
        _positive_int(payload["segment_index"], path="chunk.segment_index", minimum=0)
        chunk_id = payload["action_chunk_id"]
        if not isinstance(chunk_id, str) or _CHUNK_ID_RE.fullmatch(chunk_id) is None:
            raise ActionChunkEvidenceError("chunk.action_chunk_id is invalid")
        frame_range = payload["frame_range"]
        if not isinstance(frame_range, dict) or set(frame_range) != {
            "start_inclusive",
            "end_inclusive",
            "duration_frames",
        }:
            raise ActionChunkEvidenceError("chunk.frame_range is invalid")
        start = _positive_int(
            frame_range["start_inclusive"], path="chunk.frame_range.start", minimum=0
        )
        end = _positive_int(
            frame_range["end_inclusive"], path="chunk.frame_range.end", minimum=0
        )
        if end < start or frame_range["duration_frames"] != end - start + 1:
            raise ActionChunkEvidenceError("chunk.frame_range is inconsistent")
        arms = payload["active_arms"]
        if not isinstance(arms, list) or not arms or arms != sorted(set(arms)):
            raise ActionChunkEvidenceError(
                "chunk.active_arms must be sorted and unique"
            )
        for arm in arms:
            _safe_id(arm, path="chunk.active_arms[]")
        motion = payload["observed_motion_state"]
        if not isinstance(motion, dict) or set(motion) != {
            "source_type",
            "semantics",
            "per_arm_path_length",
            "merged_stationary_gap_frames",
        }:
            raise ActionChunkEvidenceError("chunk.observed_motion_state is invalid")
        if (
            motion["source_type"] != "observed_robot_state"
            or motion["semantics"] != "not_commanded_action"
        ):
            raise ActionChunkEvidenceError("chunk motion epistemics are invalid")
        paths = motion["per_arm_path_length"]
        if not isinstance(paths, dict) or not set(paths).issubset(set(arms)):
            raise ActionChunkEvidenceError("chunk per-arm path mapping is invalid")
        for value in paths.values():
            _finite_nonnegative(value, path="chunk.per_arm_path_length[]")
        _positive_int(
            motion["merged_stationary_gap_frames"],
            path="chunk.merged_stationary_gap_frames",
            minimum=0,
        )
        transition_refs = payload["gripper_transition_refs"]
        if not isinstance(transition_refs, list) or transition_refs != sorted(
            set(transition_refs)
        ):
            raise ActionChunkEvidenceError(
                "chunk transition refs must be sorted and unique"
            )
        if any(
            not isinstance(ref, str) or _TRANSITION_ID_RE.fullmatch(ref) is None
            for ref in transition_refs
        ):
            raise ActionChunkEvidenceError("chunk transition ref is invalid")
        if payload["inferred_phase"] not in _PHASES:
            raise ActionChunkEvidenceError("chunk inferred phase is invalid")
        if payload["confidence"] not in _CONFIDENCE:
            raise ActionChunkEvidenceError("chunk confidence is invalid")
        uncertainty = payload["uncertainty"]
        if not isinstance(uncertainty, list) or not all(
            isinstance(item, str) and item for item in uncertainty
        ):
            raise ActionChunkEvidenceError("chunk uncertainty must be strings")
        evidence = payload["evidence_frame_refs"]
        if not isinstance(evidence, dict) or set(evidence) != {
            "before",
            "during",
            "after",
            "settled",
        }:
            raise ActionChunkEvidenceError("chunk evidence roles are invalid")
        for refs in evidence.values():
            if not isinstance(refs, list) or any(
                not isinstance(ref, str)
                or _FRAME_REF_RE.fullmatch(ref) is None
                or not ref.startswith(f"{trajectory_id}#frame/")
                for ref in refs
            ):
                raise ActionChunkEvidenceError("chunk evidence frame ref is invalid")
        if not evidence["before"] or not evidence["after"]:
            raise ActionChunkEvidenceError("chunk requires before and after evidence")
        source_refs = payload["source_refs"]
        if not isinstance(source_refs, dict) or set(source_refs) != {
            "observed_robot_state",
            "segment_annotation",
        }:
            raise ActionChunkEvidenceError("chunk source refs are invalid")
        states = source_refs["observed_robot_state"]
        if not isinstance(states, list) or not states:
            raise ActionChunkEvidenceError("chunk observed-state refs are required")
        for state_ref in states:
            if not isinstance(state_ref, dict) or set(state_ref) != {
                "source_ref_id",
                "dataset_path",
                "arm_id",
                "start_inclusive",
                "end_inclusive",
            }:
                raise ActionChunkEvidenceError("chunk observed-state ref is invalid")
            _dataset_path(state_ref["dataset_path"], path="chunk.state.dataset_path")
        annotation = source_refs["segment_annotation"]
        if not isinstance(annotation, dict) or set(annotation) != {
            "segment_id",
            "source",
            "evidence_event_refs",
        }:
            raise ActionChunkEvidenceError("chunk annotation ref is invalid")
        if annotation["source"] != "benchmark_scripted_annotation":
            raise ActionChunkEvidenceError("chunk annotation source is invalid")
        provenance = payload["detector_provenance"]
        if not isinstance(provenance, dict) or set(provenance) != {
            "name",
            "version",
            "adapter_configuration_sha256",
            "detection_configuration_sha256",
        }:
            raise ActionChunkEvidenceError("chunk detector provenance is invalid")
        if (
            provenance["name"] != _DETECTOR_NAME
            or provenance["version"] != _DETECTOR_VERSION
        ):
            raise ActionChunkEvidenceError("chunk detector identity is invalid")
        expected_id = _make_action_chunk_id(
            trajectory_id=trajectory_id,
            segment_id=payload["segment_id"],
            start=start,
            end=end,
            active_arms=arms,
            transition_ids=transition_refs,
            adapter_sha256=provenance["adapter_configuration_sha256"],
            detection_sha256=provenance["detection_configuration_sha256"],
        )
        if chunk_id != expected_id:
            raise ActionChunkEvidenceError("chunk ID is not stable")


class ActionChunkBundleV1:
    """Validated action chunks and sparse frame roles for one trajectory."""

    SCHEMA = "roboharn_evo/action_chunk_bundle/v1"
    SCHEMA_VERSION = 1
    __slots__ = ("_chunks", "_payload")

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "schema_version",
            "trajectory_id",
            "source",
            "frame_count",
            "segment_order",
            "adapter",
            "detection",
            "state_scan",
            "gripper_transitions",
            "action_chunks",
            "selected_frames",
        }
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        copied = copy.deepcopy(dict(payload))
        _canonical_json(copied)
        chunks = self._validate(copied)
        self._payload = copied
        self._chunks = chunks

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionChunkBundleV1:
        return cls(payload)

    @property
    def chunks(self) -> tuple[ActionChunkV1, ...]:
        return self._chunks

    @property
    def selected_frame_indices(self) -> tuple[int, ...]:
        return tuple(item["frame_index"] for item in self._payload["selected_frames"])

    def action_chunk_records(self) -> tuple[dict[str, Any], ...]:
        """Records ready to be written one-per-line as ``action_chunks.jsonl``."""

        return tuple(chunk.to_dict() for chunk in self._chunks)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> tuple[ActionChunkV1, ...]:
        if set(payload) != ActionChunkBundleV1._KEYS:
            raise ActionChunkEvidenceError(
                "action chunk bundle fields do not match schema"
            )
        if (
            payload["schema"] not in {ActionChunkBundleV1.SCHEMA, "tcm/action_chunk_bundle/v1"}
            or payload["schema_version"] != 1
        ):
            raise ActionChunkEvidenceError("unsupported action chunk bundle schema")
        trajectory_id = _safe_id(payload["trajectory_id"], path="bundle.trajectory_id")
        source = payload["source"]
        if not isinstance(source, dict) or set(source) != {
            "kind",
            "source_ref_id",
            "content_sha256",
        }:
            raise ActionChunkEvidenceError("bundle source is invalid")
        if source["kind"] != "hdf5_expert_trajectory":
            raise ActionChunkEvidenceError("bundle source kind is invalid")
        if _SHA256_RE.fullmatch(str(source["content_sha256"])) is None:
            raise ActionChunkEvidenceError("bundle source hash is invalid")
        frame_count = _positive_int(payload["frame_count"], path="bundle.frame_count")
        segment_order = payload["segment_order"]
        if (
            not isinstance(segment_order, list)
            or not segment_order
            or segment_order != list(dict.fromkeys(segment_order))
        ):
            raise ActionChunkEvidenceError("bundle segment order is invalid")
        adapter = payload["adapter"]
        detection = payload["detection"]
        for value, label in ((adapter, "adapter"), (detection, "detection")):
            if not isinstance(value, dict) or "configuration_sha256" not in value:
                raise ActionChunkEvidenceError(f"bundle {label} is invalid")
            if _SHA256_RE.fullmatch(str(value["configuration_sha256"])) is None:
                raise ActionChunkEvidenceError(f"bundle {label} hash is invalid")
        scan = payload["state_scan"]
        if not isinstance(scan, dict) or scan.get("mode") != _SCAN_MODE:
            raise ActionChunkEvidenceError("bundle state scan is invalid")
        if scan.get("source_type") != "observed_robot_state":
            raise ActionChunkEvidenceError("bundle state scan epistemics are invalid")
        if scan.get("frame_count") != frame_count:
            raise ActionChunkEvidenceError("bundle state scan frame count mismatch")
        transitions = payload["gripper_transitions"]
        if not isinstance(transitions, list):
            raise ActionChunkEvidenceError(
                "bundle gripper transitions must be an array"
            )
        transition_ids: set[str] = set()
        for item in transitions:
            normalized = _transition_payload(item)
            if normalized["transition_id"] in transition_ids:
                raise ActionChunkEvidenceError("bundle has duplicate transition IDs")
            if normalized["after_frame_index"] >= frame_count:
                raise ActionChunkEvidenceError(
                    "bundle transition is outside trajectory"
                )
            transition_ids.add(normalized["transition_id"])
        raw_chunks = payload["action_chunks"]
        if not isinstance(raw_chunks, list):
            raise ActionChunkEvidenceError("bundle action chunks must be an array")
        chunks = tuple(ActionChunkV1.from_dict(item) for item in raw_chunks)
        seen_chunks: set[str] = set()
        chunk_segment_by_id: dict[str, str] = {}
        previous_key: tuple[int, int, int] | None = None
        for chunk in chunks:
            item = chunk.to_dict()
            if item["trajectory_id"] != trajectory_id:
                raise ActionChunkEvidenceError("bundle contains cross-trajectory chunk")
            if item["segment_id"] not in segment_order:
                raise ActionChunkEvidenceError(
                    "bundle chunk references unknown segment"
                )
            if not set(item["gripper_transition_refs"]).issubset(transition_ids):
                raise ActionChunkEvidenceError(
                    "bundle chunk has unresolved transition ref"
                )
            if chunk.action_chunk_id in seen_chunks:
                raise ActionChunkEvidenceError("bundle has duplicate action chunk ID")
            seen_chunks.add(chunk.action_chunk_id)
            chunk_segment_by_id[chunk.action_chunk_id] = str(item["segment_id"])
            key = (item["segment_index"], chunk.start_inclusive, chunk.end_inclusive)
            if previous_key is not None and key < previous_key:
                raise ActionChunkEvidenceError("bundle chunks are not ordered")
            previous_key = key
        selected = payload["selected_frames"]
        if not isinstance(selected, list):
            raise ActionChunkEvidenceError("bundle selected frames must be an array")
        seen_frames: set[int] = set()
        for item in selected:
            if not isinstance(item, dict) or set(item) != {
                "frame_index",
                "frame_ref",
                "roles",
                "action_chunk_ids",
                "segment_ids",
            }:
                raise ActionChunkEvidenceError("bundle selected frame is invalid")
            frame_index = _positive_int(
                item["frame_index"], path="bundle.selected.frame_index", minimum=0
            )
            if frame_index >= frame_count or frame_index in seen_frames:
                raise ActionChunkEvidenceError(
                    "bundle selected frame is duplicate/outside"
                )
            seen_frames.add(frame_index)
            if item["frame_ref"] != _frame_ref(trajectory_id, frame_index):
                raise ActionChunkEvidenceError("bundle selected frame ref is invalid")
            if not isinstance(item["roles"], list) or not item["roles"]:
                raise ActionChunkEvidenceError(
                    "bundle selected frame roles are required"
                )
            chunk_ids = item["action_chunk_ids"]
            selected_segments = item["segment_ids"]
            if (
                not isinstance(chunk_ids, list)
                or chunk_ids != sorted(set(chunk_ids))
                or not isinstance(selected_segments, list)
                or not selected_segments
                or selected_segments != sorted(set(selected_segments))
                or not set(selected_segments).issubset(set(segment_order))
            ):
                raise ActionChunkEvidenceError(
                    "bundle selected frame ownership is invalid"
                )
            if not set(chunk_ids).issubset(seen_chunks):
                raise ActionChunkEvidenceError(
                    "bundle selected frame has unknown chunk ref"
                )
            if any(
                chunk_segment_by_id[chunk_id] not in selected_segments
                for chunk_id in chunk_ids
            ):
                raise ActionChunkEvidenceError(
                    "bundle selected frame cites a chunk from another segment"
                )
        return chunks


def detect_action_chunks(
    trajectory_id: str,
    boundaries: Sequence[SegmentBoundary | Mapping[str, Any]],
    observed_state: Mapping[str, np.ndarray | Sequence[Sequence[float]]],
    gripper_transitions: Sequence[Mapping[str, Any]] = (),
    *,
    source_ref_id: str,
    source_content_sha256: str,
    adapter_config: ActionChunkAdapterConfig,
    detection_config: ActionChunkDetectionConfig | None = None,
    limits: ActionChunkExtractionLimits | None = None,
) -> ActionChunkBundleV1:
    """Detect action chunks from complete low-dimensional observed-state arrays.

    Motion is detected independently per arm using a median-filtered magnitude,
    separate start/stop thresholds, and persistence hysteresis.  Short/noisy
    runs are removed, nearby runs of the same arm are merged, then overlapping
    arms are represented by one parallel chunk.  Gripper transitions are
    attached structurally; closed-to-open transitions additionally request an
    independently selected post-release settled frame.
    """

    _safe_id(trajectory_id, path="trajectory_id")
    if not isinstance(source_ref_id, str) or not source_ref_id:
        raise ActionChunkEvidenceError("source_ref_id must be non-empty")
    if _SHA256_RE.fullmatch(source_content_sha256) is None:
        raise ActionChunkEvidenceError("source_content_sha256 is invalid")
    if not isinstance(adapter_config, ActionChunkAdapterConfig):
        raise TypeError("adapter_config must be ActionChunkAdapterConfig")
    config = detection_config or ActionChunkDetectionConfig()
    ceilings = limits or ActionChunkExtractionLimits()
    if len(adapter_config.arms) > ceilings.max_arms:
        raise ActionChunkEvidenceBudgetExceeded("arm count exceeds max_arms")
    if set(observed_state) != {item.arm_id for item in adapter_config.arms}:
        raise ActionChunkEvidenceError(
            "observed_state keys must exactly match configured arm identifiers"
        )
    states: dict[str, np.ndarray] = {}
    frame_count: int | None = None
    total_samples = 0
    for arm in adapter_config.arms:
        state = np.asarray(observed_state[arm.arm_id], dtype=np.float64)
        if state.ndim != 2 or state.shape[0] <= 0 or state.shape[1] <= 0:
            raise ActionChunkEvidenceError(
                f"observed_state[{arm.arm_id!r}] must have shape [frames, dimensions]"
            )
        if state.shape[1] > ceilings.max_state_elements_per_frame:
            raise ActionChunkEvidenceBudgetExceeded(
                "observed state exceeds max_state_elements_per_frame"
            )
        if not np.isfinite(state).all():
            raise ActionChunkEvidenceError("observed state contains non-finite values")
        if frame_count is None:
            frame_count = int(state.shape[0])
        elif state.shape[0] != frame_count:
            raise ActionChunkEvidenceError("observed arm states have different lengths")
        total_samples += int(state.size)
        states[arm.arm_id] = state
    assert frame_count is not None
    if frame_count > ceilings.max_frames:
        raise ActionChunkEvidenceBudgetExceeded("frame count exceeds max_frames")
    if total_samples > ceilings.max_total_state_samples:
        raise ActionChunkEvidenceBudgetExceeded(
            "state scan exceeds max_total_state_samples"
        )
    normalized_boundaries = _normalize_boundaries(
        boundaries,
        trajectory_id=trajectory_id,
        frame_count=frame_count,
    )
    transitions = [_transition_payload(item) for item in gripper_transitions]
    if len(transitions) > ceilings.max_gripper_transitions:
        raise ActionChunkEvidenceBudgetExceeded(
            "gripper transitions exceed max_gripper_transitions"
        )
    transitions.sort(
        key=lambda item: (
            item["before_frame_index"],
            item["after_frame_index"],
            item["channel_id"],
        )
    )
    if any(item["after_frame_index"] >= frame_count for item in transitions):
        raise ActionChunkEvidenceError("gripper transition lies outside trajectory")

    raw_speeds: dict[str, np.ndarray] = {}
    smooth_speeds: dict[str, np.ndarray] = {}
    for arm_id, state in states.items():
        raw_speeds[arm_id], smooth_speeds[arm_id] = _motion_signal(state, config)
    arm_for_gripper = {
        item.gripper_channel_id: item.arm_id for item in adapter_config.arms
    }
    chunks: list[dict[str, Any]] = []
    selected: dict[int, dict[str, set[str]]] = {}
    detection_sha = config.configuration_sha256
    adapter_sha = adapter_config.configuration_sha256

    def add_selected(
        frame: int, role: str, *, chunk_id: str | None, segment_id: str
    ) -> None:
        entry = selected.setdefault(
            frame,
            {"roles": set(), "action_chunk_ids": set(), "segment_ids": set()},
        )
        entry["roles"].add(role)
        entry["segment_ids"].add(segment_id)
        if chunk_id is not None:
            entry["action_chunk_ids"].add(chunk_id)

    for boundary in normalized_boundaries:
        intervals: list[_MotionInterval] = []
        for arm in adapter_config.arms:
            for start, end, path, merged_gaps in _arm_intervals(
                raw_speeds[arm.arm_id],
                smooth_speeds[arm.arm_id],
                boundary,
                config,
            ):
                intervals.append(
                    _MotionInterval(
                        start=start,
                        end=end,
                        arms={arm.arm_id},
                        motion_path_by_arm={arm.arm_id: path},
                        transition_ids=set(),
                        transition_records=[],
                        merged_gap_frames=merged_gaps,
                    )
                )
        intervals = _combine_overlapping_intervals(
            intervals, gap=config.dual_arm_overlap_gap_frames
        )
        segment_transitions = [
            item
            for item in transitions
            if boundary.start_inclusive
            <= item["after_frame_index"]
            <= boundary.end_inclusive
        ]
        transition_intervals: list[_MotionInterval] = []
        for transition in segment_transitions:
            arm_id = arm_for_gripper.get(transition["channel_id"])
            if arm_id is None:
                raise ActionChunkEvidenceError(
                    "gripper transition channel has no configured arm"
                )
            # A gripper transition is an independent observed-state change point.
            # Keeping it as its own chunk prevents a long transport interval from
            # collapsing grasp and release into one semantically mixed record.
            transition_intervals.append(
                _MotionInterval(
                    start=max(
                        boundary.start_inclusive, transition["before_frame_index"]
                    ),
                    end=min(boundary.end_inclusive, transition["after_frame_index"]),
                    arms={arm_id},
                    motion_path_by_arm={},
                    transition_ids={transition["transition_id"]},
                    transition_records=[transition],
                )
            )
        intervals.extend(_combine_overlapping_intervals(transition_intervals, gap=0))
        intervals.sort(key=lambda item: (item.start, item.end, sorted(item.arms)))

        add_selected(
            boundary.start_inclusive,
            "subtask_before",
            chunk_id=None,
            segment_id=boundary.segment_id,
        )
        add_selected(
            boundary.end_inclusive,
            "subtask_after",
            chunk_id=None,
            segment_id=boundary.segment_id,
        )
        add_selected(
            boundary.end_inclusive,
            "task_feedback",
            chunk_id=None,
            segment_id=boundary.segment_id,
        )
        segment_record = next(
            (
                item
                for item in boundaries
                if isinstance(item, Mapping)
                and item.get("segment_id") == boundary.segment_id
            ),
            None,
        )
        annotation_refs: list[str] = []
        if isinstance(segment_record, Mapping):
            derivation = segment_record.get("derivation")
            if isinstance(derivation, Mapping):
                raw_refs = derivation.get("evidence_event_refs", [])
                if isinstance(raw_refs, list):
                    annotation_refs = [str(item) for item in raw_refs]

        for interval in intervals:
            active_arms = sorted(interval.arms)
            transition_ids = sorted(interval.transition_ids)
            settled_frame = _settled_frame_after_release(
                interval,
                boundary,
                smooth_speeds,
                config,
            )
            if interval.transition_records and not interval.motion_path_by_arm:
                before_frame = min(
                    item["before_frame_index"] for item in interval.transition_records
                )
                after_frame = max(
                    item["after_frame_index"] for item in interval.transition_records
                )
            else:
                before_frame = max(boundary.start_inclusive, interval.start - 1)
                after_frame = min(boundary.end_inclusive, interval.end + 1)
            during_frames: list[int] = []
            if interval.end - interval.start + 1 >= config.during_frame_min_duration:
                during_frames = [(interval.start + interval.end) // 2]
            uncertainty = [
                "low-dimensional samples are observed robot state, not commanded action"
            ]
            if interval.merged_gap_frames:
                uncertainty.append(
                    "short stationary gaps were merged by the configured noise filter"
                )
            if (
                any(
                    item["from_state"] == "closed" and item["to_state"] == "open"
                    for item in interval.transition_records
                )
                and settled_frame is None
            ):
                uncertainty.append(
                    "no post-release settled window was detected within the configured search"
                )
            chunk_id = _make_action_chunk_id(
                trajectory_id=trajectory_id,
                segment_id=boundary.segment_id,
                start=interval.start,
                end=interval.end,
                active_arms=active_arms,
                transition_ids=transition_ids,
                adapter_sha256=adapter_sha,
                detection_sha256=detection_sha,
            )
            phase = _phase(interval, settled_frame)
            chunk = {
                "schema": ActionChunkV1.SCHEMA,
                "schema_version": ActionChunkV1.SCHEMA_VERSION,
                "trajectory_id": trajectory_id,
                "segment_id": boundary.segment_id,
                "segment_index": boundary.segment_index,
                "action_chunk_id": chunk_id,
                "frame_range": {
                    "start_inclusive": interval.start,
                    "end_inclusive": interval.end,
                    "duration_frames": interval.end - interval.start + 1,
                },
                "active_arms": active_arms,
                "observed_motion_state": {
                    "source_type": "observed_robot_state",
                    "semantics": "not_commanded_action",
                    "per_arm_path_length": {
                        arm_id: float(interval.motion_path_by_arm[arm_id])
                        for arm_id in sorted(interval.motion_path_by_arm)
                    },
                    "merged_stationary_gap_frames": interval.merged_gap_frames,
                },
                "gripper_transition_refs": transition_ids,
                "inferred_phase": phase,
                "confidence": "high" if interval.motion_path_by_arm else "medium",
                "uncertainty": uncertainty,
                "evidence_frame_refs": {
                    "before": [_frame_ref(trajectory_id, before_frame)],
                    "during": [
                        _frame_ref(trajectory_id, item) for item in during_frames
                    ],
                    "after": [_frame_ref(trajectory_id, after_frame)],
                    "settled": (
                        []
                        if settled_frame is None
                        else [_frame_ref(trajectory_id, settled_frame)]
                    ),
                },
                "source_refs": {
                    "observed_robot_state": [
                        {
                            "source_ref_id": source_ref_id,
                            "dataset_path": next(
                                arm.dataset_path
                                for arm in adapter_config.arms
                                if arm.arm_id == arm_id
                            ),
                            "arm_id": arm_id,
                            "start_inclusive": interval.start,
                            "end_inclusive": interval.end,
                        }
                        for arm_id in active_arms
                    ],
                    "segment_annotation": {
                        "segment_id": boundary.segment_id,
                        "source": "benchmark_scripted_annotation",
                        "evidence_event_refs": annotation_refs,
                    },
                },
                "detector_provenance": {
                    "name": _DETECTOR_NAME,
                    "version": _DETECTOR_VERSION,
                    "adapter_configuration_sha256": adapter_sha,
                    "detection_configuration_sha256": detection_sha,
                },
            }
            chunks.append(ActionChunkV1.from_dict(chunk).to_dict())
            for role, frames in (
                ("action_chunk_before", [before_frame]),
                ("action_chunk_during", during_frames),
                ("action_chunk_after", [after_frame]),
                (
                    "release_after_settle",
                    [] if settled_frame is None else [settled_frame],
                ),
            ):
                for frame in frames:
                    add_selected(
                        frame,
                        role,
                        chunk_id=chunk_id,
                        segment_id=boundary.segment_id,
                    )
            for transition in interval.transition_records:
                add_selected(
                    transition["before_frame_index"],
                    "gripper_transition_before",
                    chunk_id=chunk_id,
                    segment_id=boundary.segment_id,
                )
                add_selected(
                    transition["after_frame_index"],
                    "gripper_transition_after",
                    chunk_id=chunk_id,
                    segment_id=boundary.segment_id,
                )
        if len(chunks) > ceilings.max_action_chunks:
            raise ActionChunkEvidenceBudgetExceeded(
                "detected chunks exceed max_action_chunks"
            )

    selected_frames = [
        {
            "frame_index": frame,
            "frame_ref": _frame_ref(trajectory_id, frame),
            "roles": sorted(values["roles"]),
            "action_chunk_ids": sorted(values["action_chunk_ids"]),
            "segment_ids": sorted(values["segment_ids"]),
        }
        for frame, values in sorted(selected.items())
    ]
    return ActionChunkBundleV1.from_dict(
        {
            "schema": ActionChunkBundleV1.SCHEMA,
            "schema_version": ActionChunkBundleV1.SCHEMA_VERSION,
            "trajectory_id": trajectory_id,
            "source": {
                "kind": "hdf5_expert_trajectory",
                "source_ref_id": source_ref_id,
                "content_sha256": source_content_sha256,
            },
            "frame_count": frame_count,
            "segment_order": [item.segment_id for item in normalized_boundaries],
            "adapter": {
                **adapter_config.to_dict(),
                "configuration_sha256": adapter_sha,
            },
            "detection": {
                **config.to_dict(),
                "configuration_sha256": detection_sha,
            },
            "state_scan": {
                "mode": _SCAN_MODE,
                "source_type": "observed_robot_state",
                "semantics": "not_commanded_action",
                "frame_count": frame_count,
                "channel_count": len(states),
                "sample_count": total_samples,
                "channels": [
                    {
                        "arm_id": arm.arm_id,
                        "dataset_path": arm.dataset_path,
                        "dimensions": int(states[arm.arm_id].shape[1]),
                    }
                    for arm in adapter_config.arms
                ],
            },
            "gripper_transitions": transitions,
            "action_chunks": chunks,
            "selected_frames": selected_frames,
        }
    )


class ActionChunkEvidenceExtractor:
    """Read bounded arm/gripper state from an already-authorized HDF5 FD."""

    def __init__(
        self,
        adapter_config: ActionChunkAdapterConfig,
        *,
        detection_config: ActionChunkDetectionConfig | None = None,
        limits: ActionChunkExtractionLimits | None = None,
    ) -> None:
        if not isinstance(adapter_config, ActionChunkAdapterConfig):
            raise TypeError("adapter_config must be ActionChunkAdapterConfig")
        self.adapter_config = adapter_config
        self.detection_config = detection_config or ActionChunkDetectionConfig()
        self.limits = limits or ActionChunkExtractionLimits()
        if len(adapter_config.arms) > self.limits.max_arms:
            raise ActionChunkEvidenceBudgetExceeded("arm count exceeds max_arms")

    def extract(
        self,
        *,
        trajectory_id: str,
        segments: Sequence[Mapping[str, Any]],
        hdf5_handle: BinaryIO,
        source_ref_id: str,
        source_content_sha256: str,
    ) -> ActionChunkBundleV1:
        if not hasattr(hdf5_handle, "fileno"):
            raise ActionChunkEvidenceError("hdf5_handle must expose a file descriptor")
        try:
            duplicate_fd = os.dup(hdf5_handle.fileno())
        except (OSError, ValueError) as exc:
            raise ActionChunkEvidenceError("hdf5_handle is not live") from exc
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
                    )
        except (ActionChunkEvidenceError, ActionChunkEvidenceBudgetExceeded):
            raise
        except (MemoryError, OSError, RuntimeError, ValueError) as exc:
            raise ActionChunkEvidenceError(
                "action chunk HDF5 extraction failed"
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
    ) -> ActionChunkBundleV1:
        frame_count: int | None = None
        observed: dict[str, np.ndarray] = {}
        planned_state_samples = 0
        for arm in self.adapter_config.arms:
            dataset = _safe_dataset(handle, arm.dataset_path, h5py)
            current_count = _validate_numeric_timeseries(
                dataset,
                dataset_path=arm.dataset_path,
                frame_count=frame_count,
                max_elements_per_frame=self.limits.max_state_elements_per_frame,
                scalar=False,
            )
            if frame_count is None:
                frame_count = current_count
            if current_count > self.limits.max_frames:
                raise ActionChunkEvidenceBudgetExceeded(
                    "frame count exceeds max_frames"
                )
            elements_per_frame = math.prod(int(value) for value in dataset.shape[1:])
            planned_state_samples += current_count * elements_per_frame
            if planned_state_samples > self.limits.max_total_state_samples:
                raise ActionChunkEvidenceBudgetExceeded(
                    "state scan exceeds max_total_state_samples before allocation"
                )
            chunks = []
            for start in range(0, current_count, self.limits.scan_chunk_frames):
                stop = min(current_count, start + self.limits.scan_chunk_frames)
                chunks.append(np.asarray(_read_dataset_slice(dataset, start, stop)))
            observed[arm.arm_id] = np.concatenate(chunks, axis=0)
        assert frame_count is not None
        gripper_datasets: dict[str, Any] = {}
        for spec in self.adapter_config.grippers:
            dataset = _safe_dataset(handle, spec.dataset_path, h5py)
            _validate_numeric_timeseries(
                dataset,
                dataset_path=spec.dataset_path,
                frame_count=frame_count,
                max_elements_per_frame=1,
                scalar=True,
            )
            gripper_datasets[spec.channel_id] = dataset
        visual_limits = TrajectoryEvidenceLimits(
            max_gripper_scan_samples=max(
                1, frame_count * len(self.adapter_config.grippers)
            ),
            max_gripper_transitions=self.limits.max_gripper_transitions,
            gripper_scan_chunk_frames=self.limits.scan_chunk_frames,
        )
        detected, _summaries = _scan_gripper_channels(
            config=type(
                "_GripperScanAdapter",
                (),
                {"grippers": self.adapter_config.grippers},
            )(),
            gripper_datasets=gripper_datasets,
            frame_count=frame_count,
            limits=visual_limits,
        )
        transitions = [
            {
                "transition_id": _make_gripper_transition_id(
                    trajectory_id=trajectory_id,
                    channel_id=item.channel_id,
                    from_state=item.from_state,
                    to_state=item.to_state,
                    before_frame_index=item.before_frame_index,
                    after_frame_index=item.after_frame_index,
                    adapter_configuration_sha256=(
                        self.adapter_config.gripper_adapter_configuration_sha256
                    ),
                ),
                "channel_id": item.channel_id,
                "from_state": item.from_state,
                "to_state": item.to_state,
                "before_frame_index": item.before_frame_index,
                "after_frame_index": item.after_frame_index,
            }
            for item in detected
        ]
        return detect_action_chunks(
            trajectory_id,
            segments,
            observed,
            transitions,
            source_ref_id=source_ref_id,
            source_content_sha256=source_content_sha256,
            adapter_config=self.adapter_config,
            detection_config=self.detection_config,
            limits=self.limits,
        )


@dataclass(frozen=True, slots=True)
class RMBenchActionChunkEvidenceEntry:
    entry_index: int
    trajectory_id: str
    normalized_trajectory: dict[str, Any]
    segments: tuple[dict[str, Any], ...]
    action_chunks: ActionChunkBundleV1


@dataclass(frozen=True, slots=True)
class RMBenchActionChunkEvidenceBatch:
    entries: tuple[RMBenchActionChunkEvidenceEntry, ...]
    abstentions: tuple[dict[str, Any], ...]
    input_manifest: dict[str, Any]
    resource_usage: dict[str, int]


def extract_action_chunk_evidence_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    adapter_config: ActionChunkAdapterConfig | None = None,
    detection_config: ActionChunkDetectionConfig | None = None,
    extraction_limits: ActionChunkExtractionLimits | None = None,
) -> RMBenchActionChunkEvidenceBatch:
    """Run chunk detection inside the existing manifest authorization boundary."""

    from .expert_trajectory import RMBenchExpertTrajectoryImporter

    extractor = ActionChunkEvidenceExtractor(
        adapter_config or rmbench_dual_arm_action_chunk_config(),
        detection_config=detection_config,
        limits=extraction_limits,
    )

    class _ExtractingImporter(RMBenchExpertTrajectoryImporter):
        def __init__(self, path: str | os.PathLike[str]) -> None:
            super().__init__(path)
            self.extracted_by_id: dict[str, ActionChunkBundleV1] = {}
            self.evidence_abstained_ids: set[str] = set()

        def _convert_entry(self, **kwargs: Any) -> Any:
            converted = super()._convert_entry(**kwargs)
            record, segments, evidence, abstentions = converted
            artifact = kwargs["artifacts"]["hdf5"]
            try:
                with artifact.open_binary() as handle:
                    bundle = extractor.extract(
                        trajectory_id=kwargs["trajectory_id"],
                        segments=segments,
                        hdf5_handle=handle,
                        source_ref_id=artifact.source_ref_id,
                        source_content_sha256=artifact.sha256,
                    )
            except ActionChunkEvidenceBudgetExceeded:
                reason = "action_chunk_evidence_budget_exceeded"
            except (ActionChunkEvidenceError, TrajectoryEvidenceError):
                reason = "action_chunk_evidence_extraction_failed"
            else:
                self.extracted_by_id[kwargs["trajectory_id"]] = bundle
                return converted
            self.evidence_abstained_ids.add(kwargs["trajectory_id"])
            abstention = {
                "schema": "roboharn_evo/bootstrap_abstention/v1",
                "schema_version": 1,
                "status": "abstained",
                "stage": "evidence_extraction",
                "reason": reason,
                "scope": "action_chunk_evidence",
                "entry_index": kwargs["entry_index"],
                "trajectory_id": kwargs["trajectory_id"],
                "artifact_role": "hdf5",
                "details": {
                    "candidate_generated": False,
                    "raw_error_persisted": False,
                },
            }
            return record, segments, evidence, [*abstentions, abstention]

    importer = _ExtractingImporter(manifest_path)
    entries: list[RMBenchActionChunkEvidenceEntry] = []
    abstentions: list[dict[str, Any]] = []
    total_chunks = 0
    total_selected_frames = 0
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
                bundle = importer.extracted_by_id.pop(converted.trajectory_id)
            except KeyError as exc:
                raise ActionChunkEvidenceError(
                    "authorized importer did not produce action chunks"
                ) from exc
            total_chunks += len(bundle.chunks)
            total_selected_frames += len(bundle.selected_frame_indices)
            entries.append(
                RMBenchActionChunkEvidenceEntry(
                    entry_index=converted.entry_index,
                    trajectory_id=converted.trajectory_id,
                    normalized_trajectory=copy.deepcopy(converted.trajectory),
                    segments=tuple(copy.deepcopy(converted.subtask_segments)),
                    action_chunks=bundle,
                )
            )
    if importer.extracted_by_id or importer.evidence_abstained_ids:
        raise ActionChunkEvidenceError("unconsumed action chunk extraction state")
    return RMBenchActionChunkEvidenceBatch(
        entries=tuple(entries),
        abstentions=tuple(abstentions),
        input_manifest=copy.deepcopy(input_manifest),
        resource_usage={
            "entries": len(entries),
            "action_chunks": total_chunks,
            "selected_source_frames": total_selected_frames,
        },
    )


__all__ = [
    "ActionChunkAdapterConfig",
    "ActionChunkBundleV1",
    "ActionChunkDetectionConfig",
    "ActionChunkEvidenceBudgetExceeded",
    "ActionChunkEvidenceError",
    "ActionChunkEvidenceExtractor",
    "ActionChunkExtractionLimits",
    "ActionChunkV1",
    "ArmStateChannelConfig",
    "RMBenchActionChunkEvidenceBatch",
    "RMBenchActionChunkEvidenceEntry",
    "detect_action_chunks",
    "extract_action_chunk_evidence_manifest",
    "rmbench_dual_arm_action_chunk_config",
]

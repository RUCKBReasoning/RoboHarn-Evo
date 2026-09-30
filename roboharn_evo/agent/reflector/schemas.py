"""Versioned, JSON-only records for offline trajectory reflection.

The classes in this module are deliberately small persistence-boundary wrappers.
They validate dictionaries, retain fields unknown to this v1 reader, and return
deep copies from :meth:`to_dict`.  They do not write files, promote experiences,
or alter the runtime Reflector contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, ClassVar


class SchemaValidationError(ValueError):
    """Raised when a persisted v1 record violates its schema contract."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")

_SOURCE_KINDS = {"benchmark_expert", "agent_success", "agent_failure", "mixed"}
_PRODUCER_KINDS = {"benchmark", "agent", "reflector", "importer", "human"}
_ARTIFACT_KINDS = {"hdf5", "pkl", "json", "jsonl", "trace", "image", "video", "other"}
_LICENSE_STATUSES = {"verified", "unverified", "restricted"}
_CONFIDENCE_LEVELS = {"low", "medium", "high"}
_EXPERIENCE_STATUSES = {"candidate", "accepted", "deprecated"}
_EVALUATION_STATUSES = {"not_evaluated", "passed", "failed"}
_SUBTASK_OUTCOMES = {"success", "failure", "partial", "not_executed", "unknown"}

_FORBIDDEN_CROSS_TRAJECTORY_KEYS = {
    "annotation",
    "annotations",
    "benchmark_label",
    "candidate_id",
    "error_category",
    "geometry_hash",
    "gold_ood_scenario",
    "instance_id",
    "label",
    "labels",
    "s_stage",
    "d_stage",
    "target_error_category",
    "track_id",
    "world_m",
}

_PRIVATE_TRANSFER_EXACT_KEYS = frozenset(
    {
        *_FORBIDDEN_CROSS_TRAJECTORY_KEYS,
        "benchmark_hidden_outcome",
        "config_hash",
        "episode_id",
        "event_ref",
        "event_refs",
        "frame_ref",
        "frame_refs",
        "model_id",
        "oracle_object_id",
        "prompt_hash",
        "raw_model_output",
        "raw_output",
        "rgb_path",
        "depth_path",
        "mask_path",
        "video_path",
        "relative_path",
        "seed",
        "segment_ref",
        "segment_refs",
        "source_ref",
        "source_refs",
        "source_uri",
        "split",
        "trace_events",
        "trajectory_id",
    }
)
_PRIVATE_TRANSFER_EXACT_KEYS_CASEFOLD = frozenset(
    item.casefold() for item in _PRIVATE_TRANSFER_EXACT_KEYS
)
_PRIVATE_ID_TOKENS = frozenset(
    {"candidate", "geometry", "instance", "object", "oracle", "track"}
)
_PRIVATE_POSE_TOKENS = frozenset(
    {
        "orientation",
        "pose",
        "position",
        "quaternion",
        "rotation",
        "se3",
        "transform",
        "translation",
    }
)


def _fail(path: str, message: str) -> None:
    raise SchemaValidationError(f"{path}: {message}" if path else message)


def _validate_json(value: Any, *, path: str = "record") -> None:
    """Reject values that cannot be represented faithfully by strict JSON."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "NaN and Infinity are not valid persisted JSON values")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "JSON object keys must be strings")
            _validate_json(item, path=f"{path}.{key}")
        return
    _fail(path, f"unsupported non-JSON value of type {type(value).__name__}")


def _json_copy(payload: Mapping[str, Any], *, path: str = "record") -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        _fail(path, "must be an object")
    copied = copy.deepcopy(dict(payload))
    _validate_json(copied, path=path)
    return copied


def _require_keys(
    payload: Mapping[str, Any], keys: Sequence[str], *, path: str
) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        _fail(path, "missing required field(s): " + ", ".join(missing))


def _require_object(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _require_list(value: Any, *, path: str, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    if nonempty and not value:
        _fail(path, "must not be empty")
    return value


def _require_string(value: Any, *, path: str, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    if nonempty and not value.strip():
        _fail(path, "must be a non-empty string")
    return value


def _require_bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _require_int(value: Any, *, path: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _validate_utc(value: Any, *, path: str) -> str:
    text = _require_string(value, path=path)
    if not _UTC_RE.fullmatch(text):
        _fail(path, "must be RFC3339 UTC with a trailing Z")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        _fail(path, f"invalid RFC3339 UTC timestamp ({exc})")
    return text


def _validate_sha256(value: Any, *, path: str) -> str:
    text = _require_string(value, path=path)
    if not _SHA256_RE.fullmatch(text):
        _fail(path, "must be a 64-character lowercase SHA-256 hex digest")
    return text


def _validate_schema(payload: Mapping[str, Any], *, schema: str, path: str) -> None:
    _require_keys(payload, ("schema", "schema_version"), path=path)
    legacy_schema = "tcm/" + schema.removeprefix("roboharn_evo/")
    if payload["schema"] not in {schema, legacy_schema}:
        _fail(f"{path}.schema", f"expected {schema!r}, got {payload['schema']!r}")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        _fail(f"{path}.schema_version", "unsupported major version; expected integer 1")


def _validate_relative_path(value: Any, *, path: str) -> None:
    text = _require_string(value, path=path)
    if (
        "\x00" in text
        or text.startswith(("/", "\\"))
        or _WINDOWS_ABSOLUTE_RE.match(text)
    ):
        _fail(path, "must be a safe relative path")
    normalized = text.replace("\\", "/")
    raw_parts = normalized.split("/")
    posix_parts = PurePosixPath(normalized).parts
    if (
        not posix_parts
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(part == ".." for part in posix_parts)
    ):
        _fail(path, "must not contain empty, dot, or parent traversal components")


def _validate_producer(payload: Any, *, path: str) -> None:
    producer = _require_object(payload, path=path)
    _require_keys(producer, ("kind", "name", "version"), path=path)
    if producer["kind"] not in _PRODUCER_KINDS:
        _fail(f"{path}.kind", f"must be one of {sorted(_PRODUCER_KINDS)}")
    _require_string(producer["name"], path=f"{path}.name")
    _require_string(producer["version"], path=f"{path}.version")
    for key in ("model_id", "code_revision"):
        if key in producer:
            _require_string(producer[key], path=f"{path}.{key}")
    for key in ("prompt_hash", "agent_config_hash"):
        if key in producer:
            _validate_sha256(producer[key], path=f"{path}.{key}")


def _validate_source_ref(payload: Any, *, path: str) -> None:
    source = _require_object(payload, path=path)
    _require_keys(
        source, ("source_ref_id", "content_sha256", "artifact_kind"), path=path
    )
    _require_string(source["source_ref_id"], path=f"{path}.source_ref_id")
    _validate_sha256(source["content_sha256"], path=f"{path}.content_sha256")
    if source["artifact_kind"] not in _ARTIFACT_KINDS:
        _fail(f"{path}.artifact_kind", f"must be one of {sorted(_ARTIFACT_KINDS)}")
    if "relative_path" in source:
        _validate_relative_path(source["relative_path"], path=f"{path}.relative_path")
    if "source_uri" in source:
        _require_string(source["source_uri"], path=f"{path}.source_uri")
    if "size_bytes" in source:
        _require_int(source["size_bytes"], path=f"{path}.size_bytes", minimum=0)
    for key in ("trajectory_id", "segment_id", "benchmark", "task", "split"):
        if key in source:
            _require_string(source[key], path=f"{path}.{key}")
    for key in ("seed", "episode_id"):
        if key in source and not (
            isinstance(source[key], str)
            or (isinstance(source[key], int) and not isinstance(source[key], bool))
        ):
            _fail(f"{path}.{key}", "must be an integer or string")


def _validate_source_refs(
    payload: Any, *, path: str, nonempty: bool = True
) -> list[dict[str, Any]]:
    refs = _require_list(payload, path=path, nonempty=nonempty)
    source_ids: set[str] = set()
    for index, ref in enumerate(refs):
        ref_path = f"{path}[{index}]"
        _validate_source_ref(ref, path=ref_path)
        source_id = ref["source_ref_id"]
        if source_id in source_ids:
            _fail(ref_path, f"duplicate source_ref_id {source_id!r}")
        source_ids.add(source_id)
    return refs


class _JsonRecord(Mapping[str, Any]):
    """Read-only mapping facade over a validated JSON dictionary."""

    SCHEMA: ClassVar[str]
    SCHEMA_VERSION: ClassVar[int] = 1
    __slots__ = ("_data",)

    def __init__(self, payload: Mapping[str, Any], **validation_context: Any) -> None:
        data = _json_copy(payload, path=self.__class__.__name__)
        self._normalize(data)
        self._validate(data, **validation_context)
        self._data = data

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], **validation_context: Any
    ) -> "_JsonRecord":
        return cls(payload, **validation_context)

    @classmethod
    def from_json(cls, payload: str, **validation_context: Any) -> "_JsonRecord":
        def reject_constant(value: str) -> None:
            raise SchemaValidationError(f"invalid non-finite JSON number: {value}")

        try:
            decoded = json.loads(payload, parse_constant=reject_constant)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaValidationError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(decoded, **validation_context)

    @classmethod
    def _normalize(cls, payload: dict[str, Any]) -> None:
        del payload

    @classmethod
    def _validate(cls, payload: dict[str, Any], **validation_context: Any) -> None:
        del validation_context
        _validate_schema(payload, schema=cls.SCHEMA, path=cls.__name__)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_json(self, *, sort_keys: bool = False) -> str:
        return json.dumps(
            self._data,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=sort_keys,
            separators=(",", ":"),
        )

    @property
    def data(self) -> dict[str, Any]:
        return self.to_dict()

    @property
    def schema(self) -> str:
        return self._data["schema"]

    @property
    def schema_version(self) -> int:
        return self._data["schema_version"]

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"


class ProvenanceV1(_JsonRecord):
    """已验证的来源记录。"""

    SCHEMA = "roboharn_evo/provenance/v1"

    @classmethod
    def _validate(cls, payload: dict[str, Any], **validation_context: Any) -> None:
        del validation_context
        super()._validate(payload)
        path = cls.__name__
        _require_keys(
            payload,
            (
                "source_kind",
                "source_subtype",
                "source_refs",
                "producer",
                "created_at",
                "information_access",
                "oracle_derived",
                "expert_derived",
            ),
            path=path,
        )
        if payload["source_kind"] not in _SOURCE_KINDS:
            _fail(f"{path}.source_kind", f"must be one of {sorted(_SOURCE_KINDS)}")
        _require_string(payload["source_subtype"], path=f"{path}.source_subtype")
        _validate_source_refs(payload["source_refs"], path=f"{path}.source_refs")
        _validate_producer(payload["producer"], path=f"{path}.producer")
        _validate_utc(payload["created_at"], path=f"{path}.created_at")
        access = _require_object(
            payload["information_access"], path=f"{path}.information_access"
        )
        _require_keys(
            access,
            (
                "rollout_oracle_used",
                "expert_prior_used",
                "cross_rollout_memory_enabled",
            ),
            path=f"{path}.information_access",
        )
        for key in (
            "rollout_oracle_used",
            "expert_prior_used",
            "cross_rollout_memory_enabled",
        ):
            if access[key] is not None and not isinstance(access[key], bool):
                _fail(f"{path}.information_access.{key}", "must be a boolean or null")
        _require_bool(payload["oracle_derived"], path=f"{path}.oracle_derived")
        _require_bool(payload["expert_derived"], path=f"{path}.expert_derived")
        if (
            payload["source_kind"] == "benchmark_expert"
            and payload["source_subtype"] == "benchmark_scripted_expert"
            and (
                payload["oracle_derived"] is not True
                or payload["expert_derived"] is not True
            )
        ):
            _fail(
                path,
                "benchmark_scripted_expert provenance requires both "
                "oracle_derived=true and expert_derived=true",
            )
        if "legacy_record" in payload:
            _require_bool(payload["legacy_record"], path=f"{path}.legacy_record")
        if (
            "license_status" in payload
            and payload["license_status"] not in _LICENSE_STATUSES
        ):
            _fail(
                f"{path}.license_status", f"must be one of {sorted(_LICENSE_STATUSES)}"
            )
        if "notes" in payload:
            _require_string(payload["notes"], path=f"{path}.notes", nonempty=False)


def _validate_entity_ref(payload: Any, *, path: str) -> None:
    entity = _require_object(payload, path=path)
    _require_keys(entity, ("semantic_class", "task_role"), path=path)
    _require_string(entity["semantic_class"], path=f"{path}.semantic_class")
    _require_string(entity["task_role"], path=f"{path}.task_role")
    for key in ("semantic_descriptor", "geometry_class"):
        if key in entity:
            _require_string(entity[key], path=f"{path}.{key}")
    if "instance_id" in entity and entity["instance_id"] is not None:
        _require_string(entity["instance_id"], path=f"{path}.instance_id")
    if "attributes" in entity:
        _require_object(entity["attributes"], path=f"{path}.attributes")


def _validate_evidence_ref_list(
    refs: Any,
    *,
    path: str,
    trajectory_id: str,
    evidence_index: Any | None,
    nonempty: bool = False,
) -> list[str]:
    from .evidence import EvidenceRefV1

    values = _require_list(refs, path=path, nonempty=nonempty)
    seen: set[str] = set()
    for index, value in enumerate(values):
        value_path = f"{path}[{index}]"
        _require_string(value, path=value_path)
        parsed = EvidenceRefV1.parse(value)
        if parsed.trajectory_id != trajectory_id:
            _fail(
                value_path,
                f"belongs to trajectory {parsed.trajectory_id!r}, expected {trajectory_id!r}",
            )
        if value in seen:
            _fail(value_path, f"duplicate evidence reference {value!r}")
        seen.add(value)
    if evidence_index is not None:
        evidence_index.validate_refs(values, trajectory_id=trajectory_id)
    return values


class SubtaskSegmentV1(_JsonRecord):
    """Validated derived subtask with resolvable evidence references."""

    SCHEMA = "roboharn_evo/subtask_segment/v1"

    @classmethod
    def _validate(cls, payload: dict[str, Any], **validation_context: Any) -> None:
        super()._validate(payload)
        evidence_index = validation_context.get("evidence_index")
        path = cls.__name__
        _require_keys(
            payload,
            (
                "segment_id",
                "trajectory_id",
                "segment_index",
                "context",
                "transition",
                "outcome",
                "derivation",
            ),
            path=path,
        )
        _require_string(payload["segment_id"], path=f"{path}.segment_id")
        trajectory_id = _require_string(
            payload["trajectory_id"], path=f"{path}.trajectory_id"
        )
        _require_int(payload["segment_index"], path=f"{path}.segment_index", minimum=0)

        context = _require_object(payload["context"], path=f"{path}.context")
        _require_keys(
            context, ("subtask_instruction", "participants"), path=f"{path}.context"
        )
        _require_string(
            context["subtask_instruction"],
            path=f"{path}.context.subtask_instruction",
            nonempty=False,
        )
        # Empty is meaningful: an importer must not invent object/target roles
        # when the source contains no evidence for them.
        participants = _require_list(
            context["participants"], path=f"{path}.context.participants"
        )
        for index, participant in enumerate(participants):
            _validate_entity_ref(
                participant, path=f"{path}.context.participants[{index}]"
            )
        for key in ("relation", "manipulation_phase"):
            if key in context:
                _require_string(context[key], path=f"{path}.context.{key}")

        transition = _require_object(payload["transition"], path=f"{path}.transition")
        _require_keys(
            transition,
            ("before_event_refs", "action_event_refs", "after_event_refs"),
            path=f"{path}.transition",
        )
        before_refs = _validate_evidence_ref_list(
            transition["before_event_refs"],
            path=f"{path}.transition.before_event_refs",
            trajectory_id=trajectory_id,
            evidence_index=evidence_index,
        )
        action_refs = _validate_evidence_ref_list(
            transition["action_event_refs"],
            path=f"{path}.transition.action_event_refs",
            trajectory_id=trajectory_id,
            evidence_index=evidence_index,
        )
        after_refs = _validate_evidence_ref_list(
            transition["after_event_refs"],
            path=f"{path}.transition.after_event_refs",
            trajectory_id=trajectory_id,
            evidence_index=evidence_index,
        )

        outcome = _require_object(payload["outcome"], path=f"{path}.outcome")
        _require_keys(outcome, ("status",), path=f"{path}.outcome")
        if outcome["status"] not in _SUBTASK_OUTCOMES:
            _fail(
                f"{path}.outcome.status", f"must be one of {sorted(_SUBTASK_OUTCOMES)}"
            )
        if not action_refs and outcome["status"] not in {"not_executed", "unknown"}:
            _fail(
                f"{path}.outcome.status",
                "must be not_executed or unknown when action_event_refs is empty",
            )
        effects = outcome.get("observed_effects", [])
        _require_list(effects, path=f"{path}.outcome.observed_effects")
        if effects and (not before_refs or not after_refs):
            _fail(
                f"{path}.outcome.observed_effects",
                "specific state effects require both before and after evidence",
            )
        for index, effect in enumerate(effects):
            effect_path = f"{path}.outcome.observed_effects[{index}]"
            effect_obj = _require_object(effect, path=effect_path)
            _require_keys(
                effect_obj, ("metric", "value", "evidence_event_refs"), path=effect_path
            )
            _require_string(effect_obj["metric"], path=f"{effect_path}.metric")
            _validate_evidence_ref_list(
                effect_obj["evidence_event_refs"],
                path=f"{effect_path}.evidence_event_refs",
                trajectory_id=trajectory_id,
                evidence_index=evidence_index,
                nonempty=True,
            )

        derivation = _require_object(payload["derivation"], path=f"{path}.derivation")
        _require_keys(
            derivation,
            ("producer", "confidence", "evidence_event_refs", "created_at"),
            path=f"{path}.derivation",
        )
        _validate_producer(derivation["producer"], path=f"{path}.derivation.producer")
        if derivation["confidence"] not in _CONFIDENCE_LEVELS:
            _fail(
                f"{path}.derivation.confidence",
                f"must be one of {sorted(_CONFIDENCE_LEVELS)}",
            )
        _validate_evidence_ref_list(
            derivation["evidence_event_refs"],
            path=f"{path}.derivation.evidence_event_refs",
            trajectory_id=trajectory_id,
            evidence_index=evidence_index,
            nonempty=True,
        )
        _validate_utc(derivation["created_at"], path=f"{path}.derivation.created_at")


class TrajectoryRecordV1(_JsonRecord):
    """Persistence envelope that leaves the enclosed UnifiedTrajectory intact."""

    SCHEMA = "roboharn_evo/trajectory_record/v1"

    @classmethod
    def _validate(cls, payload: dict[str, Any], **validation_context: Any) -> None:
        del validation_context
        super()._validate(payload)
        path = cls.__name__
        _require_keys(
            payload,
            ("trajectory", "provenance", "evidence_index", "scene_memory"),
            path=path,
        )

        trajectory = _require_object(payload["trajectory"], path=f"{path}.trajectory")
        _require_keys(
            trajectory,
            (
                "trajectory_id",
                "instruction",
                "trace_events",
                "outcome",
                "schema_version",
            ),
            path=f"{path}.trajectory",
        )
        trajectory_id = _require_string(
            trajectory["trajectory_id"], path=f"{path}.trajectory.trajectory_id"
        )
        _require_string(
            trajectory["instruction"],
            path=f"{path}.trajectory.instruction",
            nonempty=False,
        )
        events = _require_list(
            trajectory["trace_events"], path=f"{path}.trajectory.trace_events"
        )
        for index, event in enumerate(events):
            _require_object(event, path=f"{path}.trajectory.trace_events[{index}]")
        _require_object(trajectory["outcome"], path=f"{path}.trajectory.outcome")
        if trajectory["schema_version"] != 1 or isinstance(
            trajectory["schema_version"], bool
        ):
            _fail(
                f"{path}.trajectory.schema_version",
                "unsupported UnifiedTrajectory version; expected 1",
            )

        provenance = ProvenanceV1.from_dict(payload["provenance"])
        provenance_data = provenance.to_dict()
        source_refs = provenance_data["source_refs"]
        for number, source_ref in enumerate(source_refs):
            source_trajectory_id = source_ref.get("trajectory_id")
            if (
                source_trajectory_id is not None
                and source_trajectory_id != trajectory_id
            ):
                _fail(
                    f"{path}.provenance.source_refs[{number}].trajectory_id",
                    f"expected {trajectory_id!r}, got {source_trajectory_id!r}",
                )
        from .evidence import EvidenceIndex

        evidence_entries = _require_list(
            payload["evidence_index"], path=f"{path}.evidence_index"
        )
        index = EvidenceIndex.from_entries(
            evidence_entries,
            source_refs=source_refs,
            trajectory_id=trajectory_id,
        )
        from .evidence import EvidenceRefV1

        for evidence_ref in index.refs:
            parsed_ref = EvidenceRefV1.parse(evidence_ref)
            if parsed_ref.kind == "event" and parsed_ref.index >= len(events):
                _fail(
                    f"{path}.evidence_index",
                    f"event ref {evidence_ref!r} is outside trace_events length {len(events)}",
                )

        scene_memory = _require_object(
            payload["scene_memory"], path=f"{path}.scene_memory"
        )
        _require_keys(
            scene_memory,
            ("available", "snapshot", "source"),
            path=f"{path}.scene_memory",
        )
        available = _require_bool(
            scene_memory["available"], path=f"{path}.scene_memory.available"
        )
        snapshot = _require_object(
            scene_memory["snapshot"], path=f"{path}.scene_memory.snapshot"
        )
        source = _require_string(
            scene_memory["source"], path=f"{path}.scene_memory.source"
        )
        if not available and snapshot:
            _fail(
                f"{path}.scene_memory.snapshot",
                "must be empty when scene memory is unavailable",
            )
        if available and source == "unavailable":
            _fail(
                f"{path}.scene_memory.source",
                "cannot be 'unavailable' when available is true",
            )
        if (
            provenance_data["source_kind"] == "benchmark_expert"
            and provenance_data.get("source_subtype") == "benchmark_scripted_expert"
            and any(ref.get("artifact_kind") == "hdf5" for ref in source_refs)
            and (available or snapshot or source != "unavailable")
        ):
            _fail(
                f"{path}.scene_memory",
                "RMBench scripted-expert HDF5 records must declare Scene "
                "Memory unavailable with an empty snapshot",
            )

        segments = payload.get("subtask_segments", [])
        _require_list(segments, path=f"{path}.subtask_segments")
        segment_ids: set[str] = set()
        for number, segment in enumerate(segments):
            wrapped = SubtaskSegmentV1.from_dict(segment, evidence_index=index)
            segment_data = wrapped.to_dict()
            if segment_data["trajectory_id"] != trajectory_id:
                _fail(
                    f"{path}.subtask_segments[{number}].trajectory_id",
                    f"expected {trajectory_id!r}",
                )
            segment_id = segment_data["segment_id"]
            if segment_id in segment_ids:
                _fail(
                    f"{path}.subtask_segments[{number}].segment_id",
                    "duplicate segment_id",
                )
            segment_ids.add(segment_id)
        derivations = payload.get("derivations", [])
        _require_list(derivations, path=f"{path}.derivations")
        for number, derivation in enumerate(derivations):
            _require_object(derivation, path=f"{path}.derivations[{number}]")

    def to_unified_trajectory(self) -> Any:
        """Construct the existing runtime-neutral contract without changing it."""

        from .contracts import UnifiedTrajectory

        trajectory = self._data["trajectory"]
        return UnifiedTrajectory(
            trajectory_id=trajectory["trajectory_id"],
            instruction=trajectory["instruction"],
            trace_events=copy.deepcopy(trajectory["trace_events"]),
            outcome=copy.deepcopy(trajectory["outcome"]),
            schema_version=trajectory["schema_version"],
        )


def _deduplicate_strings(values: Any, *, path: str) -> list[str]:
    items = _require_list(values, path=path)
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _require_string(item, path=f"{path}[{index}]")
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _is_private_transfer_key(key: str) -> bool:
    """Return whether a key denotes trajectory-private/non-transferable data."""

    normalized = key.casefold()
    if normalized in _PRIVATE_TRANSFER_EXACT_KEYS_CASEFOLD:
        return True
    if normalized == "world_m" or normalized.endswith("_world_m"):
        return True

    tokens = tuple(part for part in re.split(r"[^a-z0-9]+", normalized) if part)
    token_set = set(tokens)
    if "se" in token_set and "3" in token_set:
        return True
    if token_set & _PRIVATE_POSE_TOKENS:
        return True
    if normalized == "id" or (
        normalized.endswith(("_id", "_uid", "_uuid")) and token_set & _PRIVATE_ID_TOKENS
    ):
        return True
    if normalized.endswith(("_hash", "_ref")) and token_set & {
        "candidate",
        "geometry",
        "instance",
        "track",
    }:
        return True
    if "world" in token_set and token_set & {
        "camera",
        "ee",
        "matrix",
        "position",
        "rotation",
        "t",
        "transform",
        "translation",
    }:
        return True
    if normalized.endswith(("_rgb_path", "_depth_path", "_mask_path", "_video_path")):
        return True
    return False


def _find_forbidden_transfer_fields(payload: Any, *, path: str) -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{path}.{key}"
            if _is_private_transfer_key(key):
                found.append(child)
            found.extend(_find_forbidden_transfer_fields(value, path=child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(
                _find_forbidden_transfer_fields(value, path=f"{path}[{index}]")
            )
    return found


def _validate_attribute_map(payload: Any, *, path: str) -> None:
    attributes = _require_object(payload, path=path)
    forbidden = _find_forbidden_transfer_fields(attributes, path=path)
    if forbidden:
        _fail(
            path,
            "contains trajectory-private or forbidden field(s): "
            + ", ".join(forbidden),
        )


def _validate_entity_condition(payload: Any, *, path: str) -> None:
    condition = _require_object(payload, path=path)
    for key in ("semantic_class", "geometry_class"):
        if key in condition:
            _require_string(condition[key], path=f"{path}.{key}")
    if "attributes" in condition:
        _validate_attribute_map(condition["attributes"], path=f"{path}.attributes")


def _validate_transfer_condition(payload: Mapping[str, Any], *, path: str) -> None:
    for key in (
        "task_family",
        "subtask_type",
        "relation",
        "manipulation_phase",
        "failure_signature",
        "OOD_scenario",
        "semantic_class",
        "geometry_class",
    ):
        if key in payload:
            _require_string(payload[key], path=f"{path}.{key}")
    for key in ("object_condition", "target_condition"):
        if key in payload:
            _validate_entity_condition(payload[key], path=f"{path}.{key}")
    if "attributes" in payload:
        _validate_attribute_map(payload["attributes"], path=f"{path}.attributes")
    if "preconditions" in payload:
        values = _require_list(payload["preconditions"], path=f"{path}.preconditions")
        for index, value in enumerate(values):
            _require_string(value, path=f"{path}.preconditions[{index}]")


def _validate_predicted_effects(payload: Any, *, path: str) -> list[Any]:
    effects = _require_list(payload, path=path)
    for index, effect_value in enumerate(effects):
        effect_path = f"{path}[{index}]"
        effect = _require_object(effect_value, path=effect_path)
        for key in ("metric", "comparison", "unit"):
            if key in effect:
                value = _require_string(effect[key], path=f"{effect_path}.{key}")
                if key == "metric" and _is_private_transfer_key(value):
                    _fail(
                        f"{effect_path}.metric",
                        "must not name trajectory-private coordinates, poses, or IDs",
                    )
    return effects


def _validate_unique_string_list(
    payload: Any,
    *,
    path: str,
    nonempty: bool = False,
) -> list[str]:
    values = _require_list(payload, path=path, nonempty=nonempty)
    seen: set[str] = set()
    for index, value in enumerate(values):
        text = _require_string(value, path=f"{path}[{index}]")
        if text in seen:
            _fail(f"{path}[{index}]", f"duplicate value {text!r}")
        seen.add(text)
    return values


def _validate_feedback_branch(payload: Any, *, path: str) -> None:
    branch = _require_list(payload, path=path, nonempty=True)
    allowed_step_keys = {"action_pattern", "condition", "instruction"}
    for index, value in enumerate(branch):
        item_path = f"{path}[{index}]"
        if isinstance(value, str):
            _require_string(value, path=item_path)
            continue
        step = _require_object(value, path=item_path)
        unknown = sorted(set(step) - allowed_step_keys)
        if unknown:
            _fail(item_path, "unknown field(s): " + ", ".join(unknown))
        _require_string(
            step.get("action_pattern"),
            path=f"{item_path}.action_pattern",
        )
        for key in ("condition", "instruction"):
            if key in step:
                _require_string(step[key], path=f"{item_path}.{key}")


def _validate_feedback_policy(payload: Any, *, path: str) -> None:
    policy = _require_object(payload, path=path)
    required = {
        "state_variables",
        "attempt_tracking",
        "observation_rules",
        "failure_branch",
        "success_branch",
        "termination_conditions",
    }
    missing = sorted(required - set(policy))
    unknown = sorted(set(policy) - required)
    if missing:
        _fail(path, "missing required field(s): " + ", ".join(missing))
    if unknown:
        _fail(path, "unknown field(s): " + ", ".join(unknown))

    _validate_unique_string_list(
        policy["state_variables"],
        path=f"{path}.state_variables",
        nonempty=True,
    )
    attempt_tracking = _require_object(
        policy["attempt_tracking"],
        path=f"{path}.attempt_tracking",
    )
    attempt_keys = {
        "state_variable",
        "record_after_attempt",
        "exclude_previously_recorded",
    }
    if set(attempt_tracking) != attempt_keys:
        _fail(
            f"{path}.attempt_tracking",
            "expected exactly " + ", ".join(sorted(attempt_keys)),
        )
    tracked_variable = _require_string(
        attempt_tracking["state_variable"],
        path=f"{path}.attempt_tracking.state_variable",
    )
    if tracked_variable not in policy["state_variables"]:
        _fail(
            f"{path}.attempt_tracking.state_variable",
            "must name one of feedback_policy.state_variables",
        )
    for key in ("record_after_attempt", "exclude_previously_recorded"):
        enabled = _require_bool(
            attempt_tracking[key],
            path=f"{path}.attempt_tracking.{key}",
        )
        if enabled is not True:
            _fail(
                f"{path}.attempt_tracking.{key}",
                "must be true for a feedback candidate",
            )
    rules = _require_list(
        policy["observation_rules"],
        path=f"{path}.observation_rules",
        nonempty=True,
    )
    for index, value in enumerate(rules):
        rule_path = f"{path}.observation_rules[{index}]"
        rule = _require_object(value, path=rule_path)
        required_rule_keys = {"after_action_pattern", "observe_signal"}
        missing_rule_keys = sorted(required_rule_keys - set(rule))
        unknown_rule_keys = sorted(set(rule) - required_rule_keys)
        if missing_rule_keys:
            _fail(
                rule_path,
                "missing required field(s): " + ", ".join(missing_rule_keys),
            )
        if unknown_rule_keys:
            _fail(rule_path, "unknown field(s): " + ", ".join(unknown_rule_keys))
        for key in sorted(required_rule_keys):
            _require_string(rule[key], path=f"{rule_path}.{key}")

    _validate_feedback_branch(
        policy["failure_branch"], path=f"{path}.failure_branch"
    )
    _validate_feedback_branch(
        policy["success_branch"], path=f"{path}.success_branch"
    )
    _validate_unique_string_list(
        policy["termination_conditions"],
        path=f"{path}.termination_conditions",
        nonempty=True,
    )


def _validate_promotion(payload: Mapping[str, Any], *, path: str) -> None:
    promotion_key = (
        "promotion_provenance" if "promotion_provenance" in payload else "promotion"
    )
    if promotion_key not in payload:
        _fail(path, "native accepted experience requires promotion provenance")
    promotion = _require_object(payload[promotion_key], path=f"{path}.{promotion_key}")
    caller = promotion.get("caller", promotion.get("promoted_by"))
    promoted_at = promotion.get("promoted_at", promotion.get("created_at"))
    _require_string(caller, path=f"{path}.{promotion_key}.caller")
    _validate_utc(promoted_at, path=f"{path}.{promotion_key}.promoted_at")


class _ExperienceV1(_JsonRecord):
    KIND: ClassVar[str]

    @classmethod
    def _normalize(cls, payload: dict[str, Any]) -> None:
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            return
        for key in ("supporting_segment_refs", "opposing_segment_refs"):
            if isinstance(evidence.get(key), list):
                normalized: list[Any] = []
                seen: set[str] = set()
                for item in evidence[key]:
                    if isinstance(item, str):
                        if item in seen:
                            continue
                        seen.add(item)
                    normalized.append(item)
                evidence[key] = normalized

    @classmethod
    def _validate(cls, payload: dict[str, Any], **validation_context: Any) -> None:
        super()._validate(payload)
        path = cls.__name__
        _require_keys(
            payload,
            (
                "experience_id",
                "kind",
                "status",
                "confidence",
                "condition",
                "guidance",
                "predicted_effects",
                "evidence",
                "last_validated_at",
                "provenance",
            ),
            path=path,
        )
        _require_string(payload["experience_id"], path=f"{path}.experience_id")
        if payload["kind"] != cls.KIND:
            _fail(f"{path}.kind", f"must be {cls.KIND!r}")
        status = payload["status"]
        if status not in _EXPERIENCE_STATUSES:
            _fail(f"{path}.status", f"must be one of {sorted(_EXPERIENCE_STATUSES)}")
        if validation_context.get("reflector_output") and status != "candidate":
            _fail(f"{path}.status", "Reflector output must remain candidate")
        if payload["confidence"] not in _CONFIDENCE_LEVELS:
            _fail(f"{path}.confidence", f"must be one of {sorted(_CONFIDENCE_LEVELS)}")

        condition = _require_object(payload["condition"], path=f"{path}.condition")
        guidance = _require_object(payload["guidance"], path=f"{path}.guidance")
        predicted_effects = _require_list(
            payload["predicted_effects"],
            path=f"{path}.predicted_effects",
        )
        transferable = {
            "condition": condition,
            "guidance": guidance,
            "predicted_effects": predicted_effects,
        }
        forbidden = _find_forbidden_transfer_fields(
            transferable,
            path=path,
        )
        if forbidden:
            _fail(
                path,
                "transferable payload contains trajectory-private or forbidden field(s): "
                + ", ".join(forbidden),
            )
        _validate_transfer_condition(condition, path=f"{path}.condition")
        _validate_predicted_effects(
            predicted_effects,
            path=f"{path}.predicted_effects",
        )

        evidence = _require_object(payload["evidence"], path=f"{path}.evidence")
        _require_keys(
            evidence,
            (
                "supporting_segment_refs",
                "opposing_segment_refs",
                "support_count",
                "opposing_count",
                "evaluation_status",
                "evaluation_ref",
            ),
            path=f"{path}.evidence",
        )
        supporting = _deduplicate_strings(
            evidence["supporting_segment_refs"],
            path=f"{path}.evidence.supporting_segment_refs",
        )
        opposing = _deduplicate_strings(
            evidence["opposing_segment_refs"],
            path=f"{path}.evidence.opposing_segment_refs",
        )
        if set(supporting) & set(opposing):
            _fail(
                f"{path}.evidence",
                "supporting and opposing segment references must be disjoint",
            )
        support_count = _require_int(
            evidence["support_count"], path=f"{path}.evidence.support_count", minimum=0
        )
        opposing_count = _require_int(
            evidence["opposing_count"],
            path=f"{path}.evidence.opposing_count",
            minimum=0,
        )
        if support_count != len(supporting):
            _fail(
                f"{path}.evidence.support_count",
                "must equal the number of deduplicated supporting refs",
            )
        if opposing_count != len(opposing):
            _fail(
                f"{path}.evidence.opposing_count",
                "must equal the number of deduplicated opposing refs",
            )
        evaluation_status = evidence["evaluation_status"]
        if evaluation_status not in _EVALUATION_STATUSES:
            _fail(
                f"{path}.evidence.evaluation_status",
                f"must be one of {sorted(_EVALUATION_STATUSES)}",
            )
        evaluation_ref = evidence["evaluation_ref"]
        if evaluation_ref is not None:
            _require_string(evaluation_ref, path=f"{path}.evidence.evaluation_ref")

        if payload["last_validated_at"] is not None:
            _validate_utc(
                payload["last_validated_at"], path=f"{path}.last_validated_at"
            )
        provenance = ProvenanceV1.from_dict(payload["provenance"]).to_dict()

        legacy = provenance.get("legacy_record") is True
        if status == "accepted" and not legacy:
            if evaluation_status != "passed":
                _fail(
                    f"{path}.evidence.evaluation_status",
                    "native accepted experience requires passed evaluation",
                )
            if not isinstance(evaluation_ref, str) or not evaluation_ref.strip():
                _fail(
                    f"{path}.evidence.evaluation_ref",
                    "native accepted experience requires an evaluation ref",
                )
            _validate_promotion(payload, path=path)
        if status == "deprecated":
            reason = payload.get("deprecation_reason")
            if reason is None and isinstance(payload.get("deprecation"), dict):
                reason = payload["deprecation"].get("reason")
            _require_string(reason, path=f"{path}.deprecation_reason")

        cls._validate_kind_specific(payload, guidance=guidance, provenance=provenance)

    @classmethod
    def _validate_kind_specific(
        cls,
        payload: dict[str, Any],
        *,
        guidance: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        del payload, guidance, provenance

    @property
    def stable_id(self) -> str:
        return stable_experience_id(
            self._data["kind"],
            self._data["condition"],
            self._data["guidance"],
            self._data["predicted_effects"],
        )

    @property
    def retrieval_eligible(self) -> bool:
        provenance = self._data["provenance"]
        return (
            self._data["status"] == "accepted"
            and provenance["source_kind"] != "benchmark_expert"
            and provenance["oracle_derived"] is False
            and provenance["expert_derived"] is False
        )


class ProcedureExperienceV1(_ExperienceV1):
    SCHEMA = "roboharn_evo/experience/procedure/v1"
    KIND = "procedure"

    @classmethod
    def _validate_kind_specific(
        cls,
        payload: dict[str, Any],
        *,
        guidance: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        del payload, provenance
        path = f"{cls.__name__}.guidance"
        steps = _require_list(
            guidance.get("ordered_steps"),
            path=f"{path}.ordered_steps",
            nonempty=True,
        )
        for index, step in enumerate(steps):
            step_path = f"{path}.ordered_steps[{index}]"
            step_obj = _require_object(step, path=step_path)
            _require_string(
                step_obj.get("action_pattern"),
                path=f"{step_path}.action_pattern",
            )
            for key in ("manipulation_phase", "instruction"):
                if key in step_obj:
                    _require_string(step_obj[key], path=f"{step_path}.{key}")
        avoid = _require_list(guidance.get("avoid", []), path=f"{path}.avoid")
        for index, item in enumerate(avoid):
            _require_string(item, path=f"{path}.avoid[{index}]")
        if "feedback_policy" in guidance:
            _validate_feedback_policy(
                guidance["feedback_policy"],
                path=f"{path}.feedback_policy",
            )
            action_patterns = [str(step["action_pattern"]) for step in steps]
            if len(action_patterns) != len(set(action_patterns)):
                _fail(
                    f"{path}.ordered_steps",
                    "feedback procedure action_pattern values must be unique",
                )
            known_patterns = set(action_patterns)
            for index, rule in enumerate(
                guidance["feedback_policy"]["observation_rules"]
            ):
                referenced = str(rule["after_action_pattern"])
                if referenced not in known_patterns:
                    _fail(
                        f"{path}.feedback_policy.observation_rules[{index}]."
                        "after_action_pattern",
                        "must reference an ordered_steps action_pattern",
                    )


class RecoveryExperienceV1(_ExperienceV1):
    SCHEMA = "roboharn_evo/experience/recovery/v1"
    KIND = "recovery"

    @classmethod
    def _validate_kind_specific(
        cls,
        payload: dict[str, Any],
        *,
        guidance: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        path = f"{cls.__name__}.guidance"
        recommendations = _require_list(
            guidance.get("recommendations"),
            path=f"{path}.recommendations",
            nonempty=True,
        )
        for index, recommendation in enumerate(recommendations):
            _require_string(
                recommendation,
                path=f"{path}.recommendations[{index}]",
            )
        avoid = _require_list(guidance.get("avoid", []), path=f"{path}.avoid")
        for index, item in enumerate(avoid):
            _require_string(item, path=f"{path}.avoid[{index}]")
        for key in ("recovery_workflow", "post_recovery_intent"):
            if key in guidance:
                _require_string(
                    guidance[key],
                    path=f"{path}.{key}",
                    nonempty=False,
                )
        if provenance["source_kind"] not in {"agent_failure", "mixed"}:
            _fail(
                f"{cls.__name__}.provenance.source_kind",
                "recovery candidates require a failure or mixed source; expert/success-only input yields procedure candidates",
            )
        source_segments = {
            ref["segment_id"]
            for ref in provenance["source_refs"]
            if isinstance(ref.get("segment_id"), str) and ref["segment_id"]
        }
        support_segments = set(payload["evidence"]["supporting_segment_refs"])
        overlap = sorted(source_segments & support_segments)
        if overlap:
            _fail(
                f"{cls.__name__}.evidence.supporting_segment_refs",
                "failure source cannot support its own recovery recommendation: "
                + ", ".join(overlap),
            )


class ExpertBootstrapManifestV1(_JsonRecord):
    """Explicit allowlist for offline expert import; no directory discovery."""

    SCHEMA = "roboharn_evo/expert_bootstrap_manifest/v1"

    @classmethod
    def _validate(cls, payload: dict[str, Any], **validation_context: Any) -> None:
        del validation_context
        super()._validate(payload)
        path = cls.__name__
        _require_keys(payload, ("dataset_id", "dataset_root", "entries"), path=path)
        _require_string(payload["dataset_id"], path=f"{path}.dataset_id")
        _require_string(payload["dataset_root"], path=f"{path}.dataset_root")
        if "manifest_id" in payload:
            _require_string(payload["manifest_id"], path=f"{path}.manifest_id")
        if "created_at" in payload:
            _validate_utc(payload["created_at"], path=f"{path}.created_at")
        if "benchmark" in payload:
            benchmark = _require_string(
                payload["benchmark"],
                path=f"{path}.benchmark",
            )
            if benchmark != "rmbench":
                _fail(f"{path}.benchmark", "must be 'rmbench' when present")
        if (
            "license_status" in payload
            and payload["license_status"] not in _LICENSE_STATUSES
        ):
            _fail(
                f"{path}.license_status", f"must be one of {sorted(_LICENSE_STATUSES)}"
            )

        required_artifacts = {
            "hdf5",
            "planner_path",
            "language_annotation",
            "instruction",
        }
        allowed_artifacts = required_artifacts | {"qwen_annotation"}
        episode_keys: set[str] = set()
        source_ids: set[str] = set()
        content_hashes: set[str] = set()
        relative_paths: set[str] = set()
        shared_artifacts: dict[str, dict[str, Any]] = {}
        referenced_shared_artifacts: set[str] = set()
        if "shared_artifacts" in payload:
            raw_shared = _require_object(
                payload["shared_artifacts"],
                path=f"{path}.shared_artifacts",
            )
            if not raw_shared:
                _fail(f"{path}.shared_artifacts", "must not be empty when present")
            for shared_id, shared_value in raw_shared.items():
                shared_path = f"{path}.shared_artifacts.{shared_id}"
                _require_string(shared_id, path=f"{path}.shared_artifacts key")
                shared = _require_object(shared_value, path=shared_path)
                _require_keys(
                    shared,
                    ("artifact_role", "relative_path", "size_bytes", "sha256"),
                    path=shared_path,
                )
                artifact_role = _require_string(
                    shared["artifact_role"],
                    path=f"{shared_path}.artifact_role",
                )
                if artifact_role not in allowed_artifacts:
                    _fail(
                        f"{shared_path}.artifact_role",
                        f"must be one of {sorted(allowed_artifacts)}",
                    )
                _validate_relative_path(
                    shared["relative_path"],
                    path=f"{shared_path}.relative_path",
                )
                digest = _validate_sha256(
                    shared["sha256"],
                    path=f"{shared_path}.sha256",
                )
                _require_int(
                    shared["size_bytes"],
                    path=f"{shared_path}.size_bytes",
                    minimum=0,
                )
                relative_path = shared["relative_path"]
                if relative_path in relative_paths:
                    _fail(
                        shared_path,
                        f"duplicate artifact relative_path {relative_path!r} in manifest",
                    )
                if digest in content_hashes:
                    _fail(
                        shared_path, f"duplicate content SHA-256 {digest!r} in manifest"
                    )
                artifact_source_id = shared.get("source_ref_id")
                if artifact_source_id is not None:
                    _require_string(
                        artifact_source_id,
                        path=f"{shared_path}.source_ref_id",
                    )
                    if artifact_source_id in source_ids:
                        _fail(
                            shared_path,
                            f"duplicate source_ref_id {artifact_source_id!r} in manifest",
                        )
                    source_ids.add(artifact_source_id)
                relative_paths.add(relative_path)
                content_hashes.add(digest)
                shared_artifacts[shared_id] = shared

        entries = _require_list(
            payload["entries"], path=f"{path}.entries", nonempty=True
        )
        for index, entry_value in enumerate(entries):
            entry_path = f"{path}.entries[{index}]"
            entry = _require_object(entry_value, path=entry_path)
            _require_keys(
                entry,
                (
                    "task",
                    "task_config",
                    "episode_id",
                    "seed",
                    "instruction_variant",
                    "artifacts",
                    "source_kind",
                    "source_subtype",
                    "expert_derived",
                    "oracle_derived",
                ),
                path=entry_path,
            )
            task = _require_string(entry["task"], path=f"{entry_path}.task")
            task_config = _require_string(
                entry["task_config"], path=f"{entry_path}.task_config"
            )
            for key in ("episode_id", "seed"):
                value = entry[key]
                if not (
                    isinstance(value, str)
                    or (isinstance(value, int) and not isinstance(value, bool))
                ):
                    _fail(f"{entry_path}.{key}", "must be an integer or string")
            _require_string(
                entry["instruction_variant"], path=f"{entry_path}.instruction_variant"
            )
            if "task_family" in entry:
                _require_string(entry["task_family"], path=f"{entry_path}.task_family")
            if "instruction_index" in entry:
                _require_int(
                    entry["instruction_index"],
                    path=f"{entry_path}.instruction_index",
                    minimum=0,
                )
            for key in ("benchmark", "split"):
                if key in entry:
                    value = _require_string(entry[key], path=f"{entry_path}.{key}")
                    if key == "benchmark" and value != "rmbench":
                        _fail(
                            f"{entry_path}.benchmark",
                            "must be 'rmbench' when present",
                        )
            if entry["source_kind"] != "benchmark_expert":
                _fail(
                    f"{entry_path}.source_kind",
                    "expert bootstrap entry must be benchmark_expert",
                )
            if entry["source_subtype"] != "benchmark_scripted_expert":
                _fail(
                    f"{entry_path}.source_subtype",
                    "RMBench expert bootstrap entry must be benchmark_scripted_expert",
                )
            if entry["expert_derived"] is not True:
                _fail(
                    f"{entry_path}.expert_derived",
                    "expert bootstrap entry must preserve true",
                )
            if entry["oracle_derived"] is not True:
                _fail(
                    f"{entry_path}.oracle_derived",
                    "scripted expert entry must preserve true",
                )

            trajectory_id = entry.get("trajectory_id")
            if trajectory_id is not None:
                _require_string(trajectory_id, path=f"{entry_path}.trajectory_id")
                episode_key = f"trajectory:{trajectory_id}"
            else:
                episode_key = f"episode:{task}:{task_config}:{type(entry['episode_id']).__name__}:{entry['episode_id']}"
            if episode_key in episode_keys:
                _fail(entry_path, f"duplicate episode identity {episode_key!r}")
            episode_keys.add(episode_key)

            artifacts = entry["artifacts"]
            artifacts = _require_object(artifacts, path=f"{entry_path}.artifacts")
            missing_artifacts = sorted(required_artifacts - set(artifacts))
            unknown_artifacts = sorted(set(artifacts) - allowed_artifacts)
            if missing_artifacts:
                _fail(
                    f"{entry_path}.artifacts",
                    "missing required artifact(s): " + ", ".join(missing_artifacts),
                )
            if unknown_artifacts:
                _fail(
                    f"{entry_path}.artifacts",
                    "unknown artifact role(s): " + ", ".join(unknown_artifacts),
                )
            artifact_items = list(artifacts.items())
            entry_hashes: set[str] = set()
            entry_artifact_source_ids: dict[str, str] = {}
            for artifact_name, artifact_value in artifact_items:
                artifact_path = f"{entry_path}.artifacts.{artifact_name}"
                artifact = _require_object(artifact_value, path=artifact_path)
                if "shared_artifact_ref" in artifact:
                    if set(artifact) != {"shared_artifact_ref"}:
                        _fail(
                            artifact_path,
                            "shared artifact reference must contain only shared_artifact_ref",
                        )
                    shared_id = _require_string(
                        artifact["shared_artifact_ref"],
                        path=f"{artifact_path}.shared_artifact_ref",
                    )
                    shared = shared_artifacts.get(shared_id)
                    if shared is None:
                        _fail(
                            f"{artifact_path}.shared_artifact_ref",
                            f"unknown shared artifact {shared_id!r}",
                        )
                    if shared["artifact_role"] != artifact_name:
                        _fail(
                            f"{artifact_path}.shared_artifact_ref",
                            f"shared artifact role {shared['artifact_role']!r} does not match {artifact_name!r}",
                        )
                    referenced_shared_artifacts.add(shared_id)
                    digest = shared["sha256"]
                    if digest in entry_hashes:
                        _fail(
                            artifact_path,
                            f"duplicate content SHA-256 {digest!r} within manifest entry",
                        )
                    entry_hashes.add(digest)
                    artifact_source_id = shared.get("source_ref_id")
                    if artifact_source_id is not None:
                        entry_artifact_source_ids[artifact_source_id] = digest
                    continue
                _require_keys(
                    artifact,
                    ("relative_path", "size_bytes", "sha256"),
                    path=artifact_path,
                )
                _validate_relative_path(
                    artifact["relative_path"], path=f"{artifact_path}.relative_path"
                )
                digest = _validate_sha256(
                    artifact["sha256"], path=f"{artifact_path}.sha256"
                )
                _require_int(
                    artifact["size_bytes"],
                    path=f"{artifact_path}.size_bytes",
                    minimum=0,
                )
                relative_path = artifact["relative_path"]
                artifact_source_id = artifact.get("source_ref_id")
                if artifact_source_id is not None:
                    _require_string(
                        artifact_source_id, path=f"{artifact_path}.source_ref_id"
                    )
                    if artifact_source_id in source_ids:
                        _fail(
                            artifact_path,
                            f"duplicate source_ref_id {artifact_source_id!r} in manifest",
                        )
                    source_ids.add(artifact_source_id)
                    entry_artifact_source_ids[artifact_source_id] = digest
                if relative_path in relative_paths:
                    _fail(
                        artifact_path,
                        f"duplicate artifact relative_path {relative_path!r} in manifest",
                    )
                if digest in content_hashes or digest in entry_hashes:
                    _fail(
                        artifact_path,
                        f"duplicate content SHA-256 {digest!r} in manifest",
                    )
                relative_paths.add(relative_path)
                entry_hashes.add(digest)
            content_hashes.update(entry_hashes)

            if "source_refs" not in entry:
                continue
            refs = _validate_source_refs(
                entry["source_refs"], path=f"{entry_path}.source_refs"
            )
            for ref_number, ref in enumerate(refs):
                source_id = ref["source_ref_id"]
                content_hash = ref["content_sha256"]
                descriptor_digest = entry_artifact_source_ids.get(source_id)
                if descriptor_digest is not None and descriptor_digest != content_hash:
                    _fail(
                        f"{entry_path}.source_refs[{ref_number}]",
                        f"source_ref_id {source_id!r} disagrees with its artifact digest",
                    )
                if source_id in source_ids and descriptor_digest is None:
                    _fail(
                        f"{entry_path}.source_refs[{ref_number}]",
                        f"duplicate source_ref_id {source_id!r} in manifest",
                    )
                if descriptor_digest is None:
                    source_ids.add(source_id)
                # ``source_refs`` is an additive normalized view of artifacts;
                # the same content digest may therefore appear once in each.
                if content_hash not in entry_hashes and content_hash in content_hashes:
                    _fail(
                        f"{entry_path}.source_refs[{ref_number}]",
                        f"duplicate content_sha256 {content_hash!r} across manifest entries",
                    )
                if content_hash not in entry_hashes:
                    content_hashes.add(content_hash)

        unreferenced_shared = sorted(
            set(shared_artifacts) - referenced_shared_artifacts
        )
        if unreferenced_shared:
            _fail(
                f"{path}.shared_artifacts",
                "unreferenced shared artifact(s): " + ", ".join(unreferenced_shared),
            )


def stable_experience_id(
    kind: str,
    condition: Mapping[str, Any],
    guidance: Mapping[str, Any],
    predicted_effects: Sequence[Any],
    *,
    prefix: str = "exp",
) -> str:
    """Return a stable ID based only on immutable experience semantics.

    Object keys are sorted, list order is retained, UTF-8 is used, and mutable
    lifecycle/evidence/confidence fields are intentionally absent.
    """

    _require_string(kind, path="kind")
    identity = {
        "kind": kind,
        "condition": copy.deepcopy(dict(condition))
        if isinstance(condition, Mapping)
        else condition,
        "guidance": copy.deepcopy(dict(guidance))
        if isinstance(guidance, Mapping)
        else guidance,
        "predicted_effects": copy.deepcopy(list(predicted_effects))
        if isinstance(predicted_effects, Sequence)
        and not isinstance(predicted_effects, (str, bytes, bytearray))
        else predicted_effects,
    }
    _validate_json(identity, path="experience_identity")
    if not isinstance(identity["condition"], dict):
        _fail("condition", "must be an object")
    if not isinstance(identity["guidance"], dict):
        _fail("guidance", "must be an object")
    if not isinstance(identity["predicted_effects"], list):
        _fail("predicted_effects", "must be an array")
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    clean_prefix = re.sub(r"[^a-z0-9]+", "_", str(prefix).strip().lower()).strip("_")
    return f"{clean_prefix}_{digest}" if clean_prefix else digest


def validate_reflector_candidate(
    record: Mapping[str, Any] | _ExperienceV1,
) -> _ExperienceV1:
    """Validate an experience specifically at the Reflector output boundary."""

    payload = record.to_dict() if isinstance(record, _ExperienceV1) else record
    if not isinstance(payload, Mapping):
        _fail("record", "must be an experience object")
    schema = payload.get("schema")
    if schema in {ProcedureExperienceV1.SCHEMA, "tcm/experience/procedure/v1"}:
        return ProcedureExperienceV1.from_dict(payload, reflector_output=True)
    if schema in {RecoveryExperienceV1.SCHEMA, "tcm/experience/recovery/v1"}:
        return RecoveryExperienceV1.from_dict(payload, reflector_output=True)
    _fail("record.schema", "unsupported Reflector experience schema")


def validate_provenance_v1(payload: Mapping[str, Any]) -> ProvenanceV1:
    return ProvenanceV1.from_dict(payload)  # type: ignore[return-value]


def validate_trajectory_record_v1(payload: Mapping[str, Any]) -> TrajectoryRecordV1:
    return TrajectoryRecordV1.from_dict(payload)  # type: ignore[return-value]


def validate_subtask_segment_v1(
    payload: Mapping[str, Any], *, evidence_index: Any | None = None
) -> SubtaskSegmentV1:
    return SubtaskSegmentV1.from_dict(payload, evidence_index=evidence_index)  # type: ignore[return-value]


def validate_procedure_experience_v1(
    payload: Mapping[str, Any], *, reflector_output: bool = False
) -> ProcedureExperienceV1:
    return ProcedureExperienceV1.from_dict(payload, reflector_output=reflector_output)  # type: ignore[return-value]


def validate_recovery_experience_v1(
    payload: Mapping[str, Any], *, reflector_output: bool = False
) -> RecoveryExperienceV1:
    return RecoveryExperienceV1.from_dict(payload, reflector_output=reflector_output)  # type: ignore[return-value]


def validate_expert_bootstrap_manifest_v1(
    payload: Mapping[str, Any],
) -> ExpertBootstrapManifestV1:
    return ExpertBootstrapManifestV1.from_dict(payload)  # type: ignore[return-value]


__all__ = [
    "ExpertBootstrapManifestV1",
    "ProcedureExperienceV1",
    "ProvenanceV1",
    "RecoveryExperienceV1",
    "SchemaValidationError",
    "SubtaskSegmentV1",
    "TrajectoryRecordV1",
    "stable_experience_id",
    "validate_expert_bootstrap_manifest_v1",
    "validate_procedure_experience_v1",
    "validate_provenance_v1",
    "validate_recovery_experience_v1",
    "validate_reflector_candidate",
    "validate_subtask_segment_v1",
    "validate_trajectory_record_v1",
]

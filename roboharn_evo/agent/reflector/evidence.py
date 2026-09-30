"""Stable evidence references and an in-memory v1 resolver.

The index is intentionally storage- and backend-neutral.  It validates JSON
entries and resolves only refs explicitly present in the supplied trajectory
record; it never scans a directory or mutates raw trace events.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    SchemaValidationError,
    _json_copy,
    _require_int,
    _require_list,
    _require_object,
    _require_string,
    _validate_sha256,
)


_REF_RE = re.compile(
    r"^(?P<trajectory_id>[^#]+)#(?:"
    r"(?P<simple_kind>event|frame)/(?P<simple_index>\d{6})"
    r"|plan/(?P<arm>[A-Za-z0-9_.-]+)/(?P<plan_index>\d{6})"
    r")$"
)
_EVIDENCE_ENTRY_SCHEMA = "roboharn_evo/evidence_index_entry/v1"
_EVIDENCE_ENTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvidenceRefV1:
    """Parsed form of a stable trajectory-local evidence reference."""

    value: str
    trajectory_id: str = field(init=False)
    kind: str = field(init=False)
    index: int = field(init=False)
    arm: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise SchemaValidationError("evidence_ref: must be a string")
        match = _REF_RE.fullmatch(self.value)
        if match is None:
            raise SchemaValidationError(
                "evidence_ref: expected '<trajectory>#event/000000', "
                "'<trajectory>#frame/000000', or '<trajectory>#plan/<arm>/000000'"
            )
        trajectory_id = match.group("trajectory_id")
        if not trajectory_id.strip():
            raise SchemaValidationError("evidence_ref: trajectory_id must not be empty")
        simple_kind = match.group("simple_kind")
        kind = simple_kind if simple_kind is not None else "plan"
        index_text = match.group("simple_index") or match.group("plan_index")
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "index", int(index_text))
        object.__setattr__(self, "arm", match.group("arm"))

    @classmethod
    def parse(cls, value: str | "EvidenceRefV1") -> "EvidenceRefV1":
        return value if isinstance(value, cls) else cls(value)

    @classmethod
    def from_string(cls, value: str) -> "EvidenceRefV1":
        return cls(value)

    @property
    def evidence_ref(self) -> str:
        return self.value

    @property
    def ordinal(self) -> int:
        return self.index

    def to_json_value(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class EvidenceIndex:
    """Validated resolver for evidence entries belonging to one trajectory."""

    ENTRY_SCHEMA = _EVIDENCE_ENTRY_SCHEMA
    ENTRY_SCHEMA_VERSION = _EVIDENCE_ENTRY_SCHEMA_VERSION
    __slots__ = ("_entries", "_source_ref_ids", "_trajectory_id")

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        source_refs: Sequence[Mapping[str, Any] | str] | None = None,
        trajectory_id: str | None = None,
    ) -> None:
        if isinstance(entries, (str, bytes, bytearray)) or not isinstance(
            entries, Sequence
        ):
            raise SchemaValidationError("evidence_index: entries must be an array")
        if trajectory_id is not None:
            _require_string(trajectory_id, path="evidence_index.trajectory_id")

        source_ref_ids = self._collect_source_ref_ids(source_refs)
        by_ref: dict[str, dict[str, Any]] = {}
        inferred_trajectory_id = trajectory_id
        for number, raw_entry in enumerate(entries):
            path = f"evidence_index[{number}]"
            entry = _json_copy(raw_entry, path=path)
            for key in (
                "schema",
                "schema_version",
                "evidence_ref",
                "source_ref_id",
                "locator",
            ):
                if key not in entry:
                    raise SchemaValidationError(
                        f"{path}: missing required field {key!r}"
                    )
            if entry["schema"] not in {
                _EVIDENCE_ENTRY_SCHEMA, "tcm/evidence_index_entry/v1"
            }:
                raise SchemaValidationError(
                    f"{path}.schema: expected {_EVIDENCE_ENTRY_SCHEMA!r}, "
                    f"got {entry['schema']!r}"
                )
            if entry["schema_version"] != _EVIDENCE_ENTRY_SCHEMA_VERSION or isinstance(
                entry["schema_version"], bool
            ):
                raise SchemaValidationError(
                    f"{path}.schema_version: unsupported major version; expected "
                    f"integer {_EVIDENCE_ENTRY_SCHEMA_VERSION}"
                )
            ref_text = _require_string(
                entry["evidence_ref"], path=f"{path}.evidence_ref"
            )
            parsed = EvidenceRefV1.parse(ref_text)
            if inferred_trajectory_id is None:
                inferred_trajectory_id = parsed.trajectory_id
            if parsed.trajectory_id != inferred_trajectory_id:
                raise SchemaValidationError(
                    f"{path}.evidence_ref: belongs to trajectory {parsed.trajectory_id!r}, "
                    f"expected {inferred_trajectory_id!r}"
                )
            if ref_text in by_ref:
                raise SchemaValidationError(
                    f"{path}.evidence_ref: duplicate reference {ref_text!r}"
                )

            source_ref_id = _require_string(
                entry["source_ref_id"], path=f"{path}.source_ref_id"
            )
            if source_refs is not None and source_ref_id not in source_ref_ids:
                raise SchemaValidationError(
                    f"{path}.source_ref_id: {source_ref_id!r} is not present in this trajectory's provenance"
                )
            locator = _require_object(entry["locator"], path=f"{path}.locator")
            if "kind" not in locator:
                raise SchemaValidationError(
                    f"{path}.locator: missing required field 'kind'"
                )
            _require_string(locator["kind"], path=f"{path}.locator.kind")
            self._validate_locator(parsed, locator, path=f"{path}.locator")
            by_ref[ref_text] = entry

        self._entries = by_ref
        self._source_ref_ids = frozenset(source_ref_ids)
        self._trajectory_id = inferred_trajectory_id

    @staticmethod
    def _collect_source_ref_ids(
        source_refs: Sequence[Mapping[str, Any] | str] | None,
    ) -> set[str]:
        if source_refs is None:
            return set()
        if isinstance(source_refs, (str, bytes, bytearray)) or not isinstance(
            source_refs, Sequence
        ):
            raise SchemaValidationError("source_refs: must be an array")
        result: set[str] = set()
        for number, source in enumerate(source_refs):
            path = f"source_refs[{number}]"
            if isinstance(source, str):
                source_ref_id = _require_string(source, path=path)
            elif isinstance(source, Mapping):
                if "source_ref_id" not in source:
                    raise SchemaValidationError(
                        f"{path}: missing required field 'source_ref_id'"
                    )
                source_ref_id = _require_string(
                    source["source_ref_id"], path=f"{path}.source_ref_id"
                )
            else:
                raise SchemaValidationError(
                    f"{path}: must be a source_ref_id string or object"
                )
            if source_ref_id in result:
                raise SchemaValidationError(
                    f"{path}: duplicate source_ref_id {source_ref_id!r}"
                )
            result.add(source_ref_id)
        return result

    @staticmethod
    def _validate_locator(
        parsed: EvidenceRefV1, locator: Mapping[str, Any], *, path: str
    ) -> None:
        """Require the canonical locator shape implied by an evidence ref.

        A reference namespace is semantic, not merely cosmetic.  Accepting an
        ``#event`` ref backed by a frame (or a ``#plan`` ref with no arm) would
        let a caller relabel evidence without changing the stable ref.  The v1
        mapping is therefore exact and all ref-carried discriminators are
        mandatory in the locator.
        """

        expected_kind = {
            "event": "trace_event",
            "frame": "hdf5_frame",
            "plan": "planner_segment",
        }[parsed.kind]
        locator_kind = str(locator["kind"])
        if locator_kind != expected_kind:
            raise SchemaValidationError(
                f"{path}.kind: {locator_kind!r} does not match "
                f"{parsed.kind!r} evidence ref; expected {expected_kind!r}"
            )

        if parsed.kind == "event":
            if "event_ordinal" not in locator:
                raise SchemaValidationError(
                    f"{path}: trace_event locators require event_ordinal"
                )
            event_ordinal = _require_int(
                locator["event_ordinal"],
                path=f"{path}.event_ordinal",
                minimum=0,
            )
            if event_ordinal != parsed.index:
                raise SchemaValidationError(
                    f"{path}.event_ordinal: {event_ordinal} does not match "
                    f"evidence ref ordinal {parsed.index}"
                )
            if "event_index" in locator:
                event_index = _require_int(
                    locator["event_index"],
                    path=f"{path}.event_index",
                    minimum=0,
                )
                if event_index != parsed.index:
                    raise SchemaValidationError(
                        f"{path}.event_index: {event_index} does not match "
                        f"evidence ref ordinal {parsed.index}"
                    )
            if any(key in locator for key in ("frame_index", "segment_index", "arm")):
                raise SchemaValidationError(
                    f"{path}: trace_event locator contains a discriminator for another ref kind"
                )
            has_line = "line_number" in locator
            has_hash = "content_sha256" in locator
            if has_line != has_hash:
                raise SchemaValidationError(
                    f"{path}: trace_event JSONL location requires both "
                    "line_number and content_sha256"
                )
            if has_line:
                _require_int(
                    locator["line_number"],
                    path=f"{path}.line_number",
                    minimum=1,
                )
                _validate_sha256(
                    locator["content_sha256"],
                    path=f"{path}.content_sha256",
                )
            return

        if parsed.kind == "frame":
            if "frame_index" not in locator or "dataset_keys" not in locator:
                raise SchemaValidationError(
                    f"{path}: hdf5_frame locators require frame_index and dataset_keys"
                )
            frame_index = _require_int(
                locator["frame_index"],
                path=f"{path}.frame_index",
                minimum=0,
            )
            if frame_index != parsed.index:
                raise SchemaValidationError(
                    f"{path}.frame_index: {frame_index} does not match "
                    f"evidence ref ordinal {parsed.index}"
                )
            if any(
                key in locator
                for key in ("event_ordinal", "event_index", "segment_index", "arm")
            ):
                raise SchemaValidationError(
                    f"{path}: hdf5_frame locator contains a discriminator for another ref kind"
                )
            dataset_keys = _require_list(
                locator["dataset_keys"],
                path=f"{path}.dataset_keys",
                nonempty=True,
            )
            for number, dataset_key in enumerate(dataset_keys):
                _require_string(dataset_key, path=f"{path}.dataset_keys[{number}]")
            return

        if "segment_index" not in locator or "arm" not in locator:
            raise SchemaValidationError(
                f"{path}: planner_segment locators require segment_index and arm"
            )
        segment_index = _require_int(
            locator["segment_index"],
            path=f"{path}.segment_index",
            minimum=0,
        )
        if segment_index != parsed.index:
            raise SchemaValidationError(
                f"{path}.segment_index: {segment_index} does not match "
                f"evidence ref ordinal {parsed.index}"
            )
        arm = _require_string(locator["arm"], path=f"{path}.arm")
        if arm != parsed.arm:
            raise SchemaValidationError(
                f"{path}.arm: {arm!r} does not match evidence ref arm {parsed.arm!r}"
            )
        if any(
            key in locator for key in ("event_ordinal", "event_index", "frame_index")
        ):
            raise SchemaValidationError(
                f"{path}: planner_segment locator contains a discriminator for another ref kind"
            )

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[Mapping[str, Any]],
        *,
        source_refs: Sequence[Mapping[str, Any] | str] | None = None,
        trajectory_id: str | None = None,
    ) -> "EvidenceIndex":
        return cls(entries, source_refs=source_refs, trajectory_id=trajectory_id)

    @classmethod
    def from_trajectory_record(cls, record: Mapping[str, Any] | Any) -> "EvidenceIndex":
        payload = (
            record.to_dict()
            if hasattr(record, "to_dict")
            else copy.deepcopy(dict(record))
        )
        try:
            trajectory_id = payload["trajectory"]["trajectory_id"]
            source_refs = payload["provenance"]["source_refs"]
            entries = payload["evidence_index"]
        except (KeyError, TypeError) as exc:
            raise SchemaValidationError(
                "trajectory record lacks evidence index context"
            ) from exc
        return cls.from_entries(
            entries, source_refs=source_refs, trajectory_id=trajectory_id
        )

    @property
    def trajectory_id(self) -> str | None:
        return self._trajectory_id

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def resolve(self, ref: str | EvidenceRefV1) -> dict[str, Any]:
        parsed = EvidenceRefV1.parse(ref)
        if (
            self._trajectory_id is not None
            and parsed.trajectory_id != self._trajectory_id
        ):
            raise SchemaValidationError(
                f"evidence_ref {parsed.value!r} belongs to trajectory {parsed.trajectory_id!r}, "
                f"expected {self._trajectory_id!r}"
            )
        try:
            return copy.deepcopy(self._entries[parsed.value])
        except KeyError as exc:
            raise SchemaValidationError(
                f"evidence_ref {parsed.value!r} does not exist in this index"
            ) from exc

    def validate_refs(
        self,
        refs: Sequence[str | EvidenceRefV1],
        *,
        trajectory_id: str | None = None,
        allow_duplicates: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        if isinstance(refs, (str, bytes, bytearray)) or not isinstance(refs, Sequence):
            raise SchemaValidationError("evidence refs must be an array")
        expected_trajectory = (
            trajectory_id if trajectory_id is not None else self._trajectory_id
        )
        resolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        for number, ref in enumerate(refs):
            parsed = EvidenceRefV1.parse(ref)
            if (
                expected_trajectory is not None
                and parsed.trajectory_id != expected_trajectory
            ):
                raise SchemaValidationError(
                    f"evidence refs[{number}] belongs to trajectory {parsed.trajectory_id!r}, "
                    f"expected {expected_trajectory!r}"
                )
            if not allow_duplicates and parsed.value in seen:
                raise SchemaValidationError(
                    f"evidence refs[{number}] duplicates {parsed.value!r}"
                )
            seen.add(parsed.value)
            resolved.append(self.resolve(parsed))
        return tuple(resolved)

    def to_entries(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self._entries.values()))

    def __contains__(self, ref: object) -> bool:
        try:
            parsed = EvidenceRefV1.parse(ref)  # type: ignore[arg-type]
        except (SchemaValidationError, TypeError):
            return False
        return parsed.value in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)


__all__ = ["EvidenceIndex", "EvidenceRefV1"]

"""Read-only deterministic retrieval for planner procedure experiences.

This module is intentionally separate from :mod:`roboharn_evo.agent.experience.retriever`,
which serves the legacy recovery-only experience library.  The only non-off
mode implemented here is ``candidate_dev``: an explicit development override
for inspecting unvalidated, expert-derived procedure candidates in ``/plan``.
It never promotes, mutates, or writes back an experience.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from roboharn_evo.agent.reflector.schemas import ProcedureExperienceV1, SchemaValidationError


OFF_MODE = "off"
CANDIDATE_DEV_MODE = "candidate_dev"
PROCEDURE_EXPERIENCE_MODES = frozenset({OFF_MODE, CANDIDATE_DEV_MODE})

SNAPSHOT_SCHEMA = "roboharn_evo/procedure_experience_snapshot/v1"
CONTEXT_SCHEMA = "roboharn_evo/procedure_experience_context/v1"
RETIREMENT_REGISTRY_SCHEMA = "roboharn_evo/procedure_experience_retirement_registry/v1"

_RETIREMENT_REGISTRY_PATH = (
    Path(os.environ["ROBOHARN_EVO_PROCEDURE_RETIREMENT_REGISTRY"]).expanduser()
    if os.environ.get("ROBOHARN_EVO_PROCEDURE_RETIREMENT_REGISTRY")
    else Path(__file__).resolve().parents[2] / "resources" / "configs" / "procedure_retirements.json"
)

ADVISORY_NOTICE = (
    "These are unvalidated candidate experiences derived from expert data. "
    "Treat them as advisory guidance only.\n"
    "Current observations, Scene Memory, and tool results take precedence."
)

DEFAULT_TOP_K = 3
DEFAULT_MAX_PROMPT_CHARS = 6000
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_SNAPSHOT_EXPERIENCES = 1000

ENV_MODE = "ROBOHARN_EVO_PROCEDURE_EXPERIENCE_MODE"
ENV_SNAPSHOT_MANIFEST = "ROBOHARN_EVO_PROCEDURE_EXPERIENCE_SNAPSHOT_MANIFEST"
ENV_TOP_K = "ROBOHARN_EVO_PROCEDURE_EXPERIENCE_TOP_K"
ENV_MAX_PROMPT_CHARS = "ROBOHARN_EVO_PROCEDURE_EXPERIENCE_MAX_PROMPT_CHARS"
ENV_EXPECTED_CONFIG_SHA256 = "ROBOHARN_EVO_PROCEDURE_EXPERIENCE_CONFIG_SHA256"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_UTC_SECOND_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_RAW_SHA256_TOKEN_RE = re.compile(
    r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])"
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]", re.IGNORECASE)
_LOCAL_PATH_RE = re.compile(
    r"""
    (?:^|[\s"'`(\[{:=>])
    (?:
        file://[^\s"'`<>),\]}]+
        |(?:/|~/|\.\.?/)[^\s"'`<>),\]}]+
        |[A-Za-z]:[\\/][^\s"'`<>),\]}]+
        |\\\\[^\s"'`<>),\]}]+
        |(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_PUBLIC_EXPERIENCE_ID_PATH_RE = re.compile(
    r"^(?:experience|experiences\[\d+\]|audit\.selection_reasons\[\d+\])"
    r"\.experience_id$"
)

_CONDITION_KEYS = frozenset(
    {"task_family", "subtask_type", "relation", "phase", "manipulation_phase"}
)
_STEP_KEYS = frozenset({"action_pattern", "instruction", "manipulation_phase"})
_FORBIDDEN_PROJECTION_KEYS = frozenset(
    {
        "source_ref",
        "source_refs",
        "seed",
        "episode_id",
        "sha256",
        "content_sha256",
        "evidence",
        "evidence_index",
        "annotation",
        "annotations",
        "provenance",
        "world_m",
        "world_coordinate",
        "world_coordinates",
        "track_id",
        "candidate_id",
        "operation_candidate_id",
        "executor_state",
        "executor_private",
        "relative_path",
        "local_path",
        "artifact_path",
        "trajectory_id",
        "segment_id",
    }
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
_CONTEXT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "experience_mode",
        "snapshot_id",
        "snapshot_manifest_sha256",
        "query_task_family",
        "selected_experience_ids",
        "selection_reasons",
        "projection_sha256",
        "projection_char_count",
        "estimated_token_count",
        "max_prompt_chars",
        "candidate_dev_override",
        "expert_prior_used",
        "cross_rollout_memory_enabled",
        "experiences",
    }
)
_AUDIT_KEYS = frozenset(
    {
        "experience_mode",
        "snapshot_id",
        "snapshot_manifest_sha256",
        "query_task_family",
        "selected_experience_ids",
        "selection_reasons",
        "projection_sha256",
        "projection_char_count",
        "estimated_token_count",
        "candidate_dev_override",
        "expert_prior_used",
        "cross_rollout_memory_enabled",
        "injection_count",
    }
)


class ProcedureExperienceError(ValueError):
    """Base class for fail-closed procedure-experience errors."""


class ProcedureExperienceConfigurationError(ProcedureExperienceError):
    """Raised when runtime candidate-dev configuration is incomplete or unsafe."""


class ProcedureExperienceSnapshotError(ProcedureExperienceError):
    """Raised when a snapshot or one of its bound files fails validation."""


class ProcedureExperienceContextError(ProcedureExperienceError):
    """Raised when a planner context is invalid or contains private fields."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not _SHA256_RE.fullmatch(text):
        raise ProcedureExperienceSnapshotError(
            f"{label} must be a 64-character lowercase SHA-256 digest"
        )
    return text


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProcedureExperienceSnapshotError(f"{label} must be a non-empty string")
    return value.strip()


def _read_regular_file(path: Path, *, label: str, max_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProcedureExperienceSnapshotError(f"cannot stat {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ProcedureExperienceSnapshotError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ProcedureExperienceSnapshotError(f"{label} must be a regular file: {path}")
    if metadata.st_size > max_bytes:
        raise ProcedureExperienceSnapshotError(
            f"{label} exceeds the {max_bytes}-byte read budget: {metadata.st_size}"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProcedureExperienceSnapshotError(f"cannot read {label}: {path}: {exc}") from exc


def _resolve_snapshot_member(manifest_path: Path, value: Any, *, label: str) -> Path:
    relative_text = _require_nonempty_string(value, label=label)
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative_text:
        raise ProcedureExperienceSnapshotError(
            f"{label} must be a normalized relative POSIX path inside the snapshot"
        )
    snapshot_root = manifest_path.parent.resolve()
    candidate = manifest_path.parent.joinpath(*relative.parts)
    try:
        candidate_metadata = candidate.lstat()
    except OSError as exc:
        raise ProcedureExperienceSnapshotError(
            f"cannot stat {label}: {relative_text}: {exc}"
        ) from exc
    if stat.S_ISLNK(candidate_metadata.st_mode):
        raise ProcedureExperienceSnapshotError(
            f"{label} must not be a symlink: {relative_text}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProcedureExperienceSnapshotError(
            f"{label} does not resolve to a snapshot member: {relative_text}: {exc}"
        ) from exc
    if resolved != snapshot_root and snapshot_root not in resolved.parents:
        raise ProcedureExperienceSnapshotError(
            f"{label} escapes the snapshot directory: {relative_text}"
        )
    return resolved


def _snapshot_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    candidate_file = manifest.get("candidate_experiences")
    phase_a_file = manifest.get("phase_a_run_manifest")
    if not isinstance(candidate_file, Mapping) or not isinstance(phase_a_file, Mapping):
        raise ProcedureExperienceSnapshotError(
            "snapshot candidate_experiences and phase_a_run_manifest must be objects"
        )
    return {
        "schema": manifest.get("schema", SNAPSHOT_SCHEMA),
        "schema_version": 1,
        "created_for": manifest.get("created_for"),
        "task_family": manifest.get("task_family"),
        "source": manifest.get("source"),
        "bootstrap_seeds": manifest.get("bootstrap_seeds"),
        "status": manifest.get("status"),
        "experience_count": manifest.get("experience_count"),
        "experience_ids": manifest.get("experience_ids"),
        "candidate_experiences_sha256": candidate_file.get("sha256"),
        "phase_a_run_manifest_sha256": phase_a_file.get("sha256"),
        "phase_a_run_id": manifest.get("phase_a_run_id"),
    }


def snapshot_id_for_manifest(manifest: Mapping[str, Any]) -> str:
    """Return the content-derived ID required by a v1 snapshot manifest."""

    task_family = str(manifest.get("task_family", "")).strip()
    if not task_family:
        raise ProcedureExperienceSnapshotError("snapshot task_family is required")
    digest = _sha256_bytes(_canonical_json_bytes(_snapshot_identity_payload(manifest)))
    safe_family = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_family).strip("_")
    return f"pexp_{safe_family}_{digest[:24]}"


@dataclass(frozen=True)
class ProcedureExperienceRuntimeConfig:
    """Resolved runtime policy for planner-side procedure experience."""

    mode: str = OFF_MODE
    snapshot_manifest: str = ""
    task_family: str = ""
    top_k: int = DEFAULT_TOP_K
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS

    def __post_init__(self) -> None:
        if self.mode not in PROCEDURE_EXPERIENCE_MODES:
            raise ProcedureExperienceConfigurationError(
                f"procedure experience mode must be one of {sorted(PROCEDURE_EXPERIENCE_MODES)}, got {self.mode!r}"
            )
        if self.top_k < 1:
            raise ProcedureExperienceConfigurationError("procedure experience top_k must be >= 1")
        if self.max_prompt_chars < 1:
            raise ProcedureExperienceConfigurationError(
                "procedure experience max_prompt_chars must be >= 1"
            )
        if self.mode == CANDIDATE_DEV_MODE:
            if not self.snapshot_manifest.strip():
                raise ProcedureExperienceConfigurationError(
                    "candidate_dev requires an explicit snapshot_manifest"
                )
            if not self.task_family.strip():
                raise ProcedureExperienceConfigurationError(
                    "candidate_dev requires an authoritative task_family"
                )

    def identity(self) -> dict[str, Any]:
        if self.mode == OFF_MODE:
            return {"mode": OFF_MODE}
        return {
            "mode": self.mode,
            "snapshot_manifest": str(Path(self.snapshot_manifest).expanduser().resolve()),
            "task_family": self.task_family,
            "top_k": self.top_k,
            "max_prompt_chars": self.max_prompt_chars,
        }

    @property
    def config_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.identity()))


def resolve_procedure_experience_config(
    raw_config: Mapping[str, Any] | None,
    *,
    task_family: str,
    environ: Mapping[str, str] | None = None,
) -> ProcedureExperienceRuntimeConfig:
    """Resolve mapping + dedicated environment overrides without touching snapshots.

    Environment activation is explicit: snapshot-related environment variables do
    nothing unless ``ROBOHARN_EVO_PROCEDURE_EXPERIENCE_MODE=candidate_dev`` is also set.
    The authoritative benchmark task family comes from the resolved evaluation
    config, never from natural-language instructions or an environment override.
    """

    values = dict(raw_config or {})
    env = os.environ if environ is None else environ
    if ENV_MODE in env:
        values["mode"] = str(env[ENV_MODE]).strip()
    mode = str(values.get("mode", OFF_MODE)).strip() or OFF_MODE
    if mode == OFF_MODE:
        resolved = ProcedureExperienceRuntimeConfig(mode=OFF_MODE)
        expected_hash = str(env.get(ENV_EXPECTED_CONFIG_SHA256, "")).strip()
        if expected_hash and expected_hash != resolved.config_sha256:
            raise ProcedureExperienceConfigurationError(
                "procedure experience config hash mismatch: "
                f"expected {expected_hash}, resolved {resolved.config_sha256}"
            )
        return resolved
    if mode == CANDIDATE_DEV_MODE:
        if str(env.get("ROBOHARN_EVO_FORMAL_PROTOCOL", "0")).strip() not in {"", "0"}:
            raise ProcedureExperienceConfigurationError(
                "candidate_dev is a development-only override and is forbidden "
                "when ROBOHARN_EVO_FORMAL_PROTOCOL is enabled"
            )
        if ENV_SNAPSHOT_MANIFEST in env:
            values["snapshot_manifest"] = str(env[ENV_SNAPSHOT_MANIFEST]).strip()
        if ENV_TOP_K in env:
            values["top_k"] = str(env[ENV_TOP_K]).strip()
        if ENV_MAX_PROMPT_CHARS in env:
            values["max_prompt_chars"] = str(env[ENV_MAX_PROMPT_CHARS]).strip()
    try:
        resolved = ProcedureExperienceRuntimeConfig(
            mode=mode,
            snapshot_manifest=str(values.get("snapshot_manifest", "")),
            task_family=str(task_family or "").strip(),
            top_k=int(values.get("top_k", DEFAULT_TOP_K)),
            max_prompt_chars=int(
                values.get("max_prompt_chars", DEFAULT_MAX_PROMPT_CHARS)
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ProcedureExperienceConfigurationError):
            raise
        raise ProcedureExperienceConfigurationError(
            f"invalid procedure experience numeric configuration: {exc}"
        ) from exc
    expected_hash = str(env.get(ENV_EXPECTED_CONFIG_SHA256, "")).strip()
    if expected_hash:
        if not _SHA256_RE.fullmatch(expected_hash):
            raise ProcedureExperienceConfigurationError(
                f"{ENV_EXPECTED_CONFIG_SHA256} must be a lowercase SHA-256 digest"
            )
        if expected_hash != resolved.config_sha256:
            raise ProcedureExperienceConfigurationError(
                "procedure experience config hash mismatch: "
                f"expected {expected_hash}, resolved {resolved.config_sha256}"
            )
    return resolved


@dataclass(frozen=True)
class ProcedureExperienceQuery:
    task_family: str
    task_instruction: str = ""
    current_subtask: str = ""
    manipulation_phase: str = ""
    relation: str = ""
    object_semantics: tuple[str, ...] = ()
    target_semantics: tuple[str, ...] = ()

    @classmethod
    def from_planner_task(
        cls,
        task: str,
        *,
        task_family: str,
    ) -> "ProcedureExperienceQuery":
        try:
            parsed = json.loads(task)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if not isinstance(parsed, dict):
            return cls(task_family=task_family, task_instruction=str(task))

        agent_state = parsed.get("agent_state")
        if not isinstance(agent_state, dict):
            agent_state = {}
        working = agent_state.get("working")
        if not isinstance(working, dict):
            working = {}
        active_skill = agent_state.get("active_skill")
        if not isinstance(active_skill, dict):
            active_skill = {}
        current_subtask = str(
            active_skill.get("instruction")
            or working.get("active_instruction")
            or ""
        ).strip()
        manipulation_state = working.get("manipulation_state")
        if not isinstance(manipulation_state, dict):
            manipulation_state = {}
        semantic_tags = working.get("semantic_tags")
        if not isinstance(semantic_tags, dict):
            semantic_tags = {}
        scene_memory = working.get("scene_memory")
        if not isinstance(scene_memory, dict):
            scene_memory = {}

        phase = str(
            manipulation_state.get("phase")
            or semantic_tags.get("manipulation_phase")
            or ""
        ).strip()
        relation = str(
            semantic_tags.get("relation")
            or scene_memory.get("relation")
            or ""
        ).strip()

        object_semantics = _bounded_semantic_strings(
            semantic_tags.get("object") or semantic_tags.get("objects")
        )
        target_semantics = _bounded_semantic_strings(
            semantic_tags.get("target") or semantic_tags.get("targets")
        )
        return cls(
            task_family=task_family,
            task_instruction=str(parsed.get("global_task", "")).strip(),
            current_subtask=current_subtask,
            manipulation_phase=phase,
            relation=relation,
            object_semantics=object_semantics,
            target_semantics=target_semantics,
        )


def _bounded_semantic_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        return ()
    result: list[str] = []
    for item in values[:16]:
        text = str(item).strip()
        if text and text not in result:
            result.append(text[:160])
    return tuple(result)


@dataclass(frozen=True)
class ProcedureExperienceSnapshot:
    manifest_path: Path
    manifest_sha256: str
    candidate_path: Path
    candidate_sha256: str
    phase_a_manifest_path: Path
    phase_a_manifest_sha256: str
    snapshot_id: str
    task_family: str
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ProcedureExperienceRetrieval:
    context: dict[str, Any] | None
    audit: dict[str, Any]


def _parse_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProcedureExperienceSnapshotError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProcedureExperienceSnapshotError(f"{label} must contain one JSON object")
    return value


def _load_retirement_registry() -> tuple[dict[str, str], ...]:
    """Load the repository-owned, content-bound retirement registry."""

    raw = _read_regular_file(
        _RETIREMENT_REGISTRY_PATH,
        label="snapshot retirement registry",
        max_bytes=MAX_SNAPSHOT_BYTES,
    )
    registry = _parse_json_object(raw, label="snapshot retirement registry")
    if set(registry) != {"schema", "schema_version", "entries"}:
        raise ProcedureExperienceSnapshotError(
            "snapshot retirement registry fields must match the v1 schema exactly"
        )
    if (
        registry.get("schema") not in {RETIREMENT_REGISTRY_SCHEMA, "tcm/procedure_experience_retirement_registry/v1"}
        or registry.get("schema_version") != 1
    ):
        raise ProcedureExperienceSnapshotError(
            "snapshot retirement registry has an unsupported schema or version"
        )
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ProcedureExperienceSnapshotError(
            "snapshot retirement registry entries must be an array"
        )
    if len(entries) > MAX_SNAPSHOT_EXPERIENCES:
        raise ProcedureExperienceSnapshotError(
            "snapshot retirement registry exceeds its entry budget"
        )

    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    required = {
        "snapshot_id",
        "manifest_sha256",
        "status",
        "reason",
        "retired_at",
    }
    for index, value in enumerate(entries):
        label = f"snapshot retirement registry entries[{index}]"
        if not isinstance(value, dict) or set(value) != required:
            raise ProcedureExperienceSnapshotError(
                f"{label} fields must match the v1 schema exactly"
            )
        snapshot_id = value.get("snapshot_id")
        if (
            not isinstance(snapshot_id, str)
            or _SAFE_SNAPSHOT_ID_RE.fullmatch(snapshot_id) is None
        ):
            raise ProcedureExperienceSnapshotError(
                f"{label}.snapshot_id is not a safe snapshot identity"
            )
        manifest_sha256 = _require_sha256(
            value.get("manifest_sha256"),
            label=f"{label}.manifest_sha256",
        )
        if value.get("status") != "retired":
            raise ProcedureExperienceSnapshotError(
                f"{label}.status must be 'retired'"
            )
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
            raise ProcedureExperienceSnapshotError(
                f"{label}.reason must be a non-empty bounded string"
            )
        retired_at = value.get("retired_at")
        if (
            not isinstance(retired_at, str)
            or _UTC_SECOND_RE.fullmatch(retired_at) is None
        ):
            raise ProcedureExperienceSnapshotError(
                f"{label}.retired_at must be UTC at second precision"
            )
        if snapshot_id in seen_ids or manifest_sha256 in seen_hashes:
            raise ProcedureExperienceSnapshotError(
                "snapshot retirement registry contains a duplicate identity or hash"
            )
        seen_ids.add(snapshot_id)
        seen_hashes.add(manifest_sha256)
        normalized.append(
            {
                "snapshot_id": snapshot_id,
                "manifest_sha256": manifest_sha256,
                "status": "retired",
                "reason": reason.strip(),
                "retired_at": retired_at,
            }
        )
    return tuple(normalized)


def assert_snapshot_not_retired(
    *,
    snapshot_id: str,
    manifest_sha256: str,
) -> None:
    """Reject an exact retirement or either half of a conflicting binding."""

    if _SAFE_SNAPSHOT_ID_RE.fullmatch(snapshot_id) is None:
        raise ProcedureExperienceSnapshotError("snapshot_id is not a safe identity")
    digest = _require_sha256(manifest_sha256, label="snapshot manifest SHA-256")
    entries = _load_retirement_registry()
    id_match = next(
        (value for value in entries if value["snapshot_id"] == snapshot_id),
        None,
    )
    hash_match = next(
        (value for value in entries if value["manifest_sha256"] == digest),
        None,
    )
    if id_match is None and hash_match is None:
        return
    if id_match is hash_match and id_match is not None:
        raise ProcedureExperienceSnapshotError(
            "candidate_dev snapshot is retired: "
            f"{snapshot_id} ({id_match['reason']})"
        )
    raise ProcedureExperienceSnapshotError(
        "snapshot retirement identity/hash binding conflict; refusing candidate_dev"
    )


def _load_snapshot(manifest_value: str) -> ProcedureExperienceSnapshot:
    manifest_path = Path(manifest_value).expanduser()
    try:
        manifest_metadata = manifest_path.lstat()
    except OSError as exc:
        raise ProcedureExperienceSnapshotError(
            f"snapshot manifest does not exist: {manifest_value}: {exc}"
        ) from exc
    if stat.S_ISLNK(manifest_metadata.st_mode):
        raise ProcedureExperienceSnapshotError(
            f"snapshot manifest must not be a symlink: {manifest_value}"
        )
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise ProcedureExperienceSnapshotError(
            f"snapshot manifest does not exist: {manifest_value}: {exc}"
        ) from exc
    manifest_bytes = _read_regular_file(
        manifest_path,
        label="snapshot manifest",
        max_bytes=MAX_SNAPSHOT_BYTES,
    )
    manifest_digest = _sha256_bytes(manifest_bytes)
    manifest = _parse_json_object(manifest_bytes, label="snapshot manifest")
    if manifest.get("schema") not in {
        SNAPSHOT_SCHEMA, "tcm/procedure_experience_snapshot/v1"
    } or manifest.get("schema_version") != 1:
        raise ProcedureExperienceSnapshotError(
            f"snapshot schema must be {SNAPSHOT_SCHEMA!r} version 1"
        )
    if manifest.get("created_for") != CANDIDATE_DEV_MODE:
        raise ProcedureExperienceSnapshotError(
            "snapshot created_for must be 'candidate_dev'"
        )
    if manifest.get("source") != "benchmark_expert":
        raise ProcedureExperienceSnapshotError(
            "candidate_dev snapshot source must be 'benchmark_expert'"
        )
    if manifest.get("status") != "candidate":
        raise ProcedureExperienceSnapshotError(
            "candidate_dev snapshot status must remain 'candidate'"
        )
    task_family = _require_nonempty_string(
        manifest.get("task_family"), label="snapshot task_family"
    )
    bootstrap_seeds = manifest.get("bootstrap_seeds")
    if bootstrap_seeds != [1]:
        raise ProcedureExperienceSnapshotError(
            "this candidate_dev snapshot must bind bootstrap_seeds=[1]"
        )
    expected_snapshot_id = snapshot_id_for_manifest(manifest)
    if manifest.get("snapshot_id") != expected_snapshot_id:
        raise ProcedureExperienceSnapshotError(
            "snapshot_id does not match the content-derived identity: "
            f"expected {expected_snapshot_id}, got {manifest.get('snapshot_id')!r}"
        )
    assert_snapshot_not_retired(
        snapshot_id=expected_snapshot_id,
        manifest_sha256=manifest_digest,
    )

    candidate_spec = manifest.get("candidate_experiences")
    phase_a_spec = manifest.get("phase_a_run_manifest")
    if not isinstance(candidate_spec, dict) or not isinstance(phase_a_spec, dict):
        raise ProcedureExperienceSnapshotError(
            "snapshot file specifications must be JSON objects"
        )
    candidate_path = _resolve_snapshot_member(
        manifest_path,
        candidate_spec.get("path"),
        label="candidate_experiences.path",
    )
    phase_a_path = _resolve_snapshot_member(
        manifest_path,
        phase_a_spec.get("path"),
        label="phase_a_run_manifest.path",
    )
    candidate_sha256 = _require_sha256(
        candidate_spec.get("sha256"), label="candidate_experiences.sha256"
    )
    phase_a_sha256 = _require_sha256(
        phase_a_spec.get("sha256"), label="phase_a_run_manifest.sha256"
    )
    candidate_bytes = _read_regular_file(
        candidate_path,
        label="candidate experience file",
        max_bytes=MAX_SNAPSHOT_BYTES,
    )
    phase_a_bytes = _read_regular_file(
        phase_a_path,
        label="Phase A run manifest",
        max_bytes=MAX_SNAPSHOT_BYTES,
    )
    if _sha256_bytes(candidate_bytes) != candidate_sha256:
        raise ProcedureExperienceSnapshotError(
            "candidate experience file SHA-256 does not match snapshot manifest"
        )
    if _sha256_bytes(phase_a_bytes) != phase_a_sha256:
        raise ProcedureExperienceSnapshotError(
            "Phase A run manifest SHA-256 does not match snapshot manifest"
        )

    phase_a_manifest = _parse_json_object(phase_a_bytes, label="Phase A run manifest")
    if (
        phase_a_manifest.get("schema") not in {"roboharn_evo/expert_bootstrap_run/v1", "tcm/expert_bootstrap_run/v1"}
        or phase_a_manifest.get("schema_version") != 1
        or phase_a_manifest.get("mode") != "offline_candidate_only"
    ):
        raise ProcedureExperienceSnapshotError(
            "Phase A run manifest is not a v1 offline_candidate_only bootstrap run"
        )
    phase_a_config = phase_a_manifest.get("configuration")
    if not isinstance(phase_a_config, dict) or (
        phase_a_config.get("runtime_retrieval") is not False
        or phase_a_config.get("rollout_integration") is not False
        or phase_a_config.get("candidate_only") is not True
    ):
        raise ProcedureExperienceSnapshotError(
            "Phase A run manifest must prove candidate-only, disconnected runtime output"
        )
    if phase_a_manifest.get("run_id") != manifest.get("phase_a_run_id"):
        raise ProcedureExperienceSnapshotError(
            "snapshot phase_a_run_id does not match the bound run manifest"
        )
    phase_a_integrity = phase_a_manifest.get("file_integrity")
    phase_a_files = (
        phase_a_integrity.get("files")
        if isinstance(phase_a_integrity, dict)
        else None
    )
    phase_a_candidate = (
        phase_a_files.get("candidate_experiences.jsonl")
        if isinstance(phase_a_files, dict)
        else None
    )
    if not isinstance(phase_a_candidate, dict) or (
        phase_a_candidate.get("sha256") != candidate_sha256
        or phase_a_candidate.get("record_count") != manifest.get("experience_count")
    ):
        raise ProcedureExperienceSnapshotError(
            "Phase A run manifest does not bind the snapshot candidate file"
        )

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    observed_seeds: set[int] = set()
    for line_number, raw_line in enumerate(candidate_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        if len(records) >= MAX_SNAPSHOT_EXPERIENCES:
            raise ProcedureExperienceSnapshotError(
                f"candidate snapshot exceeds {MAX_SNAPSHOT_EXPERIENCES} records"
            )
        record = _parse_json_object(
            raw_line, label=f"candidate experience line {line_number}"
        )
        try:
            parsed_experience = ProcedureExperienceV1.from_dict(record)
            if parsed_experience.retrieval_eligible:
                raise ProcedureExperienceSnapshotError(
                    "candidate_dev source unexpectedly became formally retrieval eligible"
                )
            validated = parsed_experience.to_dict()
        except SchemaValidationError as exc:
            raise ProcedureExperienceSnapshotError(
                f"invalid ProcedureExperienceV1 at line {line_number}: {exc}"
            ) from exc
        experience_id = str(validated.get("experience_id", ""))
        if experience_id in seen_ids:
            raise ProcedureExperienceSnapshotError(
                f"duplicate experience_id in snapshot: {experience_id}"
            )
        seen_ids.add(experience_id)
        if validated.get("status") != "candidate":
            raise ProcedureExperienceSnapshotError(
                f"candidate_dev experience {experience_id} changed status"
            )
        condition = validated.get("condition")
        if not isinstance(condition, dict) or condition.get("task_family") != task_family:
            raise ProcedureExperienceSnapshotError(
                f"experience {experience_id} task_family does not match {task_family!r}"
            )
        provenance = validated.get("provenance")
        if not isinstance(provenance, dict) or (
            provenance.get("source_kind") != "benchmark_expert"
            or provenance.get("expert_derived") is not True
            or provenance.get("oracle_derived") is not True
        ):
            raise ProcedureExperienceSnapshotError(
                f"experience {experience_id} provenance is not the bound expert/oracle-derived benchmark data"
            )
        source_refs = provenance.get("source_refs")
        if not isinstance(source_refs, list) or not source_refs:
            raise ProcedureExperienceSnapshotError(
                f"experience {experience_id} has no source provenance refs"
            )
        for source_ref in source_refs:
            if not isinstance(source_ref, dict) or source_ref.get("task") != task_family:
                raise ProcedureExperienceSnapshotError(
                    f"experience {experience_id} source task does not match {task_family!r}"
                )
            seed = source_ref.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ProcedureExperienceSnapshotError(
                    f"experience {experience_id} source seed must be an integer"
                )
            observed_seeds.add(seed)
        records.append(validated)

    manifest_ids = manifest.get("experience_ids")
    if not isinstance(manifest_ids, list) or any(
        not isinstance(value, str) or not value for value in manifest_ids
    ):
        raise ProcedureExperienceSnapshotError(
            "snapshot experience_ids must be a non-empty string array"
        )
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ProcedureExperienceSnapshotError("snapshot manifest has duplicate experience_ids")
    if manifest_ids != [record["experience_id"] for record in records]:
        raise ProcedureExperienceSnapshotError(
            "snapshot experience_ids do not match candidate file order"
        )
    if manifest.get("experience_count") != len(records):
        raise ProcedureExperienceSnapshotError(
            "snapshot experience_count does not match candidate file"
        )
    if candidate_spec.get("record_count") != len(records):
        raise ProcedureExperienceSnapshotError(
            "candidate_experiences.record_count does not match candidate file"
        )
    phase_a_counts = phase_a_manifest.get("counts")
    if not isinstance(phase_a_counts, dict) or phase_a_counts.get(
        "candidate_experiences"
    ) != len(records):
        raise ProcedureExperienceSnapshotError(
            "Phase A run manifest candidate count does not match snapshot"
        )
    if observed_seeds != {1}:
        raise ProcedureExperienceSnapshotError(
            f"snapshot source seeds must be exactly {{1}}, got {sorted(observed_seeds)}"
        )

    return ProcedureExperienceSnapshot(
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        phase_a_manifest_path=phase_a_path,
        phase_a_manifest_sha256=phase_a_sha256,
        snapshot_id=expected_snapshot_id,
        task_family=task_family,
        records=tuple(copy.deepcopy(records)),
    )


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(value)) if token.strip()}


def _query_tokens(query: ProcedureExperienceQuery) -> dict[str, set[str]]:
    return {
        "task_instruction": _tokens(query.task_instruction),
        "current_subtask": _tokens(query.current_subtask),
        "manipulation_phase": _tokens(query.manipulation_phase),
        "relation": _tokens(query.relation),
        "object_semantics": _tokens(" ".join(query.object_semantics)),
        "target_semantics": _tokens(" ".join(query.target_semantics)),
    }


def _record_search_text(record: Mapping[str, Any]) -> str:
    transferable = {
        "condition": record.get("condition", {}),
        "guidance": record.get("guidance", {}),
        "predicted_effects": record.get("predicted_effects", []),
    }
    return json.dumps(transferable, ensure_ascii=False, sort_keys=True)


def _score_record(
    record: Mapping[str, Any], query: ProcedureExperienceQuery
) -> tuple[int, tuple[str, ...]]:
    reasons = ["task_family_exact"]
    score = 100
    condition = record.get("condition")
    if not isinstance(condition, Mapping):
        condition = {}
    subtask_type = str(condition.get("subtask_type", "")).strip().casefold()
    query_subtask_tokens = _tokens(query.current_subtask)
    if subtask_type and subtask_type in query_subtask_tokens:
        score += 40
        reasons.append("subtask_type_exact")
    for field_name, query_value, candidate_value, weight in (
        (
            "manipulation_phase",
            query.manipulation_phase,
            condition.get("phase", condition.get("manipulation_phase", "")),
            35,
        ),
        ("relation", query.relation, condition.get("relation", ""), 35),
    ):
        if query_value and str(query_value).casefold() == str(candidate_value).casefold():
            score += weight
            reasons.append(f"{field_name}_exact")
    record_tokens = _tokens(_record_search_text(record))
    for field_name, field_tokens, weight in (
        ("current_subtask", query_subtask_tokens, 8),
        ("task_instruction", _tokens(query.task_instruction), 4),
        ("object_semantics", _tokens(" ".join(query.object_semantics)), 6),
        ("target_semantics", _tokens(" ".join(query.target_semantics)), 6),
    ):
        overlap = sorted(field_tokens & record_tokens)
        if overlap:
            score += min(5, len(overlap)) * weight
            reasons.append(f"{field_name}_token_overlap:{','.join(overlap[:5])}")
    return score, tuple(reasons)


def _validate_projection_string(value: str, *, path: str) -> None:
    if _LOCAL_PATH_RE.search(value):
        raise ProcedureExperienceContextError(
            f"planner projection contains a local file path at {path}"
        )
    if (
        not _PUBLIC_EXPERIENCE_ID_PATH_RE.fullmatch(path)
        and _RAW_SHA256_TOKEN_RE.search(value)
    ):
        raise ProcedureExperienceContextError(
            f"planner projection contains a raw SHA-256 digest at {path}"
        )


def _validate_safe_projection_json(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ProcedureExperienceContextError(f"non-finite number at {path}")
        return
    if isinstance(value, str):
        _validate_projection_string(value, path=path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe_projection_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProcedureExperienceContextError(f"non-string key at {path}")
            if _is_forbidden_projection_key(key):
                raise ProcedureExperienceContextError(
                    f"forbidden planner projection field at {path}.{key}"
                )
            _validate_safe_projection_json(item, path=f"{path}.{key}")
        return
    raise ProcedureExperienceContextError(
        f"unsupported planner projection value at {path}: {type(value).__name__}"
    )


def _is_forbidden_projection_key(key: str) -> bool:
    normalized = key.casefold()
    if normalized in _FORBIDDEN_PROJECTION_KEYS:
        return True
    if normalized == "world_m" or normalized.endswith("_world_m"):
        return True
    tokens = {
        token for token in re.split(r"[^a-z0-9]+", normalized) if token
    }
    if tokens & _PRIVATE_POSE_TOKENS:
        return True
    if normalized == "id" or (
        normalized.endswith(("_id", "_uid", "_uuid"))
        and bool(tokens & _PRIVATE_ID_TOKENS)
    ):
        return True
    if "world" in tokens and tokens & {
        "camera",
        "ee",
        "matrix",
        "position",
        "rotation",
        "transform",
        "translation",
    }:
        return True
    if normalized.endswith(
        ("_rgb_path", "_depth_path", "_mask_path", "_video_path")
    ):
        return True
    return False


def _planner_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    condition = record.get("condition")
    if not isinstance(condition, Mapping):
        condition = {}
    projected_condition: dict[str, str] = {}
    for key in ("task_family", "subtask_type", "relation"):
        value = condition.get(key)
        if isinstance(value, str) and value.strip():
            projected_condition[key] = value.strip()
    phase = condition.get("phase", condition.get("manipulation_phase"))
    if isinstance(phase, str) and phase.strip():
        projected_condition["phase"] = phase.strip()

    guidance = record.get("guidance")
    if not isinstance(guidance, Mapping):
        guidance = {}
    ordered_steps: list[dict[str, str]] = []
    for raw_step in guidance.get("ordered_steps", []) or []:
        if not isinstance(raw_step, Mapping):
            continue
        step: dict[str, str] = {}
        for key in ("action_pattern", "instruction", "manipulation_phase"):
            value = raw_step.get(key)
            if isinstance(value, str) and value.strip():
                step[key] = value.strip()
        if step:
            ordered_steps.append(step)
    avoid = [
        item.strip()
        for item in (guidance.get("avoid", []) or [])
        if isinstance(item, str) and item.strip()
    ]
    projection = {
        "experience_id": str(record.get("experience_id", "")),
        "kind": str(record.get("kind", "")),
        "status": str(record.get("status", "")),
        "condition": projected_condition,
        "guidance": {"ordered_steps": ordered_steps, "avoid": avoid},
        "predicted_effects": copy.deepcopy(record.get("predicted_effects", [])),
        "confidence": str(record.get("confidence", "")),
        "candidate_dev": True,
        "expert_derived": True,
    }
    _validate_safe_projection_json(projection, path="experience")
    return projection


def render_procedure_experience_block(experiences: Sequence[Mapping[str, Any]]) -> str:
    """Render the sole bounded model-visible experience block."""

    if not experiences:
        return ""
    safe_experiences = [copy.deepcopy(dict(item)) for item in experiences]
    for index, item in enumerate(safe_experiences):
        _validate_safe_projection_json(item, path=f"experiences[{index}]")
    return (
        "<procedure_experience_context>\n"
        + ADVISORY_NOTICE
        + "\n"
        + json.dumps(
            {"experiences": safe_experiences},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n</procedure_experience_context>"
    )


def _context_for_selection(
    *,
    snapshot: ProcedureExperienceSnapshot,
    task_family: str,
    projections: list[dict[str, Any]],
    selection_reasons: list[dict[str, Any]],
    max_prompt_chars: int,
) -> dict[str, Any]:
    rendered = render_procedure_experience_block(projections)
    return {
        "schema": CONTEXT_SCHEMA,
        "schema_version": 1,
        "experience_mode": CANDIDATE_DEV_MODE,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_manifest_sha256": snapshot.manifest_sha256,
        "query_task_family": task_family,
        "selected_experience_ids": [
            item["experience_id"] for item in projections
        ],
        "selection_reasons": copy.deepcopy(selection_reasons),
        "projection_sha256": _sha256_bytes(rendered.encode("utf-8")),
        "projection_char_count": len(rendered),
        "estimated_token_count": (len(rendered) + 3) // 4,
        "max_prompt_chars": max_prompt_chars,
        "candidate_dev_override": True,
        "expert_prior_used": True,
        "cross_rollout_memory_enabled": False,
        "experiences": copy.deepcopy(projections),
    }


def validate_and_render_procedure_experience_context(
    context: Any,
) -> tuple[str, dict[str, Any]]:
    """Validate a client context and return model text plus compact audit data."""

    if not isinstance(context, dict):
        raise ProcedureExperienceContextError(
            "procedure_experience_context must be a JSON object"
        )
    unknown = sorted(set(context) - _CONTEXT_KEYS)
    missing = sorted(_CONTEXT_KEYS - set(context))
    if unknown or missing:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ProcedureExperienceContextError(
            "invalid procedure_experience_context fields: " + "; ".join(details)
        )
    if context.get("schema") not in {
        CONTEXT_SCHEMA, "tcm/procedure_experience_context/v1"
    } or context.get("schema_version") != 1:
        raise ProcedureExperienceContextError(
            f"procedure experience context must be {CONTEXT_SCHEMA!r} version 1"
        )
    if context.get("experience_mode") != CANDIDATE_DEV_MODE:
        raise ProcedureExperienceContextError(
            "only explicit candidate_dev planner context is supported"
        )
    if (
        context.get("candidate_dev_override") is not True
        or context.get("expert_prior_used") is not True
        or context.get("cross_rollout_memory_enabled") is not False
    ):
        raise ProcedureExperienceContextError(
            "candidate_dev information-access flags are inconsistent"
        )
    snapshot_id = str(context.get("snapshot_id", ""))
    query_task_family = str(context.get("query_task_family", ""))
    if not snapshot_id or not query_task_family:
        raise ProcedureExperienceContextError(
            "context snapshot_id and query_task_family are required"
        )
    snapshot_hash = str(context.get("snapshot_manifest_sha256", ""))
    projection_hash = str(context.get("projection_sha256", ""))
    if not _SHA256_RE.fullmatch(snapshot_hash) or not _SHA256_RE.fullmatch(
        projection_hash
    ):
        raise ProcedureExperienceContextError(
            "context snapshot/projection hashes must be lowercase SHA-256 digests"
        )
    experiences = context.get("experiences")
    if not isinstance(experiences, list) or not experiences:
        raise ProcedureExperienceContextError(
            "candidate_dev context must contain at least one selected experience"
        )
    selected_ids = context.get("selected_experience_ids")
    if not isinstance(selected_ids, list) or selected_ids != [
        item.get("experience_id") if isinstance(item, dict) else None
        for item in experiences
    ]:
        raise ProcedureExperienceContextError(
            "selected_experience_ids do not match planner projections"
        )
    if len(set(selected_ids)) != len(selected_ids):
        raise ProcedureExperienceContextError(
            "selected_experience_ids must not contain duplicates"
        )
    for index, experience in enumerate(experiences):
        if not isinstance(experience, dict):
            raise ProcedureExperienceContextError(
                f"experiences[{index}] must be a JSON object"
            )
        required_keys = {
            "experience_id",
            "kind",
            "status",
            "condition",
            "guidance",
            "predicted_effects",
            "confidence",
            "candidate_dev",
            "expert_derived",
        }
        if set(experience) != required_keys:
            raise ProcedureExperienceContextError(
                f"experiences[{index}] projection fields are not the exact whitelist"
            )
        if (
            experience.get("status") != "candidate"
            or experience.get("candidate_dev") is not True
            or experience.get("expert_derived") is not True
        ):
            raise ProcedureExperienceContextError(
                f"experiences[{index}] is not an explicit expert candidate projection"
            )
        condition = experience.get("condition")
        if not isinstance(condition, dict) or (
            set(condition) - {"task_family", "subtask_type", "relation", "phase"}
        ):
            raise ProcedureExperienceContextError(
                f"experiences[{index}].condition contains non-whitelisted fields"
            )
        if condition.get("task_family") != query_task_family:
            raise ProcedureExperienceContextError(
                f"experiences[{index}] task_family does not match the query"
            )
        guidance = experience.get("guidance")
        if not isinstance(guidance, dict) or set(guidance) != {
            "ordered_steps",
            "avoid",
        }:
            raise ProcedureExperienceContextError(
                f"experiences[{index}].guidance fields are not the exact whitelist"
            )
        steps = guidance.get("ordered_steps")
        if not isinstance(steps, list) or not steps:
            raise ProcedureExperienceContextError(
                f"experiences[{index}].guidance.ordered_steps must be non-empty"
            )
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict) or set(step) - _STEP_KEYS:
                raise ProcedureExperienceContextError(
                    f"experiences[{index}].guidance.ordered_steps[{step_index}] contains non-whitelisted fields"
                )
        _validate_safe_projection_json(
            experience, path=f"experiences[{index}]"
        )
    reasons = context.get("selection_reasons")
    if not isinstance(reasons, list) or len(reasons) != len(selected_ids):
        raise ProcedureExperienceContextError(
            "selection_reasons must align one-to-one with selected experiences"
        )
    for index, reason in enumerate(reasons):
        if (
            not isinstance(reason, dict)
            or set(reason) != {"experience_id", "score", "reasons"}
            or reason.get("experience_id") != selected_ids[index]
            or isinstance(reason.get("score"), bool)
            or not isinstance(reason.get("score"), int)
            or not isinstance(reason.get("reasons"), list)
        ):
            raise ProcedureExperienceContextError(
                f"selection_reasons[{index}] is invalid"
            )
    rendered = render_procedure_experience_block(experiences)
    if context.get("projection_char_count") != len(rendered):
        raise ProcedureExperienceContextError("projection_char_count mismatch")
    if context.get("estimated_token_count") != (len(rendered) + 3) // 4:
        raise ProcedureExperienceContextError("estimated_token_count mismatch")
    if projection_hash != _sha256_bytes(rendered.encode("utf-8")):
        raise ProcedureExperienceContextError("projection_sha256 mismatch")
    max_prompt_chars = context.get("max_prompt_chars")
    if (
        isinstance(max_prompt_chars, bool)
        or not isinstance(max_prompt_chars, int)
        or max_prompt_chars < 1
        or len(rendered) > max_prompt_chars
    ):
        raise ProcedureExperienceContextError(
            "rendered procedure experience exceeds max_prompt_chars"
        )
    audit = {
        "experience_mode": CANDIDATE_DEV_MODE,
        "snapshot_id": snapshot_id,
        "snapshot_manifest_sha256": snapshot_hash,
        "query_task_family": query_task_family,
        "selected_experience_ids": list(selected_ids),
        "selection_reasons": copy.deepcopy(reasons),
        "projection_sha256": projection_hash,
        "projection_char_count": len(rendered),
        "estimated_token_count": (len(rendered) + 3) // 4,
        "candidate_dev_override": True,
        "expert_prior_used": True,
        "cross_rollout_memory_enabled": False,
        "injection_count": 1,
    }
    return rendered, validate_procedure_experience_audit(audit)


def validate_procedure_experience_audit(value: Any) -> dict[str, Any]:
    """Validate the compact server-to-runtime audit sidecar."""

    if not isinstance(value, dict) or set(value) != _AUDIT_KEYS:
        raise ProcedureExperienceContextError(
            "procedure experience audit fields are not the exact compact whitelist"
        )
    if (
        value.get("experience_mode") != CANDIDATE_DEV_MODE
        or value.get("candidate_dev_override") is not True
        or value.get("expert_prior_used") is not True
        or value.get("cross_rollout_memory_enabled") is not False
        or value.get("injection_count") != 1
    ):
        raise ProcedureExperienceContextError(
            "procedure experience audit flags are inconsistent"
        )
    for key in ("snapshot_id", "query_task_family"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ProcedureExperienceContextError(f"audit {key} must be non-empty")
    for key in ("snapshot_manifest_sha256", "projection_sha256"):
        if not isinstance(value.get(key), str) or not _SHA256_RE.fullmatch(value[key]):
            raise ProcedureExperienceContextError(f"audit {key} must be SHA-256")
    selected = value.get("selected_experience_ids")
    reasons = value.get("selection_reasons")
    if (
        not isinstance(selected, list)
        or not selected
        or len(set(selected)) != len(selected)
        or any(not isinstance(item, str) or not item for item in selected)
        or not isinstance(reasons, list)
        or len(reasons) != len(selected)
    ):
        raise ProcedureExperienceContextError(
            "audit selected IDs and reasons must be non-empty and aligned"
        )
    for key in ("projection_char_count", "estimated_token_count"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ProcedureExperienceContextError(f"audit {key} must be a positive integer")
    _validate_safe_projection_json(reasons, path="audit.selection_reasons")
    return copy.deepcopy(value)


class ProcedureExperienceRetriever:
    """Deterministic, read-only procedure retriever for planner requests."""

    def __init__(self, config: ProcedureExperienceRuntimeConfig) -> None:
        self.config = config
        self._snapshot: ProcedureExperienceSnapshot | None = None
        if config.mode == CANDIDATE_DEV_MODE:
            snapshot = _load_snapshot(config.snapshot_manifest)
            if snapshot.task_family != config.task_family:
                raise ProcedureExperienceSnapshotError(
                    "configured task_family does not match snapshot: "
                    f"{config.task_family!r} vs {snapshot.task_family!r}"
                )
            self._snapshot = snapshot

    @property
    def snapshot(self) -> ProcedureExperienceSnapshot | None:
        return self._snapshot

    def _assert_snapshot_unchanged(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        for path, expected, label in (
            (
                snapshot.manifest_path,
                snapshot.manifest_sha256,
                "snapshot manifest",
            ),
            (snapshot.candidate_path, snapshot.candidate_sha256, "candidate file"),
            (
                snapshot.phase_a_manifest_path,
                snapshot.phase_a_manifest_sha256,
                "Phase A run manifest",
            ),
        ):
            current = _sha256_bytes(
                _read_regular_file(path, label=label, max_bytes=MAX_SNAPSHOT_BYTES)
            )
            if current != expected:
                raise ProcedureExperienceSnapshotError(
                    f"{label} changed after snapshot load: {path}"
                )
        assert_snapshot_not_retired(
            snapshot_id=snapshot.snapshot_id,
            manifest_sha256=snapshot.manifest_sha256,
        )

    def retrieve(
        self, query: ProcedureExperienceQuery
    ) -> ProcedureExperienceRetrieval:
        if self.config.mode == OFF_MODE:
            return ProcedureExperienceRetrieval(
                context=None,
                audit={"experience_mode": OFF_MODE, "selected_experience_ids": []},
            )
        snapshot = self._snapshot
        if snapshot is None:
            raise ProcedureExperienceConfigurationError(
                "candidate_dev retriever has no loaded snapshot"
            )
        self._assert_snapshot_unchanged()
        if query.task_family != snapshot.task_family:
            return ProcedureExperienceRetrieval(
                context=None,
                audit={
                    "experience_mode": CANDIDATE_DEV_MODE,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_manifest_sha256": snapshot.manifest_sha256,
                    "query_task_family": query.task_family,
                    "selected_experience_ids": [],
                    "selection_reasons": [],
                    "candidate_dev_override": True,
                    "expert_prior_used": True,
                    "cross_rollout_memory_enabled": False,
                    "injection_count": 0,
                },
            )

        scored: list[tuple[int, str, tuple[str, ...], dict[str, Any]]] = []
        for record in snapshot.records:
            score, reasons = _score_record(record, query)
            scored.append(
                (score, str(record["experience_id"]), reasons, copy.deepcopy(record))
            )
        scored.sort(key=lambda item: (-item[0], item[1]))

        projections: list[dict[str, Any]] = []
        selection_reasons: list[dict[str, Any]] = []
        for score, experience_id, reasons, record in scored:
            if len(projections) >= self.config.top_k:
                break
            projection = _planner_projection(record)
            tentative = [*projections, projection]
            rendered = render_procedure_experience_block(tentative)
            if len(rendered) > self.config.max_prompt_chars:
                continue
            projections.append(projection)
            selection_reasons.append(
                {
                    "experience_id": experience_id,
                    "score": score,
                    "reasons": list(reasons),
                }
            )

        if not projections:
            return ProcedureExperienceRetrieval(
                context=None,
                audit={
                    "experience_mode": CANDIDATE_DEV_MODE,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_manifest_sha256": snapshot.manifest_sha256,
                    "query_task_family": query.task_family,
                    "selected_experience_ids": [],
                    "selection_reasons": [],
                    "candidate_dev_override": True,
                    "expert_prior_used": True,
                    "cross_rollout_memory_enabled": False,
                    "injection_count": 0,
                },
            )
        context = _context_for_selection(
            snapshot=snapshot,
            task_family=query.task_family,
            projections=projections,
            selection_reasons=selection_reasons,
            max_prompt_chars=self.config.max_prompt_chars,
        )
        _, audit = validate_and_render_procedure_experience_context(context)
        return ProcedureExperienceRetrieval(context=context, audit=audit)


__all__ = [
    "ADVISORY_NOTICE",
    "CANDIDATE_DEV_MODE",
    "CONTEXT_SCHEMA",
    "DEFAULT_MAX_PROMPT_CHARS",
    "DEFAULT_TOP_K",
    "ENV_EXPECTED_CONFIG_SHA256",
    "ENV_MAX_PROMPT_CHARS",
    "ENV_MODE",
    "ENV_SNAPSHOT_MANIFEST",
    "ENV_TOP_K",
    "OFF_MODE",
    "PROCEDURE_EXPERIENCE_MODES",
    "RETIREMENT_REGISTRY_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "ProcedureExperienceConfigurationError",
    "ProcedureExperienceContextError",
    "ProcedureExperienceError",
    "ProcedureExperienceQuery",
    "ProcedureExperienceRetrieval",
    "ProcedureExperienceRetriever",
    "ProcedureExperienceRuntimeConfig",
    "ProcedureExperienceSnapshotError",
    "assert_snapshot_not_retired",
    "render_procedure_experience_block",
    "resolve_procedure_experience_config",
    "snapshot_id_for_manifest",
    "validate_and_render_procedure_experience_context",
    "validate_procedure_experience_audit",
]

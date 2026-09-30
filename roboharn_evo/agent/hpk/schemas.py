from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from .compatibility import normalize_knowledge_metadata, normalize_observe_audit


CONDITION_SCHEMA = "roboharn_evo/hpk/condition/v1"
TASK_STRATEGY_SCHEMA = "roboharn_evo/hpk/task_strategy/v1"
GEOMETRIC_STRATEGY_SCHEMA = "roboharn_evo/hpk/geometric_strategy/v1"
ABSTRACT_EFFECT_SCHEMA = "roboharn_evo/hpk/abstract_effect/v1"
EVIDENCE_SCHEMA = "roboharn_evo/hpk/evidence/v1"
ENTRY_SCHEMA = "roboharn_evo/hpk/entry/v1"
SNAPSHOT_SCHEMA = "roboharn_evo/hpk/snapshot/v1"
CONTEXT_SCHEMA = "roboharn_evo/hpk/context/v1"
AUDIT_SCHEMA = "roboharn_evo/hpk/audit/v1"
OBSERVE_AUDIT_PROFILE = "hpk_observe_audit/p0a"
HPK_GEOMETRY_SCORE_PROFILE = "hpk_geometry_score/v1"

TASK_STRATEGY_NORMALIZATION_VERSION = "task_strategy_normalizer/v1"
CONDITION_ABSTRACTION_VERSION = "hpk_condition_builder/v1"


class HPKValidationError(ValueError):
    """HPK 数据违反 v1 格式约定。"""


@dataclass(frozen=True, slots=True)
class HPKUnresolved:
    """Typed fail-closed result returned when an abstraction cannot be built.

    ``HPKUnresolved`` is a runtime control result, not a persistent HPK record.
    It deliberately carries no raw planner payload, candidate, coordinate, or
    object identity.
    """

    component: str
    reason: str
    missing_fields: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "unresolved",
            "component": self.component,
            "reason": self.reason,
            "missing_fields": list(self.missing_fields),
        }


_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

OPERATIONS = frozenset({"contact", "grasp", "place"})
ARMS = frozenset({"left", "right", "either"})
HELD_STATES = frozenset({"held", "not_held", "unknown"})
TARGET_RELATIONS = frozenset({"center_of"})
SCENE_PREDICATES = frozenset({"target_region_free", "support_valid"})
STRATEGY_FAMILIES = frozenset(
    {"placement_relation", "observed_grasp_geometry", "contact_relation"}
)
REFERENCE_FRAMES = frozenset(
    {
        "support_normal",
        "object_principal_axes",
        "current_attachment",
        "world_gravity",
        "unknown",
    }
)
APPROACH_FAMILIES = frozenset(
    {
        "clearance_first",
        "surface_normal",
        "principal_axis_relative",
        "unconstrained",
    }
)
DIRECTION_BUCKETS = frozenset({"above", "below", "lateral", "oblique", "unknown"})
ORIENTATION_RELATIONS = frozenset(
    {
        "preserve_current_attachment",
        "align_principal_axis_0",
        "align_principal_axis_1",
        "unconstrained",
        "unknown",
    }
)
GRASP_REGIONS = frozenset(
    {"observed_surface", "object_body", "semantic_part", "unknown"}
)
HARD_CONSTRAINTS = frozenset({"support_valid", "target_region_free"})
SOFT_PREFERENCES = frozenset({"lower_reach_distance"})
AVOID_CONSTRAINTS = frozenset({"repeat_equivalent_failed_candidate"})
GEOMETRY_SOURCE_CLASSES = frozenset(
    {"rgbd_observed", "runtime_relational", "oracle", "unknown"}
)
EFFECT_TYPES = frozenset({"grasp", "place", "contact", "no_effect", "unknown"})
EFFECT_PREDICATES = frozenset(
    {
        "object_attached",
        "object_supported_by_target",
        "target_relation_satisfied",
        "target_region_occupied_by_manipulated_object",
        "gripper_empty",
        "object_released",
        "placement_stable",
    }
)
VERIFIABILITY_VALUES = frozenset({"verified", "contradicted", "unverified"})
VERIFIER_SOURCES = frozenset(
    {
        "scene_memory_delta",
        "robot_state",
        "attachment_state",
        "runtime_grasp_validation",
        "runtime_place_validation",
        "vlm_action_effect_verifier",
    }
)
EVIDENCE_VERDICTS = frozenset({"support", "oppose", "unverified"})
REALIZATION_STATUSES = frozenset({"satisfied", "violated", "unknown"})
MOTION_STATUSES = frozenset(
    {"completed", "failed_before_effect", "interrupted", "unknown"}
)
ENTRY_STATUSES = frozenset({"candidate", "accepted", "deprecated"})
EVALUATION_STATUSES = frozenset({"not_evaluated", "passed", "failed"})
SOURCE_KINDS = frozenset({"agent_generated", "benchmark_expert", "human_authored"})
ACCEPTANCE_SCOPES = frozenset(
    {"formal_no_prior", "expert_prior", "integration_only", "oracle_diagnostic"}
)
SNAPSHOT_PURPOSES = frozenset({"static_integration", "static_evaluation"})
AUDIT_RETRIEVAL_STAGES = frozenset({"pre_planner_task", "post_binding_geometry"})
AUDIT_REJECTION_REASON_ORDER = (
    "insufficient_information",
    "condition_unresolved",
    "task_strategy_unresolved",
    "target_unbound",
    "no_exact_match",
    "domain_capability_mismatch",
    "context_budget_exceeded",
    "zero_compliant_candidates",
)
AUDIT_REJECTION_REASONS = frozenset(AUDIT_REJECTION_REASON_ORDER)


def _fail(path: str, message: str) -> None:
    raise HPKValidationError(f"{path}: {message}" if path else message)


def _validate_json(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "NaN and Infinity are not valid JSON values")
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


def _json_copy(value: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    copied = copy.deepcopy(dict(value))
    _validate_json(copied, path=path)
    return copied


def canonical_json_bytes(value: Any) -> bytes:
    """返回 HPK 标识符使用的规范字节表示。"""

    _validate_json(value, path="canonical_payload")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def stable_content_id(prefix: str, payload: Any) -> str:
    normalized_prefix = str(prefix or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", normalized_prefix):
        _fail("prefix", "must contain lowercase letters, digits, or underscores")
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"{normalized_prefix}_{digest}"


def validate_content_id(value: Any, *, prefix: str, path: str) -> str:
    text = _string(value, path=path)
    if not re.fullmatch(rf"{re.escape(prefix)}_[0-9a-f]{{64}}", text):
        _fail(path, f"must be a canonical {prefix}_<sha256> content ID")
    return text


def _expect_fields(
    value: Any,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    if missing:
        _fail(path, "missing required field(s): " + ", ".join(missing))
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail(path, "unknown field(s): " + ", ".join(unknown))
    return value


def _string(value: Any, *, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    if not allow_empty and not value.strip():
        _fail(path, "must be a non-empty string")
    return value


def _nullable_string(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path=path)


def _boolean(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _nullable_boolean(value: Any, *, path: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, path=path)


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, "must be an integer")
    if value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _number(
    value: Any,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be finite")
    if minimum is not None and result < minimum:
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and result > maximum:
        _fail(path, f"must be <= {maximum}")
    return result


def _enum(value: Any, *, allowed: frozenset[str], path: str) -> str:
    text = _string(value, path=path)
    if text not in allowed:
        _fail(path, f"must be one of {sorted(allowed)}")
    return text


def _nullable_enum(value: Any, *, allowed: frozenset[str], path: str) -> str | None:
    if value is None:
        return None
    return _enum(value, allowed=allowed, path=path)


def _string_list(
    value: Any,
    *,
    path: str,
    allowed: frozenset[str] | None = None,
    unique: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        text = _string(item, path=item_path)
        if allowed is not None and text not in allowed:
            _fail(item_path, f"must be one of {sorted(allowed)}")
        result.append(text)
    if unique and len(result) != len(set(result)):
        _fail(path, "must not contain duplicate values")
    return result


def _utc(value: Any, *, path: str) -> str:
    text = _string(value, path=path)
    if not _UTC_RE.fullmatch(text):
        _fail(path, "must be RFC3339 UTC with a trailing Z")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        _fail(path, f"invalid UTC timestamp ({exc})")
    return text


def _sha256(value: Any, *, path: str) -> str:
    text = _string(value, path=path)
    if not _SHA256_RE.fullmatch(text):
        _fail(path, "must be a lowercase SHA-256 digest")
    return text


_PRIVATE_EXACT_KEYS = frozenset(
    {
        "track_id",
        "track_ids",
        "instance_id",
        "instance_ids",
        "candidate_id",
        "candidate_ids",
        "operation_candidate_id",
        "operation_candidate_ids",
        "absolute_xyz",
        "world_m",
        "pose",
        "poses",
        "se3",
        "quaternion",
        "quat_wxyz",
        "transform",
        "transformation",
        "rotation_matrix",
        "translation",
        "joint_trajectory",
        "joint_trajectories",
        "action_vector",
        "action_vectors",
        "seed",
        "seeds",
        "episode_id",
        "episode_specific_answer",
        "answer",
        "image",
        "images",
        "video",
        "videos",
        "local_path",
        "file_path",
        "image_path",
        "video_path",
        "mask_path",
        "depth_path",
        "rgb_path",
        "raw_mask",
        "raw_depth",
        "raw_rgb",
        "raw_payload",
        "benchmark_hidden_state",
        "oracle_contact_matrix",
    }
)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _private_transferable_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized in _PRIVATE_EXACT_KEYS:
        return True
    if normalized.endswith("_world_m") or normalized.endswith("_absolute_xyz"):
        return True
    if normalized.endswith(("_image_path", "_video_path", "_local_path")):
        return True
    tokens = set(normalized.split("_"))
    if "pose" in tokens or "quaternion" in tokens or "se3" in tokens:
        return True
    if {"joint", "trajectory"} <= tokens or {"action", "vector"} <= tokens:
        return True
    if "candidate" in tokens and "id" in tokens:
        return True
    if "instance" in tokens and "id" in tokens:
        return True
    if "track" in tokens and "id" in tokens:
        return True
    return False


def reject_private_transferable(value: Any, *, path: str = "transferable") -> None:
    """Recursively reject runtime-private fields in transferable HPK data.

    Relative ``orientation`` is allowed because it is a typed relation in the
    HPK contract.  Only absolute pose/orientation representations are private.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            child = f"{path}.{key}"
            if _private_transferable_key(key):
                _fail(child, "runtime-private field is forbidden in transferable HPK")
            reject_private_transferable(item, path=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_private_transferable(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        # Frozen identifiers/version fields intentionally use slash-delimited
        # components.  Typed content references are also allowed only in their
        # exact structural fields; the same values embedded in free text are
        # private and rejected below.
        if _structured_transfer_text_exempt(path):
            return
        reason = _private_transfer_text_reason(value)
        if reason is not None:
            _fail(
                path,
                reason,
            )


_BARE_FILE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])[^\s/\\]+\."
    r"(?:avi|bin|csv|h5|hdf5|jpeg|jpg|json|jsonl|log|mp4|npy|npz|pkl|png|"
    r"pt|pth|text|toml|txt|yaml|yml)(?::\d+)?\b",
    re.IGNORECASE,
)
_NUMBER_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_AXIS_ASSIGNMENT_RES = tuple(
    re.compile(rf"\b{axis}\s*[:=]\s*{_NUMBER_TOKEN}\b", re.IGNORECASE)
    for axis in ("x", "y", "z")
)
_WORLD_COORDINATE_TEXT_RE = re.compile(
    rf"\bworld\s+(?:coordinates?|positions?)\b.{{0,64}}?{_NUMBER_TOKEN}"
    rf"(?:\s*,?\s+){_NUMBER_TOKEN}(?:\s*,?\s+){_NUMBER_TOKEN}",
    re.IGNORECASE | re.DOTALL,
)
_POSE_ORIENTATION_TEXT_RE = re.compile(
    rf"\b(?:absolute\s+(?:pose|position)|world\s+(?:pose|position)|pose|"
    rf"quaternion|quat_wxyz|se\s*\(?3\)?)\b.{{0,64}}?{_NUMBER_TOKEN}"
    rf"(?:\s*,?\s+){_NUMBER_TOKEN}(?:\s*,?\s+){_NUMBER_TOKEN}",
    re.IGNORECASE | re.DOTALL,
)
_SPACED_PRIVATE_ID_RE = re.compile(
    r"\b(?:operation\s+candidate|candidate|track|instance)"
    r"(?:\s+(?:id|no\.?|number))?\s+(?:#\s*)?"
    r"[A-Za-z0-9_.:-]+\b",
    re.IGNORECASE,
)
_PRIVATE_HPK_REFERENCE_RE = re.compile(
    r"\b(?:afkentry|afkc|afku|afkz|afkfx|afkev|afkeval|afksnap|"
    r"afkaudit|afkprivrank)_[0-9a-f]{64}\b",
    re.IGNORECASE,
)
_PRIVATE_GEOMETRY_DIAGNOSTIC_RE = re.compile(
    rf"(?:\b(?:reach_distance_m|candidate[_\s]+(?:rank|score)|"
    rf"private\s+(?:rank|score)|rank)\s*[:=]\s*{_NUMBER_TOKEN}\b"
    r"|\b(?:geometry_source_class|reach_distance_bucket|geometric_compliance|"
    r"afk_scores?|afk_score_components|baseline_rank|candidate_rank|"
    r"selected_candidate_private_ref|candidate_rejection_details)\b)",
    re.IGNORECASE,
)
_PRIVATE_RUN_METADATA_RE = re.compile(
    r"\b(?:seed|episode[_\s]+id|benchmark[_\s]+hidden[_\s]+state|"
    r"oracle[_\s]+contact[_\s]+matrix|correct[_\s]+combination|"
    r"ground[_\s]+truth|episode[_\s]+answer|oracle[_\s]+answer|"
    r"episode[_\s]+specific[_\s]+answer)\b",
    re.IGNORECASE,
)
_RAW_PLANNER_SOURCE_RE = re.compile(
    r"\b(?:planner[_\s]+subtask[_\s]+text|raw[_\s]+planner[_\s]+source|"
    r"planner[_\s]+source[_\s]+text)\b",
    re.IGNORECASE,
)
_HPK_CONTEXT_DELIMITER_RE = re.compile(
    r"</?(?:hierarchical_physical|action_feedback)_knowledge_context>", re.IGNORECASE
)


def _structured_transfer_text_exempt(path: str) -> bool:
    if path.endswith(
        (
            ".schema",
            ".abstraction_version",
            ".normalization_version",
            ".promotion_policy_id",
            ".scoring_profile",
            ".profile",
        )
    ):
        return True
    leaf = re.sub(r"\[\d+\]$", "", path).rsplit(".", 1)[-1]
    if leaf in {
        "entry_id",
        "evaluation_ref",
        "usage_audit_id",
        "snapshot_id",
        "snapshot_manifest_sha256",
        "selected_entry_id",
        "condition_id",
        "task_strategy_id",
        "selected_geometric_strategy_id",
        "planner_context_sha256",
    }:
        return True
    return (
        ".evidence_refs." in path
        or ".retrieved_entry_ids[" in path
        or ".missing_fields[" in path
    )


_PRIVATE_TEXT_PATTERNS = (
    re.compile(
        r"\b(?:track|instance|candidate|operation_candidate)[_-]?id\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:xyz|world_m|pose|se\s*\(?3\)?|quaternion|quat_wxyz)"
        r"\s*[:=]\s*(?:\[|\(|[+-]?(?:\d|\.\d))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:seed|episode_id|benchmark_hidden_state)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|\s)(?:/[^\s/]+){1,}(?:/[^\s]+)?"),
    re.compile(r"\b[A-Za-z]:[\\/]"),
    re.compile(r"(?:^|[\s\"'`(\[{:=>])\.\.?/[A-Za-z0-9_.-]"),
    re.compile(
        r"(?:^|[\s\"'`(\[{:=>])"
        r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,16}"
        r"(?=$|[\s\"'`),\]}])"
    ),
    re.compile(
        r"(?:^|[\s\"'`(\[{:=>])"
        r"(?:[A-Za-z0-9_.-]+/){2,}[A-Za-z0-9_.-]+"
        r"(?=$|[\s\"'`),\]}])"
    ),
    re.compile(r"data:image/|base64", re.IGNORECASE),
    re.compile(
        r"(?:\[|\()[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:\s*,\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)){2,}(?:\]|\))"
    ),
    re.compile(r"\btrack[_:-]?\d+\b", re.IGNORECASE),
    re.compile(
        r"\b(?:operation[_-]?)?candidate(?:[_-]?id)?[:=/_-]"
        r"(?:[A-Za-z0-9][\w:.-]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\binstance(?:[_-]?id)?[:=/_-]"
        r"(?:\d+|[A-Fa-f0-9]{8,}|[A-Za-z][\w.-]*[_:-]\d+)\b",
        re.IGNORECASE,
    ),
    _SPACED_PRIVATE_ID_RE,
    _PRIVATE_HPK_REFERENCE_RE,
    _PRIVATE_GEOMETRY_DIAGNOSTIC_RE,
    _PRIVATE_RUN_METADATA_RE,
    _RAW_PLANNER_SOURCE_RE,
    _HPK_CONTEXT_DELIMITER_RE,
)


def _private_transfer_text_reason(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if _HPK_CONTEXT_DELIMITER_RE.search(value):
        return "runtime-private model-visible text must not contain an HPK context delimiter"
    if "/" in value or "\\" in value or _BARE_FILE_TOKEN_RE.search(value):
        return "runtime-private local path text is forbidden"
    if all(pattern.search(value) for pattern in _AXIS_ASSIGNMENT_RES):
        return "runtime-private world coordinate text is forbidden"
    if _WORLD_COORDINATE_TEXT_RE.search(value):
        return "runtime-private world coordinate text is forbidden"
    if _POSE_ORIENTATION_TEXT_RE.search(value):
        return "runtime-private geometry or pose text is forbidden"
    if _SPACED_PRIVATE_ID_RE.search(value):
        return "runtime-private identifier text is forbidden"
    if _PRIVATE_HPK_REFERENCE_RE.search(value):
        return "runtime-private HPK reference text is forbidden"
    if _PRIVATE_GEOMETRY_DIAGNOSTIC_RE.search(value):
        return "runtime-private geometry diagnostic is forbidden"
    if _PRIVATE_RUN_METADATA_RE.search(value):
        return "runtime-private run metadata, hidden answer, or seed text is forbidden"
    if _RAW_PLANNER_SOURCE_RE.search(value):
        return "runtime-private raw planner source text is forbidden"
    if any(pattern.search(value) for pattern in _PRIVATE_TEXT_PATTERNS):
        return "runtime-private identity, geometry, or path text is forbidden"
    return None


def contains_private_transfer_text(value: Any) -> bool:
    """Return whether free text contains a recognizable runtime-private value."""

    return _private_transfer_text_reason(value) is not None


class _StrictRecord(Mapping[str, Any]):
    """Read-only mapping facade over one validated strict JSON object."""

    SCHEMA: ClassVar[str | None] = None
    LEGACY_SCHEMA: ClassVar[str | None] = None
    TRANSFERABLE: ClassVar[bool] = False
    __slots__ = ("_data",)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        data = _json_copy(payload, path=self.__class__.__name__)
        if self.TRANSFERABLE:
            reject_private_transferable(data, path=self.__class__.__name__)
        self._validate(data)
        self._data = data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_StrictRecord":
        return cls(payload)

    @classmethod
    def from_json(cls, payload: str) -> "_StrictRecord":
        def reject_constant(value: str) -> None:
            raise HPKValidationError(f"invalid non-finite JSON number: {value}")

        try:
            decoded = json.loads(payload, parse_constant=reject_constant)
        except (json.JSONDecodeError, TypeError) as exc:
            raise HPKValidationError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(decoded)

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        if cls.SCHEMA is None:
            return
        if payload.get("schema") != cls.SCHEMA and (
            cls.LEGACY_SCHEMA is None or payload.get("schema") != cls.LEGACY_SCHEMA
        ):
            _fail(
                f"{cls.__name__}.schema",
                f"unsupported schema; expected {cls.SCHEMA!r}",
            )

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_json(self) -> str:
        return canonical_json(self._data)

    @property
    def schema(self) -> str | None:
        value = self._data.get("schema")
        return str(value) if value is not None else None

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"


class ConditionV1(_StrictRecord):
    SCHEMA = CONDITION_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/condition/v1"
    TRANSFERABLE = True

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "schema",
                "task_family",
                "manipulation_phase",
                "operation",
                "manipulated_object",
                "target",
                "scene_predicates",
                "preconditions",
                "abstraction_version",
            ),
            path=path,
        )
        _string(payload["task_family"], path=f"{path}.task_family")
        _string(payload["manipulation_phase"], path=f"{path}.manipulation_phase")
        _enum(payload["operation"], allowed=OPERATIONS, path=f"{path}.operation")
        manipulated = _expect_fields(
            payload["manipulated_object"],
            required=("semantic_class", "geometry_class", "role", "held_state"),
            path=f"{path}.manipulated_object",
        )
        for key in ("semantic_class", "geometry_class", "role"):
            _string(manipulated[key], path=f"{path}.manipulated_object.{key}")
        _enum(
            manipulated["held_state"],
            allowed=HELD_STATES,
            path=f"{path}.manipulated_object.held_state",
        )
        target = _expect_fields(
            payload["target"],
            required=("semantic_class", "geometry_class", "role", "relation"),
            path=f"{path}.target",
        )
        for key in ("semantic_class", "geometry_class", "role"):
            _nullable_string(target[key], path=f"{path}.target.{key}")
        _nullable_enum(
            target["relation"],
            allowed=TARGET_RELATIONS,
            path=f"{path}.target.relation",
        )
        _string_list(
            payload["scene_predicates"],
            path=f"{path}.scene_predicates",
            allowed=SCENE_PREDICATES,
        )
        _string_list(payload["preconditions"], path=f"{path}.preconditions")
        if payload["abstraction_version"] not in {
            CONDITION_ABSTRACTION_VERSION, "afk_condition_builder/v1"
        }:
            _fail(
                f"{path}.abstraction_version",
                f"expected {CONDITION_ABSTRACTION_VERSION!r}",
            )

    @property
    def stable_id(self) -> str:
        return stable_content_id("afkc", self._data)


class TaskStrategyV1(_StrictRecord):
    SCHEMA = TASK_STRATEGY_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/task_strategy/v1"
    TRANSFERABLE = True

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "schema",
                "operation",
                "manipulated_role",
                "target_role",
                "target_relation",
                "manipulation_phase",
                "subgoal_purpose",
                "preferred_arm",
                "source",
            ),
            path=path,
        )
        _enum(payload["operation"], allowed=OPERATIONS, path=f"{path}.operation")
        _string(payload["manipulated_role"], path=f"{path}.manipulated_role")
        _nullable_string(payload["target_role"], path=f"{path}.target_role")
        _nullable_enum(
            payload["target_relation"],
            allowed=TARGET_RELATIONS,
            path=f"{path}.target_relation",
        )
        _string(payload["manipulation_phase"], path=f"{path}.manipulation_phase")
        _string(payload["subgoal_purpose"], path=f"{path}.subgoal_purpose")
        _enum(payload["preferred_arm"], allowed=ARMS, path=f"{path}.preferred_arm")
        source = _expect_fields(
            payload["source"],
            required=(
                "planner_subtask_text",
                "selected_skill",
                "action_mode",
                "normalization_version",
            ),
            path=f"{path}.source",
        )
        _string(
            source["planner_subtask_text"],
            path=f"{path}.source.planner_subtask_text",
        )
        if contains_private_transfer_text(source["planner_subtask_text"]):
            _fail(
                f"{path}.source.planner_subtask_text",
                "contains runtime-private identity, geometry, or path text",
            )
        if contains_private_transfer_text(payload["subgoal_purpose"]):
            _fail(
                f"{path}.subgoal_purpose",
                "contains runtime-private identity, geometry, or path text",
            )
        _string(source["selected_skill"], path=f"{path}.source.selected_skill")
        if source["action_mode"] != payload["operation"]:
            _fail(f"{path}.source.action_mode", "must equal operation")
        if source["normalization_version"] != TASK_STRATEGY_NORMALIZATION_VERSION:
            _fail(
                f"{path}.source.normalization_version",
                f"expected {TASK_STRATEGY_NORMALIZATION_VERSION!r}",
            )
        if payload["operation"] == "place" and (
            payload["target_role"] is None or payload["target_relation"] is None
        ):
            _fail(path, "place strategies require target_role and target_relation")

    def transferable_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("source", None)
        return payload

    @property
    def stable_id(self) -> str:
        return stable_content_id("afku", self.transferable_dict())


class GeometricStrategyV1(_StrictRecord):
    SCHEMA = GEOMETRIC_STRATEGY_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/geometric_strategy/v1"
    TRANSFERABLE = True

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "schema",
                "strategy_family",
                "reference_frame",
                "target_relation",
                "approach",
                "orientation",
                "grasp",
                "hard_constraints",
                "soft_preferences",
                "avoid",
                "capability_evidence",
            ),
            path=path,
        )
        _enum(
            payload["strategy_family"],
            allowed=STRATEGY_FAMILIES,
            path=f"{path}.strategy_family",
        )
        _enum(
            payload["reference_frame"],
            allowed=REFERENCE_FRAMES,
            path=f"{path}.reference_frame",
        )
        relation = _expect_fields(
            payload["target_relation"],
            required=("relation", "reference_role"),
            path=f"{path}.target_relation",
        )
        _nullable_enum(
            relation["relation"],
            allowed=TARGET_RELATIONS,
            path=f"{path}.target_relation.relation",
        )
        _nullable_string(
            relation["reference_role"],
            path=f"{path}.target_relation.reference_role",
        )
        approach = _expect_fields(
            payload["approach"],
            required=("family", "direction_bucket"),
            path=f"{path}.approach",
        )
        _enum(
            approach["family"],
            allowed=APPROACH_FAMILIES,
            path=f"{path}.approach.family",
        )
        _enum(
            approach["direction_bucket"],
            allowed=DIRECTION_BUCKETS,
            path=f"{path}.approach.direction_bucket",
        )
        orientation = _expect_fields(
            payload["orientation"],
            required=("relation",),
            path=f"{path}.orientation",
        )
        _enum(
            orientation["relation"],
            allowed=ORIENTATION_RELATIONS,
            path=f"{path}.orientation.relation",
        )
        grasp = _expect_fields(
            payload["grasp"],
            required=("region", "semantic_part"),
            path=f"{path}.grasp",
        )
        _enum(grasp["region"], allowed=GRASP_REGIONS, path=f"{path}.grasp.region")
        semantic_part = _nullable_string(
            grasp["semantic_part"], path=f"{path}.grasp.semantic_part"
        )
        _string_list(
            payload["hard_constraints"],
            path=f"{path}.hard_constraints",
            allowed=HARD_CONSTRAINTS,
        )
        _string_list(
            payload["soft_preferences"],
            path=f"{path}.soft_preferences",
            allowed=SOFT_PREFERENCES,
        )
        _string_list(
            payload["avoid"],
            path=f"{path}.avoid",
            allowed=AVOID_CONSTRAINTS,
        )
        capability = _expect_fields(
            payload["capability_evidence"],
            required=("semantic_part_observed", "geometry_source_class"),
            path=f"{path}.capability_evidence",
        )
        observed = _boolean(
            capability["semantic_part_observed"],
            path=f"{path}.capability_evidence.semantic_part_observed",
        )
        _enum(
            capability["geometry_source_class"],
            allowed=GEOMETRY_SOURCE_CLASSES,
            path=f"{path}.capability_evidence.geometry_source_class",
        )
        if grasp["region"] == "semantic_part":
            if semantic_part is None or not observed:
                _fail(
                    f"{path}.grasp",
                    "semantic_part requires a non-empty observed semantic part",
                )
        elif semantic_part is not None:
            _fail(
                f"{path}.grasp.semantic_part",
                "must be null unless region is semantic_part",
            )
        elif observed:
            _fail(
                f"{path}.capability_evidence.semantic_part_observed",
                "cannot be true when no semantic part is selected",
            )

    @property
    def stable_id(self) -> str:
        return stable_content_id("afkz", self._data)


class AbstractEffectV1(_StrictRecord):
    SCHEMA = ABSTRACT_EFFECT_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/abstract_effect/v1"
    TRANSFERABLE = True

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=("schema", "effect_type", "expected_predicates", "verifiability"),
            optional=("observed_predicates", "verifier_sources"),
            path=path,
        )
        _enum(payload["effect_type"], allowed=EFFECT_TYPES, path=f"{path}.effect_type")
        _string_list(
            payload["expected_predicates"],
            path=f"{path}.expected_predicates",
            allowed=EFFECT_PREDICATES,
        )
        _enum(
            payload["verifiability"],
            allowed=VERIFIABILITY_VALUES,
            path=f"{path}.verifiability",
        )
        if "observed_predicates" in payload:
            _string_list(
                payload["observed_predicates"],
                path=f"{path}.observed_predicates",
                allowed=EFFECT_PREDICATES,
            )
        if "verifier_sources" in payload:
            _string_list(
                payload["verifier_sources"],
                path=f"{path}.verifier_sources",
                allowed=VERIFIER_SOURCES,
            )

    def expected_projection(self) -> dict[str, Any]:
        return {
            "schema": self._data["schema"],
            "effect_type": self._data["effect_type"],
            "expected_predicates": copy.deepcopy(self._data["expected_predicates"]),
            "verifiability": self._data["verifiability"],
        }

    @property
    def stable_id(self) -> str:
        return stable_content_id("afkfx", self.expected_projection())


def evidence_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    data.pop("evidence_id", None)
    # Creation time is audit metadata, not event identity.  Re-importing the
    # same immutable event therefore deduplicates deterministically.
    data.pop("created_at", None)
    return data


def evidence_id_for(payload: Mapping[str, Any]) -> str:
    return stable_content_id("afkev", evidence_identity_payload(payload))


class EvidenceV1(_StrictRecord):
    SCHEMA = EVIDENCE_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/evidence/v1"

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "schema",
                "evidence_id",
                "episode_ref",
                "step_or_segment_ref",
                "condition_id",
                "task_strategy_id",
                "geometric_strategy_id",
                "expected_effect_id",
                "observed_effect",
                "verdict",
                "confidence",
                "realization_status",
                "motion_status",
                "provenance",
                "created_at",
            ),
            path=path,
        )
        for key in ("evidence_id", "episode_ref", "step_or_segment_ref"):
            _string(payload[key], path=f"{path}.{key}")
        for key, prefix in (
            ("condition_id", "afkc"),
            ("task_strategy_id", "afku"),
            ("geometric_strategy_id", "afkz"),
            ("expected_effect_id", "afkfx"),
        ):
            validate_content_id(payload[key], prefix=prefix, path=f"{path}.{key}")
        observed = _expect_fields(
            payload["observed_effect"],
            required=("effect_type", "predicates"),
            path=f"{path}.observed_effect",
        )
        _enum(
            observed["effect_type"],
            allowed=EFFECT_TYPES,
            path=f"{path}.observed_effect.effect_type",
        )
        _string_list(
            observed["predicates"],
            path=f"{path}.observed_effect.predicates",
            allowed=EFFECT_PREDICATES,
        )
        _enum(payload["verdict"], allowed=EVIDENCE_VERDICTS, path=f"{path}.verdict")
        _number(
            payload["confidence"], path=f"{path}.confidence", minimum=0.0, maximum=1.0
        )
        _enum(
            payload["realization_status"],
            allowed=REALIZATION_STATUSES,
            path=f"{path}.realization_status",
        )
        _enum(
            payload["motion_status"],
            allowed=MOTION_STATUSES,
            path=f"{path}.motion_status",
        )
        if payload["verdict"] in {"support", "oppose"} and (
            payload["motion_status"] != "completed"
            or payload["realization_status"] != "satisfied"
        ):
            _fail(
                f"{path}.verdict",
                "support/oppose require completed motion and satisfied realization",
            )
        provenance = _expect_fields(
            payload["provenance"],
            required=(
                "geometry_source_class",
                "selected_candidate_private_ref",
                "trace_event_refs",
                "verifier_summary",
                "runtime_validation_summary",
            ),
            path=f"{path}.provenance",
        )
        _enum(
            provenance["geometry_source_class"],
            allowed=GEOMETRY_SOURCE_CLASSES,
            path=f"{path}.provenance.geometry_source_class",
        )
        _nullable_string(
            provenance["selected_candidate_private_ref"],
            path=f"{path}.provenance.selected_candidate_private_ref",
        )
        _string_list(
            provenance["trace_event_refs"],
            path=f"{path}.provenance.trace_event_refs",
        )
        _string(
            provenance["verifier_summary"],
            path=f"{path}.provenance.verifier_summary",
            allow_empty=True,
        )
        if not isinstance(provenance["runtime_validation_summary"], dict):
            _fail(f"{path}.provenance.runtime_validation_summary", "must be an object")
        _utc(payload["created_at"], path=f"{path}.created_at")
        expected_id = evidence_id_for(payload)
        if payload["evidence_id"] != expected_id:
            _fail(
                f"{path}.evidence_id",
                f"content identity mismatch; expected {expected_id}",
            )


class CandidateGeometryFeaturesV1(_StrictRecord):
    """ID-free semantic projection of one private runtime candidate."""

    TRANSFERABLE = True

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "action_mode",
                "target_relation",
                "target_reference_role",
                "support_valid",
                "target_region_free",
                "approach_family",
                "approach_direction_bucket",
                "orientation_relation",
                "grasp_region",
                "semantic_part",
                "grasp_width_bucket",
                "reach_distance_bucket",
                "geometry_source_class",
            ),
            path=path,
        )
        _enum(payload["action_mode"], allowed=OPERATIONS, path=f"{path}.action_mode")
        _nullable_enum(
            payload["target_relation"],
            allowed=TARGET_RELATIONS,
            path=f"{path}.target_relation",
        )
        target_reference_role = _nullable_string(
            payload["target_reference_role"],
            path=f"{path}.target_reference_role",
        )
        if target_reference_role is not None and contains_private_transfer_text(
            target_reference_role
        ):
            _fail(
                f"{path}.target_reference_role",
                "contains runtime-private identity, geometry, or path text",
            )
        _nullable_boolean(payload["support_valid"], path=f"{path}.support_valid")
        _nullable_boolean(
            payload["target_region_free"], path=f"{path}.target_region_free"
        )
        _enum(
            payload["approach_family"],
            allowed=APPROACH_FAMILIES,
            path=f"{path}.approach_family",
        )
        _enum(
            payload["approach_direction_bucket"],
            allowed=DIRECTION_BUCKETS,
            path=f"{path}.approach_direction_bucket",
        )
        _enum(
            payload["orientation_relation"],
            allowed=ORIENTATION_RELATIONS,
            path=f"{path}.orientation_relation",
        )
        _enum(
            payload["grasp_region"], allowed=GRASP_REGIONS, path=f"{path}.grasp_region"
        )
        semantic_part = _nullable_string(
            payload["semantic_part"], path=f"{path}.semantic_part"
        )
        if payload["grasp_region"] == "semantic_part" and semantic_part is None:
            _fail(f"{path}.semantic_part", "semantic_part region requires a value")
        if payload["grasp_region"] != "semantic_part" and semantic_part is not None:
            _fail(f"{path}.semantic_part", "must be null without semantic_part region")
        _enum(
            payload["grasp_width_bucket"],
            allowed=frozenset({"narrow", "medium", "wide", "unknown"}),
            path=f"{path}.grasp_width_bucket",
        )
        _enum(
            payload["reach_distance_bucket"],
            allowed=frozenset({"near", "medium", "far", "unknown"}),
            path=f"{path}.reach_distance_bucket",
        )
        _enum(
            payload["geometry_source_class"],
            allowed=GEOMETRY_SOURCE_CLASSES,
            path=f"{path}.geometry_source_class",
        )


class ObserveAuditRecord(_StrictRecord):
    """Strict internal P0-A HPK sidecar profile."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(normalize_observe_audit(payload))

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "profile",
                "mode",
                "condition_id",
                "task_strategy_id",
                "retrieved_task_hpk_ids",
                "selected_geometric_hpk_id",
                "candidate_ids_before",
                "candidate_rank_before",
                "candidate_ids_after",
                "candidate_rank_after",
                "selected_candidate_id",
                "geometric_compliance",
                "motion_status",
                "effect_verdict",
                "evidence_id",
                "parent_snapshot_id",
                "child_snapshot_id",
            ),
            path=path,
        )
        if payload["profile"] != OBSERVE_AUDIT_PROFILE:
            _fail(
                f"{path}.profile",
                f"expected {OBSERVE_AUDIT_PROFILE!r}",
            )
        if payload["mode"] != "observe":
            _fail(f"{path}.mode", "P0-A audit mode must be observe")
        for key in (
            "condition_id",
            "task_strategy_id",
            "selected_geometric_hpk_id",
            "selected_candidate_id",
            "evidence_id",
            "parent_snapshot_id",
            "child_snapshot_id",
        ):
            _nullable_string(payload[key], path=f"{path}.{key}")
        for key, prefix in (
            ("condition_id", "afkc"),
            ("task_strategy_id", "afku"),
            ("evidence_id", "afkev"),
        ):
            if payload[key] is not None:
                validate_content_id(
                    payload[key],
                    prefix=prefix,
                    path=f"{path}.{key}",
                )
        _string_list(
            payload["retrieved_task_hpk_ids"], path=f"{path}.retrieved_task_hpk_ids"
        )
        before_ids = _string_list(
            payload["candidate_ids_before"], path=f"{path}.candidate_ids_before"
        )
        after_ids = _string_list(
            payload["candidate_ids_after"], path=f"{path}.candidate_ids_after"
        )
        for key, expected_len in (
            ("candidate_rank_before", len(before_ids)),
            ("candidate_rank_after", len(after_ids)),
        ):
            ranks = payload[key]
            if not isinstance(ranks, list) or len(ranks) != expected_len:
                _fail(f"{path}.{key}", "must align one-to-one with candidate IDs")
            for index, value in enumerate(ranks):
                _integer(value, path=f"{path}.{key}[{index}]", minimum=0)
            if len(ranks) != len(set(ranks)):
                _fail(f"{path}.{key}", "must contain unique ranks")
        compliance = payload["geometric_compliance"]
        if not isinstance(compliance, bool) and compliance != "unverified":
            _fail(
                f"{path}.geometric_compliance",
                "must be a boolean or the literal 'unverified'",
            )
        _enum(
            payload["motion_status"],
            allowed=MOTION_STATUSES,
            path=f"{path}.motion_status",
        )
        _enum(
            payload["effect_verdict"],
            allowed=EVIDENCE_VERDICTS,
            path=f"{path}.effect_verdict",
        )
        if (
            payload["parent_snapshot_id"] is not None
            or payload["child_snapshot_id"] is not None
        ):
            _fail(path, "observe mode cannot read or publish HPK snapshots")
        if (
            payload["retrieved_task_hpk_ids"]
            or payload["selected_geometric_hpk_id"] is not None
        ):
            _fail(path, "P0-A observe mode cannot retrieve or select HPK")
        if (
            before_ids != after_ids
            or payload["candidate_rank_before"] != payload["candidate_rank_after"]
        ):
            _fail(path, "P0-A observe mode cannot change candidate order")
        selected = payload["selected_candidate_id"]
        if selected is not None and selected not in after_ids:
            _fail(f"{path}.selected_candidate_id", "must reference a ranked candidate")
        if payload["geometric_compliance"] != "unverified":
            _fail(path, "P0-A observe mode has no active geometric strategy")


_PROMOTION_POLICY_RE = re.compile(r"^(?:hpk|afk)_[a-z0-9_]+/v[1-9][0-9]*$")


def _sorted_string_list(
    value: Any,
    *,
    path: str,
    allowed: frozenset[str] | None = None,
    nonempty: bool = False,
) -> list[str]:
    result = _string_list(value, path=path, allowed=allowed)
    if nonempty and not result:
        _fail(path, "must contain at least one value")
    if result != sorted(result):
        _fail(path, "must be sorted lexicographically")
    return result


def entry_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact transferable identity projection for EntryV1."""

    if not isinstance(payload, Mapping):
        _fail("EntryV1", "must be an object")
    condition = ConditionV1.from_dict(payload.get("condition"))
    task_strategy = TaskStrategyV1.from_dict(payload.get("task_strategy"))
    geometric_strategy = GeometricStrategyV1.from_dict(
        payload.get("geometric_strategy")
    )
    expected_effect = AbstractEffectV1.from_dict(payload.get("expected_effect"))
    return {
        "schema": payload.get("schema", ENTRY_SCHEMA),
        "condition": condition.to_dict(),
        "task_strategy": task_strategy.transferable_dict(),
        "geometric_strategy": geometric_strategy.to_dict(),
        "expected_effect": expected_effect.expected_projection(),
    }


def entry_id_for(payload: Mapping[str, Any]) -> str:
    return stable_content_id("afkentry", entry_identity_payload(payload))


def _validate_entry_provenance(
    provenance: Mapping[str, Any],
    *,
    acceptance_scope: str,
    path: str,
) -> dict[str, Any]:
    value = _expect_fields(
        normalize_knowledge_metadata(provenance),
        required=(
            "source_kind",
            "expert_derived",
            "oracle_derived",
            "human_prior_used",
            "learned_hpk",
            "formal_evaluation_eligible",
            "domain_ids",
        ),
        path=path,
    )
    source_kind = _enum(
        value["source_kind"], allowed=SOURCE_KINDS, path=f"{path}.source_kind"
    )
    expert_derived = _boolean(value["expert_derived"], path=f"{path}.expert_derived")
    oracle_derived = _boolean(value["oracle_derived"], path=f"{path}.oracle_derived")
    human_prior_used = _boolean(
        value["human_prior_used"], path=f"{path}.human_prior_used"
    )
    learned_hpk = _boolean(value["learned_hpk"], path=f"{path}.learned_hpk")
    formal_eligible = _boolean(
        value["formal_evaluation_eligible"],
        path=f"{path}.formal_evaluation_eligible",
    )
    _sorted_string_list(value["domain_ids"], path=f"{path}.domain_ids", nonempty=True)

    if oracle_derived:
        if acceptance_scope != "oracle_diagnostic" or formal_eligible:
            _fail(
                path,
                "oracle-derived entries require oracle_diagnostic scope and "
                "formal_evaluation_eligible=false",
            )
    if source_kind == "agent_generated":
        if expert_derived or human_prior_used or not learned_hpk:
            _fail(
                path,
                "agent_generated requires expert_derived=false, "
                "human_prior_used=false, and learned_hpk=true",
            )
        expected_scope = "oracle_diagnostic" if oracle_derived else "formal_no_prior"
        if acceptance_scope != expected_scope:
            _fail(path, f"agent_generated requires {expected_scope!r} scope")
        if not oracle_derived and not formal_eligible:
            _fail(
                path,
                "non-oracle agent_generated entries must be formal-evaluation eligible",
            )
    elif source_kind == "benchmark_expert":
        if not expert_derived or human_prior_used or not learned_hpk:
            _fail(
                path,
                "benchmark_expert requires expert_derived=true, "
                "human_prior_used=false, and learned_hpk=true",
            )
        if not oracle_derived and acceptance_scope != "expert_prior":
            _fail(path, "benchmark_expert requires expert_prior scope")
    else:
        if (
            expert_derived
            or oracle_derived
            or not human_prior_used
            or learned_hpk
            or formal_eligible
            or acceptance_scope != "integration_only"
        ):
            _fail(
                path,
                "human_authored requires integration_only, human prior disclosure, "
                "learned_hpk=false, and formal_evaluation_eligible=false",
            )
    return value


class EntryV1(_StrictRecord):
    """静态快照中的不可变 HPK 知识条目。"""

    SCHEMA = ENTRY_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/entry/v1"
    TRANSFERABLE = True

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "schema",
                "entry_id",
                "status",
                "condition",
                "task_strategy",
                "geometric_strategy",
                "expected_effect",
                "effect_statistics",
                "evidence_refs",
                "promotion_policy_id",
                "evaluation_status",
                "evaluation_ref",
                "provenance",
                "acceptance_scope",
            ),
            path=path,
        )
        validate_content_id(
            payload["entry_id"], prefix="afkentry", path=f"{path}.entry_id"
        )
        status = _enum(payload["status"], allowed=ENTRY_STATUSES, path=f"{path}.status")
        condition = ConditionV1.from_dict(payload["condition"])
        task_strategy = TaskStrategyV1.from_dict(payload["task_strategy"])
        geometric_strategy = GeometricStrategyV1.from_dict(
            payload["geometric_strategy"]
        )
        expected_effect = AbstractEffectV1.from_dict(payload["expected_effect"])
        if expected_effect.to_dict() != expected_effect.expected_projection():
            _fail(
                f"{path}.expected_effect",
                "must be the expected projection without observed fields",
            )

        if condition["operation"] != task_strategy["operation"]:
            _fail(path, "condition and task strategy operation must match")
        if condition["manipulation_phase"] != task_strategy["manipulation_phase"]:
            _fail(path, "condition and task strategy phase must match")
        if condition["manipulated_object"]["role"] != task_strategy["manipulated_role"]:
            _fail(path, "condition and task strategy manipulated role must match")
        if condition["target"]["role"] != task_strategy["target_role"]:
            _fail(path, "condition and task strategy target role must match")
        if condition["target"]["relation"] != task_strategy["target_relation"]:
            _fail(path, "condition and task strategy target relation must match")
        if expected_effect["effect_type"] != task_strategy["operation"]:
            _fail(path, "expected effect type must match task operation")
        geometric_relation = geometric_strategy["target_relation"]
        if geometric_relation["relation"] != task_strategy["target_relation"]:
            _fail(path, "geometric and task target relation must match")
        if geometric_relation["reference_role"] != task_strategy["target_role"]:
            _fail(path, "geometric reference role must match task target role")

        statistics = _expect_fields(
            payload["effect_statistics"],
            required=(
                "support_count",
                "oppose_count",
                "unverified_count",
                "posterior_alpha",
                "posterior_beta",
                "estimated_success_probability",
                "lower_confidence_bound",
                "last_updated_at",
            ),
            path=f"{path}.effect_statistics",
        )
        counts = {
            name: _integer(
                statistics[name], path=f"{path}.effect_statistics.{name}", minimum=0
            )
            for name in ("support_count", "oppose_count", "unverified_count")
        }
        for name in ("posterior_alpha", "posterior_beta"):
            if (
                _number(
                    statistics[name],
                    path=f"{path}.effect_statistics.{name}",
                    minimum=0.0,
                )
                <= 0.0
            ):
                _fail(f"{path}.effect_statistics.{name}", "must be > 0")
        _number(
            statistics["estimated_success_probability"],
            path=f"{path}.effect_statistics.estimated_success_probability",
            minimum=0.0,
            maximum=1.0,
        )
        _number(
            statistics["lower_confidence_bound"],
            path=f"{path}.effect_statistics.lower_confidence_bound",
            minimum=0.0,
            maximum=1.0,
        )
        _utc(
            statistics["last_updated_at"],
            path=f"{path}.effect_statistics.last_updated_at",
        )

        refs = _expect_fields(
            payload["evidence_refs"],
            required=("supporting", "opposing", "unverified"),
            path=f"{path}.evidence_refs",
        )
        ref_values: dict[str, list[str]] = {}
        for name in ("supporting", "opposing", "unverified"):
            values = _string_list(refs[name], path=f"{path}.evidence_refs.{name}")
            for index, value in enumerate(values):
                validate_content_id(
                    value,
                    prefix="afkev",
                    path=f"{path}.evidence_refs.{name}[{index}]",
                )
            ref_values[name] = values
        if set(ref_values["supporting"]) & set(ref_values["opposing"]):
            _fail(f"{path}.evidence_refs", "supporting and opposing refs overlap")
        if set(ref_values["supporting"]) & set(ref_values["unverified"]):
            _fail(f"{path}.evidence_refs", "supporting and unverified refs overlap")
        if set(ref_values["opposing"]) & set(ref_values["unverified"]):
            _fail(f"{path}.evidence_refs", "opposing and unverified refs overlap")
        for count_name, ref_name in (
            ("support_count", "supporting"),
            ("oppose_count", "opposing"),
            ("unverified_count", "unverified"),
        ):
            if counts[count_name] != len(ref_values[ref_name]):
                _fail(
                    f"{path}.effect_statistics.{count_name}",
                    f"must equal evidence_refs.{ref_name} length",
                )

        promotion_policy_id = _string(
            payload["promotion_policy_id"], path=f"{path}.promotion_policy_id"
        )
        if _PROMOTION_POLICY_RE.fullmatch(promotion_policy_id) is None:
            _fail(
                f"{path}.promotion_policy_id",
                "must be a versioned hpk_<name>/v<N> identifier",
            )
        evaluation_status = _enum(
            payload["evaluation_status"],
            allowed=EVALUATION_STATUSES,
            path=f"{path}.evaluation_status",
        )
        evaluation_ref = payload["evaluation_ref"]
        if evaluation_ref is not None:
            validate_content_id(
                evaluation_ref, prefix="afkeval", path=f"{path}.evaluation_ref"
            )
        if evaluation_status == "passed" and evaluation_ref is None:
            _fail(f"{path}.evaluation_ref", "passed evaluation requires a reference")
        if evaluation_status != "passed" and evaluation_ref is not None:
            _fail(
                f"{path}.evaluation_ref",
                "only passed evaluation may carry a reference",
            )

        acceptance_scope = _enum(
            payload["acceptance_scope"],
            allowed=ACCEPTANCE_SCOPES,
            path=f"{path}.acceptance_scope",
        )
        provenance = _validate_entry_provenance(
            payload["provenance"],
            acceptance_scope=acceptance_scope,
            path=f"{path}.provenance",
        )
        if status == "accepted" and evaluation_status != "passed":
            _fail(path, "accepted entries require evaluation_status=passed")
        if status == "accepted" and provenance["source_kind"] != "human_authored":
            if counts["support_count"] < 1:
                _fail(path, "accepted learned entries require supporting evidence")
        if provenance["source_kind"] == "human_authored" and (
            any(counts.values()) or any(ref_values.values())
        ):
            _fail(path, "human-authored fixtures must not manufacture evidence")

        expected_entry_id = entry_id_for(payload)
        if payload["entry_id"] != expected_entry_id:
            _fail(
                f"{path}.entry_id",
                f"content identity mismatch; expected {expected_entry_id}",
            )

    @property
    def stable_id(self) -> str:
        return entry_id_for(self._data)

    @property
    def accepted(self) -> bool:
        return self._data["status"] == "accepted"


def snapshot_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    data.pop("snapshot_id", None)
    return data


def snapshot_id_for_manifest(payload: Mapping[str, Any]) -> str:
    return stable_content_id("afksnap", snapshot_identity_payload(payload))


class SnapshotManifestV1(_StrictRecord):
    """Strict wire manifest. Exact-byte manifest SHA is a loader property."""

    SCHEMA = SNAPSHOT_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/snapshot/v1"

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "schema",
                "schema_version",
                "snapshot_id",
                "parent_snapshot_id",
                "created_at",
                "purpose",
                "member",
                "accepted_entry_ids",
                "capabilities",
                "information_access_flags",
                "runtime_status",
                "immutable",
            ),
            path=path,
        )
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != 1
        ):
            _fail(f"{path}.schema_version", "must equal 1")
        validate_content_id(
            payload["snapshot_id"], prefix="afksnap", path=f"{path}.snapshot_id"
        )
        parent = payload["parent_snapshot_id"]
        if parent is not None:
            validate_content_id(
                parent, prefix="afksnap", path=f"{path}.parent_snapshot_id"
            )
            if parent == payload["snapshot_id"]:
                _fail(f"{path}.parent_snapshot_id", "must not equal snapshot_id")
        _utc(payload["created_at"], path=f"{path}.created_at")
        _enum(payload["purpose"], allowed=SNAPSHOT_PURPOSES, path=f"{path}.purpose")
        member = _expect_fields(
            payload["member"],
            required=("path", "media_type", "sha256", "size_bytes", "record_count"),
            path=f"{path}.member",
        )
        if member["path"] != "entries.jsonl":
            _fail(f"{path}.member.path", "must equal 'entries.jsonl'")
        if member["media_type"] != "application/x-ndjson":
            _fail(f"{path}.member.media_type", "must equal 'application/x-ndjson'")
        _sha256(member["sha256"], path=f"{path}.member.sha256")
        _integer(member["size_bytes"], path=f"{path}.member.size_bytes", minimum=1)
        _integer(member["record_count"], path=f"{path}.member.record_count", minimum=1)
        accepted_ids = _sorted_string_list(
            payload["accepted_entry_ids"],
            path=f"{path}.accepted_entry_ids",
            nonempty=True,
        )
        for index, entry_id in enumerate(accepted_ids):
            validate_content_id(
                entry_id,
                prefix="afkentry",
                path=f"{path}.accepted_entry_ids[{index}]",
            )
        if len(accepted_ids) != member["record_count"]:
            _fail(
                f"{path}.accepted_entry_ids",
                "length must equal member.record_count",
            )
        capabilities = _expect_fields(
            payload["capabilities"],
            required=(
                "task_families",
                "domain_ids",
                "operations",
                "strategy_families",
                "target_relations",
                "hard_constraints",
                "geometry_source_classes",
                "semantic_part_observed",
            ),
            path=f"{path}.capabilities",
        )
        _sorted_string_list(
            capabilities["task_families"],
            path=f"{path}.capabilities.task_families",
            nonempty=True,
        )
        _sorted_string_list(
            capabilities["domain_ids"],
            path=f"{path}.capabilities.domain_ids",
            nonempty=True,
        )
        for key, allowed in (
            ("operations", OPERATIONS),
            ("strategy_families", STRATEGY_FAMILIES),
            ("target_relations", TARGET_RELATIONS),
            ("hard_constraints", HARD_CONSTRAINTS),
            ("geometry_source_classes", GEOMETRY_SOURCE_CLASSES),
        ):
            _sorted_string_list(
                capabilities[key], path=f"{path}.capabilities.{key}", allowed=allowed
            )
        _boolean(
            capabilities["semantic_part_observed"],
            path=f"{path}.capabilities.semantic_part_observed",
        )
        flags = _expect_fields(
            normalize_knowledge_metadata(payload["information_access_flags"]),
            required=(
                "expert_prior_present",
                "human_integration_prior_present",
                "oracle_derived_present",
                "learned_hpk_present",
                "all_entries_formal_evaluation_eligible",
            ),
            path=f"{path}.information_access_flags",
        )
        for key in flags:
            _boolean(flags[key], path=f"{path}.information_access_flags.{key}")
        if payload["runtime_status"] != "static_ready":
            _fail(f"{path}.runtime_status", "must equal 'static_ready'")
        if payload["immutable"] is not True:
            _fail(f"{path}.immutable", "must be the boolean true")
        expected_id = snapshot_id_for_manifest(payload)
        if payload["snapshot_id"] != expected_id:
            _fail(
                f"{path}.snapshot_id",
                f"content identity mismatch; expected {expected_id}",
            )

    @property
    def stable_id(self) -> str:
        return snapshot_id_for_manifest(self._data)


# The contract uses SnapshotV1 for the validated logical wire projection.  The
# 原始字节摘要和路径保存在 store.LoadedHPKSnapshot 中。
SnapshotV1 = SnapshotManifestV1


def _validate_source_disclosure(value: Any, *, path: str) -> dict[str, Any]:
    disclosure = _expect_fields(
        normalize_knowledge_metadata(value),
        required=(
            "source_kind",
            "acceptance_scope",
            "expert_prior_used",
            "human_prior_used",
            "oracle_evidence_used",
            "learned_hpk",
            "formal_evaluation_eligible",
        ),
        path=path,
    )
    _enum(disclosure["source_kind"], allowed=SOURCE_KINDS, path=f"{path}.source_kind")
    _enum(
        disclosure["acceptance_scope"],
        allowed=ACCEPTANCE_SCOPES,
        path=f"{path}.acceptance_scope",
    )
    for key in (
        "expert_prior_used",
        "human_prior_used",
        "oracle_evidence_used",
        "learned_hpk",
        "formal_evaluation_eligible",
    ):
        _boolean(disclosure[key], path=f"{path}.{key}")
    return disclosure


class ContextV1(_StrictRecord):
    """供规划器读取的 HPK 内容，不包含整个快照。"""

    SCHEMA = CONTEXT_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/context/v1"
    TRANSFERABLE = True

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "schema",
                "entry_id",
                "condition_summary",
                "task_guidance",
                "expected_effect",
                "confidence",
                "source_disclosure",
            ),
            path=path,
        )
        validate_content_id(
            payload["entry_id"], prefix="afkentry", path=f"{path}.entry_id"
        )
        summary = _expect_fields(
            payload["condition_summary"],
            required=(
                "task_family",
                "manipulation_phase",
                "operation",
                "manipulated_role",
                "held_state",
                "target_role",
                "target_relation",
                "scene_predicates",
                "preconditions",
            ),
            path=f"{path}.condition_summary",
        )
        for key in ("task_family", "manipulation_phase", "manipulated_role"):
            _string(summary[key], path=f"{path}.condition_summary.{key}")
        _enum(
            summary["operation"],
            allowed=OPERATIONS,
            path=f"{path}.condition_summary.operation",
        )
        _enum(
            summary["held_state"],
            allowed=HELD_STATES,
            path=f"{path}.condition_summary.held_state",
        )
        _nullable_string(
            summary["target_role"], path=f"{path}.condition_summary.target_role"
        )
        _nullable_enum(
            summary["target_relation"],
            allowed=TARGET_RELATIONS,
            path=f"{path}.condition_summary.target_relation",
        )
        _string_list(
            summary["scene_predicates"],
            path=f"{path}.condition_summary.scene_predicates",
            allowed=SCENE_PREDICATES,
        )
        _string_list(
            summary["preconditions"], path=f"{path}.condition_summary.preconditions"
        )
        guidance = _expect_fields(
            payload["task_guidance"],
            required=(
                "operation",
                "manipulated_role",
                "target_role",
                "target_relation",
                "manipulation_phase",
                "subgoal_purpose",
                "preferred_arm",
            ),
            path=f"{path}.task_guidance",
        )
        _enum(
            guidance["operation"],
            allowed=OPERATIONS,
            path=f"{path}.task_guidance.operation",
        )
        for key in ("manipulated_role", "manipulation_phase", "subgoal_purpose"):
            _string(guidance[key], path=f"{path}.task_guidance.{key}")
        _nullable_string(
            guidance["target_role"], path=f"{path}.task_guidance.target_role"
        )
        _nullable_enum(
            guidance["target_relation"],
            allowed=TARGET_RELATIONS,
            path=f"{path}.task_guidance.target_relation",
        )
        _enum(
            guidance["preferred_arm"],
            allowed=ARMS,
            path=f"{path}.task_guidance.preferred_arm",
        )
        effect = _expect_fields(
            payload["expected_effect"],
            required=("effect_type", "predicates"),
            path=f"{path}.expected_effect",
        )
        _enum(
            effect["effect_type"],
            allowed=EFFECT_TYPES,
            path=f"{path}.expected_effect.effect_type",
        )
        _string_list(
            effect["predicates"],
            path=f"{path}.expected_effect.predicates",
            allowed=EFFECT_PREDICATES,
        )
        confidence = _number(
            payload["confidence"], path=f"{path}.confidence", minimum=0.0, maximum=1.0
        )
        disclosure = _validate_source_disclosure(
            payload["source_disclosure"], path=f"{path}.source_disclosure"
        )
        if disclosure["source_kind"] == "human_authored" and confidence != 0.0:
            _fail(f"{path}.confidence", "human-authored context confidence must be 0.0")


def context_sha256(value: ContextV1 | Mapping[str, Any]) -> str:
    payload = value.to_dict() if isinstance(value, ContextV1) else dict(value)
    validated = ContextV1.from_dict(payload)
    return hashlib.sha256(canonical_json_bytes(validated.to_dict())).hexdigest()


def audit_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    data.pop("usage_audit_id", None)
    return data


def audit_id_for(payload: Mapping[str, Any]) -> str:
    return stable_content_id("afkaudit", audit_identity_payload(payload))


class AuditV1(_StrictRecord):
    """Strict public P0-C usage audit with no candidate-private identity."""

    SCHEMA = AUDIT_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/audit/v1"
    TRANSFERABLE = True

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        super()._validate(payload)
        path = cls.__name__
        _expect_fields(
            payload,
            required=(
                "schema",
                "usage_audit_id",
                "mode",
                "retrieval_stage",
                "snapshot_id",
                "snapshot_manifest_sha256",
                "retrieved_entry_ids",
                "selected_entry_id",
                "condition_id",
                "task_strategy_id",
                "selected_geometric_strategy_id",
                "match_reason",
                "rejection_reasons",
                "planner_context_sha256",
                "planner_context_rendered",
                "planner_usage",
                "behavior_changed",
                "behavior_change_channel",
                "geometric_compliance",
                "scoring_profile",
                "source_disclosure",
            ),
            path=path,
        )
        validate_content_id(
            payload["usage_audit_id"], prefix="afkaudit", path=f"{path}.usage_audit_id"
        )
        if payload["mode"] != "static":
            _fail(f"{path}.mode", "must equal 'static'")
        stage = _enum(
            payload["retrieval_stage"],
            allowed=AUDIT_RETRIEVAL_STAGES,
            path=f"{path}.retrieval_stage",
        )
        validate_content_id(
            payload["snapshot_id"], prefix="afksnap", path=f"{path}.snapshot_id"
        )
        _sha256(
            payload["snapshot_manifest_sha256"], path=f"{path}.snapshot_manifest_sha256"
        )
        retrieved = _string_list(
            payload["retrieved_entry_ids"], path=f"{path}.retrieved_entry_ids"
        )
        if len(retrieved) > 1:
            _fail(f"{path}.retrieved_entry_ids", "P0-C permits zero or one entry")
        for index, entry_id in enumerate(retrieved):
            validate_content_id(
                entry_id, prefix="afkentry", path=f"{path}.retrieved_entry_ids[{index}]"
            )
        selected = payload["selected_entry_id"]
        if selected is not None:
            validate_content_id(
                selected, prefix="afkentry", path=f"{path}.selected_entry_id"
            )
            if retrieved != [selected]:
                _fail(f"{path}.selected_entry_id", "must be the sole retrieved entry")
        elif retrieved:
            _fail(f"{path}.selected_entry_id", "must select the retrieved entry")
        for key, prefix in (
            ("condition_id", "afkc"),
            ("task_strategy_id", "afku"),
            ("selected_geometric_strategy_id", "afkz"),
        ):
            if payload[key] is not None:
                validate_content_id(payload[key], prefix=prefix, path=f"{path}.{key}")
        if selected is None and payload["selected_geometric_strategy_id"] is not None:
            _fail(f"{path}.selected_geometric_strategy_id", "requires a selected entry")
        match_reason = payload["match_reason"]
        if match_reason not in {None, "exact_match"}:
            _fail(f"{path}.match_reason", "must be exact_match or null")
        if (selected is None) != (match_reason is None):
            _fail(
                f"{path}.match_reason", "must be exact_match iff an entry is selected"
            )
        reasons = _string_list(
            payload["rejection_reasons"],
            path=f"{path}.rejection_reasons",
            allowed=AUDIT_REJECTION_REASONS,
        )
        expected_reason_order = [
            value for value in AUDIT_REJECTION_REASON_ORDER if value in reasons
        ]
        if reasons != expected_reason_order:
            _fail(f"{path}.rejection_reasons", "must use frozen enum order")
        planner_hash = payload["planner_context_sha256"]
        if planner_hash is not None:
            _sha256(planner_hash, path=f"{path}.planner_context_sha256")
        rendered = _boolean(
            payload["planner_context_rendered"], path=f"{path}.planner_context_rendered"
        )
        planner_usage = _enum(
            payload["planner_usage"],
            allowed=frozenset({"not_injected", "injected_use_unverified"}),
            path=f"{path}.planner_usage",
        )
        if rendered != (planner_hash is not None) or rendered != (
            planner_usage == "injected_use_unverified"
        ):
            _fail(path, "planner hash, rendered flag, and usage receipt must agree")
        changed = payload["behavior_changed"]
        if not isinstance(changed, bool) and changed != "unverified":
            _fail(f"{path}.behavior_changed", "must be a boolean or 'unverified'")
        channel = _enum(
            payload["behavior_change_channel"],
            allowed=frozenset({"none", "planner", "geometry", "both", "unverified"}),
            path=f"{path}.behavior_change_channel",
        )
        if changed is False and channel != "none":
            _fail(
                f"{path}.behavior_change_channel",
                "must be none when behavior did not change",
            )
        if changed == "unverified" and channel != "unverified":
            _fail(
                f"{path}.behavior_change_channel",
                "must be unverified with unverified behavior",
            )
        compliance = payload["geometric_compliance"]
        if not isinstance(compliance, bool) and compliance != "unverified":
            _fail(f"{path}.geometric_compliance", "must be a boolean or 'unverified'")
        score_profile = payload["scoring_profile"]
        if score_profile not in {None, HPK_GEOMETRY_SCORE_PROFILE, "afk_geometry_score/v1"}:
            _fail(
                f"{path}.scoring_profile",
                f"must be {HPK_GEOMETRY_SCORE_PROFILE!r} or null",
            )
        if (
            stage == "post_binding_geometry"
            and payload["selected_geometric_strategy_id"] is not None
            and score_profile is None
        ):
            _fail(
                f"{path}.scoring_profile",
                "active geometry requires the frozen score profile",
            )
        disclosure = payload["source_disclosure"]
        if disclosure is not None:
            _validate_source_disclosure(disclosure, path=f"{path}.source_disclosure")
        if selected is None and disclosure is not None:
            _fail(f"{path}.source_disclosure", "requires a selected entry")
        if selected is not None and disclosure is None:
            _fail(f"{path}.source_disclosure", "selected entry requires disclosure")
        expected_id = audit_id_for(payload)
        if payload["usage_audit_id"] != expected_id:
            _fail(
                f"{path}.usage_audit_id",
                f"content identity mismatch; expected {expected_id}",
            )

    @property
    def stable_id(self) -> str:
        return audit_id_for(self._data)


HPKConditionV1 = ConditionV1
HPKTaskStrategyV1 = TaskStrategyV1
HPKGeometricStrategyV1 = GeometricStrategyV1
HPKAbstractEffectV1 = AbstractEffectV1
HPKEvidenceV1 = EvidenceV1
HPKEntryV1 = EntryV1
HPKSnapshotV1 = SnapshotManifestV1
HPKContextV1 = ContextV1
HPKAuditV1 = AuditV1


__all__ = [
    "ABSTRACT_EFFECT_SCHEMA",
    "ACCEPTANCE_SCOPES",
    "HPK_GEOMETRY_SCORE_PROFILE",
    "HPKAbstractEffectV1",
    "HPKAuditV1",
    "HPKConditionV1",
    "HPKContextV1",
    "HPKEntryV1",
    "HPKEvidenceV1",
    "HPKGeometricStrategyV1",
    "HPKSnapshotV1",
    "HPKTaskStrategyV1",
    "HPKUnresolved",
    "HPKValidationError",
    "APPROACH_FAMILIES",
    "ARMS",
    "AUDIT_REJECTION_REASON_ORDER",
    "AUDIT_REJECTION_REASONS",
    "AUDIT_RETRIEVAL_STAGES",
    "AUDIT_SCHEMA",
    "AbstractEffectV1",
    "AuditV1",
    "CONDITION_ABSTRACTION_VERSION",
    "CONDITION_SCHEMA",
    "CONTEXT_SCHEMA",
    "CandidateGeometryFeaturesV1",
    "ConditionV1",
    "ContextV1",
    "DIRECTION_BUCKETS",
    "EFFECT_PREDICATES",
    "EFFECT_TYPES",
    "EVIDENCE_SCHEMA",
    "EVIDENCE_VERDICTS",
    "ENTRY_SCHEMA",
    "ENTRY_STATUSES",
    "EVALUATION_STATUSES",
    "EntryV1",
    "EvidenceV1",
    "GEOMETRIC_STRATEGY_SCHEMA",
    "GEOMETRY_SOURCE_CLASSES",
    "GeometricStrategyV1",
    "MOTION_STATUSES",
    "OBSERVE_AUDIT_PROFILE",
    "ObserveAuditRecord",
    "OPERATIONS",
    "ORIENTATION_RELATIONS",
    "REALIZATION_STATUSES",
    "SNAPSHOT_PURPOSES",
    "SNAPSHOT_SCHEMA",
    "SOURCE_KINDS",
    "SnapshotManifestV1",
    "SnapshotV1",
    "TASK_STRATEGY_NORMALIZATION_VERSION",
    "TASK_STRATEGY_SCHEMA",
    "TaskStrategyV1",
    "VERIFIABILITY_VALUES",
    "VERIFIER_SOURCES",
    "canonical_json",
    "canonical_json_bytes",
    "contains_private_transfer_text",
    "context_sha256",
    "audit_id_for",
    "audit_identity_payload",
    "entry_id_for",
    "entry_identity_payload",
    "evidence_id_for",
    "evidence_identity_payload",
    "reject_private_transferable",
    "stable_content_id",
    "snapshot_id_for_manifest",
    "snapshot_identity_payload",
    "validate_content_id",
]

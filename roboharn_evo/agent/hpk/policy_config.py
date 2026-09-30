from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Literal, Mapping

import yaml

from roboharn_evo.agent.hpk.schemas import canonical_json_bytes
from roboharn_evo.resources import config_path


PolicyKind = Literal["geometry", "promotion"]
DEFAULT_GEOMETRY_POLICY_RAW_SHA256 = (
    "71eb071b7399071b878c839a10f111351b740273f51d875cfd146b5573e0d5a1"
)
EVOLVING_GEOMETRY_POLICY_RAW_SHA256 = (
    "4971bc8197cf8b6105fe28fb2b82bebea2de52683c75aead656ffd394a5a2dbb"
)
SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256 = (
    "f7339e65a86360c7b9c9afcc59d5f9b16e0d778911832ef544fe9def69b36920"
)
EVOLVING_GEOMETRY_POLICY_CONFIG_SHA256 = (
    "2947b57cca7a8ffbb42ec032c34c72d9fad6e16101457aee97050170faeb1ac6"
)
SAFE_EXPLORATION_GEOMETRY_POLICY_CONFIG_SHA256 = (
    "5f71aea9c92c08d08b44127741280f41a2888fde5fbb66d38e01aea6e46eecf2"
)
SAFE_EXPLORATION_GEOMETRY_POLICY_IDENTITIES = frozenset({
    ("hpk_geometry_policy/v3", SAFE_EXPLORATION_GEOMETRY_POLICY_CONFIG_SHA256),
    ("afk_geometry_policy/v3", "f1e30464cb9f54606f8f4439fecdc9afe689a48b68438cd7c894fdbc8a4b90f8"),
})
EVOLVING_GEOMETRY_POLICY_IDENTITIES = frozenset(
    {
        (
            "hpk_geometry_policy/v2",
            EVOLVING_GEOMETRY_POLICY_CONFIG_SHA256,
        ),
        (
            "afk_geometry_policy/v2",
            "555bf479115c05e0171bc983b5be5de8d20b8af99705f25562370eb46fdfd458",
        ),
        *SAFE_EXPLORATION_GEOMETRY_POLICY_IDENTITIES,
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POLICY_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*/v[1-9][0-9]*$")
_MAX_POLICY_BYTES = 256 * 1024
_FORBIDDEN_OVERRIDE_TOKENS = frozenset(
    {"task", "seed", "arm", "object", "candidate", "track", "coordinate", "override"}
)


class HPKPolicyConfigError(ValueError):
    """HPK 策略配置不符合格式约定。"""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise HPKPolicyConfigError("policy YAML keys must be strings")
        if key in result:
            raise HPKPolicyConfigError(f"duplicate policy YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _fail(message: str) -> None:
    raise HPKPolicyConfigError(message)


def _exact(mapping: Mapping[str, Any], keys: set[str], *, path: str) -> None:
    actual = set(mapping)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        _fail(f"{path} fields mismatch: missing={missing}, extra={extra}")


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    return dict(value)


def _bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{path} must be a boolean")
    return value


def _number(
    value: Any,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{path} must be finite")
    if minimum is not None and result < minimum:
        _fail(f"{path} must be >= {minimum}")
    if maximum is not None and result > maximum:
        _fail(f"{path} must be <= {maximum}")
    return result


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{path} must be an integer >= {minimum}")
    return value


def _string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{path} must be a non-empty string")
    return value.strip()


def _reject_override_keys(value: Any, *, path: str = "policy") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in _FORBIDDEN_OVERRIDE_TOKENS
                or normalized.endswith("_override")
                or normalized.endswith("_overrides")
            ):
                _fail(f"{path}.{key} is a forbidden per-instance override key")
            _reject_override_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_override_keys(item, path=f"{path}[{index}]")


def _validate_policy_id(payload: Mapping[str, Any], *, path: str) -> str:
    value = _string(payload.get("policy_id"), path=f"{path}.policy_id")
    if _POLICY_ID_RE.fullmatch(value) is None:
        _fail(f"{path}.policy_id must be a versioned global policy ID")
    return value


def _validate_geometry(payload: dict[str, Any]) -> None:
    base_fields = {
        "schema",
        "policy_id",
        "scoring_profile",
        "grasp_width_buckets_m",
        "reach_distance_buckets_m",
        "direction_classification",
        "soft_score",
        "all_hard_mismatch_behavior",
    }
    schema = payload.get("schema")
    if schema in {"roboharn_evo/hpk/geometry_policy_config/v1", "tcm/afk/geometry_policy_config/v1"}:
        expected_fields = base_fields
        expected_policy_id = "hpk_geometry_policy/v1"
    elif schema in {"roboharn_evo/hpk/geometry_policy_config/v2", "tcm/afk/geometry_policy_config/v2"}:
        expected_fields = base_fields | {"semantic_attempt"}
        expected_policy_id = "hpk_geometry_policy/v2"
    elif schema in {"roboharn_evo/hpk/geometry_policy_config/v3", "tcm/afk/geometry_policy_config/v3"}:
        expected_fields = base_fields | {"semantic_attempt", "safe_exploration"}
        expected_policy_id = "hpk_geometry_policy/v3"
    else:
        _fail("unsupported geometry policy schema")
    _exact(
        payload,
        expected_fields,
        path="geometry_policy",
    )
    if _validate_policy_id(payload, path="geometry_policy") not in {
        expected_policy_id, "afk_" + expected_policy_id.removeprefix("hpk_")
    }:
        _fail(f"geometry policy_id must equal {expected_policy_id}")
    if payload["scoring_profile"] not in {"hpk_geometry_score/v1", "afk_geometry_score/v1"}:
        _fail("geometry scoring_profile must equal hpk_geometry_score/v1")
    if payload["all_hard_mismatch_behavior"] not in {
        "fail_closed",
        "baseline_fallback",
    }:
        _fail(
            "geometry all_hard_mismatch_behavior must be fail_closed or "
            "baseline_fallback"
        )

    if schema in {
        "roboharn_evo/hpk/geometry_policy_config/v2",
        "roboharn_evo/hpk/geometry_policy_config/v3",
        "tcm/afk/geometry_policy_config/v2",
        "tcm/afk/geometry_policy_config/v3",
    }:
        semantic_attempt = _mapping(
            payload["semantic_attempt"], path="semantic_attempt"
        )
        _exact(
            semantic_attempt,
            {"max_pending_env_step_gap", "max_segments"},
            path="semantic_attempt",
        )
        _integer(
            semantic_attempt["max_pending_env_step_gap"],
            path="semantic_attempt.max_pending_env_step_gap",
            minimum=1,
        )
        _integer(
            semantic_attempt["max_segments"],
            path="semantic_attempt.max_segments",
            minimum=1,
        )
    if schema in {"roboharn_evo/hpk/geometry_policy_config/v3", "tcm/afk/geometry_policy_config/v3"}:
        exploration = _mapping(payload["safe_exploration"], path="safe_exploration")
        _exact(
            exploration,
            {
                "mode",
                "activate_when",
                "min_eligible",
                "rank_offset",
                "max_selections_per_condition_per_episode",
                "honor_requested",
                "exclude_blocked",
                "require_guard_legal",
                "disable_after_accepted_exact_match",
            },
            path="safe_exploration",
        )
        for key in ("mode", "activate_when"):
            _string(exploration[key], path=f"safe_exploration.{key}")
        for key in (
            "min_eligible",
            "rank_offset",
            "max_selections_per_condition_per_episode",
        ):
            _integer(exploration[key], path=f"safe_exploration.{key}")
        for key in (
            "honor_requested",
            "exclude_blocked",
            "require_guard_legal",
            "disable_after_accepted_exact_match",
        ):
            _bool(exploration[key], path=f"safe_exploration.{key}")
        if exploration["mode"] != "alternate_safe_baseline_rank_v1":
            _fail("safe_exploration.mode is unsupported")
        if exploration["activate_when"] != "no_accepted_exact_match":
            _fail("safe_exploration.activate_when is unsupported")
        if not 0 < exploration["rank_offset"] < exploration["min_eligible"]:
            _fail("safe_exploration.rank_offset must be below min_eligible")
        if exploration["max_selections_per_condition_per_episode"] <= 0:
            _fail("safe_exploration episode selection limit must be positive")

    grasp = _mapping(payload["grasp_width_buckets_m"], path="grasp_width_buckets_m")
    _exact(grasp, {"narrow_max", "medium_max"}, path="grasp_width_buckets_m")
    narrow = _number(
        grasp["narrow_max"], path="grasp_width_buckets_m.narrow_max", minimum=0
    )
    medium = _number(
        grasp["medium_max"], path="grasp_width_buckets_m.medium_max", minimum=0
    )
    if not narrow < medium:
        _fail("grasp width thresholds must be strictly increasing")

    reach = _mapping(
        payload["reach_distance_buckets_m"], path="reach_distance_buckets_m"
    )
    _exact(reach, {"near_max", "medium_max"}, path="reach_distance_buckets_m")
    near = _number(
        reach["near_max"], path="reach_distance_buckets_m.near_max", minimum=0
    )
    reach_medium = _number(
        reach["medium_max"], path="reach_distance_buckets_m.medium_max", minimum=0
    )
    if not near < reach_medium:
        _fail("reach thresholds must be strictly increasing")

    direction = _mapping(
        payload["direction_classification"], path="direction_classification"
    )
    _exact(
        direction,
        {
            "epsilon",
            "vertical_to_horizontal_ratio",
            "horizontal_to_vertical_ratio",
            "fallback_above_vertical_ratio_min",
            "fallback_below_vertical_ratio_max",
            "fallback_lateral_vertical_ratio_abs_max",
        },
        path="direction_classification",
    )
    _number(direction["epsilon"], path="direction_classification.epsilon", minimum=0)
    _number(
        direction["vertical_to_horizontal_ratio"],
        path="direction_classification.vertical_to_horizontal_ratio",
        minimum=0,
    )
    _number(
        direction["horizontal_to_vertical_ratio"],
        path="direction_classification.horizontal_to_vertical_ratio",
        minimum=0,
    )
    above = _number(
        direction["fallback_above_vertical_ratio_min"],
        path="direction_classification.fallback_above_vertical_ratio_min",
        minimum=-1,
        maximum=1,
    )
    below = _number(
        direction["fallback_below_vertical_ratio_max"],
        path="direction_classification.fallback_below_vertical_ratio_max",
        minimum=-1,
        maximum=1,
    )
    lateral = _number(
        direction["fallback_lateral_vertical_ratio_abs_max"],
        path="direction_classification.fallback_lateral_vertical_ratio_abs_max",
        minimum=0,
        maximum=1,
    )
    if below >= -lateral or above <= lateral:
        _fail("fallback direction thresholds overlap")

    soft = _mapping(payload["soft_score"], path="soft_score")
    _exact(soft, {"match_weights", "reach_score", "avoid_penalties"}, path="soft_score")
    weights = _mapping(soft["match_weights"], path="soft_score.match_weights")
    expected_weights = {
        "strategy_family_match",
        "reference_frame_match",
        "target_relation_match",
        "target_reference_role_match",
        "approach_family_match",
        "approach_direction_match",
        "orientation_match",
        "grasp_region_match",
        "grasp_semantic_part_match",
        "geometry_source_class_match",
    }
    _exact(weights, expected_weights, path="soft_score.match_weights")
    for key, value in weights.items():
        _number(value, path=f"soft_score.match_weights.{key}", minimum=0)
    reach_score = _mapping(soft["reach_score"], path="soft_score.reach_score")
    _exact(
        reach_score,
        {
            "formula_id",
            "numerator",
            "denominator_offset",
            "round_digits",
            "bucket_scores",
        },
        path="soft_score.reach_score",
    )
    if reach_score["formula_id"] != "inverse_distance_v1":
        _fail("unsupported reach score formula")
    _number(
        reach_score["numerator"], path="soft_score.reach_score.numerator", minimum=0
    )
    _number(
        reach_score["denominator_offset"],
        path="soft_score.reach_score.denominator_offset",
        minimum=0,
    )
    _integer(reach_score["round_digits"], path="soft_score.reach_score.round_digits")
    buckets = _mapping(
        reach_score["bucket_scores"], path="soft_score.reach_score.bucket_scores"
    )
    _exact(
        buckets,
        {"near", "medium", "far", "unknown"},
        path="soft_score.reach_score.bucket_scores",
    )
    for key, value in buckets.items():
        _number(value, path=f"soft_score.reach_score.bucket_scores.{key}")
    penalties = _mapping(soft["avoid_penalties"], path="soft_score.avoid_penalties")
    _exact(
        penalties,
        {"repeat_equivalent_failed_candidate"},
        path="soft_score.avoid_penalties",
    )
    if (
        _number(
            penalties["repeat_equivalent_failed_candidate"],
            path="soft_score.avoid_penalties.repeat_equivalent_failed_candidate",
        )
        > 0
    ):
        _fail("avoid penalties must be non-positive")


_PROMOTION_KEYS = {
    "schema",
    "policy_id",
    "development_only",
    "formal_evaluation_eligible",
    "prior_alpha",
    "prior_beta",
    "lcb_method",
    "lcb_delta",
    "min_distinct_support_episodes",
    "min_support_count",
    "max_oppose_count",
    "accept_lcb_threshold",
    "demote_lcb_threshold",
    "min_distinct_scene_signatures",
    "min_oppose_count_for_revalidation",
    "min_distinct_oppose_episodes_for_revalidation",
    "min_oppose_count_for_deprecation",
    "min_distinct_oppose_episodes_for_deprecation",
    "allow_expert_prior",
    "allow_oracle_evidence",
}


def _validate_promotion(payload: dict[str, Any]) -> None:
    _exact(payload, _PROMOTION_KEYS, path="promotion_policy")
    if payload["schema"] not in {"roboharn_evo/hpk/promotion_policy/v1", "tcm/afk/promotion_policy/v1"}:
        _fail("unsupported promotion policy schema")
    policy_id = _validate_policy_id(payload, path="promotion_policy")
    development = _bool(payload["development_only"], path="development_only")
    formal = _bool(
        payload["formal_evaluation_eligible"], path="formal_evaluation_eligible"
    )
    if development == formal:
        _fail("development_only and formal_evaluation_eligible must be opposites")
    expected_id = (
        "hpk_promotion_integration_dev/v1" if development else "hpk_promotion_formal/v1"
    )
    if policy_id not in {expected_id, "afk_" + expected_id.removeprefix("hpk_")}:
        _fail(f"promotion policy_id must equal {expected_id}")
    if payload["lcb_method"] != "hoeffding_v1":
        _fail("lcb_method must equal hoeffding_v1")
    for key in ("prior_alpha", "prior_beta"):
        _number(payload[key], path=key, minimum=0)
        if float(payload[key]) <= 0:
            _fail(f"{key} must be > 0")
    _number(payload["lcb_delta"], path="lcb_delta", minimum=0, maximum=1)
    if float(payload["lcb_delta"]) in {0.0, 1.0}:
        _fail("lcb_delta must be strictly between 0 and 1")
    integer_keys = (
        "min_distinct_support_episodes",
        "min_support_count",
        "max_oppose_count",
        "min_distinct_scene_signatures",
        "min_oppose_count_for_revalidation",
        "min_distinct_oppose_episodes_for_revalidation",
        "min_oppose_count_for_deprecation",
        "min_distinct_oppose_episodes_for_deprecation",
    )
    for key in integer_keys:
        _integer(payload[key], path=key)
    for key in ("accept_lcb_threshold", "demote_lcb_threshold"):
        _number(payload[key], path=key, minimum=0, maximum=1)
    for key in ("allow_expert_prior", "allow_oracle_evidence"):
        if _bool(payload[key], path=key):
            _fail(f"{key} must be false in P0-D/E policies")
    if not development:
        if payload["min_distinct_support_episodes"] < 2:
            _fail("formal promotion requires multiple support episodes")
        if payload["min_distinct_scene_signatures"] < 2:
            _fail("formal promotion requires multiple scene signatures")
        if payload["min_oppose_count_for_revalidation"] < 2:
            _fail("formal revalidation cannot be triggered by one opposition")
        if payload["min_distinct_oppose_episodes_for_revalidation"] < 2:
            _fail("formal revalidation requires multiple opposing episodes")
    if (
        payload["min_oppose_count_for_deprecation"]
        < payload["min_oppose_count_for_revalidation"]
    ):
        _fail("deprecation opposition count must not be below revalidation")
    if (
        payload["min_distinct_oppose_episodes_for_deprecation"]
        < payload["min_distinct_oppose_episodes_for_revalidation"]
    ):
        _fail("deprecation episode count must not be below revalidation")


def _read_policy(path: Path) -> bytes:
    if not path.is_absolute():
        _fail("policy path must be explicit and absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HPKPolicyConfigError(
            f"policy file does not exist: {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("policy path must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if before.st_size > _MAX_POLICY_BYTES:
            _fail("policy file is too large")
        data = b""
        while len(data) <= _MAX_POLICY_BYTES:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            data += chunk
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail("policy file changed while being read")
        if len(data) != after.st_size or len(data) > _MAX_POLICY_BYTES:
            _fail("policy file size is invalid")
        return data
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class LoadedHPKPolicy:
    path: Path
    kind: PolicyKind
    policy_id: str
    sha256: str
    config_sha256: str
    _canonical_payload: bytes

    @property
    def id(self) -> str:
        return self.policy_id

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._canonical_payload.decode("utf-8"))

    def identity(self) -> dict[str, str]:
        return {"policy_id": self.policy_id, "config_sha256": self.config_sha256}


def load_policy_config(
    path: str | os.PathLike[str],
    expected_sha256: str,
    kind: PolicyKind,
) -> LoadedHPKPolicy:
    """Load one explicit raw-byte-pinned policy without override discovery."""

    if kind not in {"geometry", "promotion"}:
        _fail("kind must be 'geometry' or 'promotion'")
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        _fail("expected_sha256 must be a lowercase SHA-256 digest")
    raw_path = Path(path)
    data = _read_policy(raw_path)
    raw_sha = hashlib.sha256(data).hexdigest()
    if raw_sha != expected_sha256:
        _fail(f"policy SHA-256 mismatch: expected {expected_sha256}, got {raw_sha}")
    try:
        decoded = data.decode("utf-8")
        payload = yaml.load(decoded, Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise HPKPolicyConfigError(f"invalid policy YAML: {exc}") from exc
    payload = _mapping(payload, path="policy")
    _reject_override_keys(payload)
    if kind == "geometry":
        _validate_geometry(payload)
    else:
        _validate_promotion(payload)
    canonical = canonical_json_bytes(payload)
    return LoadedHPKPolicy(
        path=raw_path.resolve(strict=True),
        kind=kind,
        policy_id=str(payload["policy_id"]),
        sha256=raw_sha,
        config_sha256=hashlib.sha256(canonical).hexdigest(),
        _canonical_payload=canonical,
    )


def validate_geometry_policy_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached canonical geometry-policy mapping."""

    payload = _mapping(value, path="geometry_policy")
    _reject_override_keys(payload)
    _validate_geometry(payload)
    return json.loads(canonical_json_bytes(payload).decode("utf-8"))


def default_geometry_policy_path() -> Path:
    """Return the byte-frozen P0-C/static v1 policy, never an override."""

    return config_path("hpk_geometry_policy_v1.yaml")


@lru_cache(maxsize=1)
def load_default_geometry_policy() -> LoadedHPKPolicy:
    """Load the version-pinned P0-C/static v1 geometry policy."""

    return load_policy_config(
        default_geometry_policy_path(),
        DEFAULT_GEOMETRY_POLICY_RAW_SHA256,
        "geometry",
    )


def evolving_geometry_policy_path() -> Path:
    """Return the repository-owned SnapshotV2/evolving geometry policy."""

    return config_path("hpk_geometry_policy_v2.yaml")


@lru_cache(maxsize=1)
def load_evolving_geometry_policy() -> LoadedHPKPolicy:
    """Load the explicit semantic-attempt v2 policy once on active use."""

    return load_policy_config(
        evolving_geometry_policy_path(),
        EVOLVING_GEOMETRY_POLICY_RAW_SHA256,
        "geometry",
    )


def safe_exploration_geometry_policy_path() -> Path:
    """Return the repository-owned evolving v3 safe-exploration policy."""

    return config_path("hpk_geometry_policy_v3.yaml")


@lru_cache(maxsize=1)
def load_safe_exploration_geometry_policy() -> LoadedHPKPolicy:
    """Load the exact v3 policy without changing the v2 compatibility default."""

    return load_policy_config(
        safe_exploration_geometry_policy_path(),
        SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256,
        "geometry",
    )


__all__ = [
    "HPKPolicyConfigError",
    "DEFAULT_GEOMETRY_POLICY_RAW_SHA256",
    "EVOLVING_GEOMETRY_POLICY_CONFIG_SHA256",
    "EVOLVING_GEOMETRY_POLICY_IDENTITIES",
    "EVOLVING_GEOMETRY_POLICY_RAW_SHA256",
    "SAFE_EXPLORATION_GEOMETRY_POLICY_CONFIG_SHA256",
    "SAFE_EXPLORATION_GEOMETRY_POLICY_IDENTITIES",
    "SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256",
    "LoadedHPKPolicy",
    "PolicyKind",
    "default_geometry_policy_path",
    "evolving_geometry_policy_path",
    "load_default_geometry_policy",
    "load_evolving_geometry_policy",
    "load_safe_exploration_geometry_policy",
    "load_policy_config",
    "validate_geometry_policy_payload",
    "safe_exploration_geometry_policy_path",
]

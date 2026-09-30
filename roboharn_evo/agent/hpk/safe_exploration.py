from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.policy_config import (
    SAFE_EXPLORATION_GEOMETRY_POLICY_IDENTITIES,
    LoadedHPKPolicy,
    load_safe_exploration_geometry_policy,
)
from roboharn_evo.agent.hpk.runtime_binding import validate_runtime_binding
from roboharn_evo.agent.hpk.schemas import (
    canonical_json_bytes,
    stable_content_id,
    validate_content_id,
)


SAFE_EXPLORATION_PUBLIC_SCHEMA = "roboharn_evo/hpk/safe_exploration_public_audit/v1"
SAFE_EXPLORATION_PRIVATE_SCHEMA = "roboharn_evo/hpk/safe_exploration_private_audit/v1"
SAFE_EXPLORATION_PUBLIC_EVENT = "hpk_safe_exploration_public_audit"
SAFE_EXPLORATION_PRIVATE_EVENT = "hpk_safe_exploration_private_audit"
SAFE_EXPLORATION_MODE = "alternate_safe_baseline_rank_v1"
SAFE_EXPLORATION_ACTIVATION = "no_accepted_exact_match"

_PUBLIC_FIELDS = frozenset(
    {
        "schema",
        "audit_id",
        "runtime_binding_id",
        "snapshot_id",
        "snapshot_manifest_sha256",
        "geometry_policy_id",
        "geometry_policy_sha256",
        "mode",
        "activation",
        "condition_id",
        "task_strategy_id",
        "expected_effect_id",
        "eligible_count",
        "selected_baseline_rank",
        "guard_legal",
        "requested_candidate_present",
        "behavior_changed",
    }
)
_PRIVATE_FIELDS = frozenset(
    {
        "schema",
        "private_audit_id",
        "public_audit_id",
        "runtime_binding_id",
        "condition_id",
        "task_strategy_id",
        "expected_effect_id",
        "eligible_candidate_private_refs",
        "selected_candidate_private_ref",
        "selected_baseline_rank",
    }
)


class HPKSafeExplorationError(ValueError):
    """The frozen exploration policy or its audit binding is invalid."""


@dataclass(frozen=True, slots=True)
class SafeExplorationDecision:
    selected_candidate: dict[str, Any] | None
    reason: str
    public_audit: dict[str, Any] | None = None
    private_audit: dict[str, Any] | None = None

    @property
    def selected(self) -> bool:
        return self.selected_candidate is not None


def _fail(message: str) -> None:
    raise HPKSafeExplorationError(message)


def _sha(value: Any, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{path} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{path} must be an integer >= {minimum}")
    return value


def _content_id(value: Any, *, prefix: str, path: str) -> str:
    try:
        validate_content_id(value, prefix=prefix, path=path)
    except Exception as exc:
        raise HPKSafeExplorationError(str(exc)) from exc
    return str(value)


def _policy_profile(policy: LoadedHPKPolicy) -> dict[str, Any]:
    if not isinstance(policy, LoadedHPKPolicy) or policy.kind != "geometry":
        _fail("safe exploration requires a hash-pinned geometry policy")
    if (policy.policy_id, policy.config_sha256) not in SAFE_EXPLORATION_GEOMETRY_POLICY_IDENTITIES:
        _fail("safe exploration is available only in hpk_geometry_policy/v3")
    profile = policy.payload.get("safe_exploration")
    if not isinstance(profile, Mapping):
        _fail("geometry v3 lacks its frozen safe_exploration profile")
    return copy.deepcopy(dict(profile))


def _candidate_private_refs(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    refs: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            _fail("eligible candidates must be mappings")
        ref = candidate.get("candidate_id")
        if not isinstance(ref, str) or not ref.strip():
            _fail("eligible candidates require private candidate IDs")
        refs.append(ref.strip())
    if len(refs) != len(set(refs)):
        _fail("eligible candidate private IDs must be unique")
    return tuple(refs)


def validate_safe_exploration_public_audit(
    value: Mapping[str, Any],
    *,
    expected_runtime_binding: Mapping[str, Any] | None = None,
    expected_geometry_policy: LoadedHPKPolicy | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("safe exploration public audit must be an object")
    payload = copy.deepcopy(dict(value))
    if set(payload) != _PUBLIC_FIELDS:
        _fail("safe exploration public audit fields are not exact")
    if payload["schema"] not in {SAFE_EXPLORATION_PUBLIC_SCHEMA, "tcm/afk/safe_exploration_public_audit/v1"}:
        _fail("safe exploration public audit schema is unsupported")
    _content_id(payload["audit_id"], prefix="afkexplore", path="audit_id")
    _content_id(
        payload["runtime_binding_id"],
        prefix="afkruntime",
        path="runtime_binding_id",
    )
    _content_id(payload["snapshot_id"], prefix="afksnap", path="snapshot_id")
    _sha(payload["snapshot_manifest_sha256"], path="snapshot_manifest_sha256")
    if payload["geometry_policy_id"] not in {"hpk_geometry_policy/v3", "afk_geometry_policy/v3"}:
        _fail("safe exploration audit requires geometry policy v3")
    _sha(payload["geometry_policy_sha256"], path="geometry_policy_sha256")
    if payload["mode"] != SAFE_EXPLORATION_MODE:
        _fail("safe exploration public mode is unsupported")
    if payload["activation"] != SAFE_EXPLORATION_ACTIVATION:
        _fail("safe exploration activation is unsupported")
    _content_id(payload["condition_id"], prefix="afkc", path="condition_id")
    _content_id(payload["task_strategy_id"], prefix="afku", path="task_strategy_id")
    _content_id(
        payload["expected_effect_id"], prefix="afkfx", path="expected_effect_id"
    )
    eligible_count = _integer(payload["eligible_count"], path="eligible_count")
    selected_rank = _integer(
        payload["selected_baseline_rank"],
        path="selected_baseline_rank",
        minimum=0,
    )
    if selected_rank >= eligible_count:
        _fail("selected safe exploration rank is outside the eligible order")
    if (
        payload["guard_legal"] is not True
        or payload["requested_candidate_present"] is not False
        or payload["behavior_changed"] is not True
    ):
        _fail("safe exploration public audit has invalid safety flags")
    policy = expected_geometry_policy or load_safe_exploration_geometry_policy()
    profile = _policy_profile(policy)
    if (
        payload["geometry_policy_id"] != policy.policy_id
        or payload["geometry_policy_sha256"] != policy.config_sha256
        or payload["mode"] != profile["mode"]
        or payload["activation"] != profile["activate_when"]
        or payload["eligible_count"] < profile["min_eligible"]
        or payload["selected_baseline_rank"] != profile["rank_offset"]
        or payload["guard_legal"] is not profile["require_guard_legal"]
        or payload["requested_candidate_present"]
        is not (not profile["honor_requested"])
    ):
        _fail("safe exploration public audit differs from frozen policy replay")
    if expected_runtime_binding is not None:
        binding = validate_runtime_binding(expected_runtime_binding)
        geometry = binding["policy_refs"]["geometry"]
        if (
            payload["runtime_binding_id"] != binding["binding_id"]
            or payload["snapshot_id"] != binding["snapshot_ref"]["snapshot_id"]
            or payload["snapshot_manifest_sha256"]
            != binding["snapshot_ref"]["manifest_sha256"]
            or payload["geometry_policy_id"] != geometry["policy_id"]
            or payload["geometry_policy_sha256"] != geometry["config_sha256"]
        ):
            _fail("safe exploration public audit differs from runtime binding")
    identity = copy.deepcopy(payload)
    audit_id = identity.pop("audit_id")
    if audit_id != stable_content_id("afkexplore", identity):
        _fail("safe exploration public audit ID content mismatch")
    canonical_json_bytes(payload)
    return payload


def validate_safe_exploration_private_audit(
    value: Mapping[str, Any],
    *,
    expected_public_audit: Mapping[str, Any] | None = None,
    expected_geometry_policy: LoadedHPKPolicy | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("safe exploration private audit must be an object")
    payload = copy.deepcopy(dict(value))
    if set(payload) != _PRIVATE_FIELDS:
        _fail("safe exploration private audit fields are not exact")
    if payload["schema"] not in {SAFE_EXPLORATION_PRIVATE_SCHEMA, "tcm/afk/safe_exploration_private_audit/v1"}:
        _fail("safe exploration private audit schema is unsupported")
    _content_id(
        payload["private_audit_id"],
        prefix="afkprivexplore",
        path="private_audit_id",
    )
    _content_id(payload["public_audit_id"], prefix="afkexplore", path="public_audit_id")
    _content_id(
        payload["runtime_binding_id"],
        prefix="afkruntime",
        path="runtime_binding_id",
    )
    _content_id(payload["condition_id"], prefix="afkc", path="condition_id")
    _content_id(payload["task_strategy_id"], prefix="afku", path="task_strategy_id")
    _content_id(
        payload["expected_effect_id"], prefix="afkfx", path="expected_effect_id"
    )
    raw_refs = payload["eligible_candidate_private_refs"]
    if not isinstance(raw_refs, list):
        _fail("eligible_candidate_private_refs must be an array")
    refs = _candidate_private_refs([{"candidate_id": item} for item in raw_refs])
    policy = expected_geometry_policy or load_safe_exploration_geometry_policy()
    profile = _policy_profile(policy)
    if len(refs) < profile["min_eligible"]:
        _fail("safe exploration private audit has too few eligible candidates")
    selected = payload["selected_candidate_private_ref"]
    if not isinstance(selected, str) or not selected.strip():
        _fail("selected_candidate_private_ref must be non-empty")
    rank = _integer(
        payload["selected_baseline_rank"],
        path="selected_baseline_rank",
        minimum=0,
    )
    if (
        rank != profile["rank_offset"]
        or rank >= len(refs)
        or refs[rank] != selected.strip()
    ):
        _fail("private selected candidate differs from the frozen baseline rank")
    if expected_public_audit is not None:
        public = validate_safe_exploration_public_audit(
            expected_public_audit,
            expected_geometry_policy=expected_geometry_policy,
        )
        for private_key, public_key in (
            ("public_audit_id", "audit_id"),
            ("runtime_binding_id", "runtime_binding_id"),
            ("condition_id", "condition_id"),
            ("task_strategy_id", "task_strategy_id"),
            ("expected_effect_id", "expected_effect_id"),
            ("selected_baseline_rank", "selected_baseline_rank"),
        ):
            if payload[private_key] != public[public_key]:
                _fail("safe exploration public/private audit binding differs")
        if len(refs) != public["eligible_count"]:
            _fail("safe exploration public/private eligible counts differ")
    identity = copy.deepcopy(payload)
    private_id = identity.pop("private_audit_id")
    if private_id != stable_content_id("afkprivexplore", identity):
        _fail("safe exploration private audit ID content mismatch")
    canonical_json_bytes(payload)
    return payload


def choose_safe_baseline_alternative(
    *,
    geometry_policy: LoadedHPKPolicy,
    runtime_binding: Mapping[str, Any],
    condition_id: str,
    task_strategy_id: str,
    expected_effect_id: str,
    eligible_candidates: Sequence[Mapping[str, Any]],
    accepted_exact_match: bool,
    requested_candidate_id: Any,
    guard_legal: bool,
    prior_selections_for_condition: int,
) -> SafeExplorationDecision:
    """Select one frozen alternate rank from a caller-proven baseline order."""

    profile = _policy_profile(geometry_policy)
    binding = validate_runtime_binding(runtime_binding)
    if binding["policy_refs"]["geometry"] != geometry_policy.identity():
        _fail("safe exploration policy differs from the episode runtime binding")
    _content_id(condition_id, prefix="afkc", path="condition_id")
    _content_id(task_strategy_id, prefix="afku", path="task_strategy_id")
    _content_id(expected_effect_id, prefix="afkfx", path="expected_effect_id")
    if accepted_exact_match:
        return SafeExplorationDecision(None, "accepted_exact_match")
    if requested_candidate_id is not None and str(requested_candidate_id).strip():
        return SafeExplorationDecision(None, "requested_candidate_honored")
    if guard_legal is not True:
        return SafeExplorationDecision(None, "guard_legality_unproven")
    if (
        isinstance(prior_selections_for_condition, bool)
        or not isinstance(prior_selections_for_condition, int)
        or prior_selections_for_condition < 0
    ):
        _fail("prior_selections_for_condition must be a non-negative integer")
    if prior_selections_for_condition >= int(
        profile["max_selections_per_condition_per_episode"]
    ):
        return SafeExplorationDecision(None, "condition_episode_limit_reached")
    if any(not isinstance(item, Mapping) for item in eligible_candidates):
        _fail("eligible candidates must be mappings")
    candidates = tuple(copy.deepcopy(dict(item)) for item in eligible_candidates)
    if len(candidates) < int(profile["min_eligible"]):
        return SafeExplorationDecision(None, "insufficient_eligible_candidates")
    rank = int(profile["rank_offset"])
    if rank >= len(candidates):
        return SafeExplorationDecision(None, "alternate_rank_unavailable")

    # Rank is chosen before opaque private identities are read for the audit.
    selected_candidate = copy.deepcopy(candidates[rank])
    private_refs = _candidate_private_refs(candidates)
    public_payload = {
        "schema": SAFE_EXPLORATION_PUBLIC_SCHEMA,
        "runtime_binding_id": binding["binding_id"],
        "snapshot_id": binding["snapshot_ref"]["snapshot_id"],
        "snapshot_manifest_sha256": binding["snapshot_ref"]["manifest_sha256"],
        "geometry_policy_id": geometry_policy.policy_id,
        "geometry_policy_sha256": geometry_policy.config_sha256,
        "mode": profile["mode"],
        "activation": profile["activate_when"],
        "condition_id": condition_id,
        "task_strategy_id": task_strategy_id,
        "expected_effect_id": expected_effect_id,
        "eligible_count": len(candidates),
        "selected_baseline_rank": rank,
        "guard_legal": True,
        "requested_candidate_present": False,
        "behavior_changed": True,
    }
    public_payload["audit_id"] = stable_content_id("afkexplore", public_payload)
    public_audit = validate_safe_exploration_public_audit(
        public_payload,
        expected_runtime_binding=binding,
    )
    private_payload = {
        "schema": SAFE_EXPLORATION_PRIVATE_SCHEMA,
        "public_audit_id": public_audit["audit_id"],
        "runtime_binding_id": binding["binding_id"],
        "condition_id": condition_id,
        "task_strategy_id": task_strategy_id,
        "expected_effect_id": expected_effect_id,
        "eligible_candidate_private_refs": list(private_refs),
        "selected_candidate_private_ref": private_refs[rank],
        "selected_baseline_rank": rank,
    }
    private_payload["private_audit_id"] = stable_content_id(
        "afkprivexplore", private_payload
    )
    private_audit = validate_safe_exploration_private_audit(
        private_payload,
        expected_public_audit=public_audit,
    )
    return SafeExplorationDecision(
        selected_candidate=selected_candidate,
        reason="selected_alternate_safe_baseline_rank",
        public_audit=public_audit,
        private_audit=private_audit,
    )


__all__ = [
    "HPKSafeExplorationError",
    "SAFE_EXPLORATION_ACTIVATION",
    "SAFE_EXPLORATION_MODE",
    "SAFE_EXPLORATION_PRIVATE_EVENT",
    "SAFE_EXPLORATION_PRIVATE_SCHEMA",
    "SAFE_EXPLORATION_PUBLIC_EVENT",
    "SAFE_EXPLORATION_PUBLIC_SCHEMA",
    "SafeExplorationDecision",
    "choose_safe_baseline_alternative",
    "validate_safe_exploration_private_audit",
    "validate_safe_exploration_public_audit",
]

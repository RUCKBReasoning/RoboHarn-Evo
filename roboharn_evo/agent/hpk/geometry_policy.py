from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from roboharn_evo.agent.hpk.audit import (
    HPK_GEOMETRY_SCORE_PROFILE,
    LEGACY_RANKING_MAPPING_PERSISTENCE_STATUS,
    PrivateRankingAuditSink,
)
from roboharn_evo.agent.hpk.policy_config import (
    LoadedHPKPolicy,
    load_default_geometry_policy,
    validate_geometry_policy_payload,
)
from roboharn_evo.agent.hpk.schemas import GEOMETRIC_STRATEGY_SCHEMA


FAIL_CLOSED = "fail_closed"
BASELINE_FALLBACK = "baseline_fallback"
AllHardMismatchBehavior = Literal["fail_closed", "baseline_fallback"]

_HARD_CONSTRAINTS = frozenset({"support_valid", "target_region_free"})
_SOFT_PREFERENCES = frozenset({"lower_reach_distance"})
_AVOID_PREFERENCES = frozenset({"repeat_equivalent_failed_candidate"})
_UNKNOWN_TOKENS = frozenset({"", "unknown", "unconstrained"})
_SCORING_PROFILE = HPK_GEOMETRY_SCORE_PROFILE
_CANDIDATE_FEATURE_SIDECAR_KEYS = frozenset(
    {
        "hpk_geometry_features",
        "candidate_geometry_features",
        "geometry_features",
    }
)
_PLACE_ATTACHMENT_SOURCE = "verifier_confirmed_tcp_local_attachment"


@dataclass(frozen=True, slots=True)
class GeometricRankingRequest:
    """Bind a validated strategy to one runtime mismatch policy.

    The mismatch behavior is runtime configuration, not part of the
    transferable ``GeometricStrategyV1`` payload.  Keeping it in this wrapper
    avoids adding an out-of-schema field to persistent HPK.
    """

    geometric_strategy: Any
    all_hard_mismatch_behavior: AllHardMismatchBehavior = FAIL_CLOSED
    geometry_policy: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_mismatch_behavior(self.all_hard_mismatch_behavior)
        if self.geometry_policy is not None:
            validate_geometry_policy_payload(self.geometry_policy)


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    candidate: dict[str, Any]
    candidate_id: str
    baseline_rank: int
    hard_constraint_results: dict[str, bool]
    hard_constraints_satisfied: bool
    safety_rejections: tuple[str, ...]
    soft_score: float | None
    soft_score_components: dict[str, float]
    geometric_compliance: bool | Literal["unverified"]


@dataclass(frozen=True, slots=True)
class GeometryRankingResult:
    """记录使用 HPK 的候选排序结果。"""

    baseline_candidates: tuple[dict[str, Any], ...]
    ranked_candidates: tuple[dict[str, Any], ...]
    evaluations: tuple[_CandidateEvaluation, ...]
    strategy_schema: str
    geometric_strategy_id: str
    all_hard_mismatch_behavior: AllHardMismatchBehavior
    all_hard_mismatch: bool
    zero_compliant_candidates: bool
    fallback_used: bool

    def select(self, requested_candidate_id: Any = None) -> dict[str, Any] | None:
        requested = str(requested_candidate_id or "").strip()
        if requested:
            return next(
                (
                    candidate
                    for candidate in self.ranked_candidates
                    if str(candidate.get("candidate_id", "") or "").strip() == requested
                ),
                None,
            )
        return self.ranked_candidates[0] if self.ranked_candidates else None

    def write_private_audit(
        self,
        ranking_audit: MutableMapping[str, Any] | PrivateRankingAuditSink | None,
        *,
        selected_candidate: Mapping[str, Any] | None,
        requested_candidate_id: Any = None,
    ) -> None:
        """Write one private ranking result to a legacy mapping or typed sink.

        Mutable mappings remain the P0-B in-memory diagnostics API.  They are
        explicitly marked as unapproved for P0-C live persistence.  The static
        runtime must pass :class:`PrivateRankingAuditSink`, which validates the
        low-level output before the selector can return an HPK-modified action.
        """

        if ranking_audit is None:
            return
        _validate_private_audit_sink(ranking_audit)
        selected_id = (
            str(selected_candidate.get("candidate_id", "") or "").strip()
            if isinstance(selected_candidate, Mapping)
            else ""
        )
        evaluation_by_id = {item.candidate_id: item for item in self.evaluations}
        selected_evaluation = evaluation_by_id.get(selected_id)
        compliance: bool | Literal["unverified"] = "unverified"
        if selected_evaluation is not None:
            compliance = selected_evaluation.geometric_compliance
            if self.fallback_used:
                compliance = False
        elif self.zero_compliant_candidates:
            # An active typed strategy has evaluated at least one otherwise-safe
            # baseline candidate and proved that none is fully compliant.  This
            # is a negative geometric result, not missing evidence.
            compliance = False

        after_ids = [
            str(candidate.get("candidate_id", "") or "").strip()
            for candidate in self.ranked_candidates
        ]
        payload = {
            "candidate_ids_before": [
                str(candidate.get("candidate_id", "") or "").strip()
                for candidate in self.baseline_candidates
            ],
            "candidate_rank_before": list(range(len(self.baseline_candidates))),
            "candidate_ids_after": after_ids,
            "candidate_rank_after": list(range(len(after_ids))),
            "selected_geometric_hpk_id": self.geometric_strategy_id,
            "selected_candidate_id": selected_id or None,
            "geometric_compliance": compliance,
            "private_ranking_diagnostics": {
                "privacy": "private_runtime_candidate_identity",
                "scoring_profile": _SCORING_PROFILE,
                "geometric_strategy_schema": self.strategy_schema,
                "all_hard_mismatch_behavior": (self.all_hard_mismatch_behavior),
                "all_hard_mismatch": self.all_hard_mismatch,
                "zero_compliant_candidates": self.zero_compliant_candidates,
                "baseline_fallback_used": self.fallback_used,
                "baseline_rank_by_after": [
                    evaluation_by_id[candidate_id].baseline_rank
                    for candidate_id in after_ids
                ],
                "hard_constraint_results_before": [
                    dict(item.hard_constraint_results) for item in self.evaluations
                ],
                "geometric_compliance_before": [
                    item.geometric_compliance for item in self.evaluations
                ],
                "candidate_rejections_before": [
                    _candidate_rejection_reasons(item) for item in self.evaluations
                ],
                "hpk_soft_scores_after": [
                    evaluation_by_id[candidate_id].soft_score
                    for candidate_id in after_ids
                ],
                "hpk_soft_score_components_after": [
                    dict(evaluation_by_id[candidate_id].soft_score_components)
                    for candidate_id in after_ids
                ],
            },
            "persistence_status": (LEGACY_RANKING_MAPPING_PERSISTENCE_STATUS),
        }
        requested = str(requested_candidate_id or "").strip()
        baseline_selected_id = (
            requested
            if requested
            and any(
                str(candidate.get("candidate_id", "") or "").strip() == requested
                for candidate in self.baseline_candidates
            )
            else (
                str(self.baseline_candidates[0].get("candidate_id", "") or "").strip()
                if self.baseline_candidates
                else None
            )
        )
        if isinstance(ranking_audit, PrivateRankingAuditSink):
            ranking_audit.accept_low_level_ranking(
                payload,
                baseline_selected_candidate_private_ref=(baseline_selected_id),
            )
            return
        ranking_audit.clear()
        ranking_audit.update(payload)


class HPKGeometryPolicy:
    """Apply one typed geometric strategy to a baseline-ranked candidate list."""

    def __init__(
        self,
        *,
        geometry_policy: LoadedHPKPolicy | Mapping[str, Any] | None = None,
        all_hard_mismatch_behavior: AllHardMismatchBehavior | None = None,
    ) -> None:
        if geometry_policy is None:
            loaded = load_default_geometry_policy()
            self._geometry_policy = loaded.payload
            self._geometry_policy_id = loaded.policy_id
            self._geometry_policy_sha256 = loaded.config_sha256
        elif isinstance(geometry_policy, LoadedHPKPolicy):
            if geometry_policy.kind != "geometry":
                raise ValueError("HPKGeometryPolicy requires a geometry policy")
            self._geometry_policy = geometry_policy.payload
            self._geometry_policy_id = geometry_policy.policy_id
            self._geometry_policy_sha256 = geometry_policy.config_sha256
        else:
            self._geometry_policy = validate_geometry_policy_payload(geometry_policy)
            self._geometry_policy_id = str(self._geometry_policy["policy_id"])
            self._geometry_policy_sha256 = ""
        configured_mismatch = _validate_mismatch_behavior(
            self._geometry_policy["all_hard_mismatch_behavior"]
        )
        if (
            all_hard_mismatch_behavior is not None
            and _validate_mismatch_behavior(all_hard_mismatch_behavior)
            != configured_mismatch
        ):
            raise ValueError(
                "all_hard_mismatch_behavior must come from geometry policy"
            )
        self._all_hard_mismatch_behavior = configured_mismatch

    @property
    def all_hard_mismatch_behavior(self) -> AllHardMismatchBehavior:
        return self._all_hard_mismatch_behavior

    @property
    def geometry_policy_id(self) -> str:
        return self._geometry_policy_id

    @property
    def geometry_policy_sha256(self) -> str:
        return self._geometry_policy_sha256

    def activate(self, geometric_strategy: Any) -> GeometricRankingRequest:
        return GeometricRankingRequest(
            geometric_strategy=geometric_strategy,
            all_hard_mismatch_behavior=self._all_hard_mismatch_behavior,
            geometry_policy=self._geometry_policy,
        )

    def rank_candidates(
        self,
        baseline_candidates: Sequence[Mapping[str, Any]],
        *,
        geometric_strategy: Any,
        scene_state: Mapping[str, Any] | None = None,
    ) -> GeometryRankingResult:
        strategy_value = geometric_strategy
        mismatch_behavior = self._all_hard_mismatch_behavior
        if isinstance(geometric_strategy, GeometricRankingRequest):
            strategy_value = geometric_strategy.geometric_strategy
            mismatch_behavior = geometric_strategy.all_hard_mismatch_behavior
            if mismatch_behavior != self._all_hard_mismatch_behavior:
                raise ValueError(
                    "GeometricRankingRequest mismatch behavior differs from "
                    "the frozen geometry policy"
                )

        strategy, geometric_strategy_id = _validated_geometric_strategy(strategy_value)
        schema = _token(strategy.get("schema"))
        if schema not in {GEOMETRIC_STRATEGY_SCHEMA, "tcm/afk/geometric_strategy/v1"}:
            raise ValueError(
                f"geometric_strategy must use schema {GEOMETRIC_STRATEGY_SCHEMA!r}"
            )
        hard_constraints = _validated_string_sequence(
            strategy,
            "hard_constraints",
            supported=_HARD_CONSTRAINTS,
        )
        _validated_string_sequence(
            strategy,
            "soft_preferences",
            supported=_SOFT_PREFERENCES,
        )
        _validated_string_sequence(
            strategy,
            "avoid",
            supported=_AVOID_PREFERENCES,
        )

        baseline = tuple(dict(candidate) for candidate in baseline_candidates)
        _validate_baseline_candidate_ids(baseline)
        evaluations: list[_CandidateEvaluation] = []
        hard_matches: list[_CandidateEvaluation] = []
        compliant_matches: list[_CandidateEvaluation] = []
        safe_baseline: list[_CandidateEvaluation] = []
        for baseline_rank, candidate in enumerate(baseline):
            candidate_id = str(candidate.get("candidate_id", "") or "").strip()
            features = _candidate_features(
                scene_state,
                candidate,
                geometry_policy=self._geometry_policy,
            )
            safety_rejections = _candidate_safety_rejections(candidate, features)
            hard_results = {
                name: _hard_constraint_satisfied(name, candidate, features)
                for name in hard_constraints
            }
            hard_satisfied = not safety_rejections and all(hard_results.values())
            score: float | None = None
            components: dict[str, float] = {}
            compliance = _geometric_compliance(
                strategy,
                features,
                hard_constraints_satisfied=hard_satisfied,
            )
            if compliance is True:
                score, components = _soft_score(
                    strategy,
                    features,
                    geometry_policy=self._geometry_policy,
                )
            evaluation = _CandidateEvaluation(
                candidate=candidate,
                candidate_id=candidate_id,
                baseline_rank=baseline_rank,
                hard_constraint_results=hard_results,
                hard_constraints_satisfied=hard_satisfied,
                safety_rejections=safety_rejections,
                soft_score=score,
                soft_score_components=components,
                geometric_compliance=compliance,
            )
            evaluations.append(evaluation)
            if not safety_rejections:
                safe_baseline.append(evaluation)
            if hard_satisfied:
                hard_matches.append(evaluation)
            if compliance is True:
                compliant_matches.append(evaluation)

        all_hard_mismatch = bool(
            hard_constraints and safe_baseline and not hard_matches
        )
        # Once a typed strategy is active, every otherwise eligible baseline
        # candidate participates in the fail-closed decision.  Candidates that
        # current Scene Memory rejects are not "missing evidence" and must not
        # make a zero-compliance result look unverified.
        zero_compliant_candidates = not compliant_matches
        fallback_used = bool(
            zero_compliant_candidates
            and mismatch_behavior == BASELINE_FALLBACK
            and safe_baseline
        )
        if compliant_matches:
            ranked_evaluations = sorted(
                compliant_matches,
                key=lambda item: (
                    -float(item.soft_score or 0.0),
                    item.baseline_rank,
                ),
            )
        elif fallback_used:
            ranked_evaluations = safe_baseline
        else:
            ranked_evaluations = []

        return GeometryRankingResult(
            baseline_candidates=baseline,
            ranked_candidates=tuple(item.candidate for item in ranked_evaluations),
            evaluations=tuple(evaluations),
            strategy_schema=schema,
            geometric_strategy_id=geometric_strategy_id,
            all_hard_mismatch_behavior=mismatch_behavior,
            all_hard_mismatch=all_hard_mismatch,
            zero_compliant_candidates=zero_compliant_candidates,
            fallback_used=fallback_used,
        )


def rank_operation_pose_candidates(
    baseline_candidates: Sequence[Mapping[str, Any]],
    *,
    geometric_strategy: Any,
    scene_state: Mapping[str, Any] | None = None,
    geometry_policy: LoadedHPKPolicy | Mapping[str, Any] | None = None,
    all_hard_mismatch_behavior: AllHardMismatchBehavior | None = None,
) -> GeometryRankingResult:
    """Rank already-eligible candidates without changing the input sequence."""

    if (
        geometry_policy is None
        and isinstance(geometric_strategy, GeometricRankingRequest)
        and geometric_strategy.geometry_policy is not None
    ):
        geometry_policy = geometric_strategy.geometry_policy

    return HPKGeometryPolicy(
        geometry_policy=geometry_policy,
        all_hard_mismatch_behavior=all_hard_mismatch_behavior,
    ).rank_candidates(
        baseline_candidates,
        geometric_strategy=geometric_strategy,
        scene_state=scene_state,
    )


def validate_private_ranking_audit_sink(value: Any) -> None:
    """验证 HPK 选择器的可选审计输出接口。"""

    _validate_private_audit_sink(value)


def _validate_private_audit_sink(value: Any) -> None:
    if value is not None and not isinstance(
        value,
        (MutableMapping, PrivateRankingAuditSink),
    ):
        raise TypeError(
            "ranking_audit must be a legacy mutable mapping, "
            "PrivateRankingAuditSink, or None"
        )


def _validate_baseline_candidate_ids(
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    candidate_ids: list[str] = []
    for candidate in candidates:
        raw_candidate_id = candidate.get("candidate_id")
        if not isinstance(raw_candidate_id, str) or not raw_candidate_id.strip():
            raise ValueError(
                "baseline candidates require non-empty private candidate IDs"
            )
        candidate_ids.append(raw_candidate_id.strip())
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("baseline candidate IDs must be unique")


def _candidate_rejection_reasons(
    evaluation: _CandidateEvaluation,
) -> list[str]:
    if evaluation.safety_rejections:
        return list(evaluation.safety_rejections)
    if not evaluation.hard_constraints_satisfied:
        return ["hpk_hard_constraint_mismatch"]
    if evaluation.geometric_compliance is False:
        return ["hpk_geometric_compliance_mismatch"]
    if evaluation.geometric_compliance == "unverified":
        return ["hpk_geometric_compliance_unverified"]
    return []


def _validate_mismatch_behavior(value: Any) -> AllHardMismatchBehavior:
    normalized = _token(value)
    if normalized not in {FAIL_CLOSED, BASELINE_FALLBACK}:
        raise ValueError(
            "all_hard_mismatch_behavior must be 'fail_closed' or 'baseline_fallback'"
        )
    return normalized  # type: ignore[return-value]


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError(f"{label} must be a mapping or expose to_dict()")


def _validated_geometric_strategy(value: Any) -> tuple[dict[str, Any], str]:
    from roboharn_evo.agent.hpk.schemas import GeometricStrategyV1

    payload = _as_mapping(value, label="geometric_strategy")
    validated = GeometricStrategyV1.from_dict(payload)
    return validated.to_dict(), validated.stable_id


def _validated_candidate_features(value: Any, *, label: str) -> dict[str, Any]:
    from roboharn_evo.agent.hpk.schemas import CandidateGeometryFeaturesV1

    payload = _as_mapping(value, label=label)
    return CandidateGeometryFeaturesV1.from_dict(payload).to_dict()


def _validated_string_sequence(
    payload: Mapping[str, Any],
    key: str,
    *,
    supported: frozenset[str],
) -> tuple[str, ...]:
    raw = payload.get(key, [])
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"geometric_strategy.{key} must be a list")
    values: list[str] = []
    for item in raw:
        value = _token(item)
        if not value or value not in supported:
            raise ValueError(f"unsupported geometric_strategy.{key} value: {value!r}")
        if value not in values:
            values.append(value)
    return tuple(values)


def _candidate_features(
    scene_state: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    *,
    geometry_policy: Mapping[str, Any],
) -> dict[str, Any]:
    fallback = _fallback_candidate_features(
        candidate,
        geometry_policy=geometry_policy,
    )
    from roboharn_evo.agent.hpk.candidate_features import candidate_semantic_features

    projected = candidate_semantic_features(
        scene_state or {},
        candidate,
        geometry_policy=geometry_policy,
    )
    if getattr(projected, "resolved", True) is False:
        return {
            **fallback,
            "feature_projection_resolved": False,
            "reference_frame": "unknown",
            "target_relation": {
                "relation": None,
                "reference_role": None,
            },
            "support_valid": None,
            "target_region_free": None,
            "approach": {
                "family": "unconstrained",
                "direction_bucket": "unknown",
            },
            "orientation": {"relation": "unknown"},
            "grasp": {"region": "unknown", "semantic_part": None},
            "geometry_source_class": "unknown",
        }
    return _merge_candidate_features(
        fallback,
        _validated_candidate_features(
            projected,
            label="candidate_semantic_features result",
        ),
    )


def _merge_candidate_features(
    fallback: Mapping[str, Any],
    projected: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize P0-A's flat private feature schema for policy matching."""

    action_mode = _token(projected.get("action_mode"))
    return {
        **fallback,
        "feature_projection_resolved": True,
        "action_mode": action_mode,
        "strategy_family": {
            "place": "placement_relation",
            "grasp": "observed_grasp_geometry",
            "contact": "contact_relation",
        }.get(action_mode, ""),
        "target_relation": {
            "relation": projected.get("target_relation"),
            "reference_role": projected.get("target_reference_role"),
        },
        "support_valid": projected.get("support_valid"),
        "target_region_free": projected.get("target_region_free"),
        "approach": {
            "family": _token(projected.get("approach_family")),
            "direction_bucket": _token(projected.get("approach_direction_bucket")),
        },
        "orientation": {"relation": _token(projected.get("orientation_relation"))},
        "grasp": {
            "region": _token(projected.get("grasp_region")),
            "semantic_part": projected.get("semantic_part"),
        },
        "grasp_width_bucket": projected.get("grasp_width_bucket"),
        "reach_distance_bucket": projected.get("reach_distance_bucket"),
        "geometry_source_class": projected.get("geometry_source_class"),
    }


def _fallback_candidate_features(
    candidate: Mapping[str, Any],
    *,
    geometry_policy: Mapping[str, Any],
) -> dict[str, Any]:
    action_mode = _token(candidate.get("action_mode"))
    relation = _candidate_target_relation(candidate)
    target_reference_role = _candidate_target_reference_role(candidate)
    geometry_source = _token(
        candidate.get("geometry_source_class", candidate.get("geometry_source"))
    )
    geometry_source_class = _geometry_source_class(geometry_source)
    approach_family = _token(candidate.get("approach_family"))
    if not approach_family:
        if action_mode == "place" and _positive_finite(
            candidate.get("approach_clearance_m")
        ):
            approach_family = "clearance_first"
        elif "surface_normal" in geometry_source:
            approach_family = "surface_normal"
        elif "principal_ax" in geometry_source:
            approach_family = "principal_axis_relative"
        else:
            approach_family = "unconstrained"
    orientation_relation = _token(candidate.get("orientation_relation"))
    if (
        action_mode == "place"
        and orientation_relation == "preserve_current_attachment"
        and not _valid_place_attachment_geometry(candidate)
    ):
        orientation_relation = "unknown"
    if not orientation_relation:
        orientation_policy = _token(candidate.get("orientation_policy"))
        if "preserve_current" in orientation_policy:
            orientation_relation = (
                "preserve_current_attachment"
                if action_mode != "place" or _valid_place_attachment_geometry(candidate)
                else "unknown"
            )
        elif "principal_ax" in geometry_source:
            source_index = _integer(candidate.get("source_candidate_index"))
            if source_index in {0, 1}:
                orientation_relation = f"align_principal_axis_{source_index}"
    return {
        "action_mode": action_mode,
        "strategy_family": {
            "place": "placement_relation",
            "grasp": "observed_grasp_geometry",
            "contact": "contact_relation",
        }.get(action_mode, ""),
        "reference_frame": _candidate_reference_frame(
            candidate,
            geometry_source=geometry_source,
            orientation_relation=orientation_relation,
        ),
        "target_relation": {
            "relation": relation,
            "reference_role": target_reference_role,
        },
        "support_valid": candidate.get("support_valid"),
        "target_region_free": _target_region_free(candidate),
        "approach": {
            "family": approach_family,
            "direction_bucket": _approach_direction_bucket(
                candidate,
                geometry_policy=geometry_policy,
            ),
        },
        "orientation": {"relation": orientation_relation or "unknown"},
        "grasp": {
            "region": _token(candidate.get("grasp_region")) or "unknown",
            "semantic_part": candidate.get("grasp_semantic_part"),
        },
        "reach_distance_m": _nonnegative_finite(candidate.get("reach_distance_m")),
        "geometry_source_class": geometry_source_class,
        "repeat_equivalent_failed_candidate": any(
            candidate.get(key) is True
            for key in (
                "repeat_equivalent_failed_candidate",
                "equivalent_failed_candidate",
                "failed_equivalent",
            )
        ),
    }


def _candidate_safety_rejections(
    candidate: Mapping[str, Any],
    features: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if any(key in candidate for key in _CANDIDATE_FEATURE_SIDECAR_KEYS):
        reasons.append("candidate_embedded_feature_sidecar_forbidden")
    if features.get("feature_projection_resolved") is not True:
        reasons.append("candidate_feature_projection_unresolved")
    if _token(candidate.get("action_mode")) == "place":
        place_requirements = {
            "place_target_not_revalidated": "place_target_revalidated",
            "candidate_invalid": "valid",
            "candidate_unreachable": "reachable_estimate",
            "support_invalid": "support_valid",
            "target_region_occupied": "free",
        }
        for reason, key in place_requirements.items():
            if candidate.get(key) is not True:
                reasons.append(reason)
        if not _valid_place_attachment_geometry(candidate):
            reasons.append("place_attachment_geometry_invalid")
    false_flags = {
        "candidate_invalid": ("valid", "legal", "eligible", "feasible"),
        "candidate_unreachable": ("reachable", "reachable_estimate"),
        "support_invalid": ("support_valid",),
        "target_region_occupied": ("free", "target_region_free"),
    }
    for reason, keys in false_flags.items():
        if any(
            key in source
            and source.get(key) is not None
            and source.get(key) is not True
            for source in (candidate, features)
            for key in keys
        ):
            reasons.append(reason)
    if any(
        key in source and source.get(key) is not None and source.get(key) is not False
        for source in (candidate, features)
        for key in ("blocked", "unreachable", "support_invalid", "occupied")
    ):
        reasons.append("explicit_safety_block")
    occupied_by = candidate.get("occupied_by", features.get("occupied_by"))
    if occupied_by is not None and bool(occupied_by):
        reasons.append("target_region_occupied")
    return tuple(dict.fromkeys(reasons))


def _hard_constraint_satisfied(
    name: str,
    candidate: Mapping[str, Any],
    features: Mapping[str, Any],
) -> bool:
    if name == "support_valid":
        return _first_value(features, candidate, "support_valid") is True
    if name == "target_region_free":
        return _first_value(features, candidate, "target_region_free", "free") is True
    return False


def _soft_score(
    strategy: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    geometry_policy: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    score_policy = _nested_mapping(geometry_policy, "soft_score")
    match_weights = _nested_mapping(score_policy, "match_weights")

    def add_match(name: str, expected: Any, actual: Any) -> None:
        expected_token = _token(expected)
        if expected_token in _UNKNOWN_TOKENS:
            return
        weight = float(match_weights[name])
        components[name] = weight if _token(actual) == expected_token else 0.0

    add_match(
        "strategy_family_match",
        strategy.get("strategy_family"),
        features.get("strategy_family"),
    )
    add_match(
        "reference_frame_match",
        strategy.get("reference_frame"),
        features.get("reference_frame"),
    )
    target_relation = _nested_mapping(strategy, "target_relation")
    feature_target_relation = _nested_mapping(features, "target_relation")
    add_match(
        "target_relation_match",
        target_relation.get("relation"),
        feature_target_relation.get("relation"),
    )
    add_match(
        "target_reference_role_match",
        target_relation.get("reference_role"),
        feature_target_relation.get("reference_role"),
    )
    approach = _nested_mapping(strategy, "approach")
    feature_approach = _nested_mapping(features, "approach")
    add_match(
        "approach_family_match",
        approach.get("family"),
        feature_approach.get("family"),
    )
    add_match(
        "approach_direction_match",
        approach.get("direction_bucket"),
        feature_approach.get("direction_bucket"),
    )
    orientation = _nested_mapping(strategy, "orientation")
    feature_orientation = _nested_mapping(features, "orientation")
    add_match(
        "orientation_match",
        orientation.get("relation"),
        feature_orientation.get("relation"),
    )
    grasp = _nested_mapping(strategy, "grasp")
    feature_grasp = _nested_mapping(features, "grasp")
    add_match(
        "grasp_region_match",
        grasp.get("region"),
        feature_grasp.get("region"),
    )
    add_match(
        "grasp_semantic_part_match",
        grasp.get("semantic_part"),
        feature_grasp.get("semantic_part"),
    )
    capability = _nested_mapping(strategy, "capability_evidence")
    add_match(
        "geometry_source_class_match",
        capability.get("geometry_source_class"),
        features.get("geometry_source_class"),
    )

    soft_preferences = _validated_string_sequence(
        strategy,
        "soft_preferences",
        supported=_SOFT_PREFERENCES,
    )
    if "lower_reach_distance" in soft_preferences:
        reach_policy = _nested_mapping(score_policy, "reach_score")
        reach_distance = _nonnegative_finite(features.get("reach_distance_m"))
        if reach_distance is not None:
            reach_score = round(
                float(reach_policy["numerator"])
                / (float(reach_policy["denominator_offset"]) + reach_distance),
                int(reach_policy["round_digits"]),
            )
        else:
            bucket_scores = _nested_mapping(reach_policy, "bucket_scores")
            reach_score = float(
                bucket_scores.get(
                    _token(features.get("reach_distance_bucket")),
                    0.0,
                )
            )
        components["lower_reach_distance"] = reach_score
    avoid = _validated_string_sequence(
        strategy,
        "avoid",
        supported=_AVOID_PREFERENCES,
    )
    if "repeat_equivalent_failed_candidate" in avoid:
        penalties = _nested_mapping(score_policy, "avoid_penalties")
        components["repeat_equivalent_failed_candidate"] = (
            float(penalties["repeat_equivalent_failed_candidate"])
            if features.get("repeat_equivalent_failed_candidate") is True
            else 0.0
        )
    score_round_digits = int(
        _nested_mapping(score_policy, "reach_score")["round_digits"]
    )
    return round(sum(components.values()), score_round_digits), components


def _geometric_compliance(
    strategy: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    hard_constraints_satisfied: bool,
) -> bool | Literal["unverified"]:
    if not hard_constraints_satisfied:
        return False
    comparisons = [
        (strategy.get("strategy_family"), features.get("strategy_family")),
        (strategy.get("reference_frame"), features.get("reference_frame")),
        (
            _nested_mapping(strategy, "target_relation").get("relation"),
            _nested_mapping(features, "target_relation").get("relation"),
        ),
        (
            _nested_mapping(strategy, "target_relation").get("reference_role"),
            _nested_mapping(features, "target_relation").get("reference_role"),
        ),
        (
            _nested_mapping(strategy, "approach").get("family"),
            _nested_mapping(features, "approach").get("family"),
        ),
        (
            _nested_mapping(strategy, "approach").get("direction_bucket"),
            _nested_mapping(features, "approach").get("direction_bucket"),
        ),
        (
            _nested_mapping(strategy, "orientation").get("relation"),
            _nested_mapping(features, "orientation").get("relation"),
        ),
        (
            _nested_mapping(strategy, "grasp").get("region"),
            _nested_mapping(features, "grasp").get("region"),
        ),
        (
            _nested_mapping(strategy, "grasp").get("semantic_part"),
            _nested_mapping(features, "grasp").get("semantic_part"),
        ),
        (
            _nested_mapping(strategy, "capability_evidence").get(
                "geometry_source_class"
            ),
            features.get("geometry_source_class"),
        ),
    ]
    unresolved = False
    for expected, actual in comparisons:
        expected_token = _token(expected)
        if expected_token in _UNKNOWN_TOKENS:
            continue
        actual_token = _token(actual)
        if actual_token in _UNKNOWN_TOKENS:
            unresolved = True
        elif actual_token != expected_token:
            return False
    return "unverified" if unresolved else True


def _first_value(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *keys: str,
) -> Any:
    for source in (primary, secondary):
        for key in keys:
            if key in source:
                return source.get(key)
    return None


def _nested_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key)
    if isinstance(nested, Mapping):
        return dict(nested)
    if key == "target_relation" and nested is not None:
        return {"relation": nested}
    return {}


def _target_region_free(candidate: Mapping[str, Any]) -> bool | None:
    if candidate.get("free") is True:
        return True
    if candidate.get("free") is False:
        return False
    occupied_by = candidate.get("occupied_by")
    if isinstance(occupied_by, (list, tuple, set, frozenset, dict)):
        return not bool(occupied_by)
    return None


def _candidate_target_relation(candidate: Mapping[str, Any]) -> str | None:
    value = candidate.get("placement_relation", candidate.get("target_relation"))
    if isinstance(value, Mapping):
        value = value.get("relation")
    return "center_of" if _token(value) == "center_of" else None


def _candidate_target_reference_role(
    candidate: Mapping[str, Any],
) -> str | None:
    for key in ("target_reference_role", "reference_role"):
        if key in candidate:
            role = str(candidate.get(key, "") or "").strip()
            return role or None
    return None


def _valid_place_attachment_geometry(candidate: Mapping[str, Any]) -> bool:
    if _token(candidate.get("attachment_transform_source")) != _PLACE_ATTACHMENT_SOURCE:
        return False
    holding_status = _token(candidate.get("holding_status"))
    transport_policy = _token(candidate.get("grasp_transport_policy"))
    return bool(
        (
            holding_status == "verified"
            and transport_policy in {"strict", "evidence_only"}
        )
        or (
            holding_status == "provisional_evidence_only"
            and transport_policy == "evidence_only"
        )
    )


def _candidate_reference_frame(
    candidate: Mapping[str, Any],
    *,
    geometry_source: str,
    orientation_relation: str,
) -> str:
    explicit = _token(candidate.get("reference_frame"))
    if explicit:
        if (
            explicit == "current_attachment"
            and _token(candidate.get("action_mode")) == "place"
            and not _valid_place_attachment_geometry(candidate)
        ):
            return "unknown"
        return explicit
    if orientation_relation == "preserve_current_attachment":
        return "current_attachment"
    if "principal_ax" in geometry_source:
        return "object_principal_axes"
    if "surface_normal" in geometry_source:
        return "support_normal"
    return "unknown"


def _approach_direction_bucket(
    candidate: Mapping[str, Any],
    *,
    geometry_policy: Mapping[str, Any],
) -> str:
    explicit = _token(candidate.get("approach_direction_bucket"))
    if explicit:
        return explicit
    approach = _xyz_prefix(candidate.get("approach_pose"))
    target = _xyz_prefix(candidate.get("ee_target_pose"))
    if approach is None or target is None:
        return "unknown"
    direction = [approach[index] - target[index] for index in range(3)]
    norm = math.sqrt(sum(item * item for item in direction))
    direction_policy = _nested_mapping(
        geometry_policy,
        "direction_classification",
    )
    if not math.isfinite(norm) or norm <= float(direction_policy["epsilon"]):
        return "unknown"
    vertical_ratio = direction[2] / norm
    if vertical_ratio >= float(direction_policy["fallback_above_vertical_ratio_min"]):
        return "above"
    if vertical_ratio <= float(direction_policy["fallback_below_vertical_ratio_max"]):
        return "below"
    if abs(vertical_ratio) <= float(
        direction_policy["fallback_lateral_vertical_ratio_abs_max"]
    ):
        return "lateral"
    return "oblique"


def _geometry_source_class(value: str) -> str:
    if "oracle" in value:
        return "oracle"
    if value.startswith("rgbd") or "rgbd_" in value:
        return "rgbd_observed"
    if value.startswith("runtime") or "relational" in value:
        return "runtime_relational"
    return value if value in {"rgbd_observed", "runtime_relational"} else "unknown"


def _xyz_prefix(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        xyz = tuple(float(value[index]) for index in range(3))
    except (TypeError, ValueError):
        return None
    return xyz if all(math.isfinite(item) for item in xyz) else None


def _positive_finite(value: Any) -> bool:
    number = _nonnegative_finite(value)
    return number is not None and number > 0.0


def _nonnegative_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token(value: Any) -> str:
    return str(value or "").strip().lower()

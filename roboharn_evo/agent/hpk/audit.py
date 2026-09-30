from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Iterator, Literal

from roboharn_evo.agent.hpk.runtime_policy import HPKRuntimePolicy
from roboharn_evo.agent.hpk.schemas import (
    GEOMETRIC_STRATEGY_SCHEMA,
    HPK_GEOMETRY_SCORE_PROFILE,
    OBSERVE_AUDIT_PROFILE,
    ConditionV1,
    EvidenceV1,
    ObserveAuditRecord,
    TaskStrategyV1,
    canonical_json_bytes,
    stable_content_id,
)


HPK_PRIVATE_RANKING_AUDIT_PROFILE = "hpk_private_ranking_audit/v1"
LEGACY_RANKING_MAPPING_PERSISTENCE_STATUS = (
    "legacy_in_memory_mapping_not_approved_for_p0c_live_persistence"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_RANKING_FIELDS = frozenset(
    {
        "profile",
        "private_ranking_audit_id",
        "public_usage_audit_id",
        "snapshot_id",
        "snapshot_manifest_sha256",
        "selected_entry_id",
        "selected_geometric_strategy_id",
        "scoring_profile",
        "candidate_ids_before",
        "candidate_rank_before",
        "baseline_selected_candidate_private_ref",
        "candidate_ids_after",
        "candidate_rank_after",
        "selected_candidate_private_ref",
        "geometric_compliance",
        "all_hard_mismatch_behavior",
        "zero_compliant_candidates",
        "candidate_rejection_details",
        "hpk_scores_after",
        "hpk_score_components_after",
    }
)


def _content_id(value: Any, *, prefix: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        rf"{re.escape(prefix)}_[0-9a-f]{{64}}", value
    ):
        raise ValueError(f"{field} must be a canonical {prefix}_<sha256> ID")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _candidate_private_refs(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field}[{index}] must be a non-empty private ref")
        result.append(item.strip())
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique private refs")
    return result


def _rank_array(value: Any, *, field: str, expected_length: int) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{field} must align one-to-one with its candidate array")
    result: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{field}[{index}] must be a nonnegative integer")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique ranks")
    return result


def _private_ref_or_none(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be null or a non-empty private ref")
    return value.strip()


def _compliance(value: Any) -> bool | Literal["unverified"]:
    if isinstance(value, bool):
        return value
    if value == "unverified":
        return "unverified"
    raise ValueError("geometric_compliance must be boolean or 'unverified'")


def _finite_scores(value: Any, *, field: str, expected_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{field} must align one-to-one with candidates after ranking")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{field}[{index}] must be a finite number")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field}[{index}] must be a finite number")
        result.append(number)
    return result


def _score_components(value: Any, *, expected_length: int) -> list[dict[str, float]]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(
            "hpk_score_components_after must align one-to-one with candidates after ranking"
        )
    result: list[dict[str, float]] = []
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, Mapping):
            raise TypeError(
                f"hpk_score_components_after[{row_index}] must be an object"
            )
        row: dict[str, float] = {}
        for key, raw_value in raw_row.items():
            if not isinstance(key, str) or not key:
                raise ValueError("HPK score component names must be non-empty strings")
            if isinstance(raw_value, bool):
                raise ValueError(f"HPK score component {key!r} must be finite")
            try:
                number = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"HPK score component {key!r} must be finite") from exc
            if not math.isfinite(number):
                raise ValueError(f"HPK score component {key!r} must be finite")
            row[key] = number
        result.append(row)
    return result


class PrivateRankingAuditV1(Mapping[str, Any]):
    """Immutable validated private record; never transferable or planner-visible."""

    __slots__ = ("_data",)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("PrivateRankingAuditV1 must be an object")
        data = copy.deepcopy(dict(payload))
        # 使用 HPK 编码器验证 JSON 与有限数值。
        canonical_json_bytes(data)
        unknown = sorted(set(data) - _PRIVATE_RANKING_FIELDS)
        missing = sorted(_PRIVATE_RANKING_FIELDS - set(data))
        if missing:
            raise ValueError(
                "PrivateRankingAuditV1 missing required field(s): " + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "PrivateRankingAuditV1 contains unknown field(s): " + ", ".join(unknown)
            )
        if data["profile"] != HPK_PRIVATE_RANKING_AUDIT_PROFILE:
            raise ValueError("unsupported private ranking audit profile")
        _content_id(
            data["private_ranking_audit_id"],
            prefix="afkprivrank",
            field="private_ranking_audit_id",
        )
        _content_id(
            data["public_usage_audit_id"],
            prefix="afkaudit",
            field="public_usage_audit_id",
        )
        _content_id(data["snapshot_id"], prefix="afksnap", field="snapshot_id")
        _sha256(data["snapshot_manifest_sha256"], field="snapshot_manifest_sha256")
        _content_id(
            data["selected_entry_id"],
            prefix="afkentry",
            field="selected_entry_id",
        )
        _content_id(
            data["selected_geometric_strategy_id"],
            prefix="afkz",
            field="selected_geometric_strategy_id",
        )
        if data["scoring_profile"] not in {HPK_GEOMETRY_SCORE_PROFILE, "afk_geometry_score/v1"}:
            raise ValueError(f"scoring_profile must be {HPK_GEOMETRY_SCORE_PROFILE!r}")

        before = _candidate_private_refs(
            data["candidate_ids_before"], field="candidate_ids_before"
        )
        after = _candidate_private_refs(
            data["candidate_ids_after"], field="candidate_ids_after"
        )
        if not set(after).issubset(before):
            raise ValueError(
                "candidate_ids_after must be a subset of candidate_ids_before"
            )
        _rank_array(
            data["candidate_rank_before"],
            field="candidate_rank_before",
            expected_length=len(before),
        )
        _rank_array(
            data["candidate_rank_after"],
            field="candidate_rank_after",
            expected_length=len(after),
        )
        baseline = _private_ref_or_none(
            data["baseline_selected_candidate_private_ref"],
            field="baseline_selected_candidate_private_ref",
        )
        selected = _private_ref_or_none(
            data["selected_candidate_private_ref"],
            field="selected_candidate_private_ref",
        )
        if baseline is not None and baseline not in before:
            raise ValueError(
                "baseline_selected_candidate_private_ref must occur in candidate_ids_before"
            )
        if selected is not None and selected not in after:
            raise ValueError(
                "selected_candidate_private_ref must occur in candidate_ids_after"
            )
        _compliance(data["geometric_compliance"])
        if data["all_hard_mismatch_behavior"] != "fail_closed":
            raise ValueError("P0-C private ranking audit requires fail_closed behavior")
        if not isinstance(data["zero_compliant_candidates"], bool):
            raise TypeError("zero_compliant_candidates must be a boolean")
        if data["zero_compliant_candidates"] and after:
            raise ValueError(
                "fail-closed zero-compliant ranking cannot contain candidates after ranking"
            )

        rejection_details = data["candidate_rejection_details"]
        if not isinstance(rejection_details, list) or len(rejection_details) != len(
            before
        ):
            raise ValueError(
                "candidate_rejection_details must align with candidates before ranking"
            )
        rejection_refs: list[str] = []
        for index, raw_detail in enumerate(rejection_details):
            if not isinstance(raw_detail, Mapping) or set(raw_detail) != {
                "candidate_private_ref",
                "reasons",
            }:
                raise ValueError(
                    f"candidate_rejection_details[{index}] has an invalid shape"
                )
            ref = _private_ref_or_none(
                raw_detail["candidate_private_ref"],
                field=f"candidate_rejection_details[{index}].candidate_private_ref",
            )
            assert ref is not None
            rejection_refs.append(ref)
            reasons = raw_detail["reasons"]
            if not isinstance(reasons, list) or any(
                not isinstance(reason, str) or not reason.strip() for reason in reasons
            ):
                raise ValueError(
                    f"candidate_rejection_details[{index}].reasons must be strings"
                )
            if len(reasons) != len(set(reasons)):
                raise ValueError(
                    f"candidate_rejection_details[{index}].reasons must be unique"
                )
        if rejection_refs != before:
            raise ValueError(
                "candidate_rejection_details must preserve candidate_ids_before order"
            )

        _finite_scores(
            data["hpk_scores_after"],
            field="hpk_scores_after",
            expected_length=len(after),
        )
        _score_components(
            data["hpk_score_components_after"], expected_length=len(after)
        )
        identity_payload = dict(data)
        identity_payload.pop("private_ranking_audit_id")
        expected_id = stable_content_id("afkprivrank", identity_payload)
        if data["private_ranking_audit_id"] != expected_id:
            raise ValueError(
                f"private_ranking_audit_id content mismatch; expected {expected_id}"
            )
        self._data = data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrivateRankingAuditV1":
        return cls(payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_canonical_json_line(self) -> bytes:
        return canonical_json_bytes(self._data) + b"\n"

    @property
    def stable_id(self) -> str:
        return str(self._data["private_ranking_audit_id"])

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class PrivateRankingAuditSink:
    """Trusted one-decision P0-C sink for validated private ranking output.

    The sink intentionally owns no caller-provided path.  A live/CPU trace writer
    may persist :meth:`to_canonical_json_line` only inside its already validated
    private artifact root.  Arbitrary mutable mappings remain a P0-B diagnostic
    compatibility API and are never approved as this persistence boundary.
    """

    __slots__ = (
        "_snapshot_id",
        "_snapshot_manifest_sha256",
        "_selected_entry_id",
        "_selected_geometric_strategy_id",
        "_public_usage_audit_id",
        "_draft",
        "_record",
    )

    def __init__(
        self,
        *,
        snapshot_id: str,
        snapshot_manifest_sha256: str,
        selected_entry_id: str,
        selected_geometric_strategy_id: str,
        public_usage_audit_id: str | None = None,
    ) -> None:
        self._snapshot_id = _content_id(
            snapshot_id, prefix="afksnap", field="snapshot_id"
        )
        self._snapshot_manifest_sha256 = _sha256(
            snapshot_manifest_sha256, field="snapshot_manifest_sha256"
        )
        self._selected_entry_id = _content_id(
            selected_entry_id, prefix="afkentry", field="selected_entry_id"
        )
        self._selected_geometric_strategy_id = _content_id(
            selected_geometric_strategy_id,
            prefix="afkz",
            field="selected_geometric_strategy_id",
        )
        self._public_usage_audit_id = None
        if public_usage_audit_id is not None:
            self._public_usage_audit_id = _content_id(
                public_usage_audit_id,
                prefix="afkaudit",
                field="public_usage_audit_id",
            )
        self._draft: dict[str, Any] | None = None
        self._record: PrivateRankingAuditV1 | None = None

    @property
    def live_persistence_approved(self) -> bool:
        return True

    @property
    def accepted_low_level_ranking(self) -> bool:
        return self._draft is not None

    @property
    def finalized(self) -> bool:
        return self._record is not None

    @property
    def behavior_changed(self) -> bool:
        if self._draft is None:
            raise RuntimeError("private ranking sink has no accepted ranking")
        return (
            self._draft["baseline_selected_candidate_private_ref"]
            != self._draft["selected_candidate_private_ref"]
        )

    @property
    def geometric_compliance(self) -> bool | Literal["unverified"]:
        if self._draft is None:
            raise RuntimeError("private ranking sink has no accepted ranking")
        return _compliance(self._draft["geometric_compliance"])

    @property
    def zero_compliant_candidates(self) -> bool:
        if self._draft is None:
            raise RuntimeError("private ranking sink has no accepted ranking")
        return bool(self._draft["zero_compliant_candidates"])

    @property
    def selected_candidate_private_ref(self) -> str | None:
        if self._draft is None:
            raise RuntimeError("private ranking sink has no accepted ranking")
        return _private_ref_or_none(
            self._draft["selected_candidate_private_ref"],
            field="selected_candidate_private_ref",
        )

    @property
    def private_ranking_audit_id(self) -> str:
        if self._record is None:
            raise RuntimeError("private ranking sink is not finalized")
        return str(self._record["private_ranking_audit_id"])

    @property
    def record(self) -> PrivateRankingAuditV1:
        if self._record is None:
            raise RuntimeError("private ranking sink is not finalized")
        return PrivateRankingAuditV1.from_dict(self._record.to_dict())

    def accept_low_level_ranking(
        self,
        payload: Mapping[str, Any],
        *,
        baseline_selected_candidate_private_ref: str | None = None,
    ) -> None:
        """Validate and accept exactly one P0-B low-level ranking output."""

        if self._draft is not None:
            raise RuntimeError("private ranking sink accepts exactly one ranking")
        if not isinstance(payload, Mapping):
            raise TypeError("low-level ranking audit must be an object")
        raw = copy.deepcopy(dict(payload))
        canonical_json_bytes(raw)
        required = {
            "candidate_ids_before",
            "candidate_rank_before",
            "candidate_ids_after",
            "candidate_rank_after",
            "selected_geometric_hpk_id",
            "selected_candidate_id",
            "geometric_compliance",
            "private_ranking_diagnostics",
            "persistence_status",
        }
        if set(raw) != required:
            raise ValueError("low-level ranking audit has an unexpected shape")
        if raw["persistence_status"] != LEGACY_RANKING_MAPPING_PERSISTENCE_STATUS:
            raise ValueError("low-level mapping persistence status is not recognized")
        before = _candidate_private_refs(
            raw["candidate_ids_before"], field="candidate_ids_before"
        )
        after = _candidate_private_refs(
            raw["candidate_ids_after"], field="candidate_ids_after"
        )
        if not set(after).issubset(before):
            raise ValueError(
                "ranked candidates must be a subset of baseline candidates"
            )
        before_ranks = _rank_array(
            raw["candidate_rank_before"],
            field="candidate_rank_before",
            expected_length=len(before),
        )
        after_ranks = _rank_array(
            raw["candidate_rank_after"],
            field="candidate_rank_after",
            expected_length=len(after),
        )
        if raw["selected_geometric_hpk_id"] != self._selected_geometric_strategy_id:
            raise ValueError("low-level ranking used a different geometric strategy")
        selected = _private_ref_or_none(
            raw["selected_candidate_id"], field="selected_candidate_id"
        )
        if selected is not None and selected not in after:
            raise ValueError("selected_candidate_id must occur after ranking")
        baseline = _private_ref_or_none(
            baseline_selected_candidate_private_ref,
            field="baseline_selected_candidate_private_ref",
        )
        if baseline is None and before:
            baseline = before[0]
        if baseline is not None and baseline not in before:
            raise ValueError("baseline selected private ref must occur before ranking")

        diagnostics = raw["private_ranking_diagnostics"]
        if not isinstance(diagnostics, Mapping):
            raise TypeError("private_ranking_diagnostics must be an object")
        expected_diagnostic_fields = {
            "privacy",
            "scoring_profile",
            "geometric_strategy_schema",
            "all_hard_mismatch_behavior",
            "all_hard_mismatch",
            "zero_compliant_candidates",
            "baseline_fallback_used",
            "baseline_rank_by_after",
            "hard_constraint_results_before",
            "geometric_compliance_before",
            "candidate_rejections_before",
            "hpk_soft_scores_after",
            "hpk_soft_score_components_after",
        }
        if set(diagnostics) != expected_diagnostic_fields:
            raise ValueError("private_ranking_diagnostics has an unexpected shape")
        if diagnostics.get("privacy") != "private_runtime_candidate_identity":
            raise ValueError("private ranking privacy marker is invalid")
        if diagnostics.get("scoring_profile") not in {HPK_GEOMETRY_SCORE_PROFILE, "afk_geometry_score/v1"}:
            raise ValueError("low-level ranking used an unsupported score profile")
        if diagnostics.get("geometric_strategy_schema") not in {
            GEOMETRIC_STRATEGY_SCHEMA, "tcm/afk/geometric_strategy/v1"
        }:
            raise ValueError("low-level ranking used an unsupported strategy schema")
        if diagnostics.get("all_hard_mismatch_behavior") != "fail_closed":
            raise ValueError("P0-C live ranking requires fail_closed behavior")
        for field in (
            "all_hard_mismatch",
            "baseline_fallback_used",
        ):
            if not isinstance(diagnostics.get(field), bool):
                raise TypeError(f"{field} must be a boolean")
        if diagnostics.get("baseline_fallback_used") is not False:
            raise ValueError("P0-C live ranking cannot persist baseline fallback")
        zero_compliant = diagnostics.get("zero_compliant_candidates")
        if not isinstance(zero_compliant, bool):
            raise TypeError("zero_compliant_candidates must be a boolean")
        _rank_array(
            diagnostics.get("baseline_rank_by_after"),
            field="baseline_rank_by_after",
            expected_length=len(after),
        )
        hard_results = diagnostics.get("hard_constraint_results_before")
        if not isinstance(hard_results, list) or len(hard_results) != len(before):
            raise ValueError(
                "hard-constraint results must align with baseline candidates"
            )
        for row in hard_results:
            if not isinstance(row, Mapping) or any(
                not isinstance(name, str) or not isinstance(result, bool)
                for name, result in row.items()
            ):
                raise ValueError("hard-constraint result rows must contain booleans")
        compliance_before = diagnostics.get("geometric_compliance_before")
        if not isinstance(compliance_before, list) or len(compliance_before) != len(
            before
        ):
            raise ValueError("geometric compliance must align with baseline candidates")
        for value in compliance_before:
            _compliance(value)
        rejections = diagnostics.get("candidate_rejections_before")
        if not isinstance(rejections, list) or len(rejections) != len(before):
            raise ValueError("candidate rejections must align with baseline candidates")
        rejection_details: list[dict[str, Any]] = []
        for candidate_ref, raw_reasons in zip(before, rejections, strict=True):
            if not isinstance(raw_reasons, list) or any(
                not isinstance(reason, str) or not reason.strip()
                for reason in raw_reasons
            ):
                raise ValueError("candidate rejection reasons must be strings")
            reasons = [reason.strip() for reason in raw_reasons]
            if len(reasons) != len(set(reasons)):
                raise ValueError("candidate rejection reasons must be unique")
            rejection_details.append(
                {"candidate_private_ref": candidate_ref, "reasons": reasons}
            )
        scores = _finite_scores(
            diagnostics.get("hpk_soft_scores_after"),
            field="hpk_soft_scores_after",
            expected_length=len(after),
        )
        components = _score_components(
            diagnostics.get("hpk_soft_score_components_after"),
            expected_length=len(after),
        )
        self._draft = {
            "snapshot_id": self._snapshot_id,
            "snapshot_manifest_sha256": self._snapshot_manifest_sha256,
            "selected_entry_id": self._selected_entry_id,
            "selected_geometric_strategy_id": self._selected_geometric_strategy_id,
            "scoring_profile": HPK_GEOMETRY_SCORE_PROFILE,
            "candidate_ids_before": before,
            "candidate_rank_before": before_ranks,
            "baseline_selected_candidate_private_ref": baseline,
            "candidate_ids_after": after,
            "candidate_rank_after": after_ranks,
            "selected_candidate_private_ref": selected,
            "geometric_compliance": _compliance(raw["geometric_compliance"]),
            "all_hard_mismatch_behavior": "fail_closed",
            "zero_compliant_candidates": zero_compliant,
            "candidate_rejection_details": rejection_details,
            "hpk_scores_after": scores,
            "hpk_score_components_after": components,
        }
        if self._public_usage_audit_id is not None:
            self._finalize()

    def bind_public_usage_audit(self, public_usage_audit_id: str) -> None:
        """Bind a validated ID; live callers should prefer the record method."""

        validated = _content_id(
            public_usage_audit_id,
            prefix="afkaudit",
            field="public_usage_audit_id",
        )
        if (
            self._public_usage_audit_id is not None
            and self._public_usage_audit_id != validated
        ):
            raise ValueError(
                "private ranking sink is already bound to another public audit"
            )
        self._public_usage_audit_id = validated
        if self._draft is not None and self._record is None:
            self._finalize()

    def bind_public_usage_audit_record(self, value: Mapping[str, Any]) -> None:
        """Validate public/private linkage before finalizing the private record."""

        from roboharn_evo.agent.hpk.schemas import AuditV1

        audit = value if isinstance(value, AuditV1) else AuditV1.from_dict(value)
        if self._draft is None:
            raise RuntimeError("private ranking sink has no accepted ranking")
        expected = {
            "retrieval_stage": "post_binding_geometry",
            "snapshot_id": self._snapshot_id,
            "snapshot_manifest_sha256": self._snapshot_manifest_sha256,
            "selected_entry_id": self._selected_entry_id,
            "selected_geometric_strategy_id": self._selected_geometric_strategy_id,
            "scoring_profile": HPK_GEOMETRY_SCORE_PROFILE,
            "behavior_changed": self.behavior_changed,
            "geometric_compliance": self.geometric_compliance,
        }
        mismatches = [
            key
            for key, expected_value in expected.items()
            if audit[key] != expected_value
        ]
        if mismatches:
            raise ValueError(
                "public/private HPK usage audit mismatch: " + ", ".join(mismatches)
            )
        self.bind_public_usage_audit(audit.stable_id)

    def _finalize(self) -> None:
        if self._draft is None or self._public_usage_audit_id is None:
            raise RuntimeError(
                "private ranking sink requires ranking and public audit binding"
            )
        payload = {
            "profile": HPK_PRIVATE_RANKING_AUDIT_PROFILE,
            "public_usage_audit_id": self._public_usage_audit_id,
            **copy.deepcopy(self._draft),
        }
        payload["private_ranking_audit_id"] = stable_content_id("afkprivrank", payload)
        self._record = PrivateRankingAuditV1.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        if self._record is None:
            raise RuntimeError("private ranking sink is not finalized")
        return self._record.to_dict()

    def to_canonical_json_line(self) -> bytes:
        if self._record is None:
            raise RuntimeError("private ranking sink is not finalized")
        return self._record.to_canonical_json_line()


def private_ranking_sink_is_live_approved(value: Any) -> bool:
    """Return true only for the dedicated P0-C persistence boundary."""

    return isinstance(value, PrivateRankingAuditSink)


def _policy(value: HPKRuntimePolicy | Mapping[str, Any]) -> HPKRuntimePolicy:
    if isinstance(value, HPKRuntimePolicy):
        return value
    return HPKRuntimePolicy.from_mapping(value)


def _candidate_ids(value: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            candidate_id = str(item.get("candidate_id", "") or "").strip()
        else:
            candidate_id = str(item or "").strip()
        if not candidate_id:
            raise ValueError("candidate audit entries require a private candidate ID")
        result.append(candidate_id)
    if len(result) != len(set(result)):
        raise ValueError("candidate audit IDs must be unique")
    return result


def build_observe_audit(
    policy: HPKRuntimePolicy | Mapping[str, Any],
    *,
    condition: ConditionV1 | None = None,
    task_strategy: TaskStrategyV1 | None = None,
    candidates: Sequence[Any] = (),
    selected_candidate_id: str | None = None,
    motion_status: str = "unknown",
    effect_verdict: str = "unverified",
    evidence: EvidenceV1 | None = None,
) -> ObserveAuditRecord | None:
    """Return no sidecar in off mode and an inert trace sidecar in observe."""

    runtime_policy = _policy(policy)
    if not runtime_policy.enabled:
        return None
    ids = _candidate_ids(candidates)
    ranks = list(range(len(ids)))
    selected = str(selected_candidate_id or "").strip() or None
    recorded_verdict = (
        str(evidence["verdict"]) if evidence is not None else str(effect_verdict)
    )
    return ObserveAuditRecord.from_dict(
        {
            "profile": OBSERVE_AUDIT_PROFILE,
            "mode": "observe",
            "condition_id": condition.stable_id if condition is not None else None,
            "task_strategy_id": (
                task_strategy.stable_id if task_strategy is not None else None
            ),
            "retrieved_task_hpk_ids": [],
            "selected_geometric_hpk_id": None,
            "candidate_ids_before": ids,
            "candidate_rank_before": ranks,
            "candidate_ids_after": list(ids),
            "candidate_rank_after": list(ranks),
            "selected_candidate_id": selected,
            "geometric_compliance": "unverified",
            "motion_status": str(motion_status),
            "effect_verdict": recorded_verdict,
            "evidence_id": evidence["evidence_id"] if evidence is not None else None,
            "parent_snapshot_id": None,
            "child_snapshot_id": None,
        }
    )


def build_public_usage_audit(
    *,
    retrieval_stage: str,
    snapshot_id: str,
    snapshot_manifest_sha256: str,
    retrieved_entry_ids: Sequence[str] = (),
    selected_entry_id: str | None = None,
    condition_id: str | None = None,
    task_strategy_id: str | None = None,
    selected_geometric_strategy_id: str | None = None,
    match_reason: str | None = None,
    rejection_reasons: Sequence[str] = (),
    planner_context_sha256: str | None = None,
    planner_context_rendered: bool = False,
    planner_usage: str = "not_injected",
    behavior_changed: bool | Literal["unverified"] = "unverified",
    behavior_change_channel: str = "unverified",
    geometric_compliance: bool | Literal["unverified"] = "unverified",
    scoring_profile: str | None = None,
    source_disclosure: Mapping[str, Any] | None = None,
) -> Any:
    """Build the strict, candidate-free public ``AuditV1`` record.

    Imports stay local so P0-A users that only construct observe records do not
    pull the P0-C schema surface into their import path.  ``AuditV1`` performs
    the final cross-field, enum, content-ID, and recursive privacy validation.
    """

    from roboharn_evo.agent.hpk.schemas import (
        AUDIT_REJECTION_REASON_ORDER,
        AUDIT_SCHEMA,
        AuditV1,
        audit_id_for,
    )

    rejection_order = AUDIT_REJECTION_REASON_ORDER
    supplied_rejections = list(rejection_reasons)
    unknown_rejections = sorted(set(supplied_rejections) - set(rejection_order))
    if unknown_rejections:
        raise ValueError(
            "unknown HPK public rejection reason(s): " + ", ".join(unknown_rejections)
        )
    ordered_rejections = [
        reason for reason in rejection_order if reason in supplied_rejections
    ]
    if source_disclosure is not None and not isinstance(source_disclosure, Mapping):
        raise TypeError("source_disclosure must be an object or None")
    if retrieval_stage == "post_binding_geometry" and selected_entry_id is not None:
        missing_binding_ids = [
            name
            for name, value in (
                ("condition_id", condition_id),
                ("task_strategy_id", task_strategy_id),
                (
                    "selected_geometric_strategy_id",
                    selected_geometric_strategy_id,
                ),
            )
            if value is None
        ]
        if missing_binding_ids:
            raise ValueError(
                "selected geometry audit requires current binding ID(s): "
                + ", ".join(missing_binding_ids)
            )
    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "mode": "static",
        "retrieval_stage": retrieval_stage,
        "snapshot_id": snapshot_id,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "retrieved_entry_ids": list(retrieved_entry_ids),
        "selected_entry_id": selected_entry_id,
        "condition_id": condition_id,
        "task_strategy_id": task_strategy_id,
        "selected_geometric_strategy_id": selected_geometric_strategy_id,
        "match_reason": match_reason,
        "rejection_reasons": ordered_rejections,
        "planner_context_sha256": planner_context_sha256,
        "planner_context_rendered": planner_context_rendered,
        "planner_usage": planner_usage,
        "behavior_changed": behavior_changed,
        "behavior_change_channel": behavior_change_channel,
        "geometric_compliance": geometric_compliance,
        "scoring_profile": scoring_profile,
        "source_disclosure": (
            copy.deepcopy(dict(source_disclosure))
            if isinstance(source_disclosure, Mapping)
            else None
        ),
    }
    payload["usage_audit_id"] = audit_id_for(payload)
    return AuditV1.from_dict(payload)


__all__ = [
    "HPK_GEOMETRY_SCORE_PROFILE",
    "HPK_PRIVATE_RANKING_AUDIT_PROFILE",
    "LEGACY_RANKING_MAPPING_PERSISTENCE_STATUS",
    "PrivateRankingAuditSink",
    "PrivateRankingAuditV1",
    "build_observe_audit",
    "build_public_usage_audit",
    "private_ranking_sink_is_live_approved",
]

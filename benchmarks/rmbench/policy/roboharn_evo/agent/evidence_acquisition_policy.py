from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Sequence


class EvidenceSituation(str, Enum):
    """Why the runtime is considering another observation."""

    TRANSIENT_BACKEND_RETRY = "transient_backend_retry"
    OCCLUSION = "occlusion"
    STRONG_GRASP_EVIDENCE = "strong_grasp_evidence"
    PROVEN_FAILED_GRASP = "proven_failed_grasp"
    AMBIGUOUS_GRASP = "ambiguous_grasp"
    RELEASE_STABILITY = "release_stability"
    TEMPORAL_IDENTITY_REPAIR = "temporal_identity_repair"


class InformationAction(str, Enum):
    """The information-changing action recommended by the policy."""

    RETRY_TRANSIENT_BACKEND = "retry_transient_backend"
    CHANGE_CAMERA_OR_ESCALATE_BACKEND_FAILURE = (
        "change_camera_or_escalate_backend_failure"
    )
    STATIONARY_REOBSERVE_ONCE = "stationary_reobserve_once"
    CHANGE_VIEWPOINT_OR_CLEAR_OCCLUSION = (
        "change_viewpoint_or_clear_occlusion"
    )
    ACCEPT_STRONG_GRASP_EVIDENCE = (
        "accept_strong_grasp_evidence"
    )
    ACQUIRE_INDEPENDENT_CAMERA_VIEW = (
        "acquire_independent_camera_view"
    )
    PERFORM_CONTROLLED_RETURN_AND_REGRASP = (
        "perform_controlled_return_and_regrasp"
    )
    PERFORM_FAILED_GRASP_CLEARANCE = (
        "perform_failed_grasp_clearance"
    )
    ACQUIRE_DISTINCT_RELEASE_CAPTURE = (
        "acquire_distinct_release_capture"
    )
    EVALUATE_RELEASE_STABILITY = "evaluate_release_stability"
    CHANGE_VIEWPOINT_OR_PHYSICAL_RECHECK = (
        "change_viewpoint_or_physical_recheck"
    )
    ACQUIRE_DISTINCT_IDENTITY_CAPTURE = (
        "acquire_distinct_identity_capture"
    )
    EVALUATE_IDENTITY_REPAIR = "evaluate_identity_repair"


@dataclass(frozen=True, slots=True)
class EvidenceAcquisitionContext:
    """Physical/perceptual state relevant to evidence acquisition.

    ``skill_id`` is intentionally accepted for audit/integration convenience
    but is not part of :class:`EvidenceStateKey`.  The policy must live at the
    agent/runtime level so replacing a planner skill cannot replenish a
    same-state observation budget.

    ``physical_action_token`` should change after an executed physical action
    even when that action returns the end effector to the same quantized pose.
    ``viewpoint_token`` serves the analogous purpose for a calibrated camera
    whose extrinsics can change without changing ``camera_set``.
    """

    track_id: str
    arm: str
    manipulation_phase: str
    grasp_attempt_nonce: str = ""
    ee_pose: Sequence[float] | None = None
    camera_set: Sequence[str] = ()
    capture_id: str | int | None = None
    skill_id: str = ""
    physical_action_token: str | int | None = None
    viewpoint_token: str | int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceStateKey:
    """Identity of a stationary evidence-acquisition state."""

    track_id: str
    arm: str
    manipulation_phase: str
    grasp_attempt_nonce: str
    quantized_ee_pose: tuple[int, ...]
    camera_set: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "arm": self.arm,
            "manipulation_phase": self.manipulation_phase,
            "grasp_attempt_nonce": self.grasp_attempt_nonce,
            "quantized_ee_pose": list(self.quantized_ee_pose),
            "camera_set": list(self.camera_set),
        }


@dataclass(slots=True)
class _BudgetRecord:
    stationary_reobserves_used: int = 0
    distinct_release_captures: set[str] = field(
        default_factory=set
    )
    awaiting_distinct_release_capture: bool = False


@dataclass(slots=True)
class _ActiveTransaction:
    state_key: EvidenceStateKey
    physical_action_token: str = ""
    viewpoint_token: str = ""


class EvidenceAcquisitionPolicy:
    """Bound stationary observation retries by physical state.

    A same-state re-observation can recover a transient backend failure or add
    one genuinely new release-stability sample.  It cannot remove a fixed
    occluder, create a second camera view, or disambiguate a grasp forever.
    This policy makes that distinction explicit and prevents a skill/replan
    transition from silently renewing the budget.
    """

    def __init__(
        self,
        *,
        translation_quantum_m: float = 0.005,
        quaternion_quantum: float = 0.02,
        default_stationary_reobserve_limit: int = 1,
        release_distinct_capture_limit: int = 2,
    ) -> None:
        if translation_quantum_m <= 0.0:
            raise ValueError("translation_quantum_m must be positive")
        if quaternion_quantum <= 0.0:
            raise ValueError("quaternion_quantum must be positive")
        if default_stationary_reobserve_limit < 0:
            raise ValueError(
                "default_stationary_reobserve_limit must be nonnegative"
            )
        if release_distinct_capture_limit < 2:
            raise ValueError(
                "release_distinct_capture_limit must be at least two"
            )
        self.translation_quantum_m = float(
            translation_quantum_m
        )
        self.quaternion_quantum = float(quaternion_quantum)
        self.default_stationary_reobserve_limit = int(
            default_stationary_reobserve_limit
        )
        self.release_distinct_capture_limit = int(
            release_distinct_capture_limit
        )
        # Release verification normally starts with one capture already in
        # hand.  A second allowance also lets a caller recover when that first
        # capture has no usable identity token, without permitting a loop.
        self.release_stationary_reobserve_limit = max(
            self.default_stationary_reobserve_limit,
            self.release_distinct_capture_limit,
        )
        self._records: dict[EvidenceStateKey, _BudgetRecord] = {}
        self._active_transactions: dict[
            tuple[str, str, str, str], _ActiveTransaction
        ] = {}
        # Re-observation refreshes the whole camera frame, not one planner
        # label.  Track, arm, phase, and skill changes therefore cannot renew
        # the allowance while the physical state and viewpoint are unchanged.
        self._stationary_scene_reobserves: dict[
            tuple[str, str, tuple[str, ...]], int
        ] = {}

    def reset(self) -> None:
        """Forget observation budgets at an episode boundary.

        Skill replacement and replanning deliberately do not call this
        method.  The budget belongs to the physical episode state, not to a
        planner skill name.
        """

        self._records.clear()
        self._active_transactions.clear()
        self._stationary_scene_reobserves.clear()

    def stationary_scene_budget_status(
        self,
        context: EvidenceAcquisitionContext,
        *,
        situation: EvidenceSituation | str,
    ) -> dict[str, Any]:
        """Inspect the scene-wide retry budget without consuming it.

        This is used before asking the recovery planner for another plan.  If
        the current physical state has already spent its observation budget,
        the runtime can omit ``reobserve_scene`` from the advertised tools and
        ask the planner to continue with a physical action instead.  A real
        physical action or viewpoint change produces a new key automatically.
        """

        normalized_situation = EvidenceSituation(situation)
        limit = (
            self.release_stationary_reobserve_limit
            if normalized_situation
            in {
                EvidenceSituation.RELEASE_STABILITY,
                EvidenceSituation.TEMPORAL_IDENTITY_REPAIR,
            }
            else self.default_stationary_reobserve_limit
        )
        scene_key = self._stationary_scene_key(context)
        used = self._stationary_scene_reobserves.get(scene_key, 0)
        return {
            "stationary_scene_reobserves_used": int(used),
            "stationary_scene_reobserve_limit": int(limit),
            "stationary_scene_reobserve_budget_exhausted": bool(
                limit <= 0 or used >= limit
            ),
        }

    def state_key(
        self,
        context: EvidenceAcquisitionContext,
    ) -> EvidenceStateKey:
        """Return the skill-independent, quantized stationary-state key."""

        track_id = _required_text(context.track_id, "track_id")
        arm = _required_text(context.arm, "arm").lower()
        phase = _required_text(
            context.manipulation_phase,
            "manipulation_phase",
        ).lower()
        nonce = str(context.grasp_attempt_nonce or "").strip()
        return EvidenceStateKey(
            track_id=track_id,
            arm=arm,
            manipulation_phase=phase,
            grasp_attempt_nonce=nonce,
            quantized_ee_pose=_quantize_pose(
                context.ee_pose,
                translation_quantum_m=(
                    self.translation_quantum_m
                ),
                quaternion_quantum=self.quaternion_quantum,
            ),
            camera_set=_normalize_camera_set(context.camera_set),
        )

    def decide(
        self,
        context: EvidenceAcquisitionContext,
        *,
        situation: EvidenceSituation | str,
        evidence_sufficient: bool = False,
    ) -> dict[str, Any]:
        """Consume/query the budget and recommend the next information action.

        An allowed decision consumes the stationary re-observation allowance
        immediately.  This prevents multiple planner/tool rounds from issuing
        the same request before a new capture arrives.
        """

        normalized_situation = EvidenceSituation(situation)
        key, record = self._record_for_context(context)

        if normalized_situation is EvidenceSituation.OCCLUSION:
            return self._decision(
                key=key,
                record=record,
                situation=normalized_situation,
                allow_reobserve=False,
                next_action=(
                    InformationAction.CHANGE_VIEWPOINT_OR_CLEAR_OCCLUSION
                ),
                exhausted_reason=(
                    "same_state_reobserve_cannot_clear_occlusion"
                ),
                stationary_limit=0,
            )

        if normalized_situation is EvidenceSituation.STRONG_GRASP_EVIDENCE:
            if evidence_sufficient:
                return self._decision(
                    key=key,
                    record=record,
                    situation=normalized_situation,
                    allow_reobserve=False,
                    next_action=(
                        InformationAction.ACCEPT_STRONG_GRASP_EVIDENCE
                    ),
                    exhausted_reason="",
                    stationary_limit=0,
                )
            return self._decision(
                key=key,
                record=record,
                situation=normalized_situation,
                allow_reobserve=False,
                next_action=(
                    InformationAction.ACQUIRE_INDEPENDENT_CAMERA_VIEW
                ),
                exhausted_reason=(
                    "same_camera_reobserve_is_not_independent_evidence"
                ),
                stationary_limit=0,
            )

        if normalized_situation is EvidenceSituation.PROVEN_FAILED_GRASP:
            return self._decision(
                key=key,
                record=record,
                situation=normalized_situation,
                allow_reobserve=False,
                next_action=(
                    InformationAction.PERFORM_FAILED_GRASP_CLEARANCE
                ),
                exhausted_reason="grasp_failure_already_proven",
                stationary_limit=0,
            )

        if normalized_situation is EvidenceSituation.RELEASE_STABILITY:
            decision = self._decide_distinct_temporal_capture(
                key=key,
                record=record,
                context=context,
                situation=normalized_situation,
            )
            return self._apply_stationary_scene_budget(
                context,
                situation=normalized_situation,
                decision=decision,
                limit=self.release_stationary_reobserve_limit,
            )
        if (
            normalized_situation
            is EvidenceSituation.TEMPORAL_IDENTITY_REPAIR
        ):
            decision = self._decide_distinct_temporal_capture(
                key=key,
                record=record,
                context=context,
                situation=normalized_situation,
            )
            return self._apply_stationary_scene_budget(
                context,
                situation=normalized_situation,
                decision=decision,
                limit=self.release_stationary_reobserve_limit,
            )

        if (
            record.stationary_reobserves_used
            < self.default_stationary_reobserve_limit
        ):
            record.stationary_reobserves_used += 1
            next_action = (
                InformationAction.RETRY_TRANSIENT_BACKEND
                if normalized_situation
                is EvidenceSituation.TRANSIENT_BACKEND_RETRY
                else InformationAction.STATIONARY_REOBSERVE_ONCE
            )
            decision = self._decision(
                key=key,
                record=record,
                situation=normalized_situation,
                allow_reobserve=True,
                next_action=next_action,
                exhausted_reason="",
                stationary_limit=(
                    self.default_stationary_reobserve_limit
                ),
            )
            return self._apply_stationary_scene_budget(
                context,
                situation=normalized_situation,
                decision=decision,
                limit=self.default_stationary_reobserve_limit,
            )

        next_action = (
            InformationAction.CHANGE_CAMERA_OR_ESCALATE_BACKEND_FAILURE
            if normalized_situation
            is EvidenceSituation.TRANSIENT_BACKEND_RETRY
            else InformationAction.PERFORM_CONTROLLED_RETURN_AND_REGRASP
        )
        exhausted_reason = (
            "same_state_transient_backend_retry_budget_exhausted"
            if normalized_situation
            is EvidenceSituation.TRANSIENT_BACKEND_RETRY
            else "same_state_ambiguous_grasp_reobserve_budget_exhausted"
        )
        return self._decision(
            key=key,
            record=record,
            situation=normalized_situation,
            allow_reobserve=False,
            next_action=next_action,
            exhausted_reason=exhausted_reason,
            stationary_limit=self.default_stationary_reobserve_limit,
        )

    def _apply_stationary_scene_budget(
        self,
        context: EvidenceAcquisitionContext,
        *,
        situation: EvidenceSituation,
        decision: dict[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        if decision.get("allow_reobserve") is not True:
            return decision
        scene_key = self._stationary_scene_key(context)
        used = self._stationary_scene_reobserves.get(scene_key, 0)
        if used < limit:
            used += 1
            self._stationary_scene_reobserves[scene_key] = used
            return {
                **decision,
                "stationary_scene_reobserves_used": used,
                "stationary_scene_reobserve_limit": limit,
            }

        if situation is EvidenceSituation.TRANSIENT_BACKEND_RETRY:
            next_action = (
                InformationAction.CHANGE_CAMERA_OR_ESCALATE_BACKEND_FAILURE
            )
        elif situation is EvidenceSituation.AMBIGUOUS_GRASP:
            next_action = InformationAction.PERFORM_CONTROLLED_RETURN_AND_REGRASP
        else:
            next_action = InformationAction.CHANGE_VIEWPOINT_OR_PHYSICAL_RECHECK
        return {
            **decision,
            "allow_reobserve": False,
            "next_information_action": next_action.value,
            "exhausted_reason": (
                "same_physical_state_reobserve_budget_exhausted"
            ),
            "stationary_scene_reobserves_used": used,
            "stationary_scene_reobserve_limit": limit,
        }

    @staticmethod
    def _stationary_scene_key(
        context: EvidenceAcquisitionContext,
    ) -> tuple[str, str, tuple[str, ...]]:
        return (
            _token(context.physical_action_token),
            _token(context.viewpoint_token),
            _normalize_camera_set(context.camera_set),
        )

    def _decide_distinct_temporal_capture(
        self,
        *,
        key: EvidenceStateKey,
        record: _BudgetRecord,
        context: EvidenceAcquisitionContext,
        situation: EvidenceSituation,
    ) -> dict[str, Any]:
        capture = _token(context.capture_id)
        capture_is_new = bool(
            capture
            and capture not in record.distinct_release_captures
        )
        if capture_is_new:
            record.distinct_release_captures.add(capture)

        if record.awaiting_distinct_release_capture:
            if not capture_is_new:
                return self._decision(
                    key=key,
                    record=record,
                    situation=situation,
                    allow_reobserve=False,
                    next_action=(
                        InformationAction.CHANGE_VIEWPOINT_OR_PHYSICAL_RECHECK
                    ),
                    exhausted_reason=(
                        "release_reobserve_did_not_produce_distinct_capture"
                    ),
                    stationary_limit=(
                        self.release_stationary_reobserve_limit
                    ),
                )
            record.awaiting_distinct_release_capture = False

        if (
            len(record.distinct_release_captures)
            >= self.release_distinct_capture_limit
        ):
            return self._decision(
                key=key,
                record=record,
                situation=situation,
                allow_reobserve=False,
                next_action=(
                    InformationAction.EVALUATE_RELEASE_STABILITY
                    if situation is EvidenceSituation.RELEASE_STABILITY
                    else InformationAction.EVALUATE_IDENTITY_REPAIR
                ),
                exhausted_reason="",
                stationary_limit=(
                    self.release_stationary_reobserve_limit
                ),
            )

        if (
            record.stationary_reobserves_used
            < self.release_stationary_reobserve_limit
        ):
            record.stationary_reobserves_used += 1
            record.awaiting_distinct_release_capture = True
            return self._decision(
                key=key,
                record=record,
                situation=situation,
                allow_reobserve=True,
                next_action=(
                    InformationAction.ACQUIRE_DISTINCT_RELEASE_CAPTURE
                    if situation is EvidenceSituation.RELEASE_STABILITY
                    else InformationAction.ACQUIRE_DISTINCT_IDENTITY_CAPTURE
                ),
                exhausted_reason="",
                stationary_limit=(
                    self.release_stationary_reobserve_limit
                ),
            )

        return self._decision(
            key=key,
            record=record,
            situation=situation,
            allow_reobserve=False,
            next_action=(
                InformationAction.CHANGE_VIEWPOINT_OR_PHYSICAL_RECHECK
            ),
            exhausted_reason=(
                "release_stability_capture_budget_exhausted"
            ),
            stationary_limit=self.release_stationary_reobserve_limit,
        )

    def _record_for_context(
        self,
        context: EvidenceAcquisitionContext,
    ) -> tuple[EvidenceStateKey, _BudgetRecord]:
        key = self.state_key(context)
        transaction = (
            key.track_id,
            key.arm,
            key.manipulation_phase,
            key.grasp_attempt_nonce,
        )
        self._forget_previous_attempts(transaction)
        physical_token = _token(context.physical_action_token)
        viewpoint_token = _token(context.viewpoint_token)
        active = self._active_transactions.get(transaction)
        reset = active is not None and active.state_key != key
        if active is not None and physical_token:
            reset = reset or bool(
                active.physical_action_token
                and active.physical_action_token != physical_token
            )
        if active is not None and viewpoint_token:
            reset = reset or bool(
                active.viewpoint_token
                and active.viewpoint_token != viewpoint_token
            )
        if reset:
            self._clear_transaction(transaction)
            active = None
        if active is None:
            active = _ActiveTransaction(
                state_key=key,
                physical_action_token=physical_token,
                viewpoint_token=viewpoint_token,
            )
            self._active_transactions[transaction] = active
        else:
            if physical_token:
                active.physical_action_token = physical_token
            if viewpoint_token:
                active.viewpoint_token = viewpoint_token
        return key, self._records.setdefault(key, _BudgetRecord())

    def _forget_previous_attempts(
        self,
        transaction: tuple[str, str, str, str],
    ) -> None:
        track_id, arm, phase, nonce = transaction
        if not nonce:
            return
        obsolete = [
            item
            for item in self._active_transactions
            if item[:3] == (track_id, arm, phase)
            and item[3]
            and item[3] != nonce
        ]
        for item in obsolete:
            self._clear_transaction(item)

    def _clear_transaction(
        self,
        transaction: tuple[str, str, str, str],
    ) -> None:
        active = self._active_transactions.pop(transaction, None)
        if active is not None:
            self._records.pop(active.state_key, None)

    def _decision(
        self,
        *,
        key: EvidenceStateKey,
        record: _BudgetRecord,
        situation: EvidenceSituation,
        allow_reobserve: bool,
        next_action: InformationAction,
        exhausted_reason: str,
        stationary_limit: int,
    ) -> dict[str, Any]:
        return {
            "allow_reobserve": bool(allow_reobserve),
            "next_information_action": next_action.value,
            "exhausted_reason": str(exhausted_reason or ""),
            "evidence_situation": situation.value,
            "state_key": key.as_dict(),
            "stationary_reobserves_used": int(
                record.stationary_reobserves_used
            ),
            "stationary_reobserve_limit": int(stationary_limit),
            "distinct_capture_count": len(
                record.distinct_release_captures
            ),
            "distinct_capture_limit": (
                self.release_distinct_capture_limit
                if situation
                in {
                    EvidenceSituation.RELEASE_STABILITY,
                    EvidenceSituation.TEMPORAL_IDENTITY_REPAIR,
                }
                else 0
            ),
        }


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _token(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_camera_set(cameras: Sequence[str]) -> tuple[str, ...]:
    values = () if cameras is None else cameras
    return tuple(
        sorted(
            {
                str(camera or "").strip().lower()
                for camera in values
                if str(camera or "").strip()
            }
        )
    )


def _quantize_pose(
    pose: Sequence[float] | None,
    *,
    translation_quantum_m: float,
    quaternion_quantum: float,
) -> tuple[int, ...]:
    if pose is None:
        return ()
    try:
        values = [float(value) for value in pose]
    except (TypeError, ValueError):
        return ()
    if len(values) not in {3, 7} or not all(
        math.isfinite(value) for value in values
    ):
        return ()
    xyz = tuple(
        int(round(value / translation_quantum_m))
        for value in values[:3]
    )
    if len(values) == 3:
        return xyz
    quaternion = values[3:]
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-9:
        return xyz
    quaternion = [value / norm for value in quaternion]
    first_nonzero = next(
        (value for value in quaternion if abs(value) > 1e-9),
        1.0,
    )
    if first_nonzero < 0.0:
        quaternion = [-value for value in quaternion]
    return (
        *xyz,
        *(
            int(round(value / quaternion_quantum))
            for value in quaternion
        ),
    )

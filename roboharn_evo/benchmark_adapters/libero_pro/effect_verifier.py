"""Deterministic, observation-only effect verification for LIBERO actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from roboharn_evo.benchmark_adapters.base import NeutralObservation
from roboharn_evo.benchmark_adapters.libero_pro.perception import (
    PerceptionDetection,
    PerceptionFrame,
)


@dataclass(frozen=True, slots=True)
class LiberoEffectVerifierConfig:
    closed_aperture_max: float = 0.03
    open_aperture_min: float = 0.05
    minimum_lift_m: float = 0.025
    minimum_target_motion_fraction: float = 0.02
    stationary_target_fraction: float = 0.008
    support_bbox_margin_fraction: float = 0.06

    def __post_init__(self) -> None:
        values = (
            self.closed_aperture_max,
            self.open_aperture_min,
            self.minimum_lift_m,
            self.minimum_target_motion_fraction,
            self.stationary_target_fraction,
            self.support_bbox_margin_fraction,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(
                "effect verifier thresholds must be finite and non-negative"
            )
        if self.closed_aperture_max >= self.open_aperture_min:
            raise ValueError("closed aperture threshold must be below open threshold")


@dataclass(frozen=True, slots=True)
class LiberoEffectVerdict:
    action: str
    verdict: str
    observed_effect: str
    reason: str
    missing_evidence: tuple[str, ...]
    measurements: dict[str, Any]

    def __post_init__(self) -> None:
        if self.action not in {
            "grasp",
            "place",
            "contact",
            "open",
            "close",
            "move",
            "align",
            "reobserve",
            "recover",
            "other",
        }:
            raise ValueError("effect action is invalid")
        if self.verdict not in {"support", "oppose", "unverified"}:
            raise ValueError("effect verdict is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "verdict": self.verdict,
            "observed_effect": self.observed_effect,
            "reason": self.reason,
            "missing_evidence": list(self.missing_evidence),
            "measurements": dict(self.measurements),
        }


def _array(observation: NeutralObservation, key: str, size: int) -> np.ndarray:
    value = np.asarray(observation.raw.get(key), dtype=np.float64).reshape(-1)
    if value.size != size or not np.isfinite(value).all():
        raise ValueError(f"observation field {key} is invalid")
    return value


def _aperture(observation: NeutralObservation) -> float:
    qpos = _array(observation, "robot0_gripper_qpos", 2)
    return float(abs(qpos[0] - qpos[1]))


def _eef_height(observation: NeutralObservation) -> float:
    return float(_array(observation, "robot0_eef_pos", 3)[2])


def _best(
    frame: PerceptionFrame,
    *,
    role: str,
    camera: str,
) -> PerceptionDetection | None:
    values = frame.detections_for(role=role, camera=camera)
    return max(values, key=lambda value: value.score, default=None)


def _normalized_motion(
    before: PerceptionDetection,
    after: PerceptionDetection,
) -> float:
    if before.image_shape != after.image_shape:
        return 0.0
    delta = np.asarray(after.centroid_px) - np.asarray(before.centroid_px)
    diagonal = float(np.linalg.norm(np.asarray(before.image_shape, dtype=float)))
    return float(np.linalg.norm(delta) / max(diagonal, 1.0))


def _center_within_support(
    target: PerceptionDetection,
    support: PerceptionDetection,
    *,
    margin_fraction: float,
) -> bool:
    if target.image_shape != support.image_shape:
        return False
    height, width = target.image_shape
    margin_x = width * margin_fraction
    margin_y = height * margin_fraction
    x0, y0, x1, y1 = support.bbox_xyxy
    x, y = target.centroid_px
    return x0 - margin_x <= x <= x1 + margin_x and y0 - margin_y <= y <= y1 + margin_y


class LiberoDeterministicEffectVerifier:
    """Use public robot state plus fresh SAM3 observations; never simulator GT."""

    def __init__(
        self,
        config: LiberoEffectVerifierConfig | None = None,
    ) -> None:
        self.config = config or LiberoEffectVerifierConfig()

    def verify(
        self,
        *,
        action: str,
        before_observation: NeutralObservation,
        after_observation: NeutralObservation,
        before_frame: PerceptionFrame,
        after_frame: PerceptionFrame,
    ) -> LiberoEffectVerdict:
        normalized = str(action or "other").strip().casefold()
        if normalized not in {
            "grasp",
            "place",
            "contact",
            "open",
            "close",
            "move",
            "align",
            "reobserve",
            "recover",
            "other",
        }:
            normalized = "other"
        try:
            aperture_before = _aperture(before_observation)
            aperture_after = _aperture(after_observation)
            vertical_delta = _eef_height(after_observation) - _eef_height(
                before_observation
            )
        except ValueError as exc:
            return LiberoEffectVerdict(
                action=normalized,
                verdict="unverified",
                observed_effect="public robot state unavailable",
                reason="deterministic robot-state validation could not run",
                missing_evidence=(str(exc),),
                measurements={},
            )
        measurements: dict[str, Any] = {
            "gripper_aperture_before": aperture_before,
            "gripper_aperture_after": aperture_after,
            "eef_vertical_delta_m": vertical_delta,
            "before_sam3_failures": len(before_frame.failures),
            "after_sam3_failures": len(after_frame.failures),
        }
        if normalized == "grasp":
            return self._verify_grasp(
                before_frame=before_frame,
                after_frame=after_frame,
                aperture_after=aperture_after,
                vertical_delta=vertical_delta,
                measurements=measurements,
            )
        if normalized == "place":
            return self._verify_place(
                after_frame=after_frame,
                aperture_after=aperture_after,
                measurements=measurements,
            )
        return LiberoEffectVerdict(
            action=normalized,
            verdict="unverified",
            observed_effect="no deterministic typed effect extractor for this action",
            reason="only grasp and place effects are currently validated",
            missing_evidence=("typed deterministic validator",),
            measurements=measurements,
        )

    def _verify_grasp(
        self,
        *,
        before_frame: PerceptionFrame,
        after_frame: PerceptionFrame,
        aperture_after: float,
        vertical_delta: float,
        measurements: dict[str, Any],
    ) -> LiberoEffectVerdict:
        before_external = _best(before_frame, role="target", camera="external")
        after_external = _best(after_frame, role="target", camera="external")
        after_wrist = _best(after_frame, role="target", camera="wrist")
        target_motion = (
            _normalized_motion(before_external, after_external)
            if before_external is not None and after_external is not None
            else None
        )
        closed = aperture_after <= self.config.closed_aperture_max
        lifted = vertical_delta >= self.config.minimum_lift_m
        wrist_visible = after_wrist is not None
        moved = (
            target_motion is not None
            and target_motion >= self.config.minimum_target_motion_fraction
        )
        measurements.update(
            {
                "gripper_closed": closed,
                "eef_lifted": lifted,
                "target_visible_after_wrist": wrist_visible,
                "target_external_motion_fraction": target_motion,
            }
        )
        if closed and lifted and wrist_visible and moved:
            return LiberoEffectVerdict(
                action="grasp",
                verdict="support",
                observed_effect="target moved with the closed, lifted gripper",
                reason=(
                    "closed-gripper state, upward end-effector motion, external "
                    "target displacement, and fresh wrist visibility agree"
                ),
                missing_evidence=(),
                measurements=measurements,
            )
        stationary = (
            target_motion is not None
            and target_motion <= self.config.stationary_target_fraction
        )
        if closed and lifted and stationary and not wrist_visible:
            return LiberoEffectVerdict(
                action="grasp",
                verdict="oppose",
                observed_effect="target remained at its prior support while gripper lifted",
                reason=(
                    "the gripper closed and lifted, but the external target stayed "
                    "stationary and no target was visible from the wrist"
                ),
                missing_evidence=(),
                measurements=measurements,
            )
        missing: list[str] = []
        if not closed:
            missing.append("closed gripper")
        if not lifted:
            missing.append("sufficient upward end-effector motion")
        if not wrist_visible:
            missing.append("fresh wrist target detection")
        if not moved:
            missing.append("measurable external target displacement")
        return LiberoEffectVerdict(
            action="grasp",
            verdict="unverified",
            observed_effect="target attachment could not be confirmed",
            reason="deterministic grasp evidence was incomplete",
            missing_evidence=tuple(missing),
            measurements=measurements,
        )

    def _verify_place(
        self,
        *,
        after_frame: PerceptionFrame,
        aperture_after: float,
        measurements: dict[str, Any],
    ) -> LiberoEffectVerdict:
        target = _best(after_frame, role="target", camera="external")
        contexts = after_frame.detections_for(role="context", camera="external")
        support = max(contexts, key=lambda value: value.score, default=None)
        opened = aperture_after >= self.config.open_aperture_min
        supported = (
            target is not None
            and support is not None
            and _center_within_support(
                target,
                support,
                margin_fraction=self.config.support_bbox_margin_fraction,
            )
        )
        measurements.update(
            {
                "gripper_open": opened,
                "target_visible_after_external": target is not None,
                "context_visible_after_external": support is not None,
                "target_center_within_context": supported,
            }
        )
        if opened and supported:
            return LiberoEffectVerdict(
                action="place",
                verdict="support",
                observed_effect="target is visible on the destination context after release",
                reason=(
                    "the gripper opened and the fresh target center lies within "
                    "the detected destination support"
                ),
                missing_evidence=(),
                measurements=measurements,
            )
        missing: list[str] = []
        if not opened:
            missing.append("opened gripper")
        if target is None:
            missing.append("fresh external target detection")
        if support is None:
            missing.append("fresh destination support detection")
        if target is not None and support is not None and not supported:
            missing.append("target supported by destination")
        return LiberoEffectVerdict(
            action="place",
            verdict="unverified",
            observed_effect="support transfer could not be confirmed",
            reason="deterministic placement evidence was incomplete",
            missing_evidence=tuple(missing),
            measurements=measurements,
        )


__all__ = [
    "LiberoDeterministicEffectVerifier",
    "LiberoEffectVerdict",
    "LiberoEffectVerifierConfig",
]

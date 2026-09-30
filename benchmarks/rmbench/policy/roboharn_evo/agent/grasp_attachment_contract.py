from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from roboharn_evo.agent.gripper_state import gripper_command_state

from .operation_candidates import (
    pose7_to_matrix,
    valid_pending_held_object_to_tcp_attachment,
)
from .recovery.tool_specs import RecoveryToolCall


_GRASP_DIAGNOSTIC_LIFT_MARKER = "_runtime_grasp_diagnostic_lift"
_GRASP_DIAGNOSTIC_MOTION_MARKER = "_runtime_grasp_diagnostic_motion"


@dataclass(frozen=True, slots=True)
class GraspMotionLimits:
    """Conservative bounds for proving rigid motion after a diagnostic move."""

    minimum_ee_motion_m: float = 0.01
    maximum_ee_motion_m: float = 0.08
    anchor_association_m: float = 0.05
    transition_association_m: float = 0.03
    candidate_margin_m: float = 0.012
    maximum_coupled_residual_m: float = 0.01
    residual_fraction_of_motion: float = 0.5
    maximum_stationary_residual_m: float = 0.006
    minimum_hypothesis_margin_m: float = 0.006
    minimum_direction_cosine: float = 0.85
    minimum_planar_extent_ratio: float = 0.5
    minimum_independent_views: int = 2
    maximum_cross_view_position_spread_m: float = 0.05
    maximum_ee_delta_disagreement_m: float = 0.006
    # A single fixed external camera may prove attachment only when its
    # temporal evidence is substantially stronger than the ordinary
    # per-camera evidence used by the two-view contract.  These bounds do not
    # reduce ``minimum_independent_views`` and therefore do not turn every
    # one-camera observation into positive evidence.
    strong_single_view_minimum_ee_motion_m: float = 0.02
    strong_single_view_minimum_object_motion_m: float = 0.02
    strong_single_view_maximum_coupled_residual_m: float = 0.003
    strong_single_view_maximum_raw_coupled_residual_m: float = 0.008
    strong_single_view_minimum_hypothesis_margin_m: float = 0.015
    strong_single_view_minimum_direction_cosine: float = 0.98
    strong_single_view_minimum_raw_direction_cosine: float = 0.95
    # The temporal witness must also remain close to the grasp-time object
    # anchor and to the object position predicted by the exact pending
    # object-to-TCP attachment.  The v6 third-camera replay has a 13.8 mm
    # plane/anchor residual, so 20 mm leaves calibration margin without
    # retaining the ordinary 50 mm association radius as proof authority.
    strong_single_view_maximum_anchor_residual_m: float = 0.02
    strong_single_view_maximum_attachment_anchor_residual_m: float = 0.005
    strong_single_view_maximum_attachment_post_residual_m: float = 0.02


DEFAULT_GRASP_MOTION_LIMITS = GraspMotionLimits()


def capture_grasp_motion_snapshot(
    *,
    observation_preprocess: Any,
    manipulation_state: Any,
    robot_state: Any,
) -> dict[str, Any]:
    """Capture only identity-bound world points needed for grasp verification.

    The raw segmentation result can be large and its action geometry may be
    quarantined after self-occlusion.  For attachment verification we retain
    only calibrated centroids from the exact bound query, grouped by camera.
    No candidate is selected at capture time.
    """

    preprocess = (
        observation_preprocess
        if isinstance(observation_preprocess, dict)
        else {}
    )
    states = manipulation_state if isinstance(manipulation_state, dict) else {}
    robots = robot_state if isinstance(robot_state, dict) else {}
    captured: dict[str, Any] = {
        "env_step": _integer(preprocess.get("env_step")),
        "observation_generation": _integer(
            preprocess.get("observation_generation")
        ),
        "observation_capture_id": _integer(
            preprocess.get("observation_capture_id")
        ),
        "arms": {},
    }
    for arm in ("left", "right"):
        state = states.get(arm)
        if not _pending_grasp_state(state):
            continue
        assert isinstance(state, dict)
        instance_id = str(state.get("held_instance_id", "") or "").strip()
        candidate_id = str(state.get("grasp_candidate_id", "") or "").strip()
        attempt_nonce = str(
            state.get("grasp_attempt_nonce", "") or ""
        ).strip()
        attempt_step = _integer(state.get("grasp_attempt_step"))
        anchor = _xyz(state.get("pregrasp_object_world_m"))
        descriptor = state.get("held_object_perception_descriptor")
        object_id = (
            str(descriptor.get("object_id", "") or "").strip().lower()
            if isinstance(descriptor, dict)
            else ""
        )
        if (
            not instance_id
            or not candidate_id
            or not attempt_nonce
            or attempt_step is None
            or anchor is None
        ):
            continue
        observations = _bound_world_observations(
            preprocess.get("segmentation"),
            instance_id=instance_id,
            object_id=object_id,
            acting_arm=arm,
        )
        arm_robot = robots.get(arm)
        gripper = (
            _finite_float(arm_robot.get("gripper"))
            if isinstance(arm_robot, dict)
            else None
        )
        diagnostic = state.get("diagnostic_lift_evidence")
        diagnostic_delta = (
            _xyz(diagnostic.get("observed_displacement_xyz_m"))
            if isinstance(diagnostic, dict)
            else None
        )
        captured["arms"][arm] = {
            "phase": "grasp_candidate",
            "held_instance_id": instance_id,
            "grasp_candidate_id": candidate_id,
            "grasp_attempt_nonce": attempt_nonce,
            "grasp_attempt_step": attempt_step,
            "pregrasp_object_world_m": anchor,
            "object_id": object_id,
            "held_object_to_tcp_attachment": (
                dict(state.get("held_object_to_tcp_attachment"))
                if isinstance(
                    state.get("held_object_to_tcp_attachment"),
                    dict,
                )
                else None
            ),
            "gripper": gripper,
            "gripper_command": arm_robot.get("gripper_command") if isinstance(arm_robot, dict) else None,
            "ee_world_m": (
                _xyz(arm_robot.get("xyz"))
                if isinstance(arm_robot, dict)
                else None
            ),
            **(
                {
                    "pending_lift_displacement_xyz_m": diagnostic_delta,
                    "diagnostic_lift_step": _integer(
                        diagnostic.get("lift_env_step")
                    ),
                    "diagnostic_lift_observation_generation": _integer(
                        diagnostic.get("lift_observation_generation")
                    ),
                    "postlift_ee_world_m": _xyz(
                        diagnostic.get("postlift_ee_world_m")
                    ),
                }
                if diagnostic_delta is not None
                else {}
            ),
            "observations_by_camera": observations,
        }
    return captured


def validate_grasp_motion_effect(
    *,
    calls: list[RecoveryToolCall],
    results: list[Any],
    pre: Any,
    post: Any,
    post_action_feedback: bool = False,
    limits: GraspMotionLimits = DEFAULT_GRASP_MOTION_LIMITS,
) -> dict[str, Any]:
    """Prove or disprove attachment from same-object motion in world space.

    Ordinarily, a positive result requires two independently calibrated
    cameras to observe the bound object move by the same vector as the end
    effector.  One fixed external camera may also prove attachment when two
    distinct captures provide substantially stronger temporal coupling
    evidence.  Wrist-camera evidence alone can never use that exception.  A
    negative result still requires two cameras to observe the object remain
    stationary.  Any mixed, missing, or ambiguous evidence remains undecided.
    """

    diagnostic_motion = _single_diagnostic_motion(
        calls, results, post_action_feedback=post_action_feedback,
    )
    if diagnostic_motion is None:
        return {"applicable": False, "verified": None}
    arm, ee_delta = diagnostic_motion
    ee_motion = _norm(ee_delta)
    if (
        ee_motion < limits.minimum_ee_motion_m
        or ee_motion > limits.maximum_ee_motion_m
    ):
        return {
            "applicable": False,
            "verified": None,
            "reason": "diagnostic_motion_outside_validation_bounds",
        }

    before = _arm_snapshot(pre, arm)
    after = _arm_snapshot(post, arm)
    if before is None or after is None:
        return {"applicable": False, "verified": None}
    if not _same_grasp_transaction(before, after):
        return {"applicable": False, "verified": None}
    pre_step = _integer(pre.get("env_step")) if isinstance(pre, dict) else None
    post_step = _integer(post.get("env_step")) if isinstance(post, dict) else None
    attempt_step = _integer(before.get("grasp_attempt_step"))
    if (
        pre_step is None
        or post_step is None
        or attempt_step is None
        or pre_step < attempt_step
        or post_step <= pre_step
    ):
        return {
            "applicable": False,
            "verified": None,
            "reason": "snapshot_pair_does_not_belong_to_current_grasp_attempt",
        }
    common_cameras = _common_supported_cameras(
        before,
        after,
        acting_arm=arm,
    )
    if not common_cameras:
        return {
            "applicable": False,
            "verified": None,
            "reason": "insufficient_common_nonacting_camera_views",
            "available_camera_count": len(common_cameras),
            "available_cameras": common_cameras,
        }
    if not _fresh_snapshot_pair(pre, post):
        return {
            "applicable": True,
            "verified": None,
            "arm": arm,
            "held_instance_id": before["held_instance_id"],
            "failure_kind": "observation_not_fresh",
        }
    if not (
        gripper_command_state(before) == "closed"
        and gripper_command_state(after) == "closed"
    ):
        return {
            "applicable": True,
            "verified": None,
            "arm": arm,
            "held_instance_id": before["held_instance_id"],
            "failure_kind": "closing_command_not_maintained_during_lift",
        }
    before_ee = _xyz(before.get("ee_world_m"))
    after_ee = _xyz(after.get("ee_world_m"))
    if before_ee is None or after_ee is None:
        return {
            "applicable": False,
            "verified": None,
            "reason": "missing_grasp_attempt_ee_snapshot",
        }
    snapshot_ee_delta = _subtract(after_ee, before_ee)
    ee_delta_disagreement = _norm(_subtract(snapshot_ee_delta, ee_delta))
    if ee_delta_disagreement > limits.maximum_ee_delta_disagreement_m:
        return {
            "applicable": True,
            "verified": None,
            "arm": arm,
            "held_instance_id": before["held_instance_id"],
            "grasp_candidate_id": before["grasp_candidate_id"],
            "grasp_attempt_nonce": before["grasp_attempt_nonce"],
            "failure_kind": "lift_result_and_snapshot_motion_disagree",
            "result_ee_displacement_xyz_m": _round_xyz(ee_delta),
            "snapshot_ee_displacement_xyz_m": _round_xyz(
                snapshot_ee_delta
            ),
            "ee_delta_disagreement_m": round(ee_delta_disagreement, 6),
        }

    anchor = _xyz(before.get("pregrasp_object_world_m"))
    if anchor is None:
        return {"applicable": False, "verified": None}
    evidence = _camera_motion_evidence(
        before=before,
        after=after,
        anchor=anchor,
        ee_delta=ee_delta,
        acting_arm=arm,
        limits=limits,
    )
    positives = [item for item in evidence if item["classification"] == "coupled"]
    negatives = [item for item in evidence if item["classification"] == "stationary"]
    strong_single_view_support = _strong_single_fixed_external_camera_support(
        positives=positives,
        negatives=negatives,
        before=before,
        pre=pre,
        post=post,
        anchor=anchor,
        ee_delta=ee_delta,
        ee_motion=ee_motion,
        limits=limits,
    )
    verified: bool | None
    failure_kind = ""
    used_strong_single_view = False
    cross_view_consistent = _cross_view_positions_consistent(
        evidence,
        maximum_spread_m=limits.maximum_cross_view_position_spread_m,
    )
    if not cross_view_consistent:
        verified = None
        failure_kind = "cross_view_identity_position_inconsistent"
    elif (
        len(positives) >= limits.minimum_independent_views
        and not negatives
    ):
        verified = True
    elif strong_single_view_support is not None:
        verified = True
        used_strong_single_view = True
    elif (
        len(negatives) >= limits.minimum_independent_views
        and not positives
    ):
        verified = False
        failure_kind = "object_stationary_during_lift"
    else:
        verified = None
        failure_kind = (
            "conflicting_multiview_motion"
            if positives and negatives
            else "insufficient_independent_views"
        )

    supporting = [str(item["camera"]) for item in positives]
    result: dict[str, Any] = {
        "applicable": True,
        "verified": verified,
        "arm": arm,
        "held_instance_id": before["held_instance_id"],
        "grasp_candidate_id": before["grasp_candidate_id"],
        "grasp_attempt_nonce": before["grasp_attempt_nonce"],
        "ee_displacement_xyz_m": _round_xyz(ee_delta),
        "ee_displacement_m": round(ee_motion, 6),
        "supporting_camera_count": len(positives),
        "supporting_cameras": supporting,
        "negative_camera_count": len(negatives),
        "camera_evidence": evidence,
        "cross_view_position_consistent": cross_view_consistent,
    }
    if verified is True:
        result["verified_object_world_m"] = _round_xyz(
            _add(anchor, ee_delta)
        )
        if used_strong_single_view:
            result.update(
                {
                    "evidence_source": (
                        "identity_bound_single_fixed_external_camera_"
                        "strong_world_motion_coupling"
                    ),
                    "single_view_fast_path": True,
                    "pre_observation_capture_id": _integer(
                        pre.get("observation_capture_id")
                    ),
                    "post_observation_capture_id": _integer(
                        post.get("observation_capture_id")
                    ),
                    "single_view_attachment_consistency": {
                        key: strong_single_view_support.get(key)
                        for key in (
                            "pre_anchor_plane_residual_m",
                            "pre_anchor_raw_residual_m",
                            "attachment_capture_anchor_residual_m",
                            "attachment_post_plane_residual_m",
                            "attachment_post_raw_residual_m",
                            "attachment_capture_object_world_m",
                            "attachment_post_predicted_object_world_m",
                        )
                    },
                }
            )
        else:
            result["evidence_source"] = (
                "identity_bound_multiview_world_motion_coupling"
            )
    if failure_kind:
        result["failure_kind"] = failure_kind
    return result


def validate_pending_grasp_observation_effect(
    *,
    calls: list[RecoveryToolCall],
    results: list[Any],
    post: Any,
    limits: GraspMotionLimits = DEFAULT_GRASP_MOTION_LIMITS,
) -> dict[str, Any]:
    """Resolve an earlier ambiguous lift from a later static observation.

    The first lift displacement and close-time identity are persisted in the
    pending grasp state.  A later reobserve may therefore compare the exact
    object against the original anchor and the anchor translated by that
    already-executed lift, without moving the robot again.
    """

    if not calls or not all(
        call.tool_name == "reobserve_scene" for call in calls
    ):
        return {"applicable": False, "verified": None}
    if len(results) < len(calls) or not all(
        bool(getattr(results[index], "success", False))
        for index in range(len(calls))
    ):
        return {"applicable": False, "verified": None}
    if not isinstance(post, dict):
        return {"applicable": False, "verified": None}
    post_step = _integer(post.get("env_step"))
    post_generation = _integer(post.get("observation_generation"))
    arms = post.get("arms")
    candidates = arms if isinstance(arms, dict) else {}
    applicable_arms: list[tuple[str, dict[str, Any]]] = []
    for arm, snapshot in candidates.items():
        if arm not in {"left", "right"} or not isinstance(snapshot, dict):
            continue
        lift_delta = _xyz(
            snapshot.get("pending_lift_displacement_xyz_m")
        )
        lift_step = _integer(snapshot.get("diagnostic_lift_step"))
        lift_generation = _integer(
            snapshot.get("diagnostic_lift_observation_generation")
        )
        if (
            lift_delta is not None
            and lift_step is not None
            and post_step is not None
            and post_step >= lift_step
            and lift_generation is not None
            and post_generation is not None
            and post_generation > lift_generation
        ):
            applicable_arms.append((arm, snapshot))
    if len(applicable_arms) != 1:
        return {"applicable": False, "verified": None}

    arm, snapshot = applicable_arms[0]
    anchor = _xyz(snapshot.get("pregrasp_object_world_m"))
    lift_delta = _xyz(snapshot.get("pending_lift_displacement_xyz_m"))
    observations = _normalized_observations_by_camera(
        snapshot.get("observations_by_camera"),
        acting_arm=arm,
    )
    if anchor is None or lift_delta is None:
        return {"applicable": False, "verified": None}
    postlift_ee = _xyz(snapshot.get("postlift_ee_world_m"))
    current_ee = _xyz(snapshot.get("ee_world_m"))
    if not (
        gripper_command_state(snapshot) == "closed"
        and postlift_ee is not None
        and current_ee is not None
        and _distance(postlift_ee, current_ee)
        <= limits.maximum_ee_delta_disagreement_m
    ):
        return {
            "applicable": True,
            "verified": None,
            "arm": arm,
            "held_instance_id": snapshot.get("held_instance_id"),
            "grasp_candidate_id": snapshot.get("grasp_candidate_id"),
            "grasp_attempt_nonce": snapshot.get("grasp_attempt_nonce"),
            "failure_kind": "postlift_robot_state_changed",
        }
    if len(observations) < limits.minimum_independent_views:
        return {
            "applicable": True,
            "verified": None,
            "arm": arm,
            "held_instance_id": snapshot.get("held_instance_id"),
            "grasp_candidate_id": snapshot.get("grasp_candidate_id"),
            "grasp_attempt_nonce": snapshot.get("grasp_attempt_nonce"),
            "failure_kind": "insufficient_static_postlift_views",
        }

    predicted = _add(anchor, lift_delta)
    evidence: list[dict[str, Any]] = []
    for camera in sorted(observations):
        candidate = _select_unique_candidate(
            observations.get(camera),
            hypotheses=[anchor, predicted],
            maximum_distance=limits.anchor_association_m,
            margin=limits.candidate_margin_m,
        )
        if candidate is None:
            continue
        world = _xyz(candidate.get("raw_centroid_world_m"))
        if world is None:
            continue
        anchor_distance = _distance(world, anchor)
        predicted_distance = _distance(world, predicted)
        if (
            predicted_distance <= limits.transition_association_m
            and anchor_distance - predicted_distance
            >= limits.minimum_hypothesis_margin_m
        ):
            classification = "coupled"
        elif (
            anchor_distance <= limits.transition_association_m
            and predicted_distance - anchor_distance
            >= limits.minimum_hypothesis_margin_m
        ):
            classification = "stationary"
        else:
            classification = "ambiguous"
        evidence.append(
            {
                "camera": camera,
                "classification": classification,
                "pre_world_m": _round_xyz(anchor),
                "post_world_m": _round_xyz(world),
                "anchor_distance_m": round(anchor_distance, 6),
                "predicted_distance_m": round(predicted_distance, 6),
                "measurement_source": (
                    "static_postlift_identity_bound_world_position"
                ),
            }
        )

    positives = [item for item in evidence if item["classification"] == "coupled"]
    negatives = [item for item in evidence if item["classification"] == "stationary"]
    cross_view_consistent = _cross_view_positions_consistent(
        evidence,
        maximum_spread_m=limits.maximum_cross_view_position_spread_m,
    )
    verified: bool | None = None
    failure_kind = "insufficient_static_postlift_evidence"
    if cross_view_consistent and len(positives) >= limits.minimum_independent_views and not negatives:
        verified = True
        failure_kind = ""
    elif cross_view_consistent and len(negatives) >= limits.minimum_independent_views and not positives:
        verified = False
        failure_kind = "object_stationary_during_lift"
    elif not cross_view_consistent:
        failure_kind = "cross_view_identity_position_inconsistent"
    result: dict[str, Any] = {
        "applicable": True,
        "verified": verified,
        "arm": arm,
        "held_instance_id": snapshot.get("held_instance_id"),
        "grasp_candidate_id": snapshot.get("grasp_candidate_id"),
        "grasp_attempt_nonce": snapshot.get("grasp_attempt_nonce"),
        "ee_displacement_xyz_m": _round_xyz(lift_delta),
        "ee_displacement_m": round(_norm(lift_delta), 6),
        "supporting_camera_count": len(positives),
        "supporting_cameras": [item["camera"] for item in positives],
        "negative_camera_count": len(negatives),
        "camera_evidence": evidence,
        "cross_view_position_consistent": cross_view_consistent,
        "evidence_source": "deferred_static_postlift_multiview_observation",
    }
    if verified is True:
        result["verified_object_world_m"] = _round_xyz(predicted)
    if failure_kind:
        result["failure_kind"] = failure_kind
    return result


def pending_grasp_observation_attempted(
    *,
    calls: list[RecoveryToolCall],
    post: Any,
) -> bool:
    if not calls or not all(
        call.tool_name == "reobserve_scene" for call in calls
    ):
        return False
    arms = post.get("arms") if isinstance(post, dict) else None
    return bool(
        isinstance(arms, dict)
        and any(
            isinstance(snapshot, dict)
            and _xyz(
                snapshot.get("pending_lift_displacement_xyz_m")
            )
            is not None
            and str(snapshot.get("grasp_attempt_nonce", "") or "").strip()
            for snapshot in arms.values()
        )
    )


def apply_grasp_motion_validation(
    effect: Any,
    validation: Any,
    *,
    diagnostic_lift_attempted: bool = False,
) -> dict[str, Any]:
    """Attach runtime evidence and commit only a proven positive grasp."""

    result = dict(effect) if isinstance(effect, dict) else {}
    if not isinstance(validation, dict) or validation.get("applicable") is not True:
        if diagnostic_lift_attempted:
            result["runtime_grasp_validation"] = (
                dict(validation) if isinstance(validation, dict) else {}
            )
            result.update(
                {
                    "effect_verified": "unverified",
                    "effect_type": "grasp",
                    "subtask_status": "uncertain",
                    "recommended_control": "retry",
                    "failure_reason": (
                        "The runtime diagnostic motion lacked sufficient exact "
                        "multiview evidence; model-only completion is rejected."
                    ),
                    "authority": "runtime_multiview_grasp_motion",
                }
            )
        return result
    result["runtime_grasp_validation"] = dict(validation)
    if validation.get("verified") is False:
        try:
            previous_confidence = float(result.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            previous_confidence = 0.0
        result.update(
            {
                "effect_verified": "false",
                "effect_type": "grasp",
                "confidence": max(0.95, previous_confidence),
                "failure_reason": (
                    "The identity-bound object remained stationary in at "
                    "least two calibrated views during the diagnostic motion."
                ),
                "next_constraint": (
                    "Clear this unconfirmed grasp along the recorded ingress "
                    "path, then reacquire before another grasp attempt."
                ),
                "memory_update": (
                    "Deterministic multiview evidence rejected attachment; "
                    "transport remains forbidden."
                ),
                "subtask_status": "in_progress",
                "recommended_control": "retry",
                "authority": "runtime_multiview_grasp_motion",
            }
        )
        return result
    if validation.get("verified") is not True:
        result.update(
            {
                "effect_verified": "unverified",
                "effect_type": "grasp",
                "subtask_status": "uncertain",
                "recommended_control": "retry",
                "failure_reason": (
                    "Runtime grasp-motion evidence is ambiguous; do not "
                    "authorize transport from any model-only verdict."
                ),
                "authority": "runtime_multiview_grasp_motion",
            }
        )
        return result

    try:
        previous_confidence = float(result.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        previous_confidence = 0.0
    if validation.get("single_view_fast_path") is True:
        evidence_description = (
            "one fixed external camera across two distinct captures"
        )
        memory_evidence = "strong fixed-camera world-motion coupling"
        evidence_authority = (
            "runtime_single_fixed_camera_grasp_motion"
        )
    else:
        evidence_description = (
            f"{validation.get('supporting_camera_count', 0)} independent "
            "calibrated cameras"
        )
        memory_evidence = "multiview world-motion coupling"
        evidence_authority = "runtime_multiview_grasp_motion"
    result.update(
        {
            "effect_verified": "true",
            "effect_type": "grasp",
            "confidence": max(0.95, previous_confidence),
            "evidence_summary": (
                "The identity-bound object moved with the end effector in "
                f"{evidence_description} during the diagnostic motion."
            ),
            "failure_reason": "",
            "next_constraint": (
                "Treat the object as attached and continue bounded transport; "
                "do not reopen the gripper as failed-grasp recovery."
            ),
            "memory_update": (
                "Grasp attachment confirmed from identity-bound "
                f"{memory_evidence}; TCP propagation is now authorized."
            ),
            "subtask_status": "in_progress",
            "recommended_control": "continue",
            "authority": evidence_authority,
        }
    )
    return result


def matched_grasp_motion_validation(
    effect: Any,
    *,
    arm: str,
    arm_state: Any,
) -> dict[str, Any] | None:
    """Return runtime evidence only for the exact pending grasp transaction."""

    if not isinstance(effect, dict) or not isinstance(arm_state, dict):
        return None
    validation = effect.get("runtime_grasp_validation")
    if not isinstance(validation, dict) or validation.get("applicable") is not True:
        return None
    normalized_arm = str(arm or "").strip().lower()
    held_instance_id = str(
        arm_state.get("held_instance_id", "") or ""
    ).strip()
    grasp_candidate_id = str(
        arm_state.get("grasp_candidate_id", "") or ""
    ).strip()
    if (
        normalized_arm not in {"left", "right"}
        or str(validation.get("arm", "") or "").strip().lower()
        != normalized_arm
        or str(validation.get("held_instance_id", "") or "").strip()
        != held_instance_id
    ):
        return None
    validation_candidate_id = str(
        validation.get("grasp_candidate_id", "") or ""
    ).strip()
    if (
        not validation_candidate_id
        or not grasp_candidate_id
        or validation_candidate_id != grasp_candidate_id
    ):
        return None
    attempt_nonce = str(
        arm_state.get("grasp_attempt_nonce", "") or ""
    ).strip()
    if (
        not attempt_nonce
        or str(validation.get("grasp_attempt_nonce", "") or "").strip()
        != attempt_nonce
    ):
        return None
    return dict(validation)


def _pending_grasp_state(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and str(value.get("phase", "") or "").strip().lower()
        == "grasp_candidate"
        and value.get("holding_confirmed") is not True
        and value.get("transport_authorized") is not True
    )


def _bound_world_observations(
    raw_segments: Any,
    *,
    instance_id: str,
    object_id: str,
    acting_arm: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    segments = raw_segments if isinstance(raw_segments, list) else []
    normalized_ref = instance_id.strip().lower()
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("success") is not True:
            continue
        # A segment is produced from one calibrated image, so its camera is
        # authoritative for every nested detection.  Canonicalizing here also
        # prevents aliases of the same physical camera from being counted as
        # independent views.
        segment_camera = _normalized_camera_name(segment.get("camera"))
        segment_ref = _identity_ref(segment)
        segment_object = str(segment.get("object_id", "") or "").strip().lower()
        if segment_ref != normalized_ref:
            continue
        if object_id and segment_object and segment_object != object_id:
            continue
        detections = segment.get("detections")
        candidates = detections if isinstance(detections, list) else [segment]
        for detection in candidates:
            if not isinstance(detection, dict):
                continue
            if _identity_ref(detection, fallback=segment_ref) != normalized_ref:
                continue
            detection_object = str(
                detection.get("object_id", segment_object) or ""
            ).strip().lower()
            if object_id and detection_object and detection_object != object_id:
                continue
            grounding = detection.get("grounding_3d")
            measurement = _dominant_plane_measurement(grounding)
            camera = segment_camera or _normalized_camera_name(
                detection.get("camera")
            )
            if (
                measurement is None
                or not camera
                or camera == acting_arm
            ):
                continue
            grouped.setdefault(camera, []).append(
                {
                    **measurement,
                    "score": _finite_float(detection.get("score")),
                    "rank": _integer(detection.get("rank")),
                }
            )
    return grouped


def _single_diagnostic_motion(
    calls: list[RecoveryToolCall],
    results: list[Any],
    *,
    post_action_feedback: bool = False,
) -> tuple[str, list[float]] | None:
    physical = [call for call in calls if call.tool_name != "reobserve_scene"]
    if len(physical) != 1:
        return None
    motion = physical[0]
    args = dict(motion.args or {})
    has_observation = any(
        call.tool_name == "reobserve_scene" for call in calls
    )
    if motion.tool_name == "move_ee_to_pose":
        # A planner-authored pose move is never grasp evidence, even when it
        # happens to be followed by an observation.  The marker is attached
        # only after the lifecycle guard has discarded all planner markers
        # and generated a move from runtime geometry.
        if args.get(_GRASP_DIAGNOSTIC_MOTION_MARKER) is not True:
            return None
    elif motion.tool_name == "lift_ee":
        # 正常动作后观察可以提供验证输入；身份、时序和物体随动检查继续执行。
        if (
            not has_observation
            and not post_action_feedback
            and args.get(_GRASP_DIAGNOSTIC_LIFT_MARKER) is not True
        ):
            return None
    else:
        return None
    index = calls.index(motion)
    result = results[index] if index < len(results) else None
    if result is None or not bool(getattr(result, "success", False)):
        return None
    result_tool_name = str(getattr(result, "tool_name", "") or "").strip()
    if result_tool_name and result_tool_name != motion.tool_name:
        return None
    details = getattr(result, "details", {}) or {}
    arm = str(details.get("arm", args.get("arm", "")) or "").strip().lower()
    delta = _xyz(details.get("observed_displacement_xyz"))
    if arm not in {"left", "right"} or delta is None:
        return None
    return arm, delta


def _camera_motion_evidence(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    anchor: list[float],
    ee_delta: list[float],
    acting_arm: str,
    limits: GraspMotionLimits,
) -> list[dict[str, Any]]:
    before_by_camera = _normalized_observations_by_camera(
        before.get("observations_by_camera"),
        acting_arm=acting_arm,
    )
    after_by_camera = _normalized_observations_by_camera(
        after.get("observations_by_camera"),
        acting_arm=acting_arm,
    )
    evidence: list[dict[str, Any]] = []
    ee_motion = _norm(ee_delta)
    maximum_coupled_residual = min(
        limits.maximum_coupled_residual_m,
        limits.residual_fraction_of_motion * ee_motion,
    )
    for camera in sorted(set(before_by_camera).intersection(after_by_camera)):
        pre_candidate = _select_unique_candidate(
            before_by_camera.get(camera),
            hypotheses=[anchor],
            maximum_distance=limits.anchor_association_m,
            margin=limits.candidate_margin_m,
        )
        if pre_candidate is None:
            continue
        pre_world = _xyz(pre_candidate.get("world_m"))
        if pre_world is None:
            continue
        predicted = _add(pre_world, ee_delta)
        post_candidate = _select_unique_candidate(
            after_by_camera.get(camera),
            hypotheses=[pre_world, predicted],
            maximum_distance=limits.transition_association_m,
            margin=limits.candidate_margin_m,
        )
        if post_candidate is None:
            continue
        post_world = _xyz(post_candidate.get("world_m"))
        pre_raw_world = _xyz(pre_candidate.get("raw_centroid_world_m"))
        post_raw_world = _xyz(post_candidate.get("raw_centroid_world_m"))
        if (
            post_world is None
            or pre_raw_world is None
            or post_raw_world is None
        ):
            continue
        observed_delta = _subtract(post_world, pre_world)
        coupled_residual = _distance(post_world, predicted)
        stationary_residual = _norm(observed_delta)
        cosine = _cosine(observed_delta, ee_delta)
        plane_coupled = bool(
            coupled_residual <= maximum_coupled_residual
            and stationary_residual - coupled_residual
            >= limits.minimum_hypothesis_margin_m
            and cosine is not None
            and cosine >= limits.minimum_direction_cosine
        )
        plane_stationary = bool(
            stationary_residual <= limits.maximum_stationary_residual_m
            and coupled_residual - stationary_residual
            >= limits.minimum_hypothesis_margin_m
        )
        raw_delta = _subtract(post_raw_world, pre_raw_world)
        raw_coupled_residual = _norm(_subtract(raw_delta, ee_delta))
        raw_stationary_residual = _norm(raw_delta)
        raw_cosine = _cosine(raw_delta, ee_delta)
        raw_coupled = bool(
            raw_coupled_residual <= maximum_coupled_residual
            and raw_stationary_residual - raw_coupled_residual
            >= limits.minimum_hypothesis_margin_m
            and raw_cosine is not None
            and raw_cosine >= limits.minimum_direction_cosine
        )
        raw_stationary = bool(
            raw_stationary_residual
            <= limits.maximum_stationary_residual_m
            and raw_coupled_residual - raw_stationary_residual
            >= limits.minimum_hypothesis_margin_m
        )
        geometry_continuous = _planar_geometry_continuous(
            pre_candidate.get("plane_extent_xy_m"),
            post_candidate.get("plane_extent_xy_m"),
            minimum_ratio=limits.minimum_planar_extent_ratio,
        )
        coupled = plane_coupled and raw_coupled and geometry_continuous
        stationary = (
            plane_stationary and raw_stationary and geometry_continuous
        )
        classification = (
            "coupled" if coupled else "stationary" if stationary else "ambiguous"
        )
        evidence.append(
            {
                "camera": camera,
                "classification": classification,
                "pre_world_m": _round_xyz(pre_world),
                "post_world_m": _round_xyz(post_world),
                "pre_raw_centroid_world_m": _round_xyz(pre_raw_world),
                "post_raw_centroid_world_m": _round_xyz(post_raw_world),
                "observed_displacement_xyz_m": _round_xyz(observed_delta),
                "observed_displacement_m": round(stationary_residual, 6),
                "coupled_motion_residual_m": round(coupled_residual, 6),
                "stationary_residual_m": round(stationary_residual, 6),
                "direction_cosine": (
                    None if cosine is None else round(cosine, 6)
                ),
                "raw_centroid_coupled_residual_m": round(
                    raw_coupled_residual,
                    6,
                ),
                "raw_centroid_stationary_residual_m": round(
                    raw_stationary_residual,
                    6,
                ),
                "raw_centroid_direction_cosine": (
                    None if raw_cosine is None else round(raw_cosine, 6)
                ),
                "planar_geometry_continuous": geometry_continuous,
                # Both candidates came from the exact instance binding and
                # passed the per-camera uniqueness margin above.  Keep these
                # facts explicit because the one-fixed-camera path relies on
                # identity continuity rather than semantic class matching.
                "identity_bound_to_grasp_transaction": True,
                "identity_candidate_unique": True,
                "pre_plane_extent_xy_m": pre_candidate.get(
                    "plane_extent_xy_m"
                ),
                "post_plane_extent_xy_m": post_candidate.get(
                    "plane_extent_xy_m"
                ),
            }
        )
    return evidence


def _strong_single_fixed_external_camera_support(
    *,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    before: dict[str, Any],
    pre: Any,
    post: Any,
    anchor: list[float],
    ee_delta: list[float],
    ee_motion: float,
    limits: GraspMotionLimits,
) -> dict[str, Any] | None:
    """Return a sole strong fixed-camera witness, otherwise fail closed.

    This is intentionally a narrow temporal exception to the ordinary
    two-view contract.  It cannot be used by either wrist camera, by a reused
    image, by weak motion, or when any camera supplies stationary evidence.
    """

    if len(positives) != 1 or negatives:
        return None
    witness = positives[0]
    if not _is_fixed_external_camera(witness.get("camera")):
        return None
    if not _distinct_observation_captures(pre, post):
        return None
    if ee_motion < limits.strong_single_view_minimum_ee_motion_m:
        return None
    attachment = before.get("held_object_to_tcp_attachment")
    if not valid_pending_held_object_to_tcp_attachment(
        attachment,
        grasp_candidate_id=before.get("grasp_candidate_id"),
        grasp_attempt_nonce=before.get("grasp_attempt_nonce"),
        grasp_attempt_step=before.get("grasp_attempt_step"),
    ):
        return None
    attachment_capture_world = _pending_attachment_capture_object_world_m(
        attachment
    )
    pre_world = _xyz(witness.get("pre_world_m"))
    post_world = _xyz(witness.get("post_world_m"))
    pre_raw_world = _xyz(witness.get("pre_raw_centroid_world_m"))
    post_raw_world = _xyz(witness.get("post_raw_centroid_world_m"))
    if any(
        value is None
        for value in (
            attachment_capture_world,
            pre_world,
            post_world,
            pre_raw_world,
            post_raw_world,
        )
    ):
        return None
    assert attachment_capture_world is not None
    assert pre_world is not None
    assert post_world is not None
    assert pre_raw_world is not None
    assert post_raw_world is not None
    attachment_post_world = _add(attachment_capture_world, ee_delta)
    pre_anchor_plane_residual = _distance(pre_world, anchor)
    pre_anchor_raw_residual = _distance(pre_raw_world, anchor)
    attachment_capture_anchor_residual = _distance(
        attachment_capture_world,
        anchor,
    )
    attachment_post_plane_residual = _distance(
        post_world,
        attachment_post_world,
    )
    attachment_post_raw_residual = _distance(
        post_raw_world,
        attachment_post_world,
    )
    object_motion = _finite_float(witness.get("observed_displacement_m"))
    coupled_residual = _finite_float(
        witness.get("coupled_motion_residual_m")
    )
    raw_object_motion = _finite_float(
        witness.get("raw_centroid_stationary_residual_m")
    )
    raw_coupled_residual = _finite_float(
        witness.get("raw_centroid_coupled_residual_m")
    )
    direction_cosine = _finite_float(witness.get("direction_cosine"))
    raw_direction_cosine = _finite_float(
        witness.get("raw_centroid_direction_cosine")
    )
    if any(
        value is None
        for value in (
            object_motion,
            coupled_residual,
            raw_object_motion,
            raw_coupled_residual,
            direction_cosine,
            raw_direction_cosine,
        )
    ):
        return None
    assert object_motion is not None
    assert coupled_residual is not None
    assert raw_object_motion is not None
    assert raw_coupled_residual is not None
    assert direction_cosine is not None
    assert raw_direction_cosine is not None
    if not (
            witness.get("planar_geometry_continuous") is True
            and witness.get("identity_bound_to_grasp_transaction") is True
            and witness.get("identity_candidate_unique") is True
            and pre_anchor_plane_residual
            <= limits.strong_single_view_maximum_anchor_residual_m
            and pre_anchor_raw_residual
            <= limits.strong_single_view_maximum_anchor_residual_m
            and attachment_capture_anchor_residual
            <= limits.strong_single_view_maximum_attachment_anchor_residual_m
            and attachment_post_plane_residual
            <= limits.strong_single_view_maximum_attachment_post_residual_m
            and attachment_post_raw_residual
            <= limits.strong_single_view_maximum_attachment_post_residual_m
            and object_motion
            >= limits.strong_single_view_minimum_object_motion_m
            and raw_object_motion
            >= limits.strong_single_view_minimum_object_motion_m
            and coupled_residual
            <= limits.strong_single_view_maximum_coupled_residual_m
            and raw_coupled_residual
            <= limits.strong_single_view_maximum_raw_coupled_residual_m
            and object_motion - coupled_residual
            >= limits.strong_single_view_minimum_hypothesis_margin_m
            and raw_object_motion - raw_coupled_residual
            >= limits.strong_single_view_minimum_hypothesis_margin_m
            and direction_cosine
            >= limits.strong_single_view_minimum_direction_cosine
            and raw_direction_cosine
            >= limits.strong_single_view_minimum_raw_direction_cosine
    ):
        return None
    return {
        **witness,
        "pre_anchor_plane_residual_m": round(
            pre_anchor_plane_residual,
            6,
        ),
        "pre_anchor_raw_residual_m": round(
            pre_anchor_raw_residual,
            6,
        ),
        "attachment_capture_anchor_residual_m": round(
            attachment_capture_anchor_residual,
            6,
        ),
        "attachment_post_plane_residual_m": round(
            attachment_post_plane_residual,
            6,
        ),
        "attachment_post_raw_residual_m": round(
            attachment_post_raw_residual,
            6,
        ),
        "attachment_capture_object_world_m": _round_xyz(
            attachment_capture_world
        ),
        "attachment_post_predicted_object_world_m": _round_xyz(
            attachment_post_world
        ),
    }


def _pending_attachment_capture_object_world_m(
    attachment: Any,
) -> list[float] | None:
    if not isinstance(attachment, dict):
        return None
    offset = _xyz(
        attachment.get("object_centroid_to_tcp_translation_tcp_m")
    )
    transform = pose7_to_matrix(attachment.get("capture_tcp_pose"))
    if offset is None or transform is None:
        return None
    try:
        return [
            float(transform[row, 3])
            - sum(
                float(transform[row, column]) * offset[column]
                for column in range(3)
            )
            for row in range(3)
        ]
    except (IndexError, TypeError, ValueError):
        return None


def _dominant_plane_measurement(
    grounding: Any,
) -> dict[str, Any] | None:
    if not isinstance(grounding, dict) or grounding.get("success") is False:
        return None
    raw_centroid = _xyz(grounding.get("centroid_world"))
    plane = grounding.get("dominant_plane_footprint")
    if raw_centroid is None or not isinstance(plane, dict):
        return None
    if plane.get("valid") is not True:
        return None
    lower = _xyz(plane.get("bbox_world_min"))
    upper = _xyz(plane.get("bbox_world_max"))
    plane_z = _finite_float(plane.get("plane_z_world_m"))
    if lower is None or upper is None or plane_z is None:
        return None
    extent_xy = [upper[index] - lower[index] for index in range(2)]
    if not all(math.isfinite(item) and item > 0.0 for item in extent_xy):
        return None
    return {
        "world_m": [
            (lower[0] + upper[0]) / 2.0,
            (lower[1] + upper[1]) / 2.0,
            plane_z,
        ],
        "raw_centroid_world_m": raw_centroid,
        "plane_extent_xy_m": _round_xy(extent_xy),
        "plane_bbox_world_min": _round_xyz(lower),
        "plane_bbox_world_max": _round_xyz(upper),
        "plane_inlier_ratio": _finite_float(plane.get("inlier_ratio")),
        "measurement_source": "dominant_horizontal_plane_and_raw_centroid",
    }


def _planar_geometry_continuous(
    first: Any,
    second: Any,
    *,
    minimum_ratio: float,
) -> bool:
    first_extent = _finite_vector(first, length=2)
    second_extent = _finite_vector(second, length=2)
    if first_extent is None or second_extent is None:
        return False
    for before, after in zip(first_extent, second_extent):
        if before <= 0.0 or after <= 0.0:
            return False
        if min(before, after) / max(before, after) < minimum_ratio:
            return False
    return True


def _select_unique_candidate(
    raw_candidates: Any,
    *,
    hypotheses: list[list[float]],
    maximum_distance: float,
    margin: float,
) -> dict[str, Any] | None:
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        world = _xyz(candidate.get("world_m"))
        if world is None:
            continue
        score = min(_distance(world, hypothesis) for hypothesis in hypotheses)
        if score <= maximum_distance:
            scored.append((score, candidate))
    scored.sort(key=lambda item: item[0])
    if not scored:
        return None
    observation_clusters: list[list[tuple[float, dict[str, Any]]]] = []
    for item in scored:
        for cluster in observation_clusters:
            if any(
                _same_planar_observation(item[1], member[1])
                for member in cluster
            ):
                cluster.append(item)
                break
        else:
            observation_clusters.append([item])
    representatives = sorted(
        (min(cluster, key=lambda item: item[0]) for cluster in observation_clusters),
        key=lambda item: item[0],
    )
    if (
        len(representatives) > 1
        and representatives[1][0] - representatives[0][0] < margin
    ):
        return None
    return representatives[0][1]


def _same_planar_observation(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    first_lower = _xyz(first.get("plane_bbox_world_min"))
    first_upper = _xyz(first.get("plane_bbox_world_max"))
    second_lower = _xyz(second.get("plane_bbox_world_min"))
    second_upper = _xyz(second.get("plane_bbox_world_max"))
    if (
        first_lower is None
        or first_upper is None
        or second_lower is None
        or second_upper is None
        or abs(first["world_m"][2] - second["world_m"][2]) > 0.008
    ):
        return False
    intersection = 1.0
    first_area = 1.0
    second_area = 1.0
    for index in range(2):
        intersection *= max(
            0.0,
            min(first_upper[index], second_upper[index])
            - max(first_lower[index], second_lower[index]),
        )
        first_area *= max(0.0, first_upper[index] - first_lower[index])
        second_area *= max(0.0, second_upper[index] - second_lower[index])
    minimum_area = min(first_area, second_area)
    return bool(
        minimum_area > 1e-8
        and intersection / minimum_area >= 0.5
    )


def _arm_snapshot(value: Any, arm: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    arms = value.get("arms")
    state = arms.get(arm) if isinstance(arms, dict) else None
    return state if isinstance(state, dict) else None


def _same_grasp_transaction(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return bool(
        first.get("phase") == "grasp_candidate"
        and second.get("phase") == "grasp_candidate"
        and str(first.get("held_instance_id", "") or "")
        == str(second.get("held_instance_id", "") or "")
        and str(first.get("grasp_candidate_id", "") or "")
        == str(second.get("grasp_candidate_id", "") or "")
        and bool(str(first.get("grasp_attempt_nonce", "") or "").strip())
        and str(first.get("grasp_attempt_nonce", "") or "")
        == str(second.get("grasp_attempt_nonce", "") or "")
    )


def _cross_view_positions_consistent(
    evidence: list[dict[str, Any]],
    *,
    maximum_spread_m: float,
) -> bool:
    if len(evidence) < 2:
        return True
    for key in ("pre_world_m", "post_world_m"):
        positions = [
            position
            for item in evidence
            if (position := _xyz(item.get(key))) is not None
        ]
        if len(positions) != len(evidence):
            return False
        for index, first in enumerate(positions):
            if any(
                _distance(first, second) > maximum_spread_m
                for second in positions[index + 1 :]
            ):
                return False
    return True


def _fresh_snapshot_pair(first: Any, second: Any) -> bool:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    first_step = _integer(first.get("env_step"))
    second_step = _integer(second.get("env_step"))
    return (
        first_step is not None
        and second_step is not None
        and second_step > first_step
    )


def _distinct_observation_captures(first: Any, second: Any) -> bool:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    first_capture = _integer(first.get("observation_capture_id"))
    second_capture = _integer(second.get("observation_capture_id"))
    return bool(
        first_capture is not None
        and second_capture is not None
        and second_capture > first_capture
    )


def _identity_ref(value: dict[str, Any], *, fallback: str = "") -> str:
    return str(
        value.get("instance_ref")
        or value.get("query_instance_ref")
        or value.get("track_id")
        or value.get("instance_id")
        or fallback
        or ""
    ).strip().lower()


def _common_supported_cameras(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    acting_arm: str,
) -> list[str]:
    before_by_camera = _normalized_observations_by_camera(
        before.get("observations_by_camera"),
        acting_arm=acting_arm,
    )
    after_by_camera = _normalized_observations_by_camera(
        after.get("observations_by_camera"),
        acting_arm=acting_arm,
    )
    return sorted(set(before_by_camera).intersection(after_by_camera))


def _normalized_observations_by_camera(
    value: Any,
    *,
    acting_arm: str,
) -> dict[str, list[dict[str, Any]]]:
    raw = value if isinstance(value, dict) else {}
    normalized_arm = str(acting_arm or "").strip().lower()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_camera, raw_observations in raw.items():
        camera = _normalized_camera_name(raw_camera)
        if not camera or camera == normalized_arm:
            continue
        observations = (
            raw_observations
            if isinstance(raw_observations, list)
            else []
        )
        supported = [
            item for item in observations if isinstance(item, dict)
        ]
        if supported:
            grouped.setdefault(camera, []).extend(supported)
    return grouped


def _normalized_camera_name(value: Any) -> str:
    text = (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    aliases = {
        "head": "head",
        "head_camera": "head",
        "camera_head": "head",
        "cam_head": "head",
        "third": "third",
        "third_camera": "third",
        "camera_third": "third",
        "cam_third": "third",
        "third_view": "third",
        "left": "left",
        "left_camera": "left",
        "camera_left": "left",
        "left_wrist": "left",
        "left_wrist_camera": "left",
        "wrist_left": "left",
        "cam_left_wrist": "left",
        "right": "right",
        "right_camera": "right",
        "camera_right": "right",
        "right_wrist": "right",
        "right_wrist_camera": "right",
        "wrist_right": "right",
        "cam_right_wrist": "right",
    }
    return aliases.get(text, "")


def _is_fixed_external_camera(value: Any) -> bool:
    # ``head`` and ``third`` are scene cameras with fixed extrinsics in the
    # RMBench observation contract.  Both arm-labelled cameras are mounted on
    # robot wrists and are deliberately excluded from the one-view proof.
    return _normalized_camera_name(value) in {"head", "third"}


def _xyz(value: Any) -> list[float] | None:
    return _finite_vector(value, length=3)


def _finite_vector(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    try:
        result = [float(value[index]) for index in range(length)]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _add(first: list[float], second: list[float]) -> list[float]:
    return [first[index] + second[index] for index in range(3)]


def _subtract(first: list[float], second: list[float]) -> list[float]:
    return [first[index] - second[index] for index in range(3)]


def _norm(value: list[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _distance(first: list[float], second: list[float]) -> float:
    return _norm(_subtract(first, second))


def _cosine(first: list[float], second: list[float]) -> float | None:
    denominator = _norm(first) * _norm(second)
    if denominator <= 1e-12:
        return None
    return sum(first[index] * second[index] for index in range(3)) / denominator


def _round_xyz(value: list[float]) -> list[float]:
    return [round(float(item), 6) for item in value]


def _round_xy(value: list[float]) -> list[float]:
    return [round(float(item), 6) for item in value[:2]]

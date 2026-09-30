from __future__ import annotations

from copy import deepcopy
from unittest import mock

import numpy as np

from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.grasp_attachment_contract import (
    apply_grasp_motion_validation,
    capture_grasp_motion_snapshot,
    matched_grasp_motion_validation,
    pending_grasp_observation_attempted,
    validate_grasp_motion_effect,
    validate_pending_grasp_observation_effect,
)
from policy.roboharn_evo.agent.operation_candidates import (
    manipulation_state_allows_transport,
)
from policy.roboharn_evo.agent.recovery.tool_specs import (
    RecoveryToolCall,
    RecoveryToolResult,
)
from policy.roboharn_evo.tests.test_debug_recovery_pure_control import (
    DummyConfig,
    make_card,
    make_snapshot,
)


_ARM = "right"
_INSTANCE_ID = "track_0001"
_PREGRASP_WORLD_M = [0.114266, -0.218103, 0.774719]
_EE_DELTA_M = [0.000229, -0.000319, 0.021669]
_EE_PRE_WORLD_M = [0.112167, -0.214003, 0.908921]
_GRASP_ATTEMPT_NONCE = (
    "right:track_0001:rgbd_volume:grasp:right:001:0"
)

# Exact diagnostic-lift measurements from the 2026-08-05 put_back_block v6
# trace.  Only the fixed third camera retained the identity-bound object in
# both captures; its plane and raw centroid independently followed the right
# TCP through the 26.5 mm lift.
_V6_PREGRASP_WORLD_M = [0.101166, -0.193414, 0.77118]
_V6_EE_PRE_WORLD_M = [0.101822, -0.192413, 0.904063]
_V6_EE_DELTA_M = [0.000303, -0.000195, 0.026484]
_V6_RAW_PRE_WORLD_M = [0.104262, -0.189278, 0.776428]
_V6_RAW_POST_WORLD_M = [0.107187, -0.196005, 0.8035]
_V6_PLANE_PRE_WORLD_M = [0.105654, -0.199303, 0.782835]
_V6_PLANE_POST_WORLD_M = [0.106887, -0.19771, 0.809749]
_V6_PLANE_PRE_EXTENT_M = [0.034002, 0.036226, 0.00525]
_V6_PLANE_POST_EXTENT_M = [0.029528, 0.045954, 0.006088]
_V6_ATTACHMENT_OFFSET_TCP_M = [-0.01288, -0.000151, -0.001207]
_V6_ATTACHMENT_CAPTURE_TCP_POSE = [
    0.101557,
    -0.192268,
    0.78406,
    0.38174,
    0.59429,
    0.38328,
    -0.59514,
]


def _manipulation_state() -> dict:
    return {
        _ARM: {
            "phase": "grasp_candidate",
            "held_instance_id": _INSTANCE_ID,
            "holding_confirmed": False,
            "transport_authorized": False,
            "grasp_candidate_id": "rgbd_volume:grasp:right:001",
            "grasp_attempt_step": 0,
            "grasp_attempt_nonce": _GRASP_ATTEMPT_NONCE,
            "pregrasp_object_world_m": list(_PREGRASP_WORLD_M),
            "held_object_perception_descriptor": {
                "object_id": "block",
                "text_prompt": "red block",
            },
        }
    }


def _robot_state(
    *,
    after: bool = False,
    ee_delta_m: list[float] | None = None,
) -> dict:
    delta = np.asarray(
        _EE_DELTA_M if ee_delta_m is None else ee_delta_m,
        dtype=np.float64,
    )
    xyz = np.asarray(_EE_PRE_WORLD_M, dtype=np.float64)
    if after:
        xyz = xyz + delta
    return {_ARM: {"gripper": 0.0, "xyz": xyz.tolist()}}


def _detection(
    camera: str,
    world_m: list[float],
    *,
    rank: int,
    score: float,
    plane_world_m: list[float] | None = None,
    plane_extent_m: list[float] | None = None,
) -> dict:
    plane_world = list(world_m if plane_world_m is None else plane_world_m)
    plane_extent = list(
        [0.035, 0.04, 0.006]
        if plane_extent_m is None
        else plane_extent_m
    )
    return {
        "camera": camera,
        "rank": rank,
        "score": score,
        "object_id": "block",
        "instance_ref": _INSTANCE_ID,
        "identity_binding_required": True,
        "grounding_3d": {
            "success": True,
            "camera": camera,
            "centroid_world": list(world_m),
            "dominant_plane_footprint": {
                "valid": True,
                "source": "dominant_horizontal_z_band",
                "plane_z_world_m": plane_world[2],
                "inlier_ratio": 0.55,
                "bbox_world_min": [
                    plane_world[0] - plane_extent[0] / 2.0,
                    plane_world[1] - plane_extent[1] / 2.0,
                    plane_world[2] - plane_extent[2] / 2.0,
                ],
                "bbox_world_max": [
                    plane_world[0] + plane_extent[0] / 2.0,
                    plane_world[1] + plane_extent[1] / 2.0,
                    plane_world[2] + plane_extent[2] / 2.0,
                ],
                "extent_m": plane_extent,
            },
        },
    }


def _preprocess(
    *,
    env_step: int,
    target_by_camera: dict[str, list[float]],
    include_distractor: bool = True,
    target_plane_by_camera: dict[str, list[float]] | None = None,
    target_plane_extent_by_camera: dict[str, list[float]] | None = None,
    target_variants_by_camera: dict[str, list[dict]] | None = None,
    observation_generation: int | None = None,
    observation_capture_id: int | None = None,
) -> dict:
    segmentation = []
    for camera, target in target_by_camera.items():
        detections = []
        if include_distractor:
            detections.append(
                _detection(
                    camera,
                    [-0.2482, -0.1044, 0.7763],
                    rank=0,
                    score=0.65,
                )
            )
        detections.append(
            _detection(
                camera,
                target,
                rank=len(detections),
                score=0.35,
                plane_world_m=(target_plane_by_camera or {}).get(camera),
                plane_extent_m=(
                    target_plane_extent_by_camera or {}
                ).get(camera),
            )
        )
        for variant in (target_variants_by_camera or {}).get(camera, []):
            detections.append(
                _detection(
                    camera,
                    variant["world_m"],
                    rank=len(detections),
                    score=float(variant.get("score", 0.2)),
                    plane_world_m=variant.get("plane_world_m"),
                    plane_extent_m=variant.get("plane_extent_m"),
                )
            )
        segmentation.append(
            {
                "success": True,
                "camera": camera,
                "object_id": "block",
                "text_prompt": "red block",
                "instance_ref": _INSTANCE_ID,
                "identity_binding_required": True,
                "detections": detections,
            }
        )
    result = {
        "env_step": env_step,
        "observation_generation": (
            env_step
            if observation_generation is None
            else observation_generation
        ),
        "segmentation": segmentation,
    }
    if observation_capture_id is not None:
        result["observation_capture_id"] = observation_capture_id
    return result


def _effect_inputs(
    ee_delta_m: list[float] | None = None,
) -> tuple[list[RecoveryToolCall], list[RecoveryToolResult]]:
    ee_delta = list(_EE_DELTA_M if ee_delta_m is None else ee_delta_m)
    calls = [
        RecoveryToolCall(
            tool_name="lift_ee",
            args={"arm": _ARM, "distance": 0.025, "steps": 2},
        ),
        RecoveryToolCall(tool_name="reobserve_scene", args={}),
    ]
    results = [
        RecoveryToolResult(
            tool_name="lift_ee",
            success=True,
            details={
                "arm": _ARM,
                "axis": "z",
                "observed_displacement_xyz": ee_delta,
                "observed_displacement_m": float(
                    np.linalg.norm(ee_delta)
                ),
                "observed_pose": [
                    0.112396,
                    -0.214322,
                    0.93059,
                    0.296898,
                    0.648968,
                    0.26215,
                    -0.649592,
                ],
            },
        ),
        RecoveryToolResult(
            tool_name="reobserve_scene",
            success=True,
            details={"step_count": 12},
        ),
    ]
    return calls, results


def _v6_single_camera_motion_inputs(
    *,
    camera: str = "third",
    pre_capture_id: int = 11,
    post_capture_id: int = 13,
    post_plane_world_m: list[float] | None = None,
    candidate_offset_m: list[float] | None = None,
    include_pending_attachment: bool = True,
    attachment_capture_offset_m: list[float] | None = None,
) -> tuple[dict, dict, list[RecoveryToolCall], list[RecoveryToolResult]]:
    state = _manipulation_state()
    state[_ARM].update(
        {
            "grasp_attempt_step": 9,
            "pregrasp_object_world_m": list(_V6_PREGRASP_WORLD_M),
        }
    )
    if include_pending_attachment:
        attachment_capture_offset = np.asarray(
            [0.0, 0.0, 0.0]
            if attachment_capture_offset_m is None
            else attachment_capture_offset_m,
            dtype=np.float64,
        )
        if np.linalg.norm(attachment_capture_offset) == 0.0:
            attachment_offset = list(_V6_ATTACHMENT_OFFSET_TCP_M)
            attachment_pose = list(_V6_ATTACHMENT_CAPTURE_TCP_POSE)
        else:
            attachment_object = (
                np.asarray(_V6_PREGRASP_WORLD_M)
                + attachment_capture_offset
            )
            attachment_offset = [0.0, 0.0, 0.02]
            attachment_pose = [
                attachment_object[0],
                attachment_object[1],
                attachment_object[2] + 0.02,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        state[_ARM]["held_object_to_tcp_attachment"] = {
            "object_proxy_frame": "tcp_aligned_at_capture",
            "object_centroid_to_tcp_translation_tcp_m": attachment_offset,
            "capture_tcp_pose": [float(value) for value in attachment_pose],
            "grasp_candidate_id": state[_ARM]["grasp_candidate_id"],
            "grasp_attempt_nonce": state[_ARM]["grasp_attempt_nonce"],
            "capture_step": state[_ARM]["grasp_attempt_step"],
            "source": "pending_lift_verification",
        }
    candidate_offset = np.asarray(
        [0.0, 0.0, 0.0]
        if candidate_offset_m is None
        else candidate_offset_m,
        dtype=np.float64,
    )
    raw_pre_world = (
        np.asarray(_V6_RAW_PRE_WORLD_M) + candidate_offset
    ).tolist()
    raw_post_world = (
        np.asarray(_V6_RAW_POST_WORLD_M) + candidate_offset
    ).tolist()
    plane_pre_world = (
        np.asarray(_V6_PLANE_PRE_WORLD_M) + candidate_offset
    ).tolist()
    resolved_post_plane = (
        np.asarray(
            _V6_PLANE_POST_WORLD_M
            if post_plane_world_m is None
            else post_plane_world_m
        )
        + candidate_offset
    ).tolist()
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=9,
            observation_generation=7,
            observation_capture_id=pre_capture_id,
            target_by_camera={camera: raw_pre_world},
            target_plane_by_camera={camera: plane_pre_world},
            target_plane_extent_by_camera={
                camera: list(_V6_PLANE_PRE_EXTENT_M)
            },
        ),
        manipulation_state=state,
        robot_state={
            _ARM: {
                "gripper": 0.0,
                "xyz": list(_V6_EE_PRE_WORLD_M),
            }
        },
    )
    post_ee = (
        np.asarray(_V6_EE_PRE_WORLD_M) + np.asarray(_V6_EE_DELTA_M)
    ).tolist()
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=11,
            observation_generation=8,
            observation_capture_id=post_capture_id,
            target_by_camera={camera: raw_post_world},
            target_plane_by_camera={camera: resolved_post_plane},
            target_plane_extent_by_camera={
                camera: list(_V6_PLANE_POST_EXTENT_M)
            },
        ),
        manipulation_state=state,
        robot_state={_ARM: {"gripper": 0.0, "xyz": post_ee}},
    )
    calls, results = _effect_inputs(list(_V6_EE_DELTA_M))
    return pre, post, calls, results


def _pending_after_ambiguous_lift() -> dict:
    state = _manipulation_state()
    arm_state = state[_ARM]
    postlift_ee = (
        np.asarray(_EE_PRE_WORLD_M) + np.asarray(_EE_DELTA_M)
    ).tolist()
    arm_state.update(
        {
            "operation_action_mode": "grasp",
            "updated_step": 12,
            "diagnostic_lift_evidence": {
                "observed_lift_m": float(np.linalg.norm(_EE_DELTA_M)),
                "observed_displacement_xyz_m": list(_EE_DELTA_M),
                "lift_env_step": 12,
                "lift_observation_generation": 12,
                "postlift_ee_world_m": postlift_ee,
                "track_status": "multiview_motion_ambiguous",
                "track_stability": "attachment_not_proven",
            },
            "held_object_to_tcp_attachment": {
                "object_proxy_frame": "tcp_aligned_at_capture",
                "object_centroid_to_tcp_translation_tcp_m": [
                    0.0,
                    0.0,
                    0.02,
                ],
                "capture_tcp_pose": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "grasp_candidate_id": arm_state["grasp_candidate_id"],
                "grasp_attempt_nonce": arm_state["grasp_attempt_nonce"],
                "capture_step": arm_state["grasp_attempt_step"],
                "source": "pending_lift_verification",
            },
        }
    )
    return state


def _static_reobserve_inputs() -> tuple[
    list[RecoveryToolCall],
    list[RecoveryToolResult],
]:
    return (
        [RecoveryToolCall(tool_name="reobserve_scene", args={})],
        [
            RecoveryToolResult(
                tool_name="reobserve_scene",
                success=True,
            )
        ],
    )


def test_v4_multiview_coupled_motion_confirms_holding() -> None:
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera={
                "head": [0.128617, -0.235195, 0.767608],
                "third": [0.121657, -0.205207, 0.774948],
            },
            target_plane_by_camera={
                "head": [0.129889, -0.237322, 0.782334],
                "third": [0.119426, -0.210683, 0.782992],
            },
            target_plane_extent_by_camera={
                "head": [0.009638, 0.015895, 0.007886],
                "third": [0.035809, 0.045188, 0.007328],
            },
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera={
                "head": [0.129994, -0.236269, 0.789851],
                "third": [0.123956, -0.208401, 0.793879],
            },
            target_plane_by_camera={
                "head": [0.129714, -0.236694, 0.803710],
                "third": [0.124864, -0.217599, 0.804667],
            },
            target_plane_extent_by_camera={
                "head": [0.012742, 0.015905, 0.007921],
                "third": [0.027861, 0.037111, 0.006070],
            },
            target_variants_by_camera={
                "third": [
                    {
                        "world_m": [
                            0.124884,
                            -0.225284,
                            0.806306,
                        ],
                        "plane_world_m": [
                            0.124572,
                            -0.225496,
                            0.804748,
                        ],
                        "plane_extent_m": [
                            0.027895,
                            0.021328,
                            0.000453,
                        ],
                        "score": 0.126953,
                    }
                ]
            },
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs()

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is True
    assert validation["arm"] == _ARM
    assert validation["held_instance_id"] == _INSTANCE_ID
    assert validation["supporting_camera_count"] == 2
    assert set(validation["supporting_cameras"]) == {"head", "third"}
    np.testing.assert_allclose(
        validation["verified_object_world_m"],
        np.asarray(_PREGRASP_WORLD_M) + np.asarray(_EE_DELTA_M),
        atol=1e-6,
    )
    assert max(
        item["coupled_motion_residual_m"]
        for item in validation["camera_evidence"]
    ) < 0.01
    assert min(
        item["direction_cosine"]
        for item in validation["camera_evidence"]
    ) > 0.90


def test_v6_strong_single_fixed_camera_motion_confirms_holding() -> None:
    pre, post, calls, results = _v6_single_camera_motion_inputs()

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is True
    assert validation["supporting_cameras"] == ["third"]
    assert validation["single_view_fast_path"] is True
    assert validation["pre_observation_capture_id"] == 11
    assert validation["post_observation_capture_id"] == 13
    assert validation["evidence_source"] == (
        "identity_bound_single_fixed_external_camera_"
        "strong_world_motion_coupling"
    )
    evidence = validation["camera_evidence"][0]
    assert evidence["coupled_motion_residual_m"] == 0.002061
    assert evidence["raw_centroid_coupled_residual_m"] == 0.007063
    assert evidence["direction_cosine"] == 0.997206
    assert evidence["raw_centroid_direction_cosine"] == 0.968065
    assert evidence["planar_geometry_continuous"] is True
    consistency = validation["single_view_attachment_consistency"]
    assert consistency["pre_anchor_plane_residual_m"] < 0.014
    assert consistency["pre_anchor_raw_residual_m"] < 0.008
    assert consistency["attachment_capture_anchor_residual_m"] < 0.001
    assert consistency["attachment_post_plane_residual_m"] < 0.014
    assert consistency["attachment_post_raw_residual_m"] < 0.009

    effect = apply_grasp_motion_validation({}, validation)
    assert effect["authority"] == (
        "runtime_single_fixed_camera_grasp_motion"
    )


def test_runtime_diagnostic_lift_uses_physical_snapshot_when_reobserve_is_ablated() -> None:
    pre, post, calls, results = _v6_single_camera_motion_inputs()
    lift_args = dict(calls[0].args)
    lift_args["_runtime_grasp_diagnostic_lift"] = True

    validation = validate_grasp_motion_effect(
        calls=[
            RecoveryToolCall(tool_name="lift_ee", args=lift_args)
        ],
        results=[results[0]],
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is True
    assert validation["single_view_fast_path"] is True


def test_runtime_reverse_ingress_motion_uses_observed_displacement() -> None:
    pre, post, _, legacy_results = _v6_single_camera_motion_inputs()
    call = RecoveryToolCall(
        tool_name="move_ee_to_pose",
        args={
            "arm": _ARM,
            "target_pose": [
                0.101519,
                -0.192608,
                0.930547,
                0.5,
                -0.5,
                0.5,
                -0.5,
            ],
            "max_translation": 0.01,
            "steps": 3,
            "_runtime_grasp_diagnostic_motion": True,
        },
    )
    result = RecoveryToolResult(
        tool_name="move_ee_to_pose",
        success=True,
        details={
            **dict(legacy_results[0].details),
            "observed_displacement_xyz": list(_V6_EE_DELTA_M),
        },
    )

    validation = validate_grasp_motion_effect(
        calls=[call],
        results=[result],
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is True
    assert validation["single_view_fast_path"] is True


def test_unmarked_planner_pose_move_is_never_diagnostic_evidence() -> None:
    pre, post, _, legacy_results = _v6_single_camera_motion_inputs()
    calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_pose",
            args={
                "arm": _ARM,
                "target_pose": [
                    0.101519,
                    -0.192608,
                    0.930547,
                    0.5,
                    -0.5,
                    0.5,
                    -0.5,
                ],
            },
        ),
        RecoveryToolCall(tool_name="reobserve_scene", args={}),
    ]
    results = [
        RecoveryToolResult(
            tool_name="move_ee_to_pose",
            success=True,
            details={
                **dict(legacy_results[0].details),
                "observed_displacement_xyz": list(_V6_EE_DELTA_M),
            },
        ),
        RecoveryToolResult(
            tool_name="reobserve_scene",
            success=True,
            details={"step_count": 12},
        ),
    ]

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation == {"applicable": False, "verified": None}


def test_single_fixed_camera_requires_exact_pending_attachment() -> None:
    pre, post, calls, results = _v6_single_camera_motion_inputs(
        include_pending_attachment=False,
    )

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert validation["supporting_cameras"] == ["third"]
    assert "single_view_fast_path" not in validation


def test_single_fixed_camera_rejects_inconsistent_pending_attachment() -> None:
    pre, post, calls, results = _v6_single_camera_motion_inputs(
        attachment_capture_offset_m=[0.03, 0.0, 0.0],
    )

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert validation["supporting_cameras"] == ["third"]
    assert "single_view_fast_path" not in validation


def test_single_fixed_camera_rejects_temporally_coupled_off_anchor_mask() -> None:
    pre, post, calls, results = _v6_single_camera_motion_inputs(
        candidate_offset_m=[0.025, 0.0, 0.0],
    )

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    # The displaced mask still follows the EE closely enough to satisfy the
    # temporal coupling classifier, but it is not the grasp-time object.
    assert validation["supporting_cameras"] == ["third"]
    assert validation["camera_evidence"][0]["classification"] == "coupled"
    assert validation["verified"] is None
    assert "single_view_fast_path" not in validation


def test_single_fixed_camera_reused_capture_cannot_confirm_holding() -> None:
    pre, post, calls, results = _v6_single_camera_motion_inputs(
        pre_capture_id=11,
        post_capture_id=11,
    )

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert validation["supporting_cameras"] == ["third"]
    assert "single_view_fast_path" not in validation


def test_single_wrist_camera_motion_cannot_confirm_holding() -> None:
    for camera in ("right_wrist", "left_wrist"):
        pre, post, calls, results = _v6_single_camera_motion_inputs(
            camera=camera,
        )

        validation = validate_grasp_motion_effect(
            calls=calls,
            results=results,
            pre=pre,
            post=post,
        )

        assert validation["verified"] is None
        assert "single_view_fast_path" not in validation


def test_single_fixed_camera_loose_coupling_cannot_use_fast_path() -> None:
    loose_post_plane = list(_V6_PLANE_POST_WORLD_M)
    loose_post_plane[0] += 0.004
    pre, post, calls, results = _v6_single_camera_motion_inputs(
        post_plane_world_m=loose_post_plane,
    )

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert validation["supporting_cameras"] == ["third"]
    assert (
        validation["camera_evidence"][0]["coupled_motion_residual_m"]
        > 0.003
    )
    assert "single_view_fast_path" not in validation


def test_single_fixed_camera_positive_is_blocked_by_negative_camera() -> None:
    state = _manipulation_state()
    state[_ARM].update(
        {
            "grasp_attempt_step": 9,
            "pregrasp_object_world_m": list(_V6_PREGRASP_WORLD_M),
        }
    )
    cameras = ("head", "third")
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=9,
            observation_generation=7,
            observation_capture_id=11,
            target_by_camera={
                camera: list(_V6_RAW_PRE_WORLD_M) for camera in cameras
            },
            target_plane_by_camera={
                camera: list(_V6_PLANE_PRE_WORLD_M) for camera in cameras
            },
            target_plane_extent_by_camera={
                camera: list(_V6_PLANE_PRE_EXTENT_M) for camera in cameras
            },
        ),
        manipulation_state=state,
        robot_state={
            _ARM: {"gripper": 0.0, "xyz": list(_V6_EE_PRE_WORLD_M)}
        },
    )
    post_ee = (
        np.asarray(_V6_EE_PRE_WORLD_M) + np.asarray(_V6_EE_DELTA_M)
    ).tolist()
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=11,
            observation_generation=8,
            observation_capture_id=13,
            target_by_camera={
                "head": list(_V6_RAW_PRE_WORLD_M),
                "third": list(_V6_RAW_POST_WORLD_M),
            },
            target_plane_by_camera={
                "head": list(_V6_PLANE_PRE_WORLD_M),
                "third": list(_V6_PLANE_POST_WORLD_M),
            },
            target_plane_extent_by_camera={
                "head": list(_V6_PLANE_PRE_EXTENT_M),
                "third": list(_V6_PLANE_POST_EXTENT_M),
            },
        ),
        manipulation_state=state,
        robot_state={_ARM: {"gripper": 0.0, "xyz": post_ee}},
    )
    calls, results = _effect_inputs(list(_V6_EE_DELTA_M))

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert validation["supporting_cameras"] == ["third"]
    assert validation["negative_camera_count"] == 1
    assert validation["failure_kind"] == "conflicting_multiview_motion"
    assert "single_view_fast_path" not in validation


def test_stationary_same_semantic_detections_cannot_confirm_holding() -> None:
    targets = {
        "head": [0.128617, -0.235195, 0.767608],
        "third": [0.121657, -0.205207, 0.774948],
    }
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera=targets,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera={
                "head": [0.1288, -0.2350, 0.7677],
                "third": [0.1215, -0.2054, 0.7750],
            },
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs()

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is False
    assert validation["failure_kind"] == "object_stationary_during_lift"
    assert validation["negative_camera_count"] == 2


def test_single_camera_motion_without_capture_identity_remains_undecided() -> None:
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera={
                "head": [0.128617, -0.235195, 0.767608],
            },
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera={
                "head": [0.129994, -0.236269, 0.789851],
            },
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs()

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is None
    assert validation["failure_kind"] == "insufficient_independent_views"
    assert validation["supporting_cameras"] == ["head"]
    assert "single_view_fast_path" not in validation


def test_head_and_acting_wrist_do_not_count_as_two_independent_views() -> None:
    before_process = _preprocess(
        env_step=10,
        target_by_camera={
            "head_camera": [0.128617, -0.235195, 0.767608],
            "right_wrist": [0.121657, -0.205207, 0.774948],
        },
    )
    after_process = _preprocess(
        env_step=12,
        target_by_camera={
            "head_camera": [0.129994, -0.236269, 0.789851],
            "right_wrist": [0.123956, -0.208401, 0.793879],
        },
    )
    # Nested detection labels cannot create another view: the enclosing
    # segment camera is authoritative and aliases normalize to one key.
    for process in (before_process, after_process):
        for segment in process["segmentation"]:
            for detection in segment["detections"]:
                detection["camera"] = "third"
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=before_process,
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=after_process,
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs()

    assert set(
        pre["arms"][_ARM]["observations_by_camera"]
    ) == {"head"}
    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is None
    assert validation["supporting_cameras"] == ["head"]
    assert "single_view_fast_path" not in validation


def test_raw_centroid_motion_without_stable_plane_cannot_confirm_grasp() -> None:
    pre_process = _preprocess(
        env_step=10,
        target_by_camera={
            "head": [0.128617, -0.235195, 0.767608],
            "third": [0.121657, -0.205207, 0.774948],
        },
    )
    post_process = _preprocess(
        env_step=12,
        target_by_camera={
            "head": [0.129994, -0.236269, 0.789851],
            "third": [0.123956, -0.208401, 0.793879],
        },
    )
    for process in (pre_process, post_process):
        for segment in process["segmentation"]:
            for detection in segment["detections"]:
                detection["grounding_3d"].pop(
                    "dominant_plane_footprint",
                    None,
                )
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=pre_process,
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=post_process,
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs()

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is False
    assert validation["verified"] is None
    assert validation["reason"] == "insufficient_common_nonacting_camera_views"
    assert validation["available_camera_count"] == 0


def test_partial_motion_outside_absolute_residual_cannot_confirm_grasp() -> None:
    before = {
        "head": [0.128617, -0.235195, 0.767608],
        "third": [0.121657, -0.205207, 0.774948],
    }
    partial_delta = np.asarray([0.0, 0.0, 0.055])
    after = {
        camera: (np.asarray(world) + partial_delta).tolist()
        for camera, world in before.items()
    }
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera=before,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera=after,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True, ee_delta_m=[0.0, 0.0, 0.08]),
    )
    calls, results = _effect_inputs([0.0, 0.0, 0.08])

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is None
    assert validation["supporting_camera_count"] == 0
    assert all(
        item["coupled_motion_residual_m"] > 0.01
        for item in validation["camera_evidence"]
    )


def test_large_plane_shape_change_cannot_confirm_grasp() -> None:
    before = {
        "head": [0.128617, -0.235195, 0.767608],
        "third": [0.121657, -0.205207, 0.774948],
    }
    after = {
        camera: (np.asarray(world) + np.asarray(_EE_DELTA_M)).tolist()
        for camera, world in before.items()
    }
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera=before,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera=after,
            target_plane_extent_by_camera={
                "head": [0.01, 0.04, 0.006],
                "third": [0.01, 0.04, 0.006],
            },
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs()

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert all(
        item["planar_geometry_continuous"] is False
        for item in validation["camera_evidence"]
    )


def test_nearby_nonoverlapping_plane_candidates_remain_ambiguous() -> None:
    anchor = np.asarray(_PREGRASP_WORLD_M, dtype=np.float64)
    first = (anchor + np.asarray([0.006, 0.0, 0.0])).tolist()
    second = (anchor + np.asarray([-0.006, 0.0, 0.0])).tolist()
    moved_first = (np.asarray(first) + np.asarray(_EE_DELTA_M)).tolist()
    moved_second = (np.asarray(second) + np.asarray(_EE_DELTA_M)).tolist()
    variants_before = {
        camera: [
            {
                "world_m": second,
                "plane_world_m": second,
                "plane_extent_m": [0.008, 0.02, 0.006],
            }
        ]
        for camera in ("head", "third")
    }
    variants_after = {
        camera: [
            {
                "world_m": moved_second,
                "plane_world_m": moved_second,
                "plane_extent_m": [0.008, 0.02, 0.006],
            }
        ]
        for camera in ("head", "third")
    }
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera={"head": first, "third": first},
            target_plane_by_camera={"head": first, "third": first},
            target_plane_extent_by_camera={
                "head": [0.008, 0.02, 0.006],
                "third": [0.008, 0.02, 0.006],
            },
            target_variants_by_camera=variants_before,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera={
                "head": moved_first,
                "third": moved_first,
            },
            target_plane_by_camera={
                "head": moved_first,
                "third": moved_first,
            },
            target_plane_extent_by_camera={
                "head": [0.008, 0.02, 0.006],
                "third": [0.008, 0.02, 0.006],
            },
            target_variants_by_camera=variants_after,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs()

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert validation["supporting_camera_count"] == 0
    assert validation["camera_evidence"] == []


def test_cross_view_positions_must_describe_one_world_object() -> None:
    anchor = np.asarray(_PREGRASP_WORLD_M, dtype=np.float64)
    before = {
        "head": (anchor + np.asarray([-0.04, 0.0, 0.0])).tolist(),
        "third": (anchor + np.asarray([0.04, 0.0, 0.0])).tolist(),
    }
    after = {
        camera: (np.asarray(world) + np.asarray(_EE_DELTA_M)).tolist()
        for camera, world in before.items()
    }
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera=before,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera=after,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs()

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert validation["failure_kind"] == (
        "cross_view_identity_position_inconsistent"
    )
    assert validation["cross_view_position_consistent"] is False


def test_lift_result_must_match_snapshot_ee_motion() -> None:
    targets = {
        "head": [0.128617, -0.235195, 0.767608],
        "third": [0.121657, -0.205207, 0.774948],
    }
    pre = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera=targets,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera=targets,
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    calls, results = _effect_inputs([0.0, 0.0, 0.05])

    validation = validate_grasp_motion_effect(
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert validation["verified"] is None
    assert validation["failure_kind"] == (
        "lift_result_and_snapshot_motion_disagree"
    )


def test_deferred_static_reobserve_resolves_ambiguous_lift_positive() -> None:
    state = _pending_after_ambiguous_lift()
    predicted = (
        np.asarray(_PREGRASP_WORLD_M) + np.asarray(_EE_DELTA_M)
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            observation_generation=13,
            target_by_camera={
                "head": (predicted + [0.004, -0.003, 0.001]).tolist(),
                "third": (predicted + [-0.003, 0.002, -0.001]).tolist(),
            },
        ),
        manipulation_state=state,
        robot_state=_robot_state(after=True),
    )
    calls, results = _static_reobserve_inputs()

    validation = validate_pending_grasp_observation_effect(
        calls=calls,
        results=results,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["verified"] is True
    assert validation["supporting_camera_count"] == 2


def test_deferred_static_reobserve_rejects_cross_view_position_split() -> None:
    state = _pending_after_ambiguous_lift()
    predicted = (
        np.asarray(_PREGRASP_WORLD_M) + np.asarray(_EE_DELTA_M)
    )
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            observation_generation=13,
            target_by_camera={
                "head": (predicted + [0.0285, 0.0, 0.0]).tolist(),
                "third": (predicted + [-0.0285, 0.0, 0.0]).tolist(),
            },
        ),
        manipulation_state=state,
        robot_state=_robot_state(after=True),
    )
    calls, results = _static_reobserve_inputs()

    validation = validate_pending_grasp_observation_effect(
        calls=calls,
        results=results,
        post=post,
    )

    assert validation["applicable"] is True
    assert validation["supporting_camera_count"] == 2
    assert validation["cross_view_position_consistent"] is False
    assert validation["verified"] is None
    assert validation["failure_kind"] == (
        "cross_view_identity_position_inconsistent"
    )


def test_deferred_static_reobserve_resolves_stationary_object_negative() -> None:
    state = _pending_after_ambiguous_lift()
    anchor = np.asarray(_PREGRASP_WORLD_M)
    post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            observation_generation=13,
            target_by_camera={
                "head": (anchor + [0.003, -0.002, 0.001]).tolist(),
                "third": (anchor + [-0.002, 0.003, -0.001]).tolist(),
            },
        ),
        manipulation_state=state,
        robot_state=_robot_state(after=True),
    )
    calls, results = _static_reobserve_inputs()

    validation = validate_pending_grasp_observation_effect(
        calls=calls,
        results=results,
        post=post,
    )

    assert validation["verified"] is False
    assert validation["failure_kind"] == "object_stationary_during_lift"
    assert validation["negative_camera_count"] == 2


def test_deferred_static_reobserve_requires_fresh_success_and_fixed_robot() -> None:
    state = _pending_after_ambiguous_lift()
    predicted = (
        np.asarray(_PREGRASP_WORLD_M) + np.asarray(_EE_DELTA_M)
    ).tolist()
    stale_post = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            observation_generation=12,
            target_by_camera={"head": predicted, "third": predicted},
        ),
        manipulation_state=state,
        robot_state=_robot_state(after=True),
    )
    calls, results = _static_reobserve_inputs()
    assert validate_pending_grasp_observation_effect(
        calls=calls,
        results=results,
        post=stale_post,
    )["applicable"] is False
    assert validate_pending_grasp_observation_effect(
        calls=calls,
        results=[
            RecoveryToolResult(tool_name="reobserve_scene", success=False)
        ],
        post=stale_post,
    )["applicable"] is False

    fresh_process = _preprocess(
        env_step=12,
        observation_generation=13,
        target_by_camera={"head": predicted, "third": predicted},
    )
    opened_post = capture_grasp_motion_snapshot(
        observation_preprocess=fresh_process,
        manipulation_state=state,
        robot_state={
            _ARM: {
                "gripper": 1.0,
                "xyz": _robot_state(after=True)[_ARM]["xyz"],
            }
        },
    )
    opened = validate_pending_grasp_observation_effect(
        calls=calls,
        results=results,
        post=opened_post,
    )
    assert opened["verified"] is None
    assert opened["failure_kind"] == "postlift_robot_state_changed"

    drifted_robot = _robot_state(after=True)
    drifted_robot[_ARM]["xyz"][0] += 0.02
    drifted_post = capture_grasp_motion_snapshot(
        observation_preprocess=fresh_process,
        manipulation_state=state,
        robot_state=drifted_robot,
    )
    drifted = validate_pending_grasp_observation_effect(
        calls=calls,
        results=results,
        post=drifted_post,
    )
    assert drifted["failure_kind"] == "postlift_robot_state_changed"


def test_deferred_positive_promotes_exact_attachment_and_transport() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.step_count = 12
    snapshot.right_endpose = np.asarray(
        [*_robot_state(after=True)[_ARM]["xyz"], 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    snapshot.raw.setdefault("endpose", {})["right_gripper"] = 0.0
    agent.latest_snapshot = snapshot
    state = _pending_after_ambiguous_lift()
    agent.memory_store.state.working.manipulation_state = deepcopy(state)
    predicted = (
        np.asarray(_PREGRASP_WORLD_M) + np.asarray(_EE_DELTA_M)
    ).tolist()
    validation = {
        "applicable": True,
        "verified": True,
        "arm": _ARM,
        "held_instance_id": _INSTANCE_ID,
        "grasp_candidate_id": state[_ARM]["grasp_candidate_id"],
        "grasp_attempt_nonce": state[_ARM]["grasp_attempt_nonce"],
        "supporting_camera_count": 2,
        "verified_object_world_m": predicted,
    }
    calls, results = _static_reobserve_inputs()

    agent._update_manipulation_state_from_effect(
        calls=calls,
        results=results,
        action_effect={
            "effect_verified": "true",
            "effect_type": "grasp",
            "runtime_grasp_validation": validation,
        },
    )

    current = agent.memory_store.state.working.manipulation_state[_ARM]
    assert manipulation_state_allows_transport(current) is True
    assert current["held_object_to_tcp_attachment"]["authority"] == (
        "runtime_multiview_grasp_motion"
    )


def test_stale_verified_attachment_cannot_authorize_new_attempt() -> None:
    state = _pending_after_ambiguous_lift()[_ARM]
    state.update(
        {
            "phase": "holding",
            "holding_confirmed": True,
            "transport_authorized": True,
        }
    )
    attachment = state["held_object_to_tcp_attachment"]
    attachment["source"] = "runtime_multiview_grasp_motion_verified"
    attachment["authority"] = "runtime_multiview_grasp_motion"
    state["grasp_attempt_nonce"] = "new-attempt"

    assert manipulation_state_allows_transport(state) is False


def test_runtime_validation_overrides_only_verified_grasp_effect() -> None:
    original = {
        "effect_verified": "unverified",
        "effect_type": "unknown",
        "subtask_status": "uncertain",
        "recommended_control": "retry",
        "failure_reason": "the object was occluded",
    }
    validation = {
        "applicable": True,
        "verified": True,
        "arm": _ARM,
        "held_instance_id": _INSTANCE_ID,
        "supporting_camera_count": 2,
        "supporting_cameras": ["head", "third"],
        "verified_object_world_m": [0.114495, -0.218422, 0.796388],
    }
    original_copy = deepcopy(original)

    result = apply_grasp_motion_validation(original, validation)

    assert original == original_copy
    assert result["effect_verified"] == "true"
    assert result["effect_type"] == "grasp"
    assert result["subtask_status"] == "in_progress"
    assert result["recommended_control"] == "continue"
    assert result["runtime_grasp_validation"] == validation
    assert result["failure_reason"] == ""

    negative = apply_grasp_motion_validation(
        original,
        {"applicable": True, "verified": False},
    )
    assert negative["effect_verified"] == "false"
    assert negative["effect_type"] == "grasp"
    assert negative["authority"] == "runtime_multiview_grasp_motion"
    assert negative["runtime_grasp_validation"]["verified"] is False

    ambiguous_model_positive = apply_grasp_motion_validation(
        {
            "effect_verified": "true",
            "effect_type": "grasp",
            "subtask_status": "completed",
        },
        {"applicable": True, "verified": None},
    )
    assert ambiguous_model_positive["effect_verified"] == "unverified"
    assert ambiguous_model_positive["subtask_status"] == "uncertain"
    ambiguous_unknown_positive = apply_grasp_motion_validation(
        {
            "effect_verified": "true",
            "effect_type": "unknown",
            "subtask_status": "completed",
        },
        {"applicable": True, "verified": None},
    )
    assert ambiguous_unknown_positive["effect_verified"] == "unverified"
    assert ambiguous_unknown_positive["effect_type"] == "grasp"
    completed_positive = apply_grasp_motion_validation(
        {
            "effect_verified": "true",
            "effect_type": "grasp",
            "subtask_status": "completed",
        },
        validation,
    )
    assert completed_positive["subtask_status"] == "in_progress"


def test_runtime_validation_matches_only_exact_pending_grasp() -> None:
    state = _manipulation_state()[_ARM]
    validation = {
        "applicable": True,
        "verified": False,
        "arm": _ARM,
        "held_instance_id": _INSTANCE_ID,
        "grasp_candidate_id": state["grasp_candidate_id"],
        "grasp_attempt_nonce": state["grasp_attempt_nonce"],
    }

    assert matched_grasp_motion_validation(
        {"runtime_grasp_validation": validation},
        arm=_ARM,
        arm_state=state,
    ) == validation
    assert matched_grasp_motion_validation(
        {
            "runtime_grasp_validation": {
                **validation,
                "held_instance_id": "track_other",
            }
        },
        arm=_ARM,
        arm_state=state,
    ) is None
    assert matched_grasp_motion_validation(
        {
            "runtime_grasp_validation": {
                key: value
                for key, value in validation.items()
                if key != "grasp_candidate_id"
            }
        },
        arm=_ARM,
        arm_state=state,
    ) is None
    assert matched_grasp_motion_validation(
        {
            "runtime_grasp_validation": {
                **validation,
                "grasp_candidate_id": "candidate_other",
            }
        },
        arm=_ARM,
        arm_state=state,
    ) is None


def test_img_agent_commits_runtime_grasp_validation_before_llm_false_negative() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.current_instruction = "move the block"
    pre_snapshot = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=10,
            target_by_camera={
                "head": [0.128617, -0.235195, 0.767608],
                "third": [0.121657, -0.205207, 0.774948],
            },
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(),
    )
    post_snapshot = capture_grasp_motion_snapshot(
        observation_preprocess=_preprocess(
            env_step=12,
            target_by_camera={
                "head": [0.129994, -0.236269, 0.789851],
                "third": [0.123956, -0.208401, 0.793879],
            },
        ),
        manipulation_state=_manipulation_state(),
        robot_state=_robot_state(after=True),
    )
    pre = {
        "robot_state": {},
        "scene_memory": {},
        "observation_summary": "before lift",
        "runtime_evaluation": {},
        "_runtime_grasp_motion_snapshot": pre_snapshot,
    }
    post = {
        "robot_state": {},
        "scene_memory": {},
        "observation_summary": "after lift",
        "runtime_evaluation": {},
        "_runtime_grasp_motion_snapshot": post_snapshot,
    }
    calls, results = _effect_inputs()
    verifier = mock.Mock()
    verifier.verify.return_value = {
        "effect_verified": "unverified",
        "effect_type": "unknown",
        "subtask_status": "uncertain",
        "recommended_control": "retry",
        "failure_reason": "the grasped object is occluded",
    }
    agent._action_effect_verifier = verifier

    effect = agent._verify_action_effect(
        "task-level-recovery-control",
        current_subtask="move the block to the center",
        post_recovery_intent="retry",
        expected_outcome="verify the diagnostic lift",
        recovery_reason="confirm attachment",
        calls=calls,
        results=results,
        pre=pre,
        post=post,
    )

    assert effect["effect_verified"] == "true"
    assert effect["effect_type"] == "grasp"
    assert effect["runtime_grasp_validation"]["verified"] is True
    verifier_payload = verifier.verify.call_args.args[0]
    assert verifier_payload["runtime_grasp_validation"]["verified"] is True
    assert "_runtime_grasp_motion_snapshot" not in verifier_payload["pre"]
    assert "_runtime_grasp_motion_snapshot" not in verifier_payload["post"]


def test_img_agent_rejects_model_completion_when_deferred_observation_is_invalid() -> None:
    state = _pending_after_ambiguous_lift()
    predicted = (
        np.asarray(_PREGRASP_WORLD_M) + np.asarray(_EE_DELTA_M)
    ).tolist()
    calls, successful_results = _static_reobserve_inputs()
    cases = (
        (12, successful_results),
        (
            13,
            [
                RecoveryToolResult(
                    tool_name="reobserve_scene",
                    success=False,
                )
            ],
        ),
    )

    for observation_generation, results in cases:
        agent = ImgAgent(make_card(DummyConfig()))
        agent.current_instruction = "move the block"
        post_snapshot = capture_grasp_motion_snapshot(
            observation_preprocess=_preprocess(
                env_step=12,
                observation_generation=observation_generation,
                target_by_camera={
                    "head": predicted,
                    "third": predicted,
                },
            ),
            manipulation_state=state,
            robot_state=_robot_state(after=True),
        )
        verifier = mock.Mock()
        verifier.verify.return_value = {
            "effect_verified": "true",
            "effect_type": "grasp",
            "subtask_status": "completed",
            "recommended_control": "continue",
        }
        agent._action_effect_verifier = verifier

        effect = agent._verify_action_effect(
            "task-level-recovery-control",
            current_subtask="move the block to the center",
            post_recovery_intent="retry",
            expected_outcome="verify the pending grasp",
            recovery_reason="confirm attachment",
            calls=calls,
            results=results,
            pre={"_runtime_grasp_motion_snapshot": {}},
            post={"_runtime_grasp_motion_snapshot": post_snapshot},
        )

        assert effect["effect_verified"] == "unverified"
        assert effect["effect_type"] == "grasp"
        assert effect["subtask_status"] == "uncertain"
        assert effect["recommended_control"] == "retry"
        assert effect["authority"] == "runtime_multiview_grasp_motion"


def test_verified_motion_updates_holding_state_from_runtime_object_position() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.step_count = 12
    snapshot.right_endpose = np.asarray(
        [
            0.112396,
            -0.214322,
            0.93059,
            0.296898,
            0.648968,
            0.26215,
            -0.649592,
        ],
        dtype=np.float32,
    )
    snapshot.raw.setdefault("endpose", {})["right_gripper"] = 0.0
    agent.latest_snapshot = snapshot
    state = _manipulation_state()
    state[_ARM]["held_object_to_tcp_attachment"] = {
        "object_proxy_frame": "tcp_aligned_at_capture",
        "object_centroid_to_tcp_translation_tcp_m": [0.0, 0.0, 0.02],
        "capture_tcp_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "grasp_candidate_id": state[_ARM]["grasp_candidate_id"],
        "grasp_attempt_nonce": state[_ARM]["grasp_attempt_nonce"],
        "capture_step": state[_ARM]["grasp_attempt_step"],
        "source": "pending_lift_verification",
    }
    agent.memory_store.state.working.manipulation_state = deepcopy(state)
    agent.memory_store.record_scene_memory(
        {
            "env_step": 12,
            "instances": [
                {
                    "instance_id": _INSTANCE_ID,
                    "track_id": _INSTANCE_ID,
                    "status": "tracked",
                    "stability": "geometry_inconsistent_current_frame",
                    "position_state": "motion_uncertain",
                    "world_m": list(_PREGRASP_WORLD_M),
                    "latest_world_m": [9.0, 9.0, 9.0],
                }
            ],
            "manipulation_state": deepcopy(state),
        }
    )
    calls, results = _effect_inputs()
    verified_world = (
        np.asarray(_PREGRASP_WORLD_M) + np.asarray(_EE_DELTA_M)
    ).tolist()
    effect = {
        "effect_verified": "true",
        "effect_type": "grasp",
        "runtime_grasp_validation": {
            "applicable": True,
                "verified": True,
                "arm": _ARM,
                "held_instance_id": _INSTANCE_ID,
                "grasp_candidate_id": state[_ARM][
                    "grasp_candidate_id"
                ],
                "grasp_attempt_nonce": state[_ARM][
                    "grasp_attempt_nonce"
                ],
                "verified_object_world_m": verified_world,
            },
    }

    agent._update_manipulation_state_from_effect(
        calls=calls,
        results=results,
        action_effect=effect,
    )

    current = agent.memory_store.state.working.manipulation_state[_ARM]
    assert current["phase"] == "holding"
    assert current["holding_confirmed"] is True
    assert current["transport_authorized"] is True
    assert "diagnostic_lift_evidence" not in current
    attachment = current["held_object_to_tcp_attachment"]
    assert attachment["source"] == (
        "runtime_multiview_grasp_motion_verified"
    )
    assert attachment["authority"] == (
        "runtime_multiview_grasp_motion"
    )
    assert attachment["object_centroid_to_tcp_translation_tcp_m"] == [
        0.0,
        0.0,
        0.02,
    ]
    assert attachment["capture_tcp_pose"] == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def test_stale_candidate_validation_cannot_authorize_new_grasp_state() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.step_count = 12
    snapshot.raw.setdefault("endpose", {})["right_gripper"] = 0.0
    agent.latest_snapshot = snapshot
    state = _manipulation_state()
    state[_ARM]["held_object_to_tcp_attachment"] = {
        "object_proxy_frame": "tcp_aligned_at_capture",
        "object_centroid_to_tcp_translation_tcp_m": [0.0, 0.0, 0.02],
        "capture_tcp_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "grasp_candidate_id": state[_ARM]["grasp_candidate_id"],
        "grasp_attempt_nonce": state[_ARM]["grasp_attempt_nonce"],
        "capture_step": state[_ARM]["grasp_attempt_step"],
        "source": "pending_lift_verification",
    }
    agent.memory_store.state.working.manipulation_state = deepcopy(state)
    calls, results = _effect_inputs()

    agent._update_manipulation_state_from_effect(
        calls=calls,
        results=results,
        action_effect={
            "effect_verified": "true",
            "effect_type": "grasp",
            "runtime_grasp_validation": {
                "applicable": True,
                "verified": True,
                "arm": _ARM,
                "held_instance_id": _INSTANCE_ID,
                "grasp_candidate_id": "stale_candidate",
                "grasp_attempt_nonce": state[_ARM][
                    "grasp_attempt_nonce"
                ],
                "verified_object_world_m": [0.0, 0.0, 0.8],
            },
        },
    )

    current = agent.memory_store.state.working.manipulation_state[_ARM]
    assert current["phase"] == "grasp_candidate"
    assert current["holding_confirmed"] is False
    assert current["transport_authorized"] is False

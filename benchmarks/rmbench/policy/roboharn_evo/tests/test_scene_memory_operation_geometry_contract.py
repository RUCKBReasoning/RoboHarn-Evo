from __future__ import annotations

import unittest
from typing import Any

from policy.roboharn_evo.agent.perception.position_identity_contract import (
    VERIFIED_OPERATION_TARGET_ANCHOR_KEY,
)
from policy.roboharn_evo.agent.operation_candidate_lifecycle import (
    OperationCandidateLifecycle,
)
from policy.roboharn_evo.agent.operation_geometry_refresh_contract import (
    OperationGeometryRefreshContract,
)
from policy.roboharn_evo.agent.recovery.tool_specs import (
    RecoveryToolCall,
    RecoveryToolResult,
)
from policy.roboharn_evo.agent.perception.scene_memory import SceneMemoryTracker


_OLD_WORLD = [0.0, -0.15, 0.75]
_TARGET_WORLD = [0.06, -0.10, 0.75]


def _operation_candidate(
    candidate_id: str,
    world: list[float],
    *,
    lateral_m: float = 0.0,
    executable_shift_m: float = 0.0,
    arm: str = "right",
    action_mode: str = "grasp",
) -> dict[str, Any]:
    contact = [
        world[0] + lateral_m,
        world[1],
        world[2] + 0.01,
    ]
    tcp = [contact[0], contact[1], contact[2] + 0.02]
    target = [
        contact[0] + executable_shift_m,
        contact[1],
        contact[2] + 0.10,
    ]
    approach = [target[0], target[1], target[2] + 0.08]
    identity_quat = [1.0, 0.0, 0.0, 0.0]
    return {
        "candidate_id": candidate_id,
        "action_mode": action_mode,
        "arm": arm,
        "object_contact_pose": [*contact, *identity_quat],
        "tcp_pose": [*tcp, *identity_quat],
        "ee_target_pose": [*target, *identity_quat],
        "approach_pose": [*approach, *identity_quat],
        "approach_direction": [0.0, 0.0, -1.0],
        "geometry_source": "test_observed_geometry",
    }


def _detection(
    world: list[float],
    *,
    operation_candidates: list[dict[str, Any]] | None = None,
    include_action_points: bool = True,
    action_point_lateral_m: float = 0.0,
    mask_name: str,
) -> dict[str, Any]:
    grounding: dict[str, Any] = {
        "success": True,
        "centroid_world": list(world),
    }
    if include_action_points:
        top = [
            world[0] + action_point_lateral_m,
            world[1],
            world[2] + 0.03,
        ]
        approach = [
            world[0] + action_point_lateral_m,
            world[1],
            world[2] + 0.20,
        ]
        contact = [
            world[0] + action_point_lateral_m,
            world[1],
            world[2] + 0.10,
        ]
        grounding.update(
            {
                "top_surface_world": top,
                "approach_point_world": approach,
                "approach_pose_world": [
                    *approach,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "contact_point_world": contact,
                "contact_pose_world": [
                    *contact,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )
    if operation_candidates is not None:
        grounding["operation_pose_candidates"] = [
            dict(item) for item in operation_candidates
        ]
    return {
        "rank": 0,
        "score": 0.91,
        "bbox_xyxy": [180, 130, 215, 165],
        "centroid_px": [197.5, 147.5],
        "mask_path": f"/tmp/{mask_name}.png",
        "grounding_3d": grounding,
    }


def _observation(
    tracker: SceneMemoryTracker,
    *,
    detection: dict[str, Any],
    env_step: int,
    capture_id: int,
    track_id: str | None = None,
    identity_relocation_leases: list[dict[str, Any]] | None = None,
    position_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "success": True,
        "object_id": "selected_component",
        "camera": "head",
        "query_role": "target",
        "detections": [detection],
    }
    if track_id is not None:
        query.update(
            {
                "instance_ref": track_id,
                "identity_binding_required": True,
            }
        )
    return tracker.update(
        segmentation=[query],
        env_step=env_step,
        observation_capture_id=capture_id,
        global_task="move one selected component",
        current_subtask="reacquire executable geometry",
        identity_relocation_leases=identity_relocation_leases,
        position_events=position_events,
    )


def _verified_release_event(track_id: str, env_step: int) -> dict[str, Any]:
    return {
        "instance_ref": track_id,
        "position_state": "current_verified",
        "world_m": list(_OLD_WORLD),
        "confidence": 1.0,
        "tolerance_m": 0.02,
        "operation_target_world_m": list(_TARGET_WORLD),
        "operation_target_tolerance_m": 0.025,
        "operation_target_id": "operation-target:test",
        "arm": "right",
        "source": "verified_post_release_position",
        "env_step": env_step,
    }


def _seed_released_track(
    tracker: SceneMemoryTracker,
    *,
    with_initial_geometry: bool = True,
) -> tuple[str, dict[str, Any]]:
    initial_candidates = (
        [_operation_candidate("seed-candidate", _OLD_WORLD)]
        if with_initial_geometry
        else None
    )
    initial = _observation(
        tracker,
        detection=_detection(
            _OLD_WORLD,
            operation_candidates=initial_candidates,
            include_action_points=with_initial_geometry,
            mask_name="operation_contract_seed",
        ),
        env_step=0,
        capture_id=0,
    )
    track_id = initial["instances"][0]["track_id"]
    released = tracker.apply_runtime_events(
        initial,
        events=[_verified_release_event(track_id, 1)],
        env_step=1,
    )
    return track_id, released


def _internal_track(
    tracker: SceneMemoryTracker,
    track_id: str,
) -> dict[str, Any]:
    return tracker._temporal_tracker._tracks[track_id]


class SceneMemoryOperationGeometryContractTest(unittest.TestCase):
    def test_same_update_motion_uncertain_revokes_external_refresh_anchor(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate("right-grasp", _OLD_WORLD)
                ],
                mask_name="same_frame_motion_seed",
            ),
            env_step=0,
            capture_id=0,
        )
        track_id = initial["instances"][0]["track_id"]
        refresh = OperationGeometryRefreshContract()
        call = RecoveryToolCall(
            tool_name="retreat_arm",
            args={"arm": "right"},
        )
        result = RecoveryToolResult(
            tool_name="retreat_arm",
            success=True,
            details={
                "arm": "right",
                "observed_displacement_m": 0.03,
            },
        )
        refresh.authorize_from_results(
            calls=[call],
            results=[result],
            blocked_scopes=[
                {
                    "instance_id": track_id,
                    "arm": "right",
                    "action_mode": "grasp",
                    "blocked_candidate_count": 1,
                    "available_candidate_count": 0,
                }
            ],
            scene_memory=initial,
            env_step=1,
            capture_id=0,
        )
        leases = refresh.pending_relocation_leases(
            scene_memory=initial,
            env_step=1,
            capture_id=1,
        )
        updated = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate(
                        "right-grasp",
                        _OLD_WORLD,
                        executable_shift_m=0.03,
                    )
                ],
                mask_name="same_frame_motion_candidate",
            ),
            env_step=1,
            capture_id=1,
            track_id=track_id,
            identity_relocation_leases=leases,
            position_events=[
                {
                    "instance_ref": track_id,
                    "position_state": "motion_uncertain",
                    "source": "unverified_contact",
                    "reason": "same-batch contact may move object",
                }
            ],
        )
        instance = updated["instances"][0]
        self.assertIsNone(
            instance.get("verified_operation_target_anchor")
        )
        self.assertFalse(
            any(
                item.get("source")
                == "blocked_operation_geometry_refresh_after_clearance"
                for item in instance.get(
                    "identity_relocation_history",
                    [],
                )
            )
        )

    def test_refresh_rebuild_is_limited_to_authorized_arm_and_mode(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial_candidates = [
            _operation_candidate(
                "right-grasp",
                _OLD_WORLD,
                arm="right",
                action_mode="grasp",
            ),
            _operation_candidate(
                "left-grasp",
                _OLD_WORLD,
                arm="left",
                action_mode="grasp",
            ),
            _operation_candidate(
                "right-contact",
                _OLD_WORLD,
                arm="right",
                action_mode="contact",
            ),
        ]
        initial = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=initial_candidates,
                mask_name="scoped_refresh_seed",
            ),
            env_step=0,
            capture_id=0,
        )
        track_id = initial["instances"][0]["track_id"]
        refresh = OperationGeometryRefreshContract()
        created = refresh.authorize_from_results(
            calls=[
                RecoveryToolCall(
                    tool_name="retreat_arm",
                    args={"arm": "right"},
                )
            ],
            results=[
                RecoveryToolResult(
                    tool_name="retreat_arm",
                    success=True,
                    details={
                        "arm": "right",
                        "observed_displacement_m": 0.03,
                    },
                )
            ],
            blocked_scopes=[
                {
                    "instance_id": track_id,
                    "arm": "right",
                    "action_mode": "grasp",
                    "blocked_candidate_count": 1,
                    "available_candidate_count": 0,
                }
            ],
            scene_memory=initial,
            env_step=1,
            capture_id=0,
        )
        self.assertEqual(created[0]["action_modes"], ["grasp"])
        observed_candidates = [
            _operation_candidate(
                "right-grasp-new",
                _OLD_WORLD,
                executable_shift_m=0.03,
                arm="right",
                action_mode="grasp",
            ),
            _operation_candidate(
                "left-grasp-new",
                _OLD_WORLD,
                executable_shift_m=0.04,
                arm="left",
                action_mode="grasp",
            ),
            _operation_candidate(
                "right-contact-new",
                _OLD_WORLD,
                executable_shift_m=0.05,
                arm="right",
                action_mode="contact",
            ),
        ]
        first = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=observed_candidates,
                mask_name="scoped_refresh_first",
            ),
            env_step=1,
            capture_id=1,
            track_id=track_id,
            identity_relocation_leases=(
                refresh.pending_relocation_leases(
                    scene_memory=initial,
                    env_step=1,
                    capture_id=1,
                )
            ),
        )
        pending = first["instances"][0]
        self.assertEqual(
            pending["action_geometry_pending_scope"],
            {"arm": "right", "action_modes": ["grasp"]},
        )
        self.assertEqual(
            {
                item["candidate_id"]
                for item in pending["operation_pose_candidates"]
            },
            {"left-grasp", "right-contact"},
        )

        second = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=observed_candidates,
                mask_name="scoped_refresh_second",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        verified = second["instances"][0]
        by_id = {
            item["candidate_id"]: item
            for item in verified["operation_pose_candidates"]
        }
        self.assertEqual(
            set(by_id),
            {"right-grasp-new", "left-grasp", "right-contact"},
        )
        self.assertAlmostEqual(
            by_id["right-grasp-new"]["ee_target_pose"][0],
            initial_candidates[0]["ee_target_pose"][0] + 0.03,
        )
        self.assertEqual(
            by_id["left-grasp"]["ee_target_pose"],
            initial_candidates[1]["ee_target_pose"],
        )
        self.assertEqual(
            by_id["right-contact"]["ee_target_pose"],
            initial_candidates[2]["ee_target_pose"],
        )
        provenance = verified[
            "operation_pose_candidate_provenance"
        ]
        self.assertEqual(
            provenance["left-grasp"][
                "geometry_observation_capture_id"
            ],
            0,
        )
        self.assertEqual(
            provenance["right-grasp-new"][
                "geometry_observation_capture_id"
            ],
            2,
        )

    def test_measured_clearance_authorizes_two_frame_refresh_and_unblocks(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        candidate_id = "stable-runtime-slot"
        initial = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate(candidate_id, _OLD_WORLD)
                ],
                mask_name="refresh_seed",
            ),
            env_step=0,
            capture_id=0,
        )
        initial_instance = initial["instances"][0]
        track_id = initial_instance["track_id"]
        lifecycle = OperationCandidateLifecycle()
        attempt = lifecycle.bind_attempt(
            instance_id=track_id,
            candidate=initial_instance[
                "operation_pose_candidates"
            ][0],
            instance=initial_instance,
            point_key="grasp_world_m",
            dispatch_env_step=0,
        )
        lifecycle.record_failure(
            attempt,
            target_error_m=0.04,
            env_step=0,
        )
        lifecycle.record_failure(
            attempt,
            target_error_m=0.039,
            env_step=0,
        )

        refresh = OperationGeometryRefreshContract()
        created = refresh.authorize_from_results(
            calls=[
                RecoveryToolCall(
                    tool_name="retreat_arm",
                    args={"arm": "right"},
                )
            ],
            results=[
                RecoveryToolResult(
                    tool_name="retreat_arm",
                    success=True,
                    details={
                        "arm": "right",
                        "observed_displacement_m": 0.03,
                    },
                )
            ],
            blocked_scopes=lifecycle.planner_payload(
                [initial_instance]
            ),
            scene_memory=initial,
            env_step=1,
            capture_id=0,
        )
        self.assertEqual(len(created), 1)
        leases = refresh.pending_relocation_leases(
            scene_memory=initial,
            env_step=1,
            capture_id=1,
        )
        changed_geometry = [
            _operation_candidate(
                candidate_id,
                _OLD_WORLD,
                executable_shift_m=0.03,
            )
        ]
        first = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=changed_geometry,
                action_point_lateral_m=0.05,
                mask_name="refresh_first",
            ),
            env_step=1,
            capture_id=1,
            track_id=track_id,
            identity_relocation_leases=leases,
        )
        self.assertEqual(
            first["instances"][0]["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            refresh.pending_relocation_leases(
                scene_memory=first,
                env_step=2,
                capture_id=2,
            ),
            [],
        )

        second = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=changed_geometry,
                action_point_lateral_m=0.05,
                mask_name="refresh_second",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        verified = second["instances"][0]
        self.assertEqual(
            verified["action_geometry_state"],
            "verified",
        )
        self.assertAlmostEqual(
            verified["approach_world_m"][0],
            _OLD_WORLD[0] + 0.05,
        )
        self.assertAlmostEqual(
            verified["contact_world_m"][0],
            _OLD_WORLD[0] + 0.05,
        )
        self.assertAlmostEqual(
            verified["top_surface_world_m"][0],
            _OLD_WORLD[0] + 0.05,
        )
        blocked, revalidated = lifecycle.blocked_candidate_ids(
            instance_id=track_id,
            arm="right",
            action_mode="grasp",
            candidates=verified["operation_pose_candidates"],
            instance=verified,
        )
        self.assertEqual(blocked, [])
        self.assertEqual(len(revalidated), 1)

    def test_verified_release_rebuild_unblocks_only_after_two_fresh_frames(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        candidate_id = "reusable-runtime-slot"
        initial = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate(candidate_id, _OLD_WORLD)
                ],
                mask_name="lifecycle_release_seed",
            ),
            env_step=0,
            capture_id=0,
        )
        track_id = initial["instances"][0]["track_id"]
        lifecycle = OperationCandidateLifecycle()
        initial_instance = initial["instances"][0]
        initial_candidate = initial_instance[
            "operation_pose_candidates"
        ][0]
        attempt = lifecycle.bind_attempt(
            instance_id=track_id,
            candidate=initial_candidate,
            instance=initial_instance,
            point_key="grasp_world_m",
            dispatch_env_step=0,
        )
        self.assertTrue(attempt)
        lifecycle.record_failure(
            attempt,
            target_error_m=0.04,
            env_step=0,
        )
        lifecycle.record_failure(
            attempt,
            target_error_m=0.039,
            env_step=0,
        )

        tracker.apply_runtime_events(
            initial,
            events=[_verified_release_event(track_id, 1)],
            env_step=1,
        )
        relocated = [
            _operation_candidate(candidate_id, _TARGET_WORLD)
        ]
        first = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=relocated,
                mask_name="lifecycle_release_first",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        first_instance = first["instances"][0]
        self.assertEqual(
            first_instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            first_instance["operation_pose_candidates"],
            [],
        )

        second = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=relocated,
                mask_name="lifecycle_release_second",
            ),
            env_step=3,
            capture_id=3,
            track_id=track_id,
        )
        verified_instance = second["instances"][0]
        blocked, revalidated = lifecycle.blocked_candidate_ids(
            instance_id=track_id,
            arm="right",
            action_mode="grasp",
            candidates=verified_instance[
                "operation_pose_candidates"
            ],
            instance=verified_instance,
        )
        self.assertEqual(blocked, [])
        self.assertEqual(len(revalidated), 1)
        self.assertEqual(
            revalidated[0]["candidate_id"],
            candidate_id,
        )

    def test_operation_target_relocation_without_old_geometry_still_requires_two_frames(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        track_id, _ = _seed_released_track(
            tracker,
            with_initial_geometry=False,
        )
        relocated = [_operation_candidate("relocated-a", _TARGET_WORLD)]

        first = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=relocated,
                include_action_points=False,
                mask_name="operation_target_first",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        first_instance = first["instances"][0]
        self.assertEqual(
            first_instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            first_instance["action_geometry_confirmation_count"],
            1,
        )
        self.assertEqual(first_instance["operation_pose_candidate_count"], 0)
        initial_warning = _internal_track(tracker, track_id)[
            "relocation_action_geometry_initial_warning"
        ]
        self.assertNotIn(
            "temporal_action_geometry_inconsistent",
            initial_warning,
        )
        self.assertEqual(
            first["temporal_memory"]["identity_binding_outcomes"][0][
                "matched_by"
            ],
            "verified_operation_target",
        )

        second = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=relocated,
                include_action_points=False,
                mask_name="operation_target_second",
            ),
            env_step=3,
            capture_id=3,
            track_id=track_id,
        )
        second_instance = second["instances"][0]
        self.assertEqual(second_instance["action_geometry_state"], "verified")
        self.assertEqual(second_instance["operation_pose_candidate_count"], 1)
        self.assertEqual(
            second_instance["action_geometry_repair_history"][-1][
                "confirmation_count"
            ],
            2,
        )

    def test_motion_uncertain_clears_anchor_repair_state_and_executable_geometry(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        track_id, _ = _seed_released_track(tracker)
        first = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=[
                    _operation_candidate("relocated-a", _TARGET_WORLD)
                ],
                mask_name="pending_before_motion_uncertain",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        self.assertEqual(
            first["instances"][0]["action_geometry_state"],
            "relocation_pending",
        )
        internal = _internal_track(tracker, track_id)
        self.assertIn(VERIFIED_OPERATION_TARGET_ANCHOR_KEY, internal)
        self.assertIn("relocation_identity_repair_lease", internal)
        self.assertEqual(
            len(internal["relocation_action_geometry_repair_samples"]),
            1,
        )

        # Seed stale executable values to make this a cleanup contract test,
        # rather than merely asserting fields that were already empty while
        # the repair was pending.
        stale = _operation_candidate("stale-executable", _TARGET_WORLD)
        internal["operation_pose_candidates"] = [stale]
        internal["operation_pose_candidate_provenance"] = {
            "stale-executable": {
                "geometry_observation_env_step": 2,
                "geometry_observation_capture_id": 2,
            }
        }

        invalidated = tracker.apply_runtime_events(
            first,
            events=[
                {
                    "instance_ref": track_id,
                    "position_state": "motion_uncertain",
                    "source": "unverified_contact",
                    "reason": "the object may have moved",
                }
            ],
            env_step=3,
        )
        internal = _internal_track(tracker, track_id)
        self.assertNotIn(VERIFIED_OPERATION_TARGET_ANCHOR_KEY, internal)
        self.assertNotIn("relocation_identity_repair_lease", internal)
        self.assertNotIn(
            "relocation_action_geometry_repair_samples",
            internal,
        )
        self.assertEqual(internal["operation_pose_candidates"], [])
        self.assertEqual(internal["operation_pose_candidate_provenance"], {})
        self.assertEqual(internal["action_geometry_state"], "unavailable")

        public = invalidated["instances"][0]
        self.assertIsNone(public["verified_operation_target_anchor"])
        self.assertEqual(public["operation_pose_candidate_count"], 0)
        self.assertEqual(public["operation_pose_candidate_provenance"], {})
        self.assertEqual(public["action_geometry_state"], "unavailable")

    def test_equal_relative_offsets_do_not_confirm_changed_executable_pose(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        track_id, _ = _seed_released_track(tracker)
        first_geometry = [
            _operation_candidate("relocated-a", _TARGET_WORLD)
        ]
        changed_geometry = [
            _operation_candidate(
                "relocated-a",
                _TARGET_WORLD,
                executable_shift_m=0.03,
            )
        ]

        first = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=first_geometry,
                mask_name="candidate_pose_first",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        self.assertEqual(
            first["instances"][0]["action_geometry_confirmation_count"],
            1,
        )

        changed = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=changed_geometry,
                mask_name="candidate_pose_changed",
            ),
            env_step=3,
            capture_id=3,
            track_id=track_id,
        )
        changed_instance = changed["instances"][0]
        self.assertEqual(
            changed_instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            changed_instance["action_geometry_confirmation_count"],
            1,
        )
        self.assertEqual(changed_instance["operation_pose_candidate_count"], 0)

        confirmed = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=changed_geometry,
                mask_name="candidate_pose_changed_confirmed",
            ),
            env_step=4,
            capture_id=4,
            track_id=track_id,
        )
        confirmed_instance = confirmed["instances"][0]
        self.assertEqual(
            confirmed_instance["action_geometry_state"],
            "verified",
        )
        self.assertAlmostEqual(
            confirmed_instance["operation_pose_candidates"][0][
                "ee_target_pose"
            ][0],
            _TARGET_WORLD[0] + 0.03,
        )

    def test_disabled_executable_pose_consistency_keeps_two_frame_confirmation(
        self,
    ) -> None:
        tracker = SceneMemoryTracker(
            enable_executable_candidate_temporal_consistency=False,
        )
        track_id, _ = _seed_released_track(tracker)
        first_geometry = [
            _operation_candidate("relocated-a", _TARGET_WORLD)
        ]
        changed_geometry = [
            _operation_candidate(
                "relocated-a",
                _TARGET_WORLD,
                executable_shift_m=0.03,
            )
        ]

        first = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=first_geometry,
                mask_name="disabled_candidate_pose_first",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        self.assertEqual(
            first["instances"][0]["action_geometry_confirmation_count"],
            1,
        )

        changed = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=changed_geometry,
                mask_name="disabled_candidate_pose_changed",
            ),
            env_step=3,
            capture_id=3,
            track_id=track_id,
        )
        changed_instance = changed["instances"][0]
        self.assertEqual(
            changed_instance["action_geometry_state"],
            "verified",
        )
        self.assertEqual(
            changed_instance["action_geometry_repair_history"][-1][
                "confirmation_count"
            ],
            2,
        )
        self.assertAlmostEqual(
            changed_instance["operation_pose_candidates"][0][
                "ee_target_pose"
            ][0],
            _TARGET_WORLD[0] + 0.03,
        )

    def test_disabled_executable_pose_consistency_retains_offset_rejection(
        self,
    ) -> None:
        tracker = SceneMemoryTracker(
            enable_executable_candidate_temporal_consistency=False,
        )
        initial = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate("right-grasp", _OLD_WORLD)
                ],
                mask_name="disabled_pose_offset_seed",
            ),
            env_step=0,
            capture_id=0,
        )
        track_id = initial["instances"][0]["track_id"]

        rejected = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate(
                        "right-grasp",
                        _OLD_WORLD,
                        executable_shift_m=0.03,
                    )
                ],
                action_point_lateral_m=0.05,
                mask_name="disabled_pose_offset_changed",
            ),
            env_step=1,
            capture_id=1,
            track_id=track_id,
        )

        self.assertEqual(
            rejected["instances"][0]["position_state"],
            "memory_valid",
        )
        warnings = [
            warning
            for dropped in rejected["temporal_memory"][
                "dropped_candidates"
            ]
            for warning in dropped["quality"]["warnings"]
        ]
        self.assertTrue(
            any(
                warning.startswith(
                    "temporal_action_geometry_inconsistent:"
                )
                and "executable_candidate_pose_changed" not in warning
                for warning in warnings
            )
        )

    def test_candidate_id_reordering_does_not_break_physical_set_confirmation(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        track_id, _ = _seed_released_track(tracker)
        first_set = [
            _operation_candidate(
                "rank-slot-000",
                _TARGET_WORLD,
                lateral_m=-0.025,
            ),
            _operation_candidate(
                "rank-slot-001",
                _TARGET_WORLD,
                lateral_m=0.025,
            ),
        ]
        reordered_and_renamed = [
            _operation_candidate(
                "new-rank-slot-017",
                _TARGET_WORLD,
                lateral_m=0.025,
            ),
            _operation_candidate(
                "new-rank-slot-004",
                _TARGET_WORLD,
                lateral_m=-0.025,
            ),
        ]
        pending = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=first_set,
                mask_name="candidate_set_first",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        self.assertEqual(
            pending["instances"][0]["action_geometry_state"],
            "relocation_pending",
        )

        confirmed = _observation(
            tracker,
            detection=_detection(
                _TARGET_WORLD,
                operation_candidates=reordered_and_renamed,
                mask_name="candidate_set_reordered",
            ),
            env_step=3,
            capture_id=3,
            track_id=track_id,
        )
        instance = confirmed["instances"][0]
        self.assertEqual(instance["action_geometry_state"], "verified")
        self.assertEqual(
            {
                item["candidate_id"]
                for item in instance["operation_pose_candidates"]
            },
            {"new-rank-slot-017", "new-rank-slot-004"},
        )

    def test_motion_uncertain_new_single_geometry_recovers_after_two_frames(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate("old-geometry", _OLD_WORLD)
                ],
                mask_name="motion_uncertain_seed",
            ),
            env_step=0,
            capture_id=0,
        )
        track_id = initial["instances"][0]["track_id"]
        invalidated = tracker.apply_runtime_events(
            initial,
            events=[
                {
                    "instance_ref": track_id,
                    "position_state": "motion_uncertain",
                    "source": "failed_contact",
                }
            ],
            env_step=1,
        )
        self.assertEqual(
            invalidated["instances"][0]["action_geometry_state"],
            "unavailable",
        )
        new_geometry = [
            _operation_candidate(
                "new-geometry",
                _OLD_WORLD,
                executable_shift_m=0.03,
            )
        ]

        first = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=new_geometry,
                mask_name="motion_uncertain_reacquire_first",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
        )
        first_instance = first["instances"][0]
        self.assertEqual(
            first_instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            first_instance["action_geometry_confirmation_count"],
            1,
        )
        self.assertEqual(first_instance["operation_pose_candidate_count"], 0)

        second = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=new_geometry,
                mask_name="motion_uncertain_reacquire_second",
            ),
            env_step=3,
            capture_id=3,
            track_id=track_id,
        )
        second_instance = second["instances"][0]
        self.assertEqual(second_instance["action_geometry_state"], "verified")
        self.assertEqual(second_instance["operation_pose_candidate_count"], 1)
        self.assertEqual(
            second_instance["operation_pose_candidates"][0]["candidate_id"],
            "new-geometry",
        )

    def test_replayed_same_motion_boundary_does_not_erase_clean_repair_samples(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate("old-geometry", _OLD_WORLD)
                ],
                mask_name="replayed_motion_seed",
            ),
            env_step=0,
            capture_id=0,
        )
        track_id = initial["instances"][0]["track_id"]
        motion_boundary = {
            "instance_ref": track_id,
            "position_state": "motion_uncertain",
            "source": "release_recovery_required",
            "reason": "release outcome requires object reacquisition",
            # This is the physical manipulation boundary, not the later
            # observation step.  The same boundary may be projected into
            # several perception refreshes.
            "env_step": 1,
        }
        repaired_geometry = [
            _operation_candidate(
                "repaired-geometry",
                _OLD_WORLD,
                executable_shift_m=0.03,
            )
        ]

        first = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=repaired_geometry,
                mask_name="replayed_motion_clean_first",
            ),
            env_step=2,
            capture_id=2,
            track_id=track_id,
            position_events=[motion_boundary],
        )
        self.assertEqual(
            first["instances"][0]["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            first["instances"][0]["action_geometry_confirmation_count"],
            1,
        )

        second = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=repaired_geometry,
                mask_name="replayed_motion_clean_second",
            ),
            env_step=3,
            capture_id=3,
            track_id=track_id,
            position_events=[motion_boundary],
        )
        instance = second["instances"][0]
        self.assertEqual(instance["action_geometry_state"], "verified")
        self.assertEqual(instance["operation_pose_candidate_count"], 1)
        self.assertEqual(
            instance["operation_pose_candidates"][0]["candidate_id"],
            "repaired-geometry",
        )

    def test_audit_reference_is_retained_but_not_exported_as_executable_geometry(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = _observation(
            tracker,
            detection=_detection(
                _OLD_WORLD,
                operation_candidates=[
                    _operation_candidate("audited-candidate", _OLD_WORLD)
                ],
                mask_name="audit_reference_seed",
            ),
            env_step=0,
            capture_id=0,
        )
        track_id = initial["instances"][0]["track_id"]
        invalidated = tracker.apply_runtime_events(
            initial,
            events=[
                {
                    "instance_ref": track_id,
                    "position_state": "motion_uncertain",
                    "source": "unverified_collision",
                }
            ],
            env_step=1,
        )

        internal = _internal_track(tracker, track_id)
        audit = internal[
            "last_verified_observation_geometry_reference"
        ]
        self.assertEqual(
            audit["operation_pose_candidates"][0]["candidate_id"],
            "audited-candidate",
        )
        public = invalidated["instances"][0]
        self.assertNotIn(
            "last_verified_observation_geometry_reference",
            public,
        )
        self.assertEqual(public["action_geometry_state"], "unavailable")
        self.assertEqual(public["operation_pose_candidates"], [])
        self.assertEqual(public["operation_pose_candidate_count"], 0)
        self.assertEqual(public["operation_pose_candidate_provenance"], {})
        self.assertIsNone(public["approach_world_m"])
        self.assertIsNone(public["grasp_world_m"])
        self.assertIsNone(public["contact_world_m"])


if __name__ == "__main__":
    unittest.main()

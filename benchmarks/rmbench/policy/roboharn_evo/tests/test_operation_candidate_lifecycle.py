from __future__ import annotations

from copy import deepcopy

from policy.roboharn_evo.agent.operation_candidate_lifecycle import (
    OperationCandidateLifecycle,
    operation_geometry_is_trusted,
)


OLD_POSE = {
    "object_contact_pose": [
        0.096039,
        -0.149953,
        0.819638,
        0.697559,
        -0.109197,
        0.598756,
        -0.378124,
    ],
    "tcp_pose": [
        0.096177,
        -0.119476,
        0.854488,
        0.697559,
        -0.109197,
        0.598756,
        -0.378124,
    ],
    "ee_target_pose": [
        0.096534,
        -0.040481,
        0.944819,
        0.697559,
        -0.109197,
        0.598756,
        -0.378124,
    ],
    "approach_pose": [
        0.096772,
        0.012182,
        1.005039,
        0.697559,
        -0.109197,
        0.598756,
        -0.378124,
    ],
    "approach_direction": [-0.002974, -0.658293, -0.752756],
}

NEW_POSE = {
    "object_contact_pose": [
        0.091210,
        -0.093995,
        0.763599,
        0.665965,
        0.237671,
        0.666118,
        -0.237256,
    ],
    "tcp_pose": [
        0.091210,
        -0.094005,
        0.780730,
        0.665965,
        0.237671,
        0.666118,
        -0.237256,
    ],
    "ee_target_pose": [
        0.091211,
        -0.094080,
        0.900730,
        0.665965,
        0.237671,
        0.666118,
        -0.237256,
    ],
    "approach_pose": [
        0.091211,
        -0.094130,
        0.980730,
        0.665965,
        0.237671,
        0.666118,
        -0.237256,
    ],
    "approach_direction": [-0.000007, 0.000626, -1.0],
}


def candidate(
    *,
    candidate_id: str = "rgbd_volume:grasp:right:000",
    geometry: dict = OLD_POSE,
    geometry_env_step: int = 13,
    geometry_capture_id: int = 20,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "arm": "right",
        "action_mode": "grasp",
        "geometry_source": "rgbd_observed_volume_principal_axes",
        "_geometry_observation_env_step": geometry_env_step,
        "_geometry_observation_capture_id": geometry_capture_id,
        **deepcopy(geometry),
    }


def visible_instance(item: dict) -> dict:
    return {
        "instance_id": "track_0001",
        "status": "visible",
        "position_state": "current_verified",
        "action_geometry_state": "verified",
        "actionable": True,
        "operation_pose_candidates": [item],
    }


def block(
    policy: OperationCandidateLifecycle,
    item: dict,
    *,
    errors: tuple[float, float] = (0.071531, 0.070254),
) -> dict:
    attempt = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
        dispatch_env_step=24,
        observation_generation=10,
        observation_capture_id=20,
    )
    payload = None
    for error in errors:
        payload = policy.record_failure(
            attempt,
            target_error_m=error,
            env_step=24,
        )
    assert payload is not None
    return attempt


def blocked_ids(
    policy: OperationCandidateLifecycle,
    item: dict,
    *,
    instance: dict | None = None,
) -> tuple[list[str], list[dict]]:
    current = instance or visible_instance(item)
    return policy.blocked_candidate_ids(
        instance_id="track_0001",
        arm="right",
        action_mode="grasp",
        candidates=[item],
        instance=current,
    )


def test_same_candidate_id_and_geometry_remains_blocked() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate()
    block(policy, item)

    blocked, revalidated = blocked_ids(policy, item)

    assert blocked == [item["candidate_id"]]
    assert revalidated == []


def test_blocked_geometry_stays_blocked_after_late_progress_result() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate()
    attempt = block(policy, item, errors=(0.071, 0.069))

    policy.record_failure(
        attempt,
        target_error_m=0.050,
        env_step=25,
    )

    blocked, revalidated = blocked_ids(policy, item)
    assert blocked == [item["candidate_id"]]
    assert revalidated == []


def test_approach_success_does_not_clear_terminal_pose_failure() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate()
    block(policy, item)
    approach_attempt = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
        point_key="approach_world_m",
    )

    policy.record_success(approach_attempt)

    assert blocked_ids(policy, item)[0] == [item["candidate_id"]]


def test_unrelated_approach_pose_change_does_not_revalidate_terminal_block() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    changed = candidate(
        geometry_env_step=80,
        geometry_capture_id=80,
    )
    changed["approach_pose"][0] += 0.08

    blocked, revalidated = blocked_ids(policy, changed)

    assert blocked == [changed["candidate_id"]]
    assert revalidated == []


def test_attempt_revision_tracks_executed_point_and_offset() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate()
    base = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
        point_key="grasp_world_m",
    )
    offset = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
        point_key="grasp_world_m",
        offset_xyz=[0.02, 0.0, 0.0],
    )

    assert base["point_key"] == "grasp_world_m"
    assert base["geometry_revision"] == offset["geometry_revision"]
    assert (
        base["execution_spec_revision"]
        != offset["execution_spec_revision"]
    )


def test_small_modifier_changes_cannot_replenish_retry_budget() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate()
    for index, error in enumerate((0.071, 0.070, 0.069)):
        attempt = policy.bind_attempt(
            instance_id="track_0001",
            candidate=item,
            point_key="grasp_world_m",
            offset_xyz=[index * 1e-6, 0.0, 0.0],
        )
        policy.record_failure(
            attempt,
            target_error_m=error,
        )

    assert blocked_ids(policy, item)[0] == [item["candidate_id"]]


def test_explicit_none_offset_is_invalid_while_omission_defaults_to_zero() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate()

    omitted = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
    )
    explicit_none = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
        offset_xyz=None,
    )

    assert omitted["execution_spec"]["offset_xyz"] == [0.0, 0.0, 0.0]
    assert explicit_none == {}


def test_success_at_different_final_target_does_not_clear_old_block() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate()
    block(policy, item)
    different_target = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
        offset_xyz=[0.03, 0.0, 0.0],
    )

    policy.record_success(different_target)

    assert blocked_ids(policy, item)[0] == [item["candidate_id"]]


def test_success_at_same_final_target_clears_matching_block() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate()
    block(policy, item)
    same_target = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
    )

    policy.record_success(same_target)

    assert blocked_ids(policy, item)[0] == []


def test_instance_provenance_overrides_candidate_supplied_provenance() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate(
        geometry_env_step=999,
        geometry_capture_id=999,
    )
    instance = visible_instance(item)
    instance["operation_pose_candidate_provenance"] = {
        item["candidate_id"]: {
            "geometry_observation_env_step": 13,
            "geometry_observation_capture_id": 20,
        }
    }

    attempt = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
        instance=instance,
    )

    assert attempt["geometry_env_step"] == 13
    assert attempt["geometry_capture_id"] == 20


def test_instance_provenance_omission_cannot_fall_back_to_candidate_fields() -> None:
    policy = OperationCandidateLifecycle()
    item = candidate(
        geometry_env_step=999,
        geometry_capture_id=999,
    )
    instance = visible_instance(item)
    instance["operation_pose_candidate_provenance"] = {
        item["candidate_id"]: {
            "geometry_observation_env_step": 13,
        }
    }

    attempt = policy.bind_attempt(
        instance_id="track_0001",
        candidate=item,
        instance=instance,
    )

    assert attempt["geometry_env_step"] == 13
    assert attempt["geometry_capture_id"] is None


def test_verified_material_geometry_change_revalidates_same_id() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    new = candidate(
        geometry=NEW_POSE,
        geometry_env_step=80,
        geometry_capture_id=80,
    )

    blocked, revalidated = blocked_ids(policy, new)

    assert blocked == []
    assert len(revalidated) == 1
    assert revalidated[0]["candidate_id"] == new["candidate_id"]
    assert (
        revalidated[0]["max_pose_translation_change_m"]
        > 0.06
    )
    assert policy.planner_payload([visible_instance(new)]) == []


def test_fresh_candidate_shift_compensated_by_offset_stays_blocked() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    shifted = candidate(
        geometry_env_step=80,
        geometry_capture_id=80,
    )
    shifted["ee_target_pose"][0] += 0.02

    blocked, revalidated = policy.blocked_candidate_ids(
        instance_id="track_0001",
        arm="right",
        action_mode="grasp",
        candidates=[shifted],
        instance=visible_instance(shifted),
        point_key="grasp_world_m",
        offset_xyz=[-0.02, 0.0, 0.0],
    )

    assert blocked == [shifted["candidate_id"]]
    assert revalidated == []


def test_current_quaternion_cannot_turn_candidate_rotation_into_new_work() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    current_pose = [0.3, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0]
    attempt = policy.bind_attempt(
        instance_id="track_0001",
        candidate=old,
        point_key="grasp_world_m",
        target_quat_wxyz="current",
        current_pose=current_pose,
    )
    for error in (0.071, 0.070):
        policy.record_failure(attempt, target_error_m=error)

    rotated = candidate(
        geometry_env_step=80,
        geometry_capture_id=80,
    )
    rotated["ee_target_pose"][3:7] = [
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    blocked, revalidated = policy.blocked_candidate_ids(
        instance_id="track_0001",
        arm="right",
        action_mode="grasp",
        candidates=[rotated],
        instance=visible_instance(rotated),
        point_key="grasp_world_m",
        target_quat_wxyz="current",
        current_pose=current_pose,
    )

    assert blocked == [rotated["candidate_id"]]
    assert revalidated == []


def test_instance_provenance_revalidates_without_polluting_public_candidate() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    old_step = old.pop("_geometry_observation_env_step")
    old_capture = old.pop("_geometry_observation_capture_id")
    old_instance = visible_instance(old)
    old_instance["operation_pose_candidate_provenance"] = {
        old["candidate_id"]: {
            "geometry_observation_env_step": old_step,
            "geometry_observation_capture_id": old_capture,
        }
    }
    attempt = policy.bind_attempt(
        instance_id="track_0001",
        candidate=old,
        instance=old_instance,
        dispatch_env_step=24,
    )
    for error in (0.071531, 0.070254):
        policy.record_failure(
            attempt,
            target_error_m=error,
            env_step=24,
        )

    new = candidate(
        geometry=NEW_POSE,
        geometry_env_step=80,
        geometry_capture_id=80,
    )
    new_step = new.pop("_geometry_observation_env_step")
    new_capture = new.pop("_geometry_observation_capture_id")
    new_instance = visible_instance(new)
    new_instance["operation_pose_candidate_provenance"] = {
        new["candidate_id"]: {
            "geometry_observation_env_step": new_step,
            "geometry_observation_capture_id": new_capture,
        }
    }

    blocked, revalidated = policy.blocked_candidate_ids(
        instance_id="track_0001",
        arm="right",
        action_mode="grasp",
        candidates=[new],
        instance=new_instance,
    )

    assert blocked == []
    assert len(revalidated) == 1
    assert "_geometry_observation_env_step" not in new
    assert policy.planner_payload([new_instance]) == []


def test_new_candidate_id_cannot_bypass_same_geometry_block() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate(candidate_id="rgbd_volume:grasp:right:000")
    block(policy, old)
    alias = candidate(candidate_id="rgbd_volume:grasp:right:007")

    blocked, revalidated = blocked_ids(policy, alias)

    assert blocked == [alias["candidate_id"]]
    assert revalidated == []


def test_small_jitter_and_quaternion_sign_do_not_revalidate() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    jittered = candidate()
    for pose_key in (
        "object_contact_pose",
        "tcp_pose",
        "ee_target_pose",
        "approach_pose",
    ):
        jittered[pose_key][0] += 0.001
        jittered[pose_key][3:7] = [
            -value for value in jittered[pose_key][3:7]
        ]

    blocked, revalidated = blocked_ids(policy, jittered)

    assert blocked == [jittered["candidate_id"]]
    assert revalidated == []


def test_unverified_geometry_change_does_not_revalidate() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    new = candidate(
        geometry=NEW_POSE,
        geometry_env_step=80,
        geometry_capture_id=80,
    )
    pending = visible_instance(new)
    pending.update(
        {
            "status": "tracked",
            "position_state": "memory_valid",
            "action_geometry_state": "relocation_pending",
        }
    )

    blocked, revalidated = blocked_ids(
        policy,
        new,
        instance=pending,
    )

    assert blocked == [new["candidate_id"]]
    assert revalidated == []


def test_frozen_attempt_records_old_failure_not_post_action_geometry() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    old_attempt = policy.bind_attempt(
        instance_id="track_0001",
        candidate=old,
        dispatch_env_step=24,
    )
    new = candidate(
        geometry=NEW_POSE,
        geometry_env_step=80,
        geometry_capture_id=80,
    )

    policy.record_failure(old_attempt, target_error_m=0.071531)
    policy.record_failure(old_attempt, target_error_m=0.070254)

    new_blocked, _ = blocked_ids(policy, new)
    old_blocked, _ = blocked_ids(policy, old)
    assert new_blocked == []
    assert old_blocked == [old["candidate_id"]]


def test_reappearing_old_geometry_restores_its_planner_block() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    new = candidate(
        geometry=NEW_POSE,
        geometry_env_step=80,
        geometry_capture_id=80,
    )
    assert blocked_ids(policy, new)[0] == []
    assert policy.planner_payload([visible_instance(new)]) == []

    assert blocked_ids(policy, old)[0] == [old["candidate_id"]]
    payload = policy.planner_payload([visible_instance(old)])[0]
    assert "candidate_id" not in payload
    assert payload["blocked_candidate_count"] == 1
    assert payload["available_candidate_count"] == 0


def test_place_payload_exposes_target_not_private_candidate_id() -> None:
    policy = OperationCandidateLifecycle()
    place = {
        **candidate(
            candidate_id=(
                "place:held:track_0001:free-support:01:arm:right"
            )
        ),
        "action_mode": "place",
        "target_id": "place:held:track_0001:free-support:01",
        "target_kind": "free_support",
        "geometry_source": "runtime_dynamic_place_geometry",
        "holding_status": "verified",
        "support_valid": True,
        "free": True,
        "valid": True,
        "reachable_estimate": True,
    }
    block(policy, place)

    payload = policy.planner_payload([visible_instance(place)])

    assert payload == [
        {
            "instance_id": "track_0001",
            "arm": "right",
            "action_mode": "place",
            "target_id": "place:held:track_0001:free-support:01",
            "failure_count": 2,
            "last_target_error_m": 0.070254,
            "status": "all_operation_candidates_blocked",
            "blocked_candidate_count": 1,
            "available_candidate_count": 0,
            "runtime_will_select_next": False,
            "all_candidates_blocked": True,
        }
    ]


def test_evidence_only_dynamic_place_geometry_is_runtime_trusted() -> None:
    place = {
        "action_mode": "place",
        "target_id": "place:held:track_0001:free-support:01",
        "geometry_source": "runtime_dynamic_place_geometry",
        "holding_status": "provisional_evidence_only",
        "grasp_transport_policy": "evidence_only",
        "support_valid": True,
        "free": True,
        "valid": True,
        "reachable_estimate": True,
    }

    assert operation_geometry_is_trusted({}, place) is True
    assert operation_geometry_is_trusted(
        {},
        {**place, "grasp_transport_policy": "strict"},
    ) is False


def test_verified_dynamic_place_geometry_can_revalidate_while_track_occluded() -> None:
    policy = OperationCandidateLifecycle()
    target_id = "place:held:track_0001:free-support:01"
    old = {
        **candidate(
            candidate_id=f"{target_id}:arm:right",
        ),
        "action_mode": "place",
        "target_id": target_id,
        "target_kind": "free_support",
        "geometry_source": "runtime_dynamic_place_geometry",
        "holding_status": "verified",
        "support_valid": True,
        "free": True,
        "valid": True,
        "reachable_estimate": True,
    }
    block(policy, old)
    new = {
        **candidate(
            candidate_id=f"{target_id}:arm:right",
            geometry=NEW_POSE,
            geometry_env_step=80,
            geometry_capture_id=80,
        ),
        "action_mode": "place",
        "target_id": target_id,
        "target_kind": "free_support",
        "geometry_source": "runtime_dynamic_place_geometry",
        "holding_status": "verified",
        "support_valid": True,
        "free": True,
        "valid": True,
        "reachable_estimate": True,
    }
    held_track = {
        "instance_id": "track_0001",
        "status": "tracked",
        "position_state": "current_verified",
        "action_geometry_state": "verified",
        "actionable": True,
    }

    blocked, revalidated = policy.blocked_candidate_ids(
        instance_id="track_0001",
        arm="right",
        action_mode="place",
        candidates=[new],
        instance=held_track,
    )

    assert blocked == []
    assert len(revalidated) == 1


def test_success_clears_every_overlapping_equivalent_version() -> None:
    policy = OperationCandidateLifecycle()

    def shifted(x_m: float, capture_id: int) -> dict:
        item = candidate(geometry_capture_id=capture_id)
        for pose_key in (
            "object_contact_pose",
            "tcp_pose",
            "ee_target_pose",
            "approach_pose",
        ):
            item[pose_key][0] += x_m
        return item

    left = shifted(0.0, 20)
    right = shifted(0.018, 21)
    middle = shifted(0.009, 22)
    block(policy, left)
    block(policy, right)

    policy.record_success(
        policy.bind_attempt(
            instance_id="track_0001",
            candidate=middle,
            dispatch_env_step=30,
        )
    )

    assert policy.planner_payload([visible_instance(middle)]) == []
    assert blocked_ids(policy, middle)[0] == []


def test_result_target_cannot_override_frozen_place_scope() -> None:
    policy = OperationCandidateLifecycle()
    target_a = "place:held:track_0001:free-support:a"
    place = {
        **candidate(candidate_id=f"{target_a}:arm:right"),
        "action_mode": "place",
        "target_id": target_a,
        "geometry_source": "runtime_dynamic_place_geometry",
        "holding_status": "verified",
        "support_valid": True,
        "free": True,
        "valid": True,
    }
    attempt = policy.bind_attempt(
        instance_id="track_0001",
        candidate=place,
    )
    for error in (0.04, 0.039):
        policy.record_failure(
            attempt,
            target_error_m=error,
            target_id="place:held:track_0001:free-support:b",
        )

    payload = policy.planner_payload([visible_instance(place)])

    assert payload[0]["target_id"] == target_a


def test_metadata_only_change_does_not_revalidate_geometry() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    renamed_source = candidate(
        geometry_env_step=80,
        geometry_capture_id=80,
    )
    renamed_source["geometry_source"] = "another_backend_label"
    renamed_source["target_kind"] = "metadata_only"

    blocked, revalidated = blocked_ids(policy, renamed_source)

    assert blocked == [renamed_source["candidate_id"]]
    assert revalidated == []


def test_changed_geometry_without_new_provenance_stays_blocked() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    changed_but_stale = candidate(
        geometry=NEW_POSE,
        geometry_env_step=13,
        geometry_capture_id=20,
    )

    blocked, revalidated = blocked_ids(policy, changed_but_stale)

    assert blocked == [changed_but_stale["candidate_id"]]
    assert revalidated == []


def test_untrusted_new_id_and_geometry_cannot_replenish_budget() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate()
    block(policy, old)
    renumbered = candidate(
        candidate_id="rgbd_volume:grasp:right:007",
        geometry=NEW_POSE,
        geometry_env_step=80,
        geometry_capture_id=80,
    )
    pending = visible_instance(renumbered)
    pending.update(
        {
            "status": "tracked",
            "position_state": "memory_valid",
            "action_geometry_state": "relocation_pending",
        }
    )

    blocked, revalidated = blocked_ids(
        policy,
        renumbered,
        instance=pending,
    )

    assert blocked == [renumbered["candidate_id"]]
    assert revalidated == []


def test_planner_payload_uses_only_current_candidate_aliases() -> None:
    policy = OperationCandidateLifecycle()
    old = candidate(candidate_id="rgbd_volume:grasp:right:000")
    block(policy, old)
    renamed_same_geometry = candidate(
        candidate_id="rgbd_volume:grasp:right:007"
    )

    payload = policy.planner_payload(
        [visible_instance(renamed_same_geometry)]
    )

    assert "candidate_id" not in payload[0]
    assert payload[0]["blocked_candidate_count"] == 1
    assert payload[0]["available_candidate_count"] == 0

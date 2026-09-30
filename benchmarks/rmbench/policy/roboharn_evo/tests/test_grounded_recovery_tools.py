from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from policy.roboharn_evo.agent.environment import RMBenchEnvAdapter
from policy.roboharn_evo.agent.operation_candidates import (
    capture_held_object_to_tcp_attachment,
    with_dynamic_place_candidates,
)
from policy.roboharn_evo.agent.recovery.rmbench_recovery_adapter import RMBenchRecoveryAdapter
from policy.roboharn_evo.agent.recovery.tool_dispatcher import RecoveryToolDispatcher
from policy.roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall


class FakeRMBenchEnv:
    def __init__(self) -> None:
        self.take_action_cnt = 0
        self.step_lim = 100
        self.eval_success = False
        self.max_reward = 0.0
        self.instruction = "test"
        self.is_dual_arm = False
        self.left_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.right_pose = np.array([0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.left_gripper = 0.5
        self.actions: list[dict[str, object]] = []

    def check_success(self) -> bool:
        return False

    def get_obs(self) -> dict:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": np.zeros(8, dtype=np.float32)},
            "endpose": {
                "left_endpose": self.left_pose.copy(),
                "right_endpose": self.right_pose.copy(),
                "left_gripper": self.left_gripper,
                "gripper": self.left_gripper,
            },
        }

    def take_action(self, action: np.ndarray, action_type: str = "qpos") -> None:
        action = np.asarray(action, dtype=np.float32)
        self.actions.append({"action_type": action_type, "action": action.copy()})
        self.take_action_cnt += 1
        if action_type == "ee":
            self.left_pose = action[:7].copy()
            self.left_gripper = float(action[7])


class LaggedFakeRMBenchEnv(FakeRMBenchEnv):
    def take_action(self, action: np.ndarray, action_type: str = "qpos") -> None:
        action = np.asarray(action, dtype=np.float32)
        self.actions.append({"action_type": action_type, "action": action.copy()})
        self.take_action_cnt += 1
        if action_type == "ee":
            self.left_pose[:3] += 0.5 * (action[:3] - self.left_pose[:3])
            self.left_pose[3:] = action[3:7]
            self.left_gripper = float(action[7])


class SuccessOnFirstActionEnv(FakeRMBenchEnv):
    def take_action(self, action: np.ndarray, action_type: str = "qpos") -> None:
        super().take_action(action, action_type=action_type)
        if len(self.actions) == 1:
            self.eval_success = True


class CheckSuccessOnlyEnv(FakeRMBenchEnv):
    def check_success(self) -> bool:
        return True


class CountingObservationEnv(FakeRMBenchEnv):
    def __init__(self) -> None:
        super().__init__()
        self.observation_calls = 0

    def get_obs(self) -> dict:
        self.observation_calls += 1
        return super().get_obs()


def snapshot_for(env: FakeRMBenchEnv):
    return RMBenchEnvAdapter.from_env(env, env.get_obs())


def test_reobserve_scene_ablation_hides_and_rejects_tool() -> None:
    env = CountingObservationEnv()
    latest_snapshot = snapshot_for(env)
    dispatcher = RecoveryToolDispatcher()
    dispatcher.set_reobserve_scene_enabled(False)

    capabilities = dispatcher.capabilities(env)

    assert "reobserve_scene" not in capabilities.available_tools()
    assert capabilities.notes["disabled_tools_by_ablation"] == {
        "reobserve_scene": "agent.recovery.enable_reobserve"
    }
    calls_before_dispatch = env.observation_calls

    result = dispatcher.dispatch(
        RecoveryToolCall(tool_name="reobserve_scene", args={}),
        task_env=env,
        latest_snapshot=latest_snapshot,
    )

    assert result.success is False
    assert result.details["disabled_by_ablation"] is True
    assert result.details["disabled_tool"] == "reobserve_scene"
    assert (
        result.details["ablation_config_key"]
        == "agent.recovery.enable_reobserve"
    )
    assert "reobserve_scene" not in result.details["available_tools"]
    assert env.observation_calls == calls_before_dispatch
    assert dispatcher.latest_snapshot is latest_snapshot


def test_reobserve_scene_remains_enabled_by_default() -> None:
    env = FakeRMBenchEnv()
    dispatcher = RecoveryToolDispatcher()

    assert "reobserve_scene" in dispatcher.available_tools(env)


def test_close_gripper_reports_exact_close_time_pose() -> None:
    env = FakeRMBenchEnv()
    env.robot = SimpleNamespace(
        get_left_arm_jointState=lambda: np.zeros(8, dtype=np.float32)
    )
    env.left_pose = np.asarray(
        [0.11, -0.02, 0.83, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    adapter = RMBenchRecoveryAdapter(env)

    result = adapter.close_gripper(
        snapshot_for(env),
        {"arm": "left"},
    )

    assert result.result.success is True
    assert result.result.details["step_count"] == 1
    np.testing.assert_allclose(
        result.result.details["observed_pose"],
        env.left_pose,
        atol=1e-6,
    )
    assert result.result.details["observed_poses_by_arm"] == {
        "left": result.result.details["observed_pose"]
    }


def test_dispatcher_executes_bare_open_gripper() -> None:
    env = FakeRMBenchEnv()
    env.robot = SimpleNamespace(
        get_left_arm_jointState=lambda: np.zeros(8, dtype=np.float32)
    )
    dispatcher = RecoveryToolDispatcher()

    result = dispatcher.dispatch(
        RecoveryToolCall(
            tool_name="open_gripper",
            args={"arm": "left"},
        ),
        task_env=env,
        latest_snapshot=snapshot_for(env),
    )

    assert result.success is True
    assert result.details["arm"] == "left"
    assert result.details["gripper_value"] == 1.0
    assert len(env.actions) == 1


def test_dispatcher_rejects_invalid_open_gripper_arm() -> None:
    env = FakeRMBenchEnv()
    env.robot = SimpleNamespace(
        get_left_arm_jointState=lambda: np.zeros(8, dtype=np.float32)
    )
    dispatcher = RecoveryToolDispatcher()

    results = [
        dispatcher.dispatch(
            RecoveryToolCall(tool_name="open_gripper", args=args),
            task_env=env,
            latest_snapshot=snapshot_for(env),
        )
        for args in ({}, {"arm": "middle"})
    ]

    assert [result.success for result in results] == [False, False]
    assert "arm is required" in results[0].message
    assert "arm must be left, right, or both" in results[1].message
    assert env.actions == []


def test_move_ee_to_grounded_instance_executes_bounded_ee_action() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_left",
                "status": "visible",
                "world_m": [0.12, 0.0, 0.0],
                "approach_world_m": [0.10, 0.0, 0.0],
            }
        ],
        "task_focus": {"target_instances": ["object_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "instance_id": "object_left",
            "point_key": "approach",
            "max_translation": 0.04,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    assert env.actions[-1]["action_type"] == "ee"
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.04, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(env.actions[-1]["action"][3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert result.result.details["target_pose"][:3] == [0.1, 0.0, 0.0]
    assert result.result.details["executed_pose"][:3] == [0.04, 0.0, 0.0]
    assert result.result.details["observed_pose"][:3] == [0.04, 0.0, 0.0]
    assert np.isclose(result.result.details["target_observation_error_m"], 0.06)
    assert result.result.details["target_reached"] is False
    assert result.result.details["instance_id"] == "object_left"


def test_grounded_pose_move_uses_observed_feedback_until_target_converges() -> None:
    env = LaggedFakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_left",
                "status": "visible",
                "approach_world_m": [0.10, 0.0, 0.0],
            }
        ],
        "task_focus": {"target_instances": ["object_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "instance_id": "object_left",
            "point_key": "approach_world_m",
            "max_translation": 0.12,
            "steps": 4,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    assert result.result.details["target_reached"] is True
    assert result.result.details["executed_steps"] == 4
    assert result.result.details["target_error_history_m"] == [0.05, 0.025, 0.0125, 0.00625]
    assert np.isclose(result.result.details["target_observation_error_m"], 0.00625)
    assert env.take_action_cnt == 4


def test_move_ee_to_grounded_instance_supports_grasp_point_key() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_left",
                "status": "visible",
                "world_m": [0.12, 0.0, 0.0],
                "approach_world_m": [0.10, 0.0, 0.08],
                "grasp_world_m": [0.10, 0.0, 0.02],
            }
        ],
        "task_focus": {"tool_instances": ["object_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "tool",
            "point_key": "grasp",
            "max_translation": 0.2,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.10, 0.0, 0.02], atol=1e-6)
    assert result.result.details["target_pose"][:3] == [0.1, 0.0, 0.02]


def test_move_ee_to_grounded_instance_selects_next_unblocked_arm_candidate() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    first_id = "rgbd_surface:grasp:left:000"
    second_id = "rgbd_surface:grasp:left:001"

    def candidate(candidate_id: str, target_x: float, source_index: int) -> dict:
        return {
            "candidate_id": candidate_id,
            "source_candidate_index": source_index,
            "action_mode": "grasp",
            "arm": "left",
            "object_contact_pose": [target_x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "tcp_pose": [target_x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "ee_target_pose": [target_x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "approach_pose": [target_x, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0],
            "approach_direction": [0.0, 0.0, -1.0],
            "geometry_source": "rgbd_surface_normal_principal_axes",
            "priority": source_index,
        }

    scene_memory = {
        "instances": [
            {
                "instance_id": "object_left",
                "status": "visible",
                "grasp_world_m": [0.02, 0.0, 0.0],
                "operation_pose_candidates": [
                    candidate(first_id, 0.04, 0),
                    candidate(second_id, 0.08, 1),
                ],
            }
        ],
        "task_focus": {"tool_instances": ["object_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "tool",
            "point_key": "grasp",
            "max_translation": 0.2,
            "_scene_memory": scene_memory,
            "_operation_action_mode": "grasp",
            "_blocked_operation_candidate_ids": [first_id],
        },
    )

    assert result.result.success is True
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.08, 0.0, 0.0])
    assert result.result.details["operation_candidate_id"] == second_id
    assert result.result.details["operation_action_mode"] == "grasp"
    assert result.result.details["target_reached"] is True


def test_move_ee_to_grounded_instance_executes_runtime_selected_place_target() -> None:
    env = FakeRMBenchEnv()
    env.left_pose = np.asarray(
        [0.01, -0.01, 0.24, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    env.left_gripper = 0.0
    base_scene = {
        "instances": [
            {
                "instance_id": "held",
                "track_id": "held",
                "world_m": [0.0, 0.0, 0.20],
                "latest_world_m": [0.0, 0.0, 0.20],
                "first_observed_world_m": [0.0, 0.0, 0.02],
                "top_surface_world_m": [0.0, 0.0, 0.22],
                "quality": {
                    "world_extent_m": [0.04, 0.04, 0.04],
                    "world_z_max_m": 0.22,
                },
                "operation_pose_candidates": [],
            },
            {
                "instance_id": "anchor",
                "track_id": "anchor",
                "world_m": [0.10, 0.0, 0.02],
                "latest_world_m": [0.10, 0.0, 0.02],
                "first_observed_world_m": [0.10, 0.0, 0.02],
                "top_surface_world_m": [0.10, 0.0, 0.04],
                "quality": {
                    "world_extent_m": [0.04, 0.04, 0.04],
                    "world_z_max_m": 0.04,
                },
                "operation_pose_candidates": [],
            },
        ],
        "task_focus": {"tool_instances": ["held"]},
    }
    grasp_candidate_id = "candidate:left:held:001"
    grasp_attempt_step = 12
    grasp_attempt_nonce = (
        "left:held:candidate:left:held:001:12"
    )
    attachment = capture_held_object_to_tcp_attachment(
        object_world_m=[0.0, 0.0, 0.20],
        robot_arm_state={
            "xyz": env.left_pose[:3].tolist(),
            "quat_wxyz": env.left_pose[3:].tolist(),
        },
    )
    assert attachment is not None
    scene = with_dynamic_place_candidates(
        base_scene,
        manipulation_state={
            "left": {
                "phase": "holding",
                "held_instance_id": "held",
                "holding_confirmed": True,
                "transport_authorized": True,
                "grasp_candidate_id": grasp_candidate_id,
                "grasp_attempt_step": grasp_attempt_step,
                "grasp_attempt_nonce": grasp_attempt_nonce,
                "held_object_to_tcp_attachment": {
                    **attachment,
                    "grasp_candidate_id": grasp_candidate_id,
                    "grasp_attempt_nonce": grasp_attempt_nonce,
                    "capture_step": grasp_attempt_step,
                    "source": (
                        "runtime_multiview_grasp_motion_verified"
                    ),
                    "authority": "runtime_multiview_grasp_motion",
                },
            }
        },
        robot_state={
            "left": {
                "xyz": env.left_pose[:3].tolist(),
                "quat_wxyz": env.left_pose[3:].tolist(),
            }
        },
    )
    target_id = next(
        item["target_id"]
        for item in scene["operation_targets"]
        if item["target_kind"] == "vacated_pose"
    )

    result = RMBenchRecoveryAdapter(env).move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "instance_id": "held",
            "held_instance_id": "held",
            "action_mode": "place",
            "target_id": target_id,
            "point_key": "place_world_m",
            "max_translation": 0.12,
            "steps": 2,
            "_operation_action_mode": "place",
            "_scene_memory": scene,
        },
    )

    assert result.result.success is True
    assert result.result.details["operation_target_id"] == target_id
    assert result.result.details["place_target_revalidated"] is True
    assert result.result.details["support_valid"] is True
    assert result.result.details["free"] is True
    assert result.result.details["target_reached"] is True
    expected_object_delta = np.asarray([0.0, 0.0, 0.02]) - np.asarray(
        [0.0, 0.0, 0.20]
    )
    np.testing.assert_allclose(
        np.asarray(result.result.details["target_pose"][:3])
        - np.asarray([0.01, -0.01, 0.24]),
        expected_object_delta,
        atol=1e-6,
    )


def test_transient_contact_result_binds_actual_spatial_state_signature() -> None:
    env = FakeRMBenchEnv()
    result = RMBenchRecoveryAdapter(env).contact_displace(
        snapshot_for(env),
        {
            "arm": "left",
            "axis": "x",
            "direction": "positive",
            "distance": 0.01,
            "_actual_spatial_state_signature": "coordstate123",
        },
    )

    assert result.result.success is True
    assert (
        result.result.details["actual_spatial_state_signature"]
        == "coordstate123"
    )


def test_non_operation_point_key_keeps_legacy_geometry_when_candidates_exist() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_left",
                "status": "visible",
                "top_surface_world_m": [0.03, 0.0, 0.0],
                "operation_pose_candidates": [
                    {
                        "candidate_id": "rgbd_surface:grasp:left:000",
                        "source_candidate_index": 0,
                        "action_mode": "grasp",
                        "arm": "left",
                        "object_contact_pose": [0.08, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        "tcp_pose": [0.08, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        "ee_target_pose": [0.08, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        "approach_pose": [0.08, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0],
                        "approach_direction": [0.0, 0.0, -1.0],
                        "geometry_source": "rgbd_surface_normal_principal_axes",
                    }
                ],
            }
        ],
        "task_focus": {"target_instances": ["object_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "instance_id": "object_left",
            "point_key": "top_surface_world_m",
            "max_translation": 0.2,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.03, 0.0, 0.0])
    assert "operation_candidate_id" not in result.result.details


def test_move_ee_to_grounded_instance_can_preserve_height_for_carry() -> None:
    env = FakeRMBenchEnv()
    env.left_pose = np.array([0.0, 0.0, 0.95, 0.5, -0.5, 0.5, 0.5], dtype=np.float32)
    env.left_gripper = 0.0
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "held_tool_left",
                "status": "visible",
                "world_m": [0.0, 0.0, 0.85],
                "latest_world_m": [0.0, 0.0, 0.85],
                "quality": {"world_extent_m": [0.08, 0.08, 0.08], "world_z_max_m": 0.89},
            },
            {
                "instance_id": "destination_left",
                "status": "visible",
                "world_m": [0.12, -0.18, 0.76],
                "quality": {"world_extent_m": [0.04, 0.04, 0.04], "world_z_max_m": 0.78},
            }
        ],
        "task_focus": {
            "target_instances": ["destination_left"],
            "tool_instances": ["held_tool_left"],
        },
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "target",
            "point_key": "world_m",
            "preserve_height": True,
            "target_quat_wxyz": "preserve",
            "max_translation": 0.3,
            "steps": 2,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.12, -0.18, 0.95], atol=1e-6)
    np.testing.assert_allclose(env.actions[-1]["action"][3:7], [0.5, -0.5, 0.5, 0.5], atol=1e-6)
    assert result.result.details["target_pose"][:3] == [0.12, -0.18, 0.95]
    assert result.result.details["clearance_lift"] == 0.0


def test_move_ee_to_grounded_instance_lifts_before_lateral_carry_clearance() -> None:
    env = FakeRMBenchEnv()
    env.left_pose = np.array([0.0, 0.0, 0.95, 0.5, -0.5, 0.5, 0.5], dtype=np.float32)
    env.left_gripper = 0.0
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "held_tool_left",
                "world_m": [0.0, 0.0, 0.80],
                "latest_world_m": [0.0, 0.0, 0.80],
                "quality": {"world_extent_m": [0.10, 0.10, 0.10], "world_z_max_m": 0.85},
            },
            {
                "instance_id": "destination_left",
                "world_m": [0.10, -0.10, 0.75],
                "quality": {"world_extent_m": [0.04, 0.04, 0.04], "world_z_max_m": 0.77},
            },
        ],
        "task_focus": {
            "target_instances": ["destination_left"],
            "tool_instances": ["held_tool_left"],
        },
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "target",
            "point_key": "world_m",
            "preserve_height": True,
            "target_quat_wxyz": "preserve",
            "clearance_margin": 0.01,
            "clearance_steps": 2,
            "max_translation": 0.12,
            "steps": 2,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    assert len(env.actions) == 4
    np.testing.assert_allclose(env.actions[0]["action"][:3], [0.0, 0.0, 0.965], atol=1e-6)
    np.testing.assert_allclose(env.actions[1]["action"][:3], [0.0, 0.0, 0.98], atol=1e-6)
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.10, -0.10, 0.98], atol=1e-6)
    assert result.result.details["clearance_lift"] == 0.03
    assert result.result.details["held_instance_id"] == "held_tool_left"
    assert result.result.details["target_instance_id"] == "destination_left"


def test_grounded_carry_does_not_start_second_phase_after_clearance_triggers_success() -> None:
    env = SuccessOnFirstActionEnv()
    env.left_pose = np.array([0.0, 0.0, 0.95, 0.5, -0.5, 0.5, 0.5], dtype=np.float32)
    env.left_gripper = 0.0
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "held_instance",
                "world_m": [0.0, 0.0, 0.80],
                "latest_world_m": [0.0, 0.0, 0.80],
                "quality": {"world_extent_m": [0.10, 0.10, 0.10], "world_z_max_m": 0.85},
            },
            {
                "instance_id": "target_instance",
                "world_m": [0.10, -0.10, 0.75],
                "quality": {"world_extent_m": [0.04, 0.04, 0.04], "world_z_max_m": 0.77},
            },
        ],
        "task_focus": {
            "target_instances": ["target_instance"],
            "tool_instances": ["held_instance"],
        },
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "target",
            "point_key": "world_m",
            "preserve_height": True,
            "target_quat_wxyz": "preserve",
            "clearance_margin": 0.01,
            "clearance_steps": 2,
            "max_translation": 0.12,
            "steps": 2,
            "_scene_memory": scene_memory,
        },
    )

    assert len(env.actions) == 1
    assert result.result.success is False
    assert result.result.details["terminal_skip"] is True
    assert result.result.details["skip_reason"] == "environment_eval_success"
    assert result.latest_snapshot.eval_success is True


def test_move_ee_to_grounded_instance_rejects_clearance_lift_over_limit() -> None:
    env = FakeRMBenchEnv()
    env.left_pose = np.array([0.0, 0.0, 0.95, 0.5, -0.5, 0.5, 0.5], dtype=np.float32)
    env.left_gripper = 0.0
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "held_tool_left",
                "world_m": [0.0, 0.0, 0.70],
                "quality": {"world_extent_m": [0.10, 0.10, 0.10], "world_z_max_m": 0.75},
            },
            {
                "instance_id": "destination_left",
                "world_m": [0.10, -0.10, 0.80],
                "quality": {"world_extent_m": [0.04, 0.04, 0.04], "world_z_max_m": 0.82},
            },
        ],
        "task_focus": {
            "target_instances": ["destination_left"],
            "tool_instances": ["held_tool_left"],
        },
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "target",
            "point_key": "world_m",
            "preserve_height": True,
            "clearance_margin": 0.02,
            "max_clearance_lift": 0.05,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is False
    assert "required carry clearance lift exceeds" in result.result.message
    assert env.actions == []


def test_move_ee_to_grounded_instance_rejects_non_boolean_preserve_height() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [{"instance_id": "destination_left", "world_m": [0.12, -0.18, 0.76]}],
        "task_focus": {"target_instances": ["destination_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "target",
            "point_key": "world_m",
            "preserve_height": "yes",
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is False
    assert "preserve_height must be a boolean" in result.result.message
    assert env.actions == []


def test_move_ee_to_grounded_instance_uses_grounded_grasp_quaternion() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_left",
                "status": "visible",
                "grasp_world_m": [0.10, 0.0, 0.02],
                "grasp_quat_wxyz": [0.5, -0.5, 0.5, 0.5],
            }
        ],
        "task_focus": {"tool_instances": ["object_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "tool",
            "point_key": "grasp",
            "target_quat_wxyz": "grounded",
            "max_translation": 0.2,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    np.testing.assert_allclose(env.actions[-1]["action"][3:7], [0.5, -0.5, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(result.result.details["target_pose"][3:7], [0.5, -0.5, 0.5, 0.5], atol=1e-6)


def test_move_ee_to_grounded_instance_uses_explicit_numeric_quaternion() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_left",
                "status": "visible",
                "grasp_world_m": [0.10, 0.0, 0.02],
                "grasp_quat_wxyz": [0.5, -0.5, 0.5, 0.5],
            }
        ],
        "task_focus": {"tool_instances": ["object_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "tool",
            "point_key": "grasp",
            "target_quat_wxyz": [0.0, 0.0, 2.0, 0.0],
            "max_translation": 0.2,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    np.testing.assert_allclose(
        env.actions[-1]["action"][3:7],
        [0.0, 0.0, 1.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.result.details["target_pose"][3:7],
        [0.0, 0.0, 1.0, 0.0],
        atol=1e-6,
    )


def test_grounded_pose_contract_does_not_require_current_pose_for_grounded_quat() -> None:
    adapter = RMBenchRecoveryAdapter(FakeRMBenchEnv())

    target_pose = adapter._target_pose_from_grounded_instance(
        args={
            "point_key": "grasp",
            "target_quat_wxyz": "grounded",
        },
        latest_snapshot=None,
        arm="left",
        tool_name="move_ee_to_grounded_instance",
        instance={
            "instance_id": "object_left",
            "grasp_world_m": [0.10, 0.0, 0.02],
            "grasp_quat_wxyz": [0.0, 0.0, 2.0, 0.0],
        },
    )

    assert isinstance(target_pose, np.ndarray)
    np.testing.assert_allclose(
        target_pose,
        [0.10, 0.0, 0.02, 0.0, 0.0, 1.0, 0.0],
        atol=1e-6,
    )


def test_grounded_pose_contract_preserves_snapshot_failure_semantics() -> None:
    adapter = RMBenchRecoveryAdapter(FakeRMBenchEnv())

    result = adapter._target_pose_from_grounded_instance(
        args={"point_key": "world_m", "target_quat_wxyz": "current"},
        latest_snapshot=None,
        arm="left",
        tool_name="move_ee_to_grounded_instance",
        instance={"instance_id": "object_left", "world_m": [0.1, 0.0, 0.0]},
    )

    assert not isinstance(result, np.ndarray)
    assert result.result.success is False
    assert result.result.message == "latest snapshot unavailable for EE recovery"
    assert result.result.details == {}


def test_grounded_pose_contract_preserves_public_point_error_details() -> None:
    adapter = RMBenchRecoveryAdapter(FakeRMBenchEnv())
    instance = {
        "instance_id": "object_left",
        "status": "visible",
        "world_m": [0.1, 0.0, 0.0],
    }

    result = adapter._target_pose_from_grounded_instance(
        args={"point_key": "grasp"},
        latest_snapshot=None,
        arm="left",
        tool_name="move_ee_to_grounded_instance",
        instance=instance,
    )

    assert not isinstance(result, np.ndarray)
    assert result.result.success is False
    assert result.result.message == "scene instance has no finite grasp_world_m"
    assert result.result.details == {
        "instance_id": "object_left",
        "available_keys": sorted(instance),
    }


def test_move_ee_to_grounded_instance_rejects_missing_requested_grounded_quaternion() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [{"instance_id": "object_left", "status": "visible", "grasp_world_m": [0.10, 0.0, 0.02]}],
        "task_focus": {"tool_instances": ["object_left"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "role": "tool",
            "point_key": "grasp",
            "target_quat_wxyz": "grounded",
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is False
    assert "grasp_quat_wxyz" in result.result.message
    assert env.actions == []


def test_move_ee_to_grounded_instance_requires_scene_memory() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {"arm": "left", "instance_id": "object_left"},
    )

    assert result.result.success is False
    assert "scene_memory" in result.result.message
    assert env.actions == []


def test_move_ee_to_grounded_instance_does_not_guess_first_visible_instance() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_01",
                "status": "visible",
                "world_m": [0.10, 0.0, 0.0],
                "approach_world_m": [0.12, 0.0, 0.0],
            }
        ],
        "task_focus": {},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {"arm": "left", "role": "target", "_scene_memory": scene_memory},
    )

    assert result.result.success is False
    assert "could not resolve scene instance" in result.result.message
    assert env.actions == []


def test_grounded_instance_rejects_invalid_explicit_ref_without_focus_fallback() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_01",
                "track_id": "track_0001",
                "approach_world_m": [0.02, 0.0, 0.0],
            }
        ],
        "task_focus": {
            "target_instances": ["object_01"],
            "identity_binding_required": True,
        },
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "instance_ref": "track_missing",
            "role": "target",
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is False
    assert "explicit scene instance reference did not resolve uniquely" in result.result.message
    assert result.result.details["match_count"] == 0
    assert env.actions == []


def test_identity_bound_grounded_instance_requires_explicit_ref() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_01",
                "track_id": "track_0001",
                "approach_world_m": [0.02, 0.0, 0.0],
            }
        ],
        "task_focus": {
            "target_instances": ["object_01"],
            "identity_binding_required": True,
        },
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {"arm": "left", "role": "target", "_scene_memory": scene_memory},
    )

    assert result.result.success is False
    assert "requires an explicit instance_id or instance_ref" in result.result.message
    assert env.actions == []


def test_grounded_instance_rejects_ambiguous_focus_without_identity_binding() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {"instance_id": "object_01", "approach_world_m": [0.02, 0.0, 0.0]},
            {"instance_id": "object_02", "approach_world_m": [0.03, 0.0, 0.0]},
        ],
        "task_focus": {"target_instances": ["object_01", "object_02"]},
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {"arm": "left", "role": "target", "_scene_memory": scene_memory},
    )

    assert result.result.success is False
    assert "scene-memory focus is ambiguous" in result.result.message
    assert result.result.details["focused_instance_count"] == 2
    assert env.actions == []


def test_identity_bound_grounded_instance_accepts_exact_track_ref() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    scene_memory = {
        "instances": [
            {
                "instance_id": "object_01",
                "track_id": "track_0001",
                "approach_world_m": [0.02, 0.0, 0.0],
            }
        ],
        "task_focus": {
            "target_instances": ["object_01"],
            "identity_binding_required": True,
        },
    }

    result = adapter.move_ee_to_grounded_instance(
        snapshot_for(env),
        {
            "arm": "left",
            "instance_ref": "track_0001",
            "role": "target",
            "max_translation": 0.03,
            "_scene_memory": scene_memory,
        },
    )

    assert result.result.success is True
    assert result.result.details["instance_id"] == "object_01"
    assert len(env.actions) == 1


def test_move_ee_to_pose_preserves_quaternion_and_clamps_translation() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)

    result = adapter.move_ee_to_pose(
        snapshot_for(env),
        {"arm": "left", "target_xyz": [0.20, 0.0, 0.0], "max_translation": 0.03},
    )

    assert result.result.success is True
    assert env.actions[-1]["action_type"] == "ee"
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.03, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(env.actions[-1]["action"][3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert result.result.details["target_pose"][:3] == [0.2, 0.0, 0.0]
    assert result.result.details["executed_pose"][:3] == [0.03, 0.0, 0.0]
    assert result.result.details["start_pose"][:3] == [0.0, 0.0, 0.0]
    assert result.result.details["observed_displacement_xyz"] == [
        0.03,
        0.0,
        0.0,
    ]
    assert np.isclose(
        result.result.details["observed_displacement_m"],
        0.03,
    )


def test_move_ee_to_pose_can_execute_multiple_steps() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)

    result = adapter.move_ee_to_pose(
        snapshot_for(env),
        {"arm": "left", "target_xyz": [0.09, 0.0, 0.0], "max_translation": 0.03, "steps": 3},
    )

    assert result.result.success is True
    assert len(env.actions) == 3
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.09, 0.0, 0.0], atol=1e-6)
    assert result.result.details["steps"] == 3
    assert result.result.details["executed_pose"][:3] == [0.09, 0.0, 0.0]


def test_relative_ee_recovery_splits_distance_across_steps() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)

    result = adapter.retreat_arm(
        snapshot_for(env),
        {"arm": "left", "axis": "x", "direction": "positive", "distance": 0.03, "steps": 3},
    )

    assert result.result.success is True
    assert len(env.actions) == 3
    np.testing.assert_allclose(env.actions[0]["action"][:3], [0.01, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(env.actions[-1]["action"][:3], [0.03, 0.0, 0.0], atol=1e-6)
    assert result.result.details["steps"] == 3
    assert result.result.details["step_distance"] == 0.01
    assert result.result.details["start_pose"][:3] == [0.0, 0.0, 0.0]
    assert result.result.details["observed_pose"][:3] == [0.03, 0.0, 0.0]
    assert result.result.details["observed_displacement_xyz"] == [0.03, 0.0, 0.0]
    assert np.isclose(result.result.details["observed_axis_displacement_m"], 0.03)


def test_dispatch_batch_skips_contact_when_grounded_move_has_not_reached_target() -> None:
    env = FakeRMBenchEnv()
    dispatcher = RecoveryToolDispatcher()
    scene_memory = {
        "instances": [{"instance_id": "target_left", "approach_world_m": [0.10, 0.0, 0.0]}],
        "task_focus": {"target_instances": ["target_left"]},
    }

    results = dispatcher.dispatch_batch(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "arm": "left",
                    "instance_id": "target_left",
                    "point_key": "approach_world_m",
                    "max_translation": 0.04,
                    "_scene_memory": scene_memory,
                },
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "negative", "distance": 0.01},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        task_env=env,
        latest_snapshot=snapshot_for(env),
    )

    assert len(results) == 3
    assert results[0].success is True
    assert results[0].details["target_reached"] is False
    assert results[1].success is False
    assert results[1].details["skipped"] is True
    assert "stopped 0.06m from its target" in results[1].message
    assert results[2].success is True
    assert len(env.actions) == 1


def test_dispatch_batch_stops_internal_and_remaining_actions_after_environment_success() -> None:
    env = SuccessOnFirstActionEnv()
    dispatcher = RecoveryToolDispatcher()

    results = dispatcher.dispatch_batch(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_pose",
                args={"arm": "left", "target_xyz": [0.09, 0.0, 0.0], "max_translation": 0.03, "steps": 3},
            ),
            RecoveryToolCall(
                tool_name="retreat_arm",
                args={"arm": "left", "axis": "x", "direction": "negative", "distance": 0.02},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        task_env=env,
        latest_snapshot=snapshot_for(env),
    )

    assert len(env.actions) == 1
    assert len(results) == 3
    assert results[0].success is True
    assert results[0].details["executed_steps"] == 1
    assert results[0].details["environment_success"] is True
    assert results[0].details["target_reached"] is False
    for result in results[1:]:
        assert result.success is False
        assert result.details["terminal_skip"] is True
        assert result.details["skip_reason"] == "environment_eval_success"
        assert result.details["authority"] == "environment_eval_success"
    assert dispatcher.latest_snapshot is not None
    assert dispatcher.latest_snapshot.eval_success is True


def test_dispatch_batch_skips_every_call_when_environment_is_already_successful() -> None:
    env = FakeRMBenchEnv()
    dispatcher = RecoveryToolDispatcher()
    snapshot = snapshot_for(env)
    assert snapshot.eval_success is False
    env.eval_success = True

    results = dispatcher.dispatch_batch(
        [
            RecoveryToolCall(
                tool_name="retreat_arm",
                args={"arm": "left", "axis": "x", "direction": "negative", "distance": 0.02},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        task_env=env,
        latest_snapshot=snapshot,
    )

    assert env.actions == []
    assert len(results) == 2
    assert all(result.details.get("terminal_skip") is True for result in results)
    assert dispatcher.latest_snapshot is snapshot
    assert dispatcher.latest_environment_success is True


def test_dispatch_batch_does_not_treat_check_success_as_terminal_authority() -> None:
    env = CheckSuccessOnlyEnv()
    dispatcher = RecoveryToolDispatcher()

    results = dispatcher.dispatch_batch(
        [
            RecoveryToolCall(
                tool_name="retreat_arm",
                args={"arm": "left", "axis": "x", "direction": "positive", "distance": 0.02},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        task_env=env,
        latest_snapshot=snapshot_for(env),
    )

    assert len(env.actions) == 1
    assert [result.success for result in results] == [True, True]
    assert dispatcher.latest_snapshot is not None
    assert dispatcher.latest_snapshot.eval_success is False
    assert dispatcher.latest_snapshot.check_success is True


def test_physical_recovery_tools_reject_missing_arm_without_environment_action() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)
    snapshot = snapshot_for(env)

    results = [
        adapter.contact_displace(snapshot, {"axis": "z", "direction": "negative", "distance": 0.01}),
        adapter.retreat_arm(snapshot, {"axis": "x", "direction": "negative", "distance": 0.01}),
        adapter.lift_ee(snapshot, {"distance": 0.01}),
    ]

    assert all(result.result.success is False for result in results)
    assert all("arm is required" in result.result.message for result in results)
    assert env.actions == []


def test_explicit_both_arm_remains_available_for_symmetric_clearance() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)

    result = adapter.retreat_arm(
        snapshot_for(env),
        {"arm": "both", "axis": "x", "direction": "negative", "distance": 0.01},
    )

    assert result.result.success is True
    assert result.result.details["arm"] == "both"
    assert len(env.actions) == 1


def test_contact_displace_rejects_observed_pose_outside_grounded_contact_tolerance() -> None:
    env = FakeRMBenchEnv()
    adapter = RMBenchRecoveryAdapter(env)

    result = adapter.contact_displace(
        snapshot_for(env),
        {
            "arm": "left",
            "axis": "z",
            "direction": "negative",
            "distance": 0.01,
            "_contact_reference_world_m": [0.0, 0.03, 0.0],
            "_contact_reference_tolerance_m": 0.01,
        },
    )

    assert result.result.success is False
    assert "outside contact tolerance" in result.result.message
    assert np.isclose(result.result.details["target_observation_error_m"], 0.03)
    assert env.actions == []

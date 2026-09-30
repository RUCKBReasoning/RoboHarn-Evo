from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from policy.roboharn_evo.agent.runtime import build_agent_runtime


def test_build_agent_runtime_parses_debug_recovery_config() -> None:
    with mock.patch("policy.roboharn_evo.agent.runtime.build_planner_backend") as build_planner, mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend"
    ) as build_executor, mock.patch("policy.roboharn_evo.agent.runtime.AgentSession") as agent_session:
        build_planner.return_value = object()
        build_executor.return_value = object()
        agent_session.build.return_value = object()

        runtime = build_agent_runtime(
            {
                "agent": {
                    "enabled": True,
                    "observation_preprocess": {
                        "oracle_objects": {
                            "enabled": True,
                            "include_all": False,
                            "max_objects": 9,
                        },
                        "scene_memory": {
                            "temporal_action_geometry_distance_m": 0.025,
                            "min_candidate_score": 0.2,
                            "max_object_z_extent_m": 0.16,
                            "max_object_xy_extent_m": 0.28,
                            "max_object_world_z_m": 0.91,
                            "robot_self_filter_radius_m": 0.07,
                            "robot_self_filter_z_margin_m": 0.06,
                        }
                    },
                    "debug_recovery": {
                        "enabled": True,
                        "trigger_step": 3,
                        "signal": "object-not-visible",
                        "reason": "force object visibility recovery",
                        "subtask": "cover the left block",
                        "skill": "monitored-subtask-execution",
                        "wait_for_scene_memory": False,
                        "max_wait_steps": 5,
                        "retry_budget": 2,
                        "pure_control": True,
                        "max_rounds": 7,
                        "repeat_signal": True,
                        "bootstrap_with_planner": True,
                        "skip_vla_rollout": True,
                    },
                    "pure_tool_control": {
                        "enabled": True,
                        "trigger_step": 4,
                        "signal": "task-level-recovery-control",
                        "reason": "paper pure tool baseline",
                        "wait_for_scene_memory": True,
                        "max_wait_steps": 6,
                        "retry_budget": 3,
                        "max_rounds": 8,
                        "max_control_turns": 9,
                        "max_no_progress_control_turns": 6,
                        "backend_error_budget": 7,
                        "empty_plan_replan_threshold": 3,
                        "repeat_signal": False,
                        "bootstrap_with_planner": True,
                        "skip_vla_rollout": True,
                    },
                },
                "planner": {"backend": "agent_api"},
                "executor": {"backend": "pi05_api"},
            }
        )

    assert runtime.config.debug_recovery_enabled is True
    assert runtime.config.debug_recovery_trigger_step == 3
    assert runtime.config.debug_recovery_signal == "object-not-visible"
    assert runtime.config.debug_recovery_reason == "force object visibility recovery"
    assert runtime.config.debug_recovery_subtask == "cover the left block"
    assert runtime.config.debug_recovery_skill == "monitored-subtask-execution"
    assert runtime.config.debug_recovery_wait_for_scene_memory is False
    assert runtime.config.debug_recovery_max_wait_steps == 5
    assert runtime.config.debug_recovery_retry_budget == 2
    assert runtime.config.debug_recovery_pure_control is True
    assert runtime.config.debug_recovery_max_rounds == 7
    assert runtime.config.debug_recovery_repeat_signal is True
    assert runtime.config.debug_recovery_bootstrap_with_planner is True
    assert runtime.config.debug_recovery_skip_vla_rollout is True
    assert runtime.config.recovery_reobserve_scene_enabled is True
    assert runtime.config.grasp_transport_policy == "strict"
    assert runtime.config.release_guard_enabled is False
    assert runtime.config.planner_context_mode == "legacy"
    assert runtime.config.observation_trace_payload_mode == "compact_v1"
    assert runtime.config.raw_trace_sidecar_enabled is False
    assert (
        runtime.config.action_geometry_repair_pending_policy
        == "strict"
    )
    assert (
        runtime.config.enable_executable_candidate_temporal_consistency
        is True
    )
    assert runtime.config.allow_memory_valid_final_grounded_action is False
    assert (
        runtime.config.enable_automatic_self_occlusion_visual_clearance
        is True
    )
    assert runtime.config.scene_memory_min_candidate_score == 0.2
    assert runtime.config.scene_memory_temporal_action_geometry_distance_m == 0.025
    assert runtime.config.scene_memory_max_object_z_extent_m == 0.16
    assert runtime.config.scene_memory_max_object_xy_extent_m == 0.28
    assert runtime.config.scene_memory_max_object_world_z_m == 0.91
    assert runtime.config.scene_memory_robot_self_filter_radius_m == 0.07
    assert runtime.config.scene_memory_robot_self_filter_z_margin_m == 0.06
    assert runtime.config.oracle_objects_enabled is True
    assert runtime.config.oracle_objects_include_all is False
    assert runtime.config.oracle_objects_max_objects == 9
    assert runtime.config.pure_tool_control_enabled is True
    assert runtime.config.pure_tool_control_trigger_step == 4
    assert runtime.config.pure_tool_control_signal == "task-level-recovery-control"
    assert runtime.config.pure_tool_control_reason == "paper pure tool baseline"
    assert runtime.config.pure_tool_control_wait_for_scene_memory is True
    assert runtime.config.pure_tool_control_max_wait_steps == 6
    assert runtime.config.pure_tool_control_retry_budget == 3
    assert runtime.config.pure_tool_control_max_rounds == 8
    assert runtime.config.pure_tool_control_max_control_turns == 9
    assert runtime.config.pure_tool_control_max_no_progress_control_turns == 6
    assert runtime.config.pure_tool_control_backend_error_budget == 7
    assert runtime.config.pure_tool_control_empty_plan_replan_threshold == 3
    assert runtime.config.pure_tool_control_repeat_signal is False
    assert runtime.config.pure_tool_control_bootstrap_with_planner is True
    assert runtime.config.pure_tool_control_skip_vla_rollout is True


def test_deploy_policy_defaults_to_multi_camera_observation_preprocess() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "policy" / "roboharn_evo" / "deploy_policy.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    with mock.patch("policy.roboharn_evo.agent.runtime.build_planner_backend") as build_planner, mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend"
    ) as build_executor, mock.patch("policy.roboharn_evo.agent.runtime.AgentSession") as agent_session:
        build_planner.return_value = object()
        build_executor.return_value = object()
        agent_session.build.return_value = object()

        runtime = build_agent_runtime(config)

    assert runtime.config.observation_preprocess_camera == "head"
    assert runtime.config.observation_preprocess_cameras == ("head", "left", "right")
    assert runtime.config.observation_verification_cameras == ("third",)
    assert runtime.config.observation_grounding_camera == "head"
    assert runtime.config.observation_grounding_cameras == ("head", "left", "right", "third")
    assert runtime.config.scene_memory_min_candidate_score == 0.12
    assert runtime.config.scene_memory_temporal_action_geometry_distance_m == 0.04
    assert runtime.config.scene_memory_max_object_z_extent_m == 0.18
    assert runtime.config.scene_memory_max_object_xy_extent_m == 0.30
    assert runtime.config.scene_memory_max_object_world_z_m == 0.95
    assert runtime.config.scene_memory_robot_self_filter_radius_m == 0.08
    assert runtime.config.scene_memory_robot_self_filter_z_margin_m == 0.08
    assert runtime.config.oracle_objects_enabled is False
    assert runtime.config.pure_tool_control_max_rounds == 10
    assert runtime.config.pure_tool_control_max_control_turns == 64
    assert runtime.config.pure_tool_control_max_no_progress_control_turns == 10
    assert runtime.config.pure_tool_control_backend_error_budget == 5
    assert runtime.config.pure_tool_control_empty_plan_replan_threshold == 2
    assert runtime.config.recovery_reobserve_scene_enabled is True
    assert runtime.config.grasp_transport_policy == "evidence_only"
    assert runtime.config.release_guard_enabled is False
    assert runtime.config.planner_context_mode == "compact_v1"
    assert runtime.config.observation_trace_payload_mode == "compact_v1"
    assert runtime.config.raw_trace_sidecar_enabled is False
    assert (
        runtime.config.action_geometry_repair_pending_policy
        == "safe_motion"
    )
    assert (
        runtime.config.enable_executable_candidate_temporal_consistency
        is False
    )
    assert runtime.config.allow_memory_valid_final_grounded_action is True
    assert (
        runtime.config.enable_automatic_self_occlusion_visual_clearance
        is False
    )


@pytest.mark.parametrize("trace_mode", ["legacy", "raw"])
def test_raw_sidecar_requires_compact_standard_trace(trace_mode: str) -> None:
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ):
        with pytest.raises(
            ValueError,
            match="raw_trace_sidecar_enabled requires",
        ):
            build_agent_runtime(
                {
                    "agent": {
                        "rollout_dump": {
                            "trace_payload_mode": trace_mode,
                            "raw_trace_sidecar_enabled": True,
                        }
                    },
                    "planner": {"backend": "agent_api"},
                    "executor": {"backend": "pi05_api"},
                }
            )


def test_build_agent_runtime_enables_one_copy_raw_sidecar() -> None:
    session = SimpleNamespace(
        agent=SimpleNamespace(_recovery_dispatcher=mock.Mock())
    )
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.AgentSession.build",
        return_value=session,
    ):
        runtime = build_agent_runtime(
            {
                "agent": {
                    "rollout_dump": {
                        "trace_payload_mode": "compact_v1",
                        "raw_trace_sidecar_enabled": True,
                    }
                },
                "planner": {"backend": "agent_api"},
                "executor": {"backend": "pi05_api"},
            }
        )

    assert runtime.config.observation_trace_payload_mode == "compact_v1"
    assert runtime.config.raw_trace_sidecar_enabled is True
    assert runtime.config.gpt_scene_memory_sidecar_enabled is True


def test_build_agent_runtime_can_disable_gpt_scene_memory_sidecar() -> None:
    session = SimpleNamespace(
        agent=SimpleNamespace(_recovery_dispatcher=mock.Mock())
    )
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.AgentSession.build",
        return_value=session,
    ):
        runtime = build_agent_runtime(
            {
                "agent": {
                    "rollout_dump": {
                        "gpt_scene_memory_sidecar_enabled": False,
                    }
                },
                "planner": {"backend": "agent_api"},
                "executor": {"backend": "pi05_api"},
            }
        )

    assert runtime.config.gpt_scene_memory_sidecar_enabled is False


def test_build_agent_runtime_rejects_unknown_planner_context_mode() -> None:
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ):
        with pytest.raises(
            ValueError,
            match="agent.recovery.planner_context_mode",
        ):
            build_agent_runtime(
                {
                    "agent": {
                        "recovery": {
                            "planner_context_mode": "unknown",
                        }
                    },
                    "planner": {"backend": "agent_api"},
                    "executor": {"backend": "pi05_api"},
                }
            )


def test_build_agent_runtime_disables_reobserve_scene_ablation() -> None:
    dispatcher = mock.Mock()
    session = SimpleNamespace(
        agent=SimpleNamespace(_recovery_dispatcher=dispatcher)
    )
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.AgentSession.build",
        return_value=session,
    ):
        runtime = build_agent_runtime(
            {
                "agent": {
                    "enabled": True,
                    "recovery": {
                        "enable_reobserve": False,
                    },
                },
                "planner": {"backend": "agent_api"},
                "executor": {"backend": "pi05_api"},
            }
        )

    assert runtime.config.recovery_reobserve_scene_enabled is False
    dispatcher.set_reobserve_scene_enabled.assert_called_once_with(
        False
    )


def test_build_agent_runtime_can_restore_legacy_release_guard() -> None:
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.AgentSession.build",
        return_value=SimpleNamespace(agent=SimpleNamespace()),
    ):
        runtime = build_agent_runtime(
            {
                "agent": {
                    "enabled": True,
                    "recovery": {"enable_release_guard": True},
                },
                "planner": {"backend": "agent_api"},
                "executor": {"backend": "pi05_api"},
            }
        )

    assert runtime.config.release_guard_enabled is True


def test_build_agent_runtime_rejects_unknown_grasp_transport_policy() -> None:
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ):
        with pytest.raises(
            ValueError,
            match="agent.recovery.grasp_transport_policy",
        ):
            build_agent_runtime(
                {
                    "agent": {
                        "recovery": {
                            "grasp_transport_policy": "optimistic_typo",
                        }
                    },
                    "planner": {"backend": "agent_api"},
                    "executor": {"backend": "pi05_api"},
                }
            )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("safe-motion", "safe_motion"),
        ("disabled", "disabled"),
    ],
)
def test_build_agent_runtime_parses_action_geometry_repair_pending_policy(
    configured: str,
    expected: str,
) -> None:
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.AgentSession.build",
        return_value=SimpleNamespace(agent=SimpleNamespace()),
    ):
        runtime = build_agent_runtime(
            {
                "agent": {
                    "recovery": {
                        "action_geometry_repair_pending_policy": (
                            configured
                        ),
                    }
                },
                "planner": {"backend": "agent_api"},
                "executor": {"backend": "pi05_api"},
            }
        )

    assert (
        runtime.config.action_geometry_repair_pending_policy
        == expected
    )


def test_build_agent_runtime_rejects_unknown_action_geometry_repair_pending_policy() -> None:
    with mock.patch(
        "policy.roboharn_evo.agent.runtime.build_planner_backend",
        return_value=object(),
    ), mock.patch(
        "policy.roboharn_evo.agent.runtime.build_executor_backend",
        return_value=object(),
    ):
        with pytest.raises(
            ValueError,
            match=(
                "agent.recovery."
                "action_geometry_repair_pending_policy"
            ),
        ):
            build_agent_runtime(
                {
                    "agent": {
                        "recovery": {
                            "action_geometry_repair_pending_policy": (
                                "unsafe_typo"
                            ),
                        }
                    },
                    "planner": {"backend": "agent_api"},
                    "executor": {"backend": "pi05_api"},
                }
            )

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from unittest import mock

import numpy as np
import pytest

from policy.roboharn_evo.agent.core.img_agent import (
    _RECOVERY_EMPTY_PLAN,
    _RECOVERY_INTERNAL_PROGRESS,
    ImgAgent,
)
from policy.roboharn_evo.agent.decisions import ControlSignal
from policy.roboharn_evo.agent.environment import EnvSnapshot
from policy.roboharn_evo.agent.monitoring.signals import TASK_LEVEL_RECOVERY_CONTROL
from policy.roboharn_evo.agent.recovery.action_effect_verifier import ActionEffectVerifierBackendError
from policy.roboharn_evo.agent.recovery.recovery_policies import RecoveryRoute
from policy.roboharn_evo.agent.recovery.recovery_primitives import RecoveryPrimitivePlan
from policy.roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall, RecoveryToolResult


@dataclass(frozen=True)
class DummyConfig:
    initial_memory_text: str = "The task has started."
    observation_preprocess_enabled: bool = False
    observation_preprocess_every_n_steps: int = 1
    observation_preprocess_auto_objects: bool = True
    observation_preprocess_query_url: str = ""
    observation_preprocess_normalization_url: str = ""
    observation_preprocess_max_objects: int = 3
    oracle_objects_enabled: bool = False
    scene_memory_enabled: bool = True
    scene_memory_max_missing_steps: int = 20
    scene_memory_stable_distance_m: float = 0.08
    scene_memory_temporal_window_size: int = 8
    scene_memory_min_candidate_score: float = 0.12
    scene_memory_max_object_z_extent_m: float = 0.18
    scene_memory_max_object_xy_extent_m: float = 0.30
    scene_memory_max_object_world_z_m: float = 0.95
    scene_memory_robot_self_filter_radius_m: float = 0.08
    scene_memory_robot_self_filter_z_margin_m: float = 0.08
    debug_recovery_enabled: bool = True
    debug_recovery_trigger_step: int = 0
    debug_recovery_signal: str = "motion_blocked"
    debug_recovery_reason: str = "debug forced recovery loop"
    debug_recovery_subtask: str = "cover the left block"
    debug_recovery_skill: str = "monitored-subtask-execution"
    debug_recovery_wait_for_scene_memory: bool = False
    debug_recovery_max_wait_steps: int = 0
    debug_recovery_retry_budget: int = 1
    debug_recovery_pure_control: bool = True
    debug_recovery_max_rounds: int = 3
    debug_recovery_repeat_signal: bool = False
    debug_recovery_bootstrap_with_planner: bool = False
    debug_recovery_skip_vla_rollout: bool = False
    pure_tool_control_enabled: bool = False
    pure_tool_control_trigger_step: int = 0
    pure_tool_control_signal: str = TASK_LEVEL_RECOVERY_CONTROL
    pure_tool_control_reason: str = "pure tool-control ablation"
    pure_tool_control_wait_for_scene_memory: bool = False
    pure_tool_control_max_wait_steps: int = 0
    pure_tool_control_retry_budget: int = 1
    pure_tool_control_max_rounds: int = 3
    pure_tool_control_max_control_turns: int = 0
    pure_tool_control_max_no_progress_control_turns: int = 4
    pure_tool_control_backend_error_budget: int = 5
    pure_tool_control_empty_plan_replan_threshold: int = 2
    pure_tool_control_repeat_signal: bool = True
    pure_tool_control_bootstrap_with_planner: bool = True
    pure_tool_control_skip_vla_rollout: bool = True
    grasp_transport_policy: str = "strict"
    release_guard_enabled: bool = True
    planner_context_mode: str = "legacy"
    observation_trace_payload_mode: str = "legacy"
    action_geometry_repair_pending_policy: str = "strict"
    enable_executable_candidate_temporal_consistency: bool = True
    allow_memory_valid_final_grounded_action: bool = False
    enable_automatic_self_occlusion_visual_clearance: bool = True
    decision_interval: int = 1
    max_retries_per_skill: int = 2
    max_steps_per_skill: int = 80
    stall_patience: int = 12
    mode: str = "deployment"
    interrupt_on_skill_change: bool = True


class FakeControlRuntime:
    def __init__(self, prediction: dict) -> None:
        self.prediction = dict(prediction)
        self.calls = []

    def reset(self) -> None:
        return

    def predict_planner_step(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.prediction)


class FlakyControlRuntime(FakeControlRuntime):
    def __init__(self, prediction: dict) -> None:
        super().__init__(prediction)
        self.failures_remaining = 1

    def predict_planner_step(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("HTTP 502 from planner")
        return dict(self.prediction)


def make_card(config: DummyConfig, *, control_runtime=None, executor_runtime=None) -> object:
    class Card:
        control_model_name = "test"
        executor_name = "test"
        ood_backend_config = None
        recovery_backend_config = None

    card = Card()
    card.config = config
    card.control_runtime = control_runtime
    card.executor_runtime = executor_runtime
    return card


def make_snapshot() -> EnvSnapshot:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    joint_vector = np.zeros(14, dtype=np.float32)
    endpose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    return EnvSnapshot(
        raw={"joint_action": {"vector": [0.0] * 14}, "endpose": {}},
        head_rgb=image,
        left_rgb=image,
        right_rgb=image,
        joint_vector=joint_vector,
        left_endpose=endpose,
        right_endpose=endpose,
        step_count=0,
        step_limit=100,
        eval_success=False,
        check_success=False,
        max_reward=0.0,
        instruction="cover the block",
    )


def make_snapshot_with_gripper(value: float) -> EnvSnapshot:
    snapshot = make_snapshot()
    snapshot.raw.setdefault("endpose", {})
    snapshot.raw["endpose"]["left_gripper"] = value
    snapshot.raw["endpose"]["gripper"] = value
    return snapshot


def make_snapshot_with_success(*, eval_success: bool = False, check_success: bool = False) -> EnvSnapshot:
    snapshot = make_snapshot()
    return EnvSnapshot(
        raw=snapshot.raw,
        head_rgb=snapshot.head_rgb,
        left_rgb=snapshot.left_rgb,
        right_rgb=snapshot.right_rgb,
        joint_vector=snapshot.joint_vector,
        left_endpose=snapshot.left_endpose,
        right_endpose=snapshot.right_endpose,
        step_count=snapshot.step_count,
        step_limit=snapshot.step_limit,
        eval_success=eval_success,
        check_success=check_success,
        max_reward=1.0 if eval_success or check_success else 0.0,
        instruction=snapshot.instruction,
    )


def test_current_status_exposes_condition_and_budgets_for_episode_provenance() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                oracle_objects_enabled=True,
                pure_tool_control_enabled=True,
                pure_tool_control_max_rounds=10,
                pure_tool_control_max_control_turns=8,
            )
        )
    )
    agent._debug_recovery_rounds = 2

    status = agent.current_status()

    assert status["perception_condition"] == "oracle"
    assert status["oracle_objects_enabled"] is True
    assert status["semantic_round_index"] == 2
    assert status["max_semantic_rounds_per_active_subtask"] == 10
    assert status["max_control_turns"] == 8
    assert status["release_guard_enabled"] is True


def test_debug_recovery_pure_control_continues_without_pending_retry() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    with mock.patch.object(agent, "_dispatch_recovery_tools", return_value="retry") as dispatch:
        handled = agent._maybe_force_debug_recovery(task_env=object())

    assert handled is True
    assert dispatch.call_count == 1
    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.recovery.pending_action == ""
    assert agent.memory_store.state.monitor.phase == "monitoring"
    assert agent.memory_store.state.monitor.env_signal == "motion_blocked"
    assert "debug_pure_control_round:1" in agent.memory_store.state.working.recovery_history


def test_debug_recovery_pure_control_switches_to_task_level_signal_after_first_round() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    signals: list[str] = []

    def fake_dispatch(_task_env, signal):
        signals.append(signal.name)
        return "retry"

    with mock.patch.object(agent, "_dispatch_recovery_tools", side_effect=fake_dispatch):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert signals == ["motion_blocked", TASK_LEVEL_RECOVERY_CONTROL]


def test_debug_recovery_pure_control_can_repeat_forced_signal_when_requested() -> None:
    agent = ImgAgent(make_card(DummyConfig(debug_recovery_repeat_signal=True)))
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    signals: list[str] = []

    def fake_dispatch(_task_env, signal):
        signals.append(signal.name)
        return "retry"

    with mock.patch.object(agent, "_dispatch_recovery_tools", side_effect=fake_dispatch):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert signals == ["motion_blocked", "motion_blocked"]


def test_debug_recovery_pure_control_waits_for_scene_memory_without_falling_through() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_wait_for_scene_memory=True,
                debug_recovery_max_wait_steps=2,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    with mock.patch.object(agent, "_dispatch_recovery_tools", return_value="retry") as dispatch:
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert dispatch.call_count == 0
        assert agent.memory_store.state.active_skill is None

        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert dispatch.call_count == 1
    assert agent.memory_store.state.active_skill is not None
    assert agent.memory_store.state.task.task_finished is False
    assert "debug_pure_control_round:1" in agent.memory_store.state.working.recovery_history


def test_pure_tool_control_retries_identity_binding_with_fresh_preprocess_before_failure() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_wait_for_scene_memory=True,
                pure_tool_control_max_wait_steps=2,
                observation_preprocess_enabled=True,
                observation_preprocess_query_url=(
                    "http://127.0.0.1:9104/perception_queries"
                ),
            )
        )
    )
    agent.current_instruction = "complete the manipulation"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "track_0001",
                    "world_m": [0.0, 0.0, 0.8],
                }
            ],
            "task_focus": {
                "target_instances": [],
                "tool_instances": [],
                "identity_binding_required": True,
                "identity_binding_errors": [
                    "agent_returned_no_target_or_tool_binding"
                ],
            },
        }
    )
    # Generic scene waiting is a different budget; it must not consume a
    # fresh perception-query retry.
    agent._debug_recovery_scene_wait_turns = 1

    with mock.patch.object(
        agent,
        "update_snapshot",
    ) as update_snapshot:
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent._pure_tool_control_terminal_failure == ""
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert update_snapshot.call_count == 2
    for call in update_snapshot.call_args_list:
        assert call.kwargs["force_preprocess"] is True
        retry = call.kwargs["perception_binding_requirement"]
        assert retry["force_refresh"] is True
        assert retry["failure_reason"] == (
            "agent_returned_no_target_or_tool_binding"
        )
    assert (
        agent._pure_tool_control_terminal_failure
        == "pure_tool_control_identity_binding_unresolved:"
        "agent_returned_no_target_or_tool_binding"
    )


def test_pure_tool_control_continues_when_fresh_identity_query_recovers() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_wait_for_scene_memory=True,
                pure_tool_control_max_wait_steps=2,
                observation_preprocess_enabled=True,
                observation_preprocess_query_url=(
                    "http://127.0.0.1:9104/perception_queries"
                ),
            )
        )
    )
    agent.current_instruction = "complete the manipulation"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec(
        "monitored-subtask-execution"
    )
    agent.memory_store.start_or_replace_skill(
        subtask_text="perform the next grounded manipulation",
        skill_spec=skill_spec,
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "track_0001",
                    "world_m": [0.0, 0.0, 0.8],
                }
            ],
            "task_focus": {
                "target_instances": [],
                "tool_instances": [],
                "identity_binding_required": True,
                "identity_binding_errors": [
                    "agent_returned_no_target_or_tool_binding"
                ],
            },
        }
    )

    def recover_binding(*_args, **_kwargs):
        agent.memory_store.record_scene_memory(
            {
                "instances": [
                    {
                        "instance_id": "track_0001",
                        "world_m": [0.0, 0.0, 0.8],
                    }
                ],
                "task_focus": {
                    "target_instances": ["track_0001"],
                    "tool_instances": [],
                    "identity_binding_required": True,
                    "identity_binding_roles": ["target"],
                    "identity_binding_errors": [],
                },
            }
        )

    with (
        mock.patch.object(
            agent,
            "update_snapshot",
            side_effect=recover_binding,
        ) as update_snapshot,
        mock.patch.object(
            agent,
            "_dispatch_recovery_tools",
            return_value="retry",
        ) as dispatch,
    ):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert update_snapshot.call_count == 1
    assert dispatch.call_count == 1
    assert agent._pure_tool_control_terminal_failure == ""
    assert agent._debug_recovery_scene_wait_turns == 0
    assert agent._identity_binding_retry_attempts == 0


def test_pure_tool_control_bootstraps_subtask_from_planner_without_config_subtask() -> None:
    planner_subtask = "planner selected bounded manipulation step"
    control = FakeControlRuntime(
        {
            "action_mode": "start",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": planner_subtask,
            "memory_text": "planner committed memory",
            "commit_label": "state_change",
        }
    )
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                debug_recovery_subtask="",
                pure_tool_control_enabled=True,
            ),
            control_runtime=control,
        )
    )
    agent.current_instruction = "complete the task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="complete the task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    with (
        mock.patch.object(agent, "_trace") as trace,
        mock.patch.object(agent, "_dump_rollout_event") as dump_rollout_event,
    ):
        assert agent._maybe_bootstrap_debug_recovery_with_planner() is True
    expected_payload_bytes = len(control.calls[0]["task"].encode("utf-8"))
    trace_start = next(
        call
        for call in trace.call_args_list
        if call.args and call.args[0] == "control_turn_start"
    )
    dump_start = next(
        call
        for call in dump_rollout_event.call_args_list
        if call.args and call.args[0] == "control_turn_start"
    )
    assert trace_start.kwargs["control_payload_bytes"] == expected_payload_bytes
    assert dump_start.kwargs["control_payload_bytes"] == expected_payload_bytes
    assert control.calls
    assert agent.memory_store.state.active_skill is not None
    assert agent.memory_store.state.active_skill.instruction == planner_subtask

    signals: list[str] = []

    def fake_dispatch(_task_env, signal):
        signals.append(signal.name)
        return "retry"

    with mock.patch.object(agent, "_dispatch_recovery_tools", side_effect=fake_dispatch):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert signals == [TASK_LEVEL_RECOVERY_CONTROL]
    assert "pure_tool_control_round:1" in agent.memory_store.state.working.recovery_history


def test_control_decision_stores_structured_preferred_arm_on_active_skill() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    agent.apply_control_decision(
        {
            "action_mode": "start",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": "perform one grounded manipulation step",
            "preferred_arm": "right",
        }
    )

    active = agent.memory_store.state.active_skill
    assert active is not None
    assert active.preferred_arm == "right"
    assert active.to_dict()["preferred_arm"] == "right"


def test_pure_tool_control_defers_recovery_until_cycle_after_planner_bootstrap() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "complete the task"
    agent.memory_store.reset(
        task="complete the task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="new planner subtask", skill_spec=skill_spec)

    with (
        mock.patch(
            "policy.roboharn_evo.agent.core.img_agent.RMBenchEnvAdapter.from_env",
            return_value=make_snapshot(),
        ),
        mock.patch.object(agent, "update_snapshot"),
        mock.patch.object(agent, "_repair_unvalidated_pure_tool_control_finish"),
        mock.patch.object(agent, "_maybe_bootstrap_debug_recovery_with_planner", return_value=True),
        mock.patch.object(agent, "_maybe_force_debug_recovery") as force_recovery,
    ):
        agent.run_step(task_env=object(), observation={})

    force_recovery.assert_not_called()


def test_pure_tool_control_retries_control_planner_error_without_ending_episode() -> None:
    planner_subtask = "planner selected bounded manipulation step"
    control = FlakyControlRuntime(
        {
            "action_mode": "start",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": planner_subtask,
            "memory_text": "planner committed memory",
            "commit_label": "state_change",
        }
    )
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                debug_recovery_subtask="",
                pure_tool_control_enabled=True,
            ),
            control_runtime=control,
        )
    )
    agent.current_instruction = "complete the task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="complete the task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    assert agent._maybe_bootstrap_debug_recovery_with_planner() is True
    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is None
    assert agent.memory_store.state.monitor.phase == "reasoning"
    assert agent.memory_store.state.monitor.status == "needs_reasoning"
    assert agent._debug_recovery_planner_bootstrapped is False
    assert agent.current_status()["control_turn_count"] == 0
    assert agent.current_status()["control_backend_error_count"] == 1
    assert any(
        entry.startswith("pure_tool_control_control_turn_error:planner_bootstrap:")
        for entry in agent.memory_store.state.working.recovery_history
    )

    assert agent._maybe_bootstrap_debug_recovery_with_planner() is True
    assert len(control.calls) == 2
    assert agent.memory_store.state.active_skill is not None
    assert agent.memory_store.state.active_skill.instruction == planner_subtask
    assert agent.memory_store.state.task.task_finished is False
    assert agent.current_status()["control_turn_count"] == 1
    assert agent.current_status()["control_backend_error_count"] == 0


def test_pure_tool_control_does_not_fallback_to_config_or_global_subtask_without_planner_skill() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                debug_recovery_subtask="",
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "complete the task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="complete the task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    with mock.patch.object(agent, "_dispatch_recovery_tools", return_value="retry") as dispatch:
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert dispatch.call_count == 0
    assert agent.memory_store.state.active_skill is None
    assert "forced_recovery_no_active_skill" in agent.memory_store.state.working.recovery_history


def test_pure_tool_control_never_uses_configured_debug_recovery_subtask() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                debug_recovery_subtask="fixed runtime subtask must not execute",
                pure_tool_control_enabled=True,
                pure_tool_control_bootstrap_with_planner=False,
            )
        )
    )
    agent.current_instruction = "complete the task from current evidence"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    assert agent._ensure_debug_recovery_active_skill() is False
    assert agent.memory_store.state.active_skill is None


def test_planner_bootstrap_apply_error_is_not_counted_as_backend_failure() -> None:
    control = FakeControlRuntime(
        {
            "action_mode": "start",
            "selected_skill": "unknown-runtime-skill",
            "subtask_text": "agent selected subtask",
            "memory_text": "planner returned a syntactically valid response",
            "commit_label": "state_change",
        }
    )
    agent = ImgAgent(
        make_card(
            DummyConfig(debug_recovery_enabled=False, pure_tool_control_enabled=True),
            control_runtime=control,
        )
    )
    agent.current_instruction = "complete the task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    try:
        agent._maybe_bootstrap_debug_recovery_with_planner()
    except RuntimeError as exc:
        assert "No workflow-defined skill matches" in str(exc)
    else:
        raise AssertionError("local apply_control_decision error must propagate")

    assert agent.current_status()["control_turn_count"] == 1
    assert agent.current_status()["control_backend_error_count"] == 0
    assert agent._debug_recovery_planner_bootstrapped is False


def test_main_control_apply_error_is_not_counted_as_backend_failure() -> None:
    control = FakeControlRuntime(
        {
            "action_mode": "switch",
            "selected_skill": "unknown-runtime-skill",
            "subtask_text": "agent selected replacement subtask",
            "memory_text": "planner returned a syntactically valid response",
            "commit_label": "state_change",
        }
    )
    agent = ImgAgent(
        make_card(
            DummyConfig(debug_recovery_enabled=False, pure_tool_control_enabled=True),
            control_runtime=control,
        )
    )
    agent.current_instruction = "complete the task"
    snapshot = make_snapshot()
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="current active subtask",
        skill_spec=skill_spec,
    )

    with (
        mock.patch(
            "policy.roboharn_evo.agent.core.img_agent.RMBenchEnvAdapter.from_env",
            return_value=snapshot,
        ),
        mock.patch.object(agent, "_maybe_force_debug_recovery", return_value=False),
        mock.patch.object(agent, "needs_control_turn", return_value=(True, ControlSignal.INTERVAL)),
    ):
        try:
            agent.run_step(task_env=object(), observation={})
        except RuntimeError as exc:
            assert "No workflow-defined skill matches" in str(exc)
        else:
            raise AssertionError("local apply_control_decision error must propagate")

    assert agent.current_status()["control_turn_count"] == 1
    assert agent.current_status()["control_backend_error_count"] == 0


def test_forced_recovery_skip_vla_rollout_blocks_executor_path() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                debug_recovery_subtask="",
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "complete the task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="complete the task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)

    assert agent._skip_vla_rollout_for_forced_recovery() is True
    assert agent.memory_store.state.monitor.status == "rollout_active"
    assert agent.memory_store.state.monitor.env_signal == "running"


def test_recovery_guard_opens_closed_gripper_for_explicit_open_precondition() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.scene_memory = {
        "instances": [{"instance_id": "tool_01", "approach_world_m": [0.1, 0.0, 0.0]}],
        "task_focus": {"tool_instances": ["tool_01"]},
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "left", "role": "tool", "point_key": "approach_world_m", "gripper_precondition": "open"},
            )
        ]
    )

    assert [call.tool_name for call in calls] == ["open_gripper", "move_ee_to_grounded_instance"]
    assert calls[0].args["arm"] == "left"
    assert calls[0].args["_guard_reason"] == "planner requested gripper_precondition=open"


def test_recovery_guard_does_not_infer_open_gripper_for_grounded_tool_approach() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "left", "role": "tool", "point_key": "approach_world_m"},
            )
        ]
    )

    assert [call.tool_name for call in calls] == ["move_ee_to_grounded_instance"]


def test_recovery_guard_closes_open_gripper_for_explicit_closed_precondition() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(1.0)
    agent.memory_store.reset(
        task="make deliberate contact",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="contact_displace",
                args={
                    "arm": "left",
                    "axis": "z",
                    "direction": "negative",
                    "distance": 0.02,
                    "gripper_precondition": "closed",
                },
            )
        ]
    )

    assert [call.tool_name for call in calls] == ["close_gripper", "contact_displace"]
    assert calls[0].args["arm"] == "left"
    assert calls[0].args["_guard_reason"] == "planner requested gripper_precondition=closed"


def test_recovery_guard_does_not_duplicate_satisfied_closed_precondition() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="make deliberate contact",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "gripper_precondition": "closed"},
            )
        ]
    )

    assert [call.tool_name for call in calls] == ["contact_displace"]


def test_recovery_guard_does_not_infer_left_arm_when_arm_is_missing() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(1.0)
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"axis": "z", "direction": "negative", "gripper_precondition": "closed"},
            )
        ]
    )

    assert [call.tool_name for call in calls] == ["contact_displace"]
    assert "arm" not in calls[0].args


def test_recovery_guard_stages_active_contact_from_displacement_matched_offset() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="make grounded contact",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.scene_memory = {
        "instances": [
            {
                "instance_id": "target_01",
                "approach_world_m": [0.1, 0.0, 0.2],
                "contact_world_m": [0.1, 0.0, 0.1],
            }
        ],
        "task_focus": {"target_instances": ["target_01"]},
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "left", "role": "target", "point_key": "contact_world_m"},
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "negative", "distance": 0.035, "steps": 3},
            ),
        ]
    )

    assert [call.tool_name for call in calls] == ["move_ee_to_grounded_instance", "contact_displace"]
    assert calls[0].args["point_key"] == "contact_world_m"
    np.testing.assert_allclose(calls[0].args["offset_xyz"], [0.0, 0.0, 0.045], atol=1e-9)
    assert "staged active contact" in calls[0].args["_guard_reason"]
    assert calls[1].args["distance"] == 0.035
    assert "_contact_reference_world_m" not in calls[1].args


def test_recovery_guard_completes_marked_transient_contact_before_reobserve() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.latest_snapshot.head_cam2world_gl = np.eye(4, dtype=np.float64)
    agent.latest_snapshot.head_cam2world_gl[:3, 3] = [0.1, -1.0, 0.1]
    agent.memory_store.reset(
        task="press the control once",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.scene_memory = {
        "instances": [
            {
                "instance_id": "internal_instance_01",
                "track_id": "track_0001",
                "status": "visible",
                "stability": "stable",
                "camera": "head",
                "approach_world_m": [0.1, 0.0, 0.2],
                "contact_world_m": [0.1, 0.0, 0.1],
                "quality": {"world_extent_m": [0.06, 0.04, 0.02]},
            }
        ],
        "task_focus": {"target_instances": ["track_0001"]},
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"instance_ref": "track_0001", "arm": "left", "role": "target", "point_key": "contact_world_m"},
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={
                    "arm": "left",
                    "axis": "z",
                    "direction": "negative",
                    "distance": 0.012,
                    "steps": 2,
                    "complete_transient_cycle": True,
                },
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in calls] == [
        "move_ee_to_grounded_instance",
        "contact_displace",
        "move_ee_to_grounded_instance",
        "move_ee_to_grounded_instance",
        "reobserve_scene",
    ]
    assert calls[1].args["complete_transient_cycle"] is True
    assert agent._public_tool_args(calls[1].args)["complete_transient_cycle"] is True
    assert calls[2].args["instance_ref"] == "track_0001"
    assert calls[2].args["arm"] == "left"
    assert calls[2].args["point_key"] == "approach_world_m"
    assert np.isclose(calls[2].args["max_translation"], 0.112)
    assert calls[2].args["steps"] == 2
    assert "complete one transient contact cycle" in calls[2].args["_guard_reason"]
    assert "complete_transient_cycle" not in calls[2].args
    assert calls[3].args["instance_ref"] == "track_0001"
    assert calls[3].args["arm"] == "left"
    assert calls[3].args["point_key"] == "approach_world_m"
    assert calls[3].args["post_contact_clearance"] == "camera_tangent"
    np.testing.assert_allclose(calls[3].args["offset_xyz"], [-0.06, 0.0, 0.0], atol=1e-6)
    assert np.isclose(calls[3].args["max_translation"], 0.06)
    assert calls[3].args["steps"] == 2
    assert "current camera/contact geometry" in calls[3].args["_guard_reason"]
    assert agent._public_tool_args(calls[3].args)["post_contact_clearance"] == "camera_tangent"
    assert agent._grounded_setup_key(calls[3], {}) is None


def test_transient_visual_clearance_rotates_with_camera_and_contact_geometry() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(0.0)
    snapshot.head_cam2world_gl = np.eye(4, dtype=np.float64)
    snapshot.head_cam2world_gl[:3, 3] = [-1.0, 1.0, 0.0]
    snapshot.left_endpose = np.asarray([-0.1, -0.1, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    agent.latest_snapshot = snapshot

    approach = [0.1, 0.1, 0.1]
    contact = [0.0, 0.0, 0.0]
    offset = agent._transient_visual_clearance_offset(
        instance={
            "camera": "head",
            "quality": {"world_extent_m": [0.03, 0.06, 0.09]},
        },
        arm="left",
        approach=approach,
        contact=contact,
    )

    assert offset is not None
    normal = np.asarray(approach) - np.asarray(contact)
    view = np.asarray(contact) - snapshot.head_cam2world_gl[:3, 3]
    assert abs(float(np.dot(offset, normal))) < 1e-6
    assert abs(float(np.dot(offset, view))) < 1e-6
    assert sum(abs(value) > 1e-4 for value in offset) == 3


def test_transient_visual_clearance_fails_closed_without_camera_calibration() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)

    offset = agent._transient_visual_clearance_offset(
        instance={
            "camera": "head",
            "quality": {"world_extent_m": [0.06, 0.04, 0.02]},
        },
        arm="left",
        approach=[0.1, 0.0, 0.2],
        contact=[0.1, 0.0, 0.1],
    )

    assert offset is None


def test_transient_visual_clearance_does_not_use_the_moving_wrist_camera() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(0.0)
    snapshot.left_cam2world_gl = np.eye(4, dtype=np.float64)
    agent.latest_snapshot = snapshot

    offset = agent._transient_visual_clearance_offset(
        instance={
            "camera": "left",
            "quality": {"world_extent_m": [0.06, 0.04, 0.02]},
        },
        arm="left",
        approach=[0.1, 0.0, 0.2],
        contact=[0.1, 0.0, 0.1],
    )

    assert offset is None


def test_recovery_guard_does_not_infer_transient_cycle_from_subtask_text() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="press the control once",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="Press the middle control exactly once for repetition 2 of 3",
        skill_spec=skill_spec,
    )
    agent.memory_store.state.working.scene_memory = {
        "instances": [
            {
                "instance_id": "target_01",
                "status": "visible",
                "stability": "stable",
                "approach_world_m": [0.1, 0.0, 0.2],
                "contact_world_m": [0.1, 0.0, 0.1],
            }
        ],
        "task_focus": {"target_instances": ["target_01"]},
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"instance_id": "target_01", "arm": "left", "point_key": "contact_world_m"},
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "negative", "distance": 0.012},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in calls] == [
        "move_ee_to_grounded_instance",
        "contact_displace",
        "reobserve_scene",
    ]
    assert "complete_transient_cycle" not in calls[1].args


def test_recovery_guard_preserves_unmarked_release_during_active_press_subtask() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="press the control once",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="Press the middle control exactly once",
        skill_spec=skill_spec,
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "positive", "distance": 0.02},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in calls] == ["contact_displace", "reobserve_scene"]
    assert "complete_transient_cycle" not in calls[0].args


def test_recovery_guard_does_not_retract_unmarked_persistent_contact() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="lower the held object onto support",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.scene_memory = {
        "instances": [
            {
                "instance_id": "support_01",
                "approach_world_m": [0.1, 0.0, 0.2],
                "contact_world_m": [0.1, 0.0, 0.1],
            }
        ],
        "task_focus": {"target_instances": ["support_01"]},
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"instance_id": "support_01", "arm": "left", "point_key": "contact_world_m"},
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "negative", "distance": 0.012},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in calls] == [
        "move_ee_to_grounded_instance",
        "contact_displace",
        "reobserve_scene",
    ]


def test_recovery_guard_does_not_duplicate_explicit_grounded_transient_release() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.latest_snapshot.head_cam2world_gl = np.eye(4, dtype=np.float64)
    agent.latest_snapshot.head_cam2world_gl[:3, 3] = [0.1, -1.0, 0.1]
    agent.memory_store.reset(
        task="press the control once",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.scene_memory = {
        "instances": [
            {
                "instance_id": "target_01",
                "camera": "head",
                "approach_world_m": [0.1, 0.0, 0.2],
                "contact_world_m": [0.1, 0.0, 0.1],
                "quality": {"world_extent_m": [0.06, 0.04, 0.02]},
            }
        ],
        "task_focus": {"target_instances": ["target_01"]},
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"instance_id": "target_01", "arm": "left", "point_key": "contact_world_m"},
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={
                    "arm": "left",
                    "axis": "z",
                    "direction": "negative",
                    "distance": 0.012,
                    "complete_transient_cycle": True,
                },
            ),
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"instance_id": "target_01", "arm": "left", "point_key": "approach_world_m"},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in calls] == [
        "move_ee_to_grounded_instance",
        "contact_displace",
        "move_ee_to_grounded_instance",
        "move_ee_to_grounded_instance",
        "reobserve_scene",
    ]
    release_calls = [
        call
        for call in calls
        if call.tool_name == "move_ee_to_grounded_instance"
        and call.args.get("point_key") == "approach_world_m"
        and not call.args.get("post_contact_clearance")
    ]
    clearance_calls = [call for call in calls if call.args.get("post_contact_clearance") == "camera_tangent"]
    assert len(release_calls) == 1
    assert len(clearance_calls) == 1
    assert calls.index(release_calls[0]) < calls.index(clearance_calls[0]) < len(calls) - 1


def test_recovery_guard_blocks_marked_transient_contact_without_grounded_setup() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="press the control once",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="contact_displace",
                args={
                    "arm": "left",
                    "axis": "z",
                    "direction": "negative",
                    "distance": 0.012,
                    "complete_transient_cycle": True,
                },
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in calls] == ["reobserve_scene"]
    assert "without a same-batch grounded contact setup" in calls[0].args["_guard_reason"]


def test_recovery_guard_caps_unstaged_same_direction_contact_overtravel() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    scene_memory = {
        "instances": [
            {
                "instance_id": "target_01",
                "approach_world_m": [0.1, 0.0, 0.2],
                "contact_world_m": [0.1, 0.0, 0.1],
            }
        ],
        "task_focus": {"target_instances": ["target_01"]},
    }
    grounded_call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "arm": "left",
            "role": "target",
            "point_key": "contact_world_m",
            "_scene_memory": scene_memory,
        },
    )

    guarded = agent._guard_grounded_contact_overtravel(
        RecoveryToolCall(
            tool_name="contact_displace",
            args={"arm": "left", "axis": "z", "direction": "negative", "distance": 0.035},
        ),
        {"left": grounded_call},
    )

    assert guarded.args["distance"] == 0.01
    assert guarded.args["_contact_reference_world_m"] == [0.1, 0.0, 0.1]
    assert guarded.args["_contact_reference_tolerance_m"] == 0.01
    assert "capped same-direction displacement" in guarded.args["_guard_reason"]


def test_recovery_guard_preserves_retreat_from_grounded_contact() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="release grounded contact",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.scene_memory = {
        "instances": [
            {
                "instance_id": "target_01",
                "approach_world_m": [0.1, 0.0, 0.2],
                "contact_world_m": [0.1, 0.0, 0.1],
            }
        ],
        "task_focus": {"target_instances": ["target_01"]},
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "left", "role": "target", "point_key": "contact_world_m"},
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "positive", "distance": 0.035},
            ),
        ]
    )

    assert calls[1].args["distance"] == 0.035
    assert "_guard_reason" not in calls[1].args


def test_grounded_move_reached_uses_observed_pose_over_commanded_pose() -> None:
    agent = ImgAgent(make_card(DummyConfig()))

    reached = agent._grounded_move_reached_point(
        {
            "target_pose": [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
            "executed_pose": [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
            "observed_pose": [0.1, 0.03, 0.1, 1.0, 0.0, 0.0, 0.0],
        }
    )

    assert reached == "false"


def test_recovery_guard_blocks_cross_arm_move_when_other_arm_occupies_target() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.left_endpose = np.asarray([0.05, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    snapshot.right_endpose = np.asarray([0.3, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="continue grounded manipulation",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.scene_memory = {
        "instances": [{"instance_id": "target_01", "approach_world_m": [0.05, 0.0, 0.1]}],
        "task_focus": {"target_instances": ["target_01"]},
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "right", "role": "target", "point_key": "approach_world_m"},
            )
        ]
    )

    assert [call.tool_name for call in calls] == ["reobserve_scene"]
    assert "left EE occupies the target envelope" in calls[0].args["_guard_reason"]
    assert "retreat and verify" in calls[0].args["_guard_reason"]


def test_recovery_guard_truncates_static_batch_at_reobserve() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(1.0)
    agent.memory_store.reset(
        task="continue grounded manipulation",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(tool_name="lift_ee", args={"arm": "left", "distance": 0.02}),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "negative", "distance": 0.01},
            ),
        ]
    )

    assert [call.tool_name for call in calls] == ["lift_ee", "reobserve_scene"]


def test_recovery_guard_blocks_close_after_approach_only_grounded_move() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(1.0)
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "left", "role": "tool", "point_key": "approach_world_m"},
            ),
            RecoveryToolCall(tool_name="close_gripper", args={"arm": "left"}),
        ]
    )

    assert [call.tool_name for call in calls] == ["move_ee_to_grounded_instance", "reobserve_scene"]
    assert "approach-only" in calls[1].args["_guard_reason"]


def test_recovery_guard_allows_close_after_grasp_grounded_move() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(1.0)
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "left", "role": "tool", "point_key": "grasp_world_m"},
            ),
            RecoveryToolCall(tool_name="close_gripper", args={"arm": "left"}),
        ]
    )

    assert [call.tool_name for call in calls] == [
        "move_ee_to_grounded_instance",
        "close_gripper",
        "reobserve_scene",
    ]
    assert calls[1].args.get("_runtime_grasp_close_boundary") is True


def test_recovery_history_records_tool_context() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    scene_memory = {
        "instances": [{"instance_id": "lid_left", "approach_world_m": [0.1, 0.0, 0.2], "grasp_world_m": [0.1, 0.0, 0.1]}],
        "task_focus": {"tool_instances": ["lid_left"]},
    }
    calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={"arm": "left", "role": "tool", "point_key": "grasp_world_m", "_scene_memory": scene_memory},
        ),
        RecoveryToolCall(tool_name="close_gripper", args={"arm": "left"}),
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={"arm": "left", "role": "tool", "point_key": "approach_world_m", "instance_id": "lid_01"},
        ),
    ]
    results = [
        RecoveryToolResult(
            tool_name="move_ee_to_grounded_instance",
            success=True,
            message="executed bounded EE move to grounded scene-memory instance",
            details={
                "arm": "left",
                "steps": 2,
                "target_pose": [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
                "executed_pose": [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
            },
        ),
        RecoveryToolResult(
            tool_name="close_gripper",
            success=True,
            message="set left gripper qpos slots [6] to 0.0",
            details={"arm": "left", "gripper_value": 0.0},
        ),
        RecoveryToolResult(
            tool_name="move_ee_to_grounded_instance",
            success=False,
            message="scene instance has no finite approach_world_m",
            details={"instance_id": "lid_01"},
        ),
    ]

    entry = agent._format_recovery_history_entry("task-level-recovery-control", calls, results)

    assert "move_ee_to_grounded_instance=True" in entry
    assert "role=tool" in entry
    assert "instance=lid_left" in entry
    assert "point=grasp_world_m" in entry
    assert "reached_point=true" in entry
    assert "steps=2" in entry
    assert "close_gripper=True" in entry
    assert "after_point=grasp_world_m" in entry
    assert "after_point_reached=true" in entry
    assert "gripper=0" in entry
    assert "grasp_confirmed=unverified" in entry
    assert "move_ee_to_grounded_instance=False" in entry
    assert "instance=lid_01" in entry
    assert "point=approach_world_m" in entry
    assert "failure=scene instance has no finite approach_world_m" in entry
    assert "_scene_memory" not in entry


def test_recovery_history_never_selects_first_ambiguous_or_identity_bound_focus() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    base_call_args = {
        "arm": "left",
        "role": "target",
        "point_key": "approach_world_m",
    }

    ambiguous_call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            **base_call_args,
            "_scene_memory": {
                "task_focus": {"target_instances": ["target_01", "target_02"]},
            },
        },
    )
    identity_bound_call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            **base_call_args,
            "_scene_memory": {
                "task_focus": {
                    "target_instances": ["target_01"],
                    "identity_binding_required": True,
                },
            },
        },
    )

    assert agent._recovery_history_instance_id(ambiguous_call, {}) == ""
    assert agent._recovery_history_instance_id(identity_bound_call, {}) == ""


def test_debug_recovery_pure_control_stops_at_max_rounds() -> None:
    agent = ImgAgent(make_card(DummyConfig(debug_recovery_max_rounds=1)))
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    with mock.patch.object(agent, "_dispatch_recovery_tools", return_value="retry"):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert agent.memory_store.state.task.task_finished is True
    assert agent.memory_store.state.monitor.phase == "finished"
    assert agent.memory_store.state.monitor.env_signal == "debug_pure_recovery_max_rounds"


def test_pure_tool_control_round_limit_requests_replan_without_finishing_task() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_rounds=1,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)

    with mock.patch.object(agent, "_dispatch_recovery_tools", return_value="retry"):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is None
    assert agent.memory_store.state.monitor.phase == "reasoning"
    assert agent.memory_store.state.monitor.status == "needs_reasoning"
    assert agent.memory_store.state.monitor.env_signal == "running"
    assert "pure_tool_control_max_rounds_replan" in agent.memory_store.state.working.recovery_history
    assert "pure_tool_control_subtask_budget_exhausted" in agent.memory_store.state.working.recovery_history
    assert "planner selected step" in agent.memory_store.state.task.failed_skills


def test_dispatch_uses_after_action_status_instead_of_pre_action_intent() -> None:
    cases = (
        ("replan", "true", "in_progress", "continue", "continue"),
        ("retry", "true", "completed", "replan", "subtask_complete"),
        ("retry", "false", "failed", "replan", "replan"),
        ("replan", "unverified", "uncertain", "replan", "retry"),
    )

    for proposed_intent, effect_verified, subtask_status, recommended_control, expected_control in cases:
        agent = ImgAgent(
            make_card(
                DummyConfig(
                    debug_recovery_enabled=False,
                    pure_tool_control_enabled=True,
                )
            )
        )
        agent.current_instruction = "generic manipulation task"
        agent.latest_snapshot = make_snapshot()
        agent.memory_store.reset(
            task=agent.current_instruction,
            control_model_name="test",
            available_policies=[],
            available_tools=[],
            mode="deployment",
        )
        skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
        agent.memory_store.start_or_replace_skill(
            subtask_text="perform the current grounded manipulation step",
            skill_spec=skill_spec,
        )
        route = RecoveryRoute(
            signal_name=TASK_LEVEL_RECOVERY_CONTROL,
            workflow_name="task-level-recovery-control",
            plan=RecoveryPrimitivePlan(
                name="agent-selected-plan",
                tool_calls=[RecoveryToolCall(tool_name="reobserve_scene", args={})],
                expected_outcome="evaluate the current physical stop condition",
            ),
            post_recovery_intent=proposed_intent,
            reason="agent-selected recovery",
        )
        verifier_result = {
            "effect_verified": effect_verified,
            "effect_type": "unknown",
            "confidence": 0.8,
            "evidence_summary": "fresh after-action evidence",
            "failure_reason": "",
            "next_constraint": "",
            "memory_update": "checked current physical state",
            "subtask_status": subtask_status,
            "recommended_control": recommended_control,
        }
        signal = agent._make_monitor_signal(
            name=TASK_LEVEL_RECOVERY_CONTROL,
            level="warning",
            reason="pure tool-control evaluation",
        )

        with (
            mock.patch.object(agent._recovery_dispatcher, "available_tools", return_value=["reobserve_scene"]),
            mock.patch.object(agent._recovery_policy_resolver, "resolve", return_value=route),
            mock.patch.object(
                agent._recovery_dispatcher,
                "dispatch_batch",
                return_value=[RecoveryToolResult(tool_name="reobserve_scene", success=True)],
            ),
            mock.patch.object(agent._action_effect_verifier, "verify", return_value=verifier_result) as verify,
        ):
            resolved_control = agent._dispatch_recovery_tools(task_env=object(), signal=signal)

        assert resolved_control == expected_control
        assert verify.call_count == 1
        evidence_payload = verify.call_args.args[0]
        assert evidence_payload["post_recovery_intent"] == proposed_intent
        assert evidence_payload["current_subtask"] == "perform the current grounded manipulation step"
        assert evidence_payload["skill_id"] == agent.memory_store.state.active_skill.skill_id
        history_entry = agent.memory_store.state.working.recovery_history[-1]
        assert f"subtask={subtask_status}" in history_entry
        assert f"control={expected_control}" in history_entry
        if recommended_control != expected_control:
            assert f"recommended={recommended_control}" in history_entry


def test_dispatch_commits_authoritative_environment_success_without_api_verifier() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_control_turns=8,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    subtask = "perform the final grounded manipulation step"
    agent.memory_store.start_or_replace_skill(subtask_text=subtask, skill_spec=skill_spec)
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="terminal-effect-plan",
            tool_calls=[RecoveryToolCall(tool_name="reobserve_scene", args={})],
            expected_outcome="complete the committed physical effect",
        ),
        post_recovery_intent="retry",
        reason="execute the final bounded effect",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="pure tool-control evaluation",
    )
    stale_snapshot = make_snapshot_with_success(eval_success=False)
    agent._recovery_dispatcher.latest_snapshot = stale_snapshot
    agent._recovery_dispatcher.latest_environment_success = True
    agent.memory_store.set_recovery_policy(action="retry", reason="stale recovery state")
    agent.memory_store.set_last_error("stale error")
    agent._pure_tool_control_terminal_failure = "stale terminal failure"

    with (
        mock.patch.object(agent._recovery_dispatcher, "available_tools", return_value=["reobserve_scene"]),
        mock.patch.object(agent._recovery_policy_resolver, "resolve", return_value=route),
        mock.patch.object(
            agent._recovery_dispatcher,
            "dispatch_batch",
            return_value=[
                RecoveryToolResult(
                    tool_name="reobserve_scene",
                    success=True,
                    details={"environment_success": True, "step_count": 1},
                )
            ],
        ),
        mock.patch.object(agent._action_effect_verifier, "verify") as verify,
        mock.patch.object(agent, "_trace") as trace,
        mock.patch.object(agent, "_dump_rollout_event"),
    ):
        resolved_control = agent._dispatch_recovery_tools(task_env=object(), signal=signal)
        assert resolved_control == "task_complete"
        verify.assert_not_called()

        agent._debug_recovery_rounds = 1
        agent._continue_debug_recovery_pure_control(
            preferred_action=resolved_control,
            signal=signal,
            reason="pure tool-control evaluation",
        )

    status = agent.current_status()
    assert status["environment_success"] is True
    assert status["task_finished"] is True
    assert status["task_finish_validated"] is True
    assert status["recovery_pending"] == ""
    assert status["terminal_failure"] is False
    assert agent.memory_store.state.working.last_error == ""
    assert agent.memory_store.state.active_skill is None
    assert subtask in agent.memory_store.state.task.completed_skills
    assert subtask not in agent.memory_store.state.task.failed_skills
    assert agent.memory_store.state.monitor.phase == "finished"
    assert agent.memory_store.state.monitor.status == "finished"
    assert agent.memory_store.state.monitor.env_signal == "task_success"

    assert not any(call.args == ("action_effect_verification",) for call in trace.call_args_list)
    effect_trace = next(
        call for call in trace.call_args_list if call.args == ("environment_success_effect_commit",)
    )
    assert effect_trace.kwargs["authority"] == "environment_eval_success"
    assert effect_trace.kwargs["verifier_bypassed"] is True
    assert effect_trace.kwargs["result"]["global_task_success"] is True
    decision_trace = next(call for call in trace.call_args_list if call.args == ("subtask_transition_decision",))
    assert decision_trace.kwargs["authority"] == "environment_eval_success"
    assert decision_trace.kwargs["resolved_control"] == "task_complete"
    commit_trace = next(call for call in trace.call_args_list if call.args == ("global_task_success_commit",))
    assert commit_trace.kwargs["next_action"] == "stop_global_task"


def test_terminal_snapshot_skips_further_perception_preprocessing() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    terminal_snapshot = make_snapshot_with_success(eval_success=True)

    with (
        mock.patch.object(agent, "preprocess_observation") as preprocess,
        mock.patch.object(agent, "_trace") as trace,
        mock.patch.object(agent, "_dump_rollout_event"),
        mock.patch.object(agent, "_dump_snapshot_images") as dump_images,
    ):
        agent.update_snapshot(terminal_snapshot)

    preprocess.assert_not_called()
    dump_images.assert_called_once_with(terminal_snapshot)
    assert agent._runtime_evaluation_context()["environment_success"] is True
    skip_trace = next(
        call for call in trace.call_args_list if call.args == ("terminal_observation_preprocess_skipped",)
    )
    assert skip_trace.kwargs["authority"] == "environment_eval_success"


def test_recovery_route_api_is_skipped_when_task_env_is_already_terminal() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot_with_success(eval_success=False)
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    subtask = "perform a grounded manipulation step"
    agent.memory_store.start_or_replace_skill(subtask_text=subtask, skill_spec=skill_spec)
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="pure tool-control evaluation",
    )

    class TerminalTaskEnv:
        eval_success = True

    with (
        mock.patch.object(agent._recovery_dispatcher, "available_tools") as available_tools,
        mock.patch.object(agent._recovery_policy_resolver, "resolve") as resolve,
        mock.patch.object(agent, "_trace") as trace,
        mock.patch.object(agent, "_dump_rollout_event"),
    ):
        resolved_control = agent._dispatch_recovery_tools(TerminalTaskEnv(), signal)
        assert resolved_control == "task_complete"
        available_tools.assert_not_called()
        resolve.assert_not_called()

        agent._continue_debug_recovery_pure_control(
            preferred_action=resolved_control,
            signal=signal,
            reason="pure tool-control evaluation",
        )

    status = agent.current_status()
    assert status["environment_success"] is True
    assert status["task_finished"] is True
    assert status["task_finish_validated"] is True
    short_circuit = next(
        call
        for call in trace.call_args_list
        if call.args == ("recovery_environment_success_short_circuit",)
    )
    assert short_circuit.kwargs["stage"] == "before_recovery_route"


def test_after_action_subtask_status_overrides_pre_action_replan_intent() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )

    control, reason = agent._resolve_after_action_subtask_control(
        proposed_intent="replan",
        action_effect={
            "effect_verified": "true",
            "subtask_status": "in_progress",
            "recommended_control": "continue",
        },
    )

    assert control == "continue"
    assert "kept the current subtask active" in reason


def test_after_action_completion_requires_verified_physical_effect() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )

    verified, _ = agent._resolve_after_action_subtask_control(
        proposed_intent="retry",
        action_effect={
            "effect_verified": "true",
            "subtask_status": "completed",
            "recommended_control": "continue",
        },
    )
    unverified, reason = agent._resolve_after_action_subtask_control(
        proposed_intent="replan",
        action_effect={
            "effect_verified": "unverified",
            "subtask_status": "completed",
            "recommended_control": "replan",
        },
    )

    assert verified == "subtask_complete"
    assert unverified == "retry"
    assert "completion rejected" in reason


def test_pure_tool_control_verified_completion_commits_succeeded_before_next_plan() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_rounds=10,
                pure_tool_control_max_control_turns=8,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    subtask = "planner selected step"
    agent.memory_store.start_or_replace_skill(subtask_text=subtask, skill_spec=skill_spec)
    agent._debug_recovery_rounds = 3
    agent._debug_recovery_planner_bootstrapped = True
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="paper pure tool-control baseline",
    )

    agent._continue_debug_recovery_pure_control(
        preferred_action="subtask_complete",
        signal=signal,
        reason="verified stop condition",
    )

    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is None
    assert subtask in agent.memory_store.state.task.completed_skills
    assert subtask not in agent.memory_store.state.task.failed_skills
    assert agent.memory_store.state.monitor.phase == "reasoning"
    assert agent.memory_store.state.monitor.status == "needs_reasoning"
    assert agent.memory_store.state.monitor.env_signal == "running"
    assert agent._debug_recovery_rounds == 0
    assert agent._debug_recovery_triggered is False
    assert agent._debug_recovery_planner_bootstrapped is False
    assert agent.current_status()["max_control_turns"] == 8
    assert "pure_tool_control_subtask_completed" in agent.memory_store.state.working.recovery_history


def test_verified_completion_uses_agent_api_planner_for_next_subtask() -> None:
    control_runtime = FakeControlRuntime(
        {
            "action_mode": "start",
            "commit_label": "state_change",
            "memory_text": "The previous subtask is complete.",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": "agent api selected next subtask",
        }
    )
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_rounds=10,
                pure_tool_control_max_control_turns=8,
            ),
            control_runtime=control_runtime,
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    previous_subtask = "verified completed subtask"
    agent.memory_store.start_or_replace_skill(subtask_text=previous_subtask, skill_spec=skill_spec)
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="paper pure tool-control baseline",
    )

    agent._continue_debug_recovery_pure_control(
        preferred_action="subtask_complete",
        signal=signal,
        reason="verified stop condition",
    )
    assert agent._maybe_bootstrap_debug_recovery_with_planner() is True

    active = agent.memory_store.state.active_skill
    assert active is not None
    assert active.instruction == "agent api selected next subtask"
    assert previous_subtask in agent.memory_store.state.task.completed_skills
    assert previous_subtask not in agent.memory_store.state.task.failed_skills
    assert len(control_runtime.calls) == 1
    assert agent.current_status()["control_turn_count"] == 1


def test_after_action_failed_status_is_the_only_normal_replan_transition() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )

    failed, _ = agent._resolve_after_action_subtask_control(
        proposed_intent="retry",
        action_effect={
            "effect_verified": "false",
            "subtask_status": "failed",
            "recommended_control": "replan",
        },
    )
    uncertain, reason = agent._resolve_after_action_subtask_control(
        proposed_intent="replan",
        action_effect={
            "effect_verified": "unverified",
            "subtask_status": "uncertain",
            "recommended_control": "replan",
        },
    )

    assert failed == "replan"
    assert uncertain == "retry"
    assert "gather more evidence" in reason


def test_pure_tool_control_failed_transition_commits_failed_only() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    subtask = "perform the current grounded manipulation step"
    agent.memory_store.start_or_replace_skill(subtask_text=subtask, skill_spec=skill_spec)
    agent._debug_recovery_rounds = 2
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="pure tool-control evaluation",
    )

    agent._continue_debug_recovery_pure_control(
        preferred_action="replan",
        signal=signal,
        reason="after-action verifier classified the subtask as failed",
    )

    assert agent.memory_store.state.active_skill is None
    assert subtask in agent.memory_store.state.task.failed_skills
    assert subtask not in agent.memory_store.state.task.completed_skills
    assert agent.memory_store.state.monitor.phase == "reasoning"
    assert agent.memory_store.state.monitor.status == "needs_reasoning"
    assert agent._debug_recovery_rounds == 0


def test_pure_tool_control_missing_recovery_route_retries_without_finishing_task() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)

    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="paper pure tool-control baseline",
    )

    with mock.patch.object(agent._recovery_policy_resolver, "resolve", return_value=None):
        intent = agent._dispatch_recovery_tools(task_env=object(), signal=signal)
    assert intent == "retry"
    agent._continue_debug_recovery_pure_control(
        preferred_action=intent,
        signal=signal,
        reason="paper pure tool-control baseline",
    )

    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is not None
    assert agent.memory_store.state.monitor.phase == "monitoring"
    assert agent.memory_store.state.monitor.status == "rollout_active"
    assert "pure_tool_control_round:0" in agent.memory_store.state.working.recovery_history


def test_recovery_backend_errors_do_not_consume_semantic_rounds_and_end_as_infrastructure_failure() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_backend_error_budget=2,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="perform one grounded manipulation step",
        skill_spec=skill_spec,
    )
    loader = agent._recovery_policy_resolver._workflow_loader
    loader.last_error = "recovery planner failed: HTTPError: HTTP Error 502: Bad Gateway"
    loader.last_error_kind = "backend"

    with (
        mock.patch.object(agent._recovery_dispatcher, "available_tools", return_value=[]),
        mock.patch.object(agent._recovery_policy_resolver, "resolve", return_value=None),
    ):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent._debug_recovery_rounds == 0
        assert agent.current_status()["terminal_failure"] is False

        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    status = agent.current_status()
    assert status["terminal_failure"] is True
    assert status["terminal_failure_reason"] == "pure_tool_control_recovery_backend_unavailable:2"
    assert status["recovery_backend_error_count"] == 2
    assert status["control_turn_count"] == 0
    assert agent._debug_recovery_rounds == 0


def test_action_effect_backend_retry_does_not_replay_physical_batch_or_consume_round() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_backend_error_budget=3,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="perform one grounded manipulation step",
        skill_spec=skill_spec,
    )
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="one-physical-batch",
            tool_calls=[
                RecoveryToolCall(
                    tool_name="retreat_arm",
                    args={"arm": "left", "axis": "x", "direction": "negative", "distance": 0.01},
                )
            ],
            expected_outcome="clearance observed",
        ),
        post_recovery_intent="retry",
        reason="bounded generic clearance",
    )
    verified = {
        "effect_verified": "true",
        "effect_type": "move",
        "confidence": 0.8,
        "evidence_summary": "bounded displacement observed",
        "failure_reason": "",
        "next_constraint": "continue from fresh evidence",
        "memory_update": "clearance verified",
        "subtask_status": "in_progress",
        "recommended_control": "continue",
    }
    tool_result = RecoveryToolResult(
        tool_name="retreat_arm",
        success=True,
        message="executed bounded EE retreat displacement",
        details={"arm": "left"},
    )

    with (
        mock.patch.object(agent._recovery_dispatcher, "available_tools", return_value=["retreat_arm"]),
        mock.patch.object(agent._recovery_policy_resolver, "resolve", return_value=route) as resolve,
        mock.patch.object(agent._recovery_dispatcher, "dispatch_batch", return_value=[tool_result]) as dispatch,
        mock.patch.object(
            agent._action_effect_verifier,
            "verify",
            side_effect=[ActionEffectVerifierBackendError("HTTP 502"), verified],
        ) as verify,
    ):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        first_status = agent.current_status()
        assert agent._debug_recovery_rounds == 0
        assert first_status["recovery_backend_error_count"] == 1
        assert first_status["recovery_backend_error_stage"] == "action_effect_verifier"
        assert first_status["action_effect_verification_pending"] is True

        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    assert agent._debug_recovery_rounds == 1
    assert agent.current_status()["recovery_backend_error_count"] == 0
    assert agent.current_status()["action_effect_verification_pending"] is False
    assert dispatch.call_count == 1
    assert resolve.call_count == 1
    assert verify.call_count == 2


def test_action_effect_backend_error_budget_exhaustion_is_infrastructure_failure_without_replay() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_backend_error_budget=2,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="perform one grounded manipulation step",
        skill_spec=skill_spec,
    )
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="one-physical-batch",
            tool_calls=[RecoveryToolCall(tool_name="retreat_arm", args={"arm": "left"})],
            expected_outcome="clearance observed",
        ),
        post_recovery_intent="retry",
        reason="bounded generic clearance",
    )
    tool_result = RecoveryToolResult(tool_name="retreat_arm", success=True, message="ok", details={"arm": "left"})

    with (
        mock.patch.object(agent._recovery_dispatcher, "available_tools", return_value=["retreat_arm"]),
        mock.patch.object(agent._recovery_policy_resolver, "resolve", return_value=route) as resolve,
        mock.patch.object(agent._recovery_dispatcher, "dispatch_batch", return_value=[tool_result]) as dispatch,
        mock.patch.object(
            agent._action_effect_verifier,
            "verify",
            side_effect=[
                ActionEffectVerifierBackendError("HTTP 502 first"),
                ActionEffectVerifierBackendError("HTTP 502 second"),
            ],
        ) as verify,
    ):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    status = agent.current_status()
    assert status["terminal_failure"] is True
    assert status["terminal_failure_reason"] == "pure_tool_control_recovery_backend_unavailable:2"
    assert status["action_effect_verification_pending"] is False
    assert agent._debug_recovery_rounds == 0
    assert dispatch.call_count == 1
    assert resolve.call_count == 1
    assert verify.call_count == 2


def test_environment_success_supersedes_pending_action_effect_verification() -> None:
    agent = ImgAgent(
        make_card(DummyConfig(debug_recovery_enabled=False, pure_tool_control_enabled=True))
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="perform one grounded manipulation step",
        skill_spec=skill_spec,
    )
    agent._pending_action_effect_verification = {"pending": True}
    task_env = type("SuccessfulEnv", (), {"eval_success": True})()
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="retry pending verifier",
    )

    with (
        mock.patch.object(agent._recovery_policy_resolver, "resolve") as resolve,
        mock.patch.object(agent._recovery_dispatcher, "dispatch_batch") as dispatch,
        mock.patch.object(agent._action_effect_verifier, "verify") as verify,
    ):
        result = agent._dispatch_recovery_tools(task_env=task_env, signal=signal)

    assert result == "task_complete"
    assert agent.current_status()["action_effect_verification_pending"] is False
    assert agent.current_status()["environment_success"] is True
    assert resolve.call_count == 0
    assert dispatch.call_count == 0
    assert verify.call_count == 0
    assert agent._commit_pure_tool_control_environment_success(source="test") is True
    assert agent.current_status()["task_finished"] is True


def test_recovery_backend_error_budget_resets_after_valid_route() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_backend_error_budget=2,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="perform one grounded manipulation step",
        skill_spec=skill_spec,
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="paper pure tool-control baseline",
    )
    loader = agent._recovery_policy_resolver._workflow_loader
    loader.last_error = "recovery planner failed: HTTPError: HTTP Error 502: Bad Gateway"
    loader.last_error_kind = "backend"

    agent._handle_pure_tool_control_recovery_backend_error(signal=signal)
    assert agent.current_status()["recovery_backend_error_count"] == 1

    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(name="valid-empty-plan", tool_calls=[], expected_outcome="reobserve"),
        post_recovery_intent="retry",
        reason="valid recovery backend response",
    )
    with (
        mock.patch.object(agent._recovery_dispatcher, "available_tools", return_value=[]),
        mock.patch.object(agent._recovery_policy_resolver, "resolve", return_value=route),
    ):
        agent._dispatch_recovery_tools(task_env=object(), signal=signal)

    assert agent.current_status()["recovery_backend_error_count"] == 0

    agent._handle_pure_tool_control_recovery_backend_error(signal=signal)
    status = agent.current_status()
    assert status["terminal_failure"] is False
    assert status["recovery_backend_error_count"] == 1


def test_consecutive_valid_empty_recovery_plans_fast_replan() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_empty_plan_replan_threshold=2,
            )
        )
    )
    agent.current_instruction = "generic manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    subtask = "perform one grounded manipulation step"
    agent.memory_store.start_or_replace_skill(subtask_text=subtask, skill_spec=skill_spec)
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(name="empty-plan", tool_calls=[], expected_outcome="replan"),
        post_recovery_intent="replan",
        selected_arm="none",
        reason="no safe grounded tool call",
    )

    with (
        mock.patch.object(agent._recovery_dispatcher, "available_tools", return_value=[]),
        mock.patch.object(agent._recovery_policy_resolver, "resolve", return_value=route),
        mock.patch.object(agent._action_effect_verifier, "verify") as verify,
    ):
        assert agent._maybe_force_debug_recovery(task_env=object()) is True
        assert agent.memory_store.state.active_skill is not None
        assert agent._debug_recovery_rounds == 1

        assert agent._maybe_force_debug_recovery(task_env=object()) is True

    verify.assert_not_called()
    assert agent.memory_store.state.active_skill is None
    assert subtask in agent.memory_store.state.task.failed_skills
    assert "pure_tool_control_empty_plan_replan" in agent.memory_store.state.working.recovery_history
    assert agent._debug_recovery_rounds == 0


def test_spent_observation_budget_hides_reobserve_until_physical_step_changes() -> None:
    agent = _prepare_evidence_policy_agent(
        phase="grasp_candidate",
        pure_tool_control=True,
    )
    observation = RecoveryToolCall(
        tool_name="reobserve_scene",
        args={},
    )
    context, situation, _ = agent._evidence_acquisition_context(
        observation
    )
    assert agent._evidence_acquisition_policy.decide(
        context,
        situation=situation,
    )["allow_reobserve"] is True

    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="no-safe-action",
            tool_calls=[],
            expected_outcome="continue rollout",
        ),
        post_recovery_intent="retry",
        selected_arm="none",
        reason="choose among advertised tools",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="continue the active subtask",
    )
    with (
        mock.patch.object(
            agent._recovery_dispatcher,
            "available_tools",
            return_value=[
                "move_ee_to_grounded_instance",
                "reobserve_scene",
            ],
        ),
        mock.patch.object(
            agent._recovery_policy_resolver,
            "resolve",
            return_value=route,
        ) as resolve,
    ):
        assert agent._dispatch_recovery_tools(
            task_env=object(),
            signal=signal,
        ) == _RECOVERY_EMPTY_PLAN

        first_request = resolve.call_args.kwargs
        assert first_request["available_tools"] == [
            "move_ee_to_grounded_instance"
        ]
        assert first_request["recovery_state"][
            "reobserve_scene_available"
        ] is False
        assert first_request["recovery_state"][
            "stationary_observation_budget"
        ]["stationary_scene_reobserve_budget_exhausted"] is True

        assert agent.latest_snapshot is not None
        agent.latest_snapshot = replace(
            agent.latest_snapshot,
            step_count=agent.latest_snapshot.step_count + 1,
        )
        assert agent._dispatch_recovery_tools(
            task_env=object(),
            signal=signal,
        ) == _RECOVERY_EMPTY_PLAN

    second_request = resolve.call_args.kwargs
    assert "reobserve_scene" in second_request["available_tools"]
    assert second_request["recovery_state"][
        "reobserve_scene_available"
    ] is True
    assert agent.current_status()["terminal_failure"] is False


def test_repeated_no_progress_grounded_setup_is_blocked_but_alternate_arm_remains_available() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    left_call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={"instance_id": "target_01", "arm": "left", "point_key": "contact_world_m"},
    )
    for target_error in (0.0319, 0.0317):
        agent._record_grounded_setup_outcomes(
            [left_call],
            [
                RecoveryToolResult(
                    tool_name="move_ee_to_grounded_instance",
                    success=True,
                    details={
                        "instance_id": "target_01",
                        "arm": "left",
                        "target_reached": False,
                        "target_observation_error_m": target_error,
                    },
                )
            ],
        )

    blocked = agent._blocked_grounded_setup_payload()
    assert blocked == [
        {
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "contact_world_m",
            "failure_count": 2,
            "last_target_error_m": 0.0317,
            "status": "blocked_after_repeated_no_progress",
        }
    ]
    guarded_left = agent._with_internal_recovery_context([left_call])
    assert guarded_left[0].tool_name == "reobserve_scene"
    assert "select another arm or strategy" in guarded_left[0].args["_guard_reason"]

    right_call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={"instance_id": "target_01", "arm": "right", "point_key": "contact_world_m"},
    )
    guarded_right = agent._with_internal_recovery_context([right_call])
    assert guarded_right[0].tool_name == "move_ee_to_grounded_instance"


def test_repeated_candidate_no_progress_switches_candidate_without_blocking_arm() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

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

    first_id = "rgbd_surface:grasp:left:000"
    second_id = "rgbd_surface:grasp:left:001"
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "target_01",
                    "status": "visible",
                    "stability": "stable",
                    "grasp_world_m": [0.2, 0.0, 0.0],
                    "operation_pose_candidates": [
                        candidate(first_id, 0.2, 0),
                        candidate(second_id, 0.25, 1),
                    ],
                }
            ],
            "task_focus": {"target_instances": ["target_01"]},
        }
    )
    raw_call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={"instance_id": "target_01", "arm": "left", "point_key": "grasp_world_m"},
    )
    first_call = agent._with_internal_recovery_context([raw_call])[0]
    assert first_call.tool_name == "move_ee_to_grounded_instance"
    assert first_call.args["_operation_candidate_id"] == first_id

    for target_error in (0.0319, 0.0317):
        agent._record_grounded_setup_outcomes(
            [first_call],
            [
                RecoveryToolResult(
                    tool_name="move_ee_to_grounded_instance",
                    success=True,
                    details={
                        "instance_id": "target_01",
                        "arm": "left",
                        "target_reached": False,
                        "target_observation_error_m": target_error,
                        "operation_candidate_id": first_id,
                        "operation_action_mode": "grasp",
                    },
                )
            ],
        )

    blocked = agent._blocked_grounded_setup_payload()
    assert blocked == [
        {
            "instance_id": "target_01",
            "arm": "left",
            "action_mode": "grasp",
            "failure_count": 2,
            "last_target_error_m": 0.0317,
            "status": "operation_candidate_subset_blocked",
            "blocked_candidate_count": 1,
            "available_candidate_count": 1,
            "runtime_will_select_next": True,
            "all_candidates_blocked": False,
        }
    ]
    next_call = agent._with_internal_recovery_context([raw_call])[0]
    assert next_call.tool_name == "move_ee_to_grounded_instance"
    assert next_call.args["_blocked_operation_candidate_ids"] == [first_id]
    assert next_call.args["_operation_candidate_id"] == second_id


def test_blocked_candidate_reopens_only_for_fresh_verified_geometry() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    candidate_id = "rgbd_volume:grasp:right:000"

    def grasp_candidate(x: float) -> dict:
        return {
            "candidate_id": candidate_id,
            "action_mode": "grasp",
            "arm": "right",
            "object_contact_pose": [x, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
            "tcp_pose": [x, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0],
            "ee_target_pose": [x, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
            "approach_pose": [x, 0.0, 0.28, 1.0, 0.0, 0.0, 0.0],
            "approach_direction": [0.0, 0.0, -1.0],
            "geometry_source": "rgbd_observed_volume_principal_axes",
        }

    def record_scene(item: dict, *, env_step: int, capture_id: int) -> None:
        agent.memory_store.record_scene_memory(
            {
                "instances": [
                    {
                        "instance_id": "track_0001",
                        "track_id": "track_0001",
                        "status": "visible",
                        "position_state": "current_verified",
                        "action_geometry_state": "verified",
                        "actionable": True,
                        "last_verified_step": env_step,
                        "operation_pose_candidates": [item],
                        "operation_pose_candidate_provenance": {
                            candidate_id: {
                                "geometry_observation_env_step": env_step,
                                "geometry_observation_capture_id": capture_id,
                            }
                        },
                    }
                ],
                "task_focus": {"target_instances": ["track_0001"]},
            }
        )

    raw_call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "track_0001",
            "arm": "right",
            "point_key": "grasp_world_m",
        },
    )
    old = grasp_candidate(0.10)
    record_scene(old, env_step=13, capture_id=20)
    frozen_call = agent._with_internal_recovery_context([raw_call])[0]
    for target_error in (0.0715, 0.0702):
        agent._record_grounded_setup_outcomes(
            [frozen_call],
            [
                RecoveryToolResult(
                    tool_name="move_ee_to_grounded_instance",
                    success=True,
                    details={
                        "instance_id": "track_0001",
                        "arm": "right",
                        "target_reached": False,
                        "target_observation_error_m": target_error,
                        "operation_candidate_id": candidate_id,
                        "operation_action_mode": "grasp",
                    },
                )
            ],
        )

    record_scene(grasp_candidate(0.10), env_step=80, capture_id=80)
    still_blocked = agent._with_internal_recovery_context([raw_call])[0]
    assert still_blocked.tool_name == "move_ee_to_grounded_instance"
    assert still_blocked.args["_blocked_operation_candidate_ids"] == [
        candidate_id
    ]
    assert "_operation_candidate_attempt" not in still_blocked.args

    record_scene(grasp_candidate(0.18), env_step=81, capture_id=81)
    reopened = agent._with_internal_recovery_context([raw_call])[0]
    assert reopened.tool_name == "move_ee_to_grounded_instance"
    assert reopened.args["_blocked_operation_candidate_ids"] == []
    assert any(
        entry.startswith("operation_candidate_revalidated:")
        for entry in agent.memory_store.state.working.recovery_history
    )


def test_missing_frozen_candidate_attempt_is_not_reconstructed_after_action() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "track_0001",
            "arm": "right",
            "point_key": "grasp_world_m",
            "_operation_action_mode": "grasp",
        },
    )
    result = RecoveryToolResult(
        tool_name="move_ee_to_grounded_instance",
        success=False,
        details={
            "instance_id": "track_0001",
            "arm": "right",
            "target_reached": False,
            "target_observation_error_m": 0.07,
            "operation_candidate_id": "rgbd_volume:grasp:right:000",
            "operation_action_mode": "grasp",
        },
    )

    agent._record_grounded_setup_outcomes([call], [result])
    agent._record_grounded_setup_outcomes([call], [result])

    assert agent._blocked_grounded_setup_payload() == []
    assert any(
        "missing_frozen_attempt" in entry
        for entry in agent.memory_store.state.working.recovery_history
    )


def test_temporarily_occluded_grounded_instance_cannot_execute_stale_pose() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "tracked",
                    "stability": "missing_current_frame",
                    "missing_steps": 2,
                    "approach_world_m": [0.1, 0.2, 0.9],
                }
            ],
            "task_focus": {"target_instances": ["target_01"], "tool_instances": []},
        }
    )
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={"instance_id": "target_01", "arm": "left", "point_key": "approach_world_m"},
    )

    guarded = agent._with_internal_recovery_context([call])

    assert guarded[0].tool_name == "reobserve_scene"
    assert "temporarily occluded" in guarded[0].args["_guard_reason"]


def test_memory_valid_occluded_instance_can_reach_safe_approach_only() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "tracked",
                    "stability": "missing_current_frame",
                    "position_state": "memory_valid",
                    "position_source": "retained_verified_memory",
                    "action_geometry_state": "verified",
                    "missing_steps": 2,
                    "approach_world_m": [0.1, 0.2, 0.9],
                    "grasp_world_m": [0.1, 0.2, 0.82],
                }
            ],
            "task_focus": {
                "target_instances": ["target_01"],
                "tool_instances": [],
            },
        }
    )

    approach = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "approach_world_m",
            "_operation_action_mode": "grasp",
        },
    )
    guarded_approach = agent._with_internal_recovery_context(
        [approach]
    )
    assert [
        call.tool_name for call in guarded_approach
    ] == ["move_ee_to_grounded_instance"]
    assert (
        guarded_approach[0].args[
            "_runtime_memory_valid_approach"
        ]
        is True
    )

    final_grasp = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "grasp_world_m",
            "_operation_action_mode": "grasp",
        },
    )
    guarded_grasp = agent._with_internal_recovery_context(
        [final_grasp]
    )
    assert guarded_grasp[-1].tool_name == "reobserve_scene"
    assert "temporarily occluded" in guarded_grasp[-1].args[
        "_guard_reason"
    ]


def test_memory_valid_final_grasp_switch_allows_only_exact_bound_candidate() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                allow_memory_valid_final_grounded_action=True,
            )
        )
    )
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    candidate = {
        "candidate_id": "rgbd:grasp:left:000",
        "arm": "left",
        "action_mode": "grasp",
        "object_contact_pose": [0.1, 0.2, 0.82, 1.0, 0.0, 0.0, 0.0],
        "tcp_pose": [0.1, 0.2, 0.82, 1.0, 0.0, 0.0, 0.0],
        "ee_target_pose": [0.1, 0.2, 0.82, 1.0, 0.0, 0.0, 0.0],
        "approach_pose": [0.1, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "tracked",
                    "stability": "missing_current_frame",
                    "position_state": "memory_valid",
                    "position_source": "retained_verified_memory",
                    "action_geometry_state": "verified",
                    "operation_pose_candidates": [candidate],
                }
            ],
            "task_focus": {"target_instances": ["target_01"]},
        }
    )
    exact = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "grasp_world_m",
            "offset_xyz": [0.0, 0.0, 0.0],
        },
    )

    guarded = agent._with_internal_recovery_context([exact])

    assert [call.tool_name for call in guarded] == [
        "move_ee_to_grounded_instance"
    ]
    assert guarded[0].args[
        "_runtime_memory_valid_final_action"
    ] is True
    assert guarded[0].args["_operation_candidate_id"] == candidate[
        "candidate_id"
    ]

    shifted = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "grasp_world_m",
            "offset_xyz": [0.01, 0.0, 0.0],
        },
    )
    blocked = agent._with_internal_recovery_context([shifted])
    assert [call.tool_name for call in blocked] == ["reobserve_scene"]


def test_disabling_automatic_self_occlusion_clearance_does_not_inject_motion() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                enable_automatic_self_occlusion_visual_clearance=False,
            )
        )
    )
    agent.latest_snapshot = make_snapshot()
    instance = {
        "instance_id": "target_01",
        "track_id": "track_0001",
        "status": "tracked",
        "stability": "missing_current_frame",
        "position_state": "",
        "action_geometry_state": "verified",
        "approach_world_m": [0.0, 0.0, 0.0],
        "grasp_world_m": [0.0, 0.0, -0.08],
    }
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "grasp_world_m",
            "_operation_action_mode": "grasp",
            "_scene_memory": {"instances": [instance]},
        },
    )

    guarded = agent._guard_occluded_grounded_instance(call)

    assert [item.tool_name for item in guarded] == ["reobserve_scene"]
    assert all(
        item.args.get("_runtime_visual_clearance") is not True
        for item in guarded
    )


def test_motion_uncertain_position_blocks_even_safe_approach() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "tracked",
                    "stability": "missing_current_frame",
                    "position_state": "motion_uncertain",
                    "action_geometry_state": "verified",
                    "approach_world_m": [0.1, 0.2, 0.9],
                }
            ],
            "task_focus": {
                "target_instances": ["target_01"],
                "tool_instances": [],
            },
        }
    )
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "approach_world_m",
        },
    )

    guarded = agent._with_internal_recovery_context([call])

    assert [item.tool_name for item in guarded] == [
        "reobserve_scene"
    ]
    assert "may have moved" in guarded[0].args["_guard_reason"]


def test_track_reference_cannot_bypass_occluded_grounded_instance_guard() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "internal_instance_01",
                    "track_id": "track_0001",
                    "status": "tracked",
                    "stability": "missing_current_frame",
                    "missing_steps": 1,
                    "approach_world_m": [0.1, 0.2, 0.9],
                }
            ],
            "task_focus": {"target_instances": ["track_0001"], "tool_instances": []},
        }
    )
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={"instance_ref": "track_0001", "arm": "left", "point_key": "approach_world_m"},
    )

    guarded = agent._with_internal_recovery_context([call])

    assert guarded[0].tool_name == "reobserve_scene"
    assert "temporarily occluded" in guarded[0].args["_guard_reason"]


def test_geometry_inconsistent_grounded_instance_cannot_execute_stale_pose() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "visible",
                    "stability": "geometry_inconsistent_current_frame",
                    "approach_world_m": [0.1, 0.2, 0.9],
                }
            ],
            "task_focus": {"target_instances": ["target_01"], "tool_instances": []},
        }
    )
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={"instance_id": "target_01", "arm": "left", "point_key": "approach_world_m"},
    )

    guarded = agent._with_internal_recovery_context(
        [
            call,
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "negative", "distance": 0.01},
            ),
        ]
    )

    assert [item.tool_name for item in guarded] == ["reobserve_scene"]
    assert "inconsistent current-frame geometry" in guarded[0].args["_guard_reason"]


def _repair_pending_policy_agent(
    policy: str,
    *,
    position_state: str = "current_verified",
) -> ImgAgent:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                action_geometry_repair_pending_policy=policy,
            )
        )
    )
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.left_endpose[:] = np.asarray(
        [0.0, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_recovery_required",
            "held_instance_id": "target_01",
            "released_instance_id": "target_01",
            "holding_confirmed": False,
            "updated_step": 0,
        }
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "visible",
                    "stability": "relocation_geometry_pending",
                    "position_state": position_state,
                    "action_geometry_state": "relocation_pending",
                    "latest_world_m": [0.10, 0.20, 0.75],
                    "world_m": [0.10, 0.20, 0.75],
                    "quality": {
                        "world_z_max_m": 0.78,
                        "world_extent_m": [0.04, 0.04, 0.06],
                    },
                    # Deliberately stale: safe_motion must never execute these.
                    "approach_world_m": [9.0, 9.0, 9.0],
                    "grasp_world_m": [8.0, 8.0, 8.0],
                }
            ],
            "task_focus": {
                "target_instances": ["target_01"],
                "tool_instances": [],
            },
        }
    )
    return agent


def test_relocation_geometry_pending_instance_cannot_execute_unconfirmed_pose() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "visible",
                    "stability": "relocation_geometry_pending",
                    "action_geometry_state": "relocation_pending",
                    "world_m": [0.1, 0.2, 0.7],
                    "approach_world_m": [0.1, 0.2, 0.9],
                }
            ],
            "task_focus": {
                "target_instances": ["target_01"],
                "tool_instances": [],
            },
        }
    )
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "approach_world_m",
        },
    )

    guarded = agent._with_internal_recovery_context(
        [
            call,
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "left"},
            ),
        ]
    )

    assert [item.tool_name for item in guarded] == [
        "reobserve_scene"
    ]
    assert (
        "relocated action geometry is awaiting confirmation"
        in guarded[0].args["_guard_reason"]
    )


def test_relocation_pending_safe_motion_uses_runtime_high_pose_and_truncates_close() -> None:
    agent = _repair_pending_policy_agent("safe_motion")
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "grasp_world_m",
        },
    )

    guarded = agent._with_internal_recovery_context(
        [
            call,
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "left"},
            ),
        ]
    )

    assert [item.tool_name for item in guarded] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    safe = guarded[0]
    assert safe.args[
        "_runtime_action_geometry_repair_safe_motion"
    ] is True
    assert safe.args["target_pose"][:3] == pytest.approx(
        [0.0, 0.0, 0.98]
    )
    assert max(abs(value) for value in safe.args["target_pose"][:3]) < 2.0
    assert all(item.tool_name != "close_gripper" for item in guarded)


def test_relocation_pending_safe_motion_blocks_standalone_close_contact_and_spoofed_pose() -> None:
    agent = _repair_pending_policy_agent("safe_motion")

    for call in (
        RecoveryToolCall(
            tool_name="close_gripper",
            args={"arm": "left"},
        ),
        RecoveryToolCall(
            tool_name="contact_displace",
            args={
                "arm": "left",
                "axis": "z",
                "direction": "negative",
                "distance": 0.01,
            },
        ),
        RecoveryToolCall(
            tool_name="move_ee_to_pose",
            args={
                "arm": "left",
                "target_pose": [
                    0.10,
                    0.20,
                    0.76,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                # Planner-authored private markers are stripped.
                "_runtime_visual_clearance": True,
            },
        ),
    ):
        guarded = agent._with_internal_recovery_context([call])
        assert [item.tool_name for item in guarded] == [
            "reobserve_scene"
        ]


def test_relocation_pending_safe_motion_allows_only_retreat_that_increases_clearance() -> None:
    agent = _repair_pending_policy_agent("safe_motion")
    away = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="retreat_arm",
                args={
                    "arm": "left",
                    "axis": "x",
                    "direction": "negative",
                    "distance": 0.03,
                },
            )
        ]
    )
    toward = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="retreat_arm",
                args={
                    "arm": "left",
                    "axis": "x",
                    "direction": "positive",
                    "distance": 0.03,
                },
            )
        ]
    )

    assert [item.tool_name for item in away] == ["retreat_arm"]
    assert away[0].args[
        "_runtime_action_geometry_repair_retreat"
    ] is True
    assert [item.tool_name for item in toward] == [
        "reobserve_scene"
    ]


def test_relocation_pending_disabled_bypasses_only_pending_guard() -> None:
    agent = _repair_pending_policy_agent("disabled")
    grasp = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_id": "target_01",
            "arm": "left",
            "point_key": "grasp_world_m",
        },
    )
    guarded = agent._with_internal_recovery_context(
        [
            grasp,
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "left"},
            ),
        ]
    )

    # Disabling this one guard preserves the requested grounded motion.  A
    # separate grasp-close contract may still append its own observation.
    assert [item.tool_name for item in guarded[:2]] == [
        "move_ee_to_grounded_instance",
        "close_gripper",
    ]
    assert guarded[0].args[
        "_runtime_action_geometry_repair_pending_bypass"
    ] is True

    motion_uncertain = _repair_pending_policy_agent(
        "disabled",
        position_state="motion_uncertain",
    )
    still_blocked = motion_uncertain._with_internal_recovery_context(
        [grasp]
    )
    assert [item.tool_name for item in still_blocked] == [
        "reobserve_scene"
    ]
    assert "may have moved" in still_blocked[0].args[
        "_guard_reason"
    ]


def test_relocation_pending_action_state_cannot_use_occlusion_lease_bypass() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "tracked",
                    # Missing-frame bookkeeping must not hide the independent
                    # post-release geometry quarantine.
                    "stability": "missing_current_frame",
                    "action_geometry_state": "relocation_pending",
                    "world_m": [0.1, 0.2, 0.7],
                    "approach_world_m": [0.1, 0.2, 0.9],
                }
            ],
            "task_focus": {
                "target_instances": ["target_01"],
                "tool_instances": [],
            },
        }
    )
    scene_memory = (
        agent.memory_store.state.working.scene_memory
    )
    guarded = agent._guard_occluded_grounded_instance(
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={
                "instance_id": "target_01",
                "arm": "left",
                "point_key": "approach_world_m",
                "_runtime_occlusion_geometry_lease": True,
                "_scene_memory": scene_memory,
            },
        )
    )

    assert [item.tool_name for item in guarded] == [
        "reobserve_scene"
    ]
    assert (
        "awaiting confirmation from two independent clean observations"
        in guarded[0].args["_guard_reason"]
    )


def test_identity_repair_expired_action_state_requires_replan_and_blocks_stale_pose() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    expired_instance = {
        "instance_id": "target_01",
        "track_id": "track_0001",
        "status": "tracked",
        "stability": "identity_repair_expired",
        "action_geometry_state": "identity_repair_expired",
        "world_m": [0.1, 0.2, 0.7],
        # Deliberately retain a stale pose in this fixture: the action-state
        # guard must fail closed even if a malformed caller supplies one.
        "approach_world_m": [0.1, 0.2, 0.9],
    }
    scene_memory = {
        "instances": [expired_instance],
        "task_focus": {
            "target_instances": ["target_01"],
            "tool_instances": [],
        },
    }
    agent.memory_store.record_scene_memory(scene_memory)

    catalog = agent._scene_instance_catalog_for_query_payload()
    assert len(catalog) == 1
    assert catalog[0]["recovery_binding_only"] is True
    assert catalog[0]["grounded_execution_allowed"] is False
    assert (
        catalog[0]["action_geometry_state"]
        == "identity_repair_expired"
    )

    guarded = agent._guard_occluded_grounded_instance(
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={
                "instance_id": "target_01",
                "arm": "left",
                "point_key": "approach_world_m",
                "_runtime_partial_approach_continuation": True,
                "_runtime_occlusion_geometry_lease": True,
                "_scene_memory": scene_memory,
            },
        )
    )

    assert [item.tool_name for item in guarded] == [
        "reobserve_scene"
    ]
    assert "repair lease expired" in guarded[0].args[
        "_guard_reason"
    ]
    assert "replan identity recovery" in guarded[0].args[
        "_guard_reason"
    ]


def test_tracker_owned_identity_repair_lease_is_not_reinjected_by_agent() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "visible",
                    "stability": "relocation_geometry_pending",
                    "action_geometry_state": "relocation_pending",
                    "world_m": [0.028504, -0.010821, 0.771351],
                    "identity_repair_lease": {
                        "state": "active",
                        "instance_ref": "track_0001",
                        "target_world_m": [
                            0.039454,
                            -0.007076,
                            0.772605,
                        ],
                        "tolerance_m": 0.041899,
                        "started_step": 22,
                        "expires_step": 42,
                        "observation_attempts": 1,
                        "max_observation_attempts": 20,
                        "source": (
                            "post_release_action_geometry_repair"
                        ),
                    },
                }
            ],
            "task_focus": {
                "target_instances": ["target_01"],
                "tool_instances": [],
            },
        }
    )

    leases = (
        agent._pending_release_identity_relocation_leases()
    )

    # SceneMemoryTracker owns and advances an already-active repair lease.
    # Re-emitting it from cleared manipulation state can let stale coordinates
    # overwrite a same-update motion invalidation.
    assert leases == []


def test_reached_approach_creates_one_shot_lease_for_complete_occluded_contact_cycle() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(0.0)
    snapshot.step_count = 4
    snapshot.left_endpose = np.asarray([0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    snapshot.head_cam2world_gl = np.eye(4, dtype=np.float64)
    snapshot.head_cam2world_gl[:3, 3] = [0.1, 0.0, 1.0]
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="perform repeated grounded actions",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="perform the current grounded action",
        skill_spec=skill_spec,
    )
    visible_instance = {
        "instance_id": "internal_instance_01",
        "track_id": "track_0001",
        "status": "visible",
        "stability": "stable",
        "camera": "head",
        "approach_world_m": [0.1, 0.0, 0.2],
        "contact_world_m": [0.1, 0.0, 0.1],
        "quality": {"world_extent_m": [0.06, 0.04, 0.02]},
    }
    missing_instance = dict(visible_instance)
    missing_instance.update(
        {"status": "tracked", "stability": "missing_current_frame", "missing_steps": 4}
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [missing_instance],
            "task_focus": {"target_instances": ["track_0001"]},
        }
    )
    reached_approach = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_ref": "track_0001",
            "arm": "left",
            "point_key": "approach_world_m",
            "_scene_memory": {
                "instances": [visible_instance],
                "task_focus": {"target_instances": ["track_0001"]},
            },
        },
    )
    reached_result = RecoveryToolResult(
        tool_name="move_ee_to_grounded_instance",
        success=True,
        details={
            "arm": "left",
            "instance_id": "track_0001",
            "target_reached": True,
            "observed_pose": [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
        },
    )
    agent._record_grounded_setup_outcomes([reached_approach], [reached_result])

    calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={
                "instance_ref": "track_0001",
                "arm": "left",
                "point_key": "contact_world_m",
                "max_translation": 0.12,
            },
        ),
        RecoveryToolCall(
            tool_name="contact_displace",
            args={
                "arm": "left",
                "axis": "z",
                "direction": "negative",
                "distance": 0.012,
                "steps": 2,
                "complete_transient_cycle": True,
            },
        ),
        RecoveryToolCall(tool_name="reobserve_scene", args={}),
    ]

    malformed = agent._with_internal_recovery_context(
        [
            *calls[:-1],
            RecoveryToolCall(tool_name="retreat_arm", args={"arm": "right"}),
            calls[-1],
        ]
    )
    assert agent._grounded_geometry_leases == {}
    assert all(
        call.args.get("_runtime_occlusion_geometry_lease") is not True
        for call in malformed
    )

    agent._record_grounded_setup_outcomes([reached_approach], [reached_result])
    malformed_prefix = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(tool_name="retreat_arm", args={"arm": "right"}),
            *calls,
        ]
    )
    assert agent._grounded_geometry_leases == {}
    assert all(
        call.args.get("_runtime_occlusion_geometry_lease") is not True
        for call in malformed_prefix
    )

    agent._record_grounded_setup_outcomes([reached_approach], [reached_result])
    offset_setup_args = dict(calls[0].args)
    offset_setup_args["offset_xyz"] = [0.01, 0.0, 0.0]
    malformed_offset = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args=offset_setup_args,
            ),
            *calls[1:],
        ]
    )
    assert agent._grounded_geometry_leases == {}
    assert all(
        call.args.get("_runtime_occlusion_geometry_lease") is not True
        for call in malformed_offset
    )

    agent._record_grounded_setup_outcomes([reached_approach], [reached_result])
    preserve_setup_args = dict(calls[0].args)
    preserve_setup_args["preserve_height"] = True
    malformed_preserve_setup = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args=preserve_setup_args,
            ),
            *calls[1:],
        ]
    )
    assert agent._grounded_geometry_leases == {}
    assert all(
        call.args.get("_runtime_occlusion_geometry_lease") is not True
        for call in malformed_preserve_setup
    )

    agent._record_grounded_setup_outcomes([reached_approach], [reached_result])
    malformed_release = agent._with_internal_recovery_context(
        [
            calls[0],
            calls[1],
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "instance_ref": "track_0001",
                    "arm": "left",
                    "point_key": "approach_world_m",
                    "offset_xyz": [0.0, 0.0, 0.0],
                    "preserve_height": True,
                    "max_translation": 0.12,
                },
            ),
            calls[-1],
        ]
    )
    assert agent._grounded_geometry_leases == {}
    assert all(
        call.args.get("_runtime_occlusion_geometry_lease") is not True
        for call in malformed_release
    )

    agent._record_grounded_setup_outcomes([reached_approach], [reached_result])
    leased = agent._with_internal_recovery_context(calls)

    assert [call.tool_name for call in leased] == [
        "move_ee_to_grounded_instance",
        "contact_displace",
        "move_ee_to_grounded_instance",
        "move_ee_to_grounded_instance",
        "reobserve_scene",
    ]
    grounded = [call for call in leased if call.tool_name == "move_ee_to_grounded_instance"]
    assert grounded
    assert all(call.args.get("_runtime_occlusion_geometry_lease") is True for call in grounded)
    assert agent._grounded_geometry_leases == {}

    guarded_again = agent._with_internal_recovery_context(calls)
    assert [call.tool_name for call in guarded_again] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert guarded_again[0].args["_runtime_visual_clearance"] is True
    assert all(
        call.args.get("_runtime_occlusion_geometry_lease") is not True
        for call in guarded_again
    )


def test_reached_grasp_approach_creates_one_shot_lease_for_self_occluded_grasp() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.step_count = 7
    snapshot.left_endpose = np.asarray(
        [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    snapshot.head_cam2world_gl = np.eye(4, dtype=np.float64)
    snapshot.head_cam2world_gl[:3, 3] = [0.1, 0.0, 1.0]
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="grasp the grounded cube",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="grasp the grounded cube",
        skill_spec=skill_spec,
    )
    candidate = {
        "candidate_id": "rgbd_surface:grasp:left:000",
        "source_candidate_index": 0,
        "action_mode": "grasp",
        "arm": "left",
        "object_contact_pose": [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
        "tcp_pose": [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
        "ee_target_pose": [0.1, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0],
        "approach_pose": [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
        "approach_direction": [0.0, 0.0, -1.0],
        "geometry_source": "rgbd_surface_normal_principal_axes",
        "priority": 0.0,
    }
    visible_instance = {
        "instance_id": "track_0001",
        "track_id": "track_0001",
        "status": "visible",
        "stability": "stable",
        "camera": "head",
        "quality": {"world_extent_m": [0.04, 0.04, 0.04]},
        "operation_pose_candidates": [candidate],
    }
    inconsistent_instance = dict(visible_instance)
    inconsistent_instance.update(
        {
            "status": "tracked",
            "stability": "geometry_inconsistent_current_frame",
        }
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [inconsistent_instance],
            "task_focus": {"target_instances": ["track_0001"]},
        }
    )
    reached_approach = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_ref": "track_0001",
            "arm": "left",
            "point_key": "approach_world_m",
            "_operation_action_mode": "grasp",
            "_operation_candidate_id": candidate["candidate_id"],
            "_scene_memory": {
                "instances": [visible_instance],
                "task_focus": {"target_instances": ["track_0001"]},
            },
        },
    )
    reached_result = RecoveryToolResult(
        tool_name="move_ee_to_grounded_instance",
        success=True,
        details={
            "arm": "left",
            "instance_id": "track_0001",
            "target_reached": True,
            "observed_pose": [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
            "operation_action_mode": "grasp",
            "operation_candidate_id": candidate["candidate_id"],
        },
    )
    agent._record_grounded_setup_outcomes(
        [reached_approach],
        [reached_result],
    )

    calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={
                "instance_ref": "track_0001",
                "arm": "left",
                "point_key": "grasp_world_m",
                "max_translation": 0.09,
                "steps": 3,
            },
        ),
        RecoveryToolCall(tool_name="close_gripper", args={"arm": "left"}),
        RecoveryToolCall(tool_name="reobserve_scene", args={}),
    ]
    leased = agent._with_internal_recovery_context(calls)

    assert [call.tool_name for call in leased] == [
        "move_ee_to_grounded_instance",
        "close_gripper",
        "reobserve_scene",
    ]
    assert leased[0].args["_runtime_occlusion_geometry_lease"] is True
    assert leased[0].args["_operation_candidate_id"] == candidate["candidate_id"]
    assert agent._grounded_geometry_leases == {}

    replayed = agent._with_internal_recovery_context(calls)
    assert [call.tool_name for call in replayed] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert replayed[0].args["_runtime_visual_clearance"] is True
    assert all(
        call.args.get("_runtime_occlusion_geometry_lease") is not True
        for call in replayed
    )

    agent._record_grounded_setup_outcomes(
        [reached_approach],
        [reached_result],
    )
    snapshot.step_count = 8
    expired = agent._with_internal_recovery_context(calls)
    assert [call.tool_name for call in expired] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert agent._grounded_geometry_leases == {}

    agent._record_grounded_setup_outcomes(
        [reached_approach],
        [reached_result],
    )
    lease_key = next(iter(agent._grounded_geometry_leases))
    agent._grounded_geometry_leases[lease_key][
        "operation_candidate_id"
    ] = "different-candidate"
    mismatched = agent._with_internal_recovery_context(calls)
    assert [call.tool_name for call in mismatched] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert agent._grounded_geometry_leases == {}

    agent._record_grounded_setup_outcomes(
        [reached_approach],
        [reached_result],
    )
    malformed = agent._with_internal_recovery_context(
        [
            *calls[:-1],
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "left", "distance": 0.02},
            ),
            calls[-1],
        ]
    )
    assert [call.tool_name for call in malformed] == [
        "move_ee_to_grounded_instance",
        "close_gripper",
        "reobserve_scene",
    ]
    assert all(call.tool_name != "lift_ee" for call in malformed)
    assert malformed[1].args["_runtime_grasp_close_boundary"] is True
    assert agent._grounded_geometry_leases == {}


def _partial_grounded_approach_fixture():
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.step_count = 11
    snapshot.left_endpose = np.asarray(
        [0.1, 0.0, 0.225, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    snapshot.head_cam2world_gl = np.eye(4, dtype=np.float64)
    snapshot.head_cam2world_gl[:3, 3] = [0.1, 0.0, 1.0]
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="grasp the grounded cube",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec(
        "monitored-subtask-execution"
    )
    agent.memory_store.start_or_replace_skill(
        subtask_text="grasp the grounded cube",
        skill_spec=skill_spec,
    )
    candidate = {
        "candidate_id": "rgbd_volume:grasp:left:000",
        "source_candidate_index": 0,
        "action_mode": "grasp",
        "arm": "left",
        "object_contact_pose": [
            0.1,
            0.0,
            0.1,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        "tcp_pose": [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
        "ee_target_pose": [
            0.1,
            0.0,
            0.12,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        "approach_pose": [
            0.1,
            0.0,
            0.2,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        "approach_direction": [0.0, 0.0, -1.0],
        "geometry_source": "rgbd_observed_volume_principal_axes",
        "priority": 0.0,
    }
    visible_instance = {
        "instance_id": "track_0001",
        "track_id": "track_0001",
        "status": "visible",
        "stability": "stable",
        "camera": "head",
        "quality": {"world_extent_m": [0.04, 0.04, 0.04]},
        "operation_pose_candidates": [candidate],
    }
    missing_instance = dict(visible_instance)
    missing_instance.update(
        {
            "status": "tracked",
            "stability": "missing_current_frame",
            "missing_steps": 1,
        }
    )
    post_scene = {
        "instances": [missing_instance],
        "task_focus": {"target_instances": ["track_0001"]},
    }
    agent.memory_store.record_scene_memory(post_scene)
    partial_call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "instance_ref": "track_0001",
            "arm": "left",
            "point_key": "approach_world_m",
            "max_translation": 0.08,
            "steps": 4,
            "_operation_action_mode": "grasp",
            "_operation_candidate_id": candidate["candidate_id"],
            "_scene_memory": {
                "instances": [visible_instance],
                "task_focus": {
                    "target_instances": ["track_0001"]
                },
            },
        },
    )
    partial_result = RecoveryToolResult(
        tool_name="move_ee_to_grounded_instance",
        success=True,
        details={
            "arm": "left",
            "instance_id": "track_0001",
            "target_pose": [
                0.1,
                0.0,
                0.2,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "observed_pose": [
                0.1,
                0.0,
                0.225,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "target_observation_error_m": 0.025,
            "target_reached": False,
            "target_reached_tolerance_m": 0.01,
            "target_error_history_m": [
                0.265,
                0.185,
                0.105,
                0.025,
            ],
            "max_translation": 0.08,
            "steps": 4,
            "executed_steps": 4,
            "operation_action_mode": "grasp",
            "operation_candidate_id": candidate["candidate_id"],
        },
    )
    return (
        agent,
        snapshot,
        candidate,
        partial_call,
        partial_result,
    )


def test_partial_self_occluded_approach_continues_then_upgrades_to_grasp_lease() -> None:
    (
        agent,
        snapshot,
        candidate,
        partial_call,
        partial_result,
    ) = _partial_grounded_approach_fixture()
    reobserve = RecoveryToolCall(
        tool_name="reobserve_scene",
        args={},
    )
    agent._record_grounded_setup_outcomes(
        [partial_call, reobserve],
        [partial_result],
    )

    assert len(agent._partial_grounded_approach_leases) == 1
    partial_lease = next(
        iter(agent._partial_grounded_approach_leases.values())
    )
    assert partial_lease["residual_m"] == 0.025
    assert (
        partial_lease["operation_candidate_id"]
        == candidate["candidate_id"]
    )

    continuation = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "instance_ref": "track_0001",
                    "arm": "left",
                    "point_key": "approach_world_m",
                    "max_translation": 0.03,
                    "steps": 3,
                },
            ),
            reobserve,
        ]
    )

    assert [call.tool_name for call in continuation] == [
        "move_ee_to_grounded_instance",
        "reobserve_scene",
    ]
    assert (
        continuation[0].args[
            "_runtime_partial_approach_continuation"
        ]
        is True
    )
    assert (
        continuation[0].args["_operation_candidate_id"]
        == candidate["candidate_id"]
    )
    assert agent._partial_grounded_approach_leases == {}

    snapshot.left_endpose = np.asarray(
        [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    reached_result = RecoveryToolResult(
        tool_name="move_ee_to_grounded_instance",
        success=True,
        details={
            "arm": "left",
            "instance_id": "track_0001",
            "target_reached": True,
            "target_pose": [
                0.1,
                0.0,
                0.2,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "observed_pose": [
                0.1,
                0.0,
                0.2,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "operation_action_mode": "grasp",
            "operation_candidate_id": candidate["candidate_id"],
        },
    )
    agent._record_grounded_setup_outcomes(
        continuation,
        [reached_result],
    )

    assert len(agent._grounded_geometry_leases) == 1
    grasp_calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "instance_ref": "track_0001",
                    "arm": "left",
                    "point_key": "grasp_world_m",
                    "max_translation": 0.09,
                    "steps": 3,
                },
            ),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "left"},
            ),
            reobserve,
        ]
    )
    assert [call.tool_name for call in grasp_calls] == [
        "move_ee_to_grounded_instance",
        "close_gripper",
        "reobserve_scene",
    ]
    assert (
        grasp_calls[0].args["_runtime_occlusion_geometry_lease"]
        is True
    )
    assert agent._grounded_geometry_leases == {}


def test_partial_approach_lease_rejects_nonmonotonic_or_overreaching_use() -> None:
    (
        agent,
        snapshot,
        _candidate,
        partial_call,
        partial_result,
    ) = _partial_grounded_approach_fixture()
    reobserve = RecoveryToolCall(
        tool_name="reobserve_scene",
        args={},
    )
    partial_result.details["target_error_history_m"] = [
        0.105,
        0.106,
        0.025,
    ]
    partial_result.details["executed_steps"] = 3
    agent._record_grounded_setup_outcomes(
        [partial_call, reobserve],
        [partial_result],
    )
    assert agent._partial_grounded_approach_leases == {}

    (
        agent,
        snapshot,
        _candidate,
        partial_call,
        partial_result,
    ) = _partial_grounded_approach_fixture()
    agent._record_grounded_setup_outcomes(
        [partial_call, reobserve],
        [partial_result],
    )
    unauthorized_grasp = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "instance_ref": "track_0001",
                    "arm": "left",
                    "point_key": "grasp_world_m",
                    "max_translation": 0.09,
                    "steps": 3,
                },
            ),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "left"},
            ),
            reobserve,
        ]
    )
    assert agent._partial_grounded_approach_leases == {}
    assert all(
        call.args.get(
            "_runtime_partial_approach_continuation"
        )
        is not True
        for call in unauthorized_grasp
    )
    assert not any(
        call.tool_name == "close_gripper"
        for call in unauthorized_grasp
    )

    (
        agent,
        snapshot,
        _candidate,
        partial_call,
        partial_result,
    ) = _partial_grounded_approach_fixture()
    agent._record_grounded_setup_outcomes(
        [partial_call, reobserve],
        [partial_result],
    )
    snapshot.step_count = 12
    expired = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "instance_ref": "track_0001",
                    "arm": "left",
                    "point_key": "approach_world_m",
                    "max_translation": 0.03,
                    "steps": 3,
                },
            ),
            reobserve,
        ]
    )
    assert agent._partial_grounded_approach_leases == {}
    assert [call.tool_name for call in expired] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]


def test_recovery_observation_handoff_is_one_shot_and_step_bound() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    task_env = type("TaskEnv", (), {"take_action_cnt": 7})()
    raw = {"observation": {"head_camera": {"rgb": "sentinel"}}}
    agent._pending_recovery_observation = {"step_count": 7, "raw": raw}

    assert agent.consume_recovery_observation(task_env) == raw
    assert agent.consume_recovery_observation(task_env) is None

    agent._pending_recovery_observation = {"step_count": 6, "raw": raw}
    assert agent.consume_recovery_observation(task_env) is None
    assert agent._pending_recovery_observation is None


def test_recovery_observation_is_retained_across_planner_only_bootstrap() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "perform repeated grounded actions"
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    snapshot = make_snapshot()
    snapshot.step_count = 4
    raw = dict(snapshot.raw)
    task_env = type("TaskEnv", (), {"take_action_cnt": 4})()
    with (
        mock.patch(
            "policy.roboharn_evo.agent.core.img_agent.RMBenchEnvAdapter.from_env",
            return_value=snapshot,
        ),
        mock.patch.object(agent, "_maybe_bootstrap_debug_recovery_with_planner", return_value=True),
    ):
        agent.run_step(task_env=task_env, observation=raw)

    assert agent.consume_recovery_observation(task_env) == raw


def test_runtime_scene_context_overrides_planner_private_args() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    trusted_scene = {
        "instances": [
            {
                "instance_id": "trusted_target",
                "track_id": "trusted_target",
                "status": "visible",
                "stability": "stable",
                "approach_world_m": [0.1, 0.2, 0.3],
            }
        ],
        "task_focus": {"target_instances": ["trusted_target"]},
    }
    agent.memory_store.record_scene_memory(trusted_scene)
    agent.memory_store.record_observation_preprocess({"source": "runtime"})
    spoofed_scene = {
        "instances": [
            {
                "instance_id": "spoofed_target",
                "status": "tracked",
                "stability": "missing_current_frame",
                "approach_world_m": [9.0, 9.0, 9.0],
            }
        ]
    }

    enriched = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "instance_ref": "trusted_target",
                    "arm": "left",
                    "point_key": "approach_world_m",
                    "_scene_memory": spoofed_scene,
                    "_observation_preprocess": {"source": "planner"},
                },
            )
        ]
    )

    assert len(enriched) == 1
    assert enriched[0].tool_name == "move_ee_to_grounded_instance"
    assert enriched[0].args["_scene_memory"] == trusted_scene
    assert enriched[0].args["_observation_preprocess"] == {"source": "runtime"}


def test_retreat_can_precede_guarded_reobserve_but_stale_contact_is_truncated() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic manipulation task",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "target_01",
                    "track_id": "track_0001",
                    "status": "tracked",
                    "stability": "geometry_inconsistent_current_frame",
                    "approach_world_m": [0.1, 0.2, 0.9],
                }
            ],
            "task_focus": {"target_instances": ["target_01"], "tool_instances": []},
        }
    )

    guarded = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(tool_name="retreat_arm", args={"arm": "left", "distance": 0.05}),
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"instance_id": "target_01", "arm": "left", "point_key": "approach_world_m"},
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={"arm": "left", "axis": "z", "direction": "negative", "distance": 0.01},
            ),
        ]
    )

    assert [item.tool_name for item in guarded] == ["retreat_arm", "reobserve_scene"]
    assert "inconsistent current-frame geometry" in guarded[1].args["_guard_reason"]


def test_pure_tool_control_unexpected_abort_preserves_active_subtask() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="paper pure tool-control baseline",
    )

    agent._continue_debug_recovery_pure_control(
        preferred_action="abort",
        signal=signal,
        reason="paper pure tool-control baseline",
    )

    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is not None
    assert "planner selected step" not in agent.memory_store.state.task.failed_skills
    assert agent.memory_store.state.monitor.phase == "monitoring"
    assert agent.memory_store.state.monitor.status == "rollout_active"
    assert "pure_tool_control_unexpected_abort_deferred" in agent.memory_store.state.working.recovery_history


def test_pure_tool_control_rejects_finish_without_env_success() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)

    agent.apply_control_decision(
        {
            "action_mode": "finish",
            "note": "planner thought the task was complete",
            "memory_text": "finish requested",
        }
    )

    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is not None
    assert agent.memory_store.state.monitor.phase == "reasoning"
    assert agent.memory_store.state.monitor.status == "needs_reasoning"
    assert "pure_tool_control_finish_rejected" in agent.memory_store.state.working.recovery_history


def test_pure_tool_control_stops_before_exceeding_control_turn_budget() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_control_turns=8,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent._pure_tool_control_control_turns = 8

    assert agent._stop_for_pure_tool_control_turn_budget() is True

    status = agent.current_status()
    assert status["terminal_failure"] is True
    assert status["terminal_failure_reason"] == "pure_tool_control_max_control_turns_exhausted:8"
    assert status["control_turn_count"] == 8
    assert status["max_control_turns"] == 8
    assert agent.memory_store.state.monitor.status == "rollout_failed"
    assert "pure_tool_control_max_control_turns_exhausted:8" in agent.memory_store.state.working.recovery_history


def test_progress_aware_budget_allows_more_than_eight_total_turns_but_bounds_stall() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_control_turns=64,
                pure_tool_control_max_no_progress_control_turns=4,
            )
        )
    )
    agent.current_instruction = "perform repeated grounded actions"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent._pure_tool_control_control_turns = 12
    agent._pure_tool_control_no_progress_control_turns = 3

    assert agent._stop_for_pure_tool_control_turn_budget() is False

    agent._pure_tool_control_no_progress_control_turns = 4
    assert agent._stop_for_pure_tool_control_turn_budget() is True
    status = agent.current_status()
    assert status["terminal_failure_reason"] == (
        "pure_tool_control_max_no_progress_control_turns_exhausted:4"
    )
    assert status["control_turn_count"] == 12
    assert status["no_progress_control_turn_count"] == 4
    assert status["max_no_progress_control_turns"] == 4


def test_verified_subtask_completion_resets_no_progress_turn_budget() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_control_turns=64,
                pure_tool_control_max_no_progress_control_turns=4,
            )
        )
    )
    agent.current_instruction = "perform repeated grounded actions"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="perform one grounded action",
        skill_spec=skill_spec,
    )
    agent._pure_tool_control_no_progress_control_turns = 3
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="fresh verifier evidence",
    )
    agent._credit_pure_tool_control_verified_progress(
        action_effect={
            "effect_verified": "true",
            "effect_type": "move",
            "subtask_status": "completed",
        },
        calls=[
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "right"},
            )
        ],
    )

    agent._continue_debug_recovery_pure_control(
        preferred_action="subtask_complete",
        signal=signal,
        reason="fresh verifier evidence",
    )

    assert agent._pure_tool_control_no_progress_control_turns == 0
    assert agent.memory_store.state.active_skill is None


def test_observation_only_completion_does_not_reset_no_progress_budget() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_control_turns=64,
                pure_tool_control_max_no_progress_control_turns=4,
            )
        )
    )
    agent.current_instruction = "perform repeated grounded actions"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="obtain another observation",
        skill_spec=skill_spec,
    )
    agent._pure_tool_control_no_progress_control_turns = 3
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="fresh observation",
    )

    agent._continue_debug_recovery_pure_control(
        preferred_action="subtask_complete",
        signal=signal,
        reason="observation acquired",
    )

    assert agent._pure_tool_control_no_progress_control_turns == 3
    assert agent.memory_store.state.active_skill is None


def test_verified_in_progress_physical_effect_resets_no_progress_turn_budget() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_max_control_turns=64,
                pure_tool_control_max_no_progress_control_turns=4,
            )
        )
    )
    agent.current_instruction = "perform a multi-stage grounded manipulation"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent._pure_tool_control_no_progress_control_turns = 4

    agent._credit_pure_tool_control_verified_progress(
        action_effect={
            "effect_verified": "true",
            "effect_type": "move",
            "subtask_status": "in_progress",
        },
        calls=[
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={"arm": "right"},
            )
        ],
    )

    assert agent._pure_tool_control_no_progress_control_turns == 0

    agent._pure_tool_control_no_progress_control_turns = 4
    agent._credit_pure_tool_control_verified_progress(
        action_effect={
            "effect_verified": "true",
            "effect_type": "unknown",
            "subtask_status": "in_progress",
        },
        calls=[RecoveryToolCall(tool_name="reobserve_scene", args={})],
    )

    assert agent._pure_tool_control_no_progress_control_turns == 4


def test_pure_tool_control_rejects_finish_on_subtask_success_only() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot_with_success(eval_success=False, check_success=True)
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)

    agent.apply_control_decision(
        {
            "action_mode": "finish",
            "note": "planner treated subtask success as done",
            "memory_text": "finish requested",
        }
    )

    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is not None
    assert agent.memory_store.state.monitor.phase == "reasoning"
    assert agent.memory_store.state.monitor.status == "needs_reasoning"
    assert "pure_tool_control_finish_rejected" in agent.memory_store.state.working.recovery_history


def test_pure_tool_control_planner_replan_cannot_fail_active_subtask() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)
    agent._debug_recovery_planner_bootstrapped = True
    agent._debug_recovery_rounds = 2

    agent.apply_control_decision(
        {
            "action_mode": "replan",
            "note": "planner wants a different manipulation step",
            "memory_text": "replan requested",
        }
    )

    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is not None
    assert "planner selected step" not in agent.memory_store.state.task.failed_skills
    assert agent.memory_store.state.monitor.phase == "monitoring"
    assert agent.memory_store.state.monitor.status == "rollout_active"
    assert agent.memory_store.state.monitor.env_signal == "running"
    assert agent._debug_recovery_planner_bootstrapped is True
    assert agent._debug_recovery_rounds == 2
    assert "policy_replan" in agent.memory_store.state.working.recovery_history
    assert "pure_tool_control_planner_replan_deferred" in agent.memory_store.state.working.recovery_history


def test_planner_retry_without_active_skill_starts_replacement_subtask() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "complete the manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    agent.apply_control_decision(
        {
            "action_mode": "retry",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": "perform a replacement grounded manipulation step",
            "memory_text": "The previous subtask failed and no replacement has run yet.",
        }
    )

    active = agent.memory_store.state.active_skill
    assert active is not None
    assert active.instruction == "perform a replacement grounded manipulation step"
    assert agent.memory_store.state.working.recovery_history == []


def test_planner_replan_without_active_skill_starts_supplied_subtask() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "complete the manipulation task"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    agent.apply_control_decision(
        {
            "action_mode": "replan",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": "move the staged object into the empty row slot",
            "memory_text": "The prior subtask failed; this is the replacement.",
        }
    )

    active = agent.memory_store.state.active_skill
    assert active is not None
    assert active.instruction == "move the staged object into the empty row slot"


def test_planner_switch_starts_new_evidence_scope_for_repeated_instruction() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                interrupt_on_skill_change=False,
            )
        )
    )
    agent.current_instruction = "press the selected control repeatedly"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    subtask = "press the selected control once"
    first = agent.memory_store.start_or_replace_skill(
        subtask_text=subtask,
        skill_spec=skill_spec,
    )
    agent.memory_store.record_recovery("action_effect:effect=true,subtask=completed")
    agent._pending_action_effect_verification = {"skill_id": first.skill_id}

    agent.apply_control_decision(
        {
            "action_mode": "switch",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": subtask,
            "memory_text": "advance to the next repeated actuation",
        }
    )

    active = agent.memory_store.state.active_skill
    assert active is not None
    assert active.skill_id != first.skill_id
    assert agent.memory_store.state.working.recovery_history == []
    assert agent._pending_action_effect_verification is None


def test_pure_tool_control_legacy_transition_policy_cannot_fail_active_subtask() -> None:
    for action in ("replan", "abort"):
        agent = ImgAgent(
            make_card(
                DummyConfig(
                    debug_recovery_enabled=False,
                    pure_tool_control_enabled=True,
                )
            )
        )
        agent.current_instruction = "generic manipulation task"
        agent.latest_snapshot = make_snapshot()
        agent.memory_store.reset(
            task=agent.current_instruction,
            control_model_name="test",
            available_policies=[],
            available_tools=[],
            mode="deployment",
        )
        skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
        subtask = "perform the current grounded manipulation step"
        agent.memory_store.start_or_replace_skill(subtask_text=subtask, skill_spec=skill_spec)
        agent.memory_store.set_recovery_policy(action=action, reason="legacy transition proposal")

        agent._apply_recovery_policy()

        assert agent.memory_store.state.task.task_finished is False
        assert agent.memory_store.state.active_skill is not None
        assert subtask not in agent.memory_store.state.task.failed_skills
        assert agent.memory_store.state.monitor.phase == "monitoring"
        assert agent.memory_store.state.monitor.status == "rollout_active"
        assert agent.memory_store.state.monitor.env_signal == "running"
        assert f"pure_tool_control_legacy_{action}_deferred" in agent.memory_store.state.working.recovery_history


def test_non_pure_transition_policy_keeps_legacy_failure_semantics() -> None:
    planner_agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=True,
                pure_tool_control_enabled=False,
            )
        )
    )
    planner_agent.current_instruction = "generic manipulation task"
    planner_agent.latest_snapshot = make_snapshot()
    planner_agent.memory_store.reset(
        task=planner_agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = planner_agent._resolve_skill_spec("monitored-subtask-execution")
    planner_subtask = "planner-managed manipulation step"
    planner_agent.memory_store.start_or_replace_skill(subtask_text=planner_subtask, skill_spec=skill_spec)

    planner_agent.apply_control_decision(
        {
            "action_mode": "replan",
            "note": "legacy planner requested replan",
        }
    )

    assert planner_agent.memory_store.state.active_skill is None
    assert planner_subtask in planner_agent.memory_store.state.task.failed_skills

    recovery_agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=True,
                pure_tool_control_enabled=False,
            )
        )
    )
    recovery_agent.current_instruction = "generic manipulation task"
    recovery_agent.latest_snapshot = make_snapshot()
    recovery_agent.memory_store.reset(
        task=recovery_agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    recovery_subtask = "recovery-managed manipulation step"
    recovery_agent.memory_store.start_or_replace_skill(subtask_text=recovery_subtask, skill_spec=skill_spec)
    recovery_agent.memory_store.set_recovery_policy(action="abort", reason="legacy recovery abort")

    recovery_agent._apply_recovery_policy()

    assert recovery_agent.memory_store.state.active_skill is None
    assert recovery_subtask in recovery_agent.memory_store.state.task.failed_skills
    assert recovery_agent.memory_store.state.task.task_finished is True


def test_pure_tool_control_subtask_success_does_not_finish_global_task() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "cover the block"
    previous = make_snapshot()
    current = make_snapshot_with_success(eval_success=False, check_success=True)
    agent.previous_snapshot = previous
    agent.latest_snapshot = current
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)
    active = agent.memory_store.state.active_skill
    assert active is not None

    class DummyTaskEnv:
        take_action_cnt = 1
        step_lim = 100

    handled = agent.update_monitor_after_rollout_step(DummyTaskEnv(), current)

    assert handled is True
    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.active_skill is None
    assert agent.memory_store.state.monitor.phase == "reasoning"
    assert agent.memory_store.state.monitor.status == "needs_reasoning"
    assert agent.memory_store.state.monitor.env_signal == "running"
    assert "pure_tool_control_subtask_success_continue" in agent.memory_store.state.working.recovery_history


def test_pure_tool_control_check_success_does_not_trigger_global_success_control_turn() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.latest_snapshot = make_snapshot_with_success(eval_success=False, check_success=True)
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    need_decision, signal = agent.needs_control_turn()

    assert need_decision is True
    assert signal == ControlSignal.NO_ACTIVE_SKILL


def test_control_turn_payload_exposes_authoritative_environment_success() -> None:
    pure_agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    pure_agent.current_instruction = "generic manipulation task"
    pure_agent.latest_snapshot = make_snapshot_with_success(eval_success=False, check_success=True)
    pure_agent.memory_store.reset(
        task=pure_agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    payload = json.loads(pure_agent.build_control_turn_payload(ControlSignal.NO_ACTIVE_SKILL))
    runtime_evaluation = payload["runtime_evaluation"]

    assert runtime_evaluation["environment_success"] is False
    assert runtime_evaluation["check_success"] is True
    assert runtime_evaluation["global_task_success"] is False
    assert runtime_evaluation["global_success_authority"] == "environment_eval_success"
    assert runtime_evaluation["pure_tool_control"] is True


def test_control_turn_payload_compacts_private_operation_pose_candidates() -> None:
    pure_agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    pure_agent.current_instruction = "generic manipulation task"
    pure_agent.memory_store.reset(
        task=pure_agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    def operation_candidates(instance_index: int) -> list[dict]:
        candidates: list[dict] = []
        for candidate_index in range(16):
            pose = [
                round(instance_index * 0.01 + candidate_index * 0.0001, 6),
                -0.1,
                0.8,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            candidates.append(
                {
                    "candidate_id": f"private_candidate_{instance_index:02d}_{candidate_index:02d}",
                    "source_candidate_index": candidate_index,
                    "action_mode": "grasp" if candidate_index % 2 == 0 else "contact",
                    "arm": "left" if candidate_index % 4 < 2 else "right",
                    "object_contact_pose": pose,
                    "tcp_pose": pose,
                    "ee_target_pose": pose,
                    "approach_pose": pose,
                    "approach_direction": [0.0, 0.0, -1.0],
                    "geometry_source": "oracle_actor_contact_matrix",
                    "private_candidate_padding": "executor-only-" * 16,
                }
            )
        return candidates

    instances: list[dict] = []
    segments: list[dict] = []
    for instance_index in range(10):
        candidates = operation_candidates(instance_index)
        instance_id = f"track_{instance_index:04d}"
        instance = {
            "instance_id": instance_id,
            "track_id": instance_id,
            "class": "component",
            "role": "target" if instance_index == 0 else "context",
            "status": "visible",
            "world_m": [instance_index * 0.01, -0.1, 0.78],
            "approach_world_m": [instance_index * 0.01, -0.1, 0.9],
            "grasp_world_m": [instance_index * 0.01, -0.1, 0.82],
            "operation_pose_candidate_count": len(candidates),
            "operation_pose_candidates": candidates,
        }
        grounding = {
            "success": True,
            "object_id": instance_id,
            "centroid_world": instance["world_m"],
            "grasp_pose_world": [*instance["grasp_world_m"], 1.0, 0.0, 0.0, 0.0],
            "operation_pose_candidates": candidates,
            "object_contact_pose_world_candidates": [
                item["object_contact_pose"] for item in candidates
            ],
            "contact_pose_world_candidates": [item["tcp_pose"] for item in candidates],
            "grasp_pose_world_candidates": [item["tcp_pose"] for item in candidates],
            "approach_pose_world_candidates": [
                item["approach_pose"] for item in candidates
            ],
        }
        instances.append(instance)
        segments.append(
            {
                "success": True,
                "object_id": instance_id,
                "query_role": instance["role"],
                "grounding_3d": grounding,
                "detections": [
                    {
                        "rank": 0,
                        "score": 1.0,
                        "grounding_3d": grounding,
                    }
                ],
            }
        )

    scene_memory = {
        "env_step": 0,
        "task_focus": {
            "target_instances": ["track_0000"],
            "tool_instances": [],
            "reason_summary": "selected target",
        },
        "instances": instances,
        "uncertainty": [],
        "summary": "ten visible components",
    }
    observation_preprocess = {
        "stage": "observation_preprocess",
        "env_step": 0,
        "segmentation": segments,
        "scene_memory": scene_memory,
    }
    pure_agent.memory_store.record_scene_memory(scene_memory)
    pure_agent.memory_store.record_observation_preprocess(observation_preprocess)

    full_context = json.dumps(
        pure_agent.memory_store.to_reasoner_context(),
        ensure_ascii=False,
    )
    payload_text = pure_agent.build_control_turn_payload(ControlSignal.NO_ACTIVE_SKILL)
    payload = json.loads(payload_text)
    working = payload["agent_state"]["working"]

    private_keys = {
        "operation_pose_candidates",
        "object_contact_pose_world_candidates",
        "contact_pose_world_candidates",
        "grasp_pose_world_candidates",
        "approach_pose_world_candidates",
    }

    def nested_keys(value) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(nested_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    assert len(full_context.encode("utf-8")) > 500_000
    assert len(payload_text.encode("utf-8")) < 100_000
    assert len(payload_text) < len(full_context) // 5
    assert private_keys.isdisjoint(nested_keys(working))
    assert "private_candidate_" not in payload_text
    assert "private_candidate_padding" not in payload_text
    assert working["scene_memory"]["instances"][0]["operation_pose_candidate_count"] == 16
    assert working["observation_preprocess"]["segmentation"][0]["grounding_3d"][
        "grasp_pose_world"
    ]
    assert "scene_memory" not in working["observation_preprocess"]
    assert (
        pure_agent.memory_store.state.working.scene_memory["instances"][0][
            "operation_pose_candidates"
        ][0]["candidate_id"]
        == "private_candidate_00_00"
    )


def test_control_turn_compact_v1_uses_one_scene_projection() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                planner_context_mode="compact_v1",
            )
        )
    )
    agent.current_instruction = "move the block"
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    private_candidate = {
        "candidate_id": "private_grasp_candidate",
        "action_mode": "grasp",
        "arm": "right",
        "tcp_pose": [0.1, -0.1, 0.8, 1.0, 0.0, 0.0, 0.0],
        "approach_pose": [0.1, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0],
    }
    scene = {
        "env_step": 1,
        "task_focus": {
            "target_instances": ["track_0001"],
            "identity_binding_required": True,
        },
        "instances": [
            {
                "instance_id": "track_0001",
                "class": "block",
                "query_role": "target",
                "status": "visible",
                "position_state": "current_verified",
                "world_m": [0.1, -0.1, 0.77],
                "operation_pose_candidates": [private_candidate],
            }
        ],
        "temporal_memory": {"padding": "x" * 100_000},
        "uncertainty": [],
    }
    preprocess = {
        "env_step": 1,
        "segmentation": [
            {
                "object_id": "block",
                "camera": "head",
                "success": True,
                "detections": [
                    {
                        "mask_path": "/private/mask.png",
                        "grounding_3d": {
                            "operation_pose_candidates": [private_candidate]
                        },
                    }
                ],
            }
        ],
        "scene_memory": scene,
    }
    agent.memory_store.record_scene_memory(scene)
    agent.memory_store.record_observation_preprocess(preprocess)
    agent.memory_store.record_observation_summary(
        "duplicate block and robot prose"
    )

    payload_text = agent.build_control_turn_payload(
        ControlSignal.NO_ACTIVE_SKILL
    )
    payload = json.loads(payload_text)
    working = payload["agent_state"]["working"]

    assert working["planner_context_mode"] == "compact_v1"
    assert working["scene_memory"]["schema"] == (
        "planner_scene_view/compact_v1"
    )
    assert working["scene_memory"]["instances"][0]["operations"][
        "grasp"
    ]["arms"] == ["right"]
    assert "observation_preprocess" not in working
    assert "recent_observation_summary" not in working
    assert "manipulation_state" not in working
    assert "private_grasp_candidate" not in payload_text
    assert "/private/mask.png" not in payload_text
    assert "duplicate block and robot prose" not in payload_text
    assert (
        agent.memory_store.state.working.scene_memory["instances"][0][
            "operation_pose_candidates"
        ][0]["candidate_id"]
        == "private_grasp_candidate"
    )


def _verified_transport_state(
    *,
    arm: str,
    held_instance_id: str,
    step: int = 0,
) -> dict:
    candidate_id = f"candidate:{arm}:{held_instance_id}:001"
    nonce = f"{arm}:{held_instance_id}:{candidate_id}:{step}"
    return {
        "phase": "holding",
        "held_instance_id": held_instance_id,
        "holding_confirmed": True,
        "transport_authorized": True,
        "grasp_candidate_id": candidate_id,
        "grasp_attempt_step": step,
        "grasp_attempt_nonce": nonce,
        "held_object_to_tcp_attachment": {
            "object_proxy_frame": "tcp_aligned_at_capture",
            "object_centroid_to_tcp_translation_tcp_m": [0.0, 0.0, 0.04],
            "capture_tcp_pose": [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
            "grasp_candidate_id": candidate_id,
            "grasp_attempt_nonce": nonce,
            "capture_step": step,
            "source": "runtime_multiview_grasp_motion_verified",
            "authority": "runtime_multiview_grasp_motion",
        },
    }


def test_identity_bound_carry_is_canonicalized_from_confirmed_holding_state() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic rearrangement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    scene_memory = {
        "env_step": 0,
        "task_focus": {
            "target_instances": ["track_held", "track_destination"],
            "tool_instances": [],
            "identity_binding_required": True,
        },
        "instances": [
            {
                "instance_id": "track_held",
                "track_id": "track_held",
                "status": "visible",
                "world_m": [0.1, 0.0, 0.8],
                "quality": {
                    "world_extent_m": [0.04, 0.04, 0.04],
                    "world_z_max_m": 0.82,
                },
            },
            {
                "instance_id": "track_destination",
                "track_id": "track_destination",
                "status": "visible",
                "world_m": [0.2, 0.0, 0.8],
                "quality": {
                    "world_extent_m": [0.04, 0.04, 0.04],
                    "world_z_max_m": 0.82,
                },
            },
        ],
    }
    agent.memory_store.record_scene_memory(scene_memory)
    agent.memory_store.set_manipulation_arm_state(
        "right",
        _verified_transport_state(
            arm="right",
            held_instance_id="track_held",
        ),
    )

    enriched = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "arm": "right",
                    "role": "tool",
                    "point_key": "world_m",
                    "preserve_height": True,
                },
            )
        ]
    )

    assert len(enriched) == 1
    assert enriched[0].tool_name == "move_ee_to_grounded_instance"
    assert enriched[0].args["instance_id"] == "track_destination"
    assert enriched[0].args["held_instance_id"] == "track_held"


def test_contact_mode_close_does_not_create_portable_grasp_state() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(0.0)
    agent.memory_store.reset(
        task="operate an articulated mechanism",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "env_step": 0,
            "task_focus": {"target_instances": ["track_0001"]},
            "instances": [
                {
                    "instance_id": "track_0001",
                    "track_id": "track_0001",
                    "status": "visible",
                    "world_m": [0.1, 0.0, 0.8],
                    "contact_world_m": [0.1, 0.0, 0.8],
                }
            ],
        }
    )
    calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={
                "arm": "right",
                "instance_id": "track_0001",
                "point_key": "contact_world_m",
                "_operation_action_mode": "contact",
            },
        ),
        RecoveryToolCall(tool_name="close_gripper", args={"arm": "right"}),
        RecoveryToolCall(tool_name="reobserve_scene", args={}),
    ]
    results = [
        RecoveryToolResult(
            tool_name="move_ee_to_grounded_instance",
            success=True,
            details={
                "arm": "right",
                "instance_id": "track_0001",
                "target_reached": True,
                "operation_action_mode": "contact",
            },
        ),
        RecoveryToolResult(tool_name="close_gripper", success=True),
        RecoveryToolResult(tool_name="reobserve_scene", success=True),
    ]

    agent._update_manipulation_state_from_effect(
        calls=calls,
        results=results,
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "contact",
        },
    )

    assert "right" not in (
        agent.memory_store.state.working.manipulation_state
    )


def test_contact_mode_close_and_articulation_displacement_are_not_split_by_grasp_boundary() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot_with_gripper(1.0)
    agent.latest_snapshot.right_endpose[:] = np.asarray(
        [0.30, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.memory_store.reset(
        task="operate an articulated mechanism",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "track_0001",
                    "track_id": "track_0001",
                    "status": "visible",
                    "stability": "stable",
                    "position_state": "current_verified",
                    "action_geometry_state": "verified",
                    "approach_world_m": [0.0, 0.0, 0.10],
                    "contact_world_m": [0.0, 0.0, 0.0],
                }
            ],
            "task_focus": {"target_instances": ["track_0001"]},
        }
    )

    guarded = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "instance_id": "track_0001",
                    "arm": "left",
                    "point_key": "contact_world_m",
                },
            ),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "left"},
            ),
            RecoveryToolCall(
                tool_name="contact_displace",
                args={
                    "arm": "left",
                    "axis": "z",
                    "direction": "negative",
                    "distance": 0.01,
                    "steps": 2,
                },
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in guarded] == [
        "move_ee_to_grounded_instance",
        "close_gripper",
        "contact_displace",
        "reobserve_scene",
    ]
    assert all(
        call.args.get("_runtime_grasp_close_boundary") is not True
        and call.args.get("_runtime_grasp_diagnostic_motion") is not True
        for call in guarded
    )


def test_pending_portable_grasp_uses_runtime_reverse_ingress_not_fixed_lift() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.right_endpose[:] = np.asarray(
        [0.10, -0.20, 0.80, 0.5, -0.5, 0.5, -0.5],
        dtype=np.float32,
    )
    snapshot.raw.setdefault("endpose", {})[
        "right_gripper"
    ] = 0.0
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="transport a portable object",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "right": {
            "phase": "grasp_candidate",
            "held_instance_id": "track_0001",
            "holding_confirmed": False,
            "transport_authorized": False,
            "operation_action_mode": "grasp",
            "grasp_candidate_id": "candidate:right:001",
            "grasp_attempt_step": 7,
            "grasp_attempt_nonce": (
                "right:track_0001:candidate:right:001:7"
            ),
            "grasp_ee_target_world_m": [0.10, -0.20, 0.80],
            "grasp_approach_world_m": [0.13, -0.16, 0.82],
            "held_object_to_tcp_attachment": {
                "object_proxy_frame": "tcp_aligned_at_capture",
                "object_centroid_to_tcp_translation_tcp_m": [
                    0.0,
                    0.0,
                    0.04,
                ],
                "capture_tcp_pose": [
                    0.10,
                    -0.20,
                    0.80,
                    0.5,
                    -0.5,
                    0.5,
                    -0.5,
                ],
                "grasp_candidate_id": "candidate:right:001",
                "grasp_attempt_nonce": (
                    "right:track_0001:candidate:right:001:7"
                ),
                "capture_step": 7,
                "source": "pending_lift_verification",
            },
        }
    }

    guarded = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="lift_ee",
                args={
                    "arm": "right",
                    "axis": "z",
                    "distance": 0.03,
                },
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in guarded] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    diagnostic = guarded[0]
    assert diagnostic.args[
        "_runtime_grasp_diagnostic_motion"
    ] is True
    assert diagnostic.args["target_pose"][:3] != [
        0.10,
        -0.20,
        0.83,
    ]
    assert diagnostic.args["target_pose"][3:] == [
        0.5,
        -0.5,
        0.5,
        -0.5,
    ]


def test_evidence_only_pending_grasp_does_not_block_same_arm_or_take_over_open() -> None:
    agent = ImgAgent(
        make_card(DummyConfig(grasp_transport_policy="evidence_only"))
    )
    snapshot = make_snapshot_with_gripper(0.0)
    snapshot.right_endpose[:] = np.asarray(
        [0.10, -0.20, 0.80, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="transport a portable object",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "right": {
            "phase": "grasp_candidate",
            "held_instance_id": "track_0001",
            "holding_confirmed": False,
            "transport_authorized": False,
            "grasp_candidate_id": "candidate:right:001",
            "grasp_attempt_step": 7,
            "grasp_attempt_nonce": (
                "right:track_0001:candidate:right:001:7"
            ),
        }
    }

    move = agent._with_internal_recovery_context(
        [RecoveryToolCall(tool_name="move_to_home", args={"arm": "right"})]
    )
    explicit_open = agent._with_internal_recovery_context(
        [RecoveryToolCall(tool_name="open_gripper", args={"arm": "right"})]
    )

    assert [call.tool_name for call in move] == ["move_to_home"]
    assert [call.tool_name for call in explicit_open] == ["open_gripper"]
    assert all(
        "_runtime_failed_grasp_clearance" not in call.args
        for call in explicit_open
    )


def test_evidence_only_policy_is_exposed_to_recovery_planner() -> None:
    agent = ImgAgent(
        make_card(DummyConfig(grasp_transport_policy="evidence_only"))
    )
    agent.current_instruction = "transport a portable object"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.start_or_replace_skill(
        subtask_text="pick the selected object",
        skill_spec=agent._resolve_skill_spec(
            "monitored-subtask-execution"
        ),
    )
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="valid-empty-plan",
            tool_calls=[],
            expected_outcome="continue planning",
        ),
        post_recovery_intent="retry",
        reason="inspect runtime policy",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="pure tool-control evaluation",
    )

    with (
        mock.patch.object(
            agent._recovery_dispatcher,
            "available_tools",
            return_value=[],
        ),
        mock.patch.object(
            agent._recovery_policy_resolver,
            "resolve",
            return_value=route,
        ) as resolve,
    ):
        agent._dispatch_recovery_tools(task_env=object(), signal=signal)

    assert resolve.call_args.kwargs["recovery_state"][
        "grasp_transport_policy"
    ] == "evidence_only"


def test_compact_v1_recovery_planner_receives_one_scene_projection() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                grasp_transport_policy="evidence_only",
                planner_context_mode="compact_v1",
            )
        )
    )
    agent.current_instruction = "pick and move the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task=agent.current_instruction,
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.start_or_replace_skill(
        subtask_text="pick the selected block",
        skill_spec=agent._resolve_skill_spec(
            "monitored-subtask-execution"
        ),
    )
    private_candidate = {
        "candidate_id": "private_candidate",
        "action_mode": "grasp",
        "arm": "right",
        "tcp_pose": [0.1, -0.1, 0.8, 1.0, 0.0, 0.0, 0.0],
        "approach_pose": [0.1, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0],
    }
    scene = {
        "env_step": 0,
        "task_focus": {
            "target_instances": ["track_0001"],
            "identity_binding_required": True,
        },
        "instances": [
            {
                "instance_id": "track_0001",
                "class": "block",
                "query_role": "target",
                "status": "visible",
                "position_state": "current_verified",
                "world_m": [0.1, -0.1, 0.77],
                "operation_pose_candidates": [private_candidate],
            }
        ],
        "temporal_memory": {"padding": "x" * 100_000},
    }
    preprocess = {
        "env_step": 0,
        "segmentation": [
            {
                "object_id": "block",
                "camera": "head",
                "success": True,
                "detections": [
                    {
                        "mask_path": "/private/mask.png",
                        "grounding_3d": {
                            "operation_pose_candidates": [private_candidate]
                        },
                    }
                ],
            }
        ],
        "scene_memory": scene,
    }
    agent.memory_store.record_scene_memory(scene)
    agent.memory_store.record_observation_preprocess(preprocess)
    agent.memory_store.record_observation_summary("duplicate scene prose")
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="valid-empty-plan",
            tool_calls=[],
            expected_outcome="continue planning",
        ),
        post_recovery_intent="retry",
        reason="inspect compact context",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="pure tool-control evaluation",
    )

    with (
        mock.patch.object(
            agent._recovery_dispatcher,
            "available_tools",
            return_value=[],
        ),
        mock.patch.object(
            agent._recovery_policy_resolver,
            "resolve",
            return_value=route,
        ) as resolve,
    ):
        agent._dispatch_recovery_tools(task_env=object(), signal=signal)

    kwargs = resolve.call_args.kwargs
    assert kwargs["scene_memory"]["schema"] == (
        "planner_scene_view/compact_v1"
    )
    assert kwargs["scene_memory"]["instances"][0]["operations"][
        "grasp"
    ]["arms"] == ["right"]
    assert kwargs["observation_preprocess"] == {}
    assert kwargs["observation_summary"] == ""
    assert kwargs["recovery_state"]["planner_context_mode"] == "compact_v1"
    assert "manipulation_state" not in kwargs["recovery_state"]
    serialized = json.dumps(kwargs["scene_memory"])
    assert "private_candidate" not in serialized
    assert "/private/mask.png" not in serialized
    assert agent.memory_store.state.working.observation_preprocess[
        "segmentation"
    ][0]["detections"][0]["mask_path"] == "/private/mask.png"


def test_evidence_only_close_uses_close_time_pose_and_stays_unconfirmed() -> None:
    agent = ImgAgent(
        make_card(DummyConfig(grasp_transport_policy="evidence_only"))
    )
    snapshot = make_snapshot_with_gripper(0.0)
    snapshot.step_count = 8
    snapshot.right_endpose[:] = np.asarray(
        [0.50, 0.00, 0.90, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="transport a portable object",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    candidate_id = "candidate:right:track_0001:001"
    scene_memory = {
        "env_step": 4,
        "task_focus": {"target_instances": ["track_0001"]},
        "instances": [
            {
                "instance_id": "track_0001",
                "track_id": "track_0001",
                "status": "visible",
                "stability": "stable",
                "position_state": "current_verified",
                "action_geometry_state": "verified",
                "world_m": [0.10, 0.00, 0.80],
                "latest_world_m": [0.10, 0.00, 0.80],
                "quality": {
                    "actionable": True,
                    "world_extent_m": [0.04, 0.04, 0.04],
                    "world_z_max_m": 0.82,
                },
                "operation_pose_candidates": [
                    {
                        "candidate_id": candidate_id,
                        "arm": "right",
                        "action_mode": "grasp",
                        "object_contact_pose": [
                            0.10, 0.00, 0.82, 1.0, 0.0, 0.0, 0.0
                        ],
                        "ee_target_pose": [
                            0.10, 0.00, 0.84, 1.0, 0.0, 0.0, 0.0
                        ],
                        "approach_pose": [
                            0.10, 0.00, 0.92, 1.0, 0.0, 0.0, 0.0
                        ],
                    }
                ],
            }
        ],
    }
    agent.memory_store.record_scene_memory(scene_memory)
    setup = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "arm": "right",
            "instance_id": "track_0001",
            "point_key": "grasp_world_m",
            "_operation_action_mode": "grasp",
            "_operation_candidate_id": candidate_id,
            "_scene_memory": scene_memory,
        },
    )
    calls = [
        setup,
        RecoveryToolCall(tool_name="close_gripper", args={"arm": "right"}),
        RecoveryToolCall(tool_name="move_to_home", args={"arm": "right"}),
    ]
    results = [
        RecoveryToolResult(
            tool_name="move_ee_to_grounded_instance",
            success=True,
            details={
                "arm": "right",
                "instance_id": "track_0001",
                "target_reached": True,
                "operation_action_mode": "grasp",
                "operation_candidate_id": candidate_id,
                "target_pose": [
                    0.10, 0.00, 0.84, 1.0, 0.0, 0.0, 0.0
                ],
            },
        ),
        RecoveryToolResult(
            tool_name="close_gripper",
            success=True,
            details={
                "arm": "right",
                "step_count": 5,
                "observed_pose": [
                    0.10, 0.00, 0.84, 1.0, 0.0, 0.0, 0.0
                ],
            },
        ),
        RecoveryToolResult(tool_name="move_to_home", success=True),
    ]

    agent._update_manipulation_state_from_effect(
        calls=calls,
        results=results,
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "grasp",
        },
    )

    state = agent.memory_store.state.working.manipulation_state["right"]
    assert state["phase"] == "holding_provisional"
    assert state["holding_confirmed"] is False
    assert state["transport_authorized"] is True
    assert state["grasp_attempt_step"] == 5
    assert state["held_object_to_tcp_attachment"]["capture_tcp_pose"][:3] == [
        0.1,
        0.0,
        0.84,
    ]
    assert agent._transport_holding_state("right") is not None

    agent._agent_card.config = replace(
        agent.config,
        grasp_transport_policy="strict",
    )
    assert agent._transport_holding_state("right") is None
    agent._sync_runtime_manipulation_state_to_scene_memory()
    assert agent.memory_store.state.working.scene_memory[
        "operation_targets"
    ] == []


def test_verified_grasp_state_survives_replan_and_blocks_unbound_release() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="generic rearrangement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(
        subtask_text="pick the selected object",
        skill_spec=skill_spec,
    )
    agent.memory_store.record_scene_memory(
        {
            "env_step": 0,
            "task_focus": {
                "target_instances": ["track_0001"],
                "tool_instances": [],
                "identity_binding_required": True,
            },
            "instances": [
                {
                    "instance_id": "track_0001",
                    "track_id": "track_0001",
                    "status": "visible",
                    "grasp_world_m": [0.1, 0.0, 0.8],
                }
            ],
        }
    )
    grasp_calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={
                "arm": "right",
                "instance_id": "track_0001",
                "action_mode": "grasp",
                "point_key": "grasp_world_m",
            },
        ),
        RecoveryToolCall(tool_name="close_gripper", args={"arm": "right"}),
        RecoveryToolCall(tool_name="reobserve_scene", args={}),
    ]
    grasp_results = [
        RecoveryToolResult(
            tool_name="move_ee_to_grounded_instance",
            success=True,
            details={
                "arm": "right",
                "instance_id": "track_0001",
                "operation_action_mode": "grasp",
                "target_reached": True,
            },
        ),
        RecoveryToolResult(tool_name="close_gripper", success=True),
        RecoveryToolResult(tool_name="reobserve_scene", success=True),
    ]
    agent._update_manipulation_state_from_effect(
        calls=grasp_calls,
        results=grasp_results,
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "grasp",
        },
    )
    assert agent.memory_store.state.working.manipulation_state["right"][
        "holding_confirmed"
    ] is False

    agent.memory_store.set_manipulation_arm_state(
        "right",
        _verified_transport_state(
            arm="right",
            held_instance_id="track_0001",
        ),
    )
    held_state = agent.memory_store.state.working.manipulation_state["right"]
    assert held_state["holding_confirmed"] is True
    assert held_state["held_instance_id"] == "track_0001"

    agent.memory_store.record_recovery("attempt-local verifier evidence")
    agent.memory_store.mark_active_skill_failed(note="force replan")
    agent.memory_store.start_or_replace_skill(
        subtask_text="continue placement from the current physical state",
        skill_spec=skill_spec,
    )

    assert agent.memory_store.state.working.recovery_history == []
    assert agent.memory_store.state.working.manipulation_state["right"][
        "held_instance_id"
    ] == "track_0001"

    blocked = agent._with_internal_recovery_context(
        [RecoveryToolCall(tool_name="open_gripper", args={"arm": "right"})]
    )
    assert [call.tool_name for call in blocked] == ["reobserve_scene"]
    assert "verifier-confirmed held instance" in blocked[0].args["_guard_reason"]

    matching_but_unplaced = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "right",
                    "release_held_instance_id": "track_0001",
                },
            )
        ]
    )
    assert [call.tool_name for call in matching_but_unplaced] == [
        "reobserve_scene"
    ]
    assert "runtime-validated place target" in matching_but_unplaced[
        0
    ].args["_guard_reason"]


def test_occluded_diagnostic_lift_does_not_authorize_transport() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.step_count = 16
    snapshot.right_endpose = np.asarray(
        [0.16, -0.10, 0.928, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    snapshot.raw.setdefault("endpose", {})
    snapshot.raw["endpose"]["right_gripper"] = 0.0
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic rearrangement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "env_step": 16,
            "task_focus": {
                "target_instances": ["track_0002"],
                "tool_instances": [],
                "identity_binding_required": True,
            },
            "instances": [
                {
                    "instance_id": "track_0002",
                    "track_id": "track_0002",
                    "status": "tracked",
                    "stability": "missing_current_frame",
                    "camera": "head",
                    "world_m": [0.16, -0.10, 0.76],
                    "latest_world_m": [0.16, -0.10, 0.76],
                    "first_observed_world_m": [0.16, -0.10, 0.76],
                    "top_surface_world_m": [0.16, -0.10, 0.78],
                    "quality": {
                        "actionable": True,
                        "world_extent_m": [0.04, 0.04, 0.04],
                        "world_z_max_m": 0.78,
                    },
                    "operation_pose_candidates": [],
                }
            ],
        }
    )
    agent.memory_store.set_manipulation_arm_state(
        "right",
        {
            "phase": "grasp_candidate",
            "held_instance_id": "track_0002",
            "holding_confirmed": False,
            "transport_authorized": False,
            "operation_action_mode": "grasp",
            "grasp_candidate_id": "rgbd_volume:grasp:right:000",
            "grasp_occlusion_geometry_lease": True,
            "grasp_ee_target_world_m": [0.16, -0.10, 0.90],
            "held_object_to_tcp_attachment": {
                "object_proxy_frame": "tcp_aligned_at_capture",
                "object_centroid_to_tcp_translation_tcp_m": [
                    0.0,
                    0.0,
                    0.168,
                ],
                "source": "reached_grasp_close_hypothesis",
            },
        },
    )

    agent._update_manipulation_state_from_effect(
        calls=[
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.03},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        results=[
            RecoveryToolResult(
                tool_name="lift_ee",
                success=True,
                details={
                    "arm": "right",
                    "axis": "z",
                    "signed_distance": 0.03,
                    "observed_axis_displacement_m": 0.026,
                    "observed_pose": [
                        0.16,
                        -0.10,
                        0.928,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                },
            ),
            RecoveryToolResult(
                tool_name="reobserve_scene",
                success=True,
            ),
        ],
        # Missing visual evidence after the lift does not prove attachment.
        action_effect={
            "effect_verified": "false",
            "effect_type": "grasp",
        },
    )

    state = agent.memory_store.state.working.manipulation_state["right"]
    assert state["phase"] == "grasp_candidate"
    assert state["holding_confirmed"] is False
    assert state["transport_authorized"] is False
    assert (
        state["diagnostic_lift_evidence"][
            "fresh_visible_negative_evidence"
        ]
        is False
    )
    assert (
        state["diagnostic_lift_evidence"]["verifier_effect"]
        == "false"
    )
    assert (
        state["diagnostic_lift_evidence"]["transport_authorized"]
        is False
    )

    clear_failed_grasp = agent._with_internal_recovery_context(
        [RecoveryToolCall(tool_name="open_gripper", args={"arm": "right"})]
    )
    assert [call.tool_name for call in clear_failed_grasp] == [
        "reobserve_scene"
    ]
    assert "runtime-proven two-view failure" in clear_failed_grasp[
        0
    ].args["_guard_reason"]


def test_opening_after_unconfirmed_grasp_is_not_an_object_release_event() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.step_count = 9
    agent.latest_snapshot = snapshot
    grasp_candidate = {
        "phase": "grasp_candidate",
        "held_instance_id": "track_0001",
        "holding_confirmed": False,
        "transport_authorized": False,
    }

    failed_grasp_events = agent._scene_events_from_recovery_effect(
        calls=[
            RecoveryToolCall(
                tool_name="open_gripper",
                args={"arm": "right"},
            )
        ],
        results=[
            RecoveryToolResult(
                tool_name="open_gripper",
                success=True,
            )
        ],
        action_effect={
            "effect_verified": "false",
            "effect_type": "grasp",
        },
        previous_manipulation_state={"right": grasp_candidate},
        updated_manipulation_state={},
    )

    assert not any(
        event.get("source") == "object_release"
        for event in failed_grasp_events
    )

    confirmed_release_events = agent._scene_events_from_recovery_effect(
        calls=[
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "right",
                    "release_held_instance_id": "track_0001",
                },
            )
        ],
        results=[
            RecoveryToolResult(
                tool_name="open_gripper",
                success=True,
            )
        ],
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "release",
        },
        previous_manipulation_state={
            "right": {
                **grasp_candidate,
                "phase": "holding",
                "holding_confirmed": True,
                "transport_authorized": True,
            }
        },
        updated_manipulation_state={},
    )

    assert any(
        event.get("source") == "object_release"
        and event.get("instance_ref") == "track_0001"
        for event in confirmed_release_events
    )


def test_failed_grasp_open_then_clean_multiview_reacquisition_allows_approach() -> None:
    """Cover the full recovery chain seen in the failed rollout.

    Clearing an unconfirmed grasp must not create a second motion event.  A
    clean view can then replace a self-contaminated view of the same bound
    object, after which the ordinary grounded-approach guard is usable again.
    """

    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.step_count = 9
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="reposition one selected object",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    def observed_object(
        *,
        camera: str,
        bbox: list[int],
        world: list[float],
        approach_offset_y: float,
        score: float,
    ) -> dict:
        extent = [0.052, 0.056, 0.040]
        top = [world[0], world[1], world[2] + 0.04]
        approach = [
            world[0],
            world[1] + approach_offset_y,
            world[2] + 0.20,
        ]
        contact = [
            world[0],
            world[1] + approach_offset_y,
            world[2] + 0.12,
        ]
        return {
            "rank": 0,
            "score": score,
            "camera": camera,
            "bbox_xyxy": bbox,
            "centroid_px": [
                (bbox[0] + bbox[2]) / 2.0,
                (bbox[1] + bbox[3]) / 2.0,
            ],
            "mask_path": f"/tmp/failed_grasp_recovery_{camera}.png",
            "grounding_3d": {
                "success": True,
                "centroid_world": world,
                "bbox_world_min": [
                    world[index] - extent[index] / 2.0
                    for index in range(3)
                ],
                "bbox_world_max": [
                    world[index] + extent[index] / 2.0
                    for index in range(3)
                ],
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
            },
        }

    initial_world = [0.097478, -0.203906, 0.775435]
    initial = agent._scene_memory_tracker.update(
        segmentation=[
            {
                "success": True,
                "object_id": "selected_component",
                "camera": "head",
                "query_role": "target",
                "detections": [
                    observed_object(
                        camera="head",
                        bbox=[150, 150, 190, 185],
                        world=initial_world,
                        approach_offset_y=0.0,
                        score=0.90,
                    )
                ],
            }
        ],
        env_step=8,
        global_task="reposition one selected object",
        current_subtask="acquire the selected object",
    )
    track_id = initial["instances"][0]["track_id"]
    agent.memory_store.record_scene_memory(initial)

    uncertain = agent._scene_memory_tracker.update(
        segmentation=[],
        env_step=9,
        global_task="reposition one selected object",
        current_subtask="recover from an unconfirmed grasp",
        position_events=[
            {
                "instance_ref": track_id,
                "position_state": "motion_uncertain",
                "source": "grasp_closure",
            }
        ],
    )
    agent.memory_store.record_scene_memory(uncertain)
    agent.memory_store.state.working.manipulation_state = {
        "right": {
            "phase": "grasp_candidate",
            "held_instance_id": track_id,
            "holding_confirmed": False,
            "transport_authorized": False,
        }
    }

    agent._update_manipulation_state_from_effect(
        calls=[
            RecoveryToolCall(
                tool_name="open_gripper",
                args={"arm": "right"},
            )
        ],
        results=[
            RecoveryToolResult(
                tool_name="open_gripper",
                success=True,
            )
        ],
        action_effect={
            "effect_verified": "false",
            "effect_type": "grasp",
        },
    )

    assert agent.memory_store.state.working.manipulation_state == {}
    after_open = agent.memory_store.state.working.scene_memory
    after_open_instance = next(
        item
        for item in after_open["instances"]
        if item["track_id"] == track_id
    )
    assert after_open_instance["position_state"] == "motion_uncertain"

    contaminated_head = observed_object(
        camera="head",
        bbox=[155, 150, 198, 188],
        world=[0.095360, -0.214045, 0.778300],
        approach_offset_y=0.073,
        score=0.91,
    )
    clean_third = observed_object(
        camera="third",
        bbox=[205, 148, 245, 185],
        world=[0.102513, -0.193762, 0.770866],
        approach_offset_y=0.0,
        score=0.82,
    )
    reacquired = agent._scene_memory_tracker.update(
        segmentation=[
            {
                "success": True,
                "object_id": "selected_component",
                "camera": "head",
                "query_role": "target",
                "instance_ref": track_id,
                "identity_binding_required": True,
                "detections": [contaminated_head],
            },
            {
                "success": True,
                "object_id": "selected_component",
                "camera": "third",
                "query_role": "target",
                "instance_ref": track_id,
                "identity_binding_required": True,
                "detections": [clean_third],
            },
        ],
        env_step=10,
        global_task="reposition one selected object",
        current_subtask="reacquire after a failed grasp",
    )
    agent.memory_store.record_scene_memory(reacquired)

    reacquired_instance = next(
        item
        for item in reacquired["instances"]
        if item["track_id"] == track_id
    )
    assert reacquired_instance["position_state"] == "current_verified"
    assert reacquired_instance["camera"] == "third"

    guarded = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "instance_ref": track_id,
                    "arm": "right",
                    "point_key": "approach_world_m",
                },
            )
        ]
    )
    assert [call.tool_name for call in guarded] == [
        "move_ee_to_grounded_instance"
    ]


def test_visible_stationary_target_cannot_gain_provisional_hold_capability() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.step_count = 16
    snapshot.raw.setdefault("endpose", {})
    snapshot.raw["endpose"]["right_gripper"] = 0.0
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic rearrangement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.record_scene_memory(
        {
            "env_step": 16,
            "instances": [
                {
                    "instance_id": "track_0002",
                    "track_id": "track_0002",
                    "status": "visible",
                    "stability": "stable",
                    "world_m": [0.16, -0.10, 0.76],
                }
            ],
        }
    )
    agent.memory_store.set_manipulation_arm_state(
        "right",
        {
            "phase": "grasp_candidate",
            "held_instance_id": "track_0002",
            "holding_confirmed": False,
            "transport_authorized": False,
            "operation_action_mode": "grasp",
            "grasp_candidate_id": "rgbd_volume:grasp:right:000",
            "grasp_occlusion_geometry_lease": True,
            "held_object_to_tcp_attachment": {
                "object_centroid_to_tcp_translation_tcp_m": [
                    0.0,
                    0.0,
                    0.12,
                ],
            },
        },
    )

    agent._update_manipulation_state_from_effect(
        calls=[
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.03},
            )
        ],
        results=[
            RecoveryToolResult(
                tool_name="lift_ee",
                success=True,
                details={
                    "axis": "z",
                    "signed_distance": 0.03,
                    "observed_axis_displacement_m": 0.026,
                },
            )
        ],
        action_effect={
            "effect_verified": "false",
            "effect_type": "grasp",
        },
    )

    state = agent.memory_store.state.working.manipulation_state["right"]
    assert state["phase"] == "grasp_candidate"
    assert state["transport_authorized"] is False


def test_compact_recovery_context_keeps_bound_context_instance_identities() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    scene_memory = {
        "env_step": 4,
        "task_focus": {
            "target_instances": ["track_blue"],
            "tool_instances": [],
            "identity_binding_required": True,
        },
        "instances": [
            {
                "instance_id": "track_blue",
                "track_id": "track_blue",
                "class": "block",
                "status": "visible",
                "world_m": [0.1, 0.0, 0.8],
            },
            {
                "instance_id": "track_green",
                "track_id": "track_green",
                "class": "block",
                "status": "visible",
                "world_m": [0.2, 0.0, 0.8],
            },
            {
                "instance_id": "oracle_extra",
                "track_id": "oracle_extra",
                "class": "block",
                "status": "visible",
                "world_m": [0.3, 0.0, 0.8],
            },
        ],
    }
    observation_preprocess = {
        "stage": "observation_preprocess",
        "env_step": 4,
        "segmentation": [
            {
                "success": True,
                "object_id": "blue_object",
                "query_role": "target",
                "instance_ref": "track_blue",
            },
            {
                "success": True,
                "object_id": "green_reference",
                "query_role": "context",
                "instance_ref": "track_green",
            },
            {
                "success": True,
                "object_id": "unbound_oracle_context",
                "query_role": "context",
                "instance_ref": "",
            },
        ],
    }
    agent.memory_store.record_scene_memory(scene_memory)
    agent.memory_store.record_observation_preprocess(observation_preprocess)

    compact_scene = agent._compact_scene_memory_for_effect(scene_memory)
    compact_observation = agent._compact_observation_preprocess_for_recovery(
        observation_preprocess
    )

    assert [item["instance_id"] for item in compact_scene["instances"]] == [
        "track_blue",
        "track_green",
    ]
    assert compact_observation["segmentation"][1]["instance_ref"] == "track_green"


def test_pure_tool_control_check_success_does_not_skip_forced_recovery() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
                pure_tool_control_bootstrap_with_planner=False,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot_with_success(eval_success=False, check_success=True)
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
    agent.memory_store.start_or_replace_skill(subtask_text="planner selected step", skill_spec=skill_spec)

    with mock.patch.object(agent, "_dispatch_recovery_tools", return_value="retry") as dispatch:
        handled = agent._maybe_force_debug_recovery(task_env=object())

    assert handled is True
    assert dispatch.call_count == 1
    assert agent.memory_store.state.task.task_finished is False
    assert "pure_tool_control_round:1" in agent.memory_store.state.working.recovery_history


def test_pure_tool_control_reopens_unvalidated_internal_finish() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                debug_recovery_enabled=False,
                pure_tool_control_enabled=True,
            )
        )
    )
    agent.current_instruction = "cover the block"
    agent.latest_snapshot = make_snapshot()
    agent.memory_store.reset(
        task="cover the block",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.mark_task_finished(note="invalid internal finish")

    status = agent.current_status()
    assert status["task_finished"] is True
    assert status["task_finish_validated"] is False
    assert agent._repair_unvalidated_pure_tool_control_finish() is True

    assert agent.memory_store.state.task.task_finished is False
    assert agent.memory_store.state.monitor.phase == "reasoning"
    assert agent.memory_store.state.monitor.status == "needs_reasoning"
    assert agent.memory_store.state.monitor.env_signal == "running"
    assert "pure_tool_control_unvalidated_finish_reopened" in agent.memory_store.state.working.recovery_history


def _place_test_agent(
    config: DummyConfig | None = None,
) -> tuple[ImgAgent, dict]:
    agent = ImgAgent(make_card(config or DummyConfig()))
    snapshot = make_snapshot_with_gripper(0.0)
    snapshot.left_endpose[:] = np.asarray(
        [0.01, -0.01, 0.24, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            **_verified_transport_state(
                arm="left",
                held_instance_id="held",
            ),
            "pregrasp_object_world_m": [0.0, 0.0, 0.02],
            "pregrasp_object_contact_world_m": [0.0, 0.0, 0.04],
        }
    }
    scene = agent._with_runtime_manipulation_state(
        {
            "instances": [
                {
                    "instance_id": "held",
                    "track_id": "held",
                    "status": "tracked",
                    "stability": "missing_current_frame",
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
            "task_focus": {
                "tool_instances": ["held"],
                "target_instances": ["anchor"],
                "identity_binding_required": True,
            },
        }
    )
    agent.memory_store.record_scene_memory(scene)
    return agent, scene


def test_place_target_is_public_but_runtime_selects_private_arm_candidate() -> None:
    agent, scene = _place_test_agent()
    public = agent._compact_scene_memory_for_effect(scene)
    target_id = next(
        item["target_id"]
        for item in public["operation_targets"]
        if item["target_kind"] == "vacated_pose"
    )
    assert all(
        "candidate_id" not in item for item in public["operation_targets"]
    )

    guarded = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "arm": "left",
                    "action_mode": "place",
                    "target_id": target_id,
                    "point_key": "place_world_m",
                },
            ),
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "left",
                    "release_held_instance_id": "held",
                    "release_target_id": target_id,
                },
            ),
        ]
    )

    assert [call.tool_name for call in guarded] == [
        "move_ee_to_grounded_instance",
        "open_gripper",
    ]
    assert guarded[0].args["instance_id"] == "held"
    assert guarded[0].args["_operation_action_mode"] == "place"
    assert guarded[0].args["_operation_candidate_id"].endswith(":arm:left")
    assert guarded[0].args["_runtime_place_motion_authorized"] is True
    assert guarded[1].args["_release_place_validation"]["valid"] is True
    assert (
        guarded[1].args["_release_place_candidate"]["target_id"]
        == target_id
    )


def test_legacy_provisional_hold_cannot_create_place_targets() -> None:
    agent, scene = _place_test_agent()
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "holding_provisional",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "transport_authorized": True,
            "operation_action_mode": "grasp",
            "grasp_candidate_id": "rgbd_volume:grasp:left:000",
            "held_object_to_tcp_attachment": {
                "object_proxy_frame": "tcp_aligned_at_capture",
                "object_centroid_to_tcp_translation_tcp_m": [
                    0.01,
                    -0.01,
                    0.04,
                ],
                "capture_tcp_pose": [
                    0.01,
                    -0.01,
                    0.24,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "source": (
                    "reached_grasp_close_plus_bounded_lift_under_"
                    "proven_self_occlusion"
                ),
            },
            "provisional_transport_evidence": {
                "observed_lift_m": 0.03,
                "minimum_lift_m": 0.01,
                "track_status": "tracked",
                "track_stability": "missing_current_frame",
                "fresh_visible_negative_evidence": False,
                "post_release_verification_required": True,
            },
        }
    }
    scene = agent._with_runtime_manipulation_state(scene)
    agent.memory_store.record_scene_memory(scene)
    assert scene.get("operation_targets", []) == []
    held = next(
        item
        for item in scene["instances"]
        if item["instance_id"] == "held"
    )
    assert not any(
        candidate.get("action_mode") == "place"
        for candidate in held.get("operation_pose_candidates", [])
    )

    clear_failed_grasp = agent._with_internal_recovery_context(
        [RecoveryToolCall(tool_name="open_gripper", args={"arm": "left"})]
    )
    assert [call.tool_name for call in clear_failed_grasp] == [
        "open_gripper"
    ]


def test_place_release_guard_rejects_mismatched_public_target() -> None:
    agent, scene = _place_test_agent()
    target_id = next(
        item["target_id"]
        for item in scene["operation_targets"]
        if item["target_kind"] == "vacated_pose"
    )

    guarded = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "arm": "left",
                    "action_mode": "place",
                    "target_id": target_id,
                    "point_key": "place_world_m",
                },
            ),
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "left",
                    "release_held_instance_id": "held",
                    "release_target_id": "place:unknown",
                },
            ),
        ]
    )

    assert [call.tool_name for call in guarded] == [
        "move_ee_to_grounded_instance",
        "reobserve_scene",
    ]
    assert "does not match" in guarded[-1].args["_guard_reason"]


def test_disabled_release_guard_does_not_rewrite_explicit_open() -> None:
    agent, scene = _place_test_agent(
        DummyConfig(release_guard_enabled=False)
    )
    target_id = next(
        item["target_id"]
        for item in scene["operation_targets"]
        if item["target_kind"] == "vacated_pose"
    )
    open_args = {
        "arm": "left",
        "release_held_instance_id": "wrong-object-label",
        "release_target_id": "place:wrong-target-label",
    }

    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "arm": "left",
                    "action_mode": "place",
                    "target_id": target_id,
                    "point_key": "place_world_m",
                },
            ),
            RecoveryToolCall(tool_name="open_gripper", args=open_args),
        ]
    )

    assert [call.tool_name for call in calls] == [
        "move_ee_to_grounded_instance",
        "open_gripper",
    ]
    assert calls[1].args == open_args


def test_disabled_release_guard_bypasses_release_batch_rewriters() -> None:
    agent = ImgAgent(
        make_card(DummyConfig(release_guard_enabled=False))
    )
    request = RecoveryToolCall(
        tool_name="open_gripper",
        args={"arm": "left"},
    )

    with mock.patch.object(
        agent,
        "_guard_recently_resolved_release_clearance",
        side_effect=AssertionError("release clearance guard was called"),
    ), mock.patch.object(
        agent,
        "_guard_pending_release_verification",
        side_effect=AssertionError("release verification guard was called"),
    ), mock.patch.object(
        agent,
        "_drop_redundant_support_contact_settle",
        side_effect=AssertionError("support settle rewrite was called"),
    ), mock.patch.object(
        agent,
        "_guard_open_gripper_while_holding",
        side_effect=AssertionError("open guard was called"),
    ):
        guarded = agent._apply_recovery_manipulation_guards([request])

    assert guarded == [request]


def test_disabled_release_guard_keeps_post_release_verification() -> None:
    agent, scene = _place_test_agent(
        DummyConfig(release_guard_enabled=False)
    )
    target_id = next(
        item["target_id"]
        for item in scene["operation_targets"]
        if item["target_kind"] == "vacated_pose"
    )
    place_calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "arm": "left",
                    "action_mode": "place",
                    "target_id": target_id,
                    "point_key": "place_world_m",
                },
            )
        ]
    )
    materialized = agent._grounded_instance_for_call(place_calls[0])
    selected = materialized["selected_operation_candidate"]
    agent._update_manipulation_state_from_effect(
        calls=place_calls,
        results=[
            RecoveryToolResult(
                tool_name="move_ee_to_grounded_instance",
                success=True,
                details={
                    "instance_id": "held",
                    "arm": "left",
                    "point_key": "place_world_m",
                    "target_reached": True,
                    "target_pose": selected["ee_target_pose"],
                    "operation_candidate_id": selected["candidate_id"],
                    "operation_target_id": target_id,
                    "operation_action_mode": "place",
                    "held_object_target_world_m": selected[
                        "held_object_target_world_m"
                    ],
                    "place_target_revalidated": True,
                    "support_valid": True,
                    "free": True,
                },
            )
        ],
        action_effect={
            "effect_verified": "true",
            "effect_type": "place_alignment",
        },
    )
    assert agent.memory_store.state.working.manipulation_state["left"][
        "phase"
    ] == "place_aligned"

    bare_open = agent._with_internal_recovery_context(
        [RecoveryToolCall(tool_name="open_gripper", args={"arm": "left"})]
    )
    assert [call.tool_name for call in bare_open] == ["open_gripper"]
    assert bare_open[0].args == {"arm": "left"}
    agent.latest_snapshot.raw["endpose"]["left_gripper"] = 1.0
    open_result = RecoveryToolResult(
        tool_name="open_gripper",
        success=True,
        details={"gripper_value": 1.0},
    )
    validation = agent._runtime_place_effect_validation(
        calls=bare_open,
        results=[open_result],
    )
    assert validation["applicable"] is True
    assert validation["release_verification_required"] is True
    assert validation["held_instance_id"] == "held"
    assert validation["target_id"] == target_id

    agent._update_manipulation_state_from_effect(
        calls=bare_open,
        results=[open_result],
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "release",
            "runtime_place_validation": validation,
        },
    )
    pending = agent.memory_store.state.working.manipulation_state["left"]
    assert pending["phase"] == "release_pending_verification"
    assert pending["held_instance_id"] == "held"
    assert pending["place_target_id"] == target_id
    follow_up = agent._runtime_place_effect_validation(
        calls=[],
        results=[],
        pending_arm="left",
    )
    assert follow_up["applicable"] is True
    assert follow_up["release_verification_required"] is True


def test_opened_place_release_remains_pending_until_post_verification() -> None:
    agent, scene = _place_test_agent()
    target_id = next(
        item["target_id"]
        for item in scene["operation_targets"]
        if item["target_kind"] == "vacated_pose"
    )
    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "arm": "left",
                    "action_mode": "place",
                    "target_id": target_id,
                    "point_key": "place_world_m",
                },
            ),
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "left",
                    "release_held_instance_id": "held",
                    "release_target_id": target_id,
                },
            ),
        ]
    )
    selected = calls[1].args["_release_place_candidate"]
    results = [
        RecoveryToolResult(
            tool_name="move_ee_to_grounded_instance",
            success=True,
            details={
                "instance_id": "held",
                "arm": "left",
                "point_key": "place_world_m",
                "target_reached": True,
                "target_pose": selected["ee_target_pose"],
                "operation_candidate_id": selected["candidate_id"],
                "operation_target_id": target_id,
                "operation_action_mode": "place",
                "held_object_target_world_m": selected[
                    "held_object_target_world_m"
                ],
                "place_target_revalidated": True,
                "support_valid": True,
                "free": True,
            },
        ),
        RecoveryToolResult(
            tool_name="open_gripper",
            success=True,
            details={"gripper_value": 1.0},
        ),
    ]

    agent._update_manipulation_state_from_effect(
        calls=calls,
        results=results,
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "release",
            "runtime_place_validation": {
                "verified": False,
                "arm": "left",
            },
        },
    )

    pending = agent.memory_store.state.working.manipulation_state["left"]
    assert pending["phase"] == "release_pending_verification"
    assert pending["holding_confirmed"] is False
    assert pending["place_target_id"] == target_id
    assert pending["held_instance_id"] == "held"
    assert pending["pregrasp_object_world_m"] == [0.0, 0.0, 0.02]
    assert pending["pregrasp_object_contact_world_m"] == [
        0.0,
        0.0,
        0.04,
    ]
    leases = agent._pending_release_identity_relocation_leases()
    assert len(leases) == 1
    assert leases[0]["instance_ref"] == "held"
    assert leases[0]["recovery_anchors"]


def test_failed_release_validation_transitions_pending_to_recovery() -> None:
    agent, _ = _place_test_agent()
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "place_target_id": "place:generic-target",
            "held_object_target_world_m": [0.0, 0.0, 0.02],
            "held_extent_m": [0.04, 0.04, 0.04],
            "release_ee_target_world_m": [0.0, 0.0, 0.10],
            "pre_release_validated": True,
            "source_skill_id": "place-skill",
            "updated_step": 1,
        }
    }
    validation = {
        "applicable": True,
        "release_verification_required": True,
        "verified": False,
        "arm": "left",
        "held_instance_id": "held",
        "target_id": "place:generic-target",
        "pre_release_validated": True,
        "release_executed": True,
        "gripper_open": True,
        "object_position_observed": True,
        "object_at_target": False,
        "object_target_error_m": 0.04,
        "target_tolerance_m": 0.02,
        "ee_cleared_from_release_pose": True,
        "detachment_verified": True,
        "stable_across_fresh_observations": False,
        "placement_recovery_required": True,
    }

    agent._update_manipulation_state_from_effect(
        calls=[],
        results=[],
        action_effect={
            "effect_verified": "false",
            "effect_type": "place",
            "runtime_place_validation": validation,
        },
    )

    recovery = agent.memory_store.state.working.manipulation_state[
        "left"
    ]
    assert recovery["phase"] == "release_recovery_required"
    assert recovery["released_instance_id"] == "held"
    assert recovery["holding_confirmed"] is False
    assert recovery["object_target_error_m"] == 0.04
    assert recovery["target_tolerance_m"] == 0.02
    assert any(
        entry.startswith("runtime_release_recovery_required:")
        for entry in agent.memory_store.state.working.recovery_history
    )


def test_place_release_cannot_complete_when_runtime_validation_is_false() -> None:
    agent, scene = _place_test_agent()
    target_id = next(
        item["target_id"]
        for item in scene["operation_targets"]
        if item["target_kind"] == "vacated_pose"
    )
    calls = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args={
                    "arm": "left",
                    "action_mode": "place",
                    "target_id": target_id,
                    "point_key": "place_world_m",
                },
            ),
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "left",
                    "release_held_instance_id": "held",
                    "release_target_id": target_id,
                },
            ),
        ]
    )
    results = [
        RecoveryToolResult(
            tool_name="move_ee_to_grounded_instance",
            success=True,
            details={
                "instance_id": "held",
                "arm": "left",
                "target_reached": True,
                "operation_action_mode": "place",
                "operation_target_id": target_id,
                "place_target_revalidated": True,
                "support_valid": True,
                "free": True,
            },
        ),
        RecoveryToolResult(
            tool_name="open_gripper",
            success=True,
            details={"gripper_value": 1.0},
        ),
    ]
    with mock.patch.object(
        agent._action_effect_verifier,
        "verify",
        return_value={
            "effect_verified": "true",
            "effect_type": "align",
            "confidence": 0.9,
            "evidence_summary": "planner claimed completion",
            "failure_reason": "",
            "next_constraint": "",
            "memory_update": "",
            "subtask_status": "completed",
            "recommended_control": "continue",
        },
    ):
        result = agent._verify_action_effect(
            "task-level-recovery-control",
            current_subtask="place the held object",
            post_recovery_intent="retry",
            expected_outcome="object placed",
            recovery_reason="test",
            calls=calls,
            results=results,
            pre={},
            post={},
        )

    assert result["effect_verified"] == "unverified"
    assert result["subtask_status"] == "uncertain"
    assert result["recommended_control"] == "retry"
    assert result["runtime_place_validation"]["verified"] is False


def test_detached_release_outside_target_forces_recovery_handoff() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    validation = {
        "applicable": True,
        "release_verification_required": True,
        "verified": False,
        "arm": "left",
        "held_instance_id": "released",
        "target_id": "place:generic-target",
        "pre_release_validated": True,
        "release_executed": True,
        "gripper_open": True,
        "object_position_observed": True,
        "object_at_target": False,
        "object_target_error_m": 0.04,
        "target_tolerance_m": 0.02,
        "ee_cleared_from_release_pose": True,
        "detachment_verified": True,
        "stable_across_fresh_observations": False,
        "placement_recovery_required": True,
    }
    with (
        mock.patch.object(
            agent,
            "_runtime_place_effect_validation",
            return_value=validation,
        ),
        mock.patch.object(
            agent._action_effect_verifier,
            "verify",
            return_value={
                "effect_verified": "true",
                "effect_type": "place",
                "confidence": 0.9,
                "evidence_summary": "planner claimed placement",
                "failure_reason": "",
                "next_constraint": "",
                "memory_update": "",
                "subtask_status": "completed",
                "recommended_control": "continue",
            },
        ),
    ):
        result = agent._verify_action_effect(
            "task-level-recovery-control",
            current_subtask="place the released object",
            post_recovery_intent="retry",
            expected_outcome="object placed",
            recovery_reason="test",
            calls=[],
            results=[],
            pre={},
            post={},
        )

    assert result["effect_verified"] == "false"
    assert result["effect_type"] == "place"
    assert result["subtask_status"] == "failed"
    assert result["recommended_control"] == "replan"
    assert "reacquire the released instance" in result["next_constraint"]


def test_pending_place_release_clears_only_after_detachment_and_stability() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.left_endpose[:] = np.asarray(
        [0.10, 0.10, 0.20, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "place_target_id": "place:vacated:held",
            "held_object_target_world_m": [0.0, 0.0, 0.02],
            "held_extent_m": [0.04, 0.04, 0.04],
            "release_ee_target_world_m": [0.01, -0.01, 0.06],
            "pre_release_validated": True,
        }
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "held",
                    "track_id": "held",
                    "world_m": [0.0, 0.0, 0.02],
                    "latest_world_m": [0.0, 0.0, 0.02],
                    "quality": {
                        "world_extent_m": [0.04, 0.04, 0.04]
                    },
                }
            ],
            "temporal_memory": {
                "tracks": [
                    {
                        "track_id": "held",
                        "history": [
                            {
                                "observation_capture_id": 1,
                                "world_m": [0.001, 0.0, 0.02],
                            },
                            {
                                "observation_capture_id": 2,
                                "world_m": [0.0, 0.0, 0.02],
                            },
                        ],
                    }
                ]
            },
        }
    )

    validation = agent._runtime_place_effect_validation(
        calls=[],
        results=[],
    )
    assert validation["verified"] is True
    assert validation["object_at_target"] is True
    assert validation["ee_cleared_from_release_pose"] is True
    assert validation["stable_across_fresh_observations"] is True
    assert validation["placement_recovery_required"] is False

    agent._update_manipulation_state_from_effect(
        calls=[],
        results=[],
        action_effect={
            "effect_verified": "true",
            "effect_type": "place",
            "runtime_place_validation": validation,
        },
    )
    assert "left" not in agent.memory_store.state.working.manipulation_state


def test_pending_release_accepts_position_with_action_geometry_quarantined() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.step_count = 13
    snapshot.left_endpose[:] = np.asarray(
        [0.20, 0.10, 0.95, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    target_world = [0.110159, -0.112888, 0.774796]
    first_world = [0.093469, -0.110745, 0.777810]
    second_world = [0.094169, -0.110445, 0.777610]
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "place_target_id": "place:generic-target",
            "held_object_target_world_m": target_world,
            "held_extent_m": [0.04, 0.04, 0.04],
            "release_ee_target_world_m": [
                0.110159,
                -0.112888,
                0.90,
            ],
            "pre_release_validated": True,
            "updated_step": 12,
        }
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "held",
                    "track_id": "held",
                    "status": "visible",
                    "stability": "relocation_geometry_pending",
                    "position_state": "current_verified",
                    "last_verified_step": 13,
                    "world_m": second_world,
                    "latest_world_m": second_world,
                    "top_surface_world_m": None,
                    "approach_world_m": None,
                    "grasp_world_m": None,
                    "contact_world_m": None,
                    "operation_pose_candidates": [],
                    "action_geometry_state": "relocation_pending",
                }
            ],
            "temporal_memory": {
                "tracks": [
                    {
                        "track_id": "held",
                        "history": [
                            {
                                "env_step": 12,
                                "observation_capture_id": 12,
                                "world_m": first_world,
                            },
                            {
                                "env_step": 13,
                                "observation_capture_id": 13,
                                "world_m": second_world,
                            },
                        ],
                    }
                ]
            },
        }
    )

    validation = agent._runtime_place_effect_validation(
        calls=[],
        results=[],
    )

    assert validation["object_position_fresh"] is True
    assert validation["object_position_observed"] is True
    assert validation["object_at_target"] is True
    assert validation["stable_across_fresh_observations"] is True
    assert validation["detachment_verified"] is True
    assert validation["verified"] is True

    stored_scene = agent.memory_store.state.working.scene_memory
    stored_scene["temporal_memory"]["tracks"][0]["history"][1][
        "observation_capture_id"
    ] = 12
    duplicate_capture = agent._runtime_place_effect_validation(
        calls=[],
        results=[],
    )
    assert duplicate_capture["object_position_fresh"] is True
    assert duplicate_capture["object_at_target"] is True
    assert duplicate_capture["stable_across_fresh_observations"] is False
    assert duplicate_capture["verified"] is False


def test_place_validation_reports_ee_clearance_independently_of_target_error() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.left_endpose[:] = np.asarray(
        [0.10, 0.10, 0.20, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "place_target_id": "place:vacated:held",
            "held_object_target_world_m": [0.0, 0.0, 0.02],
            "held_extent_m": [0.04, 0.04, 0.04],
            "release_ee_target_world_m": [0.01, -0.01, 0.06],
            "pre_release_validated": True,
            "updated_step": 0,
        }
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "held",
                    "track_id": "held",
                    "world_m": [0.05, 0.0, 0.02],
                    "latest_world_m": [0.05, 0.0, 0.02],
                }
            ],
            "temporal_memory": {
                "tracks": [
                    {
                        "track_id": "held",
                        "history": [
                            {
                                "env_step": 0,
                                "world_m": [0.05, 0.0, 0.02],
                            },
                            {
                                "env_step": 0,
                                "world_m": [0.05, 0.0, 0.02],
                            },
                        ],
                    }
                ]
            },
        }
    )

    validation = agent._runtime_place_effect_validation(
        calls=[],
        results=[],
    )

    assert validation["verified"] is False
    assert validation["object_at_target"] is False
    assert validation["ee_release_clearance_m"] > 0.02
    assert validation["ee_cleared_from_release_pose"] is True
    assert validation["detachment_verified"] is True
    assert validation["object_position_observed"] is True
    assert validation["placement_recovery_required"] is True


def test_pending_release_ignores_pre_release_tracked_geometry() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.left_endpose[:] = np.asarray(
        [0.10, 0.10, 0.20, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "place_target_id": "place:generic-target",
            "held_object_target_world_m": [0.0, 0.0, 0.02],
            "held_extent_m": [0.04, 0.04, 0.04],
            "release_ee_target_world_m": [0.01, -0.01, 0.06],
            "pre_release_validated": True,
            "updated_step": 5,
        }
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "held",
                    "track_id": "held",
                    "status": "tracked",
                    "stability": "missing_current_frame",
                    "last_seen_step": 2,
                    "world_m": [0.20, 0.0, 0.02],
                    "latest_world_m": [0.20, 0.0, 0.02],
                }
            ],
            "temporal_memory": {
                "tracks": [
                    {
                        "track_id": "held",
                        "history": [
                            {
                                "env_step": 2,
                                "world_m": [0.20, 0.0, 0.02],
                            }
                        ],
                    }
                ]
            },
        }
    )

    validation = agent._runtime_place_effect_validation(
        calls=[],
        results=[],
    )

    assert validation["object_position_fresh"] is False
    assert validation["object_position_observed"] is False
    assert validation["object_target_error_m"] is None
    assert validation["object_at_target"] is False
    assert validation["placement_recovery_required"] is False


def test_pending_release_at_target_blocks_additional_physical_clearance() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.left_endpose[:] = np.asarray(
        [0.10, 0.10, 0.20, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "place_target_id": "place:vacated:held",
            "held_object_target_world_m": [0.0, 0.0, 0.02],
            "held_extent_m": [0.04, 0.04, 0.04],
            "release_ee_target_world_m": [0.01, -0.01, 0.06],
            "pre_release_validated": True,
            "updated_step": 0,
        }
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "held",
                    "track_id": "held",
                    "world_m": [0.0, 0.0, 0.02],
                    "latest_world_m": [0.0, 0.0, 0.02],
                }
            ],
            "temporal_memory": {
                "tracks": [
                    {
                        "track_id": "held",
                        "history": [
                            {
                                "env_step": 0,
                                "world_m": [0.0, 0.0, 0.02],
                            }
                        ],
                    }
                ]
            },
        }
    )

    before = agent._runtime_place_effect_validation(calls=[], results=[])
    guarded = agent._apply_recovery_manipulation_guards(
        [
            RecoveryToolCall(
                tool_name="retreat_arm",
                args={
                    "arm": "left",
                    "axis": "x",
                    "direction": "negative",
                    "distance": 0.03,
                    "steps": 2,
                },
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert before["object_at_target"] is True
    assert before["ee_cleared_from_release_pose"] is True
    assert before["detachment_verified"] is True
    assert before["stable_across_fresh_observations"] is False
    assert before["placement_recovery_required"] is False
    assert [call.tool_name for call in guarded] == ["reobserve_scene"]
    assert guarded[0].args["_release_verification_arms"] == ["left"]
    assert "without moving the robot" in guarded[0].args["_guard_reason"]
    assert (
        agent.memory_store.state.working.manipulation_state["left"]["phase"]
        == "release_pending_verification"
    )


def test_pending_release_outside_target_allows_physical_recovery() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.left_endpose[:] = np.asarray(
        [0.10, 0.10, 0.20, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "place_target_id": "place:vacated:held",
            "held_object_target_world_m": [0.0, 0.0, 0.02],
            "held_extent_m": [0.04, 0.04, 0.04],
            "release_ee_target_world_m": [0.01, -0.01, 0.06],
            "pre_release_validated": True,
            "updated_step": 0,
        }
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "held",
                    "track_id": "held",
                    "world_m": [0.05, 0.0, 0.02],
                    "latest_world_m": [0.05, 0.0, 0.02],
                }
            ],
            "temporal_memory": {
                "tracks": [
                    {
                        "track_id": "held",
                        "history": [
                            {
                                "env_step": 0,
                                "world_m": [0.05, 0.0, 0.02],
                            }
                        ],
                    }
                ]
            },
        }
    )
    calls = [
        RecoveryToolCall(
            tool_name="retreat_arm",
            args={
                "arm": "left",
                "axis": "x",
                "direction": "negative",
                "distance": 0.03,
            },
        )
    ]

    before = agent._runtime_place_effect_validation(calls=[], results=[])
    guarded = agent._apply_recovery_manipulation_guards(calls)

    assert before["object_at_target"] is False
    assert before["placement_recovery_required"] is True
    assert guarded == calls
    recovery = agent.memory_store.state.working.manipulation_state[
        "left"
    ]
    assert recovery["phase"] == "release_recovery_required"
    assert recovery["released_instance_id"] == "held"
    assert recovery["object_target_error_m"] > recovery[
        "target_tolerance_m"
    ]
    assert (
        recovery["placement_failure_reason"]
        == "released_object_outside_target_tolerance"
    )


def test_trace_off_target_observation_resolves_release_to_recovery() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.step_count = 38
    snapshot.left_endpose[:] = np.asarray(
        [0.20, -0.10, 1.00, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic multi-stage manipulation",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )

    def observed_component(
        rank: int,
        bbox: list[int],
        world: list[float],
        extent: list[float],
        score: float,
    ) -> dict:
        return {
            "rank": rank,
            "score": score,
            "bbox_xyxy": bbox,
            "centroid_px": [
                (bbox[0] + bbox[2]) / 2.0,
                (bbox[1] + bbox[3]) / 2.0,
            ],
            "mask_path": f"/tmp/release_trace_{rank}.png",
            "grounding_3d": {
                "success": True,
                "centroid_world": world,
                "bbox_world_min": [
                    world[index] - extent[index] / 2.0
                    for index in range(3)
                ],
                "bbox_world_max": [
                    world[index] + extent[index] / 2.0
                    for index in range(3)
                ],
                "top_surface_world": [
                    world[0],
                    world[1],
                    world[2] + extent[2] / 2.0,
                ],
                "approach_point_world": [
                    world[0],
                    world[1],
                    world[2] + extent[2] / 2.0 + 0.08,
                ],
            },
        }

    initial = agent._scene_memory_tracker.update(
        segmentation=[
            {
                "success": True,
                "object_id": "component",
                "query_role": "target",
                "detections": [
                    observed_component(
                        0,
                        [126, 160, 138, 174],
                        [0.102096, -0.114293, 0.771531],
                        [0.039063, 0.062372, 0.039967],
                        0.72,
                    )
                ],
            }
        ],
        env_step=22,
        global_task="generic multi-stage manipulation",
        current_subtask="move the selected component",
    )
    held_id = initial["instances"][0]["track_id"]
    agent.memory_store.record_scene_memory(initial)
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": held_id,
            "holding_confirmed": False,
            "place_target_id": "place:generic-target",
            "held_object_target_world_m": [
                0.098879,
                -0.201065,
                0.793909,
            ],
            "held_extent_m": [0.037854, 0.076441, 0.103114],
            "release_ee_target_world_m": [
                0.097855,
                -0.200231,
                0.962013,
            ],
            "pregrasp_object_world_m": [
                0.102096,
                -0.114293,
                0.771531,
            ],
            "pregrasp_object_contact_world_m": [
                0.093796,
                -0.130835,
                0.797822,
            ],
            "pre_release_validated": True,
            "updated_step": 36,
        }
    }
    observed = agent._scene_memory_tracker.update(
        segmentation=[
            {
                "success": True,
                "object_id": "component",
                "query_role": "context",
                "instance_ref": held_id,
                "identity_binding_required": True,
                "detections": [
                    observed_component(
                        0,
                        [126, 160, 138, 174],
                        [0.102377, -0.113884, 0.771712],
                        [0.038657, 0.058686, 0.039961],
                        0.777344,
                    ),
                    observed_component(
                        1,
                        [219, 162, 237, 179],
                        [-0.244555, -0.094243, 0.778213],
                        [0.119198, 0.094158, 0.168748],
                        0.621094,
                    ),
                    observed_component(
                        2,
                        [127, 160, 138, 174],
                        [0.101836, -0.113577, 0.771952],
                        [0.037443, 0.054045, 0.039961],
                        0.148438,
                    ),
                ],
            }
        ],
        env_step=38,
        global_task="generic multi-stage manipulation",
        current_subtask="verify the released component",
        identity_relocation_leases=(
            agent._pending_release_identity_relocation_leases()
        ),
        position_events=agent._runtime_scene_position_events(),
    )
    agent.memory_store.record_scene_memory(
        agent._with_runtime_manipulation_state(observed)
    )

    validation = agent._resolve_pending_release_from_runtime(
        source="trace_replay",
    )

    assert validation["object_position_observed"] is True
    assert validation["object_at_target"] is False
    assert validation["object_target_error_m"] > 0.08
    assert validation["placement_recovery_required"] is True
    recovery = agent.memory_store.state.working.manipulation_state[
        "left"
    ]
    assert recovery["phase"] == "release_recovery_required"
    assert recovery["released_instance_id"] == held_id


def test_fresh_runtime_release_resolution_blocks_stale_clearance_batch() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.left_endpose[:] = np.asarray(
        [0.10, 0.10, 0.20, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic placement",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "release_pending_verification",
            "held_instance_id": "held",
            "holding_confirmed": False,
            "place_target_id": "place:vacated:held",
            "held_object_target_world_m": [0.0, 0.0, 0.02],
            "held_extent_m": [0.04, 0.04, 0.04],
            "release_ee_target_world_m": [0.01, -0.01, 0.06],
            "pre_release_validated": True,
            "updated_step": 0,
        }
    }
    agent.memory_store.record_scene_memory(
        {
            "instances": [
                {
                    "instance_id": "held",
                    "track_id": "held",
                    "world_m": [0.0, 0.0, 0.02],
                    "latest_world_m": [0.0, 0.0, 0.02],
                }
            ],
            "temporal_memory": {
                "tracks": [
                    {
                        "track_id": "held",
                        "history": [
                            {
                                "env_step": 0,
                                "observation_capture_id": 1,
                                "world_m": [0.001, 0.0, 0.02],
                            },
                            {
                                "env_step": 0,
                                "observation_capture_id": 2,
                                "world_m": [0.0, 0.0, 0.02],
                            },
                        ],
                    }
                ]
            },
        }
    )

    validation = agent._resolve_pending_release_from_runtime(
        source="test_scene_memory_update",
    )
    agent._recent_release_resolutions["left"][
        "active_skill_id"
    ] = "release_skill"
    with mock.patch.object(
        agent,
        "_active_skill_id",
        return_value="replacement_clearance_skill",
    ):
        guarded = agent._apply_recovery_manipulation_guards(
            [
                RecoveryToolCall(
                    tool_name="retreat_arm",
                    args={
                        "arm": "left",
                        "axis": "x",
                        "direction": "negative",
                        "distance": 0.05,
                    },
                ),
                RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={},
                ),
            ]
        )

    assert validation["verified"] is True
    assert "left" not in agent.memory_store.state.working.manipulation_state
    assert [call.tool_name for call in guarded] == ["reobserve_scene"]
    assert "already resolved" in guarded[0].args["_guard_reason"]


def _prepare_evidence_policy_agent(
    *,
    phase: str,
    capture_id: int = 11,
    pure_tool_control: bool = False,
    grasp_transport_policy: str = "strict",
    release_guard_enabled: bool = True,
) -> ImgAgent:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                pure_tool_control_enabled=pure_tool_control,
                grasp_transport_policy=grasp_transport_policy,
                release_guard_enabled=release_guard_enabled,
            )
        )
    )
    snapshot = make_snapshot_with_gripper(0.0)
    snapshot.step_count = 11
    snapshot.left_endpose[:] = np.asarray(
        [0.10, -0.19, 0.93, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic manipulation",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": phase,
            "held_instance_id": "track_0001",
            "holding_confirmed": False,
            "transport_authorized": False,
            "grasp_attempt_nonce": "left:track_0001:candidate:9",
        }
    }
    agent.memory_store.record_observation_preprocess(
        {
            "observation_capture_id": capture_id,
            "segmentation": [
                {"camera": "head", "success": True},
                {"camera": "third", "success": True},
            ],
        }
    )
    return agent


def test_stationary_reobserve_budget_survives_skill_replacement() -> None:
    agent = _prepare_evidence_policy_agent(phase="grasp_candidate")
    agent.memory_store.state.working.manipulation_state["left"][
        "diagnostic_lift_evidence"
    ] = {"lift_env_step": 10}
    observation = RecoveryToolCall(
        tool_name="reobserve_scene",
        args={},
    )

    with mock.patch.object(agent, "_active_skill_id", return_value="skill_1"):
        first = agent._apply_evidence_acquisition_policy([observation])
    with mock.patch.object(agent, "_active_skill_id", return_value="skill_2"):
        second = agent._apply_evidence_acquisition_policy([observation])

    assert [call.tool_name for call in first] == ["reobserve_scene"]
    assert first[0].args["_evidence_acquisition"][
        "stationary_reobserves_used"
    ] == 1
    assert second == []
    assert agent._last_evidence_acquisition_decision[
        "next_information_action"
    ] == "perform_controlled_return_and_regrasp"
    assert agent._last_evidence_acquisition_decision["skill_id"] == "skill_2"


def test_evidence_only_ambiguous_grasp_never_stages_controlled_return() -> None:
    agent = _prepare_evidence_policy_agent(
        phase="grasp_candidate",
        grasp_transport_policy="evidence_only",
    )
    agent.memory_store.state.working.manipulation_state["left"][
        "diagnostic_lift_evidence"
    ] = {"lift_env_step": 10}
    observation = RecoveryToolCall(tool_name="reobserve_scene", args={})

    first = agent._apply_evidence_acquisition_policy([observation])
    second = agent._apply_evidence_acquisition_policy([observation])

    assert [call.tool_name for call in first] == ["reobserve_scene"]
    assert second == []
    assert agent._last_evidence_acquisition_decision[
        "next_information_action"
    ] == "continue_without_attachment_guard"
    assert agent.memory_store.state.working.manipulation_state["left"][
        "phase"
    ] == "grasp_candidate"


def test_disabled_release_guard_preserves_planner_open_through_evidence_policy() -> None:
    agent = _prepare_evidence_policy_agent(
        phase="ambiguous_grasp_return_pending",
        release_guard_enabled=False,
    )
    requested = RecoveryToolCall(
        tool_name="open_gripper",
        args={"arm": "left"},
    )

    with mock.patch.object(
        agent,
        "_stage_ambiguous_grasp_return",
        side_effect=AssertionError("planner open was taken over"),
    ):
        guarded = agent._apply_evidence_acquisition_policy(
            [requested],
            planner_explicit_open=True,
        )

    assert guarded == [requested]
    assert guarded[0].args == {"arm": "left"}


def test_disabled_release_guard_without_planner_open_keeps_return_takeover() -> None:
    agent = _prepare_evidence_policy_agent(
        phase="ambiguous_grasp_return_pending",
        release_guard_enabled=False,
    )
    observation = RecoveryToolCall(
        tool_name="reobserve_scene",
        args={},
    )
    replacement = [
        RecoveryToolCall(
            tool_name="lift_ee",
            args={"arm": "left", "distance": 0.02},
        )
    ]

    with mock.patch.object(
        agent,
        "_stage_ambiguous_grasp_return",
        return_value=replacement,
    ) as takeover:
        guarded = agent._apply_evidence_acquisition_policy([observation])

    assert guarded == replacement
    takeover.assert_called_once()


def test_dispatch_preserves_original_planner_open_when_release_guard_disabled() -> None:
    agent = _prepare_evidence_policy_agent(
        phase="ambiguous_grasp_return_pending",
        release_guard_enabled=False,
    )
    requested = RecoveryToolCall(
        tool_name="open_gripper",
        args={"arm": "left"},
    )
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="planner-explicit-release",
            tool_calls=[requested],
            expected_outcome="open the selected gripper",
        ),
        post_recovery_intent="retry",
        selected_arm="left",
        reason="planner explicitly selected release",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="execute planner release",
    )
    open_result = RecoveryToolResult(
        tool_name="open_gripper",
        success=True,
    )

    with (
        mock.patch.object(
            agent._recovery_dispatcher,
            "available_tools",
            return_value=["open_gripper"],
        ),
        mock.patch.object(
            agent._recovery_policy_resolver,
            "resolve",
            return_value=route,
        ),
        mock.patch.object(
            agent,
            "_stage_ambiguous_grasp_return",
            side_effect=AssertionError("planner open was taken over"),
        ),
        mock.patch.object(
            agent._recovery_dispatcher,
            "dispatch_batch",
            return_value=[open_result],
        ) as dispatch,
        mock.patch.object(
            agent,
            "_verify_action_effect",
            return_value={"effect_verified": "unverified"},
        ),
        mock.patch.object(
            agent,
            "_complete_recovery_effect_transition",
            return_value="retry",
        ),
    ):
        preferred_action = agent._dispatch_recovery_tools(
            task_env=object(),
            signal=signal,
        )

    assert preferred_action == "retry"
    dispatched = dispatch.call_args.args[0]
    assert dispatched == [requested]
    assert dispatched[0].args == {"arm": "left"}


def test_grasp_without_diagnostic_lift_requests_lift_instead_of_empty_return() -> None:
    agent = _prepare_evidence_policy_agent(
        phase="grasp_candidate",
        pure_tool_control=True,
    )
    observation = RecoveryToolCall(
        tool_name="reobserve_scene",
        args={},
    )

    first = agent._apply_evidence_acquisition_policy([observation])
    assert [call.tool_name for call in first] == ["reobserve_scene"]

    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="diagnostic-lift-pending",
            tool_calls=[observation],
            expected_outcome="acquire grasp evidence",
        ),
        post_recovery_intent="retry",
        selected_arm="none",
        reason="grasp attachment remains unverified",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="continue grasp verification",
    )
    with (
        mock.patch.object(
            agent._recovery_dispatcher,
            "available_tools",
            return_value=["reobserve_scene", "lift_ee"],
        ),
        mock.patch.object(
            agent._recovery_policy_resolver,
            "resolve",
            return_value=route,
        ),
        mock.patch.object(
            agent._recovery_dispatcher,
            "dispatch_batch",
        ) as dispatch,
    ):
        preferred_action = agent._dispatch_recovery_tools(
            task_env=object(),
            signal=signal,
        )

    assert preferred_action == "retry"
    dispatch.assert_not_called()
    decision = agent._last_evidence_acquisition_decision
    assert decision["next_information_action"] == (
        "perform_bounded_grasp_verification_motion"
    )
    assert decision["diagnostic_lift_required"] is True
    assert decision["exhausted_reason"] == (
        "diagnostic_lift_evidence_missing"
    )
    assert "controlled_return" not in decision["next_information_action"]
    assert agent.memory_store.state.working.manipulation_state["left"][
        "phase"
    ] == "grasp_candidate"


def test_occlusion_never_spends_a_stationary_reobserve() -> None:
    agent = _prepare_evidence_policy_agent(phase="holding")

    guarded = agent._apply_evidence_acquisition_policy(
        [
            RecoveryToolCall(
                tool_name="reobserve_scene",
                args={
                    "_evidence_situation": "occlusion",
                    "_evidence_track_id": "track_0001",
                    "_evidence_arm": "left",
                    "_evidence_phase": "self_occluded_grounded_instance",
                },
            )
        ]
    )

    assert guarded == []
    decision = agent._last_evidence_acquisition_decision
    assert decision["stationary_reobserves_used"] == 0
    assert decision["stationary_reobserve_limit"] == 0
    assert decision["next_information_action"] == (
        "change_viewpoint_or_clear_occlusion"
    )


def test_release_stability_counts_distinct_captures_not_skill_names() -> None:
    agent = _prepare_evidence_policy_agent(
        phase="release_pending_verification",
        capture_id=20,
    )
    observation = RecoveryToolCall(tool_name="reobserve_scene", args={})

    with mock.patch.object(agent, "_active_skill_id", return_value="release"):
        first = agent._apply_evidence_acquisition_policy([observation])
    agent.memory_store.record_observation_preprocess(
        {
            "observation_capture_id": 21,
            "segmentation": [
                {"camera": "head", "success": True},
                {"camera": "third", "success": True},
            ],
        }
    )
    with mock.patch.object(agent, "_active_skill_id", return_value="verify"):
        second = agent._apply_evidence_acquisition_policy([observation])

    assert [call.tool_name for call in first] == ["reobserve_scene"]
    assert second == []
    assert agent._last_evidence_acquisition_decision[
        "next_information_action"
    ] == "evaluate_release_stability"
    assert agent._last_evidence_acquisition_decision[
        "distinct_capture_count"
    ] == 2


def test_reobserve_ablation_keeps_physical_action_and_removes_observation() -> None:
    agent = _prepare_evidence_policy_agent(phase="grasp_candidate")
    agent._recovery_dispatcher.set_reobserve_scene_enabled(False)
    lift = RecoveryToolCall(
        tool_name="lift_ee",
        args={
            "arm": "left",
            "distance": 0.025,
            "_runtime_grasp_diagnostic_lift": True,
        },
    )

    guarded = agent._apply_evidence_acquisition_policy(
        [lift, RecoveryToolCall(tool_name="reobserve_scene", args={})]
    )

    assert guarded == [lift]
    assert agent.current_status()["reobserve_scene_enabled"] is False
    assert agent._last_evidence_acquisition_decision[
        "next_information_action"
    ] == "use_fresh_snapshot_from_physical_action"


def test_reobserve_ablation_blocks_stationary_internal_call() -> None:
    agent = _prepare_evidence_policy_agent(phase="holding")
    agent._recovery_dispatcher.set_reobserve_scene_enabled(False)

    guarded = agent._apply_evidence_acquisition_policy(
        [RecoveryToolCall(tool_name="reobserve_scene", args={})]
    )

    assert guarded == []
    assert agent._last_evidence_acquisition_decision[
        "exhausted_reason"
    ] == "disabled_by_ablation"


def test_zero_call_controlled_return_reports_internal_progress() -> None:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                pure_tool_control_enabled=True,
            )
        )
    )
    snapshot = make_snapshot_with_gripper(1.0)
    snapshot.step_count = 34
    grasp = [0.12, -0.08, 0.90]
    approach = [0.12, -0.08, 0.98]
    quaternion = [1.0, 0.0, 0.0, 0.0]
    snapshot.left_endpose[:] = np.asarray(
        [*approach, *quaternion],
        dtype=np.float32,
    )
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic manipulation",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "left": {
            "phase": "ambiguous_grasp_return_pending",
            "held_instance_id": "track_0001",
            "grasp_attempt_nonce": "left:track_0001:candidate:9",
            "holding_confirmed": False,
            "transport_authorized": False,
            "grasp_ee_target_world_m": grasp,
            "grasp_approach_world_m": approach,
            "diagnostic_lift_evidence": {
                "lift_env_step": 31,
            },
            "ambiguous_grasp_return": {
                "status": "approach_reached_confirmation_pending",
                "arm": "left",
                "held_instance_id": "track_0001",
                "grasp_attempt_nonce": "left:track_0001:candidate:9",
                "grasp_world_m": grasp,
                "approach_world_m": approach,
                "quat_wxyz": quaternion,
                "last_physical_step": 33,
            },
        }
    }
    agent._recovery_dispatcher.set_reobserve_scene_enabled(False)
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="resume-ambiguous-grasp-return",
            tool_calls=[
                RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={},
                )
            ],
            expected_outcome="confirm exact return",
        ),
        post_recovery_intent="retry",
        selected_arm="none",
        reason="resume the persisted return transaction",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="resume ambiguous grasp return",
    )
    with (
        mock.patch.object(
            agent._recovery_dispatcher,
            "available_tools",
            return_value=["lift_ee"],
        ),
        mock.patch.object(
            agent._recovery_policy_resolver,
            "resolve",
            return_value=route,
        ),
        mock.patch.object(
            agent._recovery_dispatcher,
            "dispatch_batch",
        ) as dispatch,
    ):
        preferred_action = agent._dispatch_recovery_tools(
            task_env=object(),
            signal=signal,
        )

    assert preferred_action == _RECOVERY_INTERNAL_PROGRESS
    dispatch.assert_not_called()
    assert "left" not in (
        agent.memory_store.state.working.manipulation_state
    )
    assert agent._last_evidence_acquisition_decision[
        "controlled_return_zero_call_completion"
    ] is True

    agent._pure_tool_control_empty_plan_turns = 1
    agent._pure_tool_control_no_progress_control_turns = 3
    agent._continue_debug_recovery_pure_control(
        preferred_action=preferred_action,
        signal=signal,
        reason=signal.reason,
    )

    assert agent._pure_tool_control_empty_plan_turns == 0
    assert agent._pure_tool_control_no_progress_control_turns == 3
    history = agent.memory_store.state.working.recovery_history
    assert any(
        item.startswith("pure_tool_control_internal_recovery_progress:")
        for item in history
    )
    assert not any(
        item.startswith("pure_tool_control_empty_recovery_plan:")
        for item in history
    )

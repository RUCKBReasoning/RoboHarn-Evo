from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest import mock

from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.decisions import ControlSignal
from policy.roboharn_evo.agent.monitoring.signals import TASK_LEVEL_RECOVERY_CONTROL
from policy.roboharn_evo.agent.recovery.recovery_policies import RecoveryRoute
from policy.roboharn_evo.agent.recovery.recovery_primitives import RecoveryPrimitivePlan
from policy.roboharn_evo.agent.recovery.action_effect_verifier import (
    ActionEffectVerifier,
)
from policy.roboharn_evo.agent.recovery.skill_workflow_loader import (
    SkillRecoveryWorkflowLoader,
)
from policy.roboharn_evo.tests.test_debug_recovery_pure_control import (
    DummyConfig,
    FakeControlRuntime,
    make_card,
    make_snapshot,
)
from policy.roboharn_evo.tests.test_recovery_semantic_tags import (
    FakeBackend as FakeRecoveryBackend,
    FakeRegistry as FakeRecoveryRegistry,
)


def _configured_agent(
    *,
    control_runtime: FakeControlRuntime | None = None,
) -> ImgAgent:
    config = replace(
        DummyConfig(),
        planner_context_mode="compact_v1",
    )
    # Keep this test compatible while the sidecar flag is added to DummyConfig:
    # the production AgentRuntimeConfig exposes this exact attribute.
    object.__setattr__(
        config,
        "gpt_scene_memory_sidecar_enabled",
        True,
    )
    agent = ImgAgent(
        make_card(
            config,
            control_runtime=control_runtime,
        )
    )
    agent.current_instruction = "move the block"
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
            "env_step": 0,
            "task_focus": {
                "target_instances": ["track_0001"],
                "identity_binding_required": True,
            },
            "instances": [
                {
                    "instance_id": "track_0001",
                    "class": "block",
                    "status": "visible",
                    "position_state": "current_verified",
                    "world_m": [0.1, -0.1, 0.77],
                    "operation_pose_candidates": [
                        {
                            "candidate_id": "private:candidate",
                            "arm": "right",
                            "action_mode": "grasp",
                            "tcp_pose": [
                                0.1,
                                -0.1,
                                0.8,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                            ],
                        }
                    ],
                }
            ],
        }
    )
    agent.memory_store.record_observation_preprocess(
        {
            "observation_generation": 7,
            "observation_capture_id": 9,
            "segmentation": [
                {
                    "object_id": "block",
                    "camera": "head",
                    "success": True,
                }
            ],
        }
    )
    return agent


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_sidecar_prefers_rollout_path_and_falls_back_to_trace_sibling(
    tmp_path: Path,
) -> None:
    agent = _configured_agent()
    trace = tmp_path / "episode_0000_agent_trace.jsonl"
    agent.set_trace_file(str(trace), episode_id=0, seed=8)

    agent._record_gpt_scene_memory_request(
        consumer="control_planner",
        scene_memory={"schema": "planner_scene_view/compact_v1"},
    )
    fallback = tmp_path / "episode_0000_gpt_scene_memory.jsonl"
    assert fallback.is_file()

    rollout = tmp_path / "episode_0000_rollout"
    agent.set_rollout_dump_dir(str(rollout))
    agent._record_gpt_scene_memory_request(
        consumer="recovery_planner",
        scene_memory={"schema": "planner_scene_view/compact_v1"},
    )
    assert (rollout / "gpt_scene_memory.jsonl").is_file()
    assert len(_records(fallback)) == 1


def test_identical_requests_are_recorded_individually_with_exact_scene(
    tmp_path: Path,
) -> None:
    agent = _configured_agent()
    rollout = tmp_path / "episode_0000_rollout"
    agent.set_trace_file(
        str(tmp_path / "episode_0000_agent_trace.jsonl"),
        episode_id=0,
        seed=8,
    )
    agent.set_rollout_dump_dir(str(rollout))
    scene = {
        "schema": "planner_scene_view/compact_v1",
        "instances": [
            {
                "instance_id": "track_0001",
                "world_m": [0.1, -0.1, 0.77],
            }
        ],
    }

    first_id = agent._record_gpt_scene_memory_request(
        consumer="recovery_planner",
        scene_memory=scene,
        request_metadata={
            "signal": TASK_LEVEL_RECOVERY_CONTROL,
            "observation_generation": 7,
            "observation_capture_id": 9,
        },
    )
    second_id = agent._record_gpt_scene_memory_request(
        consumer="recovery_planner",
        scene_memory=scene,
        request_metadata={
            "signal": TASK_LEVEL_RECOVERY_CONTROL,
            "observation_generation": 7,
            "observation_capture_id": 9,
        },
    )

    records = _records(rollout / "gpt_scene_memory.jsonl")
    assert len(records) == 2
    assert first_id and second_id and first_id != second_id
    assert records[0]["scene_memory"] == scene
    assert records[1]["scene_memory"] == scene
    assert records[0]["scene_memory_sha256"] == records[1][
        "scene_memory_sha256"
    ]
    assert records[0]["request_index"] + 1 == records[1]["request_index"]
    assert records[0]["consumer"] == "recovery_planner"
    assert records[0]["episode_id"] == 0
    assert records[0]["seed"] == 8


def test_sidecar_can_be_disabled_without_changing_request_state(
    tmp_path: Path,
) -> None:
    agent = _configured_agent()
    object.__setattr__(
        agent.config,
        "gpt_scene_memory_sidecar_enabled",
        False,
    )
    rollout = tmp_path / "episode_0000_rollout"
    agent.set_trace_file(
        str(tmp_path / "episode_0000_agent_trace.jsonl"),
        episode_id=0,
        seed=8,
    )
    agent.set_rollout_dump_dir(str(rollout))

    request_id = agent._record_gpt_scene_memory_request(
        consumer="control_planner",
        scene_memory={"schema": "planner_scene_view/compact_v1"},
    )

    assert request_id == ""
    assert not (rollout / "gpt_scene_memory.jsonl").exists()


def test_control_sidecar_scene_exactly_matches_the_model_request(
    tmp_path: Path,
) -> None:
    control = FakeControlRuntime(
        {
            "action_mode": "start",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": "pick the block",
            "memory_text": "continue",
            "commit_label": "state_change",
        }
    )
    agent = _configured_agent(control_runtime=control)
    rollout = tmp_path / "episode_0000_rollout"
    agent.set_trace_file(
        str(tmp_path / "episode_0000_agent_trace.jsonl"),
        episode_id=0,
        seed=8,
    )
    agent.set_rollout_dump_dir(str(rollout))

    agent.run_control_turn(ControlSignal.NO_ACTIVE_SKILL)

    sent = json.loads(control.calls[0]["task"])
    exact_scene = sent["agent_state"]["working"]["scene_memory"]
    records = _records(rollout / "gpt_scene_memory.jsonl")
    assert len(records) == 1
    assert records[0]["consumer"] == "control_planner"
    assert records[0]["scene_memory"] == exact_scene
    serialized = json.dumps(records[0])
    assert "private:candidate" not in serialized
    assert "tcp_pose" not in serialized


def test_recovery_sidecar_scene_exactly_matches_the_resolver_request(
    tmp_path: Path,
) -> None:
    agent = _configured_agent()
    rollout = tmp_path / "episode_0000_rollout"
    agent.set_trace_file(
        str(tmp_path / "episode_0000_agent_trace.jsonl"),
        episode_id=0,
        seed=8,
    )
    agent.set_rollout_dump_dir(str(rollout))
    agent.memory_store.start_or_replace_skill(
        subtask_text="pick the block",
        skill_spec=agent._resolve_skill_spec(
            "monitored-subtask-execution"
        ),
    )
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="empty-plan",
            tool_calls=[],
            expected_outcome="continue planning",
        ),
        post_recovery_intent="retry",
        reason="inspect current scene",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="test request",
    )

    def resolve_and_notify(**kwargs):
        agent._record_recovery_planner_scene_memory_request(
            {
                "signal_name": kwargs["signal"].name,
                "OOD_scenario": kwargs["signal"].name,
                "current_subtask": kwargs["current_subtask"],
                "preferred_arm": kwargs["preferred_arm"],
                "skill_payload_mode": "signal_scoped",
                "scene_memory": kwargs["scene_memory"],
            }
        )
        return route

    with (
        mock.patch.object(
            agent._recovery_dispatcher,
            "available_tools",
            return_value=[],
        ),
        mock.patch.object(
            agent._recovery_policy_resolver,
            "resolve",
            side_effect=resolve_and_notify,
        ) as resolve,
    ):
        agent._dispatch_recovery_tools(task_env=object(), signal=signal)

    exact_scene = resolve.call_args.kwargs["scene_memory"]
    records = _records(rollout / "gpt_scene_memory.jsonl")
    assert len(records) == 1
    assert records[0]["consumer"] == "recovery_planner"
    assert records[0]["scene_memory"] == exact_scene


def test_recovery_loader_sidecar_exactly_matches_backend_payload(
    tmp_path: Path,
) -> None:
    agent = _configured_agent()
    rollout = tmp_path / "episode_0000_rollout"
    agent.set_trace_file(
        str(tmp_path / "episode_0000_agent_trace.jsonl"),
        episode_id=0,
        seed=8,
    )
    agent.set_rollout_dump_dir(str(rollout))
    backend = FakeRecoveryBackend()
    loader = SkillRecoveryWorkflowLoader(
        FakeRecoveryRegistry(),
        recovery_backend=backend,
        request_observer=(
            agent._record_recovery_planner_scene_memory_request
        ),
    )
    scene = {
        "schema": "planner_scene_view/compact_v1",
        "env_step": 4,
        "instances": [
            {"instance_id": "track_0001", "world_m": [0.1, 0, 0.77]}
        ],
    }

    loader.resolve(
        signal_name="motion_blocked",
        reason="test request",
        current_subtask="pick the block",
        scene_memory=scene,
        available_tools=["reobserve_scene"],
    )

    assert backend.payload is not None
    records = _records(rollout / "gpt_scene_memory.jsonl")
    assert len(records) == 1
    assert records[0]["consumer"] == "recovery_planner"
    assert records[0]["scene_memory"] == backend.payload["scene_memory"]


def test_sidecar_write_failure_is_fail_open_for_direct_control_and_recovery(
    tmp_path: Path,
) -> None:
    control = FakeControlRuntime(
        {
            "action_mode": "start",
            "selected_skill": "monitored-subtask-execution",
            "subtask_text": "pick the block",
            "memory_text": "continue",
            "commit_label": "state_change",
        }
    )
    agent = _configured_agent(control_runtime=control)
    # Opening a directory as an append-only JSONL file must fail without
    # preventing either planner request.
    agent._gpt_scene_memory_sidecar_file = tmp_path
    direct = agent._record_gpt_scene_memory_request(
        consumer="control_planner",
        scene_memory={"schema": "planner_scene_view/compact_v1"},
    )
    assert direct == ""

    prediction = agent.run_control_turn(ControlSignal.NO_ACTIVE_SKILL)
    assert prediction["subtask_text"] == "pick the block"
    assert len(control.calls) == 1

    agent.memory_store.start_or_replace_skill(
        subtask_text="pick the block",
        skill_spec=agent._resolve_skill_spec(
            "monitored-subtask-execution"
        ),
    )
    route = RecoveryRoute(
        signal_name=TASK_LEVEL_RECOVERY_CONTROL,
        workflow_name="task-level-recovery-control",
        plan=RecoveryPrimitivePlan(
            name="empty-plan",
            tool_calls=[],
            expected_outcome="continue planning",
        ),
        post_recovery_intent="retry",
        reason="inspect current scene",
    )
    signal = agent._make_monitor_signal(
        name=TASK_LEVEL_RECOVERY_CONTROL,
        level="warning",
        reason="test request",
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
        result = agent._dispatch_recovery_tools(
            task_env=object(),
            signal=signal,
        )

    assert result == "retry"
    assert resolve.call_count == 1


class _EffectBackend:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def plan_recovery(self, *, recovery_payload, media=None):
        self.payloads.append(recovery_payload)
        return {
            "effect_verified": "true",
            "effect_type": "placement",
            "confidence": 0.9,
            "subtask_status": "completed",
            "recommended_control": "continue",
        }


def test_action_effect_sidecar_exactly_matches_one_backend_request_and_is_fail_open(
    tmp_path: Path,
) -> None:
    agent = _configured_agent()
    rollout = tmp_path / "episode_0000_rollout"
    agent.set_trace_file(
        str(tmp_path / "episode_0000_agent_trace.jsonl"),
        episode_id=0,
        seed=8,
    )
    agent.set_rollout_dump_dir(str(rollout))
    backend = _EffectBackend()
    verifier = ActionEffectVerifier(
        skill_registry=agent._recovery_skill_registry,
        backend=backend,
        request_observer=agent._record_action_effect_scene_memory_request,
    )
    pre_scene = {
        "schema": "planner_scene_view/compact_v1",
        "env_step": 3,
        "instances": [{"instance_id": "track_0001", "world_m": [0, 0, 1]}],
    }
    post_scene = {
        "schema": "planner_scene_view/compact_v1",
        "env_step": 4,
        "instances": [{"instance_id": "track_0001", "world_m": [0.1, 0, 1]}],
    }
    evidence = {
        "workflow": "place-object",
        "current_subtask": "place the block",
        "pre": {"scene_memory": pre_scene},
        "post": {"scene_memory": post_scene},
    }

    result = verifier.verify(evidence)

    assert result["effect_verified"] == "true"
    assert len(backend.payloads) == 1
    sent_evidence = backend.payloads[0]["evidence_payload"]
    records = _records(rollout / "gpt_scene_memory.jsonl")
    assert len(records) == 1
    assert records[0]["consumer"] == "action_effect_verifier"
    assert records[0]["scene_memory"] == sent_evidence["post"][
        "scene_memory"
    ]
    assert records[0]["scene_memory_views"]["pre"] == sent_evidence[
        "pre"
    ]["scene_memory"]
    assert records[0]["scene_memory_views"]["post"] == sent_evidence[
        "post"
    ]["scene_memory"]

    # An observer failure is diagnostic-only: the same backend call proceeds.
    failing_backend = _EffectBackend()
    failing_observer_verifier = ActionEffectVerifier(
        skill_registry=agent._recovery_skill_registry,
        backend=failing_backend,
        request_observer=mock.Mock(
            side_effect=OSError("sidecar unavailable")
        ),
    )
    fail_open_result = failing_observer_verifier.verify(evidence)
    assert fail_open_result["effect_verified"] == "true"
    assert len(failing_backend.payloads) == 1

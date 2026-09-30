from __future__ import annotations

from dataclasses import dataclass
import json
from unittest import mock

import numpy as np

from policy.roboharn_evo.agent.components.agent_tools.local_skill_registry import LocalSkillRegistry
from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.environment import EnvSnapshot
from policy.roboharn_evo.agent.paths import rmbench_root, roboharn_skills_dir
from policy.roboharn_evo.agent.recovery.action_effect_verifier import (
    ActionEffectVerifier,
    ActionEffectVerifierBackendError,
)
from policy.roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall, RecoveryToolResult
from policy.roboharn_evo.models.agent_api_recovery_adapter import AgentApiRecoveryAdapter, AgentApiRecoveryConfig
from policy.roboharn_evo.models.prompt_skills import load_prompt_skill
from policy.roboharn_evo.scripts.serve_qwen_planner import normalize_action_effect_prediction


class FakeEffectBackend:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = {} if response is None else response
        self.error = error
        self.payloads = []

    def plan_recovery(self, *, recovery_payload, media=None):
        self.payloads.append(recovery_payload)
        if self.error is not None:
            raise self.error
        return dict(self.response) if isinstance(self.response, dict) else self.response


@dataclass(frozen=True)
class DummyConfig:
    initial_memory_text: str = "The task has started."
    observation_preprocess_enabled: bool = False
    observation_preprocess_every_n_steps: int = 1
    observation_preprocess_normalization_url: str = ""
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
    decision_interval: int = 1
    max_retries_per_skill: int = 2
    max_steps_per_skill: int = 80
    stall_patience: int = 12
    mode: str = "deployment"
    interrupt_on_skill_change: bool = True
    debug_recovery_enabled: bool = False
    debug_recovery_trigger_step: int = 0
    debug_recovery_signal: str = "motion_blocked"
    debug_recovery_reason: str = ""
    debug_recovery_subtask: str = ""
    debug_recovery_skill: str = "monitored-subtask-execution"
    debug_recovery_wait_for_scene_memory: bool = False
    debug_recovery_max_wait_steps: int = 0
    debug_recovery_retry_budget: int = 1
    debug_recovery_pure_control: bool = False
    debug_recovery_max_rounds: int = 1
    debug_recovery_repeat_signal: bool = False
    debug_recovery_bootstrap_with_planner: bool = False
    debug_recovery_skip_vla_rollout: bool = False
    pure_tool_control_enabled: bool = False
    pure_tool_control_trigger_step: int = 0
    pure_tool_control_signal: str = "task_level_recovery_control"
    pure_tool_control_reason: str = "pure tool-control ablation"
    pure_tool_control_wait_for_scene_memory: bool = False
    pure_tool_control_max_wait_steps: int = 0
    pure_tool_control_retry_budget: int = 1
    pure_tool_control_max_rounds: int = 1
    pure_tool_control_repeat_signal: bool = True
    pure_tool_control_bootstrap_with_planner: bool = True
    pure_tool_control_skip_vla_rollout: bool = True


def make_card(config: DummyConfig) -> object:
    class Card:
        control_runtime = None
        executor_runtime = None
        control_model_name = "test"
        executor_name = "test"
        ood_backend_config = None
        recovery_backend_config = {"backend": "agent_api", "agent_api": {"server_url": "http://127.0.0.1:9/recover"}}

    card = Card()
    card.config = config
    return card


def make_snapshot() -> EnvSnapshot:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    joint_vector = np.zeros(14, dtype=np.float32)
    left_endpose = np.array([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    right_endpose = np.array([0.3, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return EnvSnapshot(
        raw={"joint_action": {"vector": [0.0] * 14}, "endpose": {"left_gripper": 1.0, "right_gripper": 1.0}},
        head_rgb=image,
        left_rgb=image,
        right_rgb=image,
        joint_vector=joint_vector,
        left_endpose=left_endpose,
        right_endpose=right_endpose,
        step_count=7,
        step_limit=100,
        eval_success=False,
        check_success=False,
        max_reward=0.0,
        instruction="cover the block",
    )


def skill_registry() -> LocalSkillRegistry:
    return LocalSkillRegistry(
        configured_paths=[str(roboharn_skills_dir())],
        workspace_root=str(rmbench_root()),
    )


def test_action_effect_skill_contract_loads() -> None:
    prompt = load_prompt_skill("action-effect-verification")
    assert "Do not treat tool execution success as task-effect success" in prompt
    assert "latest_world_m" in prompt
    assert "history-smoothed position" in prompt
    assert "effect_verified" in prompt
    assert "unverified" in prompt
    assert "expected_outcome" in prompt
    assert "empty tool sequence" in prompt
    assert "perception-query metadata as retrieval intent" in prompt
    assert "backward-compatible payloads" in prompt
    assert "immutable historical reference pose" in prompt
    assert "subtask_status" in prompt
    assert "recommended_control" in prompt
    assert "only a hint" in prompt


def test_action_effect_verifier_calls_backend_with_temporary_payload() -> None:
    backend = FakeEffectBackend(
        {
            "effect_verified": "true",
            "effect_type": "move",
            "confidence": 0.8,
            "evidence_summary": "tool moved as intended",
            "next_constraint": "continue",
            "subtask_status": "in_progress",
            "recommended_control": "continue",
        }
    )
    verifier = ActionEffectVerifier(skill_registry=skill_registry(), backend=backend)

    result = verifier.verify({"current_subtask": "cover", "pre": {}, "post": {}})

    assert result["effect_verified"] == "true"
    assert result["effect_type"] == "move"
    assert result["confidence"] == 0.8
    assert result["subtask_status"] == "in_progress"
    assert result["recommended_control"] == "continue"
    assert backend.payloads[0]["mode"] == "action_effect_verification"
    assert backend.payloads[0]["evidence_payload"]["current_subtask"] == "cover"
    assert backend.payloads[0]["verification_skill"]["name"] == "action-effect-verification"


def test_action_effect_agent_api_prompt_comes_from_skill() -> None:
    verifier = ActionEffectVerifier(
        skill_registry=skill_registry(),
        backend_config={"backend": "agent_api", "agent_api": {"server_url": "http://127.0.0.1:9/recover"}},
    )
    skill = skill_registry().get_skill("action-effect-verification")
    assert skill is not None

    backend = verifier._backend_for_skill(skill.body)

    assert isinstance(backend, AgentApiRecoveryAdapter)
    assert backend.config.prompt_template == skill.body
    assert "Verification payload: {recovery_payload}" in backend.config.prompt_template


def test_agent_api_recovery_does_not_duplicate_payload_in_prompt() -> None:
    backend = AgentApiRecoveryAdapter(
        AgentApiRecoveryConfig(
            server_url="http://127.0.0.1:9/recover",
            timeout_sec=120,
            prompt_template="Plan recovery. Payload: {recovery_payload}",
        )
    )
    recovery_payload = {"large_unique_marker": "payload-only-value"}

    with mock.patch(
        "policy.roboharn_evo.models.agent_api_recovery_adapter._post_json",
        return_value={"post_recovery_intent": "retry"},
    ) as post_json:
        backend.plan_recovery(recovery_payload=recovery_payload)

    request_payload = post_json.call_args.kwargs["payload"]
    assert request_payload["recovery_payload"] == recovery_payload
    assert "payload-only-value" not in request_payload["prompt"]
    assert "provided separately in the user message" in request_payload["prompt"]


def test_recovery_observation_context_drops_masks_and_robot_copies() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    payload = {
        "stage": "observation_preprocess",
        "env_step": 7,
        "segmentation": [
            {
                "success": True,
                "object_id": "lid",
                "query_role": "tool",
                "mask_path": "/tmp/large-mask.png",
                "robot_state": {"large_duplicate": [1] * 100},
                "detections": [
                    {
                        "rank": 0,
                        "score": 0.9,
                        "mask_path": "/tmp/detection-mask.png",
                        "robot_state": {"large_duplicate": [1] * 100},
                        "grounding_3d": {
                            "success": True,
                            "centroid_world": [0.1, 0.2, 0.8],
                            "approach_point_world": [0.1, 0.2, 0.9],
                            "sampled_points": [[0.0, 0.0, 0.0]] * 100,
                        },
                    }
                ],
            }
        ],
        "scene_memory": {"temporal_memory": {"history": [1] * 100}},
    }

    compact = agent._compact_observation_preprocess_for_recovery(payload)
    encoded = json.dumps(compact)

    assert "large-mask" not in encoded
    assert "detection-mask" not in encoded
    assert "large_duplicate" not in encoded
    assert "sampled_points" not in encoded
    assert "temporal_memory" not in encoded
    assert compact["segmentation"][0]["detections"][0]["grounding_3d"]["approach_point_world"] == [0.1, 0.2, 0.9]


def test_action_effect_verifier_propagates_backend_error() -> None:
    verifier = ActionEffectVerifier(skill_registry=skill_registry(), backend=FakeEffectBackend(error=RuntimeError("offline")))

    try:
        verifier.verify({"current_subtask": "cover"})
    except ActionEffectVerifierBackendError as exc:
        assert "RuntimeError" in str(exc)
        assert "offline" in str(exc)
    else:
        raise AssertionError("backend outage must remain distinguishable from a semantic retry")


def test_action_effect_verifier_invalid_response_fails_closed_without_backend_error() -> None:
    verifier = ActionEffectVerifier(skill_registry=skill_registry(), backend=FakeEffectBackend(response=[]))

    result = verifier.verify({"current_subtask": "cover"})

    assert result["effect_verified"] == "unverified"
    assert result["effect_type"] == "unknown"
    assert result["subtask_status"] == "uncertain"
    assert result["recommended_control"] == "retry"
    assert result["failure_reason"] == "invalid_verifier_response"


def test_qwen_recover_service_preserves_action_effect_schema() -> None:
    result = normalize_action_effect_prediction(
        {
            "effect_verified": "false",
            "effect_type": "grasp",
            "confidence": 1.5,
            "evidence_summary": "object moved away after close",
            "failure_reason": "no coupled motion",
            "next_constraint": "do not carry",
            "memory_update": "close did not verify grasp",
            "subtask_status": "in-progress",
            "recommended_control": "continue",
            "tool_calls": [{"tool_name": "close_gripper"}],
            "post_recovery_intent": "retry",
        }
    )

    assert result == {
        "effect_verified": "false",
        "effect_type": "grasp",
        "confidence": 1.0,
        "evidence_summary": "object moved away after close",
        "failure_reason": "no coupled motion",
        "next_constraint": "do not carry",
        "memory_update": "close did not verify grasp",
        "subtask_status": "in_progress",
        "recommended_control": "continue",
    }


def test_qwen_recover_service_defaults_invalid_transition_fields_conservatively() -> None:
    result = normalize_action_effect_prediction(
        {
            "effect_verified": "true",
            "subtask_status": "done-ish",
            "recommended_control": "advance",
        }
    )

    assert result["effect_verified"] == "true"
    assert result["subtask_status"] == "uncertain"
    assert result["recommended_control"] == "retry"


def test_img_agent_action_effect_payload_is_temporary_and_compact() -> None:
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
    agent.memory_store.state.working.scene_memory = {
        "env_step": 7,
        "task_focus": {"tool_instances": ["lid_left"], "target_instances": ["green_block_01"]},
        "instances": [
            {
                "instance_id": "lid_left",
                "track_id": "track_0001",
                "status": "visible",
                "world_m": [0.1, 0.0, 0.8],
                "first_observed_world_m": [0.1, -0.2, 0.8],
                "latest_world_m": [0.1, 0.0, 0.83],
                "approach_world_m": [0.1, 0.0, 0.9],
                "grasp_world_m": [0.1, 0.0, 0.82],
            },
            {"instance_id": "background", "world_m": [9.0, 9.0, 9.0]},
        ],
        "summary": "focus tool lid_left",
    }
    agent.memory_store.record_observation_summary("focus=tool:lid_left")

    calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={"arm": "left", "role": "tool", "point_key": "grasp_world_m", "_scene_memory": agent.memory_store.state.working.scene_memory},
        )
    ]
    results = [
        RecoveryToolResult(
            tool_name="move_ee_to_grounded_instance",
            success=True,
            message="executed bounded EE move to grounded scene-memory instance",
            details={"arm": "left", "steps": 2},
        )
    ]
    agent._action_effect_verifier = ActionEffectVerifier(
        skill_registry=skill_registry(),
        backend=FakeEffectBackend(
            {
                "effect_verified": "unverified",
                "effect_type": "grasp",
                "memory_update": "grasp not verified",
                "subtask_status": "uncertain",
                "recommended_control": "retry",
            }
        ),
    )

    pre = agent._capture_action_effect_state()
    effect = agent._verify_action_effect(
        "task-level-recovery-control",
        current_subtask="cover",
        post_recovery_intent="replan",
        expected_outcome="return to planning when the observed scene is ready",
        recovery_reason="current evidence supports a planning handoff",
        calls=calls,
        results=results,
        pre=pre,
        post=pre,
    )
    entry = agent._format_action_effect_history_entry(effect, calls, results)
    agent.memory_store.record_recovery(entry)

    assert "action_effect:effect=unverified" in agent.memory_store.state.working.recovery_history[-1]
    assert "type=grasp" in agent.memory_store.state.working.recovery_history[-1]
    assert "subtask=uncertain" in agent.memory_store.state.working.recovery_history[-1]
    assert "control=retry" in agent.memory_store.state.working.recovery_history[-1]
    assert "memory=grasp not verified" in agent.memory_store.state.working.recovery_history[-1]
    assert "pre_scene_memory" not in agent.memory_store.state.working.to_dict()
    payload = agent._action_effect_verifier._backend.payloads[0]["evidence_payload"]
    assert payload["post_recovery_intent"] == "replan"
    assert payload["expected_outcome"] == "return to planning when the observed scene is ready"
    assert payload["recovery_reason"] == "current evidence supports a planning handoff"
    assert payload["pre"]["scene_memory"]["instances"][0]["instance_id"] == "lid_left"
    assert payload["pre"]["scene_memory"]["instances"][0]["world_m"] == [0.1, 0.0, 0.83]
    assert payload["pre"]["scene_memory"]["instances"][0]["first_observed_world_m"] == [0.1, -0.2, 0.8]
    assert payload["pre"]["scene_memory"]["instances"][0]["latest_world_m"] == [0.1, 0.0, 0.83]
    assert payload["pre"]["scene_memory"]["instances"][0]["stable_world_m"] == [0.1, 0.0, 0.8]
    assert payload["pre"]["scene_memory"]["instances"][0]["position_source"] == "latest_observation"
    assert all(item.get("instance_id") != "background" for item in payload["pre"]["scene_memory"]["instances"])
    assert "_scene_memory" not in str(payload["last_tool_calls"])


def test_recovery_scene_compaction_keeps_stable_grounding_by_default() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    scene_memory = {
        "instances": [
            {
                "instance_id": "tool_left",
                "world_m": [0.0, 0.0, 0.8],
                "latest_world_m": [0.0, 0.0, 0.84],
            }
        ]
    }

    compact = agent._compact_scene_memory_for_effect(scene_memory)

    assert compact["instances"][0]["world_m"] == [0.0, 0.0, 0.8]
    assert "stable_world_m" not in compact["instances"][0]
    assert "latest_world_m" not in compact["instances"][0]


def test_action_effect_history_handles_missing_dispatch_result() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    call = RecoveryToolCall(tool_name="reobserve_scene", args={"reason": "refresh evidence"})

    entry = agent._format_action_effect_history_entry(
        {"effect_verified": "unverified", "effect_type": "move", "failure_reason": "no result returned"},
        [call],
        [],
    )

    assert entry.startswith("action_effect:effect=unverified")
    assert "tool=reobserve_scene" in entry
    assert "failure=no result returned" in entry

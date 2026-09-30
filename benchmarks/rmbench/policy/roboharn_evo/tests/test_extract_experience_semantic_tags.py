from __future__ import annotations

from pathlib import Path

from policy.roboharn_evo.scripts.extract_experience_from_trace import build_context, extract_recovery_trials


def _records() -> list[dict]:
    return build_context(
        [
            {
                "event": "instruction_set",
                "instruction": "put the mug in the tray",
                "episode_id": 1,
                "seed": 2,
                "env_step": 0,
            },
            {
                "event": "control_decision",
                "rendered_instruction": "grasp the mug",
                "semantic_tags": {
                    "task_family": "pick_and_place",
                    "subtask_type": "grasp",
                    "state_tags": {"visibility_state": "visible"},
                    "tag_source": "planner_vlm",
                },
                "episode_id": 1,
                "seed": 2,
                "env_step": 1,
            },
            {
                "event": "monitor_signal",
                "signal": "motion_blocked",
                "reason": "blocked",
                "robot_state": {
                    "step": 3,
                    "joint_norm": 1.2,
                    "left": {"xyz": [0.1, 0.2, 0.3], "rpy": [0.0, 0.0, 0.0], "gripper": 0.4},
                    "right": {"xyz": [0.4, 0.5, 0.6], "rpy": [0.0, 0.0, 1.57], "gripper": 0.8},
                },
                "episode_id": 1,
                "seed": 2,
                "env_step": 3,
            },
            {
                "event": "recovery_router",
                "signal_name": "motion_blocked",
                "ood_scenario": "motion_blocked",
                "workflow": "recover-motion-blocked",
                "post_recovery_intent": "retry",
                "tools": ["reobserve_scene"],
                "available_tools": ["reobserve_scene"],
                "reason": "blocked",
                "episode_id": 1,
                "seed": 2,
                "env_step": 3,
            },
            {
                "event": "recovery_result",
                "results": [{"tool_name": "reobserve_scene", "success": True, "message": "ok"}],
                "episode_id": 1,
                "seed": 2,
                "env_step": 4,
            },
        ]
    )


def test_trace_semantic_tags_are_used_without_fallback() -> None:
    trials = extract_recovery_trials(_records(), Path("trace.jsonl"), use_fallback_tags=False)

    assert len(trials) == 1
    trial = trials[0]
    assert trial["task_family"] == "pick_and_place"
    assert trial["subtask_type"] == "grasp"
    assert trial["semantic_tags"]["tag_source"] == "planner_vlm"
    assert "pick_and_place" in trial["retrieval_tags"]
    assert "grasp" in trial["retrieval_tags"]
    assert trial["post_window_outcome"] == "unknown"
    assert trial["retry_success_after_recovery"] is False
    assert trial["robot_state"]["left"]["xyz"] == [0.1, 0.2, 0.3]


def test_missing_trace_tags_default_to_other_without_fallback() -> None:
    records = build_context(
        [
            {"event": "instruction_set", "instruction": "put the mug in the tray"},
            {"event": "control_decision", "rendered_instruction": "grasp the mug"},
            {"event": "monitor_signal", "signal": "motion_blocked"},
            {
                "event": "recovery_router",
                "signal_name": "motion_blocked",
                "ood_scenario": "motion_blocked",
                "workflow": "recover-motion-blocked",
                "post_recovery_intent": "retry",
                "tools": [],
            },
            {"event": "recovery_result", "results": []},
        ]
    )

    trials = extract_recovery_trials(records, Path("trace.jsonl"), use_fallback_tags=False)

    assert trials[0]["task_family"] == "other"
    assert trials[0]["subtask_type"] == "other"
    assert trials[0]["semantic_tags"]["tag_source"] == "unknown"


def test_missing_trace_tags_can_opt_into_fallback() -> None:
    records = build_context(
        [
            {"event": "instruction_set", "instruction": "put the mug in the tray"},
            {"event": "control_decision", "rendered_instruction": "grasp the mug"},
            {"event": "monitor_signal", "signal": "motion_blocked"},
            {
                "event": "recovery_router",
                "signal_name": "motion_blocked",
                "ood_scenario": "motion_blocked",
                "workflow": "recover-motion-blocked",
                "post_recovery_intent": "retry",
                "tools": [],
            },
            {"event": "recovery_result", "results": []},
        ]
    )

    trials = extract_recovery_trials(records, Path("trace.jsonl"), use_fallback_tags=True)

    assert trials[0]["task_family"] == ""
    assert trials[0]["subtask_type"] == ""
    assert trials[0]["semantic_tags"]["tag_source"] == "fallback"


def test_post_recovery_attribution_detects_retry_reentry() -> None:
    records = build_context(
        [
            {"event": "instruction_set", "instruction": "put the mug in the tray"},
            {"event": "control_decision", "rendered_instruction": "grasp the mug"},
            {"event": "monitor_signal", "signal": "grasp_lost", "reason": "lost"},
            {
                "event": "recovery_router",
                "signal_name": "grasp_lost",
                "ood_scenario": "grasp_lost",
                "workflow": "recover-grasp-lost",
                "post_recovery_intent": "retry",
                "tools": ["open_gripper", "reobserve_scene"],
                "available_tools": ["open_gripper", "reobserve_scene"],
            },
            {
                "event": "recovery_result",
                "results": [
                    {"tool_name": "open_gripper", "success": True, "message": "ok"},
                    {"tool_name": "reobserve_scene", "success": True, "message": "ok"},
                ],
            },
            {"event": "recovery_policy", "action": "retry", "reason": "retry after recovery"},
            {"event": "vla_request", "subtask": "grasp the mug"},
        ]
    )

    trials = extract_recovery_trials(records, Path("trace.jsonl"), use_fallback_tags=True)

    assert trials[0]["post_window_outcome"] == "retry_reentered_vla"
    assert trials[0]["retry_success_after_recovery"] is False
    assert trials[0]["outcome_evidence"]["effective_action"] == "retry"


def test_post_recovery_attribution_detects_success_after_retry() -> None:
    records = build_context(
        [
            {"event": "instruction_set", "instruction": "put the mug in the tray"},
            {"event": "control_decision", "rendered_instruction": "grasp the mug"},
            {"event": "monitor_signal", "signal": "grasp_lost", "reason": "lost"},
            {
                "event": "recovery_router",
                "signal_name": "grasp_lost",
                "ood_scenario": "grasp_lost",
                "workflow": "recover-grasp-lost",
                "post_recovery_intent": "retry",
                "tools": ["open_gripper", "reobserve_scene"],
                "available_tools": ["open_gripper", "reobserve_scene"],
            },
            {"event": "recovery_result", "results": [{"tool_name": "reobserve_scene", "success": True}]},
            {"event": "recovery_policy", "action": "retry", "reason": "retry after recovery"},
            {"event": "vla_request", "subtask": "grasp the mug"},
            {"event": "monitor_signal", "signal": "task_success", "reason": "done"},
        ]
    )

    trials = extract_recovery_trials(records, Path("trace.jsonl"), use_fallback_tags=True)

    assert trials[0]["post_window_outcome"] == "success"
    assert trials[0]["retry_success_after_recovery"] is True

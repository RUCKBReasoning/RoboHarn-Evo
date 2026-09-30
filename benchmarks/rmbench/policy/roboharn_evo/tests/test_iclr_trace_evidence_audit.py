from __future__ import annotations

import json
from pathlib import Path

from policy.roboharn_evo.scripts.audit_iclr_trace_evidence import analyze_trace, build_summary


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_trace_audit_detects_complete_recovery_chain_and_feedback(tmp_path: Path) -> None:
    trace = tmp_path / "episode_0000_agent_trace.jsonl"
    write_jsonl(
        trace,
        [
            {"event": "episode_start", "task_name": "cover_blocks", "seed": 0, "instruction": "cover the block"},
            {
                "event": "observation_preprocess",
                "env_step": 0,
                "latency_sec": 0.2,
                "segmentation": [
                    {
                        "success": True,
                        "object_id": "lid",
                        "camera": "head",
                        "detections": [
                            {
                                "camera": "head",
                                "grounding_3d": {"success": True, "centroid_world": [0.0, 0.0, 0.8]},
                            }
                        ],
                        "grounding_3d": {"success": True},
                    }
                ],
            },
            {
                "event": "scene_memory_update",
                "env_step": 0,
                "scene_memory": {
                    "instances": [
                        {
                            "instance_id": "lid_01",
                            "track_id": "track_0001",
                            "camera": "head",
                            "stability": "stable",
                            "status": "visible",
                        }
                    ],
                    "task_focus": {"tool_instances": ["lid_01"], "target_instances": [], "source": "vlm"},
                    "temporal_memory": {"tracks": [{"track_id": "track_0001"}]},
                    "summary": "scene_instances=['lid_01']",
                },
            },
            {
                "event": "recovery_router",
                "env_step": 0,
                "workflow": "task-level-recovery-control",
                "tools": ["move_ee_to_grounded_instance"],
                "available_tools": ["move_ee_to_grounded_instance"],
            },
            {"event": "recovery_dispatch", "env_step": 0, "workflow": "task-level-recovery-control"},
            {
                "event": "action_effect_verification",
                "env_step": 1,
                "workflow": "task-level-recovery-control",
                "result": {
                    "effect_verified": "false",
                    "effect_type": "grasp",
                    "confidence": 0.8,
                    "memory_update": "grasp failed",
                },
            },
            {
                "event": "recovery_result",
                "env_step": 1,
                "workflow": "task-level-recovery-control",
                "results": [{"tool_name": "move_ee_to_grounded_instance", "success": True}],
            },
            {
                "event": "recovery_router",
                "env_step": 2,
                "workflow": "task-level-recovery-control",
                "tools": ["reobserve_scene"],
                "available_tools": ["reobserve_scene"],
                "reason": "recent_recovery_history=['action_effect:effect=false,type=grasp,memory=grasp failed']",
            },
            {"event": "episode_end", "success": False, "result": "Fail", "failure_reason": "test"},
        ],
    )

    report = analyze_trace(trace)

    assert report["level0_recovery_chain"]["complete"]
    assert report["recovery"]["complete_chains"] == 1
    assert report["recovery"]["action_effect_counts"] == {"false": 1}
    assert report["recovery"]["action_effect_memory_feedback_mentions"] == 1
    assert report["scene_memory"]["unique_track_ids"] == 1
    assert report["evidence_flags"]["has_complete_recovery_chain"]
    assert report["evidence_flags"]["has_action_effect_memory_feedback"]


def test_trace_audit_warns_on_empty_scene_memory_and_segmentation_errors(tmp_path: Path) -> None:
    trace = tmp_path / "episode_0001_agent_trace.jsonl"
    write_jsonl(
        trace,
        [
            {"event": "episode_start", "task_name": "cover_blocks"},
            {
                "event": "observation_preprocess",
                "env_step": 0,
                "segmentation": [{"success": False, "error": "SAM3 unavailable", "camera": "head"}],
            },
            {"event": "scene_memory_update", "env_step": 0, "scene_memory": {"instances": []}},
            {"event": "episode_end", "success": False},
        ],
    )

    summary = build_summary([trace])
    report = summary["traces"][0]

    assert "perception_preprocess_present_but_no_successful_segmentation" in report["warnings"]
    assert "scene_memory_present_but_empty_instances" in report["warnings"]
    assert summary["aggregate"]["segmentation_error_traces"] == 1
    assert summary["claim_evidence"][1]["status"] == "trace_supported"


def test_trace_audit_accepts_environment_terminal_effect_without_counting_agent_verifier(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "episode_0002_agent_trace.jsonl"
    write_jsonl(
        trace,
        [
            {"event": "episode_start", "task_name": "terminal_task"},
            {
                "event": "observation_preprocess",
                "segmentation": [{"success": True, "camera": "head"}],
            },
            {
                "event": "scene_memory_update",
                "scene_memory": {
                    "instances": [{"instance_id": "target", "track_id": "track_0001"}],
                    "task_focus": {"target_instances": ["target"]},
                },
            },
            {
                "event": "recovery_router",
                "workflow": "task-level-recovery-control",
                "tools": ["contact_displace", "reobserve_scene"],
                "available_tools": ["contact_displace", "reobserve_scene"],
            },
            {"event": "recovery_dispatch"},
            {
                "event": "environment_success_effect_commit",
                "authority": "environment_eval_success",
                "verifier_bypassed": True,
            },
            {
                "event": "recovery_result",
                "results": [
                    {"tool_name": "contact_displace", "success": True},
                    {
                        "tool_name": "reobserve_scene",
                        "success": False,
                        "details": {"terminal_skip": True},
                    },
                ],
            },
            {"event": "episode_end", "success": True, "environment_success": True},
        ],
    )

    report = analyze_trace(trace)

    assert report["level0_recovery_chain"]["complete"] is True
    assert report["level0_recovery_chain"]["accepted_effect_authority"] == "environment_eval_success"
    assert report["recovery"]["complete_chains"] == 1
    assert report["recovery"]["action_effect_events"] == 0
    assert report["recovery"]["environment_success_effect_events"] == 1
    assert report["recovery"]["tool_result_successes"] == 1
    assert report["recovery"]["tool_result_failures"] == 0
    assert report["recovery"]["tool_result_terminal_skips"] == 1
    assert report["evidence_flags"]["has_action_effect_verification"] is False
    assert report["evidence_flags"]["has_environment_success_effect_commit"] is True

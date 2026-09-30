from __future__ import annotations

import copy

from policy.roboharn_evo.scripts.replay_recovery_payload_ablation import apply_ablation, normalize_backend_output


def sample_payload() -> dict:
    return {
        "scene_memory": {"instances": [{"instance_id": "lid_01"}]},
        "observation_preprocess": {"segmentation": [{"object_id": "lid"}]},
        "recovery_history": ["action_effect:effect=false,type=grasp"],
        "retrieved_experience": {"lessons": [{"lesson_id": "x"}], "similar_cases": [], "avoid_patterns": []},
    }


def test_apply_ablation_masks_scene_memory_without_mutating_original() -> None:
    payload = sample_payload()
    original = copy.deepcopy(payload)

    ablated = apply_ablation(payload, "no_scene_memory")

    assert ablated["scene_memory"] == {}
    assert ablated["observation_preprocess"] == original["observation_preprocess"]
    assert payload == original


def test_apply_ablation_masks_recovery_history_and_retrieval() -> None:
    payload = sample_payload()

    ablated = apply_ablation(payload, "no_recovery_history")

    assert ablated["recovery_history"] == []
    assert ablated["retrieved_experience"] == {"lessons": [], "similar_cases": [], "avoid_patterns": []}


def test_normalize_backend_output_validates_tool_schema() -> None:
    valid = normalize_backend_output(
        {
            "recovery_workflow": "task-level-recovery-control",
            "post_recovery_intent": "retry",
            "tool_calls": [{"tool_name": "reobserve_scene", "args": {}}],
        }
    )
    invalid = normalize_backend_output(
        {
            "recovery_workflow": "task-level-recovery-control",
            "post_recovery_intent": "retry",
            "tool_calls": [{"tool_name": "imaginary_tool", "args": []}],
        }
    )

    assert valid["valid"] is True
    assert valid["tool_sequence"] == ["reobserve_scene"]
    assert invalid["valid"] is False
    assert "unknown tool: imaginary_tool" in invalid["errors"]
    assert "args for imaginary_tool is not a dict" in invalid["errors"]

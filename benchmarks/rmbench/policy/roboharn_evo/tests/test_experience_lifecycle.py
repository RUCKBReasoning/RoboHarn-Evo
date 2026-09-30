from __future__ import annotations

import json
from pathlib import Path

from policy.roboharn_evo.agent.experience import ExperienceQuery, ExperienceRetriever
from policy.roboharn_evo.scripts.build_experience_index import index_record
from policy.roboharn_evo.scripts.mine_experience_lessons import cluster_key, render_lesson


def _record(outcome: str = "recovery_success_retry") -> dict:
    return {
        "trial_id": f"trial_{outcome}",
        "trace_file": "trace.jsonl",
        "OOD_scenario": "grasp_lost",
        "task_family": "pick_and_place",
        "subtask_type": "grasp",
        "semantic_tags": {
            "task_family": "pick_and_place",
            "subtask_type": "grasp",
            "state_tags": {"visibility_state": "visible", "gripper_state": "empty"},
            "tag_source": "ood_vlm",
        },
        "recovery_workflow": "recover-grasp-lost",
        "tool_calls": [{"tool_name": "open_gripper", "args": {}}, {"tool_name": "reobserve_scene", "args": {}}],
        "available_tools": ["open_gripper", "reobserve_scene"],
        "post_recovery_intent": "retry",
        "outcome": outcome,
        "retrieval_tags": ["grasp_lost", "pick_and_place", "grasp", "visible", "empty"],
        "robot_state": {
            "step": 12,
            "joint_norm": 1.23,
            "left": {"xyz": [0.1, 0.2, 0.3], "rpy": [0.0, 0.1, 0.2], "gripper": 0.0},
            "right": {"xyz": [0.4, 0.5, 0.6], "rpy": [0.3, 0.4, 0.5], "gripper": 1.0},
        },
    }


def test_mined_lesson_is_candidate_by_default() -> None:
    record = _record()
    key = cluster_key(record)
    text = render_lesson(key, [record, {**record, "trial_id": "trial_2"}], opposing_records=[], min_support=2)

    assert "- Status: `candidate`" in text
    assert "recover-grasp-lost" in text
    assert "open_gripper" in text


def test_index_record_adds_quality_fields(tmp_path: Path) -> None:
    record = _record()
    indexed = index_record(record, tmp_path)

    assert indexed["support_count"] == 1
    assert indexed["opposing_count"] == 0
    assert indexed["confidence"] == "medium"
    assert indexed["avoid_pattern"] is False
    assert indexed["grounding_summary"]["left"]["xyz"] == [0.1, 0.2, 0.3]


def test_retriever_returns_only_accepted_lessons_and_avoid_patterns(tmp_path: Path) -> None:
    experience_root = tmp_path
    (experience_root / "retrieval-index").mkdir(parents=True)
    lesson_dir = experience_root / "learned-lessons"
    lesson_dir.mkdir()
    accepted = lesson_dir / "accepted.md"
    candidate = lesson_dir / "candidate.md"
    accepted.write_text("- Lesson ID: `accepted`\n- Status: `accepted`\n\naccepted text", encoding="utf-8")
    candidate.write_text("- Lesson ID: `candidate`\n- Status: `candidate`\n\ncandidate text", encoding="utf-8")
    records = [
        {
            **index_record(_record(), experience_root),
            "lesson_paths": ["learned-lessons/accepted.md", "learned-lessons/candidate.md"],
            "support_count": 3,
            "confidence": "high",
        },
        {
            **index_record(_record("recovery_failed"), experience_root),
            "avoid_pattern": True,
            "failure_penalty": 2,
            "opposing_count": 2,
        },
    ]
    with (experience_root / "retrieval-index" / "index.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    payload = ExperienceRetriever(experience_root).retrieve(
        ExperienceQuery(OOD_scenario="grasp_lost", task_family="pick_and_place", subtask_type="grasp")
    )

    assert [lesson["lesson_id"] for lesson in payload["lessons"]] == ["accepted"]
    assert payload["avoid_patterns"]
    assert payload["avoid_patterns"][0]["failure_penalty"] == 2
    assert payload["similar_cases"][0]["grounding_summary"]["left"]["gripper"] == 0.0

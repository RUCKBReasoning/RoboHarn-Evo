from __future__ import annotations

from policy.roboharn_evo.agent.experience import fallback_semantic_tags, merge_semantic_tags, normalize_semantic_tags


def test_valid_vlm_tags_pass_through() -> None:
    tags = normalize_semantic_tags(
        {
            "task_family": "pick_and_place",
            "subtask_type": "grasp",
            "state_tags": {
                "object_state": "grasped",
                "visibility_state": "visible",
                "gripper_state": "holding_object",
                "motion_state": "moving",
            },
        },
        default_source="planner_vlm",
    )

    assert tags == {
        "task_family": "pick_and_place",
        "subtask_type": "grasp",
        "state_tags": {
            "object_state": "grasped",
            "visibility_state": "visible",
            "gripper_state": "holding_object",
            "motion_state": "moving",
        },
        "tag_source": "planner_vlm",
    }


def test_invalid_tags_are_normalized_to_other_or_empty() -> None:
    tags = normalize_semantic_tags(
        {
            "task_family": "sorting",
            "subtask_type": "teleport",
            "state_tags": {
                "object_state": "magic",
                "visibility_state": "behind Mars",
                "gripper_state": "clenched",
                "motion_state": "flying",
            },
            "tag_source": "invented",
        },
        default_source="ood_vlm",
    )

    assert tags["task_family"] == "other"
    assert tags["subtask_type"] == "other"
    assert tags["state_tags"] == {
        "object_state": "",
        "visibility_state": "",
        "gripper_state": "",
        "motion_state": "",
    }
    assert tags["tag_source"] == "ood_vlm"


def test_ood_state_tags_update_planner_state_only() -> None:
    planner_tags = normalize_semantic_tags(
        {
            "task_family": "pick_and_place",
            "subtask_type": "grasp",
            "state_tags": {"visibility_state": "visible"},
        },
        default_source="planner_vlm",
    )
    ood_tags = normalize_semantic_tags(
        {
            "task_family": "open_drawer",
            "subtask_type": "open",
            "state_tags": {
                "visibility_state": "object_not_visible",
                "motion_state": "blocked",
            },
        },
        default_source="ood_vlm",
    )

    merged = merge_semantic_tags(planner_tags, ood_tags, merge_task_fields=False, merge_state_fields=True)

    assert merged["task_family"] == "pick_and_place"
    assert merged["subtask_type"] == "grasp"
    assert merged["state_tags"]["visibility_state"] == "object_not_visible"
    assert merged["state_tags"]["motion_state"] == "blocked"
    assert merged["tag_source"] == "ood_vlm"


def test_fallback_semantic_tags_ignore_task_and_object_words() -> None:
    first = fallback_semantic_tags(
        global_task="put the mug into the tray",
        current_subtask="grasp the mug",
        signal_name="stall_detected",
    )
    second = fallback_semantic_tags(
        global_task="open the cabinet drawer",
        current_subtask="pull the drawer",
        signal_name="stall_detected",
    )

    assert first == second
    assert first["task_family"] == ""
    assert first["subtask_type"] == ""
    assert first["state_tags"]["motion_state"] == "stalled"
    assert first["tag_source"] == "fallback"

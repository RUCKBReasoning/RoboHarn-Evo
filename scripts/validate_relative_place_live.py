from __future__ import annotations

import argparse
import base64
import copy
import json
from pathlib import Path
import sys
from urllib import request

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboharn_evo.agent.hpk.compatibility import normalize_v3_rollout_record
from roboharn_evo.agent.hpk.hierarchical_retriever import build_hierarchical_hpk_runtime
from roboharn_evo.agent.operation_candidates import operation_pose_candidates, pose7_to_matrix, validate_place_candidate, with_dynamic_place_candidates
from roboharn_evo.agent.perception.scene_memory import SceneMemoryTracker
from roboharn_evo.services.agent_api.qwen_planner import PERCEPTION_QUERY_SYSTEM_PROMPT, validate_perception_queries


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def recorded_placement_state(job: Path) -> dict:
    event_path, = (job / "rollout").glob("episode_*/events.jsonl")
    events = [normalize_v3_rollout_record(json.loads(line)) for line in event_path.read_text().splitlines() if line.strip()]
    goal_index, goal_event = next((index, event) for index, event in enumerate(events) if event["event"] == "hpk_v3_subtask_usage" and event.get("subtask_goal", {}).get("operation") == "place")
    instruction = next(event["instruction"] for event in events if event["event"] == "instruction_set")
    tracker = SceneMemoryTracker()
    scene, manipulation, robot, close_result = None, None, None, None
    subtask = ""
    for event in events[:goal_index]:
        if event["event"] == "hpk_v3_subtask_usage":
            subtask = event["subtask_after"]
        elif event["event"] == "observation_preprocess":
            scene = tracker.update(segmentation=event["segmentation"], env_step=event["env_step"], global_task=instruction, current_subtask=subtask, observation_capture_id=event["observation_capture_id"])
            robot = copy.deepcopy(event["robot_state"])
        elif event["event"] == "agent_instance_binding":
            scene = tracker.rebind_task_focus(scene, perception_queries=event["queries"], global_task=instruction, current_subtask=subtask, identity_binding_required=event["binding_requested"], allow_context_binding=event["allow_context_binding"])
        elif event["event"] == "scene_memory_runtime_events":
            scene = tracker.apply_runtime_events(scene, events=event["events"], env_step=event["env_step"])
        elif event["event"] == "manipulation_state_update":
            manipulation = event["current"]
        elif event["event"] == "recovery_result":
            for result in event["results"]:
                if result["tool_name"] == "close_gripper" and result["success"] is True:
                    close_result = result["details"]
    if scene is None or manipulation is None or robot is None or close_result is None:
        raise ValueError("recorded placement requires real perception, closure, and attachment state")
    arm = close_result["arm"]
    pose = close_result["observed_pose"]
    robot[arm].update(xyz=pose[:3], quat_wxyz=pose[3:], gripper=close_result["gripper_value"])
    attachment = manipulation[arm]["held_object_to_tcp_attachment"]
    # 从同一闭合时刻记录的 action pose 和 TCP pose 还原真实变换。
    transform = np.linalg.inv(pose7_to_matrix(pose)) @ pose7_to_matrix(attachment["capture_tcp_pose"])
    return {
        "scene": scene, "manipulation_state": manipulation, "robot_state": robot,
        "calibration": {arm: {"action_to_tcp_matrix": transform.tolist()}},
        "arm": arm, "instruction": instruction, "subtask": goal_event["subtask_after"],
        "goal": goal_event["subtask_goal"], "step": goal_event["env_step"],
        "episode_directory": event_path.parent, "tracker": tracker,
    }


def evaluate_placement(state: dict, queries: list[dict], *, source_root: Path, server_url: str) -> dict:
    scene = state["tracker"].rebind_task_focus(
        copy.deepcopy(state["scene"]), perception_queries=queries,
        global_task=state["instruction"], current_subtask=state["subtask"],
        identity_binding_required=True, allow_context_binding=True,
    )
    state["tracker"].bind_placement_queries(scene, queries, current_subtask=state["subtask"], goal_contract=state["goal"])
    before = with_dynamic_place_candidates(state["scene"], manipulation_state=state["manipulation_state"], robot_state=state["robot_state"], tcp_calibration_by_arm=state["calibration"], active_grasp_transport_policy="evidence_only")
    after = with_dynamic_place_candidates(scene, manipulation_state=state["manipulation_state"], robot_state=state["robot_state"], tcp_calibration_by_arm=state["calibration"], active_grasp_transport_policy="evidence_only")
    held_id = state["manipulation_state"][state["arm"]]["held_instance_id"]
    held = next(item for item in after["instances"] if item["instance_id"] == held_id)
    runtime = build_hierarchical_hpk_runtime({"mode": "full", "store_root": str(source_root), "hpk_goal_consistency_enabled": True, "knowledge_updates_enabled": False}, planner_agent_api={"server_url": server_url + "/plan", "timeout_sec": 1200})
    runtime.set_current_task_strategy(state["goal"])
    candidates = [item for item in operation_pose_candidates(held) if item.get("target_kind") == "reference_region"]
    if not candidates:
        raise ValueError("no relative placement candidate from the real recorded state")
    checks = []
    for candidate in candidates:
        validity = validate_place_candidate(after, held_instance=held, candidate=candidate)
        resolution = runtime.resolve_goal_consistent_candidates(scene_memory=after, instance=held, action_mode="place", eligible_candidates=[candidate], requested_target_id=candidate["target_id"], all_runtime_candidates=operation_pose_candidates(held), arm=state["arm"])
        checks.append({"candidate": candidate, "validation": validity, "unresolved": resolution.unresolved, "feasibility": resolution.feasibility_report.to_dict()})
        if validity["valid"] is not True or resolution.unresolved:
            raise ValueError("relative placement did not pass runtime geometry and goal checks")
    return {"targets_before": before["operation_targets"], "targets_after": after["operation_targets"], "reference_regions": after["reference_regions"], "checks": checks, "motion_executed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = recorded_placement_state(args.job)
    write_json(args.output / "recorded_state.json", {key: value for key, value in state.items() if key not in {"tracker", "episode_directory"}})
    fields = ("instance_id", "track_id", "class", "query_role", "status", "position_state", "world_m", "top_surface_world_m", "bbox_xyxy", "camera", "score")
    payload = {
        "global_task": state["instruction"], "current_subtask": state["subtask"], "subtask_goal": state["goal"],
        "robot_state": state["robot_state"], "committed_memory": {"manipulation_state": state["manipulation_state"]},
        "scene_instances": [{key: item[key] for key in fields if key in item} for item in state["scene"]["instances"]],
        "instance_binding_phase": "post_detection_selection", "require_instance_binding": True,
        "binding_requirement": {"required_any_roles": ["target", "tool", "context"]}, "max_queries": 3,
        "prompt": PERCEPTION_QUERY_SYSTEM_PROMPT,
    }
    images = {}
    for camera in ("head", "left", "right", "third"):
        image = state["episode_directory"] / camera / f"step_{state['step']:06d}.png"
        content = image.read_bytes()
        (args.output / f"{camera}.png").write_bytes(content)
        images[camera] = base64.b64encode(content).decode("ascii")
    write_json(args.output / "request.json", {**payload, "images": [f"{name}.png" for name in images]})
    if args.prepare_only:
        print(json.dumps({"output": str(args.output), "instances": payload["scene_instances"], "goal": state["goal"]}, ensure_ascii=False), flush=True)
        return
    if args.response is None:
        call = request.Request(args.server_url + "/perception_queries", data=json.dumps({**payload, "image_b64_by_camera": images}).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(call, timeout=1200) as response:
            result = json.loads(response.read())
    else:
        result = json.loads(args.response.read_text())
    write_json(args.output / "response.json", result)
    queries = validate_perception_queries(result)["queries"]
    report = evaluate_placement(state, queries, source_root=args.source_root, server_url=args.server_url)
    write_json(args.output / "report.json", report)
    print(json.dumps({"output": str(args.output), "target_count": len(report["targets_after"]), "relative_candidates": len(report["checks"]), "motion_executed": False}), flush=True)


if __name__ == "__main__":
    main()

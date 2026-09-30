from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboharn_evo.agent.hpk.family_store import load_knowledge_family_catalog
from roboharn_evo.agent.hpk.compatibility import normalize_agent_knowledge_config, normalize_v3_rollout_record
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex


def readable_knowledge(root: Path, tasks, actions, evidence: RGBEvidenceIndex) -> str:
    lines = ["# Cover Blocks HPK", "", "知识来自固定的十条仿真示范轨迹。每条知识的证据计数及适用状态以实际文件为准。", ""]
    for kind, title, values in (("task", "Task Knowledge", tasks), ("action", "Action Knowledge", actions)):
        lines.extend([f"## {title}", ""])
        for index, value in enumerate(values):
            record = value.to_dict()
            lines.extend([f"### {title} {index + 1}", "", "```json", json.dumps(record, ensure_ascii=False, indent=2), "```", ""])
            events = evidence.for_knowledge(kind, index)
            applicable = [event for event in events if event.get("valid_for_current_knowledge", True)]
            examples = [matching[-1] for verdict in ("support", "oppose", "unverified") if (matching := [event for event in applicable if event["verdict"] == verdict])]
            lines.extend([f"关联 {len(events)} 个实际执行事件。完整对应关系保存在 `evidence_index.jsonl`；下面列出实际观察示例。", ""])
            if len(applicable) != len(events):
                lines.extend([f"其中 {len(events) - len(applicable)} 次执行不适用于当前知识，未计入该条知识的证据统计。", ""])
            for event in examples:
                observation = next(item for item in event["before"] if item["camera"] == "head_camera")
                after = next(item for item in event["after"] if item["camera"] == "head_camera")
                lines.extend([f"- 来源：`{event['source_trajectory']}`；动作帧范围：{event['action_range']['start_frame']}–{event['action_range']['end_frame']}；判断：`{event['verdict']}`。", f"  [操作前图片]({observation['path']}) · [操作后图片]({after['path']})", f"  实际观察：{event['observed_result']}"])
            lines.append("")
    return "\n".join(lines)


def verify_source(root: Path) -> dict:
    tasks, actions = load_hierarchical_store(root)
    evidence = RGBEvidenceIndex.load(root)
    evidence.validate_knowledge(tasks, actions)
    catalog = load_knowledge_family_catalog(root, task_knowledge=tasks, action_knowledge=actions)
    manifest = json.loads((root / "source_trajectories.json").read_text())
    expected = set()
    for source in manifest["source_trajectories"]:
        index = RGBEvidenceIndex.load(root, filename=f"episode_{source['episode']}/evidence_index.jsonl")
        expected.update(record["execution_event"] for record in index.records)
    if {record["execution_event"] for record in evidence.records} != expected:
        raise ValueError("exported knowledge does not retain all source execution events")
    images = {image["path"] for record in evidence.records for role in ("before", "after", "delayed") for image in record[role]}
    for image_path in images:
        with Image.open(evidence.image_path(image_path)) as image:
            image.verify()
    return {"source_trajectories": len(manifest["source_trajectories"]), "task_knowledge": len(tasks), "action_knowledge": len(actions), "task_skills": len(catalog.task_families), "action_skills": len(catalog.action_families), "source_execution_events": len(expected), "indexed_rgb_images": len(images)}


def executed_hpk_candidates(events: list[dict]) -> list[dict]:
    events = [normalize_v3_rollout_record(event) for event in events]
    executed = {
        (details.get("step_count"), details.get("operation_candidate_id"), details.get("instance_id"), details.get("arm"))
        for event in events if event["event"] == "recovery_result"
        for result in event["results"] if result["success"]
        if (details := result.get("details", {})).get("operation_candidate_id")
    }
    records = []
    for event in events:
        if event["event"] != "operation_candidate_selected" or event.get("result_env_step") is None:
            continue
        usage = event.get("hpk_v3_action_usage", {})
        key = (event["result_env_step"], event["candidate_id"], event["instance_id"], event["arm"])
        if usage.get("knowledge_adopted") is True and key in executed:
            records.append({"dispatch_step": event["dispatch_env_step"], "result_step": event["result_env_step"], "action": event["action_mode"], "arm": event["arm"], "knowledge": usage["retrieved_knowledge"], "geometry": usage["selected_candidate_features"], "behavior_changed": usage["behavior_changed"]})
    return records


def verify_run_artifacts(directory: Path, *, instruction: str, mode: str) -> dict:
    event_path, = (directory / "rollout").glob("episode_*/events.jsonl")
    events = [normalize_v3_rollout_record(json.loads(line)) for line in event_path.read_text().splitlines() if line.strip()]
    if {event["instruction"] for event in events if event["event"] == "instruction_set"} != {instruction}:
        raise ValueError("integration instruction differs from the fixed source protocol")
    calls = [call for event in events if event["event"] == "recovery_router" for call in event["tool_calls"]]
    results = [result for event in events if event["event"] == "recovery_result" for result in event["results"]]
    effects = [event for event in events if event["event"] == "action_effect_verification"]
    executed = [result for result in results if result["success"] and result.get("details", {}).get("step_count", 0) > 0]
    if not calls or not executed or not effects:
        raise ValueError("integration requires actual tool execution and recorded physical-effect checks")
    snapshots = {}
    for camera in ("head", "left", "right", "third"):
        paths = sorted((event_path.parent / camera).glob("*.png"))
        if not paths:
            raise ValueError(f"integration has no {camera} RGB snapshots")
        for path in paths:
            with Image.open(path) as image:
                image.verify()
        snapshots[camera] = len(paths)
    video_paths = sorted((directory / "rollout").rglob("episode_0000_rollout/video/*.mp4"))
    if len(video_paths) != 4 or {path.stem for path in video_paths} != set(snapshots):
        raise ValueError("integration requires all four camera videos")
    videos = []
    for path in video_paths:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,nb_frames,width,height", "-of", "json", str(path)], check=True, capture_output=True, text=True)
        stream, = json.loads(probe.stdout)["streams"]
        if Fraction(stream["avg_frame_rate"]) != 30 or int(stream["nb_frames"]) <= 0:
            raise ValueError("integration video must contain real frames at 30 fps")
        videos.append({"path": path.relative_to(directory).as_posix(), **stream})
    retrievals = []
    request_paths = set(directory.rglob("hpk_rgb_retrieval/**/request.json"))
    request_paths.update(directory.rglob("afk_rgb_retrieval/**/request.json"))
    for path in sorted(request_paths):
        request = json.loads(path.read_text())
        bindings = request["input"]["image_bindings"]
        if len(bindings) != len(request["images"]):
            raise ValueError("retrieval image bindings and saved PNG files differ")
        roles = {binding["role"] for binding in bindings}
        if not {"current observation", "before", "after"} <= roles:
            raise ValueError("retrieval must include current RGB and paired historical RGB")
        for name in request["images"]:
            with Image.open(path.parent / name) as image:
                image.verify()
        response_path = path.parent / "response.json"
        response = json.loads(response_path.read_text()) if response_path.is_file() else None
        retrievals.append({"directory": path.parent.relative_to(directory).as_posix(), "schema": request["schema_name"], "image_count": len(bindings), "response_received": response is not None, "selected_knowledge_index": None if response is None else response["selected_knowledge_index"]})
    legacy_schemas = {
        "afk_v31_subtask_retrieval": "hpk_v31_subtask_retrieval",
        "afk_v3_action_retrieval": "hpk_v3_action_retrieval",
    }
    completed_schemas = {legacy_schemas.get(request["schema"], request["schema"]) for request in retrievals if request["response_received"]}
    if mode == "full" and not {"hpk_v31_subtask_retrieval", "hpk_v3_action_retrieval"} <= completed_schemas:
        raise ValueError("Full HPK requires actual Task and Action RGB retrieval responses")
    return {"tool_calls": len(calls), "executed_tool_results": len(executed), "physical_effect_checks": len(effects), "task_knowledge_adoptions": sum(event["event"] == "hpk_v3_subtask_usage" and event.get("knowledge_adopted") is True for event in events), "executed_hpk_candidates": executed_hpk_candidates(events), "snapshots": snapshots, "videos": videos, "retrievals": retrievals}


def verify(root: Path) -> dict:
    report = verify_source(root)
    manifest = json.loads((root / "source_trajectories.json").read_text())
    build = json.loads((root / "store_build_report.json").read_text())
    for field in ("model", "reasoning_effort"):
        if build[field] != manifest["config"]["model"][field]:
            raise ValueError("knowledge build model differs from the fixed source configuration")
    report["knowledge_build"] = {name: build[name] for name in ("model", "reasoning_effort", "source_episodes", "bootstrap_episodes", "incremental_episode", "model_usage")}
    case_manifest = json.loads((root / "integration_runs/cases.json").read_text())
    results = []
    run_reports = []
    integration = manifest["config"]["integration"]
    expected_runs = {(seed, method) for seed in integration["seeds"] for method in integration["methods"]}
    actual_runs = set()
    for case in case_manifest["cases"]:
        directory = root / "integration_runs" / case["directory"]
        result = json.loads((directory / "status.json").read_text())
        results.append(result)
        if result["status"] != "completed" or type(result["success"]) is not bool:
            raise ValueError("every integration execution must include its actual final benchmark result")
        config = json.loads((directory / "run_config.json").read_text())
        config["agent"] = normalize_agent_knowledge_config(config["agent"])
        mode = config["agent"]["hpk_v3"]["mode"]
        if (result["seed"], mode) != (case["seed"], case["method"]):
            raise ValueError("integration case metadata differs from the executed configuration")
        if result["automatic_retry"] is not False or config["agent"]["hpk_v3"]["knowledge_updates_enabled"] is not False:
            raise ValueError("integration must use read-only knowledge without automatic retries")
        if config["eval"]["step_limit"] != integration["action_limit"] or config["agent"]["pure_tool_control"]["max_control_turns"] != integration["control_turn_limit"]:
            raise ValueError("integration budgets differ from the fixed source protocol")
        actual_runs.add((result["seed"], mode))
        if mode == "full":
            if result.get("knowledge_store_unchanged") is not True:
                raise ValueError("Full HPK integration must preserve source knowledge")
        run_reports.append({**case, "success": result["success"], "total_steps": result["total_steps"], **verify_run_artifacts(directory, instruction=manifest["instruction"], mode=mode)})
    if actual_runs != expected_runs or len(results) != len(expected_runs):
        raise ValueError("integration executions differ from the fixed source protocol")
    return {**report, "integration_executions": len(results), "integration_successes": sum(result["success"] for result in results), "integration_evidence": run_reports}


def package_files(root: Path) -> set[Path]:
    files = {root / name for name in ("task_knowledge.jsonl", "action_knowledge.jsonl", "knowledge_families.json", "evidence_index.jsonl", "source_trajectories.json", "preparation_summary.json", "store_build_report.json", "build/semantic_groups.json", "build/model_responses.json", "HPK_READABLE.md", "USE_ON_REAL_ROBOT.md", "integration_plan.json", "package_report.json", "handoff_terminal_tests.xml", "relative_placement_tests.xml", "relative_place_motion_tests.xml", "integration_runs/cases.json")}
    manifest = json.loads((root / "source_trajectories.json").read_text())
    for source in manifest["source_trajectories"]:
        directory = root / f"episode_{source['episode']}"
        files.update(directory / name for name in ("reflector_input.json", "reflection_request.json", "reflection_response.json", "trajectory_knowledge_package.json", "atomic_knowledge.json", "evidence_index.jsonl", "source_observations.json", "robot_state_and_actions.npz", "planned_joint_actions.npz", "planned_joint_actions.json"))
    for name in ("evidence", "build/maintenance", "runtime_validation"):
        files.update(path for path in (root / name).rglob("*") if path.is_file())
    cases = json.loads((root / "integration_runs/cases.json").read_text())["cases"]
    for case in cases:
        case_path = Path(case["directory"])
        if len(case_path.parts) > 1:
            files.add(root / f"integration_plan_{case_path.parts[0]}.json")
        for name in ("integration_runs", "integration_segmentation"):
            directory = root / name / case_path
            files.update(path for path in directory.rglob("*") if path.is_file() and not {"runtime_workspace", "runtime_caches", "runtime_configs", "__pycache__"}.intersection(path.relative_to(directory).parts))
    return files


def write_documents(root: Path) -> tuple[Path, Path]:
    tasks, actions = load_hierarchical_store(root)
    evidence = RGBEvidenceIndex.load(root)
    evidence.validate_knowledge(tasks, actions)
    (root / "HPK_READABLE.md").write_text(readable_knowledge(root, tasks, actions, evidence), encoding="utf-8")
    (root / "USE_ON_REAL_ROBOT.md").write_text(
        "# 真机读取\n\n解压后保留知识、图片和索引的相对位置。使用 RoboHarn-Evo 的共享 Runtime 接收当前 RGB 观察。\n\n"
        "读取接口：`load_hierarchical_store(package_root)`、`RGBEvidenceIndex.load(package_root)` 和 `load_knowledge_family_catalog(package_root, task_knowledge=tasks, action_knowledge=actions)`。\n\n"
        "```python\nfrom pathlib import Path\nfrom roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store\nfrom roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex\nfrom roboharn_evo.agent.hpk.family_store import load_knowledge_family_catalog\n\npackage_root = Path(\"cover_blocks_source\").resolve()\ntasks, actions = load_hierarchical_store(package_root)\nevidence = RGBEvidenceIndex.load(package_root)\ncatalog = load_knowledge_family_catalog(package_root, task_knowledge=tasks, action_knowledge=actions)\nimage_path = evidence.image_path(evidence.records[0][\"before\"][0][\"path\"])\n```\n\n"
        "当前 RGB 由真机相机提供，当前物体与目标由真机 Scene Memory 绑定；历史图片通过 `RGBEvidenceIndex.image_path` 读取。Task 检索接收 baseline subtask；Action 检索接收本次已经生成的合法几何候选。\n\n"
        "zero-shot 配置使用 `mode: full`、`knowledge_updates_enabled: false`、`retrieval_rgb_enabled: true`，设置 `store_root`、独立的 `rgb_audit_root`、实际模型与图片容量。源图像的相机名称为 `head_camera`。\n\n"
        "```yaml\nagent:\n  hpk_v3:\n    mode: full\n    store_root: /absolute/path/to/cover_blocks_source\n    knowledge_updates_enabled: false\n    retrieval_rgb_enabled: true\n    historical_rgb_cameras: [head_camera]\n    max_rgb_images: 32\n    retrieval_model: gpt-5.5\n    retrieval_reasoning_effort: xhigh\n    hpk_goal_consistency_enabled: true\n    rgb_audit_root: /absolute/path/to/current_run/hpk_rgb_retrieval\n```\n\n"
        "使用现有 `build_hierarchical_hpk_runtime` 和 `runtime.set_current_rgb` 接入当前观察。保留控制器、碰撞检查、抓取验证和完整执行日志。真机使用不修改源 HPK。\n\n"
        "当前相机数组必须为 `uint8` 的 RGB 三通道格式。使用 `current_rgb_image(rgb_array, camera=camera_name)` 编码，再将图片元组交给 `runtime.set_current_rgb`。该调用仅设置当前观察；模型请求由 Task/Action 检索触发。每次决策使用本次新观察，历史图片从包中按相对路径加载。\n\n"
        "Task 检索保留总体子任务的目的和顺序，当前 Goal Contract 描述下一项可执行的 `grasp`、`place` 或 `contact`。抓取阶段完成条件来自物体随夹爪运动等执行证据，放置阶段保留本次目标及关系。知识中的历史颜色和位置不替代当前 Scene Memory 的绑定。\n\n"
        "在 RoboHarn-Evo 项目中读取并检查解压后的知识、Skill catalog 和全部索引图片：\n\n```bash\n"
        "python scripts/export_cover_blocks_hpk.py --source-root /absolute/path/to/cover_blocks_source --verify-source-only\n```\n\n"
        "核验包内四条仿真轨迹、实际检索图片、动作、效果记录及 30 fps 视频：\n\n```bash\n"
        "python scripts/export_cover_blocks_hpk.py --source-root /absolute/path/to/cover_blocks_source --verify-only\n```\n\n"
        "本包使用的四条仿真轨迹由 `integration_runs/cases.json` 指定。`package_report.json` 逐条列出实际结果、检索目录、图片数量、工具执行次数和视频位置。终局结果以各条轨迹的 benchmark 评分为准。\n",
        encoding="utf-8",
    )
    return root / "HPK_READABLE.md", root / "USE_ON_REAL_ROBOT.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "eval_result/hpk/q4/cover_blocks_source")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-source-only", action="store_true")
    parser.add_argument("--write-readable", action="store_true", help="write readable knowledge and real-robot loading instructions")
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    if args.verify_source_only:
        print(json.dumps(verify_source(root), indent=2))
        return
    if args.write_readable:
        for path in write_documents(root):
            print(str(path))
        return
    report = verify(root)
    if args.verify_only:
        print(json.dumps(report, indent=2))
        return
    write_documents(root)
    (root / "package_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target = root / "cover_blocks_hpk.zip"
    with ZipFile(target, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(package_files(root)):
            archive.write(path, (Path("cover_blocks_source") / path.relative_to(root)).as_posix(), compress_type=ZIP_STORED if path.suffix.lower() in {".png", ".mp4", ".npz", ".gz"} else ZIP_DEFLATED)
        if archive.testzip() is not None:
            raise ValueError("archive verification failed")
    print(json.dumps({**report, "archive": str(target), "bytes": target.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

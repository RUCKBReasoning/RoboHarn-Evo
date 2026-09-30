#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.agent.experience import FAILURE_OUTCOMES, SUCCESS_OUTCOMES, normalize_ood_scenario, stable_experience_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic RoboHarn-Evo experience retrieval index.")
    parser.add_argument("--experience-root", type=Path, default=Path("policy/roboharn_evo/skills/experience"))
    parser.add_argument("--recovery-trials", type=Path, default=Path("policy/roboharn_evo/skills/experience/raw-traces/recovery_trials.jsonl"))
    parser.add_argument("--avoid-patterns", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/avoid_patterns.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/index.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def case_summary_path(record: dict[str, Any], experience_root: Path) -> Path:
    case_id = str(record.get("case_id", "")).strip()
    if not case_id:
        scenario = normalize_ood_scenario(record.get("OOD_scenario", ""))
        prefix = f"{record.get('task_family', 'task')}_{scenario}_{record.get('outcome', 'outcome')}"
        case_id = stable_experience_id(prefix, {"trial_id": record.get("trial_id", ""), "trace": record.get("trace_file", "")})
    return experience_root / "case-summaries" / f"{case_id}.md"


def discover_lesson_paths(experience_root: Path, record: dict[str, Any]) -> list[str]:
    scenario = normalize_ood_scenario(record.get("OOD_scenario", ""))
    task_family = str(record.get("task_family", "")).strip()
    matched: list[str] = []
    lessons_dir = experience_root / "learned-lessons"
    if not lessons_dir.exists():
        return matched
    for path in sorted(lessons_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").lower()
        if "status: `accepted`" not in text and "status: accepted" not in text:
            continue
        if scenario != "none" and scenario not in text:
            continue
        if task_family and task_family not in text:
            continue
        matched.append(relpath(path, experience_root))
    return matched


def tool_sequence(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in record.get("tool_calls", []) or []:
        if isinstance(item, dict):
            name = str(item.get("tool_name", "")).strip()
            if name:
                names.append(name)
    return names


def grounding_summary(record: dict[str, Any]) -> dict[str, Any]:
    robot_state = record.get("robot_state", {})
    if not isinstance(robot_state, dict) or not robot_state:
        return {}
    summary: dict[str, Any] = {
        "step": robot_state.get("step", record.get("step", "")),
        "joint_norm": robot_state.get("joint_norm", ""),
    }
    for arm in ("left", "right"):
        arm_state = robot_state.get(arm, {})
        if not isinstance(arm_state, dict):
            continue
        summary[arm] = {
            "xyz": arm_state.get("xyz", []),
            "rpy": arm_state.get("rpy", []),
            "gripper": arm_state.get("gripper", None),
        }
    return summary


def confidence_for_record(record: dict[str, Any]) -> str:
    if int(record.get("support_count", 0) or 0) >= 5:
        return "high"
    if int(record.get("support_count", 0) or 0) >= 2:
        return "medium"
    outcome = str(record.get("outcome", "")).strip()
    return "medium" if outcome in SUCCESS_OUTCOMES else "low"


def support_count_for_record(record: dict[str, Any]) -> int:
    if "support_count" in record:
        return max(0, int(record.get("support_count", 0) or 0))
    return 1 if str(record.get("outcome", "")).strip() in SUCCESS_OUTCOMES else 0


def opposing_count_for_record(record: dict[str, Any]) -> int:
    if "opposing_count" in record:
        return max(0, int(record.get("opposing_count", 0) or 0))
    return 1 if str(record.get("outcome", "")).strip() in FAILURE_OUTCOMES else 0


def index_record(record: dict[str, Any], experience_root: Path) -> dict[str, Any]:
    case_path = case_summary_path(record, experience_root)
    case_id = case_path.stem
    tags = {str(tag).strip() for tag in record.get("retrieval_tags", []) if str(tag).strip()}
    tags.update(
        value for value in [
            normalize_ood_scenario(record.get("OOD_scenario", "")),
            str(record.get("task_family", "")).strip(),
            str(record.get("subtask_type", "")).strip(),
            str(record.get("post_recovery_intent", "")).strip(),
            str(record.get("outcome", "")).strip(),
        ] if value
    )
    tools = tool_sequence(record)
    tags.update(tools)
    support_count = support_count_for_record(record)
    opposing_count = opposing_count_for_record(record)
    avoid_pattern = bool(record.get("avoid_pattern", False)) or str(record.get("outcome", "")).strip() in FAILURE_OUTCOMES
    confidence = str(record.get("confidence", "")).strip() or confidence_for_record({**record, "support_count": support_count})
    failure_penalty = max(0, int(record.get("failure_penalty", 0) or 0))
    if avoid_pattern:
        failure_penalty = max(failure_penalty, opposing_count, 1)
    return {
        "case_id": case_id,
        "trial_id": str(record.get("trial_id", "")),
        "trace_id": str(record.get("trace_id", record.get("trace_file", ""))),
        "OOD_scenario": normalize_ood_scenario(record.get("OOD_scenario", "")),
        "task_family": str(record.get("task_family", "")),
        "object": str(record.get("object", "")),
        "subtask_type": str(record.get("subtask_type", "")),
        "recovery_workflow": str(record.get("recovery_workflow", "")),
        "recovery_primitives": tools,
        "tool_sequence": tools,
        "post_recovery_intent": str(record.get("post_recovery_intent", "")),
        "outcome": str(record.get("outcome", "")),
        "summary_path": relpath(case_path, experience_root),
        "trace_path": relpath(Path(str(record.get("trace_file", ""))), experience_root) if record.get("trace_file") else "",
        "lesson_paths": discover_lesson_paths(experience_root, record),
        "grounding_summary": grounding_summary(record),
        "retrieval_tags": sorted(tags),
        "support_count": support_count,
        "opposing_count": opposing_count,
        "last_validated_at": str(record.get("last_validated_at", "")),
        "confidence": confidence,
        "avoid_pattern": avoid_pattern,
        "failure_penalty": failure_penalty,
        "notes": str(record.get("notes", record.get("failure_reason", ""))),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Index already exists: {args.output}. Use --overwrite.")
    experience_root = args.experience_root.resolve()
    records = read_jsonl(args.recovery_trials) + read_jsonl(args.avoid_patterns)
    indexed = [index_record(record, experience_root) for record in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in indexed:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"index_records": len(indexed), "output": str(args.output)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

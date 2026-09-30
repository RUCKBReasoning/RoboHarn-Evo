#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.agent.experience import FAILURE_OUTCOMES, SUCCESS_OUTCOMES, normalize_ood_scenario, stable_experience_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine candidate RoboHarn-Evo recovery lessons and avoid patterns from recovery trial ledgers.")
    parser.add_argument("--recovery-trials", type=Path, default=Path("policy/roboharn_evo/skills/experience/raw-traces/recovery_trials.jsonl"))
    parser.add_argument("--lesson-dir", type=Path, default=Path("policy/roboharn_evo/skills/experience/learned-lessons"))
    parser.add_argument("--avoid-output", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/avoid_patterns.jsonl"))
    parser.add_argument("--min-support", type=int, default=2)
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


def slugify(value: str, *, fallback: str = "lesson") -> str:
    slug = str(value or "").strip().lower().replace("-", "_")
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or fallback


def state_tags(record: dict[str, Any]) -> dict[str, str]:
    tags = record.get("semantic_tags", {})
    if not isinstance(tags, dict):
        return {}
    state = tags.get("state_tags", {})
    if not isinstance(state, dict):
        return {}
    return {str(key): str(value) for key, value in state.items() if str(value).strip()}


def tool_sequence(record: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for item in record.get("tool_calls", []) or []:
        if isinstance(item, dict):
            name = str(item.get("tool_name", "")).strip()
            if name:
                names.append(name)
    return tuple(names)


def cluster_key(record: dict[str, Any]) -> tuple[str, ...]:
    state = state_tags(record)
    return (
        normalize_ood_scenario(record.get("OOD_scenario", "")),
        str(record.get("task_family", "")).strip(),
        str(record.get("subtask_type", "")).strip(),
        str(state.get("object_state", "")),
        str(state.get("visibility_state", "")),
        str(state.get("gripper_state", "")),
        str(state.get("motion_state", "")),
        str(record.get("recovery_workflow", "")).strip(),
        ",".join(tool_sequence(record)),
        str(record.get("post_recovery_intent", "")).strip(),
    )


def case_id(record: dict[str, Any]) -> str:
    existing = str(record.get("case_id", "")).strip()
    if existing:
        return existing
    prefix = "_".join(
        part
        for part in [
            slugify(str(record.get("task_family", "")), fallback="task"),
            slugify(normalize_ood_scenario(record.get("OOD_scenario", "")), fallback="ood"),
            slugify(str(record.get("outcome", "")), fallback="outcome"),
        ]
        if part
    )
    return stable_experience_id(prefix, {"trial_id": record.get("trial_id", ""), "trace": record.get("trace_file", "")})


def confidence(support_count: int, opposing_count: int) -> str:
    if support_count >= 5 and opposing_count == 0:
        return "high"
    if support_count >= 2 and opposing_count <= 1:
        return "medium"
    return "low"


def title_for_cluster(key: tuple[str, ...], *, avoid: bool = False) -> str:
    scenario, task_family, subtask_type, _object_state, visibility_state, gripper_state, motion_state, workflow, _tools, intent = key
    prefix = "Avoid" if avoid else "Prefer"
    state = ", ".join(part for part in [visibility_state, gripper_state, motion_state] if part)
    suffix = f" when {state}" if state else ""
    return f"{prefix} {workflow or intent or 'recovery'} for {scenario or 'OOD'} {subtask_type or 'subtask'} in {task_family or 'task'}{suffix}"


def render_lesson(key: tuple[str, ...], records: list[dict[str, Any]], *, opposing_records: list[dict[str, Any]], min_support: int) -> str:
    scenario, task_family, subtask_type, object_state, visibility_state, gripper_state, motion_state, workflow, tools, intent = key
    support_cases = [case_id(record) for record in records]
    opposing_cases = [case_id(record) for record in opposing_records]
    support_count = len(support_cases)
    opposing_count = len(opposing_cases)
    lesson_id = stable_experience_id(
        "lesson",
        {
            "key": key,
            "support": sorted(support_cases),
            "opposing": sorted(opposing_cases),
        },
        length=12,
    )
    state_lines = [
        f"  - Object state: `{object_state}`" if object_state else "",
        f"  - Visibility state: `{visibility_state}`" if visibility_state else "",
        f"  - Gripper state: `{gripper_state}`" if gripper_state else "",
        f"  - Motion state: `{motion_state}`" if motion_state else "",
    ]
    tool_text = ", ".join(f"`{tool}`" for tool in tools.split(",") if tool) or "no recovery tools"
    status = "candidate"
    return f"""# Lesson: {title_for_cluster(key)}

## Metadata

- Lesson ID: `{lesson_id}`
- Status: `{status}`
- Confidence: `{confidence(support_count, opposing_count)}`
- Support Count: `{support_count}`
- Opposing Count: `{opposing_count}`
- Last Validated At: ``

## Applies When

- OOD_scenario: `{scenario}`
- Task family: `{task_family}`
- Subtask type: `{subtask_type}`
- Preconditions:
{chr(10).join(line for line in state_lines if line) or "  - Similar semantic tags are present in the recovery query."}

## Recommendation

Prefer recovery workflow `{workflow}` with tool sequence {tool_text}, then use post-recovery intent `{intent}` when the current observation matches the preconditions.

## Avoid

Avoid immediately changing to a more disruptive recovery route unless current adapter capabilities or recovery history make `{workflow}` unavailable or repeatedly failed.

## Supporting Cases

{chr(10).join(f"- `{item}`" for item in support_cases)}

## Opposing Cases

{chr(10).join(f"- `{item}`" for item in opposing_cases) or "- None recorded in the mined ledger."}

## Evaluation Result

- Status: `not_evaluated`
- Required support before promotion: `{max(1, min_support)}`

## Promotion Criteria

Promote to `accepted` only after regression evaluation shows that this lesson improves workflow/tool/post-intent selection without increasing invalid recovery plans.

## Notes

Generated by `mine_experience_lessons.py` from factual recovery ledgers. Keep this lesson short and downgrade it if future opposing cases accumulate.
"""


def avoid_pattern_record(key: tuple[str, ...], records: list[dict[str, Any]], *, support_records: list[dict[str, Any]]) -> dict[str, Any]:
    scenario, task_family, subtask_type, object_state, visibility_state, gripper_state, motion_state, workflow, tools, intent = key
    cases = [case_id(record) for record in records]
    support_cases = [case_id(record) for record in support_records]
    return {
        "case_id": stable_experience_id("avoid_pattern", {"key": key, "cases": sorted(cases)}, length=12),
        "trial_id": "",
        "trace_id": "",
        "OOD_scenario": scenario,
        "task_family": task_family,
        "object": object_state,
        "subtask_type": subtask_type,
        "recovery_workflow": workflow,
        "recovery_primitives": [tool for tool in tools.split(",") if tool],
        "tool_sequence": [tool for tool in tools.split(",") if tool],
        "post_recovery_intent": intent,
        "outcome": "recovery_failed",
        "summary_path": "",
        "trace_path": "",
        "lesson_paths": [],
        "retrieval_tags": sorted(
            tag
            for tag in {
                scenario,
                task_family,
                subtask_type,
                object_state,
                visibility_state,
                gripper_state,
                motion_state,
                workflow,
                intent,
                *[tool for tool in tools.split(",") if tool],
            }
            if tag
        ),
        "support_count": len(support_cases),
        "opposing_count": len(cases),
        "last_validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "confidence": confidence(len(support_cases), len(cases)),
        "avoid_pattern": True,
        "failure_penalty": max(1, len(cases)),
        "notes": f"Avoid pattern mined from failed/aborted cases: {', '.join(cases)}",
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.recovery_trials)
    success_clusters: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    failure_clusters: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        outcome = str(record.get("outcome", "")).strip()
        key = cluster_key(record)
        if outcome in SUCCESS_OUTCOMES:
            success_clusters[key].append(record)
        elif outcome in FAILURE_OUTCOMES:
            failure_clusters[key].append(record)

    args.lesson_dir.mkdir(parents=True, exist_ok=True)
    written_lessons: list[str] = []
    for key, cluster_records in sorted(success_clusters.items(), key=lambda item: item[0]):
        if len(cluster_records) < max(1, args.min_support):
            continue
        lesson_id = stable_experience_id(
            "lesson",
            {
                "key": key,
                "support": sorted(case_id(record) for record in cluster_records),
                "opposing": sorted(case_id(record) for record in failure_clusters.get(key, [])),
            },
            length=12,
        )
        output_path = args.lesson_dir / f"{lesson_id}.md"
        if output_path.exists() and not args.overwrite:
            continue
        output_path.write_text(
            render_lesson(key, cluster_records, opposing_records=failure_clusters.get(key, []), min_support=args.min_support),
            encoding="utf-8",
        )
        written_lessons.append(str(output_path))

    avoid_records = [
        avoid_pattern_record(key, cluster_records, support_records=success_clusters.get(key, []))
        for key, cluster_records in sorted(failure_clusters.items(), key=lambda item: item[0])
    ]
    args.avoid_output.parent.mkdir(parents=True, exist_ok=True)
    with args.avoid_output.open("w", encoding="utf-8") as f:
        for record in avoid_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "records_read": len(records),
                "success_clusters": len(success_clusters),
                "failure_clusters": len(failure_clusters),
                "lessons_written": len(written_lessons),
                "avoid_patterns_written": len(avoid_records),
                "lesson_paths": written_lessons,
                "avoid_output": str(args.avoid_output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

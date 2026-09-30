#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.agent.experience import normalize_ood_scenario, stable_experience_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create factual case summaries from RoboHarn-Evo recovery trial ledgers.")
    parser.add_argument("--recovery-trials", type=Path, default=Path("policy/roboharn_evo/skills/experience/raw-traces/recovery_trials.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("policy/roboharn_evo/skills/experience/case-summaries"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
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


def slugify(value: str, *, fallback: str = "case") -> str:
    slug = str(value or "").strip().lower().replace("-", "_")
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or fallback


def case_id_for_trial(record: dict[str, Any]) -> str:
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


def tool_sequence(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in record.get("tool_calls", []) or []:
        if isinstance(item, dict):
            name = str(item.get("tool_name", "")).strip()
            if name:
                names.append(name)
    return names


def result_lines(record: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in record.get("tool_results", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name", "")).strip()
        success = item.get("success")
        message = str(item.get("message", "")).strip()
        lines.append(f"- {name}: success={success}; message={message}")
    return lines or ["- No recovery tool result was recorded."]


def _arm_grounding_lines(record: dict[str, Any]) -> list[str]:
    robot_state = record.get("robot_state", {})
    if not isinstance(robot_state, dict) or not robot_state:
        return ["- No structured robot grounding state was recorded."]
    lines = [
        f"- Robot state step: `{robot_state.get('step', record.get('step', ''))}`",
        f"- Joint norm: `{robot_state.get('joint_norm', '')}`",
    ]
    for arm in ("left", "right"):
        arm_state = robot_state.get(arm, {})
        if not isinstance(arm_state, dict):
            continue
        lines.append(
            f"- {arm} arm: xyz=`{arm_state.get('xyz', [])}`, "
            f"rpy=`{arm_state.get('rpy', [])}`, gripper=`{arm_state.get('gripper', None)}`"
        )
    return lines


def render_case_summary(record: dict[str, Any], *, case_id: str) -> str:
    scenario = normalize_ood_scenario(record.get("OOD_scenario", ""))
    tools = tool_sequence(record)
    tool_steps = "\n".join(f"{index}. `{name}`" for index, name in enumerate(tools, start=1)) or "No recovery tool call was recorded."
    trace_file = str(record.get("trace_file", "")).strip()
    failure_reason = str(record.get("failure_reason", "")).strip()
    observation_summary = str(record.get("observation_summary", "")).strip()
    if not observation_summary:
        observation_summary = "No compact observation summary was recorded in the extracted trace."

    return f"""# Case: {scenario} during {record.get("subtask_type", "subtask")}

## Metadata

- Case ID: `{case_id}`
- Trial ID: `{record.get("trial_id", "")}`
- Trace file: `{trace_file}`
- Episode ID: `{record.get("episode_id", "")}`
- Seed: `{record.get("seed", "")}`
- Step: `{record.get("step", "")}`
- Task family: `{record.get("task_family", "")}`
- Task: `{record.get("global_task", "")}`
- Subtask: `{record.get("subtask", "")}`
- OOD_scenario: `{scenario}`
- Recovery workflow: `{record.get("recovery_workflow", "")}`
- Post recovery intent: `{record.get("post_recovery_intent", "")}`
- Post recovery decision: `{record.get("post_recovery_decision", "")}`
- Outcome: `{record.get("outcome", "")}`
- Status: `candidate`

## Context

{observation_summary}

## Robot Grounding

{chr(10).join(_arm_grounding_lines(record))}

## Evidence

- Monitor signal: `{record.get("monitor_signal", "")}`
- Failure reason: {failure_reason or "No failure reason was recorded."}

## Intervention

{tool_steps}

## Outcome

{chr(10).join(result_lines(record))}

Final outcome: `{record.get("outcome", "")}`.

## Notes

This is a factual case summary generated from a runtime trace. It is not a learned lesson and should not be treated as a hard recovery rule.
"""


def eligible(record: dict[str, Any]) -> bool:
    return bool(record.get("recovery_workflow")) or normalize_ood_scenario(record.get("OOD_scenario", "")) != "none"


def main() -> None:
    args = parse_args()
    records = [record for record in read_jsonl(args.recovery_trials) if eligible(record)]
    if args.limit > 0:
        records = records[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for record in records:
        case_id = case_id_for_trial(record)
        output_path = args.output_dir / f"{case_id}.md"
        if output_path.exists() and not args.overwrite:
            continue
        output_path.write_text(render_case_summary(record, case_id=case_id), encoding="utf-8")
        written.append(str(output_path))
    print(json.dumps({"cases_written": len(written), "paths": written}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

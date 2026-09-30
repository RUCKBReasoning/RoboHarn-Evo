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

from policy.roboharn_evo.agent.experience import normalize_ood_scenario, stable_experience_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact /recover regression payload fixtures from recovery trials.")
    parser.add_argument("--recovery-trials", type=Path, default=Path("policy/roboharn_evo/skills/experience/raw-traces/recovery_trials.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/recovery_regression_payloads.jsonl"))
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


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    scenario = normalize_ood_scenario(record.get("OOD_scenario", ""))
    payload_id = stable_experience_id(
        "recover_payload",
        {"trial_id": record.get("trial_id", ""), "trace": record.get("trace_file", "")},
        length=12,
    )
    return {
        "payload_id": payload_id,
        "case_id": str(record.get("case_id", "")),
        "signal_name": str(record.get("monitor_signal", scenario)),
        "OOD_scenario": scenario,
        "reason": str(record.get("failure_reason", "")),
        "current_subtask": str(record.get("subtask", "")),
        "observation_summary": str(record.get("observation_summary", "")),
        "robot_state": dict(record.get("robot_state", {}) or {}),
        "recovery_state": {},
        "recovery_history": [],
        "available_tools": list(record.get("available_tools", []) or []),
        "semantic_tags": dict(record.get("semantic_tags", {}) or {}),
        "retrieved_experience": {"lessons": [], "similar_cases": [], "avoid_patterns": []},
        "expected": {
            "recovery_workflow": str(record.get("recovery_workflow", "")),
            "post_recovery_intent": str(record.get("post_recovery_intent", "")),
            "tool_sequence": [
                str(item.get("tool_name", "")).strip()
                for item in record.get("tool_calls", []) or []
                if isinstance(item, dict) and str(item.get("tool_name", "")).strip()
            ],
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Regression payload file already exists: {args.output}. Use --overwrite.")
    records = read_jsonl(args.recovery_trials)
    payloads = [
        build_payload(record)
        for record in records
        if normalize_ood_scenario(record.get("OOD_scenario", "")) != "none"
        and str(record.get("recovery_workflow", "")).strip()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for payload in payloads:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"payloads": len(payloads), "output": str(args.output)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

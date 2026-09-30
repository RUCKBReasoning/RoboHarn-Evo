#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.agent.experience import ACCEPTED_LESSON_STATUS, normalize_ood_scenario
from policy.roboharn_evo.models.backend_factory import build_recovery_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate experience lessons against recovery regression payloads.")
    parser.add_argument("--lesson-dir", type=Path, default=Path("policy/roboharn_evo/skills/experience/learned-lessons"))
    parser.add_argument("--payloads", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/recovery_regression_payloads.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/lesson_eval_results.jsonl"))
    parser.add_argument("--backend-config", type=Path, default=None, help="Optional JSON backend config for live /recover evaluation.")
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


def metadata_value(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*-\s*{re.escape(key)}:\s*`?([^`\n]+)`?\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def section(text: str, name: str) -> str:
    pattern = re.compile(rf"^##\s+{re.escape(name)}\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_match = re.search(r"^##\s+", text[start:], re.MULTILINE)
    end = start + next_match.start() if next_match else len(text)
    return text[start:end].strip()


def lesson_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "lesson_id": metadata_value(text, "Lesson ID") or path.stem,
        "path": str(path),
        "status": metadata_value(text, "Status").lower(),
        "confidence": metadata_value(text, "Confidence").lower(),
        "OOD_scenario": metadata_value(text, "OOD_scenario"),
        "task_family": metadata_value(text, "Task family"),
        "subtask_type": metadata_value(text, "Subtask type"),
        "recommendation": section(text, "Recommendation"),
        "text": text,
    }


def candidate_lessons(lesson_dir: Path) -> list[dict[str, Any]]:
    if not lesson_dir.exists():
        return []
    lessons = []
    for path in sorted(lesson_dir.glob("*.md")):
        payload = lesson_payload(path)
        if payload["status"] == "candidate":
            lessons.append(payload)
    return lessons


def payload_matches_lesson(payload: dict[str, Any], lesson: dict[str, Any]) -> bool:
    scenario = normalize_ood_scenario(payload.get("OOD_scenario") or payload.get("signal_name", ""))
    if normalize_ood_scenario(lesson.get("OOD_scenario", "")) not in {"none", scenario}:
        return False
    semantic_tags = payload.get("semantic_tags", {})
    if not isinstance(semantic_tags, dict):
        semantic_tags = {}
    task_family = str(semantic_tags.get("task_family", payload.get("task_family", ""))).strip()
    subtask_type = str(semantic_tags.get("subtask_type", payload.get("subtask_type", ""))).strip()
    if lesson.get("task_family") and lesson["task_family"] != task_family:
        return False
    if lesson.get("subtask_type") and lesson["subtask_type"] != subtask_type:
        return False
    return True


def expected_from_lesson(lesson: dict[str, Any]) -> dict[str, str]:
    text = str(lesson.get("recommendation", ""))
    workflow_match = re.search(r"workflow\s+`([^`]+)`", text)
    intent_match = re.search(r"intent\s+`([^`]+)`", text)
    return {
        "recovery_workflow": workflow_match.group(1).strip() if workflow_match else "",
        "post_recovery_intent": intent_match.group(1).strip() if intent_match else "",
    }


def local_eval(payload: dict[str, Any], lesson: dict[str, Any]) -> dict[str, Any]:
    expected = expected_from_lesson(lesson)
    baseline = payload.get("expected", {})
    if not isinstance(baseline, dict):
        baseline = {}
    workflow_ok = not baseline.get("recovery_workflow") or expected["recovery_workflow"] == str(baseline.get("recovery_workflow", "")).strip()
    intent_ok = not baseline.get("post_recovery_intent") or expected["post_recovery_intent"] == str(baseline.get("post_recovery_intent", "")).strip()
    return {
        "mode": "local",
        "prediction": expected,
        "expected": baseline,
        "passed": bool(workflow_ok and intent_ok and expected["recovery_workflow"]),
        "invalid_output": False,
    }


def live_eval(payload: dict[str, Any], lesson: dict[str, Any], backend_config: dict[str, Any]) -> dict[str, Any]:
    backend = build_recovery_backend(backend_config)
    recovery_payload = dict(payload)
    retrieved_experience = dict(recovery_payload.get("retrieved_experience", {}) or {})
    lessons = list(retrieved_experience.get("lessons", []) or [])
    lessons.insert(
        0,
        {
            "lesson_id": lesson["lesson_id"],
            "status": "candidate",
            "text": lesson["text"],
            "score": 999,
            "path": lesson["path"],
        },
    )
    retrieved_experience["lessons"] = lessons
    recovery_payload["retrieved_experience"] = retrieved_experience
    prediction = backend.plan_recovery(recovery_payload=recovery_payload)
    expected = payload.get("expected", {})
    if not isinstance(expected, dict):
        expected = {}
    workflow = str(prediction.get("recovery_workflow", "")).strip()
    intent = str(prediction.get("post_recovery_intent", "")).strip()
    workflow_ok = not expected.get("recovery_workflow") or workflow == str(expected.get("recovery_workflow", "")).strip()
    intent_ok = not expected.get("post_recovery_intent") or intent == str(expected.get("post_recovery_intent", "")).strip()
    invalid_output = not workflow or intent not in {"retry", "replan", "abort"}
    return {
        "mode": "live",
        "prediction": prediction,
        "expected": expected,
        "passed": bool(workflow_ok and intent_ok and not invalid_output),
        "invalid_output": invalid_output,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Eval output already exists: {args.output}. Use --overwrite.")
    lessons = candidate_lessons(args.lesson_dir)
    payloads = read_jsonl(args.payloads)
    backend_config = None
    if args.backend_config is not None:
        backend_config = json.loads(args.backend_config.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    evaluated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for lesson in lessons:
        matched_payloads = [payload for payload in payloads if payload_matches_lesson(payload, lesson)]
        passed = 0
        invalid = 0
        details: list[dict[str, Any]] = []
        for payload in matched_payloads:
            result = live_eval(payload, lesson, backend_config) if backend_config else local_eval(payload, lesson)
            passed += int(bool(result["passed"]))
            invalid += int(bool(result["invalid_output"]))
            details.append(
                {
                    "payload_id": str(payload.get("payload_id", payload.get("case_id", ""))),
                    "passed": result["passed"],
                    "invalid_output": result["invalid_output"],
                    "prediction": result["prediction"],
                    "expected": result["expected"],
                }
            )
        total = len(matched_payloads)
        pass_rate = float(passed / total) if total else 0.0
        results.append(
            {
                "lesson_id": lesson["lesson_id"],
                "lesson_path": lesson["path"],
                "evaluated_at": evaluated_at,
                "payloads_matched": total,
                "passed": passed,
                "invalid_outputs": invalid,
                "pass_rate": pass_rate,
                "promotion_ready": bool(total > 0 and pass_rate >= 0.8 and invalid == 0),
                "details": details,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"lessons": len(lessons), "payloads": len(payloads), "results": len(results), "output": str(args.output)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

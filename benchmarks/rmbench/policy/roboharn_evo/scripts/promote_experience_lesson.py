#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote or deprecate RoboHarn-Evo experience lessons based on evaluation results.")
    parser.add_argument("--lesson-dir", type=Path, default=Path("policy/roboharn_evo/skills/experience/learned-lessons"))
    parser.add_argument("--eval-results", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/lesson_eval_results.jsonl"))
    parser.add_argument("--lesson-id", default="", help="Optional single lesson id/path stem to update.")
    parser.add_argument("--deprecate", action="store_true", help="Deprecate matching lesson(s) instead of promoting.")
    parser.add_argument("--reason", default="")
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    parser.add_argument("--min-payloads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
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


def replace_metadata(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*-\s*{re.escape(key)}:\s*)`?[^`\n]*`?\s*$", re.IGNORECASE | re.MULTILINE)
    replacement = rf"\1`{value}`"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    marker = "## Metadata\n\n"
    if marker in text:
        return text.replace(marker, marker + f"- {key}: `{value}`\n", 1)
    return f"- {key}: `{value}`\n" + text


def append_deprecation_reason(text: str, reason: str) -> str:
    if not reason:
        return text
    line = f"- Deprecation Reason: `{reason}`"
    if "Deprecation Reason:" in text:
        return re.sub(r"^\s*-\s*Deprecation Reason:.*$", line, text, count=1, flags=re.MULTILINE)
    marker = "## Metadata\n\n"
    if marker in text:
        return text.replace(marker, marker + line + "\n", 1)
    return line + "\n" + text


def eval_by_lesson_id(eval_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("lesson_id", "")).strip(): record for record in eval_results if str(record.get("lesson_id", "")).strip()}


def should_promote(record: dict[str, Any], *, min_pass_rate: float, min_payloads: int) -> bool:
    return (
        int(record.get("payloads_matched", 0) or 0) >= min_payloads
        and float(record.get("pass_rate", 0.0) or 0.0) >= min_pass_rate
        and int(record.get("invalid_outputs", 0) or 0) == 0
    )


def main() -> None:
    args = parse_args()
    eval_results = eval_by_lesson_id(read_jsonl(args.eval_results))
    updated: list[str] = []
    skipped: list[str] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for path in sorted(args.lesson_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lesson_id = metadata_value(text, "Lesson ID") or path.stem
        if args.lesson_id and args.lesson_id not in {lesson_id, path.stem, str(path)}:
            continue
        status = metadata_value(text, "Status").lower()
        if args.deprecate:
            new_text = replace_metadata(text, "Status", "deprecated")
            new_text = append_deprecation_reason(new_text, args.reason or "deprecated by promote_experience_lesson.py")
        else:
            result = eval_results.get(lesson_id)
            if status != "candidate" or result is None or not should_promote(result, min_pass_rate=args.min_pass_rate, min_payloads=args.min_payloads):
                skipped.append(str(path))
                continue
            new_text = replace_metadata(text, "Status", "accepted")
            new_text = replace_metadata(new_text, "Last Validated At", str(result.get("evaluated_at", now)))
            new_text = re.sub(
                r"(^## Evaluation Result\s*\n\n)(.*?)(?=^##\s+|\Z)",
                rf"\1- Status: `passed`\n- Pass Rate: `{float(result.get('pass_rate', 0.0)):.3f}`\n- Payloads Matched: `{int(result.get('payloads_matched', 0))}`\n\n",
                new_text,
                count=1,
                flags=re.MULTILINE | re.DOTALL,
            )
        updated.append(str(path))
        if not args.dry_run:
            path.write_text(new_text, encoding="utf-8")

    print(json.dumps({"updated": updated, "skipped": skipped, "dry_run": args.dry_run}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

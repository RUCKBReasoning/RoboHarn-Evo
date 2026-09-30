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

from policy.roboharn_evo.agent.experience import (
    ExperienceValidationError,
    validate_retrieval_index_record,
    validate_recovery_trial,
    validate_retrieval_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RoboHarn-Evo experience ledgers and retrieval payloads.")
    parser.add_argument("--recovery-trials", type=Path, default=Path("policy/roboharn_evo/skills/experience/raw-traces/recovery_trials.jsonl"))
    parser.add_argument("--retrieval-index", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/index.jsonl"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on validation errors.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return []
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append((line_no, payload))
    return records


def main() -> None:
    args = parse_args()
    errors: list[str] = []

    for line_no, record in read_jsonl(args.recovery_trials):
        for error in validate_recovery_trial(record):
            errors.append(f"{args.recovery_trials}:{line_no}: {error}")
        try:
            validate_retrieval_payload(record)
        except ExperienceValidationError as exc:
            errors.append(f"{args.recovery_trials}:{line_no}: {exc}")

    for line_no, record in read_jsonl(args.retrieval_index):
        for error in validate_retrieval_index_record(record):
            errors.append(f"{args.retrieval_index}:{line_no}: {error}")
        try:
            validate_retrieval_payload(record)
        except ExperienceValidationError as exc:
            errors.append(f"{args.retrieval_index}:{line_no}: {exc}")

    payload = {
        "recovery_trial_records": len(read_jsonl(args.recovery_trials)),
        "retrieval_index_records": len(read_jsonl(args.retrieval_index)),
        "errors": errors,
        "ok": not errors,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    if errors and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

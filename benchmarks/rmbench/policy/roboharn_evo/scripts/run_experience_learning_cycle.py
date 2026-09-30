#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline RoboHarn-Evo experience learning cycle.")
    parser.add_argument("--trace", type=Path, action="append", default=[], help="Trace JSONL to extract. Can be passed multiple times.")
    parser.add_argument("--experience-root", type=Path, default=Path("policy/roboharn_evo/skills/experience"))
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--promote", action="store_true", help="Promote candidate lessons that pass evaluation.")
    parser.add_argument("--use-fallback-tags", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True, cwd=repo_root, env=env)
    if completed.returncode != 0:
        error_payload = {
            "cmd": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(completed.returncode)
    stdout = completed.stdout.strip()
    payload: dict[str, object]
    try:
        payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except json.JSONDecodeError:
        payload = {"stdout": stdout}
    return {"cmd": cmd, "result": payload}


def main() -> None:
    args = parse_args()
    for trace in args.trace:
        if not trace.exists():
            raise FileNotFoundError(
                f"Trace file does not exist: {trace}. Replace the example path with a real agent trace JSONL."
            )
    root = args.experience_root
    recovery_trials = root / "raw-traces" / "recovery_trials.jsonl"
    ood_trials = root / "raw-traces" / "ood_trials.jsonl"
    reentry_trials = root / "raw-traces" / "reentry_trials.jsonl"
    case_dir = root / "case-summaries"
    lesson_dir = root / "learned-lessons"
    avoid_patterns = root / "retrieval-index" / "avoid_patterns.jsonl"
    regression_payloads = root / "retrieval-index" / "recovery_regression_payloads.jsonl"
    eval_results = root / "retrieval-index" / "lesson_eval_results.jsonl"
    index_path = root / "retrieval-index" / "index.jsonl"

    script_dir = Path(__file__).resolve().parent
    steps: list[dict[str, object]] = []

    for trace in args.trace:
        cmd = [
            sys.executable,
            str(script_dir / "extract_experience_from_trace.py"),
            "--trace",
            str(trace),
            "--recovery-output",
            str(recovery_trials),
            "--ood-output",
            str(ood_trials),
            "--reentry-output",
            str(reentry_trials),
            "--append",
        ]
        if args.use_fallback_tags:
            cmd.append("--use-fallback-tags")
        steps.append(run(cmd))

    steps.append(
        run(
            [
                sys.executable,
                str(script_dir / "summarize_experience_cases.py"),
                "--recovery-trials",
                str(recovery_trials),
                "--output-dir",
                str(case_dir),
                "--overwrite",
            ]
        )
    )
    steps.append(
        run(
            [
                sys.executable,
                str(script_dir / "mine_experience_lessons.py"),
                "--recovery-trials",
                str(recovery_trials),
                "--lesson-dir",
                str(lesson_dir),
                "--avoid-output",
                str(avoid_patterns),
                "--min-support",
                str(args.min_support),
                "--overwrite",
            ]
        )
    )
    steps.append(
        run(
            [
                sys.executable,
                str(script_dir / "build_recovery_regression_payloads.py"),
                "--recovery-trials",
                str(recovery_trials),
                "--output",
                str(regression_payloads),
                "--overwrite",
            ]
        )
    )
    steps.append(
        run(
            [
                sys.executable,
                str(script_dir / "eval_experience_lessons.py"),
                "--lesson-dir",
                str(lesson_dir),
                "--payloads",
                str(regression_payloads),
                "--output",
                str(eval_results),
                "--overwrite",
            ]
        )
    )
    if args.promote:
        steps.append(
            run(
                [
                    sys.executable,
                    str(script_dir / "promote_experience_lesson.py"),
                    "--lesson-dir",
                    str(lesson_dir),
                    "--eval-results",
                    str(eval_results),
                ]
            )
        )
    steps.append(
        run(
            [
                sys.executable,
                str(script_dir / "build_experience_index.py"),
                "--experience-root",
                str(root),
                "--recovery-trials",
                str(recovery_trials),
                "--avoid-patterns",
                str(avoid_patterns),
                "--output",
                str(index_path),
                "--overwrite",
            ]
        )
    )
    validate_cmd = [
        sys.executable,
        str(script_dir / "validate_experience_records.py"),
        "--recovery-trials",
        str(recovery_trials),
        "--retrieval-index",
        str(index_path),
    ]
    if args.strict:
        validate_cmd.append("--strict")
    steps.append(run(validate_cmd))
    print(json.dumps({"steps": steps}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

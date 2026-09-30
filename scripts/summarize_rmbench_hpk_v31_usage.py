#!/usr/bin/env python3
"""Summarize one isolated RMBench cell's Agent API usage and wall time."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


USAGE_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_usage"
_PREFIX = "[openai-planner-audit] "


class UsageSummaryError(RuntimeError):
    """The isolated service-log window cannot prove exact cell usage."""


def _records(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        start = line.find(_PREFIX)
        if start < 0:
            continue
        try:
            value = json.loads(line[start + len(_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise UsageSummaryError(
                f"{path}:{line_number} contains a malformed Agent API audit"
            ) from exc
        if not isinstance(value, dict):
            raise UsageSummaryError(
                f"{path}:{line_number} Agent API audit must be an object"
            )
        if value.get("event") == "agent_model_request_complete":
            result.append(value)
    return result


def _episode_wall_time(trace_path: Path) -> float:
    starts: list[Mapping[str, Any]] = []
    ends: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        trace_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageSummaryError(
                f"{trace_path}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise UsageSummaryError(
                f"{trace_path}:{line_number} must contain one object"
            )
        if value.get("event") == "episode_start":
            starts.append(value)
        elif value.get("event") == "episode_end":
            ends.append(value)
    if len(starts) != 1 or len(ends) != 1:
        raise UsageSummaryError("trace must contain exactly one episode start and end")
    start = starts[0].get("timestamp")
    end = ends[0].get("timestamp")
    if (
        not isinstance(start, (int, float))
        or isinstance(start, bool)
        or not isinstance(end, (int, float))
        or isinstance(end, bool)
        or end < start
    ):
        raise UsageSummaryError("episode timestamps are invalid")
    return float(end) - float(start)


def _token(usage: Mapping[str, Any], name: str) -> int | None:
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def summarize(
    audit_log: Path,
    trace_path: Path,
    *,
    expected_model: str,
    expected_reasoning_effort: str,
) -> dict[str, Any]:
    records = _records(audit_log.resolve(strict=True))
    expected_model = str(expected_model or "").strip()
    expected_effort = str(expected_reasoning_effort or "").strip()
    if not expected_model or not expected_effort:
        raise UsageSummaryError("expected model and reasoning effort must be explicit")
    identity_errors = [
        record
        for record in records
        if record.get("model") != expected_model
        or record.get("reasoning_effort", expected_effort) != expected_effort
    ]
    failed = [record for record in records if record.get("success") is not True]
    retry_counts: list[int] = []
    usages: list[Mapping[str, Any]] = []
    for record in records:
        retry = record.get("retry_count", 0)
        if isinstance(retry, bool) or not isinstance(retry, int) or retry < 0:
            raise UsageSummaryError("Agent API retry_count is invalid")
        retry_counts.append(retry)
        usage = record.get("usage")
        if isinstance(usage, Mapping):
            usages.append(usage)
    usage_complete = len(usages) == len(records) and all(
        _token(value, key) is not None
        for value in usages
        for key in ("input_tokens", "output_tokens", "total_tokens")
    )
    if identity_errors:
        raise UsageSummaryError("Agent API model identity differs within the cell")
    return {
        "schema": USAGE_SCHEMA,
        "agent_api_calls": len(records),
        "successful_calls": len(records) - len(failed),
        "failed_calls": len(failed),
        "input_tokens": sum(_token(value, "input_tokens") or 0 for value in usages),
        "output_tokens": sum(_token(value, "output_tokens") or 0 for value in usages),
        "total_tokens": sum(_token(value, "total_tokens") or 0 for value in usages),
        "automatic_retries": sum(retry_counts),
        "usage_complete": usage_complete,
        "wall_time_sec": _episode_wall_time(trace_path.resolve(strict=True)),
        "endpoint_counts": dict(
            sorted(
                Counter(str(record.get("endpoint", "")) for record in records).items()
            )
        ),
        "model": expected_model,
        "reasoning_effort": expected_effort,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-reasoning-effort", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise UsageSummaryError(f"refusing to overwrite {output}")
    value = summarize(
        args.audit_log,
        args.trace,
        expected_model=args.expected_model,
        expected_reasoning_effort=args.expected_reasoning_effort,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge immutable ESI source runs into one receipt-bound trace set."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    ESIKnowledgeError,
    ESITraceRecord,
)

_RECEIPT_STATUSES = {
    "completed",
    "attempt_cap_exhausted",
    "failed_before_completion",
}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ESIKnowledgeError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise ESIKnowledgeError(f"{label} must be one object")
    return dict(value)


def _read_trace(path: Path) -> ESITraceRecord:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise ESIKnowledgeError("source trace member must contain exactly one record")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ESIKnowledgeError("source trace member is invalid JSON") from exc
    return ESITraceRecord.from_mapping(value)


def merge_collections(
    roots: Sequence[Path],
    *,
    required_correct: int,
    attempt_cap: int,
) -> tuple[dict[str, Any], tuple[ESITraceRecord, ...], dict[str, Any]]:
    if not roots:
        raise ESIKnowledgeError("at least one source collection root is required")
    if (
        isinstance(required_correct, bool)
        or not isinstance(required_correct, int)
        or required_correct < 1
        or isinstance(attempt_cap, bool)
        or not isinstance(attempt_cap, int)
        or attempt_cap < required_correct
    ):
        raise ESIKnowledgeError("merge counts are invalid")
    attempts = []
    traces = []
    seen_refs = set()
    seen_scenes = set()
    for collection_index, raw_root in enumerate(roots, 1):
        root = raw_root.expanduser().resolve()
        receipt = _read_object(root / "collection_receipt.json", "receipt")
        if (
            receipt.get("schema") not in {"roboharn_evo/esi_bench/source_collection_receipt/v1", "tcm/esi_bench/source_collection_receipt/v1"}
            or receipt.get("status") not in _RECEIPT_STATUSES
            or receipt.get("automatic_retries") != 0
            or not isinstance(receipt.get("attempts"), list)
        ):
            raise ESIKnowledgeError("source collection receipt is invalid")
        for attempt in receipt["attempts"]:
            if not isinstance(attempt, Mapping):
                raise ESIKnowledgeError("source collection attempt is invalid")
            ref = str(attempt.get("instance_ref") or "").strip()
            scene = str(attempt.get("scene_group") or "").strip()
            position = attempt.get("queue_position")
            if (
                not ref
                or not scene
                or isinstance(position, bool)
                or not isinstance(position, int)
                or position < 1
            ):
                raise ESIKnowledgeError("source collection attempt identity is invalid")
            if ref in seen_refs or scene.casefold() in seen_scenes:
                raise ESIKnowledgeError(
                    "source collection attempts are not independent"
                )
            seen_refs.add(ref)
            seen_scenes.add(scene.casefold())
            merged_attempt = {
                **dict(attempt),
                "collection_index": collection_index,
                "collection_status": receipt["status"],
            }
            attempts.append(merged_attempt)
            eligible = (
                attempt.get("benchmark_correct") is True
                and attempt.get("rollout_exit_code") == 0
                and attempt.get("summary_exit_code") == 0
            )
            trace_path = (
                root / f"episode_{position:02d}" / "source_trace" / "source_trace.jsonl"
            )
            if eligible:
                if not trace_path.is_file():
                    raise ESIKnowledgeError("eligible source attempt has no trace")
                trace = _read_trace(trace_path)
                if trace.source_ref != ref or trace.benchmark_correct is not True:
                    raise ESIKnowledgeError("source trace identity/outcome mismatch")
                traces.append(trace)
            elif trace_path.exists():
                raise ESIKnowledgeError("ineligible source attempt has a trace")
    if len(attempts) > attempt_cap:
        raise ESIKnowledgeError("merged source attempts exceed the authorized cap")
    if len(traces) != required_correct:
        raise ESIKnowledgeError("merged source traces do not meet the support gate")
    receipt = {
        "schema": "roboharn_evo/esi_bench/source_collection_receipt/v1",
        "status": "completed",
        "required_correct": required_correct,
        "attempt_cap": attempt_cap,
        "attempts": attempts,
        "correct_trace_count": len(traces),
        "automatic_retries": 0,
    }
    audit = {
        "schema": "roboharn_evo/esi_bench/source_collection_merge_audit/v1",
        "collection_count": len(roots),
        "attempt_count": len(attempts),
        "eligible_trace_count": len(traces),
        "correct_but_rejected_materialization_count": sum(
            item.get("benchmark_correct") is True
            and item.get("summary_exit_code") not in {None, 0}
            for item in attempts
        ),
        "infrastructure_failure_count": sum(
            item.get("rollout_exit_code") != 0 for item in attempts
        ),
        "automatic_retries": 0,
    }
    return receipt, tuple(traces), audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge preregistered ESI source collection receipts and traces."
    )
    parser.add_argument("--collection-root", type=Path, action="append", required=True)
    parser.add_argument("--required-correct", type=int, required=True)
    parser.add_argument("--attempt-cap", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"merged source output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    receipt, traces, audit = merge_collections(
        args.collection_root,
        required_correct=args.required_correct,
        attempt_cap=args.attempt_cap,
    )
    (output / "collection_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source_traces.jsonl").write_text(
        "".join(
            json.dumps(trace.to_dict(), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for trace in traces
        ),
        encoding="utf-8",
    )
    (output / "merge_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

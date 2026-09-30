#!/usr/bin/env python3
"""Recompute frozen ESI probe metrics from completed public artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_esi_frozen_probe import (  # noqa: E402
    _complete_call_audit,
    modes_for_profile,
    probe_call_budget,
    summarize_probe,
)


def _read_jsonl(path: Path) -> tuple[dict, ...]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("decision line must be one object")
            values.append(dict(value))
    return tuple(values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a completed frozen probe.")
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--probe-receipt", type=Path, required=True)
    parser.add_argument("--retrievals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _complete_receipts(
    receipt: Mapping[str, object],
    retrievals: tuple[dict, ...],
    decisions: tuple[dict, ...],
    *,
    state_count: int,
) -> bool:
    profile = str(receipt.get("probe_profile") or "five_condition")
    try:
        modes = modes_for_profile(profile)
    except Exception:
        return False
    budget = probe_call_budget(state_count, modes=modes)
    retrieval_scopes = {}
    for item in retrievals:
        state_index = item.get("state_index")
        scope = item.get("scope")
        if not isinstance(state_index, int) or not isinstance(scope, str):
            return False
        retrieval_scopes.setdefault(state_index, set()).add(scope)
        if not _complete_call_audit(item.get("provider_call_audit")):
            return False
    expected_scopes = {"flat", "task_core_full_shared"}
    if "shuffled_full" in modes:
        expected_scopes.add("shuffled_full")
    return bool(
        receipt.get("schema") in {"roboharn_evo/esi_bench/frozen_probe_receipt/v2", "tcm/esi_bench/frozen_probe_receipt/v2"}
        and receipt.get("status") == "completed"
        and receipt.get("annotation_calls") == budget["annotation_calls"]
        and receipt.get("evaluated_model_calls") == budget["evaluated_model_calls"]
        and receipt.get("retrieval_model_calls") == budget["retrieval_model_calls"]
        and receipt.get("image_payloads") == budget["total_image_payloads"]
        and receipt.get("automatic_retries") == 0
        and receipt.get("simulator_episodes") == 0
        and receipt.get("executed_actions") == 0
        and len(decisions) == state_count * len(modes)
        and all(
            _complete_call_audit(item.get("evaluated_call_audit")) for item in decisions
        )
        and len(retrievals) == state_count * len(expected_scopes)
        and set(retrieval_scopes) == set(range(1, state_count + 1))
        and all(scopes == expected_scopes for scopes in retrieval_scopes.values())
    )


def main() -> int:
    args = _parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(annotations, Mapping) or not isinstance(
        annotations.get("states"), list
    ):
        raise ValueError("annotation artifact is invalid")
    decisions = _read_jsonl(args.decisions)
    retrievals = _read_jsonl(args.retrievals)
    receipt = json.loads(args.probe_receipt.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("probe receipt is invalid")
    summary = summarize_probe(
        decisions,
        annotations["states"],
        receipts_complete=_complete_receipts(
            receipt,
            retrievals,
            decisions,
            state_count=len(annotations["states"]),
        ),
        modes=modes_for_profile(str(receipt.get("probe_profile") or "five_condition")),
    )
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_event_name


class RestorePairEvaluationError(RuntimeError):
    """The pair is incomplete or cannot support an unambiguous comparison."""


def _one(root: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise RestorePairEvaluationError(
            f"{label} must contain exactly one {pattern}; found {len(matches)}"
        )
    return matches[0]


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RestorePairEvaluationError(f"{path} must contain one JSON object")
    return value


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RestorePairEvaluationError(
                f"{path}:{line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise RestorePairEvaluationError(
                f"{path}:{line_number} must contain one JSON object"
            )
        record = dict(value)
        if isinstance(record.get("event"), str):
            record["event"] = normalize_hpk_event_name(record["event"])
        records.append(record)
    if not records:
        raise RestorePairEvaluationError(f"{path} is empty")
    return records


def _event(records: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    return [record for record in records if record.get("event") == name]


def _restore_is_exact(summary: Mapping[str, Any]) -> bool:
    boundary = summary.get("boundary")
    if not isinstance(boundary, Mapping):
        return False
    match = boundary.get("state_match")
    correction = match.get("physics_correction") if isinstance(match, Mapping) else None
    return bool(
        isinstance(match, Mapping)
        and match.get("matches") is True
        and match.get("agent_state_matches") is True
        and match.get("mismatches") == []
        and isinstance(correction, Mapping)
        and isinstance(
            correction.get("post_correction_max_numeric_error"), (int, float)
        )
        and not isinstance(correction.get("post_correction_max_numeric_error"), bool)
        and float(correction["post_correction_max_numeric_error"]) <= 1e-6
    )


def _first_selected_candidate(
    records: Sequence[Mapping[str, Any]], boundary_step: int
) -> Mapping[str, Any] | None:
    for record in records:
        if record.get("event") != "operation_candidate_selected":
            continue
        step = record.get("env_step")
        if isinstance(step, int) and step >= boundary_step:
            return {
                "operation": record.get("operation"),
                "arm": record.get("arm"),
                "source_candidate_index": record.get("source_candidate_index"),
                "geometry_source": record.get("geometry_source"),
            }
    return None


def _proposal_chain(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposals = _event(records, "hpk_v2_strategy_proposal")
    scheduled = _event(records, "hpk_v2_proposal_scheduled")
    resolutions = _event(records, "hpk_v2_proposal_resolution")
    proposal_ids = {
        str(proposal["proposal"]["proposal_id"])
        for proposal in proposals
        if proposal.get("scheduled") is True
        and isinstance(proposal.get("proposal"), Mapping)
        and proposal["proposal"].get("proposal_id")
    }
    scheduled_ids = {
        str(record["proposal_id"])
        for record in scheduled
        if record.get("selected") is True and record.get("proposal_id")
    }
    resolved = [
        record
        for record in resolutions
        if record.get("proposal_id") in proposal_ids & scheduled_ids
    ]
    deterministic = [
        record
        for record in resolved
        if record.get("motion_status") == "completed"
        and record.get("realization_status") == "satisfied"
        and record.get("verdict") in {"support", "oppose"}
    ]
    return {
        "backend_proposal_count": len(proposals),
        "scheduled_for_execution_count": len(scheduled),
        "resolved_count": len(resolved),
        "deterministic_physical_verdict_count": len(deterministic),
        "physical_gate_passed": bool(deterministic),
        "verdicts": [record.get("verdict") for record in resolved],
    }


def _condition(root: Path, *, expect_hpk: bool) -> dict[str, Any]:
    trace_path = _one(root, "episode_*_agent_trace.jsonl", label=str(root))
    result_path = _one(root, "failure_boundary_result.json", label=str(root))
    records = _jsonl(trace_path)
    result = _json(result_path)
    if result.get("schema") not in {
        "roboharn_evo/rmbench_failure_boundary_result/v1",
        "tcm/rmbench_failure_boundary_result/v1",
    }:
        raise RestorePairEvaluationError("failure-boundary result schema mismatch")
    boundary = result.get("boundary")
    outcome = result.get("outcome")
    if not isinstance(boundary, Mapping) or not isinstance(outcome, Mapping):
        raise RestorePairEvaluationError("failure-boundary result is incomplete")
    starts = _event(records, "episode_start")
    ends = _event(records, "episode_end")
    if len(starts) != 1 or len(ends) != 1:
        raise RestorePairEvaluationError(
            "condition must have one episode start and end"
        )
    hpk_counts = Counter(
        str(record.get("event"))
        for record in records
        if str(record.get("event", "")).startswith("hpk_")
    )
    if not expect_hpk and hpk_counts:
        raise RestorePairEvaluationError("HPK-off trace contains HPK events")
    proposal = _proposal_chain(records)
    finalizations = _event(records, "hpk_episode_finalization")
    advanced = any(record.get("advanced") is True for record in finalizations)
    return {
        "trace_path": str(trace_path),
        "result_path": str(result_path),
        "seed": result.get("seed"),
        "instruction": result.get("instruction"),
        "boundary_path": boundary.get("path"),
        "boundary_env_step": boundary.get("env_step"),
        "exact_state_restored": _restore_is_exact(result),
        "outcome": dict(outcome),
        "first_selected_candidate": _first_selected_candidate(
            records, int(boundary.get("env_step", 0))
        ),
        "control_turn_count": sum(
            record.get("event") == "control_turn_start" for record in records
        ),
        "hpk_event_counts": dict(sorted(hpk_counts.items())),
        "proposal_chain": proposal,
        "store_advanced": advanced,
    }


def evaluate(
    boundary_path: Path, off_root: Path, evolving_root: Path
) -> dict[str, Any]:
    boundary = _json(boundary_path)
    off = _condition(off_root, expect_hpk=False)
    evolving = _condition(evolving_root, expect_hpk=True)
    identity_matches = bool(
        off["seed"] == evolving["seed"] == boundary.get("seed")
        and off["instruction"] == evolving["instruction"] == boundary.get("instruction")
        and Path(str(off["boundary_path"])).resolve() == boundary_path
        and Path(str(evolving["boundary_path"])).resolve() == boundary_path
    )
    candidate_changed = (
        off["first_selected_candidate"] is not None
        and evolving["first_selected_candidate"] is not None
        and off["first_selected_candidate"] != evolving["first_selected_candidate"]
    )
    gate = bool(
        identity_matches
        and off["exact_state_restored"]
        and evolving["exact_state_restored"]
        and evolving["proposal_chain"]["physical_gate_passed"]
    )
    return {
        "schema": "roboharn_evo/hpk/rmbench_restore_pair_report/v1",
        "boundary": {
            "path": str(boundary_path),
            "task": boundary.get("task"),
            "seed": boundary.get("seed"),
            "env_step": boundary.get("boundary", {}).get("env_step"),
        },
        "pair_identity_matches": identity_matches,
        "off": off,
        "evolving": evolving,
        "comparison": {
            "first_candidate_changed": candidate_changed,
            "additional_environment_actions_delta": (
                int(evolving["outcome"]["additional_environment_actions"])
                - int(off["outcome"]["additional_environment_actions"])
            ),
            "success_changed": bool(evolving["outcome"]["success"])
            != bool(off["outcome"]["success"]),
        },
        "e05_proposal_execution_physical_verdict_gate_passed": gate,
        "scientific_boundary": (
            "This paired restore is an integration smoke from one captured state; "
            "it is not a benchmark success-rate estimate."
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HPK-off and evolving recoveries from one RMBench failure boundary."
    )
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--off-root", type=Path, required=True)
    parser.add_argument("--evolving-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise RestorePairEvaluationError(f"refusing to overwrite {output}")
    report = evaluate(
        args.boundary.resolve(strict=True),
        args.off_root.resolve(strict=True),
        args.evolving_root.resolve(strict=True),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

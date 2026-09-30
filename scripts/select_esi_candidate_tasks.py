#!/usr/bin/env python3
"""Select primary/reserve ESI source tasks with one frozen semantic judgment."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.agent.hpk.hierarchical_retriever import (  # noqa: E402
    AgentApiHierarchicalRetrievalBackend,
)
from roboharn_evo.benchmark_adapters.esi_bench import ESIKnowledgeError  # noqa: E402
from roboharn_evo.benchmark_adapters.esi_bench.candidate_selection import (  # noqa: E402
    ESI_SOURCE_TASK_SELECTION_SCHEMA_NAME,
    candidate_selection_json_schema,
)

_INSTRUCTIONS = """Select ESI-Bench source tasks for a Task Knowledge development experiment.

Use only the public split counts, representative public question, official prompt text, and declared public action vocabulary in the input. Do not assume a preferred task name. Prefer tasks where successful trajectories could reveal a reusable evidence-acquisition subtask, applicability condition, completion condition, ordering branch, or stop criterion that the official prompt does not already prescribe. Reject tasks whose plausible lesson would merely repeat the official prompt. Require enough source and development instances for at least three independent source supports and a later matched mechanism probe. This selection is an experiment-design hypothesis, not Task Knowledge and not a benchmark result. Return strict JSON only."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose primary/reserve ESI tasks from a public audit packet."
    )
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-source-instances", type=int, default=4)
    parser.add_argument("--min-development-instances", type=int, default=12)
    parser.add_argument(
        "--include-small-task",
        action="append",
        default=[],
        help=(
            "Repeat to restrict the semantic judge to a preregistered public "
            "small-task set. No task names are built into this script."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request-only", action="store_true")
    group.add_argument("--planner-url")
    group.add_argument("--judge-response", type=Path)
    return parser.parse_args()


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ESIKnowledgeError(
                f"candidate inventory line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise ESIKnowledgeError("candidate inventory entries must be objects")
        records.append(dict(value))
    if not records:
        raise ESIKnowledgeError("candidate inventory is empty")
    tasks = [str(item.get("small_task") or "").strip() for item in records]
    if any(not item for item in tasks) or len(tasks) != len(set(tasks)):
        raise ESIKnowledgeError("candidate inventory task names must be unique")
    return tuple(records)


def build_selection_input(records: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        {"candidate_tasks": [dict(item) for item in records]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def select_requested_records(
    records: Sequence[Mapping[str, Any]],
    requested_tasks: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Apply an explicit public-task scope without encoding benchmark rules."""

    normalized = tuple(
        " ".join(str(value or "").strip().casefold().split())
        for value in requested_tasks
    )
    if not normalized:
        return tuple(dict(item) for item in records)
    if any(not value for value in normalized) or len(normalized) != len(
        set(normalized)
    ):
        raise ESIKnowledgeError("included small tasks must be non-empty and distinct")
    by_task = {
        " ".join(str(item.get("small_task") or "").strip().casefold().split()): dict(
            item
        )
        for item in records
    }
    missing = sorted(set(normalized) - set(by_task))
    if missing:
        raise ESIKnowledgeError(
            f"included small tasks are absent from audit: {missing}"
        )
    return tuple(by_task[value] for value in normalized)


def filter_eligible_records(
    records: Sequence[Mapping[str, Any]],
    *,
    min_source_instances: int,
    min_development_instances: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    for label, value in (
        ("min_source_instances", min_source_instances),
        ("min_development_instances", min_development_instances),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ESIKnowledgeError(f"{label} must be a positive integer")
    eligible = []
    excluded = []
    for item in records:
        reasons = []
        if int(item["source_instance_count"]) < min_source_instances:
            reasons.append("insufficient source instances")
        if int(item["development_instance_count"]) < min_development_instances:
            reasons.append("insufficient development instances")
        if reasons:
            excluded.append({"small_task": item["small_task"], "reasons": reasons})
        else:
            eligible.append(dict(item))
    if len(eligible) < 2:
        raise ESIKnowledgeError("fewer than two tasks pass the sample-count gate")
    return tuple(eligible), tuple(excluded)


def _response_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ESIKnowledgeError("candidate judge returned invalid JSON") from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ESIKnowledgeError("candidate judge must return one JSON object")


def validate_selection(
    value: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = set(candidate_selection_json_schema()["properties"])
    if set(value) != expected:
        raise ESIKnowledgeError("candidate selection response fields mismatch")
    selections = value.get("selections")
    controls = value.get("prompt_covered_controls")
    summary = " ".join(str(value.get("selection_summary") or "").strip().split())
    if not isinstance(selections, list) or len(selections) != 2:
        raise ESIKnowledgeError("candidate selection must contain two selections")
    if not isinstance(controls, list) or any(
        not isinstance(item, str) for item in controls
    ):
        raise ESIKnowledgeError("prompt-covered controls must be an array of strings")
    if not summary:
        raise ESIKnowledgeError("candidate selection summary must be non-empty")
    by_task = {str(item["small_task"]): item for item in records}
    roles = []
    selected_tasks = []
    normalized = []
    required_fields = set(
        candidate_selection_json_schema()["properties"]["selections"]["items"][
            "properties"
        ]
    )
    for item in selections:
        if not isinstance(item, Mapping) or set(item) != required_fields:
            raise ESIKnowledgeError("candidate selection item fields mismatch")
        role = str(item["role"])
        task = str(item["small_task"])
        coverage = str(item["official_prompt_coverage"])
        if role not in {"primary", "reserve"}:
            raise ESIKnowledgeError("candidate role is invalid")
        if task not in by_task:
            raise ESIKnowledgeError("candidate judge selected an unknown task")
        if coverage == "fully prescribed":
            raise ESIKnowledgeError("selected task is fully prescribed by its prompt")
        if int(by_task[task]["source_instance_count"]) < 3:
            raise ESIKnowledgeError("selected task lacks three source instances")
        if int(by_task[task]["development_instance_count"]) < 1:
            raise ESIKnowledgeError("selected task lacks a development instance")
        roles.append(role)
        selected_tasks.append(task)
        normalized.append({key: item[key] for key in required_fields})
    if set(roles) != {"primary", "reserve"} or len(set(selected_tasks)) != 2:
        raise ESIKnowledgeError("candidate roles/tasks must be distinct")
    known_controls = sorted(set(controls) & set(by_task))
    unknown_controls = sorted(set(controls) - set(by_task))
    return {
        "selections": sorted(normalized, key=lambda item: item["role"]),
        "prompt_covered_controls": known_controls,
        "unmapped_prompt_covered_controls": unknown_controls,
        "selection_summary": summary,
    }


def _write_request(output: Path, *, input_text: str) -> None:
    schema = candidate_selection_json_schema()
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    (output / "judge_instructions.txt").write_text(
        _INSTRUCTIONS + "\n", encoding="utf-8"
    )
    (output / "judge_input.json").write_text(input_text + "\n", encoding="utf-8")
    (output / "judge_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    combined = _INSTRUCTIONS + input_text + schema_text
    (output / "call_budget.json").write_text(
        json.dumps(
            {
                "external_model_calls": 1,
                "images": 0,
                "input_bytes": len(combined.encode("utf-8")),
                "deterministic_whitespace_token_estimate": len(
                    re.findall(r"\S+", combined)
                ),
                "automatic_retries": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"candidate selection output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = select_requested_records(
        _read_jsonl(args.candidate_inventory.expanduser().resolve()),
        args.include_small_task,
    )
    eligible, excluded = filter_eligible_records(
        records,
        min_source_instances=args.min_source_instances,
        min_development_instances=args.min_development_instances,
    )
    (output / "eligibility.json").write_text(
        json.dumps(
            {
                "min_source_instances": args.min_source_instances,
                "min_development_instances": args.min_development_instances,
                "eligible_task_count": len(eligible),
                "excluded": list(excluded),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    input_text = build_selection_input(eligible)
    _write_request(output, input_text=input_text)
    if args.request_only:
        print((output / "call_budget.json").read_text(encoding="utf-8"))
        return 0
    if args.judge_response is not None:
        raw_response: Any = json.loads(
            args.judge_response.expanduser().resolve().read_text(encoding="utf-8")
        )
    else:
        completion = AgentApiHierarchicalRetrievalBackend(args.planner_url).complete(
            instructions=_INSTRUCTIONS,
            input_text=input_text,
            images=(),
            output_schema=candidate_selection_json_schema(),
            schema_name=ESI_SOURCE_TASK_SELECTION_SCHEMA_NAME,
        )
        raw_response = completion.output
    (output / "raw_judge_response.json").write_text(
        json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected = validate_selection(_response_mapping(raw_response), eligible)
    (output / "selection.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

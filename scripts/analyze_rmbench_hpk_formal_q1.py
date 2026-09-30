#!/usr/bin/env python3
"""Generate Q1 per-episode, table, paired-test, and LaTeX artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboharn_evo.benchmark_adapters.rmbench.formal_analysis import (  # noqa: E402
    METHOD_ORDER,
    TASK_ORDER,
    FormalQ1AnalysisError,
    aggregate_cells,
    collect_cells,
    exact_mcnemar,
    hierarchical_bootstrap_delta,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _paired(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(cell["task"]), str(cell["method"]), int(cell["evaluation_seed"])): cell
        for cell in cells
        if cell.get("denominator_eligible") is True
    }
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER[1:]:
        bootstrap_pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        method_pairs: list[tuple[bool, bool]] = []
        for task in TASK_ORDER:
            task_pairs: list[tuple[bool, bool]] = []
            seeds = sorted(
                seed
                for observed_task, observed_method, seed in by_key
                if observed_task == task and observed_method == "off"
            )
            for seed in seeds:
                baseline = by_key.get((task, "off", seed))
                treatment = by_key.get((task, method, seed))
                if baseline is None or treatment is None:
                    continue
                task_pairs.append(
                    (baseline.get("success") is True, treatment.get("success") is True)
                )
            method_pairs.extend(task_pairs)
            bootstrap_pairs[task].extend(task_pairs)
            test = exact_mcnemar(
                [value[0] for value in task_pairs],
                [value[1] for value in task_pairs],
            )
            rows.append({"scope": task, "method": method, **test})
        rows.append(
            {
                "scope": "all_tasks_micro",
                "method": method,
                **exact_mcnemar(
                    [value[0] for value in method_pairs],
                    [value[1] for value in method_pairs],
                ),
                "hierarchical_bootstrap": hierarchical_bootstrap_delta(bootstrap_pairs),
            }
        )
    return rows


def _latex_main(aggregates: Sequence[Mapping[str, Any]]) -> str:
    lookup = {(str(row["task"]), str(row["method"])): row for row in aggregates}
    header = "Method & " + " & ".join(task.replace("_", r"\_") for task in TASK_ORDER)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", header + r" \\", r"\midrule"]
    for method in METHOD_ORDER:
        values = []
        for task in TASK_ORDER:
            value = lookup[(task, method)]["success_rate"]
            values.append("--" if value is None else f"{100.0 * float(value):.1f}")
        lines.append(method.replace("_", r"\_") + " & " + " & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _latex_appendix(cells: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Task & Method & $n$ & Actions & API calls & Task adopted & Action adopted & Effect support \\",
        r"\midrule",
    ]
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            group = [
                cell
                for cell in cells
                if cell.get("task") == task
                and cell.get("method") == method
                and cell.get("denominator_eligible") is True
            ]

            def total(field: str) -> int:
                return sum(int(cell.get(field, 0) or 0) for cell in group)

            lines.append(
                " & ".join(
                    (
                        task.replace("_", r"\_"),
                        method,
                        str(len(group)),
                        str(total("environment_actions")),
                        str(total("agent_api_calls")),
                        str(total("task_knowledge_adopted")),
                        str(total("action_knowledge_adopted")),
                        str(total("action_evidence_support")),
                    )
                )
                + r" \\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def write_outputs(
    *,
    experiment_root: Path,
    matrix_path: Path,
    output_root: Path,
    allow_incomplete: bool,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FormalQ1AnalysisError(f"refusing to overwrite {output_root}")
    cells = collect_cells(experiment_root, matrix_path)
    missing = [cell["cell_id"] for cell in cells if cell.get("status") == "missing"]
    if missing and not allow_incomplete:
        raise FormalQ1AnalysisError(
            f"formal Q1 is incomplete: {len(missing)} of 300 cells are missing"
        )
    output_root.mkdir(parents=True)
    with (output_root / "per_episode.jsonl").open("x", encoding="utf-8") as stream:
        for cell in cells:
            stream.write(json.dumps(cell, ensure_ascii=False, sort_keys=True) + "\n")
    aggregates = aggregate_cells(cells)
    with (output_root / "rmbench_q1_main.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    paired = _paired(cells)
    result = {
        "schema": "roboharn_evo/rmbench/hpk_formal_q1_results/v1",
        "planned_cells": 300,
        "observed_cells": 300 - len(missing),
        "missing_cells": missing,
        "denominator_eligible": sum(
            cell.get("denominator_eligible") is True for cell in cells
        ),
        "infrastructure_invalid": sum(
            cell.get("status") == "infrastructure_invalid" for cell in cells
        ),
        "task_method": aggregates,
        "paired_tests": paired,
    }
    _write_json(output_root / "rmbench_q1_main.json", result)
    (output_root / "rmbench_q1_table_ready.tex").write_text(
        _latex_main(aggregates), encoding="utf-8"
    )
    (output_root / "rmbench_q1_appendix_tables.tex").write_text(
        _latex_appendix(cells), encoding="utf-8"
    )
    valid = [cell for cell in cells if cell.get("denominator_eligible") is True]
    summaries = [
        "# RMBench HPK Q1 Result Summary",
        "",
        "- Planned cells: 300",
        f"- Observed cells: {300 - len(missing)}",
        f"- Denominator-eligible cells: {len(valid)}",
        f"- Infrastructure-invalid cells: {result['infrastructure_invalid']}",
        f"- Exact successes: {sum(cell.get('success') is True for cell in valid)}",
        "",
    ]
    for method in METHOD_ORDER:
        group = [cell for cell in valid if cell.get("method") == method]
        rate = mean(cell.get("success") is True for cell in group) if group else None
        summaries.append(
            f"- {method}: " + ("N/A" if rate is None else f"{100.0 * rate:.1f}%")
        )
    if missing:
        summaries.extend(
            (
                "",
                "This is an incomplete operational snapshot, not the final paper table.",
            )
        )
    (output_root / "result_summary.md").write_text(
        "\n".join(summaries) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_outputs(
        experiment_root=args.experiment_root,
        matrix_path=args.matrix,
        output_root=args.output_root,
        allow_incomplete=args.allow_incomplete,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "planned_cells",
                    "observed_cells",
                    "denominator_eligible",
                    "infrastructure_invalid",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".release_work" / "matplotlib"),
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"report must be an object: {path}")
    return dict(value)


def _metrics(report: Mapping[str, Any], scale: str, method: str) -> Mapping[str, Any]:
    value = report["scales"][scale]["metrics"][method]
    if not isinstance(value, Mapping):
        raise TypeError(f"metrics are invalid for scale={scale}, method={method}")
    return value


def _method_cases(
    report: Mapping[str, Any],
    scale: str,
    method: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = report["scales"][scale].get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"retrieval rows are missing for scale={scale}")
    selected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("method") != method:
            continue
        key = (str(row.get("case_name")), str(row.get("kind")))
        if key in selected:
            raise ValueError(f"duplicate retrieval case for scale={scale}: {key}")
        selected[key] = row
    if not selected:
        raise ValueError(f"no {method} rows are available for scale={scale}")
    return selected


def _validate_case_alignment(
    exhaustive_report: Mapping[str, Any],
    optimized_report: Mapping[str, Any],
    scale: str,
) -> None:
    exhaustive = _method_cases(exhaustive_report, scale, "exhaustive")
    family = _method_cases(optimized_report, scale, "family-routed")
    if set(exhaustive) != set(family):
        raise ValueError(f"retrieval case sets differ for scale={scale}")
    for key in exhaustive:
        for field in ("query", "expected_atomic_index"):
            if exhaustive[key].get(field) != family[key].get(field):
                raise ValueError(
                    f"retrieval case field differs for scale={scale}, case={key}, "
                    f"field={field}"
                )


def _maintenance_atomic_inputs(report: Mapping[str, Any], *, incremental: bool) -> int:
    if incremental:
        return int(report["maintenance"]["consolidation_atomic_input_count"])
    before = report["counts_before"]
    added = report["new_counts"]
    return sum(int(before[key]) + int(added[key]) for key in before)


def _model_calls(report: Mapping[str, Any]) -> int:
    usage = report.get("model_usage")
    if isinstance(usage, Mapping) and isinstance(usage.get("model_calls"), int):
        return int(usage["model_calls"])
    return len(report.get("model_call_schemas", []))


def build_efficiency_artifacts(
    *,
    exhaustive_report_path: Path,
    optimized_report_path: Path,
    incremental_report_path: Path,
    full_rebuild_report_path: Path,
    output_dir: Path,
    agreement_report_path: Path | None = None,
) -> dict[str, Any]:
    """Build machine-readable metrics plus PNG/PDF paper figures."""

    exhaustive_report = _load(exhaustive_report_path)
    optimized_report = _load(optimized_report_path)
    incremental_report = _load(incremental_report_path)
    full_rebuild_report = _load(full_rebuild_report_path)
    agreement_report = (
        None if agreement_report_path is None else _load(agreement_report_path)
    )
    exhaustive_representation = exhaustive_report.get(
        "atomic_prompt_representation",
        "undeclared",
    )
    family_representation = optimized_report.get(
        "atomic_prompt_representation",
        "undeclared",
    )
    fair_family_ablation = exhaustive_representation == family_representation
    scales = sorted(
        set(exhaustive_report["scales"]).intersection(optimized_report["scales"]),
        key=int,
    )
    if not scales:
        raise ValueError("retrieval reports do not share a scale")

    retrieval: list[dict[str, Any]] = []
    for scale in scales:
        _validate_case_alignment(exhaustive_report, optimized_report, scale)
        exhaustive = _metrics(exhaustive_report, scale, "exhaustive")
        family = _metrics(optimized_report, scale, "family-routed")
        exhaustive_tokens = exhaustive.get("retrieval_input_tokens")
        family_tokens = family.get("retrieval_input_tokens")
        if not isinstance(exhaustive_tokens, int) or not isinstance(family_tokens, int):
            raise ValueError("paper comparison requires provider input-token counts")
        retrieval.append(
            {
                "scale": int(scale),
                "exhaustive_input_tokens": exhaustive_tokens,
                "family_input_tokens": family_tokens,
                "input_token_reduction": 1.0 - family_tokens / exhaustive_tokens,
                "exhaustive_model_calls": int(exhaustive["retrieval_calls"]),
                "family_model_calls": int(family["retrieval_calls"]),
                "exhaustive_latency_ms": exhaustive["retrieval_latency_ms"],
                "family_latency_ms": family["retrieval_latency_ms"],
                "latency_reduction": 1.0
                - family["retrieval_latency_ms"] / exhaustive["retrieval_latency_ms"],
                "exhaustive_final_accuracy": exhaustive["final_selection_accuracy"],
                "family_final_accuracy": family["final_selection_accuracy"],
                "family_recall_at_2": family["family_recall_at_2"],
                "atomic_recall_at_8": family["atomic_recall_at_8"],
                "family_harmful_changes": family["harmful_behavior_change_count"],
            }
        )

    incremental_atomic = _maintenance_atomic_inputs(
        incremental_report,
        incremental=True,
    )
    full_atomic = _maintenance_atomic_inputs(full_rebuild_report, incremental=False)
    maintenance = {
        "full_rebuild_atomic_inputs": full_atomic,
        "incremental_atomic_inputs": incremental_atomic,
        "atomic_input_reduction": 1.0 - incremental_atomic / full_atomic,
        "full_rebuild_model_calls": _model_calls(full_rebuild_report),
        "incremental_model_calls": _model_calls(incremental_report),
        "incremental_evidence_conservation_error": incremental_report["maintenance"][
            "evidence_conservation_error"
        ],
        "full_rebuild_evidence_conservation_error": full_rebuild_report["evidence"][
            "evidence_conservation_error"
        ],
        "incremental_vs_full_rebuild_retrieval_agreement": (
            None
            if agreement_report is None
            else agreement_report.get("final_retrieval_agreement")
        ),
    }
    family_recalls = [
        row["family_recall_at_2"]
        for row in retrieval
        if row["family_recall_at_2"] is not None
    ]
    atomic_recalls = [
        row["atomic_recall_at_8"]
        for row in retrieval
        if row["atomic_recall_at_8"] is not None
    ]
    retrieval_gates = {
        "family_recall_at_2_at_least_095": (
            bool(family_recalls) and mean(family_recalls) >= 0.95
        ),
        "atomic_recall_at_8_at_least_090": (
            bool(atomic_recalls) and mean(atomic_recalls) >= 0.90
        ),
        "final_accuracy_drop_at_most_003": (
            mean(row["exhaustive_final_accuracy"] for row in retrieval)
            - mean(row["family_final_accuracy"] for row in retrieval)
            <= 0.03
        ),
        "input_token_reduction_at_300_at_least_060": (
            retrieval[-1]["scale"] == 300
            and retrieval[-1]["input_token_reduction"] >= 0.60
        ),
        "harmful_behavior_change_not_increased": all(
            row["family_harmful_changes"] == 0 for row in retrieval
        ),
    }
    maintenance_gates = {
        "incremental_evidence_conservation_error_is_zero": (
            maintenance["incremental_evidence_conservation_error"] == 0
        ),
        "full_rebuild_evidence_conservation_error_is_zero": (
            maintenance["full_rebuild_evidence_conservation_error"] == 0
        ),
        "incremental_vs_full_rebuild_retrieval_agreement_at_least_090": (
            maintenance["incremental_vs_full_rebuild_retrieval_agreement"] is not None
            and maintenance["incremental_vs_full_rebuild_retrieval_agreement"] >= 0.90
        ),
    }
    comparison = {
        "schema": "roboharn_evo/hpk/family_efficiency_comparison/v1",
        "retrieval": retrieval,
        "maintenance": maintenance,
        "retrieval_gates": retrieval_gates,
        "maintenance_gates": maintenance_gates,
        "retrieval_go": all(retrieval_gates.values()),
        "family_only_ablation_go": (
            all(retrieval_gates.values()) if fair_family_ablation else None
        ),
        "maintenance_go": all(maintenance_gates.values()),
        "overall_integration_go": all(retrieval_gates.values())
        and all(maintenance_gates.values()),
        "comparison_scope": (
            "family only ablation"
            if fair_family_ablation
            else "integrated optimized pipeline versus legacy exhaustive baseline"
        ),
        "fair_family_ablation": fair_family_ablation,
        "case_alignment_exact": True,
        "exhaustive_atomic_prompt_representation": exhaustive_representation,
        "family_atomic_prompt_representation": family_representation,
        "measurement_notes": {
            "retrieval": "live provider input tokens and completed model calls",
            "maintenance": (
                "atomic records presented to consolidation and completed model calls"
            ),
            "incremental_maintenance_authority": incremental_report.get(
                "evaluation_authority",
                "not declared by report",
            ),
            "full_rebuild_authority": full_rebuild_report.get(
                "evaluation_authority",
                "not declared by report",
            ),
            "scientific_boundary": (
                "held-in pure-text Family-only ablation; not task success or transfer. "
                "The exhaustive and Family-routed paths use the same atomic prompt "
                "representation and exactly aligned evaluation cases."
                if fair_family_ablation
                else (
                    "held-in pure-text integration evaluation; not task success or "
                    "transfer. A prompt-representation mismatch means this is an "
                    "integrated-system comparison, not a Family-only ablation."
                )
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "comparison_table.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retrieval[0]))
        writer.writeheader()
        writer.writerows(retrieval)

    x = np.arange(len(retrieval))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    figure.suptitle(
        "HPK Knowledge Family Retrieval and Incremental Maintenance",
        fontsize=16,
        fontweight="bold",
    )

    exhaustive_tokens = [row["exhaustive_input_tokens"] / 1000 for row in retrieval]
    family_tokens = [row["family_input_tokens"] / 1000 for row in retrieval]
    exhaustive_label = (
        "Compact exhaustive" if fair_family_ablation else "Legacy exhaustive"
    )
    family_label = "Family routed" if fair_family_ablation else "Optimized Family"
    axes[0, 0].bar(x - width / 2, exhaustive_tokens, width, label=exhaustive_label)
    axes[0, 0].bar(x + width / 2, family_tokens, width, label=family_label)
    axes[0, 0].set_xticks(x, [str(row["scale"]) for row in retrieval])
    axes[0, 0].set_xlabel("Atomic knowledge units")
    axes[0, 0].set_ylabel("Provider input tokens (thousands)")
    axes[0, 0].set_title("A. Retrieval input cost")
    axes[0, 0].legend(frameon=False)

    reductions = [row["input_token_reduction"] * 100 for row in retrieval]
    bars = axes[0, 1].bar(x, reductions, color="#2A9D8F")
    axes[0, 1].axhline(60, color="#D1495B", linestyle="--", label="60% Gate")
    axes[0, 1].set_xticks(x, [str(row["scale"]) for row in retrieval])
    axes[0, 1].set_xlabel("Atomic knowledge units")
    axes[0, 1].set_ylabel("Input-token reduction (%)")
    axes[0, 1].set_title(
        "B. Savings from Family routing"
        if fair_family_ablation
        else "B. Integrated input-token savings"
    )
    axes[0, 1].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0, 1].legend(frameon=False)

    exhaustive_calls = [row["exhaustive_model_calls"] for row in retrieval]
    family_calls = [row["family_model_calls"] for row in retrieval]
    axes[1, 0].bar(x - width / 2, exhaustive_calls, width, label=exhaustive_label)
    axes[1, 0].bar(x + width / 2, family_calls, width, label=family_label)
    axes[1, 0].set_xticks(x, [str(row["scale"]) for row in retrieval])
    axes[1, 0].set_xlabel("Atomic knowledge units")
    axes[1, 0].set_ylabel("Model calls")
    axes[1, 0].set_title("C. Retrieval calls")
    axes[1, 0].legend(frameon=False)

    labels = ("Atomic inputs", "Model calls")
    full_values = (full_atomic, maintenance["full_rebuild_model_calls"])
    incremental_values = (
        incremental_atomic,
        maintenance["incremental_model_calls"],
    )
    mx = np.arange(len(labels))
    axes[1, 1].bar(mx - width / 2, full_values, width, label="Full rebuild")
    axes[1, 1].bar(mx + width / 2, incremental_values, width, label="Incremental")
    axes[1, 1].set_xticks(mx, labels)
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("D. Maintenance work (lower is better)")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_dir / "hpk_family_efficiency.png", dpi=220)
    figure.savefig(output_dir / "hpk_family_efficiency.pdf")
    plt.close(figure)

    largest = retrieval[-1]
    note = f"""# HPK Family efficiency — experiment note

## Result

At {largest["scale"]} atomic units, Family-routed retrieval used {largest["family_input_tokens"]:,} provider input tokens versus {largest["exhaustive_input_tokens"]:,} for exhaustive retrieval, a {largest["input_token_reduction"]:.1%} reduction. Final selection accuracy was {largest["family_final_accuracy"]:.1%}; Family Recall@2 was {largest["family_recall_at_2"]:.1%}.

Incremental maintenance presented {incremental_atomic} atomic records to consolidation versus {full_atomic} for a full rebuild ({maintenance["atomic_input_reduction"]:.1%} fewer). It used {maintenance["incremental_model_calls"]} model calls versus {maintenance["full_rebuild_model_calls"]} for full rebuild. This second number must be reported even if it is unfavorable: scoped maintenance reduces knowledge read volume but does not automatically reduce calls when several Families are affected.

## Experimental boundary

Preregistered retrieval Gate: {"Go" if comparison["retrieval_go"] else "No-Go"}. Maintenance integrity/agreement Gate: {"Go" if comparison["maintenance_go"] else "No-Go"}.

Comparison scope: {comparison["comparison_scope"]}. Exhaustive prompt representation: {comparison["exhaustive_atomic_prompt_representation"]}. Family prompt representation: {comparison["family_atomic_prompt_representation"]}.

This is a held-in, pure-text integration comparison. It measures retrieval and maintenance efficiency, recall, selection accuracy, and evidence conservation. Incremental maintenance authority: {comparison["measurement_notes"]["incremental_maintenance_authority"]}. Full-rebuild authority: {comparison["measurement_notes"]["full_rebuild_authority"]}. It does not establish downstream task-success improvement, cross-task transfer, or statistical significance. Use the PDF as a draft experiment figure and regenerate it from frozen multi-seed results before publication.
"""
    (output_dir / "ICLR_EXPERIMENT_NOTE.md").write_text(note, encoding="utf-8")
    return comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HPK Family retrieval and maintenance efficiency.")
    parser.add_argument("--exhaustive-report", type=Path, required=True)
    parser.add_argument("--optimized-report", type=Path, required=True)
    parser.add_argument("--incremental-report", type=Path, required=True)
    parser.add_argument("--full-rebuild-report", type=Path, required=True)
    parser.add_argument("--agreement-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    comparison = build_efficiency_artifacts(
        exhaustive_report_path=args.exhaustive_report,
        optimized_report_path=args.optimized_report,
        incremental_report_path=args.incremental_report,
        full_rebuild_report_path=args.full_rebuild_report,
        output_dir=args.output_dir,
        agreement_report_path=args.agreement_report,
    )
    print(json.dumps(comparison, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

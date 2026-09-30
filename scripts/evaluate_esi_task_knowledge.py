from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_experiment_metadata
from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    ESIRetrievalAudit,
    action_family_accuracy,
    aggregate_cost,
    answer_accuracy,
    behavior_change_at_frozen_state,
    confidence_diagnostics,
    conservative_redundant_action_rate,
    holm_adjust,
    invalid_noop_rate,
    knowledge_adoption,
    load_split_manifest,
    next_subtask_accuracy,
    paired_bootstrap_difference,
    paired_mcnemar,
    premature_and_early_wrong_answer,
    step_distribution,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute ESI Task-HPK tables from ordinary raw artifacts."
    )
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="MODE=/absolute/audit/root; repeat for off/flat/task/shuffled",
    )
    parser.add_argument("--frozen-probes", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence-bins", type=int, default=10)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    parsed: dict[str, Path] = {}
    for raw in args.condition:
        if "=" not in raw:
            parser.error("--condition must be MODE=ROOT")
        mode, root = raw.split("=", 1)
        mode = mode.strip().casefold()
        if mode not in {"off", "flat", "task", "shuffled"} or mode in parsed:
            parser.error("condition modes must be unique off/flat/task/shuffled")
        parsed[mode] = Path(root).expanduser().resolve()
    if args.bootstrap_samples <= 0 or args.confidence_bins <= 0:
        parser.error("bootstrap samples and confidence bins must be positive")
    if not 0.0 <= args.high_confidence_threshold <= 1.0:
        parser.error("high-confidence threshold must be between 0 and 1")
    args.conditions = parsed
    return args


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected one JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    result = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} line {line_number} must be an object")
        result.append(dict(value))
    return tuple(result)


def _episode_key(answer: Mapping[str, Any]) -> str:
    return json.dumps(
        [answer.get(field) for field in ("task", "scene", "room", "question_id")],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _upstream_integer_option(config: Mapping[str, Any], name: str) -> int | None:
    values = config.get("upstream_arguments")
    if not isinstance(values, list):
        return None
    for index, value in enumerate(values[:-1]):
        if value == name:
            try:
                return int(values[index + 1])
            except (TypeError, ValueError):
                return None
    return None


def _public_observation_labels(
    answer: Mapping[str, Any], history: list[Any]
) -> tuple[str | None, ...]:
    root_value = answer.get("step_image_dir")
    if not isinstance(root_value, str) or not root_value.strip():
        return tuple(None for _ in history)
    root = Path(root_value).expanduser()
    known_states: list[bytes] = []
    labels = []
    for item in history:
        image = item.get("image") if isinstance(item, Mapping) else None
        if not isinstance(image, str) or not image.strip():
            labels.append(None)
            continue
        path = Path(image)
        if not path.is_absolute():
            path = root / path
        try:
            content = path.read_bytes()
        except OSError:
            labels.append(None)
            continue
        try:
            index = known_states.index(content)
        except ValueError:
            index = len(known_states)
            known_states.append(content)
        labels.append(f"public observation state {index}")
    return tuple(labels)


def _collect_condition(root: Path, mode: str) -> dict[str, Any]:
    episodes = []
    actions = []
    adoption = []
    costs = []
    excluded = []
    run_configs = []
    frozen_checks = []
    raw_files = sorted(root.rglob("upstream_answer.json"))
    for answer_path in raw_files:
        answer = _read_json(answer_path)
        config_path = answer_path.parent / "run_config.json"
        run_config = _read_json(config_path) if config_path.is_file() else {}
        if not isinstance(answer.get("correct"), bool):
            excluded.append(
                {
                    "answer_file": str(answer_path),
                    "reason": answer.get("skip_reason")
                    or "official result has no binary correctness",
                }
            )
            continue
        final = answer.get("final_answer")
        if not isinstance(final, Mapping):
            final = {}
        history = answer.get("history")
        if not isinstance(history, list):
            history = []
        observation_labels = _public_observation_labels(answer, history)
        key = _episode_key(answer)
        episode = {
            "instance_key": key,
            "task": answer.get("task"),
            "source_metadata": answer.get("source_metadata"),
            "question_index": answer.get("question_index"),
            "correct": answer.get("correct"),
            "steps": int(final.get("steps", len(history))),
            "answer_step": int(final.get("answer_step", -1)),
            "max_steps": _upstream_integer_option(run_config, "--max-steps")
            or max(int(final.get("steps", len(history))), 1),
            "premature": answer.get("premature"),
            "confidence": final.get("confidence"),
        }
        episodes.append(episode)
        run_configs.append(run_config)
        frozen_path = answer_path.parent / "frozen_store_before_after_check.json"
        frozen_checks.append(_read_json(frozen_path) if frozen_path.is_file() else {})
        for item, observation_label in zip(history, observation_labels):
            if not isinstance(item, Mapping):
                continue
            result = item.get("action_result")
            if not isinstance(result, Mapping):
                result = {}
            action = str(item.get("action", ""))
            terminal = action in {"stop", "force_final_choice"}
            actions.append(
                {
                    "instance_key": key,
                    "action": action,
                    "parser_valid": bool(action),
                    "execution_handled": (
                        True if terminal else bool(result.get("handled", False))
                    ),
                    "operation": result.get("operation", ""),
                    "terminal": terminal,
                    "public_observation_summary": observation_label,
                }
            )
        retrieval_path = answer_path.parent / "hpk_retrieval.jsonl"
        legacy_retrieval_path = answer_path.parent / "afk_retrieval.jsonl"
        if legacy_retrieval_path.exists():
            if retrieval_path.exists():
                raise ValueError(f"ambiguous retrieval artifacts: {answer_path.parent}")
            retrieval_path = legacy_retrieval_path
        for item in _read_jsonl(retrieval_path):
            audit = ESIRetrievalAudit.from_mapping(item)
            adoption.append(
                {
                    "instance_key": key,
                    "knowledge_adopted": audit.knowledge_adopted,
                    "adoption_label": None,
                }
            )
        metrics_path = answer_path.parent / "metrics.json"
        if metrics_path.is_file():
            costs.append(_read_json(metrics_path))
    return {
        "mode": mode,
        "episodes": episodes,
        "actions": actions,
        "adoption": adoption,
        "costs": costs,
        "excluded": excluded,
        "run_configs": run_configs,
        "frozen_checks": frozen_checks,
        "answer_files": [str(path) for path in raw_files],
    }


def _condition_metrics(
    raw: Mapping[str, Any],
    *,
    confidence_bins: int,
    high_confidence_threshold: float,
) -> dict[str, Any]:
    episodes = raw["episodes"]
    actions = raw["actions"]
    result = {
        "answer_accuracy": answer_accuracy(episodes) if episodes else None,
        "invalid_noop": invalid_noop_rate(actions),
        "steps": step_distribution(episodes),
        "cost": aggregate_cost(raw["costs"]),
        "knowledge_adoption": (
            knowledge_adoption(raw["adoption"]) if raw["adoption"] else None
        ),
        "confidence": (
            confidence_diagnostics(
                episodes,
                bins=confidence_bins,
                high_confidence_threshold=high_confidence_threshold,
            )
            if episodes and all(item.get("confidence") is not None for item in episodes)
            else {
                "available": False,
                "reason": "official final confidence is unavailable",
            }
        ),
    }
    if all(
        item.get("terminal") or item.get("public_observation_summary") is not None
        for item in actions
    ):
        result["redundant_action"] = conservative_redundant_action_rate(actions)
    else:
        result["redundant_action"] = {
            "available": False,
            "reason": "one or more public observation images are unavailable",
        }
    if episodes and all(item.get("premature") is not None for item in episodes):
        result["premature_and_early_wrong"] = premature_and_early_wrong_answer(episodes)
    else:
        result["premature_and_early_wrong"] = {
            "available": False,
            "reason": "objective premature-answer annotations are unavailable",
        }
    adopted_keys = {
        item["instance_key"] for item in raw["adoption"] if item["knowledge_adopted"]
    }
    retrieved = [item for item in episodes if item["instance_key"] in adopted_keys]
    not_retrieved = [
        item for item in episodes if item["instance_key"] not in adopted_keys
    ]
    result["analysis_slices"] = {
        "by_small_task": (
            result["answer_accuracy"]["per_task_accuracy"]
            if result["answer_accuracy"] is not None
            else {}
        ),
        "retrieved": answer_accuracy(retrieved) if retrieved else None,
        "not_retrieved": answer_accuracy(not_retrieved) if not_retrieved else None,
        "selection_bias_warning": (
            "retrieved/not-retrieved is mechanism analysis, not intention-to-treat"
        ),
        "action_selection_stratum": {
            "available": False,
            "reason": "no frozen stratum annotations were supplied",
        },
        "initial_confidence_band": {
            "available": False,
            "reason": "confidence-band boundaries were not supplied",
        },
        "source_support_count": {
            "available": False,
            "reason": "source-support annotations were not supplied",
        },
    }
    return result


def _paired_comparison(
    off: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    def indexed(values):
        result = {}
        for item in values:
            key = item["instance_key"]
            if key in result:
                raise ValueError(f"duplicate paired ESI instance key: {key}")
            result[key] = item
        return result

    off_map = indexed(off["episodes"])
    task_map = indexed(task["episodes"])
    if set(off_map) != set(task_map):
        raise ValueError("paired ESI conditions contain different instance sets")
    keys = sorted(off_map)
    if not keys:
        return {"available": False, "reason": "no paired instances"}
    off_outcomes = [bool(off_map[key]["correct"]) for key in keys]
    task_outcomes = [bool(task_map[key]["correct"]) for key in keys]
    return {
        "paired_instance_count": len(keys),
        "mcnemar": paired_mcnemar(off_outcomes, task_outcomes),
        "accuracy_difference": paired_bootstrap_difference(
            [float(value) for value in off_outcomes],
            [float(value) for value in task_outcomes],
            seed=seed,
            samples=samples,
        ),
        "steps_difference": paired_bootstrap_difference(
            [float(off_map[key]["steps"]) for key in keys],
            [float(task_map[key]["steps"]) for key in keys],
            seed=seed,
            samples=samples,
        ),
    }


def _probe_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False, "reason": "no frozen probe annotations"}
    probes = _read_jsonl(path.expanduser().resolve())
    return {
        "available": True,
        "next_subtask": next_subtask_accuracy(probes),
        "action_family": action_family_accuracy(probes),
        "behavior_change": behavior_change_at_frozen_state(probes),
        "raw_probe_count": len(probes),
    }


def _write_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "answer_accuracy_micro",
                "answer_accuracy_macro",
                "invalid_noop_rate",
                "steps_correct_mean",
                "evaluated_model_calls",
                "retrieval_model_calls",
            ]
        )
        for mode, value in sorted(metrics.items()):
            accuracy = value.get("answer_accuracy") or {}
            steps = value.get("steps", {}).get("correct_episodes", {})
            cost = value.get("cost", {})
            writer.writerow(
                [
                    mode,
                    accuracy.get("micro_accuracy"),
                    accuracy.get("macro_accuracy"),
                    value.get("invalid_noop", {}).get("invalid_noop_rate"),
                    steps.get("mean"),
                    cost.get("evaluated_model_calls"),
                    cost.get("retrieval_model_calls"),
                ]
            )


def _semantic_upstream_arguments(config: Mapping[str, Any]) -> tuple[str, ...]:
    values = config.get("upstream_arguments")
    if not isinstance(values, list):
        raise ValueError("formal ESI run is missing upstream arguments")
    ignored_with_value = {"--results-root", "--step-image-root"}
    result = []
    index = 0
    while index < len(values):
        value = str(values[index])
        if value in ignored_with_value:
            index += 2
            continue
        result.append(value)
        index += 1
    return tuple(result)


def _validate_formal_inputs(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required_modes = {"off", "flat", "task", "shuffled"}
    if set(raw) != required_modes:
        raise ValueError("formal ESI analysis requires all four exact conditions")
    reference_keys = None
    reference_upstream_by_instance = None
    split_bytes = None
    active_identity = None
    for mode in sorted(required_modes):
        condition = raw[mode]
        if condition["excluded"] or not condition["episodes"]:
            raise ValueError("formal ESI conditions cannot be empty or excluded")
        keys = tuple(sorted(item["instance_key"] for item in condition["episodes"]))
        if len(keys) != len(set(keys)):
            raise ValueError("formal ESI condition contains duplicate instances")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("formal ESI conditions use different heldout instances")
        if len(condition["run_configs"]) != len(condition["episodes"]):
            raise ValueError("formal ESI run-config coverage is incomplete")
        if len(condition["frozen_checks"]) != len(condition["episodes"]):
            raise ValueError("formal ESI frozen-check coverage is incomplete")
        upstream_by_instance = {}
        for episode, config, frozen in zip(
            condition["episodes"],
            condition["run_configs"],
            condition["frozen_checks"],
        ):
            config = normalize_hpk_experiment_metadata(config)
            if (
                config.get("scientific_scope") != "ESI RQ2 Task Knowledge only"
                or config.get("upstream_commit")
                != "3c1756396f32b1a90c1f72356a7fde45f418e179"
                or config.get("hpk_mode") != mode
                or config.get("formal") is not True
                or config.get("split_part") != "heldout"
                or config.get("heldout_store_update_enabled") is not False
                or config.get("retrieval_current_image_grounded") is not (mode != "off")
            ):
                raise ValueError("formal ESI run configuration is invalid")
            split_entry = config.get("split_entry")
            if not isinstance(split_entry, Mapping):
                raise ValueError("formal ESI split entry is missing")
            try:
                same_metadata = (
                    Path(str(split_entry.get("metadata_path"))).resolve()
                    == Path(str(episode.get("source_metadata"))).resolve()
                )
            except (OSError, RuntimeError, ValueError):
                same_metadata = False
            if (
                not same_metadata
                or split_entry.get("question_index") != episode.get("question_index")
                or " ".join(
                    str(split_entry.get("small_task") or "")
                    .casefold()
                    .replace("_", " ")
                    .split()
                )
                != " ".join(
                    str(episode.get("task") or "").casefold().replace("_", " ").split()
                )
            ):
                raise ValueError("formal ESI result does not match its heldout entry")
            renderer = config.get("renderer_preflight")
            render_audit = config.get("canonical_render_audit")
            if (
                not isinstance(renderer, Mapping)
                or renderer.get("compatible_for_formal") is not True
                or not isinstance(render_audit, Mapping)
                or render_audit.get("human_inspected") is not True
                or render_audit.get("renderer_valid") is not True
            ):
                raise ValueError("formal ESI renderer gates are incomplete")
            manifest_path = Path(str(config.get("split_manifest_path") or ""))
            if not manifest_path.is_file():
                raise ValueError("formal ESI split manifest is unavailable")
            current_split_bytes = manifest_path.read_bytes()
            if split_bytes is None:
                split_bytes = current_split_bytes
            elif current_split_bytes != split_bytes:
                raise ValueError("formal ESI conditions use different split manifests")
            manifest = load_split_manifest(manifest_path)
            if dict(split_entry) not in [item.to_dict() for item in manifest.heldout]:
                raise ValueError("formal ESI split entry is not in heldout manifest")
            semantic_args = _semantic_upstream_arguments(config)
            upstream_by_instance[episode["instance_key"]] = semantic_args
            if mode == "off":
                if frozen.get("heldout_update_performed") is not False:
                    raise ValueError("formal ESI-Off reports a heldout Store update")
                continue
            store_contents = config.get("frozen_store_contents")
            if not isinstance(store_contents, Mapping) or not store_contents:
                raise ValueError("formal ESI frozen Store content audit is missing")
            identity = (
                config.get("frozen_store_root"),
                dict(store_contents),
                config.get("retrieval_provider"),
                config.get("retrieval_model"),
                config.get("candidate_cap"),
                config.get("context_token_cap"),
                config.get("retrieval_current_image_grounded"),
            )
            if active_identity is None:
                active_identity = identity
            elif identity != active_identity:
                raise ValueError("formal ESI active-condition controls differ")
            if (
                frozen.get("checked") is not True
                or frozen.get("unchanged") is not True
                or frozen.get("heldout_update_performed") is not False
            ):
                raise ValueError("formal ESI frozen Store check failed")
            if mode == "shuffled" and (
                frozen.get("shuffle_manifest_checked") is not True
                or frozen.get("shuffle_manifest_unchanged") is not True
            ):
                raise ValueError("formal ESI shuffle manifest check failed")
            if mode == "shuffled" and not isinstance(
                config.get("shuffle_manifest_content"), str
            ):
                raise ValueError("formal ESI shuffle manifest content is missing")
        if reference_upstream_by_instance is None:
            reference_upstream_by_instance = upstream_by_instance
        elif upstream_by_instance != reference_upstream_by_instance:
            raise ValueError("formal ESI upstream model/task arguments differ")
    return {
        "passed": True,
        "condition_count": len(required_modes),
        "paired_instance_count": len(reference_keys or ()),
        "store_updates": 0,
    }


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"analysis output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw = {
        mode: _collect_condition(root, mode) for mode, root in args.conditions.items()
    }
    formal_gate = (
        _validate_formal_inputs(raw)
        if args.formal
        else {
            "passed": False,
            "reason": "analysis was not requested as a formal run",
        }
    )
    condition_metrics = {
        mode: _condition_metrics(
            value,
            confidence_bins=args.confidence_bins,
            high_confidence_threshold=args.high_confidence_threshold,
        )
        for mode, value in raw.items()
    }
    comparisons = {}
    for baseline in ("off", "flat", "shuffled"):
        name = f"task_vs_{baseline}"
        comparisons[name] = (
            _paired_comparison(
                raw[baseline],
                raw["task"],
                seed=args.bootstrap_seed,
                samples=args.bootstrap_samples,
            )
            if {baseline, "task"}.issubset(raw)
            else {
                "available": False,
                "reason": f"{baseline}/task conditions not both supplied",
            }
        )
    secondary_p = {
        name: value["mcnemar"]["exact_two_sided_p"]
        for name, value in comparisons.items()
        if name in {"task_vs_flat", "task_vs_shuffled"}
        and value.get("available") is not False
    }
    frozen_probe_records = (
        _read_jsonl(args.frozen_probes.expanduser().resolve())
        if args.frozen_probes is not None
        else ()
    )
    result = {
        "scientific_scope": "ESI RQ2 Task Knowledge only",
        "analysis_parameters": {
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence_bins": args.confidence_bins,
            "high_confidence_threshold": args.high_confidence_threshold,
        },
        "formal_gate": formal_gate,
        "condition_metrics": condition_metrics,
        "paired_comparisons": comparisons,
        "secondary_holm_adjusted_p": (holm_adjust(secondary_p) if secondary_p else {}),
        "frozen_probe_metrics": _probe_metrics(args.frozen_probes),
        "formal_scientific_numbers": args.formal,
    }
    (output / "raw_metric_inputs.json").write_text(
        json.dumps(
            {**raw, "_frozen_probe_records": list(frozen_probe_records)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "table_2a.csv", condition_metrics)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

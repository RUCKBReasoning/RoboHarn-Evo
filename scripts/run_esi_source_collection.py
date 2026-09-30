#!/usr/bin/env python3
"""Execute one preregistered ESI source queue without episode retries."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_experiment_metadata

RUNS_ROOT = (REPOSITORY_ROOT / "eval_result" / "esi_bench" / "runs").resolve()


class SourceCollectionError(RuntimeError):
    """Raised when a preregistered source collection cannot continue."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceCollectionError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise SourceCollectionError(f"{label} must be one object")
    return dict(value)


def validate_collection_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "primary_small_task",
        "selection_policy",
        "episode_config",
        "maximum_budget",
        "ordered_instances",
        "transfer_boundary",
    }
    if set(value) != expected or value.get("schema") not in {
        "roboharn_evo/esi_bench/source_collection_plan/v1",
        "tcm/esi_bench/source_collection_plan/v1",
    }:
        raise SourceCollectionError("source collection plan fields mismatch")
    policy = value.get("selection_policy")
    config = value.get("episode_config")
    instances = value.get("ordered_instances")
    if not isinstance(policy, Mapping) or not isinstance(config, Mapping):
        raise SourceCollectionError("source collection policy/config is invalid")
    config = normalize_hpk_experiment_metadata(config)
    if not isinstance(instances, list) or not instances:
        raise SourceCollectionError("source collection queue is empty")
    required = int(policy.get("required_correct_trajectories", 0))
    attempt_cap = int(policy.get("attempt_cap", 0))
    if (
        policy.get("uses_source_split_only") is not True
        or policy.get("independence_key") != "scene_group"
        or policy.get("stop_after_required_correct") is not True
        or not 1 <= required <= attempt_cap == len(instances)
    ):
        raise SourceCollectionError("source collection stop policy is invalid")
    if (
        config.get("hpk_mode") != "off"
        or config.get("model") != "gpt-5.5"
        or config.get("reasoning_effort") != "xhigh"
        or config.get("automatic_retries") != 0
    ):
        raise SourceCollectionError("source collection runtime policy is invalid")
    positions = []
    scenes = []
    for item in instances:
        if not isinstance(item, Mapping):
            raise SourceCollectionError("source collection instance is invalid")
        positions.append(item.get("queue_position"))
        scenes.append(str(item.get("scene_group") or "").casefold())
        for field in (
            "instance_ref",
            "small_task",
            "big_task",
            "scene_group",
            "room_group",
            "metadata_path",
            "runner_task",
        ):
            if not str(item.get(field) or "").strip():
                raise SourceCollectionError(f"source collection instance lacks {field}")
    if positions != list(range(1, len(instances) + 1)):
        raise SourceCollectionError("source collection queue positions are invalid")
    if len(scenes) != len(set(scenes)):
        raise SourceCollectionError("source collection scenes are not independent")
    result = json.loads(json.dumps(dict(value), ensure_ascii=False))
    result["episode_config"] = config
    return result


def episode_command(
    *,
    python: str,
    instance: Mapping[str, Any],
    plan: Mapping[str, Any],
    split_manifest: Path,
    episode_root: Path,
    evaluated_model_url: str,
) -> list[str]:
    config = plan["episode_config"]
    return [
        python,
        str(REPOSITORY_ROOT / "scripts" / "run_esi_bench.py"),
        "--hpk-mode",
        "off",
        "--hpk-audit-root",
        str(episode_root / "audit"),
        "--evaluated-model-agent-api-url",
        evaluated_model_url,
        "--evaluated-model-reasoning-effort",
        "xhigh",
        "--split-manifest",
        str(split_manifest),
        "--split-part",
        "source",
        "--task",
        str(instance["runner_task"]),
        "--metadata",
        str(instance["metadata_path"]),
        "--question-index",
        str(instance["question_index"]),
        "--results-root",
        str(episode_root / "results"),
        "--step-image-root",
        str(episode_root / "steps"),
        "--provider",
        "gpt",
        "--model",
        "gpt-5.5",
        "--max-steps",
        str(config["max_steps"]),
        "--min-steps",
        str(config["min_steps"]),
        "--threshold",
        str(config["confidence_threshold"]),
        "--robot",
        "R1",
    ]


def summary_command(
    *,
    python: str,
    episode_root: Path,
    evaluated_model_url: str,
) -> list[str]:
    return [
        python,
        str(REPOSITORY_ROOT / "scripts" / "materialize_esi_source_trace.py"),
        "--audit-root",
        str(episode_root / "audit"),
        "--output-root",
        str(episode_root / "source_trace"),
        "--planner-url",
        evaluated_model_url,
    ]


def _run_logged(command: Sequence[str], root: Path, label: str) -> int:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{label}.stdout.log").open("w", encoding="utf-8") as stdout:
        with (root / f"{label}.stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                list(command),
                cwd=REPOSITORY_ROOT,
                env=dict(os.environ),
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
    return int(completed.returncode)


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen ESI source queue and stop after enough correct traces."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluated-model-url", required=True)
    parser.add_argument(
        "--resume-after-materialization",
        action="store_true",
        help=(
            "Continue a stopped queue only after its last correct rollout has "
            "subsequently produced the missing source trace."
        ),
    )
    return parser.parse_args()


def resume_materialized_collection(
    plan: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path], int]:
    receipt = _read_object(output / "collection_receipt.json", "receipt")
    attempts = receipt.get("attempts")
    instances = plan["ordered_instances"]
    if (
        receipt.get("schema") not in {"roboharn_evo/esi_bench/source_collection_receipt/v1", "tcm/esi_bench/source_collection_receipt/v1"}
        or receipt.get("status") != "failed_before_completion"
        or receipt.get("automatic_retries") != 0
        or not isinstance(attempts, list)
        or not attempts
        or len(attempts) > len(instances)
    ):
        raise SourceCollectionError("source collection is not resumable")
    trace_paths = []
    for index, attempt in enumerate(attempts):
        instance = instances[index]
        position = index + 1
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("queue_position") != position
            or attempt.get("instance_ref") != instance["instance_ref"]
            or attempt.get("scene_group") != instance["scene_group"]
            or attempt.get("rollout_exit_code") != 0
        ):
            raise SourceCollectionError("source collection resume prefix mismatch")
        trace_path = (
            output / f"episode_{position:02d}" / "source_trace" / ("source_trace.jsonl")
        )
        if attempt.get("benchmark_correct") is True:
            if index == len(attempts) - 1 and attempt.get("summary_exit_code") != 0:
                if not trace_path.is_file() or trace_path.stat().st_size == 0:
                    raise SourceCollectionError(
                        "last failed materialization has no recovered source trace"
                    )
                attempt["summary_exit_code"] = 0
            if attempt.get("summary_exit_code") != 0 or not trace_path.is_file():
                raise SourceCollectionError(
                    "correct source attempt is not materialized"
                )
            trace_paths.append(trace_path)
        elif attempt.get("summary_exit_code") is not None or trace_path.exists():
            raise SourceCollectionError("incorrect attempt cannot own a source trace")
    receipt["correct_trace_count"] = len(trace_paths)
    receipt["status"] = "running"
    _write_receipt(output / "collection_receipt.json", receipt)
    return receipt, trace_paths, len(attempts)


def main() -> int:
    args = _parse_args()
    plan = validate_collection_plan(
        _read_object(args.plan.expanduser().resolve(), "source collection plan")
    )
    split_manifest = args.split_manifest.expanduser().resolve()
    if not split_manifest.is_file():
        raise SourceCollectionError("split manifest is missing")
    output = args.output_root.expanduser().resolve()
    if not output.is_relative_to(RUNS_ROOT):
        raise SourceCollectionError("source collection output must stay under runs")
    receipt_path = output / "collection_receipt.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume_after_materialization:
            raise FileExistsError(f"source collection output is not empty: {output}")
        receipt, trace_paths, consumed = resume_materialized_collection(plan, output)
    else:
        output.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": "roboharn_evo/esi_bench/source_collection_receipt/v1",
            "status": "running",
            "required_correct": plan["selection_policy"][
                "required_correct_trajectories"
            ],
            "attempt_cap": plan["selection_policy"]["attempt_cap"],
            "attempts": [],
            "correct_trace_count": 0,
            "automatic_retries": 0,
        }
        _write_receipt(receipt_path, receipt)
        trace_paths = []
        consumed = 0
    for instance in plan["ordered_instances"][consumed:]:
        position = int(instance["queue_position"])
        episode = output / f"episode_{position:02d}"
        print(f"[ESI source] starting queue position {position}", flush=True)
        code = _run_logged(
            episode_command(
                python=sys.executable,
                instance=instance,
                plan=plan,
                split_manifest=split_manifest,
                episode_root=episode,
                evaluated_model_url=args.evaluated_model_url,
            ),
            episode,
            "rollout",
        )
        attempt = {
            "queue_position": position,
            "instance_ref": instance["instance_ref"],
            "scene_group": instance["scene_group"],
            "rollout_exit_code": code,
            "benchmark_correct": None,
            "summary_exit_code": None,
        }
        receipt["attempts"].append(attempt)
        if code != 0:
            receipt["status"] = "failed_before_completion"
            _write_receipt(receipt_path, receipt)
            print(f"[ESI source] rollout failed at queue position {position}")
            return code or 2
        result = _read_object(episode / "audit" / "upstream_answer.json", "result")
        correct = result.get("correct") is True
        attempt["benchmark_correct"] = correct
        if correct:
            summary_code = _run_logged(
                summary_command(
                    python=sys.executable,
                    episode_root=episode,
                    evaluated_model_url=args.evaluated_model_url,
                ),
                episode,
                "materialize",
            )
            attempt["summary_exit_code"] = summary_code
            if summary_code != 0:
                receipt["status"] = "failed_before_completion"
                _write_receipt(receipt_path, receipt)
                print(f"[ESI source] summary failed at queue position {position}")
                return summary_code or 2
            trace_path = episode / "source_trace" / "source_trace.jsonl"
            if not trace_path.is_file():
                raise SourceCollectionError("materializer produced no source trace")
            trace_paths.append(trace_path)
            receipt["correct_trace_count"] = len(trace_paths)
        _write_receipt(receipt_path, receipt)
        print(
            f"[ESI source] position {position} correct={correct}; "
            f"traces={len(trace_paths)}/{receipt['required_correct']}",
            flush=True,
        )
        if len(trace_paths) == receipt["required_correct"]:
            break
    if len(trace_paths) != receipt["required_correct"]:
        receipt["status"] = "attempt_cap_exhausted"
        _write_receipt(receipt_path, receipt)
        return 3
    (output / "source_traces.jsonl").write_text(
        "".join(path.read_text(encoding="utf-8") for path in trace_paths),
        encoding="utf-8",
    )
    receipt["status"] = "completed"
    _write_receipt(receipt_path, receipt)
    print("[ESI source] collection completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

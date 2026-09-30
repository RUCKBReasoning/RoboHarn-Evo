#!/usr/bin/env python3
"""Capture preregistered ESI public initial states without model execution."""

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
RUNS_ROOT = (REPOSITORY_ROOT / "eval_result" / "esi_bench" / "runs").resolve()


class FrozenCaptureError(RuntimeError):
    pass


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FrozenCaptureError(f"{label} must be one object")
    return dict(value)


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    schema = value.get("schema")
    legacy_fields = {
        "schema",
        "split_part",
        "selection_policy",
        "budgets",
        "ordered_states",
    }
    current_fields = {
        *legacy_fields,
        "probe_profile",
        "excluded_instance_refs",
    }
    if not (
        (schema in {"roboharn_evo/esi_bench/frozen_probe_plan/v1", "tcm/esi_bench/frozen_probe_plan/v1"} and set(value) == legacy_fields)
        or (
            schema in {"roboharn_evo/esi_bench/frozen_probe_plan/v2", "tcm/esi_bench/frozen_probe_plan/v2"}
            and set(value) == current_fields
        )
    ):
        raise FrozenCaptureError("frozen probe plan fields mismatch")
    if schema in {"roboharn_evo/esi_bench/frozen_probe_plan/v2", "tcm/esi_bench/frozen_probe_plan/v2"} and value.get(
        "probe_profile"
    ) not in {"five_condition", "four_condition_exploratory"}:
        raise FrozenCaptureError("frozen probe plan profile is invalid")
    policy = value.get("selection_policy")
    budgets = value.get("budgets")
    states = value.get("ordered_states")
    if (
        value.get("split_part") != "development"
        or not isinstance(policy, Mapping)
        or policy.get("outcome_blind") is not True
        or policy.get("answer_or_score_used") is not False
        or not isinstance(budgets, Mapping)
        or not isinstance(states, list)
        or not states
    ):
        raise FrozenCaptureError("frozen probe plan policy is invalid")
    capture = budgets.get("capture_phase")
    if not isinstance(capture, Mapping) or dict(capture) != {
        "simulator_episodes": len(states),
        "unique_images": len(states),
        "evaluated_model_calls": 0,
        "retrieval_model_calls": 0,
        "executed_actions": 0,
        "automatic_retries": 0,
    }:
        raise FrozenCaptureError("frozen capture budget is invalid")
    if [
        item.get("state_index") for item in states if isinstance(item, Mapping)
    ] != list(range(1, len(states) + 1)):
        raise FrozenCaptureError("frozen state indices are invalid")
    return json.loads(json.dumps(dict(value), ensure_ascii=False))


def capture_command(
    *,
    python: str,
    state: Mapping[str, Any],
    split_manifest: Path,
    state_root: Path,
) -> list[str]:
    return [
        python,
        str(REPOSITORY_ROOT / "scripts" / "run_esi_bench.py"),
        "--hpk-mode",
        "off",
        "--capture-frozen-state-only",
        "--hpk-audit-root",
        str(state_root / "audit"),
        "--split-manifest",
        str(split_manifest),
        "--split-part",
        "development",
        "--task",
        str(state["runner_task"]),
        "--metadata",
        str(state["metadata_path"]),
        "--question-index",
        str(state["question_index"]),
        "--results-root",
        str(state_root / "results"),
        "--step-image-root",
        str(state_root / "steps"),
        "--provider",
        "gpt",
        "--model",
        "gpt-5.5",
        "--max-steps",
        "1",
        "--min-steps",
        "1",
        "--threshold",
        "0.99",
        "--robot",
        "R1",
    ]


def _run_logged(command: Sequence[str], root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    with (
        (root / "capture.stdout.log").open("w", encoding="utf-8") as stdout,
        (root / "capture.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        return int(
            subprocess.run(
                list(command),
                cwd=REPOSITORY_ROOT,
                env=dict(os.environ),
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            ).returncode
        )


def validate_packet(
    packet: Mapping[str, Any], *, state: Mapping[str, Any], state_root: Path
) -> dict[str, Any]:
    if (
        packet.get("schema") not in {"roboharn_evo/esi_bench/frozen_public_state/v1", "tcm/esi_bench/frozen_public_state/v1"}
        or packet.get("split_part") != "development"
        or packet.get("instance_ref") != state.get("instance_ref")
        or packet.get("public_history") != []
        or packet.get("evaluated_model_calls") != 0
        or packet.get("retrieval_model_calls") != 0
        or packet.get("executed_actions") != 0
        or packet.get("answer_read") is not False
        or packet.get("score_read") is not False
        or packet.get("hidden_state_exported") is not False
    ):
        raise FrozenCaptureError("frozen public state packet is invalid")
    image = Path(str(packet.get("current_image_path") or "")).resolve()
    if not image.is_file() or not image.is_relative_to(state_root.resolve()):
        raise FrozenCaptureError("frozen public image is outside its state root")
    return dict(packet)


def load_committed_capture(
    *,
    state: Mapping[str, Any],
    state_root: Path,
    exit_code: int,
) -> tuple[dict[str, Any], bool]:
    if exit_code not in {0, -11}:
        raise FrozenCaptureError("capture process failed before a recognized commit")
    packet = validate_packet(
        _read_object(state_root / "audit" / "frozen_state.json", "state packet"),
        state=state,
        state_root=state_root,
    )
    metrics = _read_object(state_root / "audit" / "metrics.json", "capture metrics")
    if (
        metrics.get("capture_only") is not True
        or metrics.get("correct") is not None
        or metrics.get("steps") != 0
        or metrics.get("evaluated_model_calls") != 0
        or metrics.get("retrieval_model_calls") != 0
        or metrics.get("knowledge_adoption_count") != 0
    ):
        raise FrozenCaptureError("capture metrics do not prove the zero-call boundary")
    audit = state_root / "audit"
    forbidden = (
        "upstream_answer.json",
        "hpk_action_trace.jsonl",
        "hpk_query.jsonl",
        "hpk_retrieval.jsonl",
        "afk_action_trace.jsonl",
        "afk_query.jsonl",
        "afk_retrieval.jsonl",
    )
    if any((audit / name).exists() for name in forbidden):
        raise FrozenCaptureError("capture-only state contains action/answer artifacts")
    return packet, exit_code == -11


def resume_committed_capture(
    plan: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    receipt = _read_object(output / "capture_receipt.json", "capture receipt")
    recorded = receipt.get("states")
    states = plan["ordered_states"]
    if (
        receipt.get("schema") not in {"roboharn_evo/esi_bench/frozen_capture_receipt/v1", "tcm/esi_bench/frozen_capture_receipt/v1"}
        or receipt.get("status") != "failed_before_completion"
        or receipt.get("automatic_retries") != 0
        or not isinstance(recorded, list)
        or not recorded
        or len(recorded) > len(states)
    ):
        raise FrozenCaptureError("frozen capture receipt is not resumable")
    packets = []
    for index, item in enumerate(recorded, 1):
        state = states[index - 1]
        if (
            not isinstance(item, Mapping)
            or item.get("state_index") != index
            or item.get("instance_ref") != state["instance_ref"]
        ):
            raise FrozenCaptureError("frozen capture resume prefix mismatch")
        packet, shutdown_warning = load_committed_capture(
            state=state,
            state_root=output / f"state_{index:02d}",
            exit_code=int(item.get("exit_code")),
        )
        item["capture_committed"] = True
        item["shutdown_status"] = (
            "native_cleanup_failed_after_commit" if shutdown_warning else "clean"
        )
        packets.append(packet)
    if recorded[-1].get("exit_code") != -11:
        raise FrozenCaptureError("resume requires a committed native shutdown failure")
    receipt["completed_state_count"] = len(packets)
    receipt["status"] = "running"
    _write_receipt(output / "capture_receipt.json", receipt)
    return receipt, packets, len(recorded)


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture frozen ESI development states."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resume-after-committed-shutdown-crash",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = validate_plan(_read_object(args.plan.resolve(), "frozen probe plan"))
    split_manifest = args.split_manifest.expanduser().resolve()
    if not split_manifest.is_file():
        raise FrozenCaptureError("split manifest is missing")
    output = args.output_root.expanduser().resolve()
    if not output.is_relative_to(RUNS_ROOT):
        raise FrozenCaptureError("frozen capture output must stay under runs")
    receipt_path = output / "capture_receipt.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume_after_committed_shutdown_crash:
            raise FileExistsError(f"frozen capture output is not empty: {output}")
        receipt, packets, consumed = resume_committed_capture(plan, output)
    else:
        output.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": "roboharn_evo/esi_bench/frozen_capture_receipt/v1",
            "probe_profile": plan.get("probe_profile", "five_condition"),
            "status": "running",
            "authorized_episode_cap": len(plan["ordered_states"]),
            "completed_state_count": 0,
            "evaluated_model_calls": 0,
            "retrieval_model_calls": 0,
            "executed_actions": 0,
            "automatic_retries": 0,
            "states": [],
        }
        _write_receipt(receipt_path, receipt)
        packets = []
        consumed = 0
    for state in plan["ordered_states"][consumed:]:
        index = int(state["state_index"])
        state_root = output / f"state_{index:02d}"
        code = _run_logged(
            capture_command(
                python=sys.executable,
                state=state,
                split_manifest=split_manifest,
                state_root=state_root,
            ),
            state_root,
        )
        item = {
            "state_index": index,
            "instance_ref": state["instance_ref"],
            "exit_code": code,
        }
        receipt["states"].append(item)
        try:
            packet, shutdown_warning = load_committed_capture(
                state=state,
                state_root=state_root,
                exit_code=code,
            )
        except (FrozenCaptureError, FileNotFoundError, json.JSONDecodeError):
            receipt["status"] = "failed_before_completion"
            _write_receipt(receipt_path, receipt)
            return code or 2
        item["capture_committed"] = True
        item["shutdown_status"] = (
            "native_cleanup_failed_after_commit" if shutdown_warning else "clean"
        )
        packets.append(packet)
        receipt["completed_state_count"] = len(packets)
        _write_receipt(receipt_path, receipt)
    (output / "frozen_states.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in packets
        ),
        encoding="utf-8",
    )
    receipt["status"] = (
        "completed_with_shutdown_warnings"
        if any(
            item.get("shutdown_status") == "native_cleanup_failed_after_commit"
            for item in receipt["states"]
        )
        else "completed"
    )
    _write_receipt(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

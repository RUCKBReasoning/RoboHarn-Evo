#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.agent.components.agent_tools.local_skill_registry import LocalSkillRegistry
from policy.roboharn_evo.agent.experience import normalize_semantic_tags
from policy.roboharn_evo.agent.recovery.skill_workflow_loader import SkillRecoveryWorkflowLoader
from policy.roboharn_evo.agent.recovery.tool_specs import RECOVERY_TOOLS
from policy.roboharn_evo.models.agent_api_recovery_adapter import AgentApiRecoveryAdapter, AgentApiRecoveryConfig


ABLATIONS = {
    "full",
    "no_scene_memory",
    "no_observation_preprocess",
    "no_recovery_history",
    "no_scene_and_preprocess",
}


class CaptureBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def plan_recovery(self, *, recovery_payload: dict[str, Any], media: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(recovery_payload))
        workflow_names = [
            str(item.get("name", "")).strip()
            for item in recovery_payload.get("workflow_skills", []) or []
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]
        workflow = workflow_names[0] if workflow_names else "unknown"
        return {
            "recovery_workflow": workflow,
            "tool_calls": [],
            "post_recovery_intent": "retry",
            "reason": "capture backend",
            "stop_condition": "payload captured",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay recovery planner payloads with memory/perception ablations.")
    parser.add_argument("--trace", type=Path, action="append", default=[], required=True, help="Agent trace JSONL. Can be passed multiple times.")
    parser.add_argument("--output", type=Path, default=Path("research_iclr/results/recovery_payload_ablation.jsonl"))
    parser.add_argument("--max-payloads", type=int, default=20)
    parser.add_argument("--ablations", nargs="+", default=sorted(ABLATIONS), choices=sorted(ABLATIONS))
    parser.add_argument("--skill-root", type=Path, default=Path("policy/roboharn_evo/skills"))
    parser.add_argument("--call-backend", action="store_true", help="Actually call the recovery backend for each ablated payload.")
    parser.add_argument("--server-url", default="http://127.0.0.1:9101/recover")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when backend output is invalid or a call fails.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                payload["_line_no"] = line_no
                records.append(payload)
    return records


def build_registry(skill_root: Path) -> LocalSkillRegistry:
    return LocalSkillRegistry(
        configured_paths=[str(skill_root)],
        workspace_root=str(REPO_ROOT),
    )


def capture_payload_from_router(record: dict[str, Any], *, registry: LocalSkillRegistry) -> dict[str, Any] | None:
    if record.get("event") != "recovery_router" or record.get("route_found") is False:
        return None
    capture_backend = CaptureBackend()
    loader = SkillRecoveryWorkflowLoader(
        registry,
        recovery_backend=capture_backend,
    )
    route = loader.resolve(
        signal_name=str(record.get("signal_name", "task_level_recovery_control") or "task_level_recovery_control"),
        reason=str(record.get("reason", "")),
        ood_scenario=str(record.get("ood_scenario", record.get("signal_name", ""))),
        global_task=str(record.get("global_task", "")),
        current_subtask=current_subtask_from_record(record),
        observation_summary=str(record.get("observation_summary", "")),
        robot_state=dict(record.get("robot_state", {}) or {}),
        recovery_state=dict(record.get("recovery_state", {}) or {}),
        recovery_history=recovery_history_from_record(record),
        available_tools=list(record.get("available_tools", []) or sorted(RECOVERY_TOOLS)),
        semantic_tags=normalize_semantic_tags(record.get("semantic_tags")),
        scene_memory=dict(record.get("scene_memory", {}) or {}),
        observation_preprocess=dict(record.get("observation_preprocess", {}) or {}),
    )
    if route is None or not capture_backend.payloads:
        return None
    payload = capture_backend.payloads[-1]
    payload["_trace_router"] = {
        "line": record.get("_line_no"),
        "env_step": record.get("env_step", ""),
        "expected_workflow": str(record.get("workflow", "")),
        "expected_tools": list(record.get("tools", []) or []),
        "expected_post_recovery_intent": str(record.get("post_recovery_intent", "")),
    }
    return payload


def current_subtask_from_record(record: dict[str, Any]) -> str:
    scene_memory = record.get("scene_memory")
    if isinstance(scene_memory, dict):
        focus = scene_memory.get("task_focus")
        if isinstance(focus, dict):
            subtask = str(focus.get("current_subtask", "")).strip()
            if subtask:
                return subtask
    return str(record.get("current_subtask", record.get("subtask", ""))).strip()


def recovery_history_from_record(record: dict[str, Any]) -> list[str]:
    raw = record.get("recovery_history")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    reason = str(record.get("reason", ""))
    marker = "recent_recovery_history="
    if marker not in reason:
        return []
    return [reason[reason.index(marker) + len(marker) :]]


def apply_ablation(payload: dict[str, Any], ablation: str) -> dict[str, Any]:
    ablated = copy.deepcopy(payload)
    ablated["ablation"] = ablation
    if ablation == "full":
        return ablated
    if ablation == "no_scene_memory":
        ablated["scene_memory"] = {}
    elif ablation == "no_observation_preprocess":
        ablated["observation_preprocess"] = {}
    elif ablation == "no_recovery_history":
        ablated["recovery_history"] = []
        ablated["retrieved_experience"] = {"lessons": [], "similar_cases": [], "avoid_patterns": []}
    elif ablation == "no_scene_and_preprocess":
        ablated["scene_memory"] = {}
        ablated["observation_preprocess"] = {}
    else:
        raise ValueError(f"Unsupported ablation: {ablation}")
    return ablated


def build_backend(args: argparse.Namespace) -> AgentApiRecoveryAdapter:
    return AgentApiRecoveryAdapter(
        AgentApiRecoveryConfig(
            server_url=args.server_url,
            timeout_sec=args.timeout_sec,
            prompt_template="{recovery_payload}",
        )
    )


def normalize_backend_output(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"valid": False, "error": "backend output is not a dict", "raw": payload}
    workflow = str(payload.get("recovery_workflow", "")).strip()
    intent = str(payload.get("post_recovery_intent", "")).strip().lower()
    calls = payload.get("tool_calls", [])
    errors: list[str] = []
    if not workflow:
        errors.append("missing recovery_workflow")
    if intent not in {"retry", "replan", "abort"}:
        errors.append("invalid post_recovery_intent")
    if not isinstance(calls, list):
        errors.append("tool_calls is not a list")
        calls = []
    tool_names: list[str] = []
    for item in calls:
        if not isinstance(item, dict):
            errors.append("tool call is not a dict")
            continue
        tool_name = str(item.get("tool_name", "")).strip()
        if tool_name:
            tool_names.append(tool_name)
        if tool_name not in RECOVERY_TOOLS:
            errors.append(f"unknown tool: {tool_name}")
        args = item.get("args", {})
        if args is not None and not isinstance(args, dict):
            errors.append(f"args for {tool_name or '<missing>'} is not a dict")
    return {
        "valid": not errors,
        "errors": errors,
        "recovery_workflow": workflow,
        "post_recovery_intent": intent,
        "tool_sequence": tool_names,
        "raw": payload,
    }


def build_ablation_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    registry = build_registry(args.skill_root)
    base_payloads: list[dict[str, Any]] = []
    for trace in args.trace:
        for record in read_jsonl(trace):
            payload = capture_payload_from_router(record, registry=registry)
            if payload is None:
                continue
            payload["_source_trace"] = str(trace)
            base_payloads.append(payload)
            if len(base_payloads) >= args.max_payloads:
                break
        if len(base_payloads) >= args.max_payloads:
            break

    backend = build_backend(args) if args.call_backend else None
    output_records: list[dict[str, Any]] = []
    for payload_index, payload in enumerate(base_payloads):
        for ablation in args.ablations:
            ablated = apply_ablation(payload, ablation)
            output: dict[str, Any] | None = None
            error = ""
            if backend is not None:
                try:
                    output = normalize_backend_output(backend.plan_recovery(recovery_payload=ablated))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    output = {"valid": False, "errors": [error], "raw": {}}
            output_records.append(
                {
                    "payload_index": payload_index,
                    "ablation": ablation,
                    "source_trace": payload.get("_source_trace", ""),
                    "trace_router": dict(payload.get("_trace_router", {}) or {}),
                    "payload": ablated,
                    "backend_called": backend is not None,
                    "backend_output": output,
                    "error": error,
                }
            )
    return output_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_ablation: dict[str, dict[str, Any]] = {}
    for record in records:
        ablation = str(record.get("ablation", "unknown"))
        bucket = per_ablation.setdefault(ablation, {"records": 0, "backend_called": 0, "valid": 0, "errors": 0})
        bucket["records"] += 1
        if record.get("backend_called"):
            bucket["backend_called"] += 1
        output = record.get("backend_output")
        if isinstance(output, dict):
            if output.get("valid"):
                bucket["valid"] += 1
            else:
                bucket["errors"] += 1
    return {
        "records": len(records),
        "payload_groups": len({record.get("payload_index") for record in records}),
        "ablations": per_ablation,
    }


def main() -> None:
    args = parse_args()
    records = build_ablation_records(args)
    write_jsonl(args.output, records)
    summary = summarize(records)
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, sort_keys=True, indent=2))
    if args.strict:
        failures = [
            record
            for record in records
            if record.get("backend_called")
            and isinstance(record.get("backend_output"), dict)
            and not record["backend_output"].get("valid")
        ]
        if failures:
            raise SystemExit(f"{len(failures)} backend outputs failed validation")


if __name__ == "__main__":
    main()

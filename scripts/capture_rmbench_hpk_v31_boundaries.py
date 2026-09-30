#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_rmbench_hpk_v31_gate_with_services import (  # noqa: E402
    UtilityGateServiceError,
    _health,
    _start,
    _stop,
    _wait_health,
    agent_service_command,
    sam3_service_command,
)
from scripts.run_rmbench_hpk_v31_utility_gate import _launcher_environment  # noqa: E402
from scripts.summarize_rmbench_hpk_v31_usage import summarize  # noqa: E402
from roboharn_evo.agent.hpk.hierarchical_retriever import (  # noqa: E402
    AgentApiHierarchicalRetrievalBackend,
    VLMSubtaskKnowledgeRetriever,
    build_subtask_knowledge_query,
)
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store  # noqa: E402
from roboharn_evo.agent.failure_boundary_replay import load_failure_boundary  # noqa: E402
from roboharn_evo.benchmark_adapters.rmbench.v31_boundary_bank import (  # noqa: E402
    classify_failure_boundary_v31,
    recoverable_failure_boundary_from_agent_payload_v31,
    validate_failure_category_coverage_v31,
)


CAPTURE_PLAN_SCHEMA = "roboharn_evo/rmbench/v31/boundary_capture_plan"
CAPTURE_RESULT_SCHEMA = "roboharn_evo/rmbench/v31/boundary_capture_result"
BOUNDARY_SPEC_SCHEMA = "roboharn_evo/rmbench/v31/utility_boundary_spec"


class BoundaryCaptureError(RuntimeError):
    """The real boundary capture protocol could not be completed."""


@contextmanager
def _loopback_proxy_bypass():
    """Keep local Agent requests off the host HTTP proxy."""

    previous = {key: os.environ.get(key) for key in ("NO_PROXY", "no_proxy")}
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise BoundaryCaptureError(
            f"{root} must contain exactly one {pattern}; found {len(matches)}"
        )
    return matches[0]


def _seeds(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item or not item.isdigit():
            raise BoundaryCaptureError(
                "seeds must be comma-separated non-negative integers"
            )
        value = int(item)
        if value in values:
            raise BoundaryCaptureError("capture seeds must be unique")
        values.append(value)
    if not values or len(values) > 20:
        raise BoundaryCaptureError("capture requires 1 to 20 explicit seeds")
    return tuple(values)


def _artifact_dir(seed_root: Path, seed: int) -> Path:
    return seed_root / "segmentation_artifacts" / f"worker_0_gpu_0_seed_0_e{seed}"


def _validated_capture_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryCaptureError("cannot read boundary-bank result") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") not in {"roboharn_evo/rmbench_failure_boundary_bank_result/v1", "tcm/rmbench_failure_boundary_bank_result/v1"}
        or payload.get("mode") != "capture_bank"
    ):
        raise BoundaryCaptureError("boundary-bank result schema mismatch")
    raw_boundaries = payload.get("boundaries")
    if not isinstance(raw_boundaries, list) or not raw_boundaries:
        raise BoundaryCaptureError("capture produced no confirmed-holding boundary")
    boundaries: list[dict[str, Any]] = []
    for raw in raw_boundaries:
        if not isinstance(raw, Mapping):
            raise BoundaryCaptureError("captured boundary summary is malformed")
        path_value = raw.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise BoundaryCaptureError("captured boundary path is missing")
        boundary_path = Path(path_value).resolve(strict=True)
        boundary = load_failure_boundary(boundary_path)
        failure_boundary = recoverable_failure_boundary_from_agent_payload_v31(
            boundary.agent
        )
        if failure_boundary is None:
            raise BoundaryCaptureError(
                "captured state is not one recoverable strict-hold failure boundary"
            )
        holding, failure_evidence = failure_boundary
        boundaries.append(
            {
                "path": str(boundary_path),
                "seed": boundary.seed,
                "instruction": boundary.instruction,
                "env_step": int(
                    boundary.environment["task_state"].get("take_action_cnt", 0)
                ),
                "holding_boundary": holding.to_private_dict(),
                "failure_evidence": failure_evidence,
            }
        )
    return {
        "task": payload.get("task"),
        "seed": payload.get("seed"),
        "instruction": payload.get("instruction"),
        "boundaries": boundaries,
        "outcome": payload.get("outcome"),
    }


def _target_references(value: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for key in (
        "target_id",
        "support_instance_id",
        "reference_object_id",
        "reference_region_id",
    ):
        ref = str(value.get(key, "") or "").strip()
        if ref and ref not in result:
            result.append(ref)
    raw = value.get("reference_instance_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            ref = str(item or "").strip()
            if ref and ref not in result:
                result.append(ref)
    return tuple(result)


def _resolve_unique_runtime_target(
    scene_memory: Mapping[str, Any],
    *,
    held_instance_ref: str,
) -> tuple[str, str] | None:
    raw_targets = scene_memory.get("operation_targets")
    targets = (
        [value for value in raw_targets if isinstance(value, Mapping)]
        if isinstance(raw_targets, list)
        else []
    )
    eligible = [
        value
        for value in targets
        if str(value.get("action_mode", "") or "").strip().casefold() == "place"
        and str(value.get("held_instance_id", "") or "").strip() == held_instance_ref
        and value.get("support_valid") is True
        and value.get("free") is True
        and str(value.get("target_id", "") or "").strip()
    ]
    focus = scene_memory.get("task_focus")
    focus_refs: list[str] = []
    if isinstance(focus, Mapping):
        for field in ("target_instances", "context_instances"):
            values = focus.get(field)
            if isinstance(values, list):
                for value in values:
                    ref = str(value or "").strip()
                    if ref and ref != held_instance_ref and ref not in focus_refs:
                        focus_refs.append(ref)
    matches: list[tuple[str, str]] = []
    for target in eligible:
        target_id = str(target["target_id"]).strip()
        references = _target_references(target)
        for ref in focus_refs:
            if ref in references and (ref, target_id) not in matches:
                matches.append((ref, target_id))
    if len({target for _, target in matches}) == 1:
        return matches[0]
    if len(eligible) != 1:
        return None
    target = eligible[0]
    target_id = str(target["target_id"]).strip()
    semantic_refs = [
        value
        for value in _target_references(target)
        if value not in {target_id, held_instance_ref}
    ]
    return (semantic_refs[0] if len(semantic_refs) == 1 else target_id, target_id)


def _active_skill(value: Mapping[str, Any]) -> SimpleNamespace | None:
    raw = value.get("active_skill")
    if not isinstance(raw, Mapping):
        return None
    return SimpleNamespace(**dict(raw))


def build_boundary_specs(
    boundaries: Sequence[Mapping[str, Any]],
    *,
    store_root: Path,
    agent_url: str,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_knowledge, action_knowledge = load_hierarchical_store(store_root)
    place_indices = [
        index
        for index, value in enumerate(action_knowledge)
        if value["status"] == "supported" and value["condition"]["action"] == "place"
    ]
    if len(place_indices) != 1:
        raise BoundaryCaptureError(
            "the frozen Store must contain exactly one supported place Action Knowledge record"
        )
    retriever = VLMSubtaskKnowledgeRetriever(
        AgentApiHierarchicalRetrievalBackend(
            agent_url.rstrip("/") + "/plan",
            timeout_sec=600,
        )
    )
    specs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for ordinal, summary in enumerate(boundaries):
        boundary_path = Path(str(summary["path"])).resolve(strict=True)
        boundary = load_failure_boundary(boundary_path)
        failure = recoverable_failure_boundary_from_agent_payload_v31(boundary.agent)
        if failure is None:
            raise BoundaryCaptureError("captured boundary lost its failure predicate")
        holding, failure_evidence = failure
        memory_state = boundary.agent.get("memory_state")
        working = (
            memory_state.get("working") if isinstance(memory_state, Mapping) else None
        )
        scene_memory = (
            working.get("scene_memory") if isinstance(working, Mapping) else None
        )
        if not isinstance(scene_memory, Mapping):
            raise BoundaryCaptureError("captured boundary has no Scene Memory")
        active_skill = _active_skill(memory_state)
        baseline = (
            str(getattr(active_skill, "instruction", "") or "").strip()
            if active_skill is not None
            else ""
        )
        if not baseline:
            audits.append(
                {
                    "boundary_path": str(boundary_path),
                    "accepted": False,
                    "reason": "active skill has no current subtask",
                }
            )
            continue
        query = build_subtask_knowledge_query(
            overall_goal=boundary.instruction,
            scene_memory=scene_memory,
            active_skill=active_skill,
        )
        selected = retriever.retrieve(
            task_knowledge,
            query,
            baseline_subtask=baseline,
        )
        if selected is None:
            audits.append(
                {
                    "boundary_path": str(boundary_path),
                    "accepted": False,
                    "reason": "VLM selected no supported Task Knowledge",
                    "retrieval_audit": retriever.last_call_audit,
                }
            )
            continue
        task_index = next(
            (
                index
                for index, value in enumerate(task_knowledge)
                if value.to_dict() == selected.knowledge.to_dict()
            ),
            None,
        )
        target = _resolve_unique_runtime_target(
            scene_memory,
            held_instance_ref=holding.held_instance_ref,
        )
        if task_index is None or target is None:
            audits.append(
                {
                    "boundary_path": str(boundary_path),
                    "accepted": False,
                    "reason": (
                        "selected Task Knowledge or semantic Runtime target is not unique"
                    ),
                    "retrieval_audit": retriever.last_call_audit,
                }
            )
            continue
        semantic_target_ref, runtime_target_ref = target
        category = classify_failure_boundary_v31(
            boundary.agent,
            held_instance_ref=holding.held_instance_ref,
            expected_target_ref=runtime_target_ref,
        )
        boundary_id = f"boundary-{ordinal:02d}"
        spec = {
            "schema": BOUNDARY_SPEC_SCHEMA,
            "boundary_id": boundary_id,
            "category": category,
            "boundary_path": str(boundary_path),
            "task_knowledge_index": task_index,
            "action_knowledge_index": place_indices[0],
            "required_target_scene_ref": semantic_target_ref,
            "expected_runtime_target_ref": runtime_target_ref,
        }
        specs.append(spec)
        audits.append(
            {
                "boundary_path": str(boundary_path),
                "accepted": True,
                "boundary_id": boundary_id,
                "failure_evidence": failure_evidence,
                "task_knowledge_index": task_index,
                "grounded_subtask": selected.grounded_subtask,
                "retrieval_reason": selected.reason,
                "retrieval_audit": retriever.last_call_audit,
                "runtime_target_binding": {
                    "required_target_scene_ref": semantic_target_ref,
                    "expected_runtime_target_ref": runtime_target_ref,
                },
                "category": category,
            }
        )
    specs_path = output_root / "boundary_specs.jsonl"
    with specs_path.open("x", encoding="utf-8") as stream:
        for value in specs:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(
        output_root / "boundary_spec_audit.json",
        {"accepted_count": len(specs), "records": audits},
    )
    return specs, audits


def capture(
    *,
    output_root: Path,
    seeds: Sequence[int],
    task: str,
    task_config: str,
    instruction_set: str,
    store_root: Path,
    launcher: Path,
    assets_root: Path,
    gpu: str,
    agent_port: int,
    sam3_port: int,
    rmbench_python: Path,
    sam3_python: Path,
    codex_bin: Path,
    codex_auth_file: Path,
    codex_config_file: Path,
    sam3_repo: Path,
    checkpoint: Path,
    bpe_path: Path,
    max_boundaries_per_seed: int,
    minimum_total_boundaries: int,
    environment_action_budget: int = 150,
    control_turn_budget: int = 64,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise BoundaryCaptureError("capture output root already exists")
    if (
        isinstance(max_boundaries_per_seed, bool)
        or not 1 <= max_boundaries_per_seed <= 20
        or isinstance(minimum_total_boundaries, bool)
        or not 1 <= minimum_total_boundaries <= 20
    ):
        raise BoundaryCaptureError("boundary count limits must be between 1 and 20")
    project_root = Path(__file__).resolve().parents[1]
    contract = {
        "task": str(task).strip(),
        "task_config": str(task_config).strip(),
        "instruction_set": str(instruction_set).strip(),
        "agent_model": "gpt-5.5",
        "reasoning_effort": "xhigh",
        "environment_action_budget": environment_action_budget,
        "control_turn_budget": control_turn_budget,
    }
    if not all(contract[key] for key in ("task", "task_config", "instruction_set")):
        raise BoundaryCaptureError(
            "task, task config, and instruction set are required"
        )
    for label, value in (
        ("environment_action_budget", environment_action_budget),
        ("control_turn_budget", control_turn_budget),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BoundaryCaptureError(f"{label} must be a positive integer")
    plan = {
        "schema": CAPTURE_PLAN_SCHEMA,
        "seeds": list(seeds),
        "task": contract["task"],
        "task_config": contract["task_config"],
        "instruction_set": contract["instruction_set"],
        "store_root": str(store_root),
        "grasp_transport_policy": "strict",
        "max_boundaries_per_seed": max_boundaries_per_seed,
        "minimum_total_boundaries": minimum_total_boundaries,
        "environment_action_budget": environment_action_budget,
        "control_turn_budget": control_turn_budget,
        "retry_budget": 0,
        "seed_replacement": False,
    }
    agent_url = f"http://127.0.0.1:{agent_port}"
    sam3_url = f"http://127.0.0.1:{sam3_port}"
    if _health(agent_url) is not None or _health(sam3_url) is not None:
        raise BoundaryCaptureError("exclusive capture service ports are already in use")
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "capture_plan.json", plan)
    service_root = output_root / "services"
    agent_runtime = service_root / "agent/runtime"
    agent_runtime.mkdir(parents=True)
    agent_log = service_root / "agent/service.log"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(project_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    agent_process: subprocess.Popen[bytes] | None = None
    agent_stream: BinaryIO | None = None
    seed_results: list[dict[str, Any]] = []
    all_boundaries: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    spec_audits: list[dict[str, Any]] = []
    try:
        agent_command = agent_service_command(
            python=rmbench_python,
            project_root=project_root,
            port=agent_port,
            contract=contract,
            codex_bin=codex_bin,
            codex_auth_file=codex_auth_file,
            codex_config_file=codex_config_file,
            runtime_root=agent_runtime,
        )
        agent_process, agent_stream = _start(
            agent_command,
            cwd=project_root,
            environment=environment,
            log_path=agent_log,
        )
        agent_health = _wait_health(agent_process, agent_url, timeout_sec=60)
        _write_json(
            service_root / "agent/launch.json",
            {"command": agent_command, "health": agent_health},
        )
        frozen_instruction: str | None = None
        for seed in seeds:
            seed_root = output_root / "seeds" / f"seed_{seed:06d}"
            boundary_root = seed_root / "boundaries"
            boundary_root.mkdir(parents=True)
            artifact_dir = _artifact_dir(seed_root, seed)
            (artifact_dir / "inputs").mkdir(parents=True)
            (artifact_dir / "masks").mkdir(parents=True)
            sam_service_root = seed_root / "sam3_service"
            sam_cache = service_root / "sam3_cache"
            sam_cache.mkdir(exist_ok=True)
            sam_environment = dict(environment)
            sam_environment.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            sam_command = sam3_service_command(
                python=sam3_python,
                project_root=project_root,
                port=sam3_port,
                sam3_repo=sam3_repo,
                checkpoint=checkpoint,
                bpe_path=bpe_path,
                cache_root=sam_cache,
                artifact_dir=artifact_dir,
                instance_id=f"hpk-v31-boundary-capture-seed-{seed}",
            )
            sam_process: subprocess.Popen[bytes] | None = None
            sam_stream: BinaryIO | None = None
            try:
                sam_process, sam_stream = _start(
                    sam_command,
                    cwd=project_root,
                    environment=sam_environment,
                    log_path=sam_service_root / "service.log",
                )
                sam_health = _wait_health(sam_process, sam3_url, timeout_sec=300)
                _write_json(
                    sam_service_root / "launch.json",
                    {"command": sam_command, "health": sam_health},
                )
                cell = {
                    "cell_root": seed_root,
                    "boundary_id": f"capture-seed-{seed}",
                    "condition": "c0",
                    "seed": seed,
                }
                launcher_environment = _launcher_environment(
                    cell=cell,
                    contract=contract,
                    assets_root=assets_root,
                    agent_api_base_url=agent_url,
                    sam3_service_url=sam3_url,
                    gpu=gpu,
                )
                before = agent_log.stat().st_size
                command = [
                    str(launcher),
                    "--eval.exact_seed_fail_closed",
                    "True",
                    "--eval.step_limit",
                    str(environment_action_budget),
                    "--agent.hpk.mode",
                    "off",
                    "--agent.hpk_v3.mode",
                    "off",
                    "--agent.recovery.grasp_transport_policy",
                    "strict",
                    "--agent.failure_boundary_replay.mode",
                    "capture_bank",
                    "--agent.failure_boundary_replay.output_dir",
                    str(boundary_root),
                    "--agent.failure_boundary_replay.max_boundaries",
                    str(max_boundaries_per_seed),
                    "--agent.failure_boundary_replay.min_boundaries",
                    "1",
                    "--agent.failure_boundary_replay.capture_at_or_after_env_step",
                    "0",
                    "--agent.failure_boundary_replay.stop_after_bank_full",
                    "False",
                    "--agent.failure_boundary_replay.label_prefix",
                    "strict confirmed holding boundary for v3.1 utility gate",
                ]
                completed = subprocess.run(
                    command,
                    cwd=launcher.parent.parent.parent,
                    env=launcher_environment,
                    check=False,
                )
                audit_window = seed_root / "agent_api_audit_window.log"
                with agent_log.open("rb") as stream:
                    stream.seek(before)
                    audit_window.write_bytes(stream.read())
                if completed.returncode != 0:
                    raise BoundaryCaptureError(
                        f"seed {seed} capture failed with exit {completed.returncode}"
                    )
                run_output = seed_root / "run_output"
                summary = _validated_capture_summary(
                    _one(run_output, "failure_boundary_result.json")
                )
                trace = _one(run_output, "episode_*_agent_trace.jsonl")
                usage = summarize(
                    audit_window,
                    trace,
                    expected_model="gpt-5.5",
                    expected_reasoning_effort="xhigh",
                )
                _write_json(seed_root / "usage.json", usage)
                if frozen_instruction is None:
                    frozen_instruction = str(summary["instruction"])
                elif summary["instruction"] != frozen_instruction:
                    raise BoundaryCaptureError(
                        "capture seeds did not use one frozen instruction"
                    )
                seed_results.append(summary)
                all_boundaries.extend(summary["boundaries"])
            finally:
                _stop(sam_process)
                if sam_stream is not None:
                    sam_stream.close()
        if not minimum_total_boundaries <= len(all_boundaries) <= 20:
            raise BoundaryCaptureError(
                "capture did not produce the preregistered 12–20 boundary set: "
                f"captured={len(all_boundaries)}, minimum={minimum_total_boundaries}"
            )
        preregistration_log_start = agent_log.stat().st_size
        with _loopback_proxy_bypass():
            specs, spec_audits = build_boundary_specs(
                all_boundaries,
                store_root=store_root,
                agent_url=agent_url,
                output_root=output_root,
            )
        with agent_log.open("rb") as stream:
            stream.seek(preregistration_log_start)
            (output_root / "boundary_spec_agent_api_audit.log").write_bytes(
                stream.read()
            )
    except (UtilityGateServiceError, OSError, subprocess.SubprocessError) as exc:
        raise BoundaryCaptureError(str(exc)) from exc
    finally:
        _stop(agent_process)
        if agent_stream is not None:
            agent_stream.close()
    if len(specs) < minimum_total_boundaries:
        raise BoundaryCaptureError(
            "fewer than the preregistered minimum boundaries have both a VLM-selected "
            "Task Knowledge record and one unambiguous Runtime target: "
            f"accepted={len(specs)}, minimum={minimum_total_boundaries}"
        )
    if minimum_total_boundaries >= 12:
        try:
            validate_failure_category_coverage_v31(
                value.get("category") for value in specs
            )
        except ValueError as exc:
            raise BoundaryCaptureError(str(exc)) from exc
    result = {
        "schema": CAPTURE_RESULT_SCHEMA,
        "capture_plan": str(output_root / "capture_plan.json"),
        "instruction": frozen_instruction,
        "boundary_count": len(all_boundaries),
        "preparable_boundary_count": len(specs),
        "boundary_specs_path": str(output_root / "boundary_specs.jsonl"),
        "boundaries": all_boundaries,
        "boundary_spec_audits": spec_audits,
        "seed_results": seed_results,
        "automatic_retry": False,
        "seed_replacement": False,
    }
    _write_json(output_root / "capture_result.json", result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--instruction-set", required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--agent-port", type=int, default=19114)
    parser.add_argument("--sam3-port", type=int, default=19314)
    parser.add_argument(
        "--rmbench-python",
        type=Path,
        default=Path("/path/to/simulator/environment/bin/python"),
    )
    parser.add_argument(
        "--sam3-python",
        type=Path,
        default=Path("/path/to/SAM3/environment/bin/python"),
    )
    parser.add_argument("--codex-bin", type=Path, default=Path("/usr/bin/codex"))
    parser.add_argument(
        "--codex-auth-file", type=Path, default=Path("/path/to/provider/auth.json")
    )
    parser.add_argument(
        "--codex-config-file", type=Path, default=Path("/path/to/provider/config.toml")
    )
    parser.add_argument("--sam3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe-path", type=Path, required=True)
    parser.add_argument("--max-boundaries-per-seed", type=int, default=4)
    parser.add_argument("--minimum-total-boundaries", type=int, default=12)
    parser.add_argument("--environment-action-budget", type=int, default=150)
    parser.add_argument("--control-turn-budget", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = capture(
        output_root=args.output_root.absolute(),
        seeds=_seeds(args.seeds),
        task=args.task,
        task_config=args.task_config,
        instruction_set=args.instruction_set,
        store_root=args.store_root.resolve(strict=True),
        launcher=args.launcher.resolve(strict=True),
        assets_root=args.assets_root.resolve(strict=True),
        gpu=str(args.gpu).strip(),
        agent_port=args.agent_port,
        sam3_port=args.sam3_port,
        rmbench_python=args.rmbench_python.resolve(strict=True),
        sam3_python=args.sam3_python.resolve(strict=True),
        codex_bin=args.codex_bin.resolve(strict=True),
        codex_auth_file=args.codex_auth_file.resolve(strict=True),
        codex_config_file=args.codex_config_file.resolve(strict=True),
        sam3_repo=args.sam3_repo.resolve(strict=True),
        checkpoint=args.checkpoint.resolve(strict=True),
        bpe_path=args.bpe_path.resolve(strict=True),
        max_boundaries_per_seed=args.max_boundaries_per_seed,
        minimum_total_boundaries=args.minimum_total_boundaries,
        environment_action_budget=args.environment_action_budget,
        control_turn_budget=args.control_turn_budget,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BoundaryCaptureError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

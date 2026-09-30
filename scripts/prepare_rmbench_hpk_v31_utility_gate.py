#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from roboharn_evo.agent.failure_boundary_replay import load_failure_boundary  # noqa: E402
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store  # noqa: E402
from roboharn_evo.benchmark_adapters.rmbench.v31_boundary_bank import (  # noqa: E402
    classify_failure_boundary_v31,
    recoverable_failure_boundary_from_agent_payload_v31,
    validate_failure_category_coverage_v31,
)
from roboharn_evo.benchmark_adapters.rmbench.v31_goal_bridge import (  # noqa: E402
    RMBenchFrozenGoalContextV31,
    RMBenchGoalContextBridgeV31,
    build_rmbench_v31_goal_context_from_knowledge,
)


SPEC_SCHEMA = "roboharn_evo/rmbench/v31/utility_boundary_spec"
MANIFEST_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_manifest"
CELL_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_cell"
CONDITIONS = ("c0", "c1", "c2")
CATEGORIES = frozenset({"goal_inconsistency", "realization_failure"})
_BOUNDARY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")


class UtilityGatePreparationError(RuntimeError):
    """The boundary set cannot support the preregistered experiment."""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UtilityGatePreparationError(
                f"{path}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise UtilityGatePreparationError(
                f"{path}:{line_number} must contain one JSON object"
            )
        records.append(value)
    if not records:
        raise UtilityGatePreparationError("boundary spec JSONL is empty")
    return records


def _path(base: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise UtilityGatePreparationError(f"{label} must be an explicit path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise UtilityGatePreparationError(f"{label} does not exist: {path}") from exc


def _strict_spec(value: Mapping[str, Any], base: Path) -> dict[str, Any]:
    expected = {
        "schema",
        "boundary_id",
        "category",
        "boundary_path",
        "task_knowledge_index",
        "action_knowledge_index",
        "required_target_scene_ref",
        "expected_runtime_target_ref",
    }
    if set(value) != expected or value.get("schema") not in {SPEC_SCHEMA, "tcm/rmbench/v31/utility_boundary_spec"}:
        raise UtilityGatePreparationError(
            "each boundary spec must use the exact v3.1 field set"
        )
    boundary_id = str(value.get("boundary_id", "") or "").strip()
    category = str(value.get("category", "") or "").strip()
    target_ref = str(value.get("expected_runtime_target_ref", "") or "").strip()
    if _BOUNDARY_ID.fullmatch(boundary_id) is None:
        raise UtilityGatePreparationError(
            "boundary_id must be a bounded filesystem-safe identifier"
        )
    if category not in CATEGORIES:
        raise UtilityGatePreparationError(f"unsupported boundary category: {category}")
    if not target_ref or len(target_ref) > 1000:
        raise UtilityGatePreparationError(
            "expected_runtime_target_ref must be bounded and non-empty"
        )
    semantic_target_ref = str(value.get("required_target_scene_ref", "") or "").strip()
    if not semantic_target_ref or len(semantic_target_ref) > 1000:
        raise UtilityGatePreparationError(
            "required_target_scene_ref must be bounded and non-empty"
        )
    indices: dict[str, int] = {}
    for field in ("task_knowledge_index", "action_knowledge_index"):
        index = value.get(field)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise UtilityGatePreparationError(f"{field} must be non-negative")
        indices[field] = index
    return {
        "boundary_id": boundary_id,
        "category": category,
        "boundary_path": _path(base, value["boundary_path"], label="boundary_path"),
        **indices,
        "required_target_scene_ref": semantic_target_ref,
        "expected_runtime_target_ref": target_ref,
    }


def _validate_boundary(
    spec: Mapping[str, Any],
    *,
    task_knowledge: Sequence[Any],
    action_knowledge: Sequence[Any],
) -> tuple[dict[str, Any], RMBenchFrozenGoalContextV31]:
    boundary_path = Path(spec["boundary_path"])
    boundary = load_failure_boundary(boundary_path)
    failure_boundary = recoverable_failure_boundary_from_agent_payload_v31(
        boundary.agent
    )
    if failure_boundary is None:
        raise UtilityGatePreparationError(
            f"{boundary_path} is not one recoverable strict-hold failure state"
        )
    holding, failure_evidence = failure_boundary
    observed_category = classify_failure_boundary_v31(
        boundary.agent,
        held_instance_ref=holding.held_instance_ref,
        expected_target_ref=spec["expected_runtime_target_ref"],
    )
    if spec["category"] != observed_category:
        raise UtilityGatePreparationError(
            "preregistered category differs from captured Runtime failure facts"
        )
    memory_state = boundary.agent.get("memory_state")
    working = memory_state.get("working") if isinstance(memory_state, Mapping) else None
    scene_memory = working.get("scene_memory") if isinstance(working, Mapping) else None
    if not isinstance(scene_memory, Mapping):
        raise UtilityGatePreparationError("captured boundary has no Scene Memory")
    task_index = int(spec["task_knowledge_index"])
    action_index = int(spec["action_knowledge_index"])
    if task_index >= len(task_knowledge) or action_index >= len(action_knowledge):
        raise UtilityGatePreparationError("knowledge index is outside the frozen Store")
    try:
        context = build_rmbench_v31_goal_context_from_knowledge(
            task_knowledge=task_knowledge[task_index],
            action_knowledge=action_knowledge[action_index],
            scene_memory=scene_memory,
            manipulated_object_ref=holding.held_instance_ref,
            required_target_scene_ref=spec["required_target_scene_ref"],
            expected_runtime_target_ref=spec["expected_runtime_target_ref"],
            local_trace_ref=holding.active_skill_ref,
        )
    except Exception as exc:  # noqa: BLE001 - normalize preregistration errors
        raise UtilityGatePreparationError(
            "cannot derive a Goal Context from the selected frozen knowledge"
        ) from exc
    prepared = RMBenchGoalContextBridgeV31(context).prepare(
        scene_memory=scene_memory,
        active_skill_ref=holding.active_skill_ref,
    )
    expected_target = str(spec["expected_runtime_target_ref"])
    if (
        not prepared.installed
        or prepared.runtime_binding.required_target_ref != expected_target
    ):
        raise UtilityGatePreparationError(
            "Goal Context does not resolve to the preregistered Runtime target"
        )
    return {
        "task": boundary.task,
        "seed": boundary.seed,
        "instruction": boundary.instruction,
        "boundary_env_step": int(
            boundary.environment["task_state"].get("take_action_cnt", 0)
        ),
        "holding_boundary": holding.to_private_dict(),
        "failure_evidence": failure_evidence,
        "observed_category": observed_category,
        "task_knowledge_index": task_index,
        "action_knowledge_index": action_index,
        "selected_task_knowledge": task_knowledge[task_index].to_dict(),
        "selected_action_knowledge": action_knowledge[action_index].to_dict(),
        "goal_contract": context.goal_contract.to_dict(),
        "runtime_binding": prepared.runtime_binding.to_private_dict(),
    }, context


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare(
    *,
    specs_path: Path,
    store_root: Path,
    output_root: Path,
    runtime_commit: str,
    agent_model: str,
    reasoning_effort: str,
    decoding: Mapping[str, Any],
    task_config: str,
    instruction_set: str,
    environment_action_budget: int = 150,
    control_turn_budget: int = 64,
    require_complete: bool = True,
) -> dict[str, Any]:
    specs_path = specs_path.resolve(strict=True)
    store_root = store_root.resolve(strict=True)
    try:
        task_knowledge, action_knowledge = load_hierarchical_store(store_root)
    except Exception as exc:  # noqa: BLE001 - normalize Store preflight failures
        raise UtilityGatePreparationError("cannot load the frozen v3 Store") from exc
    if not task_knowledge or not action_knowledge:
        raise UtilityGatePreparationError(
            "the utility gate requires non-empty Task and Action Knowledge"
        )
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise UtilityGatePreparationError(f"refusing to overwrite {output_root}")
    raw_specs = _jsonl(specs_path)
    if len(raw_specs) > 20 or (require_complete and len(raw_specs) < 12):
        raise UtilityGatePreparationError(
            "the final utility gate requires 12 to 20 boundary specs"
        )
    if not raw_specs:
        raise UtilityGatePreparationError("at least one boundary spec is required")
    if (
        isinstance(environment_action_budget, bool)
        or not isinstance(environment_action_budget, int)
        or environment_action_budget <= 0
        or isinstance(control_turn_budget, bool)
        or not isinstance(control_turn_budget, int)
        or control_turn_budget <= 0
    ):
        raise UtilityGatePreparationError(
            "experiment budgets must be positive integers"
        )
    runtime_commit = str(runtime_commit or "").strip()
    agent_model = str(agent_model or "").strip()
    reasoning_effort = str(reasoning_effort or "").strip()
    task_config = str(task_config or "").strip()
    instruction_set = str(instruction_set or "").strip()
    if not all(
        (
            runtime_commit,
            agent_model,
            reasoning_effort,
            task_config,
            instruction_set,
        )
    ):
        raise UtilityGatePreparationError(
            "runtime commit, Agent model, reasoning effort, task config, and "
            "instruction set must be explicit"
        )
    if not isinstance(decoding, Mapping) or not decoding:
        raise UtilityGatePreparationError("decoding config must be non-empty")

    validated: list[
        tuple[dict[str, Any], dict[str, Any], RMBenchFrozenGoalContextV31]
    ] = []
    seen_ids: set[str] = set()
    seen_boundaries: set[Path] = set()
    identity: tuple[str, str] | None = None
    for raw in raw_specs:
        spec = _strict_spec(raw, specs_path.parent)
        if spec["boundary_id"] in seen_ids or spec["boundary_path"] in seen_boundaries:
            raise UtilityGatePreparationError("boundary specs must be unique")
        seen_ids.add(spec["boundary_id"])
        seen_boundaries.add(spec["boundary_path"])
        audit, context = _validate_boundary(
            spec,
            task_knowledge=task_knowledge,
            action_knowledge=action_knowledge,
        )
        current_identity = (audit["task"], audit["instruction"])
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise UtilityGatePreparationError(
                "all boundaries must use the same task and frozen instruction"
            )
        validated.append((spec, audit, context))

    if require_complete:
        try:
            validate_failure_category_coverage_v31(
                spec["category"] for spec, _audit, _context in validated
            )
        except ValueError as exc:
            raise UtilityGatePreparationError(str(exc)) from exc

    common = {
        "task": identity[0] if identity is not None else "",
        "shared_core_baseline": "8276000",
        "runtime_commit": runtime_commit,
        "store_root": str(store_root),
        "agent_model": agent_model,
        "reasoning_effort": reasoning_effort,
        "decoding": dict(decoding),
        "task_config": task_config,
        "instruction_set": instruction_set,
        "grasp_transport_policy": "strict",
        "environment_action_budget": environment_action_budget,
        "control_turn_budget": control_turn_budget,
    }
    manifest_boundaries: list[dict[str, Any]] = []
    run_cells: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=False)
    for spec, audit, context in validated:
        goal_context_path = (
            output_root / "preregistered_goal_contexts" / f"{spec['boundary_id']}.json"
        )
        _write_json(goal_context_path, context.to_private_dict())
        cells: dict[str, str] = {}
        for condition in CONDITIONS:
            cell_root = output_root / "cells" / spec["boundary_id"] / condition
            cells[condition] = str(cell_root)
            cell = {
                "schema": CELL_SCHEMA,
                "condition": condition,
                "boundary_id": spec["boundary_id"],
                "boundary_path": str(spec["boundary_path"]),
                "goal_context_path": str(goal_context_path),
                "hpk_v3_mode": "off" if condition == "c0" else "full",
                "hpk_goal_consistency_enabled": condition == "c2",
                "store_writeback_enabled": False,
                "retry_budget": 0,
                **common,
            }
            _write_json(cell_root / "condition.json", cell)
            overrides = [
                "--agent.hpk.mode",
                "off",
                "--agent.hpk_v3.mode",
                cell["hpk_v3_mode"],
                "--agent.failure_boundary_replay.mode",
                "restore",
                "--agent.failure_boundary_replay.path",
                str(spec["boundary_path"]),
                "--agent.recovery.grasp_transport_policy",
                "strict",
            ]
            if condition != "c0":
                overrides.extend(
                    (
                        "--agent.hpk_v3.store_root",
                        str(store_root),
                        "--agent.hpk_v3.hpk_goal_consistency_enabled",
                        str(condition == "c2"),
                        "--agent.hpk_v3.rmbench_frozen_goal_context",
                        str(goal_context_path),
                        "--agent.hpk_v3.rmbench_read_only_utility_gate",
                        "True",
                    )
                )
            run_cells.append(
                {
                    "boundary_id": spec["boundary_id"],
                    "condition": condition,
                    "seed": audit["seed"],
                    "cell_root": str(cell_root),
                    "overrides": overrides,
                }
            )
        manifest_boundaries.append(
            {
                "boundary_id": spec["boundary_id"],
                "category": spec["category"],
                "boundary_path": str(spec["boundary_path"]),
                "goal_context_path": str(goal_context_path),
                "task_knowledge_index": spec["task_knowledge_index"],
                "action_knowledge_index": spec["action_knowledge_index"],
                "required_target_scene_ref": spec["required_target_scene_ref"],
                "expected_runtime_target_ref": spec["expected_runtime_target_ref"],
                "cells": cells,
            }
        )
        _write_json(
            output_root / "boundary_audits" / f"{spec['boundary_id']}.json",
            audit,
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        **common,
        "boundaries": manifest_boundaries,
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(
        output_root / "run_plan.json",
        {
            "schema": "roboharn_evo/rmbench/v31/utility_gate_run_plan",
            "cell_count": len(run_cells),
            "conditions": list(CONDITIONS),
            "automatic_retry": False,
            "experiment_contract": {
                **common,
            },
            "cells": run_cells,
        },
    )
    return {
        "output_root": str(output_root),
        "manifest_path": str(output_root / "manifest.json"),
        "run_plan_path": str(output_root / "run_plan.json"),
        "boundary_count": len(validated),
        "cell_count": len(run_cells),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-specs", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--instruction-set", required=True)
    parser.add_argument("--decoding-json", type=Path, required=True)
    parser.add_argument("--environment-action-budget", type=int, default=150)
    parser.add_argument("--control-turn-budget", type=int, default=64)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    decoding = json.loads(args.decoding_json.read_text(encoding="utf-8"))
    result = prepare(
        specs_path=args.boundary_specs,
        store_root=args.store_root,
        output_root=args.output_root,
        runtime_commit=args.runtime_commit,
        agent_model=args.agent_model,
        reasoning_effort=args.reasoning_effort,
        decoding=decoding,
        task_config=args.task_config,
        instruction_set=args.instruction_set,
        environment_action_budget=args.environment_action_budget,
        control_turn_budget=args.control_turn_budget,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

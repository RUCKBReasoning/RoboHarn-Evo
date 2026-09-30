from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_SCHEMA = "roboharn_evo/rmbench/hpk_formal_q1_protocol/v1"
RUN_CONFIG_SCHEMA = "roboharn_evo/rmbench/hpk_formal_q1_run_config/v1"
SOURCE_POOL_SCHEMA = "roboharn_evo/rmbench/hpk_formal_q1_source_pool/v1"

TASK_ROLES = {
    "rearrange_blocks": "task_dominant",
    "swap_blocks": "task_dominant",
    "press_button": "action_dominant",
    "place_block_mat": "action_dominant",
    "put_back_block": "coupled",
    "cover_blocks": "coupled",
}

METHODS = {
    "off": ("HPK-Off", False, False, False),
    "flat": ("Flat Free-Form Reflection", True, False, False),
    "task": ("Task-HPK", False, True, False),
    "action": ("Action-HPK", False, False, True),
    "full": ("Full HPK (Frozen)", False, True, True),
}


class FormalQ1ProtocolError(ValueError):
    """The frozen Q1 protocol or its benchmark inputs are inconsistent."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalQ1ProtocolError(f"{label} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FormalQ1ProtocolError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _int_list(value: Any, *, label: str, count: int) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise FormalQ1ProtocolError(f"{label} must be an integer array")
    result = list(value)
    if (
        len(result) != count
        or any(isinstance(item, bool) or not isinstance(item, int) for item in result)
        or len(set(result)) != len(result)
    ):
        raise FormalQ1ProtocolError(
            f"{label} must contain exactly {count} distinct integers"
        )
    return result


def load_protocol(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, label="protocol")
    _exact_keys(
        root,
        {
            "schema",
            "protocol_id",
            "task_config",
            "instruction_set",
            "instruction_type",
            "source",
            "evaluation",
            "tasks",
            "methods",
            "store_policy",
            "split_policy",
        },
        label="protocol",
    )
    if root["schema"] != PROTOCOL_SCHEMA:
        raise FormalQ1ProtocolError("protocol schema mismatch")
    if root["task_config"] != "demo_clean":
        raise FormalQ1ProtocolError("Q1 task_config must remain demo_clean")

    source_config = _mapping(root["source"], label="source")
    evaluation = _mapping(root["evaluation"], label="evaluation")
    source_count = source_config.get("trajectory_count_per_task")
    evaluation_count = evaluation.get("paired_seed_count_per_task")
    if source_count != 10 or evaluation_count != 10:
        raise FormalQ1ProtocolError("Q1 requires 10 source and 10 evaluation seeds")
    source_indices = _int_list(
        source_config.get("episode_indices"),
        label="source.episode_indices",
        count=10,
    )
    evaluation_indices = _int_list(
        evaluation.get("episode_indices"),
        label="evaluation.episode_indices",
        count=10,
    )
    if set(source_indices) & set(evaluation_indices):
        raise FormalQ1ProtocolError("source and evaluation episode indices overlap")
    expected_evaluation = {
        "episode_indices",
        "paired_seed_count_per_task",
        "exact_seed_fail_closed",
        "perception_condition",
        "action_limit",
        "control_turn_limit",
        "no_progress_control_turn_limit",
        "semantic_round_limit_per_subtask",
        "retry_budget",
    }
    _exact_keys(evaluation, expected_evaluation, label="evaluation")
    if evaluation["exact_seed_fail_closed"] is not True:
        raise FormalQ1ProtocolError("exact-seed fail-closed must be enabled")
    if evaluation["perception_condition"] != "no_oracle":
        raise FormalQ1ProtocolError("Q1 must remain no-oracle")
    for field, expected in {
        "action_limit": 150,
        "control_turn_limit": 64,
        "no_progress_control_turn_limit": 10,
        "semantic_round_limit_per_subtask": 10,
        "retry_budget": 0,
    }.items():
        if evaluation[field] != expected:
            raise FormalQ1ProtocolError(f"evaluation.{field} must equal {expected}")

    raw_tasks = root["tasks"]
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(TASK_ROLES):
        raise FormalQ1ProtocolError("protocol must contain the six frozen tasks")
    parsed_tasks: list[dict[str, Any]] = []
    for index, raw_task in enumerate(raw_tasks):
        task = _mapping(raw_task, label=f"tasks[{index}]")
        _exact_keys(
            task,
            {"name", "role", "source_seeds", "evaluation_seeds"},
            label=f"tasks[{index}]",
        )
        name = task.get("name")
        if name not in TASK_ROLES or task.get("role") != TASK_ROLES[name]:
            raise FormalQ1ProtocolError(f"task role mismatch for {name!r}")
        task["source_seeds"] = _int_list(
            task["source_seeds"], label=f"{name}.source_seeds", count=10
        )
        task["evaluation_seeds"] = _int_list(
            task["evaluation_seeds"], label=f"{name}.evaluation_seeds", count=10
        )
        if set(task["source_seeds"]) & set(task["evaluation_seeds"]):
            raise FormalQ1ProtocolError(f"source/evaluation seed overlap for {name}")
        parsed_tasks.append(task)
    if {task["name"] for task in parsed_tasks} != set(TASK_ROLES):
        raise FormalQ1ProtocolError("frozen task set is incomplete or duplicated")

    raw_methods = root["methods"]
    if not isinstance(raw_methods, list) or len(raw_methods) != len(METHODS):
        raise FormalQ1ProtocolError("protocol must contain the five matched methods")
    parsed_methods: list[dict[str, Any]] = []
    for index, raw_method in enumerate(raw_methods):
        method = _mapping(raw_method, label=f"methods[{index}]")
        _exact_keys(
            method,
            {
                "id",
                "paper_name",
                "flat_reflection",
                "task_knowledge",
                "action_knowledge",
            },
            label=f"methods[{index}]",
        )
        method_id = method.get("id")
        if method_id not in METHODS:
            raise FormalQ1ProtocolError(f"unknown method {method_id!r}")
        expected = METHODS[method_id]
        observed = (
            method.get("paper_name"),
            method.get("flat_reflection"),
            method.get("task_knowledge"),
            method.get("action_knowledge"),
        )
        if observed != expected:
            raise FormalQ1ProtocolError(f"method definition mismatch for {method_id}")
        parsed_methods.append(method)
    if {method["id"] for method in parsed_methods} != set(METHODS):
        raise FormalQ1ProtocolError("matched method set is incomplete or duplicated")

    store_policy = _mapping(root["store_policy"], label="store_policy")
    if store_policy != {
        "evaluation_read_only": True,
        "online_reflection": False,
        "online_update": False,
    }:
        raise FormalQ1ProtocolError("Q1 Store policy must remain frozen/read-only")
    split_policy = _mapping(root["split_policy"], label="split_policy")
    if set(split_policy.values()) != {True}:
        raise FormalQ1ProtocolError("all split invariants must be enabled")

    root["source"] = source_config
    root["evaluation"] = evaluation
    root["tasks"] = parsed_tasks
    root["methods"] = parsed_methods
    return root


def _read_seed_manifest(path: Path) -> list[int]:
    values = path.read_text(encoding="utf-8").split()
    try:
        seeds = [int(value) for value in values]
    except ValueError as exc:
        raise FormalQ1ProtocolError(f"invalid seed manifest {path}") from exc
    if len(seeds) != len(set(seeds)):
        raise FormalQ1ProtocolError(f"seed manifest contains duplicates: {path}")
    return seeds


def validate_benchmark_inputs(
    protocol: Mapping[str, Any],
    *,
    repo_root: str | Path,
    asset_root: str | Path,
) -> list[dict[str, Any]]:
    repo = Path(repo_root).resolve(strict=True)
    assets = Path(asset_root).resolve(strict=True)
    task_config = str(protocol["task_config"])
    source_indices = list(protocol["source"]["episode_indices"])
    evaluation_indices = list(protocol["evaluation"]["episode_indices"])
    resolved: list[dict[str, Any]] = []
    for task in protocol["tasks"]:
        name = str(task["name"])
        module = repo / "benchmarks" / "rmbench" / "envs" / f"{name}.py"
        if not module.is_file():
            raise FormalQ1ProtocolError(f"missing task module: {module}")
        data_root = assets / "data" / "data" / name / task_config
        seed_manifest = data_root / "seed.txt"
        annotation = data_root / "language_annotation.json"
        if not seed_manifest.is_file() or not annotation.is_file():
            raise FormalQ1ProtocolError(f"incomplete benchmark data root: {data_root}")
        seeds = _read_seed_manifest(seed_manifest)
        all_indices = source_indices + evaluation_indices
        if max(all_indices) >= len(seeds):
            raise FormalQ1ProtocolError(f"seed manifest too short for {name}")
        actual_source = [seeds[index] for index in source_indices]
        actual_evaluation = [seeds[index] for index in evaluation_indices]
        if actual_source != list(task["source_seeds"]):
            raise FormalQ1ProtocolError(f"frozen source seeds drifted for {name}")
        if actual_evaluation != list(task["evaluation_seeds"]):
            raise FormalQ1ProtocolError(f"frozen evaluation seeds drifted for {name}")
        for episode in source_indices:
            required = (
                data_root / "data" / f"episode{episode}.hdf5",
                data_root / "_traj_data" / f"episode{episode}.pkl",
                data_root / "instructions" / f"episode{episode}.json",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FormalQ1ProtocolError(
                    f"source trajectory {name}/{episode} is incomplete: {missing}"
                )
        resolved.append(
            {
                "task": name,
                "role": task["role"],
                "data_root": str(data_root),
                "source_episode_indices": source_indices,
                "source_seeds": actual_source,
                "evaluation_episode_indices": evaluation_indices,
                "evaluation_seeds": actual_evaluation,
                "instance_scope": "fresh simulator reset per seed",
            }
        )
    return resolved


def build_matrix(
    protocol: Mapping[str, Any],
    *,
    code_commit: str,
) -> list[dict[str, Any]]:
    if len(code_commit) != 40 or any(
        ch not in "0123456789abcdef" for ch in code_commit
    ):
        raise FormalQ1ProtocolError("code_commit must be a full lowercase Git commit")
    rows: list[dict[str, Any]] = []
    for task in protocol["tasks"]:
        for method in protocol["methods"]:
            for episode_index, seed in zip(
                protocol["evaluation"]["episode_indices"],
                task["evaluation_seeds"],
                strict=True,
            ):
                rows.append(
                    {
                        "cell_id": f"q1_{task['name']}_{method['id']}_seed_{seed}",
                        "protocol_id": protocol["protocol_id"],
                        "code_commit": code_commit,
                        "task": task["name"],
                        "task_role": task["role"],
                        "task_config": protocol["task_config"],
                        "method": method["id"],
                        "paper_method": method["paper_name"],
                        "evaluation_episode_index": episode_index,
                        "evaluation_seed": seed,
                        "source_pool": f"{protocol['protocol_id']}:{task['name']}:source_0_9",
                        "flat_reflection": method["flat_reflection"],
                        "task_knowledge": method["task_knowledge"],
                        "action_knowledge": method["action_knowledge"],
                        "store_read_only": True,
                        "retry_budget": protocol["evaluation"]["retry_budget"],
                    }
                )
    if len(rows) != 300 or len({row["cell_id"] for row in rows}) != 300:
        raise FormalQ1ProtocolError("Q1 matrix must contain exactly 300 unique cells")
    return rows


def current_git_commit(repo_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_preparation(
    *,
    protocol_path: str | Path,
    repo_root: str | Path,
    asset_root: str | Path,
    output_root: str | Path,
    code_commit: str,
) -> Path:
    protocol = load_protocol(protocol_path)
    sources = validate_benchmark_inputs(
        protocol,
        repo_root=repo_root,
        asset_root=asset_root,
    )
    matrix = build_matrix(protocol, code_commit=code_commit)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    run_config = {
        "schema": RUN_CONFIG_SCHEMA,
        "protocol": protocol,
        "code_commit": code_commit,
        "repo_root": str(Path(repo_root).resolve(strict=True)),
        "asset_root": str(Path(asset_root).resolve(strict=True)),
        "formal_shard_count": len(matrix),
        "formal_episode_count": len(matrix),
        "source_pools": sources,
        "outcomes_observed_before_freeze": False,
    }
    (output / "run_config.yaml").write_text(
        yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output / "source_pool_manifest.json").write_text(
        json.dumps(
            {"schema": SOURCE_POOL_SCHEMA, "tasks": sources},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fields = list(matrix[0])
    with (output / "method_task_seed_matrix.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix)
    return output


__all__ = [
    "METHODS",
    "PROTOCOL_SCHEMA",
    "RUN_CONFIG_SCHEMA",
    "SOURCE_POOL_SCHEMA",
    "TASK_ROLES",
    "FormalQ1ProtocolError",
    "build_matrix",
    "current_git_commit",
    "load_protocol",
    "validate_benchmark_inputs",
    "write_preparation",
]

"""Bind one preregistered Q1 method to the unchanged RMBench runtime."""

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.compatibility import normalize_agent_knowledge_config
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store


CELL_BINDING_SCHEMA = "roboharn_evo/rmbench/hpk_formal_q1_cell_binding/v1"
METHOD_VIEW_SCHEMA = "roboharn_evo/rmbench/hpk_formal_q1_method_views/v1"
PROTOCOL_ID = "rmbench_hpk_v31_q1_main_v1"
PROTOCOL_IDS = frozenset(
    {
        PROTOCOL_ID,
        "rmbench_hpk_v31_q1_clarified_six_tasks_v1",
        "rmbench_afk_v31_q1_main_v1",
        "rmbench_afk_v31_q1_clarified_six_tasks_v1",
    }
)
METHODS = frozenset({"off", "flat", "task", "action", "full"})


class FormalQ1BindingError(ValueError):
    """A formal cell does not match its frozen task, seed, source, or method."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalQ1BindingError(f"{label} must be an object")
    return copy.deepcopy(dict(value))


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalQ1BindingError(f"{label} must be non-empty text")
    return value.strip()


def _current_commit() -> str:
    repo = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalQ1BindingError(f"cannot read {label}: {path}") from exc
    return _mapping(value, label=label)


def _runtime_seed(arguments: Mapping[str, Any]) -> int:
    evaluation = _mapping(arguments.get("eval", {}), label="eval")
    seed = evaluation.get("start_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FormalQ1BindingError("eval.start_seed must be one explicit seed")
    return seed


@dataclass(frozen=True, slots=True)
class ResolvedFormalQ1Binding:
    arguments: dict[str, Any]
    method: str
    cell_id: str
    task: str
    evaluation_seed: int
    source_pool: str


def resolve_formal_q1_binding(
    arguments: Mapping[str, Any],
    request: Mapping[str, Any],
) -> ResolvedFormalQ1Binding:
    """Resolve a path-only CLI request into one frozen benchmark method.

    The formal launcher still rejects arbitrary ``agent.*`` overrides.  This
    adapter derives the required Agent configuration from a preregistered cell
    file instead, so the five methods cannot drift independently at launch.
    """

    requested = _mapping(request, label="rmbench_formal")
    if set(requested) != {"config_path"}:
        raise FormalQ1BindingError(
            "formal Q1 requires exactly rmbench_formal.config_path"
        )
    config_path = (
        Path(_text(requested["config_path"], label="rmbench_formal.config_path"))
        .expanduser()
        .resolve(strict=True)
    )
    payload = _load_json(config_path, label="formal cell binding")
    expected_fields = {
        "schema",
        "protocol_id",
        "code_commit",
        "cell_id",
        "task",
        "method",
        "evaluation_seed",
        "source_pool",
        "method_views_root",
    }
    if set(payload) != expected_fields or payload.get("schema") not in {
        CELL_BINDING_SCHEMA, "tcm/rmbench/afk_formal_q1_cell_binding/v1"
    }:
        raise FormalQ1BindingError("formal cell binding fields or schema mismatch")
    protocol_id = _text(payload.get("protocol_id"), label="protocol_id")
    if protocol_id not in PROTOCOL_IDS:
        raise FormalQ1BindingError("formal cell protocol ID mismatch")
    method = _text(payload.get("method"), label="method").lower()
    if method not in METHODS:
        raise FormalQ1BindingError(f"unknown formal Q1 method: {method}")
    task = _text(payload.get("task"), label="task")
    if task != _text(arguments.get("task_name"), label="task_name"):
        raise FormalQ1BindingError("formal cell task differs from runtime task")
    evaluation_seed = payload.get("evaluation_seed")
    if (
        isinstance(evaluation_seed, bool)
        or not isinstance(evaluation_seed, int)
        or evaluation_seed < 0
        or evaluation_seed != _runtime_seed(arguments)
    ):
        raise FormalQ1BindingError("formal cell seed differs from runtime seed")
    expected_cell_id = f"q1_{task}_{method}_seed_{evaluation_seed}"
    if payload.get("cell_id") != expected_cell_id:
        raise FormalQ1BindingError("formal cell ID differs from task, method, or seed")
    source_pool = _text(payload.get("source_pool"), label="source_pool")
    if source_pool != f"{protocol_id}:{task}:source_0_9":
        raise FormalQ1BindingError("formal source-pool identity mismatch")
    code_commit = _text(payload.get("code_commit"), label="code_commit")
    if code_commit != _current_commit():
        raise FormalQ1BindingError("formal cell code commit differs from runtime HEAD")

    views_root = (
        Path(_text(payload.get("method_views_root"), label="method_views_root"))
        .expanduser()
        .resolve(strict=True)
    )
    views = _load_json(views_root / "method_views.json", label="method views")
    if views.get("schema") not in {
        METHOD_VIEW_SCHEMA, "tcm/rmbench/afk_formal_q1_method_views/v1"
    }:
        raise FormalQ1BindingError("method view schema mismatch")
    if views.get("same_source_pool_for_all_methods") is not True:
        raise FormalQ1BindingError("method views do not share one source pool")
    if views.get("source_collections") != [f"episode_{index}" for index in range(10)]:
        raise FormalQ1BindingError(
            "method views do not use source episodes 0 through 9"
        )
    source_complete = _load_json(
        views_root.parent / "source_complete.json",
        label="completed source pool",
    )
    if (
        source_complete.get("task") != task
        or source_complete.get("source_trajectory_count") != 10
        or source_complete.get("automatic_retries") != 0
        or source_complete.get("method_views")
        != ["off", "flat", "task", "action", "full"]
    ):
        raise FormalQ1BindingError("completed source pool does not match this cell")

    merged = _mapping(arguments, label="runtime arguments")
    agent = normalize_agent_knowledge_config(_mapping(merged.get("agent", {}), label="agent"))
    agent["procedure_experience"] = {"mode": "off"}
    agent["hpk"] = {"mode": "off"}
    internal_method: dict[str, Any] = {"method": method}
    if method in {"off", "flat"}:
        agent["hpk_v3"] = {"mode": "off"}
        if method == "flat":
            lesson = (views_root / "flat" / "flat_reflection.txt").resolve(strict=True)
            internal_method["flat_reflection"] = {
                "lesson_path": str(lesson),
                "max_prompt_chars": 4000,
            }
    else:
        store_root = (views_root / method / "store").resolve(strict=True)
        task_knowledge, action_knowledge = load_hierarchical_store(store_root)
        expected_presence = {
            "task": (True, False),
            "action": (False, True),
            "full": (True, True),
        }[method]
        observed_presence = (bool(task_knowledge), bool(action_knowledge))
        if observed_presence != expected_presence:
            raise FormalQ1BindingError(
                f"{method} Store layer presence differs from the frozen ablation"
            )
        agent["hpk_v3"] = {
            "mode": "full",
            "store_root": str(store_root),
            "hierarchical_reflection": False,
            "hpk_goal_consistency_enabled": method in {"task", "full"},
            "rmbench_read_only_utility_gate": True,
        }
    merged["agent"] = agent
    merged["rmbench_formal"] = internal_method
    return ResolvedFormalQ1Binding(
        arguments=merged,
        method=method,
        cell_id=expected_cell_id,
        task=task,
        evaluation_seed=evaluation_seed,
        source_pool=source_pool,
    )


__all__ = [
    "CELL_BINDING_SCHEMA",
    "FormalQ1BindingError",
    "ResolvedFormalQ1Binding",
    "resolve_formal_q1_binding",
]

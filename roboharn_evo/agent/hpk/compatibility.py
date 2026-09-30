from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _rename_fields(value: Mapping[str, Any], names: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    for previous, current in names:
        if previous not in result:
            continue
        item = result.pop(previous)
        if current in result and result[current] != item:
            raise ValueError(f"conflicting fields: {previous}, {current}")
        result[current] = item
    return result


def normalize_hpk_v3_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return _rename_fields(config, (("afk_goal_consistency_enabled", "hpk_goal_consistency_enabled"),))


def normalize_knowledge_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    return _rename_fields(record, (
        ("learned_afk", "learned_hpk"),
        ("learned_afk_present", "learned_hpk_present"),
    ))


def normalize_observe_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    result = _rename_fields(record, (
        ("retrieved_task_afk_ids", "retrieved_task_hpk_ids"),
        ("selected_geometric_afk_id", "selected_geometric_hpk_id"),
    ))
    if result.get("profile") == "afk_observe_audit/p0a":
        result["profile"] = "hpk_observe_audit/p0a"
    return result


def normalize_hpk_experiment_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    return _rename_fields(record, (
        ("afk_mode", "hpk_mode"),
        ("afk_v3_mode", "hpk_v3_mode"),
        ("afk_goal_consistency_enabled", "hpk_goal_consistency_enabled"),
    ))


def normalize_hpk_event_name(event: str) -> str:
    return "hpk_" + event[len("afk_"):] if event.startswith("afk_") else event


def normalize_hpk_runtime_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    result = _rename_fields(record, (("afk", "hpk"),))
    if isinstance(result.get("simulator"), Mapping):
        result["simulator"] = _rename_fields(result["simulator"], (("tcm_agent_loop", "roboharn_agent_loop"),))
    return result


def normalize_agent_knowledge_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = _rename_fields(config, (("afk", "hpk"), ("afk_v3", "hpk_v3")))
    if isinstance(result.get("hpk_v3"), Mapping):
        result["hpk_v3"] = normalize_hpk_v3_config(result["hpk_v3"])
    return result


def normalize_v3_rollout_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    for previous, current in (
        ("afk_v3_action_usage", "hpk_v3_action_usage"),
        ("afk_v3_subtask_usage", "hpk_v3_subtask_usage"),
        ("prompt_before_action_afk", "prompt_before_action_hpk"),
    ):
        if previous in result:
            value = result.pop(previous)
            if current in result and result[current] != value:
                raise ValueError(f"conflicting rollout fields: {previous}, {current}")
            result[current] = value
    event = result.get("event")
    if isinstance(event, str) and event.startswith(("afk_v3_", "afk_v31_")):
        result["event"] = "hpk" + event[len("afk"):]
    usage = result.get("hpk_v3_action_usage")
    if isinstance(usage, dict) and usage.get("schema") == "tcm/afk/v3/action_usage":
        usage["schema"] = "roboharn_evo/hpk/v3/action_usage"
    return result

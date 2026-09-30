
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_event_name
from roboharn_evo.agent.hpk.schemas import EntryV1
from roboharn_evo.agent.hpk.semantic_knowledge import (
    semantic_evidence_from_runtime_verification,
    semantic_evidence_from_transition,
    semantic_knowledge_from_strategy,
    semantic_object_from_mapping,
)
from roboharn_evo.agent.hpk.semantic_store import save_semantic_knowledge
from roboharn_evo.agent.hpk.updater import apply_semantic_promotion


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _object_for_transition(
    transition: Mapping[str, Any],
    prepared: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    source = prepared.get(str(transition.get("action_attempt_nonce") or ""), {})
    for call in source.get("guarded_calls", []):
        if not isinstance(call, Mapping):
            continue
        args = call.get("args")
        if not isinstance(args, Mapping):
            continue
        instance_ref = args.get("instance_id")
        scene = args.get("_scene_memory")
        if not isinstance(scene, Mapping):
            continue
        for instance in scene.get("instances", []):
            if (
                isinstance(instance, Mapping)
                and instance.get("instance_id") == instance_ref
            ):
                semantic = semantic_object_from_mapping(instance)
                if semantic["category"] != "unknown":
                    return semantic
    return semantic_object_from_mapping({"class": "unknown"})


def replay_semantic_trace(
    *,
    legacy_entry_path: Path,
    private_trace_path: Path,
    output_path: Path,
    agent_trace_path: Path | None = None,
    promotion_policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    entry = EntryV1.from_dict(
        json.loads(legacy_entry_path.read_text(encoding="utf-8").splitlines()[0])
    )
    rows = _read_json_lines(private_trace_path)
    prepared = {
        str(row.get("action_attempt_nonce") or ""): row
        for row in rows
        if normalize_hpk_event_name(str(row.get("event", ""))) == "hpk_action_attempt_prepared_private"
    }
    expected_operation = str(entry["task_strategy"]["operation"])
    expected_orientation = str(entry["geometric_strategy"]["orientation"]["relation"])
    matching: list[tuple[float, dict[str, Any], dict[str, str]]] = []
    for row in rows:
        if row.get("event") != "action_effect_transition":
            continue
        transition = row.get("transition")
        if not isinstance(transition, Mapping):
            continue
        features = transition.get("candidate_geometry_features")
        if not isinstance(features, Mapping):
            continue
        if (
            transition.get("operation") != expected_operation
            or features.get("orientation_relation") != expected_orientation
        ):
            continue
        matching.append(
            (
                float(row.get("timestamp", 0.0) or 0.0),
                dict(transition),
                _object_for_transition(transition, prepared),
            )
        )
    if not matching:
        raise ValueError("the trace contains no action matching the legacy strategy")

    known_objects = [
        value
        for _, _, value in matching
        if value["category"] != "unknown" and value["color"] != "unknown"
    ]
    object_semantics = known_objects[0] if known_objects else matching[0][2]
    object_semantics = {
        **object_semantics,
        "role": str(entry["condition"]["manipulated_object"]["role"]).replace("_", " "),
        "held_state": str(
            entry["condition"]["manipulated_object"]["held_state"]
        ).replace("_", " "),
    }
    evidence = [
        semantic_evidence_from_transition(
            transition,
            object_semantics=object_semantics,
        )
        for _, transition, _ in matching
    ]
    if agent_trace_path is not None:
        for row in _read_json_lines(agent_trace_path):
            result = row.get("result")
            if row.get("event") != "action_effect_verification" or not isinstance(
                result, Mapping
            ):
                continue
            validation = result.get("runtime_grasp_validation")
            if not isinstance(validation, Mapping):
                continue
            candidate_ref = validation.get("grasp_candidate_id")
            arm = validation.get("arm")
            candidates = [
                (timestamp, transition)
                for timestamp, transition, _ in matching
                if transition.get("selected_candidate_private_ref") == candidate_ref
                and transition.get("arm") == arm
                and timestamp <= float(row.get("timestamp", 0.0) or 0.0)
            ]
            if not candidates:
                continue
            _, transition = max(candidates, key=lambda value: value[0])
            evidence.append(
                semantic_evidence_from_runtime_verification(
                    result,
                    object_semantics=object_semantics,
                    operation=str(transition["operation"]),
                    arm=str(transition["arm"]),
                )
            )
    record = semantic_knowledge_from_strategy(
        condition=entry["condition"],
        task_strategy=entry["task_strategy"],
        geometric_strategy=entry["geometric_strategy"],
        expected_effect=entry["expected_effect"],
        object_semantics=object_semantics,
        reasoning={
            "observed_problem": "not available",
            "failure_analysis": "not available",
            "strategy_rationale": "not available",
            "causal_hypothesis": "not available",
            "expected_observation": "the object moves with the lifted gripper",
            "failure_condition": "the object does not remain attached after lift",
            "source": "runtime extraction",
            "confidence": 0.0,
        },
        evidence=evidence,
        status="candidate",
    )
    if promotion_policy is not None:
        record = apply_semantic_promotion(record, promotion_policy)
    save_semantic_knowledge(output_path, [record])
    return (record,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-entry", type=Path, required=True)
    parser.add_argument("--private-trace", type=Path, required=True)
    parser.add_argument("--agent-trace", type=Path)
    parser.add_argument("--promotion-policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = None
    if args.promotion_policy is not None:
        import yaml

        policy = yaml.safe_load(args.promotion_policy.read_text(encoding="utf-8"))
    records = replay_semantic_trace(
        legacy_entry_path=args.legacy_entry,
        private_trace_path=args.private_trace,
        output_path=args.output,
        agent_trace_path=args.agent_trace,
        promotion_policy=policy,
    )
    print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

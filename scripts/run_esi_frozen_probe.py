from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = (REPOSITORY_ROOT / "eval_result" / "esi_bench" / "runs").resolve()
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    AgentApiESIEvaluatedModel,
    AgentApiESIMultimodalRetrievalBackend,
    ESIPublicStepContext,
    ESITaskOnlyRetrievalRuntime,
    VLMCurrentImageSubtaskKnowledgeRetriever,
    VLMFlatLessonRetriever,
    behavior_change_at_frozen_state,
    load_frozen_task_store,
    load_shuffle_manifest,
    official_action_family,
)

FIVE_CONDITION_MODES = (
    "off",
    "flat",
    "task_core",
    "task_full",
    "shuffled_full",
)
FOUR_CONDITION_MODES = ("off", "flat", "task_core", "task_full")
PROBE_PROFILES = {
    "five_condition": FIVE_CONDITION_MODES,
    "four_condition_exploratory": FOUR_CONDITION_MODES,
}
MODES = FIVE_CONDITION_MODES
PUBLIC_CAMERA_ACTIONS = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "turn_left",
    "turn_right",
    "turn_up",
    "turn_down",
    "stop",
)


class FrozenProbeError(RuntimeError):
    pass


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FrozenProbeError(f"{label} must be one object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise FrozenProbeError(f"{label} line {line_number} must be one object")
        records.append(dict(value))
    return tuple(records)


def modes_for_profile(profile: str) -> tuple[str, ...]:
    normalized = str(profile or "").strip().casefold()
    try:
        return PROBE_PROFILES[normalized]
    except KeyError as exc:
        raise FrozenProbeError("unknown frozen probe profile") from exc


def probe_call_budget(
    state_count: int,
    *,
    modes: Sequence[str] = MODES,
) -> dict[str, int]:
    if (
        isinstance(state_count, bool)
        or not isinstance(state_count, int)
        or state_count < 1
    ):
        raise FrozenProbeError("state_count must be positive")
    active_modes = tuple(modes)
    if active_modes not in PROBE_PROFILES.values():
        raise FrozenProbeError("probe modes do not match a frozen profile")
    retrievals_per_state = 3 if "shuffled_full" in active_modes else 2
    return {
        "annotation_calls": 1,
        "evaluated_model_calls": state_count * len(active_modes),
        "retrieval_model_calls": state_count * retrievals_per_state,
        "total_model_calls": 1
        + state_count * (len(active_modes) + retrievals_per_state),
        "annotation_image_payloads": state_count,
        "evaluated_image_payloads": state_count * len(active_modes),
        "retrieval_image_payloads": state_count * retrievals_per_state,
        "total_image_payloads": state_count
        * (1 + len(active_modes) + retrievals_per_state),
        "automatic_retries": 0,
        "simulator_episodes": 0,
        "executed_actions": 0,
    }


def rotated_mode_order(
    state_index: int,
    *,
    modes: Sequence[str] = MODES,
) -> tuple[str, ...]:
    if (
        isinstance(state_index, bool)
        or not isinstance(state_index, int)
        or state_index < 1
    ):
        raise FrozenProbeError("state_index must be positive")
    active_modes = tuple(modes)
    if active_modes not in PROBE_PROFILES.values():
        raise FrozenProbeError("probe modes do not match a frozen profile")
    offset = (state_index - 1) % len(active_modes)
    return active_modes[offset:] + active_modes[:offset]


def annotation_schema(state_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "states": {
                "type": "array",
                "minItems": state_count,
                "maxItems": state_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "state_index": {"type": "integer", "minimum": 1},
                        "evidence_need": {"type": "string", "minLength": 1},
                        "acceptable_next_subtasks": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 6,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "acceptable_actions": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 6,
                            "items": {
                                "type": "string",
                                "enum": list(PUBLIC_CAMERA_ACTIONS),
                            },
                        },
                        "acceptable_action_families": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "items": {
                                "type": "string",
                                "enum": ["viewpoint-or-locomotion", "answer"],
                            },
                        },
                        "stop_or_continue": {
                            "type": "string",
                            "enum": ["stop", "continue"],
                        },
                        "rationale": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "state_index",
                        "evidence_need",
                        "acceptable_next_subtasks",
                        "acceptable_actions",
                        "acceptable_action_families",
                        "stop_or_continue",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["states"],
        "additionalProperties": False,
    }


def decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "minLength": 1},
            "next_subtask": {"type": "string", "minLength": 1},
            "stop_or_continue": {
                "type": "string",
                "enum": ["stop", "continue"],
            },
            "current_answer": {"type": "string", "minLength": 1},
            "reasoning": {"type": "string", "minLength": 1},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [
            "action",
            "next_subtask",
            "stop_or_continue",
            "current_answer",
            "reasoning",
            "confidence",
        ],
        "additionalProperties": False,
    }


def validate_annotations(
    value: Mapping[str, Any], *, state_count: int
) -> tuple[dict[str, Any], ...]:
    if set(value) != {"states"} or not isinstance(value["states"], list):
        raise FrozenProbeError("annotation response fields mismatch")
    records = tuple(dict(item) for item in value["states"] if isinstance(item, Mapping))
    if len(records) != state_count or [
        item.get("state_index") for item in records
    ] != list(range(1, state_count + 1)):
        raise FrozenProbeError("annotations do not cover ordered frozen states")
    allowed = set(PUBLIC_CAMERA_ACTIONS)
    for item in records:
        actions = item.get("acceptable_actions")
        subtasks = item.get("acceptable_next_subtasks")
        families = item.get("acceptable_action_families")
        stop_or_continue = item.get("stop_or_continue")
        derived_families = (
            {
                official_action_family(action)
                for action in actions
                if isinstance(action, str)
            }
            if isinstance(actions, list)
            else set()
        )
        if (
            not isinstance(actions, list)
            or not actions
            or len(actions) != len(set(actions))
            or any(action not in allowed for action in actions)
            or not isinstance(subtasks, list)
            or not subtasks
            or len(subtasks) != len(set(subtasks))
            or any(not str(subtask or "").strip() for subtask in subtasks)
            or not isinstance(families, list)
            or set(families) != derived_families
            or stop_or_continue not in {"stop", "continue"}
            or (stop_or_continue == "stop") != (set(actions) == {"stop"})
            or not str(item.get("evidence_need") or "").strip()
            or not str(item.get("rationale") or "").strip()
        ):
            raise FrozenProbeError("frozen-state annotation is invalid")
    return records


def _normalize_action(value: Any) -> str:
    return "_".join(str(value or "").strip().casefold().replace("-", " ").split())


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _complete_call_audit(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("transport") == "existing_agent_api_responses"
        and value.get("endpoint_path") == "/v1/responses"
        and value.get("model") == "gpt-5.5"
        and value.get("reasoning_effort") == "xhigh"
        and value.get("retry_count") == 0
        and isinstance(value.get("usage"), Mapping)
    )


def _paired_comparison(
    grouped: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    baseline_mode: str,
    active_mode: str,
    adopted_only: bool = False,
) -> dict[str, Any]:
    paired = []
    corrected = []
    harmed = []
    adoption = {"helpful": 0, "harmful": 0, "neutral": 0, "not_adopted": 0}
    eligible = []
    baseline_acceptable_count = 0
    active_acceptable_count = 0
    for state_index in sorted(grouped):
        baseline = grouped[state_index][baseline_mode]
        active = grouped[state_index][active_mode]
        adopted = bool(active["retrieval"]["knowledge_adopted"])
        if adopted_only and not adopted:
            continue
        eligible.append(state_index)
        baseline_ok = bool(baseline["exact_action_acceptable"])
        active_ok = bool(active["exact_action_acceptable"])
        baseline_acceptable_count += int(baseline_ok)
        active_acceptable_count += int(active_ok)
        changed = _normalize_action(
            baseline["decision"]["action"]
        ) != _normalize_action(active["decision"]["action"])
        label = "neutral"
        if not baseline_ok and active_ok:
            label = "helpful"
            corrected.append(state_index)
        elif baseline_ok and not active_ok:
            label = "harmful"
            harmed.append(state_index)
        if adopted:
            adoption[label] += 1
        else:
            adoption["not_adopted"] += 1
        paired.append(
            {
                "state_index": state_index,
                "off_action": baseline["decision"]["action"],
                "task_action": active["decision"]["action"],
                "change_label": label,
                "changed": changed,
            }
        )
    return {
        "baseline_mode": baseline_mode,
        "active_mode": active_mode,
        "adopted_only": adopted_only,
        "eligible_state_indices": eligible,
        "baseline_acceptable_count": baseline_acceptable_count,
        "active_acceptable_count": active_acceptable_count,
        "behavior": behavior_change_at_frozen_state(paired),
        "wrong_to_acceptable_state_indices": corrected,
        "acceptable_to_wrong_state_indices": harmed,
        "adoption_outcomes": adoption,
    }


def summarize_probe(
    records: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    receipts_complete: bool = False,
    modes: Sequence[str] = MODES,
) -> dict[str, Any]:
    if not isinstance(receipts_complete, bool):
        raise FrozenProbeError("receipts_complete must be a boolean")
    active_modes = tuple(modes)
    if active_modes not in PROBE_PROFILES.values():
        raise FrozenProbeError("probe modes do not match a frozen profile")
    profile = next(
        name for name, values in PROBE_PROFILES.items() if values == active_modes
    )
    annotation_by_state = {int(item["state_index"]): item for item in annotations}
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for item in records:
        state_index = int(item["state_index"])
        grouped.setdefault(state_index, {})[str(item["mode"])] = item
    if set(grouped) != set(annotation_by_state) or any(
        set(items) != set(active_modes) for items in grouped.values()
    ):
        raise FrozenProbeError("decision records do not form complete paired states")
    for state_index, items in grouped.items():
        core = items["task_core"]
        full = items["task_full"]
        if _shared_retrieval_view(core["retrieval"]) != _shared_retrieval_view(
            full["retrieval"]
        ):
            raise FrozenProbeError(
                f"state {state_index} Core/Full retrieval results differ"
            )
        if "retrieval_scope" in core or "retrieval_scope" in full:
            if (
                core.get("retrieval_scope") != "task_core_full_shared"
                or full.get("retrieval_scope") != "task_core_full_shared"
            ):
                raise FrozenProbeError(
                    f"state {state_index} lacks the shared Task retrieval receipt"
                )
    condition_metrics = {}
    for mode in active_modes:
        selected = [grouped[index][mode] for index in sorted(grouped)]
        subtask_outcomes = [
            _normalize_text(item["decision"]["next_subtask"])
            in {
                _normalize_text(value)
                for value in annotation_by_state[int(item["state_index"])][
                    "acceptable_next_subtasks"
                ]
            }
            for item in selected
        ]
        stop_outcomes = [
            item["decision"]["stop_or_continue"]
            == annotation_by_state[int(item["state_index"])]["stop_or_continue"]
            for item in selected
        ]
        condition_metrics[mode] = {
            "state_count": len(selected),
            "exact_acceptable_count": sum(
                bool(item["exact_action_acceptable"]) for item in selected
            ),
            "family_acceptable_count": sum(
                bool(item["action_family_acceptable"]) for item in selected
            ),
            "invalid_action_count": sum(
                item["action_family"] == "invalid-or-noop" for item in selected
            ),
            "knowledge_adoption_count": sum(
                bool(item["retrieval"]["knowledge_adopted"]) for item in selected
            ),
            "next_subtask_correct_count": sum(subtask_outcomes),
            "next_subtask_accuracy": sum(subtask_outcomes) / len(subtask_outcomes),
            "stop_continue_correct_count": sum(stop_outcomes),
            "stop_continue_accuracy": sum(stop_outcomes) / len(stop_outcomes),
        }
    mode_vs_off = {
        mode: _paired_comparison(
            grouped,
            baseline_mode="off",
            active_mode=mode,
        )
        for mode in active_modes
        if mode != "off"
    }
    task_full_vs_core = _paired_comparison(
        grouped,
        baseline_mode="task_core",
        active_mode="task_full",
    )
    task_full_vs_core_at_adopted = _paired_comparison(
        grouped,
        baseline_mode="task_core",
        active_mode="task_full",
        adopted_only=True,
    )
    task_full_vs_flat = _paired_comparison(
        grouped,
        baseline_mode="flat",
        active_mode="task_full",
    )
    task_full_vs_shuffled = (
        _paired_comparison(
            grouped,
            baseline_mode="shuffled_full",
            active_mode="task_full",
        )
        if "shuffled_full" in active_modes
        else None
    )
    task_comparison = mode_vs_off["task_full"]
    full_corrected = set(task_comparison["wrong_to_acceptable_state_indices"])
    if "shuffled_full" not in active_modes:
        attributable_behavior = task_full_vs_core_at_adopted["behavior"]
        gate_checks = {
            "conditional_rule_corrects_core_error": bool(
                task_full_vs_core_at_adopted["wrong_to_acceptable_state_indices"]
            ),
            "conditional_rule_helpful_exceeds_harmful": (
                attributable_behavior["helpful_changes"]
                > attributable_behavior["harmful_changes"]
            ),
            "task_full_not_worse_than_core_on_adopted_states": (
                task_full_vs_core_at_adopted["active_acceptable_count"]
                >= task_full_vs_core_at_adopted["baseline_acceptable_count"]
            ),
            "receipts_complete": receipts_complete,
        }
    else:
        shuffled_corrected = set(
            mode_vs_off["shuffled_full"]["wrong_to_acceptable_state_indices"]
        )
        gate_checks = {
            "task_full_corrects_off_error": bool(full_corrected),
            "helpful_exceeds_harmful": (
                task_comparison["behavior"]["helpful_changes"]
                > task_comparison["behavior"]["harmful_changes"]
            ),
            "acceptable_rate_not_lower_than_off": (
                condition_metrics["task_full"]["exact_acceptable_count"]
                >= condition_metrics["off"]["exact_acceptable_count"]
            ),
            "conditional_rule_positive_change": bool(
                task_full_vs_core_at_adopted["wrong_to_acceptable_state_indices"]
            ),
            "shuffled_does_not_share_full_correction": not bool(
                full_corrected.intersection(shuffled_corrected)
            ),
            "receipts_complete": receipts_complete,
        }
    return {
        "schema": "roboharn_evo/esi_bench/frozen_probe_summary/v3",
        "probe_profile": profile,
        "state_count": len(grouped),
        "condition_metrics": condition_metrics,
        "mode_vs_off": mode_vs_off,
        "task_full_vs_core": task_full_vs_core,
        "task_full_vs_core_at_adopted_states": task_full_vs_core_at_adopted,
        "task_full_vs_flat": task_full_vs_flat,
        "task_full_vs_shuffled_full": task_full_vs_shuffled,
        "task_vs_off_behavior": task_comparison["behavior"],
        "wrong_to_acceptable_state_indices": task_comparison[
            "wrong_to_acceptable_state_indices"
        ],
        "acceptable_to_wrong_state_indices": task_comparison[
            "acceptable_to_wrong_state_indices"
        ],
        "mechanism_gate_checks": gate_checks,
        "mechanism_gate_passed": all(gate_checks.values()),
    }


def _annotation_contents(packets: Sequence[Mapping[str, Any]]) -> list[Any]:
    contents: list[Any] = []
    for index, packet in enumerate(packets, 1):
        contents.append(
            f"STATE {index}\nPublic question and action contract:\n{packet['question_or_goal']}"
        )
        contents.append(Path(str(packet["current_image_path"])))
    return contents


def _runtime_by_mode(
    *,
    service_url: str,
    store_root: Path,
    shuffle_path: Path | None,
    modes: Sequence[str],
) -> tuple[
    dict[str, ESITaskOnlyRetrievalRuntime], AgentApiESIMultimodalRetrievalBackend
]:
    store = load_frozen_task_store(store_root)
    active_modes = tuple(modes)
    shuffle = (
        load_shuffle_manifest(shuffle_path)
        if "shuffled_full" in active_modes and shuffle_path is not None
        else None
    )
    backend = AgentApiESIMultimodalRetrievalBackend(
        service_url, model="gpt-5.5", reasoning_effort="xhigh"
    )
    task_retriever = VLMCurrentImageSubtaskKnowledgeRetriever(backend)
    flat_retriever = VLMFlatLessonRetriever(backend)
    return (
        {
            "off": ESITaskOnlyRetrievalRuntime(
                mode="off", candidate_cap=2, context_token_cap=256
            ),
            "flat": ESITaskOnlyRetrievalRuntime(
                mode="flat",
                candidate_cap=2,
                context_token_cap=256,
                store=store,
                flat_retriever=flat_retriever,
            ),
            "task": ESITaskOnlyRetrievalRuntime(
                mode="task",
                candidate_cap=2,
                context_token_cap=256,
                store=store,
                task_retriever=task_retriever,
            ),
            **(
                {
                    "shuffled": ESITaskOnlyRetrievalRuntime(
                        mode="shuffled",
                        candidate_cap=2,
                        context_token_cap=256,
                        store=store,
                        task_retriever=task_retriever,
                        shuffle_manifest=shuffle,
                    )
                }
                if "shuffled_full" in active_modes
                else {}
            ),
        },
        backend,
    )


def _shared_retrieval_view(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must be identical for Core and Full ablation prompts."""

    ignored = {"prompt_token_count", "context_token_count"}
    return {
        key: copy.deepcopy(item) for key, item in value.items() if key not in ignored
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen ESI mechanism probe.")
    parser.add_argument("--frozen-states", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--shuffle-manifest", type=Path)
    parser.add_argument(
        "--probe-profile",
        choices=tuple(PROBE_PROFILES),
        default="five_condition",
    )
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    modes = modes_for_profile(args.probe_profile)
    if ("shuffled_full" in modes) != (args.shuffle_manifest is not None):
        raise FrozenProbeError(
            "shuffle manifest must be supplied exactly for the five-condition profile"
        )
    output = args.output_root.expanduser().resolve()
    if not output.is_relative_to(RUNS_ROOT):
        raise FrozenProbeError("frozen probe output must stay under runs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"frozen probe output is not empty: {output}")
    packets = _read_jsonl(args.frozen_states.resolve(), "frozen states")
    capture_receipt = _read_object(args.capture_receipt.resolve(), "capture receipt")
    if (
        capture_receipt.get("status")
        not in {"completed", "completed_with_shutdown_warnings"}
        or capture_receipt.get("completed_state_count") != len(packets)
        or capture_receipt.get("evaluated_model_calls") != 0
        or capture_receipt.get("retrieval_model_calls") != 0
        or capture_receipt.get("executed_actions") != 0
        or capture_receipt.get("automatic_retries") != 0
        or capture_receipt.get("probe_profile", "five_condition") != args.probe_profile
    ):
        raise FrozenProbeError("capture receipt is incomplete or contaminated")
    expected_state_count = 12 if args.probe_profile == "five_condition" else 8
    if len(packets) != expected_state_count or [
        item.get("instance_ref") for item in packets
    ] != [item.get("instance_ref") for item in capture_receipt["states"]]:
        raise FrozenProbeError("frozen states do not match the capture receipt")
    for packet in packets:
        image = Path(str(packet.get("current_image_path") or ""))
        if (
            packet.get("public_history") != []
            or packet.get("answer_read") is not False
            or packet.get("score_read") is not False
            or not image.is_file()
        ):
            raise FrozenProbeError("frozen state violates the public-state boundary")
    output.mkdir(parents=True, exist_ok=True)
    budget = probe_call_budget(len(packets), modes=modes)
    receipt: dict[str, Any] = {
        "schema": "roboharn_evo/esi_bench/frozen_probe_receipt/v2",
        "probe_profile": args.probe_profile,
        "status": "running",
        "budget": budget,
        "annotation_calls": 0,
        "evaluated_model_calls": 0,
        "retrieval_model_calls": 0,
        "image_payloads": 0,
        "automatic_retries": 0,
        "simulator_episodes": 0,
        "executed_actions": 0,
    }
    _write_json(output / "probe_receipt.json", receipt)
    evaluated = AgentApiESIEvaluatedModel(
        args.service_url,
        model="gpt-5.5",
        reasoning_effort="xhigh",
    )
    health = evaluated.preflight()
    if int(health["responses_capabilities"]["max_images_per_request"]) < len(packets):
        raise FrozenProbeError("service cannot accept the blinded annotation image set")
    annotations_raw, _raw, _status = evaluated.generate_json(
        contents=_annotation_contents(packets),
        system_instruction=(
            "Blindly annotate the next evidence-acquisition subtask and action for each ordered "
            "ESI state. Use concise, transferable lower-case verb phrases for acceptable next "
            "subtasks. "
            "Use only its public image and public question. Do not infer a benchmark answer, "
            "hidden state, pose, or outcome. Give every camera action that is reasonably useful "
            "for acquiring the missing visible evidence; use stop only when the current image "
            "already contains enough decisive visual evidence. The acceptable action families "
            "must exactly describe the acceptable actions. Return the states in exact order."
        ),
        response_schema=annotation_schema(len(packets)),
        schema_name="hpk_v3_esi_frozen_state_annotations",
        max_output_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        fallback=None,
    )
    annotations = validate_annotations(annotations_raw, state_count=len(packets))
    _write_json(
        output / "annotations.json",
        {
            "states": list(annotations),
            "call_audit": copy.deepcopy(evaluated.last_call_audit),
        },
    )
    receipt["annotation_calls"] = 1
    receipt["image_payloads"] = len(packets)
    _write_json(output / "probe_receipt.json", receipt)
    runtimes, retrieval_backend = _runtime_by_mode(
        service_url=args.service_url,
        store_root=args.store_root.resolve(),
        shuffle_path=(
            None if args.shuffle_manifest is None else args.shuffle_manifest.resolve()
        ),
        modes=modes,
    )
    retrieval_backend.preflight()
    records = []
    retrieval_records = []
    annotation_by_state = {int(item["state_index"]): item for item in annotations}
    decisions_path = output / "decisions.jsonl"
    retrievals_path = output / "retrievals.jsonl"
    for state_index, packet in enumerate(packets, 1):
        annotation = annotation_by_state[state_index]
        acceptable = {
            _normalize_action(item) for item in annotation["acceptable_actions"]
        }
        acceptable_families = set(annotation["acceptable_action_families"])
        image = Path(str(packet["current_image_path"]))
        context = ESIPublicStepContext(
            small_task=str(packet["small_task"]),
            big_task=str(packet["big_task"]),
            question_or_goal=str(packet["question_or_goal"]),
            step=1,
            history=(),
            public_object_roles=(),
            evidence_status="no public action-observation evidence has been collected",
        )
        projection_cache = {}
        retrieval_call_audit_by_mode = {}
        retrieval_scope_by_mode = {}
        for call_position, mode in enumerate(
            rotated_mode_order(state_index, modes=modes), 1
        ):
            if mode not in projection_cache:
                if mode == "off":
                    projection_cache[mode] = runtimes["off"].project(
                        official_prompt=str(packet["question_or_goal"]),
                        model_arguments={},
                        context=context,
                    )
                    retrieval_call_audit_by_mode[mode] = None
                    retrieval_scope_by_mode[mode] = "none"
                elif mode == "flat":
                    projection_cache[mode] = runtimes["flat"].project(
                        official_prompt=str(packet["question_or_goal"]),
                        model_arguments={},
                        context=context,
                        current_image_path=image,
                    )
                    provider_audit = copy.deepcopy(retrieval_backend.last_call_audit)
                    retrieval_call_audit_by_mode[mode] = provider_audit
                    retrieval_scope_by_mode[mode] = "flat"
                    retrieval_record = {
                        "schema": "roboharn_evo/esi_bench/frozen_retrieval/v2",
                        "state_index": state_index,
                        "scope": "flat",
                        "trigger_mode": mode,
                        "retrieval": projection_cache[mode].audit.to_dict(),
                        "provider_call_audit": provider_audit,
                    }
                    retrieval_records.append(retrieval_record)
                    with retrievals_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                retrieval_record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    receipt["retrieval_model_calls"] += 1
                    receipt["image_payloads"] += 1
                elif mode in {"task_core", "task_full"}:
                    pair = runtimes["task"].project_task_variants(
                        official_prompt=str(packet["question_or_goal"]),
                        model_arguments={},
                        context=context,
                        current_image_path=image,
                    )
                    core_audit = pair.core.audit.to_dict()
                    full_audit = pair.full.audit.to_dict()
                    if _shared_retrieval_view(core_audit) != _shared_retrieval_view(
                        full_audit
                    ):
                        raise FrozenProbeError(
                            "Task-Core and Task-Full do not share one retrieval result"
                        )
                    projection_cache["task_core"] = pair.core
                    projection_cache["task_full"] = pair.full
                    provider_audit = copy.deepcopy(retrieval_backend.last_call_audit)
                    retrieval_call_audit_by_mode["task_core"] = provider_audit
                    retrieval_call_audit_by_mode["task_full"] = provider_audit
                    retrieval_scope_by_mode["task_core"] = "task_core_full_shared"
                    retrieval_scope_by_mode["task_full"] = "task_core_full_shared"
                    retrieval_record = {
                        "schema": "roboharn_evo/esi_bench/frozen_retrieval/v2",
                        "state_index": state_index,
                        "scope": "task_core_full_shared",
                        "trigger_mode": mode,
                        "retrieval": _shared_retrieval_view(core_audit),
                        "provider_call_audit": provider_audit,
                    }
                    retrieval_records.append(retrieval_record)
                    with retrievals_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                retrieval_record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    receipt["retrieval_model_calls"] += 1
                    receipt["image_payloads"] += 1
                elif mode == "shuffled_full":
                    pair = runtimes["shuffled"].project_task_variants(
                        official_prompt=str(packet["question_or_goal"]),
                        model_arguments={},
                        context=context,
                        current_image_path=image,
                    )
                    projection_cache[mode] = pair.full
                    provider_audit = copy.deepcopy(retrieval_backend.last_call_audit)
                    retrieval_call_audit_by_mode[mode] = provider_audit
                    retrieval_scope_by_mode[mode] = "shuffled_full"
                    retrieval_record = {
                        "schema": "roboharn_evo/esi_bench/frozen_retrieval/v2",
                        "state_index": state_index,
                        "scope": "shuffled_full",
                        "trigger_mode": mode,
                        "retrieval": pair.full.audit.to_dict(),
                        "provider_call_audit": provider_audit,
                    }
                    retrieval_records.append(retrieval_record)
                    with retrievals_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                retrieval_record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    receipt["retrieval_model_calls"] += 1
                    receipt["image_payloads"] += 1
                else:
                    raise FrozenProbeError(f"unsupported probe mode: {mode}")
            projection = projection_cache[mode]
            retrieval = projection.audit.to_dict()
            if mode == "off":
                retrieval_call_audit = None
            else:
                if retrieval["retrieval_calls"] != 1:
                    raise FrozenProbeError(
                        "active condition lost its matched retrieval call"
                    )
                retrieval_call_audit = retrieval_call_audit_by_mode[mode]
            decision, raw_text, status = evaluated.generate_json(
                contents=["[CURRENT VIEW - step 1]", image, projection.prompt],
                system_instruction=(
                    "Follow the supplied ESI task and choose the next public action using only "
                    "the current image and prompt. HPK text, when present, is advisory. Return "
                    "one concise lower-case next-subtask verb phrase, the normalized action, and "
                    "whether to stop or continue. Do not repeat the prompt's example JSON field "
                    "names or infer hidden simulator state."
                ),
                response_schema=decision_schema(),
                schema_name="hpk_v3_esi_frozen_decision",
                max_output_tokens=1024,
                temperature=0.0,
                top_p=1.0,
                fallback=None,
            )
            if status != "completed" or set(decision) != {
                "action",
                "next_subtask",
                "stop_or_continue",
                "current_answer",
                "reasoning",
                "confidence",
            }:
                raise FrozenProbeError("evaluated frozen decision is invalid")
            action = _normalize_action(decision["action"])
            family = official_action_family(action)
            record = {
                "schema": "roboharn_evo/esi_bench/frozen_decision/v2",
                "probe_profile": args.probe_profile,
                "state_index": state_index,
                "instance_ref": packet["instance_ref"],
                "small_task": packet["small_task"],
                "mode": mode,
                "call_position": call_position,
                "decision": {**decision, "action": action},
                "raw_output": raw_text,
                "retrieval": retrieval,
                "retrieval_scope": retrieval_scope_by_mode[mode],
                "retrieval_call_audit": retrieval_call_audit,
                "evaluated_call_audit": copy.deepcopy(evaluated.last_call_audit),
                "acceptable_actions": sorted(acceptable),
                "acceptable_next_subtasks": list(
                    annotation["acceptable_next_subtasks"]
                ),
                "exact_action_acceptable": action in acceptable,
                "action_family": family,
                "acceptable_families": sorted(acceptable_families),
                "action_family_acceptable": family in acceptable_families,
                "next_subtask_acceptable": _normalize_text(decision["next_subtask"])
                in {
                    _normalize_text(item)
                    for item in annotation["acceptable_next_subtasks"]
                },
                "stop_continue_acceptable": (
                    decision["stop_or_continue"] == annotation["stop_or_continue"]
                ),
            }
            records.append(record)
            with decisions_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            receipt["evaluated_model_calls"] += 1
            receipt["image_payloads"] += 1
            _write_json(output / "probe_receipt.json", receipt)
    usage_mismatch = (
        receipt["annotation_calls"] != budget["annotation_calls"]
        or receipt["evaluated_model_calls"] != budget["evaluated_model_calls"]
        or receipt["retrieval_model_calls"] != budget["retrieval_model_calls"]
        or receipt["image_payloads"] != budget["total_image_payloads"]
        or len(retrieval_records)
        != len(packets) * (3 if "shuffled_full" in modes else 2)
        or any(
            not _complete_call_audit(item["provider_call_audit"])
            for item in retrieval_records
        )
        or any(
            not _complete_call_audit(item["evaluated_call_audit"]) for item in records
        )
    )
    if usage_mismatch:
        raise FrozenProbeError("frozen probe usage differs from preregistered budget")
    summary = summarize_probe(
        records,
        annotations,
        receipts_complete=True,
        modes=modes,
    )
    _write_json(output / "summary.json", summary)
    receipt["status"] = "completed"
    receipt["mechanism_gate_passed"] = summary["mechanism_gate_passed"]
    _write_json(output / "probe_receipt.json", receipt)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Materialize one benchmark-correct ESI audit as a model-visible source trace."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_experiment_metadata
from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    AgentApiESIEvaluatedModel,
    ESIKnowledgeError,
    ESITraceRecord,
    ESITraceStep,
    validate_no_transfer_leakage,
)

_SUMMARY_INSTRUCTIONS = """Describe only visible evidence in each ordered ESI image.

For every supplied step, report one concise observation summary describing visible shape, parts, surface structure, occlusion, scale, and the change from earlier views when relevant. Use category-neutral phrases such as "the target" or "the visible form". Never name an object category, product, species, or answer option, even when visually obvious. Do not use model reasoning as evidence. Do not mention paths, IDs, scene names, room names, camera poses, coordinates, hidden state, ground truth, or oracle information. Return strict JSON only."""

_PUBLIC_RESULT_KEYS = (
    "handled",
    "operation",
    "action",
    "success",
    "error",
    "reason",
)


def observation_summary_schema(step_count: int) -> dict[str, Any]:
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 1
    ):
        raise ESIKnowledgeError("step_count must be a positive integer")
    item = {
        "type": "object",
        "properties": {
            "step": {"type": "integer", "minimum": 1},
            "observation_summary": {"type": "string", "minLength": 1},
        },
        "required": ["step", "observation_summary"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "summaries": {
                "type": "array",
                "items": item,
                "minItems": step_count,
                "maxItems": step_count,
            }
        },
        "required": ["summaries"],
        "additionalProperties": False,
    }


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ESIKnowledgeError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise ESIKnowledgeError(f"{label} must be one object")
    return dict(value)


def _structured_public_choices(value: Any) -> tuple[str, ...]:
    choices: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            text = " ".join(item.strip().split())
            if text and text.casefold() not in {"not sure", "unsure", "unknown"}:
                choices.append(text)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(dict.fromkeys(choices))


def _public_choice_names(
    prompt: str,
    structured_options: Any = None,
) -> tuple[str, ...]:
    structured = _structured_public_choices(structured_options)
    if structured:
        return structured
    match = re.search(
        r"Answer choices:\s*(.*?)\s*Available camera actions:",
        prompt,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ESIKnowledgeError("cannot locate public answer choices")
    section = match.group(1)
    choices = re.findall(
        r"(?:^|\s)[A-Z]\.\s*(.*?)(?=\s+[A-Z]\.\s*|$)",
        section,
    )
    if len(choices) < 2:
        choices = re.findall(
            r"(?:^|\s)-\s+(.*?)(?=\s+-\s+|$)",
            section,
        )
    normalized = tuple(
        text
        for item in choices
        if (text := " ".join(item.strip().split())).casefold()
        not in {"not sure", "unsure", "unknown"}
    )
    if len(normalized) < 2:
        raise ESIKnowledgeError("public answer choices are incomplete")
    return normalized


def load_source_episode(audit_root: Path) -> dict[str, Any]:
    root = audit_root.expanduser().resolve()
    run_config = normalize_hpk_experiment_metadata(
        _read_object(root / "run_config.json", "run config")
    )
    result = _read_object(root / "upstream_answer.json", "upstream answer")
    question = _read_object(root / "question_public.json", "public question")
    if run_config.get("hpk_mode") != "off" or run_config.get("split_part") != "source":
        raise ESIKnowledgeError("source trace requires HPK-off source-split audit")
    split_entry = run_config.get("split_entry")
    if not isinstance(split_entry, Mapping):
        raise ESIKnowledgeError("source trace audit has no split entry")
    if result.get("correct") is not True:
        raise ESIKnowledgeError(
            "only benchmark-correct source episodes are materialized"
        )
    history = result.get("history")
    if not isinstance(history, list):
        raise ESIKnowledgeError("source episode history must be an array")
    action_steps = [
        dict(item)
        for item in history
        if isinstance(item, Mapping) and item.get("action") != "force_final_choice"
    ]
    if [item.get("step") for item in action_steps] != list(
        range(1, len(action_steps) + 1)
    ):
        raise ESIKnowledgeError("source episode action steps must be contiguous")
    image_root = Path(str(result.get("step_image_dir") or "")).expanduser().resolve()
    image_paths = []
    for item in action_steps:
        image = image_root / str(item.get("image") or "")
        if not image.is_file():
            raise ESIKnowledgeError("source episode step image is missing")
        image_paths.append(image)
    final = result.get("final_answer")
    if not isinstance(final, Mapping):
        raise ESIKnowledgeError("source episode final answer is missing")
    prompt = str(question.get("question_or_goal") or "")
    return {
        "source_ref": str(split_entry.get("instance_ref") or ""),
        "small_task": str(split_entry.get("small_task") or ""),
        "big_task": str(split_entry.get("big_task") or ""),
        "question_text": str(split_entry.get("question_text") or ""),
        "public_object_names": _public_choice_names(prompt, result.get("options")),
        "action_steps": tuple(action_steps),
        "image_paths": tuple(image_paths),
        "final_answer": str(final.get("answer") or ""),
        "confidence": float(final.get("confidence", 0.0)),
        "stop_reason": str(final.get("stopped_by") or "unknown"),
    }


def validate_summary_response(
    value: Mapping[str, Any],
    *,
    step_count: int,
    forbidden_literals: Sequence[str],
) -> tuple[str, ...]:
    if set(value) != {"summaries"} or not isinstance(value["summaries"], list):
        raise ESIKnowledgeError("observation summary response fields mismatch")
    summaries = value["summaries"]
    if [
        item.get("step") if isinstance(item, Mapping) else None for item in summaries
    ] != list(range(1, step_count + 1)):
        raise ESIKnowledgeError("observation summaries must cover ordered trace steps")
    texts = tuple(
        " ".join(str(item.get("observation_summary") or "").strip().split())
        for item in summaries
    )
    if any(not item for item in texts):
        raise ESIKnowledgeError("observation summaries must be non-empty")
    validate_no_transfer_leakage(
        list(texts),
        forbidden_literals=forbidden_literals,
        label="source observation summaries",
    )
    return texts


def materialize_trace(
    episode: Mapping[str, Any], summaries: Sequence[str]
) -> ESITraceRecord:
    action_steps = episode["action_steps"]
    if len(action_steps) != len(summaries):
        raise ESIKnowledgeError("trace action/summary counts differ")
    steps = []
    for item, image, summary in zip(
        action_steps, episode["image_paths"], summaries, strict=True
    ):
        result = item.get("action_result")
        if not isinstance(result, Mapping):
            result = {}
        steps.append(
            ESITraceStep(
                step=int(item["step"]),
                observation_path=str(image),
                observation_summary=summary,
                model_reasoning=str(item.get("reasoning") or "no reasoning provided"),
                selected_action=str(item.get("action") or "no action"),
                action_result_public={
                    key: result[key] for key in _PUBLIC_RESULT_KEYS if key in result
                },
            )
        )
    return ESITraceRecord(
        source_ref=str(episode["source_ref"]),
        small_task=str(episode["small_task"]),
        big_task=str(episode["big_task"]),
        question_text=str(episode["question_text"]),
        public_object_names=tuple(episode["public_object_names"]),
        steps=tuple(steps),
        final_answer=str(episode["final_answer"]),
        confidence=float(episode["confidence"]),
        benchmark_correct=True,
        stop_reason=str(episode["stop_reason"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add image-grounded summaries to one correct ESI source audit."
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--planner-url")
    group.add_argument("--summary-response", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"source trace output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    episode = load_source_episode(args.audit_root)
    if args.summary_response is not None:
        response = _read_object(
            args.summary_response.expanduser().resolve(), "summary response"
        )
        call_audit = {"external_model_calls": 0, "precomputed_response": True}
    else:
        transport = AgentApiESIEvaluatedModel(
            args.planner_url,
            model="gpt-5.5",
            reasoning_effort="xhigh",
        )
        health = transport.preflight()
        contents: list[Any] = [
            "Summarize the following ordered model-visible images without naming an answer option."
        ]
        for index, image in enumerate(episode["image_paths"], 1):
            contents.extend((f"[SOURCE VIEW STEP {index}]", image))
        response, _raw, _status = transport.generate_json(
            contents=contents,
            system_instruction=_SUMMARY_INSTRUCTIONS,
            response_schema=observation_summary_schema(len(episode["image_paths"])),
            schema_name="esi_source_observation_summaries_v1",
            max_output_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            fallback=None,
        )
        call_audit = {
            "external_model_calls": 1,
            "precomputed_response": False,
            "health_model": health.get("model"),
            "call": transport.last_call_audit,
        }
    (output / "raw_observation_summary.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summaries = validate_summary_response(
        response,
        step_count=len(episode["image_paths"]),
        forbidden_literals=episode["public_object_names"],
    )
    trace = materialize_trace(episode, summaries)
    (output / "source_trace.json").write_text(
        json.dumps(trace.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "source_trace.jsonl").write_text(
        json.dumps(trace.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (output / "materialization_audit.json").write_text(
        json.dumps(
            {
                **call_audit,
                "step_count": len(trace.steps),
                "benchmark_correct": True,
                "source_answer_sent_to_summary_model": False,
                "model_reasoning_used_as_observation": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(call_audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

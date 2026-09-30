from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    HPKV3ValidationError,
    TrajectoryKnowledgePackageV3,
    _enum,
    _mapping,
    _strings,
    _temporal_state,
    _text,
    trajectory_knowledge_package_json_schema,
)

_ACTIONS = frozenset({"grasp", "place", "contact"})
_VERDICTS = frozenset({"support", "oppose", "unverified"})

_SYSTEM_PROMPT = """You organize one robot trajectory into reusable manipulation knowledge.

Build exactly one ordered hierarchy:
1. the overall task and plan summary;
2. ordered subtasks;
3. for every subtask, its purpose, before state, observed facts, short inferred rationale, completion condition, and planned next subtask;
4. within every subtask, group action chunks into grasp, place, or contact knowledge;
5. for every action, describe relative geometry, a short strategy rationale, the expected physical effect, and one temporally correct evidence event per real execution.

The ordered input chunks are observations, not preclassified task stages. Infer action type, phase boundaries, before state, and after state from the instruction, raw annotation, sensor facts, ordered images, and optional runtime hints. Never infer them from chunk number or a repeated fixed schedule.

A grasp establishes control of an object with the selected gripper. A place transfers support of an already carried object to a destination and releases it. When a subtask contains acquisition followed by placement, represent the grasp and the place as separate action knowledge with their own execution boundaries. Group approach, closure, and the confirming lift within the grasp; begin place evidence after acquisition. Every action evidence before_state must describe the state at that action's start and satisfy its applicability condition. Account for the observed manipulation executions throughout the trajectory, including prerequisite acquisition, without relabelling a whole pick-and-place subtask as one place action.

Keep observed facts separate from inferred rationale. Keep held state, support relation, attachment, and other changing properties only in before_state or after_state, never in trajectory_context.objects. A closed gripper is not proof of attachment, and reaching a pose is not proof of a physical effect. A later observation may update the same action evidence with evidence_timing "delayed"; do not count immediate and delayed views as two executions. A supplied runtime verdict is authoritative. Without a runtime verdict, use support or oppose only when the ordered observations clearly establish the expected physical effect; otherwise use unverified and say what is missing.

For chunks carrying runtime evidence, preserve the global execution order. When the output is read in subtask order, action-knowledge order, then evidence order, copy that execution's action, observed arm, observed strategy, runtime verdict, and evidence timing exactly into the matching evidence event. Do not exchange evidence between two executions even when they have the same action type.

Use snake_case keys and short natural-language values. Do not output IDs, hashes, file paths, frame numbers, absolute positions, poses, candidate names, or motion trajectories. Return only strict JSON matching TrajectoryKnowledgePackageV3."""


class HierarchicalReflectionBackend(Protocol):
    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[Any],
        output_schema: Mapping[str, Any],
        schema_name: str,
    ) -> HierarchicalBackendCompletion: ...


@dataclass(frozen=True, slots=True)
class HierarchicalBackendCompletion:
    output: Mapping[str, Any] | str
    audit: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class HierarchicalReflectionResult:
    package: TrajectoryKnowledgePackageV3
    raw_response: dict[str, Any] | str
    evidence_bindings: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package.to_dict(),
            "raw_response": copy.deepcopy(self.raw_response),
            "evidence_bindings": copy.deepcopy(list(self.evidence_bindings)),
        }


class ExistingGPTMultimodalReflectorBackend:
    """Thin adapter over the repository's existing GPT multimodal transport."""

    def __init__(self, transport: Any, *, authorization: Any) -> None:
        if not callable(getattr(transport, "complete", None)):
            raise TypeError("transport must expose complete")
        self._transport = transport
        self._authorization = authorization

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[Any],
        output_schema: Mapping[str, Any],
        schema_name: str,
    ) -> HierarchicalBackendCompletion:
        if images:
            completion = self._transport.complete(
                instructions=instructions,
                input_text=input_text,
                images=images,
                output_schema=output_schema,
                schema_name=schema_name,
                authorization=self._authorization,
            )
        else:
            complete_text = getattr(self._transport, "complete_text", None)
            if not callable(complete_text):
                raise TypeError(
                    "transport must expose complete_text for image-free input"
                )
            completion = complete_text(
                instructions=instructions,
                input_text=input_text,
                output_schema=output_schema,
                schema_name=schema_name,
                authorization=self._authorization,
                purpose="hpk_v3_hierarchical_reflection",
            )
        return HierarchicalBackendCompletion(
            output=completion.output,
            audit=getattr(completion, "audit", None),
        )


def _input_state(value: Any, *, path: str) -> dict[str, Any]:
    return _temporal_state(value, path=path)


def _input_chunk(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    required = {
        "annotation",
        "observed_facts",
        "visual_observations",
    }
    optional = {
        "action_hint",
        "phase_hint",
        "active_arm_observation",
        "before_state",
        "after_state",
        "runtime_observed_strategy",
        "runtime_verdict",
        "runtime_evidence_timing",
    }
    actual = set(payload)
    if actual - required - optional or required - actual:
        raise HPKV3ValidationError(
            f"{path}: fields mismatch: missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - required - optional)}"
        )
    result = {
        "annotation": _text(payload["annotation"], path=f"{path}.annotation"),
        "observed_facts": _strings(
            payload["observed_facts"], path=f"{path}.observed_facts"
        ),
        "visual_observations": _strings(
            payload["visual_observations"], path=f"{path}.visual_observations"
        ),
    }
    if payload.get("action_hint") is not None:
        result["action_hint"] = _enum(
            payload["action_hint"], _ACTIONS, path=f"{path}.action_hint"
        )
    for field in (
        "phase_hint",
        "active_arm_observation",
        "runtime_observed_strategy",
    ):
        if payload.get(field) is not None:
            result[field] = _text(payload[field], path=f"{path}.{field}", max_chars=300)
    for field in ("before_state", "after_state"):
        if payload.get(field) is not None:
            result[field] = _input_state(payload[field], path=f"{path}.{field}")
    if payload.get("runtime_verdict") is not None:
        result["runtime_verdict"] = _enum(
            payload["runtime_verdict"],
            _VERDICTS,
            path=f"{path}.runtime_verdict",
        )
    if payload.get("runtime_evidence_timing") is not None:
        result["runtime_evidence_timing"] = _enum(
            payload["runtime_evidence_timing"],
            {"immediate", "delayed"},
            path=f"{path}.runtime_evidence_timing",
        )
    return {key: item for key, item in result.items() if item is not None}


def build_hierarchical_reflection_input(
    *,
    instruction: str,
    ordered_action_chunks: Sequence[Mapping[str, Any]],
    key_observations: Sequence[str] = (),
    available_state: Sequence[str] = (),
    image_count: int = 0,
) -> str:
    """Build the compact, semantic-only JSON sent to the reflector backend."""

    task = _text(instruction, path="instruction", max_chars=2000)
    if isinstance(ordered_action_chunks, (str, bytes)) or not isinstance(
        ordered_action_chunks, Sequence
    ):
        raise HPKV3ValidationError("ordered_action_chunks: must be an array")
    if not ordered_action_chunks:
        raise HPKV3ValidationError(
            "ordered_action_chunks: must contain at least one action chunk"
        )
    chunks = [
        _input_chunk(item, path=f"ordered_action_chunks[{index}]")
        for index, item in enumerate(ordered_action_chunks)
    ]
    observations = _strings(
        key_observations,
        path="key_observations",
        max_items=64,
        unique=False,
    )
    state = _strings(available_state, path="available_state", max_items=64)
    if (
        isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or image_count < 0
    ):
        raise HPKV3ValidationError("image_count: must be a non-negative integer")
    if image_count and image_count != len(observations):
        raise HPKV3ValidationError(
            "key_observations: must describe each supplied image in the same order"
        )
    payload = {
        "instruction": task,
        "ordered_action_chunks": chunks,
        "key_observations": observations,
        "available_state": state,
        "image_binding": (
            "each supplied image corresponds to the key observation at the same position"
            if image_count
            else "no images were supplied"
        ),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _strict_response(value: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        try:
            raw = json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise HPKV3ValidationError("model output must be finite JSON") from exc
        assert isinstance(parsed, dict)
        return parsed
    if not isinstance(value, str):
        raise HPKV3ValidationError("model output must be a JSON object or JSON text")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise HPKV3ValidationError(
                    f"model output contains duplicate key {key!r}"
                )
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                HPKV3ValidationError(
                    f"model output contains non-standard constant {constant}"
                )
            ),
        )
    except HPKV3ValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise HPKV3ValidationError("model output is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise HPKV3ValidationError("model output must be a JSON object")
    return parsed


class VLMHierarchicalReflector:
    """Produce one strict hierarchical package in one backend call, with no retry."""

    def __init__(self, backend: HierarchicalReflectionBackend) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must expose complete")
        self._backend = backend
        self._last_result: HierarchicalReflectionResult | None = None
        self._last_raw_response: dict[str, Any] | str | None = None

    @property
    def last_result(self) -> HierarchicalReflectionResult | None:
        return self._last_result

    @property
    def last_raw_response(self) -> dict[str, Any] | str | None:
        return copy.deepcopy(self._last_raw_response)

    def reflect(
        self,
        *,
        instruction: str,
        ordered_action_chunks: Sequence[Mapping[str, Any]],
        key_observations: Sequence[str] = (),
        available_state: Sequence[str] = (),
        images: Sequence[Any] = (),
        bind_rgb_evidence: bool = False,
        review_package: Mapping[str, Any] | None = None,
        bind_existing_package: bool = False,
    ) -> HierarchicalReflectionResult:
        self._last_result = None
        self._last_raw_response = None
        if bind_existing_package and (not bind_rgb_evidence or review_package is None):
            raise ValueError("binding an existing package requires its content and RGB observations")
        bounded_images = tuple(images)
        input_text = build_hierarchical_reflection_input(
            instruction=instruction,
            ordered_action_chunks=ordered_action_chunks,
            key_observations=key_observations,
            available_state=available_state,
            image_count=len(bounded_images),
        )
        schema = trajectory_knowledge_package_json_schema()
        instructions = _SYSTEM_PROMPT
        if bind_rgb_evidence:
            indexed_input = json.loads(input_text)
            indexed_input["ordered_action_chunks"] = [{"chunk_index": index, **chunk} for index, chunk in enumerate(indexed_input["ordered_action_chunks"])]
            indexed_input["key_observations"] = [{"observation_index": index, "description": description} for index, description in enumerate(indexed_input["key_observations"])]
            input_text = json.dumps(indexed_input, ensure_ascii=False, separators=(",", ":"))
            instructions += "\nUse the explicit chunk_index and observation_index fields in evidence bindings. Cover the complete supplied trajectory: the ordered subtask execution ranges must account for every supplied chunk, from the first to the last, including preparatory and withdrawal motion within the appropriate subtask."
        if review_package is not None:
            review_payload = json.loads(input_text)
            review_payload["draft_package"] = TrajectoryKnowledgePackageV3(review_package).to_dict()
            input_text = json.dumps(review_payload, ensure_ascii=False, separators=(",", ":"))
            if not bind_existing_package:
                instructions += "\nRe-examine the supplied draft hierarchy using the full ordered observations. Return a complete package whose primitive execution boundaries, before states, applicability conditions, after states, and evidence observations describe the same actual actions. Include the observed acquisition and support-transfer actions separately."
        if bind_rgb_evidence:
            integer = {"type": "integer", "minimum": 0}
            nullable_integer = {"anyOf": [integer, {"type": "null"}]}
            properties = {
                "subtask_index": integer,
                "action_index": nullable_integer,
                "evidence_index": nullable_integer,
                "chunk_start": integer,
                "chunk_end": integer,
                "before_observation": integer,
                "after_observation": integer,
                "delayed_observations": {"type": "array", "items": integer},
                "task_verdict": {"anyOf": [{"type": "string", "enum": ["support", "oppose", "unverified"]}, {"type": "null"}]},
                "observed_result": {"type": "string"},
                "missing_evidence": {"type": "string"},
            }
            schema = {"type": "object", "properties": {
                "package": schema,
                "evidence_bindings": {"type": "array", "items": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}},
            }, "required": ["package", "evidence_bindings"], "additionalProperties": False}
            instructions += "\nReturn a package and a separate evidence_bindings array. The package follows the knowledge schema. Bindings contain temporary zero-based array positions only: subtask_index, action_index within that subtask, evidence_index within that action, inclusive chunk_start and chunk_end in the ordered input, and before, after, and delayed observation positions. Produce exactly one binding per subtask (action_index and evidence_index null), and one per action evidence event. Before and after images must bracket the entire declared execution interval. Every delayed_observations index must be strictly greater than after_observation. When the after image already contains the confirming observation, use an empty delayed_observations array; never repeat the after image as delayed evidence. Subtask task_verdict evaluates its own completion_condition independently of its actions and of overall task success. For action bindings task_verdict is null and observed_result describes that same package evidence event. The before image precedes execution, the after image follows execution, and later transport images can supply delayed attachment evidence. Keep all indices outside the package."
            if bind_existing_package:
                schema = {"type": "object", "properties": {"evidence_bindings": schema["properties"]["evidence_bindings"]}, "required": ["evidence_bindings"], "additionalProperties": False}
                instructions += "\nFor this request the draft_package is fixed. Return only evidence_bindings for its existing subtask and action execution events. Inspect the actual ordered source images and chunk boundaries for every binding."
        completion = self._backend.complete(
            instructions=instructions,
            input_text=input_text,
            images=bounded_images,
            output_schema=schema,
            schema_name="hpk_v3_trajectory_knowledge_package",
        )
        if isinstance(completion.output, Mapping):
            self._last_raw_response = copy.deepcopy(dict(completion.output))
        else:
            self._last_raw_response = completion.output
        parsed = _strict_response(completion.output)
        package = TrajectoryKnowledgePackageV3(review_package if bind_existing_package else parsed["package"] if bind_rgb_evidence else parsed)
        raw_response: dict[str, Any] | str
        if isinstance(completion.output, Mapping):
            raw_response = copy.deepcopy(dict(completion.output))
        else:
            raw_response = completion.output
        result = HierarchicalReflectionResult(
            package=package,
            raw_response=raw_response,
            evidence_bindings=tuple(parsed["evidence_bindings"]) if bind_rgb_evidence else (),
        )
        self._last_result = result
        return result


__all__ = [
    "ExistingGPTMultimodalReflectorBackend",
    "HierarchicalBackendCompletion",
    "HierarchicalReflectionBackend",
    "HierarchicalReflectionResult",
    "VLMHierarchicalReflector",
    "build_hierarchical_reflection_input",
]

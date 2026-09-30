from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from roboharn_evo.agent.hpk.compatibility import normalize_v3_rollout_record
from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    HPKV3ValidationError,
    TrajectoryKnowledgePackageV3,
    atomize_package,
)
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    VLMHierarchicalReflector,
)
from roboharn_evo.agent.reflector.multimodal_transport import MultimodalImage
from roboharn_evo.agent.hpk.rgb_evidence import bind_reflection_evidence


@dataclass(frozen=True, slots=True)
class RealRolloutReflectionInput:
    instruction: str
    ordered_action_chunks: tuple[dict[str, Any], ...]
    key_observations: tuple[str, ...]
    available_state: tuple[str, ...]
    images: tuple[MultimodalImage, ...]
    evidence_signature: tuple[tuple[str, str, str, str, str], ...]
    source_root: Path | None = None
    source_observations: Mapping[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HPKV3ValidationError(f"cannot read {path.name}") from exc
    if not isinstance(value, dict):
        raise HPKV3ValidationError(f"{path.name} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HPKV3ValidationError(f"cannot read {path.name}") from exc
    records: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HPKV3ValidationError(
                f"{path.name} line {ordinal} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise HPKV3ValidationError(
                f"{path.name} line {ordinal} must contain an object"
            )
        records.append(normalize_v3_rollout_record(value))
    return tuple(records)


def _local_ref(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    metadata = value.get("_runtime_metadata")
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("local_ref", "") or "").strip()


def _natural_state(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result = {
        key: value[key]
        for key in (
            "held_state",
            "support_relation",
            "target_relation",
            "relevant_relations",
        )
        if key in value
    }
    if "relevant_relations" not in result:
        result["relevant_relations"] = []
    return result


def _nearest_frame(directory: Path, step: int, *, side: str) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("step_*.png"):
        try:
            value = int(path.stem.removeprefix("step_"))
        except ValueError:
            continue
        if (side == "before" and value <= step) or (side == "after" and value >= step):
            candidates.append((value, path))
    if not candidates:
        raise HPKV3ValidationError(f"rollout has no {side} head-camera observation bracketing step {step}")
    return min(candidates, key=lambda item: (abs(item[0] - step), item[0]))[1]


def _paired_image(
    before_path: Path,
    after_path: Path,
    *,
    output_path: Path,
) -> bytes:
    with (
        Image.open(before_path) as before_source,
        Image.open(after_path) as after_source,
    ):
        before = before_source.convert("RGB")
        after = after_source.convert("RGB")
        height = max(before.height, after.height)
        canvas = Image.new("RGB", (before.width + after.width, height), "white")
        canvas.paste(before, (0, 0))
        canvas.paste(after, (before.width, 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="PNG")
        buffer = BytesIO()
        canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def build_real_rollout_reflection_input(
    rollout_dir: str | Path,
    *,
    result: str,
    max_action_executions: int = 16,
    bind_rgb_evidence: bool = False,
) -> RealRolloutReflectionInput:
    """Project v3 action/evidence events into one ID-free reflector input."""

    root = Path(rollout_dir)
    if isinstance(max_action_executions, bool) or max_action_executions <= 0:
        raise ValueError("max_action_executions must be positive")
    meta = _read_json(root / "meta.json")
    instruction = str(meta.get("instruction", "") or "").strip()
    if not instruction:
        raise HPKV3ValidationError("rollout meta has no instruction")
    records = _read_jsonl(root / "events.jsonl")

    selections: dict[str, tuple[int, dict[str, Any]]] = {}
    evidence_updates: dict[str, tuple[int, dict[str, Any]]] = {}
    for ordinal, record in enumerate(records):
        if record.get("event") == "operation_candidate_selected":
            usage = record.get("hpk_v3_action_usage")
            ref = _local_ref(usage)
            if ref and ref not in selections and isinstance(usage, Mapping):
                selections[ref] = (ordinal, record)
        elif record.get("event") == "hpk_v3_action_evidence":
            ref = _local_ref(record)
            evidence = record.get("evidence")
            if ref and isinstance(evidence, Mapping):
                evidence_updates[ref] = (ordinal, record)

    ordered_refs = [
        ref
        for ref, _ in sorted(selections.items(), key=lambda item: item[1][0])
        if ref in evidence_updates
    ]
    if not ordered_refs:
        raise HPKV3ValidationError("rollout contains no completed v3 action evidence")
    if len(ordered_refs) > max_action_executions:
        raise HPKV3ValidationError(
            "rollout exceeds the configured complete-reflection action budget"
        )

    chunks: list[dict[str, Any]] = []
    observation_text: list[str] = []
    images: list[MultimodalImage] = []
    signatures: list[tuple[str, str, str, str, str]] = []
    source_chunks, source_frames = [], {}
    image_root = root / "hpk_v3_reflection" / "key_observations"
    for index, ref in enumerate(ordered_refs):
        _, selection = selections[ref]
        _, update = evidence_updates[ref]
        usage = selection["hpk_v3_action_usage"]
        evidence = update["evidence"]
        condition = usage.get("learning_condition")
        executed = evidence.get("executed_action")
        if not isinstance(condition, Mapping) or not isinstance(executed, Mapping):
            raise HPKV3ValidationError("v3 action evidence lacks semantic context")
        action = str(executed.get("action", "") or "").strip()
        if action != str(condition.get("action", "") or "").strip():
            raise HPKV3ValidationError("v3 action and learning condition disagree")
        verdict = str(evidence.get("verdict", "") or "").strip()
        timing = str(evidence.get("evidence_timing", "") or "").strip()
        executed_arm = str(executed.get("executed_arm", "") or "").strip()
        observed_strategy = str(executed.get("observed_strategy", "") or "").strip()
        signatures.append((action, executed_arm, observed_strategy, verdict, timing))
        facts = [
            str(condition.get("object_description", "") or "").strip(),
            str(evidence.get("observed_result", "") or "").strip(),
            (
                "the action execution status was "
                + str(evidence.get("execution_status", "") or "").strip()
            ),
        ]
        missing = str(evidence.get("missing_evidence", "") or "").strip()
        if missing:
            facts.append("missing evidence: " + missing)
        chunk: dict[str, Any] = {
            "annotation": (
                f"the runtime executed one {action} action with "
                f"{executed.get('executed_arm', 'the selected arm')!s}"
            ),
            "observed_facts": [value for value in facts if value],
            "visual_observations": [
                "the paired image shows the head-camera scene before and after this action execution"
            ],
            "action_hint": action,
            "active_arm_observation": str(
                executed.get("executed_arm", "the selected arm")
            ),
            "runtime_observed_strategy": observed_strategy,
            "runtime_verdict": verdict,
            "runtime_evidence_timing": timing,
        }
        before_state = _natural_state(evidence.get("before_state"))
        after_state = _natural_state(evidence.get("after_state"))
        if before_state is not None:
            chunk["before_state"] = before_state
        if after_state is not None:
            chunk["after_state"] = after_state
        chunks.append(chunk)

        try:
            before_step = int(selection.get("env_step", 0))
            after_step = int(update.get("env_step", before_step))
        except (TypeError, ValueError) as exc:
            raise HPKV3ValidationError("v3 action event has an invalid step") from exc
        before_path = _nearest_frame(root / "head", before_step, side="before")
        after_path = _nearest_frame(root / "head", after_step, side="after")
        if bind_rgb_evidence:
            for path in (before_path, after_path):
                frame = int(path.stem.removeprefix("step_"))
                source_frames[frame] = {"frame": frame, "timestamp_seconds": None, "images": {"head_camera": str(path.relative_to(root))}}
            source_chunks.append({"chunk_index": index, "start_frame": before_step, "end_frame": after_step})
            continue
        image_bytes = _paired_image(
            before_path,
            after_path,
            output_path=image_root / f"action_{index + 1:03d}.png",
        )
        observation_text.append(
            "paired head-camera observation before and after one ordered action execution"
        )
        images.append(
            MultimodalImage(
                evidence_id=f"action observation {index + 1}",
                mime_type="image/png",
                content=image_bytes,
                detail="high",
            )
        )

    source_observations = None
    if bind_rgb_evidence:
        observations = [source_frames[frame] for frame in sorted(source_frames)]
        for index, observation in enumerate(observations):
            images.append(MultimodalImage(evidence_id=f"observation {index}", mime_type="image/png", content=(root / observation["images"]["head_camera"]).read_bytes(), detail="high"))
            observation_text.append(f"Head-camera observation at position {index}")
        for chunk, source in zip(chunks, source_chunks, strict=True):
            before_index = max(index for index, observation in enumerate(observations) if observation["frame"] <= source["start_frame"])
            after_index = min(index for index, observation in enumerate(observations) if observation["frame"] >= source["end_frame"])
            chunk["visual_observations"] = [f"Before image at position {before_index}; after image at position {after_index}"]
        source_observations = {"source": {"execution_source": str((root / "events.jsonl").resolve()), "raw_files": {"events": "events.jsonl"}}, "observations": observations, "chunks": source_chunks}
    outcome = "success" if str(result).strip().casefold() == "success" else "failure"
    return RealRolloutReflectionInput(
        instruction=instruction,
        ordered_action_chunks=tuple(chunks),
        key_observations=tuple(observation_text),
        available_state=(
            f"the real episode ended with task {outcome}",
            f"the runtime recorded {len(chunks)} physical action executions",
        ),
        images=tuple(images),
        evidence_signature=tuple(signatures),
        source_root=root,
        source_observations=source_observations,
    )


def validate_reflected_evidence_authority(
    package: TrajectoryKnowledgePackageV3 | Mapping[str, Any],
    *,
    expected: Sequence[tuple[str, str, str, str, str]],
) -> None:
    """Preserve each ordered runtime execution and verdict exactly once."""

    typed = (
        package
        if isinstance(package, TrajectoryKnowledgePackageV3)
        else TrajectoryKnowledgePackageV3(package)
    )
    actual: list[tuple[str, str, str, str, str]] = []
    for subtask in typed["task_strategy"]["subtasks"]:
        for action in subtask["action_knowledge"]:
            for evidence in action["evidence"]:
                actual.append(
                    (
                        evidence["executed_action"]["action"],
                        evidence["executed_action"].get("executed_arm", ""),
                        evidence["executed_action"]["observed_strategy"],
                        evidence["verdict"],
                        evidence["evidence_timing"],
                    )
                )
    if actual != list(expected):
        raise HPKV3ValidationError(
            "reflected package changed, reordered, or duplicated authoritative "
            "action evidence"
        )


def reflect_real_rollout(
    reflector: VLMHierarchicalReflector,
    rollout_input: RealRolloutReflectionInput,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one no-retry reflection and save its package and atomic units."""

    result = reflector.reflect(
        instruction=rollout_input.instruction,
        ordered_action_chunks=rollout_input.ordered_action_chunks,
        key_observations=rollout_input.key_observations,
        available_state=rollout_input.available_state,
        images=rollout_input.images,
        bind_rgb_evidence=rollout_input.source_observations is not None,
    )
    validate_reflected_evidence_authority(
        result.package,
        expected=rollout_input.evidence_signature,
    )
    package = result.package.to_dict()
    atomic = atomize_package(result.package)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if rollout_input.source_observations is not None:
        atomic, evidence_index = bind_reflection_evidence(root=rollout_input.source_root, source_name=str(directory.relative_to(rollout_input.source_root)), source_observations=rollout_input.source_observations, package=result.package, atomic_knowledge=atomic, bindings=result.evidence_bindings)
        evidence_index.save(filename=str((directory / "evidence_index.jsonl").relative_to(rollout_input.source_root)))

    def write(name: str, value: Any) -> None:
        (directory / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    write(
        "reflector_input.json",
        {
            "instruction": rollout_input.instruction,
            "ordered_action_chunks": list(rollout_input.ordered_action_chunks),
            "key_observations": list(rollout_input.key_observations),
            "available_state": list(rollout_input.available_state),
        },
    )
    write("trajectory_knowledge_package.json", package)
    write("atomic_knowledge.json", atomic)
    if rollout_input.source_observations is not None:
        shutil.copy2(rollout_input.source_root / "events.jsonl", directory / "events.jsonl")
        write("source_observations.json", rollout_input.source_observations)
        write("reflection_response.json", result.to_dict())
    return {
        "status": "trajectory package generated",
        "subtask_count": len(package["task_strategy"]["subtasks"]),
        "task_knowledge_count": len(atomic["task_knowledge"]),
        "action_knowledge_count": len(atomic["action_knowledge"]),
        "action_execution_count": len(rollout_input.evidence_signature),
    }


__all__ = [
    "RealRolloutReflectionInput",
    "build_real_rollout_reflection_input",
    "reflect_real_rollout",
    "validate_reflected_evidence_authority",
]

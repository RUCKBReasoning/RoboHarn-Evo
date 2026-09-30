
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image

from roboharn_evo.agent.hpk.hierarchical_knowledge import atomize_package
from roboharn_evo.agent.hpk.hierarchical_retriever import (
    AgentApiHierarchicalRetrievalBackend,
    HierarchicalHPKRetrievalRuntime,
    VLMActionKnowledgeRetriever,
    VLMSubtaskKnowledgeRetriever,
)
from roboharn_evo.agent.hpk.hierarchical_store import (
    load_hierarchical_store,
    save_hierarchical_store,
)
from roboharn_evo.agent.hpk.semantic_consolidator import VLMKnowledgeConsolidator
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    ExistingGPTMultimodalReflectorBackend,
    VLMHierarchicalReflector,
)
from roboharn_evo.agent.reflector.multimodal_transport import (
    ImageEgressAuthorization,
    MultimodalImage,
    OpenAICompatibleMultimodalTransport,
)


class LiberoExpertKnowledgeError(ValueError):
    """The expert source or generated task-only Store is invalid."""


@dataclass(frozen=True, slots=True)
class LiberoExpertReflectionSource:
    dataset_root: Path
    episode_index: int
    instruction: str
    task_index: int
    frame_count: int
    boundary_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LiberoTaskKnowledgeBuildResult:
    store_root: Path
    instruction: str
    episode_index: int
    frame_count: int
    temporal_windows: int
    key_observations: int
    task_knowledge_count: int
    action_knowledge_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "real LIBERO expert trajectory",
            "instruction": self.instruction,
            "episode_index": self.episode_index,
            "frame_count": self.frame_count,
            "temporal_windows": self.temporal_windows,
            "key_observations": self.key_observations,
            "store_root": str(self.store_root),
            "task_knowledge_count": self.task_knowledge_count,
            "action_knowledge_count": self.action_knowledge_count,
            "action_knowledge_enabled": False,
        }


def _jsonl_records(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LiberoExpertKnowledgeError(f"cannot read {path}") from exc
    result: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LiberoExpertKnowledgeError(
                f"{path.name} line {ordinal} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise LiberoExpertKnowledgeError(
                f"{path.name} line {ordinal} must contain an object"
            )
        result.append(value)
    return tuple(result)


def uniform_temporal_boundaries(
    frame_count: int,
    *,
    window_steps: int,
    max_images: int,
) -> tuple[int, ...]:
    """Return inclusive observation boundaries without semantic assumptions."""

    for name, value in (
        ("frame_count", frame_count),
        ("window_steps", window_steps),
        ("max_images", max_images),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LiberoExpertKnowledgeError(f"{name} must be a positive integer")
    if frame_count < 2:
        raise LiberoExpertKnowledgeError("expert trajectory must contain two frames")
    boundaries = list(range(0, frame_count, window_steps))
    if boundaries[-1] != frame_count - 1:
        boundaries.append(frame_count - 1)
    if len(boundaries) > max_images:
        raise LiberoExpertKnowledgeError(
            "temporal frequency produces more observations than the model budget"
        )
    return tuple(boundaries)


def load_expert_reflection_source(
    dataset_root: str | Path,
    *,
    episode_index: int,
    window_steps: int,
    max_images: int,
) -> tuple[LiberoExpertReflectionSource, Any]:
    """Load one LeRobot-format episode and verify its task/video alignment."""

    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise LiberoExpertKnowledgeError("episode_index must be an integer")
    if episode_index < 0:
        raise LiberoExpertKnowledgeError("episode_index must be non-negative")
    root = Path(dataset_root).expanduser().resolve(strict=True)
    episodes = _jsonl_records(root / "meta" / "episodes.jsonl")
    episode = next(
        (item for item in episodes if item.get("episode_index") == episode_index),
        None,
    )
    if episode is None:
        raise LiberoExpertKnowledgeError("expert episode is absent from metadata")
    tasks = episode.get("tasks")
    if (
        isinstance(tasks, (str, bytes))
        or not isinstance(tasks, Sequence)
        or len(tasks) != 1
        or not isinstance(tasks[0], str)
        or not tasks[0].strip()
    ):
        raise LiberoExpertKnowledgeError("expert episode must bind exactly one task")
    frame_count = episode.get("length")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise LiberoExpertKnowledgeError("expert episode length is invalid")
    boundaries = uniform_temporal_boundaries(
        frame_count,
        window_steps=window_steps,
        max_images=max_images,
    )
    parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    try:
        import pandas as pd

        table = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise LiberoExpertKnowledgeError("cannot read expert trajectory table") from exc
    required = {
        "observation.states.ee_state",
        "observation.states.gripper_state",
        "action",
        "frame_index",
        "task_index",
    }
    if required - set(table.columns) or len(table) != frame_count:
        raise LiberoExpertKnowledgeError("expert trajectory table contract mismatch")
    frame_indices = tuple(int(value) for value in table["frame_index"].tolist())
    if frame_indices != tuple(range(frame_count)):
        raise LiberoExpertKnowledgeError("expert frame indices are not contiguous")
    task_indices = {int(value) for value in table["task_index"].tolist()}
    if len(task_indices) != 1:
        raise LiberoExpertKnowledgeError("expert episode mixes multiple tasks")
    source = LiberoExpertReflectionSource(
        dataset_root=root,
        episode_index=episode_index,
        instruction=" ".join(tasks[0].split()),
        task_index=next(iter(task_indices)),
        frame_count=frame_count,
        boundary_frames=boundaries,
    )
    return source, table


def _decode_selected_frames(
    path: Path,
    indices: Sequence[int],
) -> tuple[np.ndarray, ...]:
    selected = set(indices)
    frames: dict[int, np.ndarray] = {}
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for ordinal, frame in enumerate(container.decode(stream)):
                if ordinal in selected:
                    frames[ordinal] = frame.to_ndarray(format="rgb24")
    except Exception as exc:
        raise LiberoExpertKnowledgeError(f"cannot decode {path.name}") from exc
    missing = selected - set(frames)
    if missing:
        raise LiberoExpertKnowledgeError(
            f"video is missing requested frame indices: {sorted(missing)}"
        )
    return tuple(frames[index] for index in indices)


def _paired_observation(external: np.ndarray, wrist: np.ndarray) -> Image.Image:
    external_image = Image.fromarray(np.asarray(external, dtype=np.uint8), mode="RGB")
    wrist_image = Image.fromarray(np.asarray(wrist, dtype=np.uint8), mode="RGB")
    height = max(external_image.height, wrist_image.height)
    canvas = Image.new(
        "RGB",
        (external_image.width + wrist_image.width, height),
        color="white",
    )
    canvas.paste(external_image, (0, 0))
    canvas.paste(wrist_image, (external_image.width, 0))
    return canvas


def _relative_window_facts(table: Any, start: int, end: int) -> list[str]:
    start_ee = np.asarray(table.iloc[start]["observation.states.ee_state"], dtype=float)
    end_ee = np.asarray(table.iloc[end]["observation.states.ee_state"], dtype=float)
    start_gripper = np.asarray(
        table.iloc[start]["observation.states.gripper_state"], dtype=float
    )
    end_gripper = np.asarray(
        table.iloc[end]["observation.states.gripper_state"], dtype=float
    )
    actions = np.stack(
        [np.asarray(value, dtype=float) for value in table.iloc[start:end]["action"]]
    )
    translation = end_ee[:3] - start_ee[:3]
    orientation = end_ee[3:6] - start_ee[3:6]
    return [
        (
            "the end effector changes relative position by "
            f"{float(np.linalg.norm(translation)):.4f} metres"
        ),
        f"the end effector changes height by {float(translation[2]):+.4f} metres",
        (
            "the end effector orientation state changes by "
            f"{float(np.linalg.norm(orientation)):.4f} radians"
        ),
        (
            "the measured gripper state changes by "
            f"{float(np.linalg.norm(end_gripper - start_gripper)):.4f}"
        ),
        (
            "the mean native translation command magnitude in this window is "
            f"{float(np.linalg.norm(actions[:, :3], axis=1).mean()):.4f}"
        ),
    ]


def prepare_expert_reflection(
    source: LiberoExpertReflectionSource,
    table: Any,
    *,
    output_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[str], list[MultimodalImage]]:
    """Prepare ID-free ordered windows and paired real-camera observations."""

    root = Path(output_dir)
    image_root = root / "key_observations"
    image_root.mkdir(parents=True, exist_ok=False)
    video_root = source.dataset_root / "videos" / "chunk-000"
    external = _decode_selected_frames(
        video_root
        / "observation.images.image"
        / f"episode_{source.episode_index:06d}.mp4",
        source.boundary_frames,
    )
    wrist = _decode_selected_frames(
        video_root
        / "observation.images.wrist_image"
        / f"episode_{source.episode_index:06d}.mp4",
        source.boundary_frames,
    )
    descriptions: list[str] = []
    images: list[MultimodalImage] = []
    for ordinal, (external_frame, wrist_frame) in enumerate(
        zip(external, wrist, strict=True)
    ):
        paired = _paired_observation(external_frame, wrist_frame)
        path = image_root / f"observation_{ordinal:02d}.png"
        paired.save(path, format="PNG")
        buffer = BytesIO()
        paired.save(buffer, format="PNG")
        descriptions.append(
            "paired external-camera and wrist-camera views at the next ordered boundary"
        )
        images.append(
            MultimodalImage(
                evidence_id=f"ordered observation {ordinal + 1}",
                mime_type="image/png",
                content=buffer.getvalue(),
                detail="high",
            )
        )
    chunks: list[dict[str, Any]] = []
    for start, end in zip(
        source.boundary_frames[:-1],
        source.boundary_frames[1:],
        strict=True,
    ):
        chunks.append(
            {
                "annotation": "one contiguous expert action window",
                "observed_facts": _relative_window_facts(table, start, end),
                "visual_observations": [
                    "the paired views after this window are supplied in temporal order"
                ],
            }
        )
    return chunks, descriptions, images


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_task_only_store_from_expert(
    *,
    dataset_root: str | Path,
    episode_index: int,
    output_dir: str | Path,
    planner_url: str,
    timeout_sec: int = 600,
    window_steps: int = 10,
    max_images: int = 16,
) -> LiberoTaskKnowledgeBuildResult:
    """Reflect one real expert episode and publish a read-only task-only Store."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, table = load_expert_reflection_source(
        dataset_root,
        episode_index=episode_index,
        window_steps=window_steps,
        max_images=max_images,
    )
    chunks, descriptions, images = prepare_expert_reflection(
        source,
        table,
        output_dir=output,
    )
    parsed_url = urlsplit(str(planner_url).strip())
    service_url = urlunsplit((parsed_url.scheme, parsed_url.netloc, "", "", ""))
    authorization = ImageEgressAuthorization.operator_granted(
        assertion="LIBERO expert trajectory HPK v3 task reflection",
        scope="libero_expert_task_knowledge_key_observations",
    )
    transport = OpenAICompatibleMultimodalTransport(
        service_url=service_url,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        timeout_sec=timeout_sec,
        max_images=max_images,
    )
    preflight = transport.capability_preflight(authorization=authorization)
    reflector = VLMHierarchicalReflector(
        ExistingGPTMultimodalReflectorBackend(
            transport,
            authorization=authorization,
        )
    )
    reflection = reflector.reflect(
        instruction=source.instruction,
        ordered_action_chunks=chunks,
        key_observations=descriptions,
        available_state=(
            "the source is a real expert demonstration",
            "semantic action boundaries must be inferred from ordered evidence",
        ),
        images=images,
    )
    atomized = atomize_package(reflection.package)
    consolidation_backend = AgentApiHierarchicalRetrievalBackend(
        planner_url,
        timeout_sec=timeout_sec,
    )
    consolidated = VLMKnowledgeConsolidator(consolidation_backend).consolidate(
        task_knowledge=atomized["task_knowledge"],
        action_knowledge=atomized["action_knowledge"],
    )
    store_root = output / "store"
    save_hierarchical_store(
        store_root,
        task_knowledge=consolidated.task_knowledge,
        action_knowledge=(),
    )
    _write_json(
        output / "trajectory_knowledge_package.json", reflection.package.to_dict()
    )
    _write_json(output / "atomic_knowledge.json", atomized)
    _write_json(
        output / "semantic_consolidation_response.json",
        consolidated.raw_response,
    )
    _write_json(
        output / "source_provenance.json",
        {
            "source": "real LIBERO expert trajectory",
            "dataset_root": str(source.dataset_root),
            "episode_index": source.episode_index,
            "task_index": source.task_index,
            "instruction": source.instruction,
            "frame_count": source.frame_count,
            "temporal_window_steps": window_steps,
            "boundary_frames": list(source.boundary_frames),
            "semantic_stage_assignment": "GPT-5.5 xhigh v3 hierarchical reflector",
            "task_consolidation": "GPT-5.5 xhigh semantic consolidator",
            "action_knowledge_written_to_store": False,
            "simulator_executed": False,
            "automatic_retries": 0,
            "capability_preflight": preflight.to_dict(),
        },
    )
    result = LiberoTaskKnowledgeBuildResult(
        store_root=store_root,
        instruction=source.instruction,
        episode_index=source.episode_index,
        frame_count=source.frame_count,
        temporal_windows=len(chunks),
        key_observations=len(images),
        task_knowledge_count=len(consolidated.task_knowledge),
        action_knowledge_count=0,
    )
    _write_json(output / "build_result.json", result.to_dict())
    return result


def build_task_only_runtime(
    *,
    store_root: str | Path,
    planner_url: str,
    timeout_sec: int = 600,
) -> HierarchicalHPKRetrievalRuntime:
    """Load Task Knowledge read-only; reject any Action Knowledge member."""

    tasks, actions = load_hierarchical_store(store_root)
    if not tasks:
        raise LiberoExpertKnowledgeError("task-only Store contains no Task Knowledge")
    if actions:
        raise LiberoExpertKnowledgeError(
            "LIBERO task-only smoke rejects non-empty Action Knowledge"
        )
    backend = AgentApiHierarchicalRetrievalBackend(
        planner_url,
        timeout_sec=timeout_sec,
    )
    return HierarchicalHPKRetrievalRuntime(
        mode="full",
        task_knowledge=tasks,
        action_knowledge=(),
        subtask_retriever=VLMSubtaskKnowledgeRetriever(backend),
        action_retriever=VLMActionKnowledgeRetriever(backend),
    )


__all__ = [
    "LiberoExpertKnowledgeError",
    "LiberoExpertReflectionSource",
    "LiberoTaskKnowledgeBuildResult",
    "build_task_only_runtime",
    "build_task_only_store_from_expert",
    "load_expert_reflection_source",
    "prepare_expert_reflection",
    "uniform_temporal_boundaries",
]

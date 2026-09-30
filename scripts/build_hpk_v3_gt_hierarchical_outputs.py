from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import request

import h5py
import numpy as np
from PIL import Image

from roboharn_evo.agent.hpk.hierarchical_knowledge import HPKV3ValidationError, atomize_package
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    ExistingGPTMultimodalReflectorBackend,
    VLMHierarchicalReflector,
)
from roboharn_evo.agent.reflector.multimodal_transport import (
    ImageEgressAuthorization,
    MultimodalImage,
    OpenAICompatibleMultimodalTransport,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "eval_result/hpk/v3/rmbench_gt_hierarchical_generic"
)


def _post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = request.build_opener(request.ProxyHandler({}))
    with opener.open(http_request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("service response must be an object")
    return value


def _decode_hdf5_rgb(value: Any) -> Image.Image:
    """Undo the historical RGB-as-BGR OpenCV encoding in RMBench HDF5."""

    encoded = Image.open(BytesIO(bytes(value))).convert("RGB")
    swapped = np.asarray(encoded, dtype=np.uint8)[..., ::-1]
    return Image.fromarray(np.ascontiguousarray(swapped), mode="RGB")


def _arm_sensor_facts(handle: h5py.File, start: int, end: int) -> list[str]:
    """Expose measured changes without assigning an action or semantic phase."""

    facts: list[str] = []
    for arm in ("left", "right"):
        poses = np.asarray(handle[f"endpose/{arm}_endpose"])
        delta = poses[end, :3] - poses[start, :3]
        horizontal = float(np.linalg.norm(delta[:2]))
        vertical = float(delta[2])
        gripper = np.asarray(handle[f"endpose/{arm}_gripper"])
        facts.extend(
            [
                (
                    f"the {arm} arm end effector changes horizontal position by "
                    f"{horizontal:.4f} metres and height by {vertical:+.4f} metres"
                ),
                (
                    f"the {arm} gripper signal changes from "
                    f"{float(gripper[start]):.4f} to {float(gripper[end]):.4f}"
                ),
            ]
        )
    return facts


def _sam_description(response: dict[str, Any], *, concept: str) -> str:
    detections = response.get("detections")
    if not isinstance(detections, list) or not detections:
        return f"SAM3 did not find a clear region matching the {concept}"
    if len(detections) == 1:
        return f"SAM3 found one visible region matching the {concept}"
    return f"SAM3 found multiple visible regions matching the {concept}"


def _load_chunks(annotation_path: Path, episode: int) -> list[tuple[str, int, int]]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    values = annotation[f"episode_{episode}"]
    chunks: list[tuple[str, int, int]] = []
    cursor = 0
    for subtask, length in values:
        chunks.append((str(subtask), cursor, cursor + int(length) - 1))
        cursor += int(length)
    return chunks


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_episode(
    *,
    episode: int,
    instruction: str,
    available_state: Sequence[str],
    data_root: Path,
    output_root: Path,
    sam_url: str | None,
    sam_concept: str,
    sam_input_root: Path,
    sam_mask_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[MultimodalImage]]:
    chunks = _load_chunks(data_root / "language_annotation.json", episode)
    if not chunks:
        raise ValueError(f"episode {episode} has no ordered action chunks")
    episode_root = output_root / f"episode_{episode}"
    image_root = episode_root / "key_observations"
    image_root.mkdir(parents=True, exist_ok=True)
    if sam_url is not None:
        sam_input_root.mkdir(parents=True, exist_ok=True)
        sam_mask_root.mkdir(parents=True, exist_ok=True)
    chunk_inputs: list[dict[str, Any]] = []
    descriptions: list[str] = []
    images: list[MultimodalImage] = []
    hdf5_path = data_root / "data" / f"episode{episode}.hdf5"
    with h5py.File(hdf5_path, "r") as handle:
        boundary_frames = [0, *(end for _, _, end in chunks)]
        boundary_descriptions = ["the initial observation"] + [
            "the observation after the corresponding ordered action chunk"
            for _ in chunks
        ]
        sam_descriptions: list[str] = []
        for ordinal, (frame, description) in enumerate(
            zip(boundary_frames, boundary_descriptions, strict=True)
        ):
            image = _decode_hdf5_rgb(handle["observation/head_camera/rgb"][frame])
            filename = f"observation_{ordinal:02d}.png"
            output_path = image_root / filename
            service_path = sam_input_root / f"hpk_v3_gt_e{episode}_{filename}"
            image.save(output_path)
            if sam_url is None:
                sam_descriptions.append(
                    f"SAM3 observation for {sam_concept} is pending"
                )
            else:
                image.save(service_path)
                sam = _post_json(
                    f"{sam_url.rstrip('/')}/segment_image",
                    {
                        "image_path": str(service_path),
                        "text_prompt": sam_concept,
                        "object_id": "HPK v3 task relevant region",
                        "top_k": 3,
                        "confidence_threshold": 0.1,
                        "output_dir": str(sam_mask_root),
                    },
                    timeout=120,
                )
                if sam.get("success") is not True:
                    raise RuntimeError(
                        f"SAM3 failed for episode {episode}: {sam.get('error')}"
                    )
                sam_descriptions.append(_sam_description(sam, concept=sam_concept))
            descriptions.append(description)
            images.append(
                MultimodalImage(
                    evidence_id=f"episode {episode} observation {ordinal}",
                    mime_type="image/png",
                    content=output_path.read_bytes(),
                    detail="high",
                )
            )

        for index, (annotation, start, end) in enumerate(chunks):
            chunk_inputs.append(
                {
                    "annotation": annotation.rstrip("."),
                    "observed_facts": _arm_sensor_facts(handle, start, end),
                    "visual_observations": [sam_descriptions[index + 1]],
                }
            )
    _write_json(
        episode_root / "reflector_input.json",
        {
            "input_format": "raw ordered action chunk observations",
            "instruction": instruction,
            "ordered_action_chunks": chunk_inputs,
            "key_observations": descriptions,
            "available_state": list(available_state),
        },
    )
    return chunk_inputs, descriptions, images


def _load_prepared_episode(
    output_root: Path,
    episode: int,
    *,
    expected_instruction: str | None = None,
    expected_available_state: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[MultimodalImage]]:
    episode_root = output_root / f"episode_{episode}"
    payload = json.loads(
        (episode_root / "reflector_input.json").read_text(encoding="utf-8")
    )
    if payload.get("input_format") != "raw ordered action chunk observations":
        raise ValueError("prepared input predates the generic observation-only builder")
    if (
        expected_instruction is not None
        and payload.get("instruction") != expected_instruction
    ):
        raise ValueError("prepared input instruction does not match this run")
    if expected_available_state is not None and payload.get("available_state") != list(
        expected_available_state
    ):
        raise ValueError("prepared available state does not match this run")
    chunks = payload["ordered_action_chunks"]
    descriptions = payload["key_observations"]
    image_paths = sorted((episode_root / "key_observations").glob("*.png"))
    if len(image_paths) != len(descriptions):
        raise ValueError("prepared key observations and images do not align")
    images = [
        MultimodalImage(
            evidence_id=f"episode {episode} observation {ordinal}",
            mime_type="image/png",
            content=path.read_bytes(),
            detail="high",
        )
        for ordinal, path in enumerate(image_paths)
    ]
    return chunks, descriptions, images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--available-state",
        action="append",
        default=[],
        help="repeat for each dataset-provided structured fact available to the VLM",
    )
    parser.add_argument(
        "--sam-concept",
        required=True,
        help="dataset-provided task-relevant visual concept; never inferred by chunk index",
    )
    parser.add_argument("--gpt-url", default="http://127.0.0.1:9104")
    parser.add_argument("--sam-url", default="http://127.0.0.1:9314")
    parser.add_argument(
        "--sam-artifact-root",
        type=Path,
        default=None,
        help="defaults to OUTPUT_ROOT/sam3_artifacts",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare corrected observations and semantic inputs without service calls",
    )
    parser.add_argument(
        "--reuse-prepared",
        type=int,
        nargs="*",
        default=[],
        help="reuse existing semantic inputs and images for these episodes without SAM3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.episodes:
        raise ValueError("at least one real trajectory is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sam_artifact_root = args.sam_artifact_root or args.output_root / "sam3_artifacts"
    if args.prepare_only:
        prepared: list[dict[str, Any]] = []
        for episode in args.episodes:
            chunks, descriptions, images = _prepare_episode(
                episode=episode,
                instruction=args.instruction,
                available_state=args.available_state,
                data_root=args.data_root,
                output_root=args.output_root,
                sam_url=None,
                sam_concept=args.sam_concept,
                sam_input_root=sam_artifact_root / "inputs",
                sam_mask_root=sam_artifact_root / "masks",
            )
            assert len(descriptions) == len(images)
            prepared.append(
                {
                    "episode": episode,
                    "action_chunks": len(chunks),
                    "key_observations": len(images),
                    "service_calls": 0,
                }
            )
        _write_json(
            args.output_root / "preparation_summary.json",
            {
                "status": "prepared without GPT or SAM3 calls",
                "rgb_channel_correction": "applied to RMBench HDF5 images",
                "episodes": prepared,
            },
        )
        return
    authorization = ImageEgressAuthorization.operator_granted(
        assertion="HPK v3 offline GT hierarchical reflection",
        scope="hpk_v3_offline_gt_key_observations",
    )
    transport = OpenAICompatibleMultimodalTransport(
        service_url=args.gpt_url,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        timeout_sec=600,
        max_images=16,
    )
    transport.capability_preflight(authorization=authorization)
    reflector = VLMHierarchicalReflector(
        ExistingGPTMultimodalReflectorBackend(
            transport,
            authorization=authorization,
        )
    )
    completed: list[dict[str, Any]] = []
    reused = set(args.reuse_prepared)
    if not reused.issubset(set(args.episodes)):
        raise ValueError("--reuse-prepared must be a subset of --episodes")
    for episode in args.episodes:
        if episode in reused:
            chunks, descriptions, images = _load_prepared_episode(
                args.output_root,
                episode,
                expected_instruction=args.instruction,
                expected_available_state=args.available_state,
            )
        else:
            chunks, descriptions, images = _prepare_episode(
                episode=episode,
                instruction=args.instruction,
                available_state=args.available_state,
                data_root=args.data_root,
                output_root=args.output_root,
                sam_url=args.sam_url,
                sam_concept=args.sam_concept,
                sam_input_root=sam_artifact_root / "inputs",
                sam_mask_root=sam_artifact_root / "masks",
            )
        episode_root = args.output_root / f"episode_{episode}"
        try:
            result = reflector.reflect(
                instruction=args.instruction,
                ordered_action_chunks=chunks,
                key_observations=descriptions,
                available_state=args.available_state,
                images=images,
            )
        except HPKV3ValidationError as exc:
            raw = reflector.last_raw_response
            if isinstance(raw, dict):
                _write_json(episode_root / "rejected_raw_reflector_output.json", raw)
            elif isinstance(raw, str):
                (episode_root / "rejected_raw_reflector_output.json").write_text(
                    raw.rstrip() + "\n", encoding="utf-8"
                )
            _write_json(
                episode_root / "reflection_error.json",
                {
                    "status": "rejected by local HPK v3 validation",
                    "reason": str(exc),
                    "automatic_retry": False,
                },
            )
            completed.append(
                {
                    "episode": episode,
                    "action_chunks": len(chunks),
                    "key_observations": len(images),
                    "status": "rejected",
                    "task_knowledge_units": 0,
                    "action_knowledge_units": 0,
                }
            )
            continue
        package = result.package.to_dict()
        atomized = atomize_package(result.package)
        _write_json(episode_root / "trajectory_knowledge_package.json", package)
        _write_json(episode_root / "atomic_knowledge.json", atomized)
        completed.append(
            {
                "episode": episode,
                "action_chunks": len(chunks),
                "key_observations": len(images),
                "status": "accepted",
                "subtasks": len(package["task_strategy"]["subtasks"]),
                "task_knowledge_units": len(atomized["task_knowledge"]),
                "action_knowledge_units": len(atomized["action_knowledge"]),
            }
        )
    _write_json(
        args.output_root / "run_summary.json",
        {
            "source": "RMBench ground-truth trajectories",
            "instruction": args.instruction,
            "model": "GPT-5.5 xhigh through the existing Agent API backend",
            "segmentation": "SAM3",
            "external_model_calls": len(completed),
            "sam3_calls": sum(
                item["key_observations"]
                for item in completed
                if item["episode"] not in reused
            ),
            "automatic_retries": 0,
            "episodes": completed,
        },
    )


if __name__ == "__main__":
    main()

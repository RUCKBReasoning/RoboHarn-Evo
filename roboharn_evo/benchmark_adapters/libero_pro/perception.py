"""Online SAM3 perception for the LIBERO benchmark-neutral Agent loop."""

from __future__ import annotations

import base64
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np
from PIL import Image

from roboharn_evo.agent.hpk.schemas import reject_private_transferable
from roboharn_evo.benchmark_adapters.base import NeutralObservation


class LiberoPerceptionError(ValueError):
    """A perception query, image, or SAM3 response is invalid."""


def _natural_text(value: Any, *, label: str, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        raise LiberoPerceptionError(f"{label} must be non-empty")
    if len(text) > max_chars:
        raise LiberoPerceptionError(f"{label} is too long")
    if "/" in text or "\\" in text:
        raise LiberoPerceptionError(f"{label} must not contain a path")
    return text


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = request.build_opener(request.ProxyHandler({}))
    with opener.open(http_request, timeout=timeout_sec) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise LiberoPerceptionError("perception service response must be an object")
    return value


PostJson = Callable[[str, dict[str, Any], int], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PerceptionQuery:
    """One model-authored visual query; no persistent identity is carried."""

    text_prompt: str
    role: str
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text_prompt",
            _natural_text(self.text_prompt, label="text_prompt", max_chars=160),
        )
        role = str(self.role).strip().casefold()
        if role not in {"target", "tool", "context"}:
            raise LiberoPerceptionError("query role must be target, tool, or context")
        object.__setattr__(self, "role", role)
        reason = " ".join(str(self.reason).strip().split())[:240]
        reject_private_transferable(
            {
                "text_prompt": self.text_prompt,
                "role": role,
                "reason": reason,
            },
            path="perception_query",
        )
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, str]:
        return {
            "text_prompt": self.text_prompt,
            "role": self.role,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PerceptionDetection:
    query: PerceptionQuery
    camera: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    centroid_px: tuple[float, float]
    area_px: int
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.camera not in {"external", "wrist"}:
            raise LiberoPerceptionError("camera must be external or wrist")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise LiberoPerceptionError("detection score must be within [0, 1]")
        if len(self.bbox_xyxy) != 4 or not all(
            math.isfinite(float(value)) for value in self.bbox_xyxy
        ):
            raise LiberoPerceptionError("bbox must contain four finite values")
        if len(self.centroid_px) != 2 or not all(
            math.isfinite(float(value)) for value in self.centroid_px
        ):
            raise LiberoPerceptionError("centroid must contain two finite values")
        if self.area_px <= 0:
            raise LiberoPerceptionError("detection area must be positive")
        if (
            len(self.image_shape) != 2
            or self.image_shape[0] <= 0
            or self.image_shape[1] <= 0
        ):
            raise LiberoPerceptionError("image shape must be positive")

    def to_public_dict(self) -> dict[str, Any]:
        height, width = self.image_shape
        return {
            "query": self.query.to_dict(),
            "camera": self.camera,
            "score": self.score,
            "detected": True,
            "area_fraction": round(self.area_px / float(height * width), 6),
        }


@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    step: int
    phase: str
    queries: tuple[PerceptionQuery, ...]
    detections: tuple[PerceptionDetection, ...]
    failures: tuple[str, ...]
    sam3_calls: int
    elapsed_ms: int

    def detections_for(
        self,
        *,
        role: str,
        camera: str | None = None,
    ) -> tuple[PerceptionDetection, ...]:
        return tuple(
            value
            for value in self.detections
            if value.query.role == role and (camera is None or value.camera == camera)
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "phase": self.phase,
            "queries": [value.to_dict() for value in self.queries],
            "detections": [value.to_public_dict() for value in self.detections],
            "failures": list(self.failures),
            "sam3_calls": self.sam3_calls,
            "elapsed_ms": self.elapsed_ms,
        }


_PERCEPTION_QUERY_PROMPT = """Choose at most three visual segmentation queries for the current robot subtask.
Return {"queries":[{"object_id":"short semantic label","text_prompt":"short English visual prompt","role":"target | tool | context","reason":"short reason"}]}.
Use role=target for the object whose state determines success, role=tool for a manipulated tool, and role=context for a relevant support, destination, or obstacle. Each text_prompt must be an appearance-only singular noun phrase for exactly one segmented object or part. Put spatial relations and disambiguation in reason or separate context queries; do not put words such as left, right, between, on, under, near, or destination in text_prompt. object_id is an ephemeral request label only; do not use a simulator, track, candidate, or dataset identity. Do not output coordinates, poses, action vectors, file paths, or robot instructions. Return JSON only."""


class LiberoSAM3PerceptionClient:
    """Generate task-grounded queries, then run local SAM3 on real cameras."""

    def __init__(
        self,
        *,
        planner_service_url: str,
        sam3_service_url: str,
        artifact_root: str | Path,
        timeout_sec: int = 120,
        max_queries: int = 3,
        confidence_threshold: float = 0.1,
        post_json: PostJson | None = None,
    ) -> None:
        planner_base = str(planner_service_url).rstrip("/")
        planner_base = planner_base.removesuffix("/plan")
        self.planner_service_url = planner_base
        self.sam3_service_url = str(sam3_service_url).rstrip("/")
        if not self.planner_service_url or not self.sam3_service_url:
            raise ValueError("planner and SAM3 service URLs must be non-empty")
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if not 1 <= max_queries <= 3:
            raise ValueError("max_queries must be within [1, 3]")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        self.timeout_sec = int(timeout_sec)
        self.max_queries = int(max_queries)
        self.confidence_threshold = float(confidence_threshold)
        self._post_json = post_json or _post_json
        self.query_calls = 0
        self.sam3_calls = 0

    def reset(self) -> None:
        self.query_calls = 0
        self.sam3_calls = 0

    @staticmethod
    def _upright_rgb(value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise LiberoPerceptionError("perception camera must be HWC uint8 RGB")
        return np.ascontiguousarray(np.rot90(image, 2))

    @classmethod
    def _encoded_image(cls, value: Any) -> str:
        buffer = BytesIO()
        Image.fromarray(cls._upright_rgb(value), mode="RGB").save(
            buffer,
            format="PNG",
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def generate_queries(
        self,
        *,
        task: str,
        subtask: str,
        memory: str,
        observation: NeutralObservation,
    ) -> tuple[PerceptionQuery, ...]:
        payload = {
            "prompt": _PERCEPTION_QUERY_PROMPT,
            "global_task": _natural_text(task, label="task", max_chars=800),
            "current_subtask": _natural_text(
                subtask,
                label="subtask",
                max_chars=800,
            ),
            "committed_memory": " ".join(str(memory).strip().split())[:800],
            "observation_summary": "two current real LIBERO RGB camera views",
            "robot_state": {"available": "public gripper and end-effector state"},
            "active_skill": "monitored-subtask-execution",
            "max_queries": self.max_queries,
            "cameras": ["external", "wrist"],
            "image_b64_by_camera": {
                "external": self._encoded_image(observation.cameras["agentview_image"]),
                "wrist": self._encoded_image(
                    observation.cameras["robot0_eye_in_hand_image"]
                ),
            },
            "oracle_objects": [],
            "scene_instances": [],
            "require_instance_binding": False,
        }
        response = self._post_json(
            f"{self.planner_service_url}/perception_queries",
            payload,
            self.timeout_sec,
        )
        self.query_calls += 1
        raw_queries = response.get("queries")
        if not isinstance(raw_queries, list):
            raise LiberoPerceptionError("query service omitted queries array")
        result: list[PerceptionQuery] = []
        seen: set[tuple[str, str]] = set()
        for raw in raw_queries:
            if not isinstance(raw, Mapping):
                continue
            try:
                query = PerceptionQuery(
                    text_prompt=raw.get("text_prompt", ""),
                    role=raw.get("role", "context"),
                    reason=raw.get("reason", ""),
                )
            except LiberoPerceptionError:
                continue
            key = (query.text_prompt.casefold(), query.role)
            if key in seen:
                continue
            seen.add(key)
            result.append(query)
            if len(result) >= self.max_queries:
                break
        if not any(value.role == "target" for value in result):
            return ()
        return tuple(result)

    def observe(
        self,
        *,
        observation: NeutralObservation,
        queries: Sequence[PerceptionQuery],
        step: int,
        phase: str,
    ) -> PerceptionFrame:
        started = time.monotonic()
        normalized_phase = str(phase).strip().casefold()
        if normalized_phase not in {"before_action", "after_action"}:
            raise ValueError("phase must be before_action or after_action")
        if step < 0:
            raise ValueError("step must be non-negative")
        query_tuple = tuple(queries)
        input_root = self.artifact_root / "inputs"
        mask_root = self.artifact_root / "masks"
        input_root.mkdir(parents=True, exist_ok=True)
        mask_root.mkdir(parents=True, exist_ok=True)
        cameras = {
            "external": observation.cameras["agentview_image"],
            "wrist": observation.cameras["robot0_eye_in_hand_image"],
        }
        image_paths: dict[str, Path] = {}
        for camera, value in cameras.items():
            path = input_root / f"step_{step:04d}_{normalized_phase}_{camera}.png"
            Image.fromarray(self._upright_rgb(value), mode="RGB").save(path)
            image_paths[camera] = path
        detections: list[PerceptionDetection] = []
        failures: list[str] = []
        calls = 0
        for query_index, query in enumerate(query_tuple):
            query_cameras = (
                ("external", "wrist") if query.role == "target" else ("external",)
            )
            for camera in query_cameras:
                calls += 1
                try:
                    response = self._post_json(
                        f"{self.sam3_service_url}/segment_image",
                        {
                            "image_path": str(image_paths[camera]),
                            "text_prompt": query.text_prompt,
                            "object_id": f"{query.role}_{query_index}",
                            "top_k": 1,
                            "confidence_threshold": self.confidence_threshold,
                            "output_dir": str(
                                mask_root
                                / f"step_{step:04d}_{normalized_phase}_{camera}"
                            ),
                        },
                        self.timeout_sec,
                    )
                    self.sam3_calls += 1
                    if response.get("success") is not True:
                        raise LiberoPerceptionError("SAM3 reported failure")
                    raw = response.get("detections")
                    if not isinstance(raw, list) or not raw:
                        continue
                    best = raw[0]
                    if not isinstance(best, Mapping):
                        continue
                    bbox = best.get("bbox_xyxy", best.get("box_xyxy", []))
                    centroid = best.get("centroid_px", [])
                    shape = best.get("mask_shape", [])
                    if not (
                        isinstance(bbox, list)
                        and isinstance(centroid, list)
                        and isinstance(shape, list)
                    ):
                        continue
                    detections.append(
                        PerceptionDetection(
                            query=query,
                            camera=camera,
                            score=float(best.get("score", 0.0)),
                            bbox_xyxy=tuple(float(value) for value in bbox[:4]),
                            centroid_px=tuple(float(value) for value in centroid[:2]),
                            area_px=int(best.get("area_px", 0)),
                            image_shape=tuple(int(value) for value in shape[:2]),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - unknown is fail-safe
                    failures.append(f"{query.role}:{camera}:{type(exc).__name__}")
        return PerceptionFrame(
            step=step,
            phase=normalized_phase,
            queries=query_tuple,
            detections=tuple(detections),
            failures=tuple(failures),
            sam3_calls=calls,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )


__all__ = [
    "LiberoPerceptionError",
    "LiberoSAM3PerceptionClient",
    "PerceptionDetection",
    "PerceptionFrame",
    "PerceptionQuery",
]

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def resolve_gpt_scene_memory_sidecar_path(
    *,
    trace_file: Path | None,
    rollout_dir: Path | None,
) -> Path | None:
    """Return the audit path without creating or mutating runtime state."""

    if rollout_dir is not None:
        return rollout_dir / "gpt_scene_memory.jsonl"
    if trace_file is None:
        return None
    stem = trace_file.stem
    if stem.endswith("_agent_trace"):
        stem = stem[: -len("_agent_trace")] + "_gpt_scene_memory"
    else:
        stem = stem + "_gpt_scene_memory"
    return trace_file.with_name(stem + ".jsonl")


def append_gpt_scene_memory_request(
    *,
    path: Path,
    request_index: int,
    consumer: str,
    scene_memory: dict[str, Any],
    episode_id: int,
    seed: int,
    env_step: Any,
    planner_context_mode: str,
    observation_generation: Any = None,
    observation_capture_id: Any = None,
    request_metadata: dict[str, Any] | None = None,
    scene_memory_views: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Append one model-request audit record, failing open on every error.

    This function only serializes a copy of data that has already been placed
    in a model request.  It never returns data to the planner, memory tracker,
    guards, or executor.
    """

    request_id = f"gpt_scene_memory_request_{int(request_index):06d}"
    try:
        normalized_scene = json.loads(
            json.dumps(scene_memory, ensure_ascii=False)
        )
        normalized_views = {
            str(name): json.loads(json.dumps(view, ensure_ascii=False))
            for name, view in (scene_memory_views or {}).items()
            if isinstance(view, dict)
        }
        normalized_metadata = json.loads(
            json.dumps(request_metadata or {}, ensure_ascii=False)
        )
        hash_payload: dict[str, Any] = {
            "scene_memory": normalized_scene,
        }
        if normalized_views:
            hash_payload["scene_memory_views"] = normalized_views
        canonical_scene = json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        record: dict[str, Any] = {
            "schema": "roboharn_evo/gpt_scene_memory_request/v1",
            "schema_version": 1,
            "event": "gpt_scene_memory_request",
            "timestamp": time.time(),
            "episode_id": int(episode_id),
            "seed": int(seed),
            "env_step": env_step,
            "request_id": request_id,
            "request_index": int(request_index),
            "request_status": "submitted",
            "consumer": str(consumer or "unknown"),
            "planner_context_mode": str(planner_context_mode or ""),
            "observation_generation": observation_generation,
            "observation_capture_id": observation_capture_id,
            "scene_memory_schema": normalized_scene.get("schema", ""),
            "scene_memory_sha256": hashlib.sha256(
                canonical_scene.encode("utf-8")
            ).hexdigest(),
            "request_metadata": normalized_metadata,
            "scene_memory": normalized_scene,
        }
        if normalized_views:
            record["scene_memory_views"] = normalized_views

        path.parent.mkdir(parents=True, exist_ok=True)
        needs_separator = False
        if path.is_file() and path.stat().st_size > 0:
            with path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                needs_separator = existing.read(1) != b"\n"
        with path.open("a", encoding="utf-8") as stream:
            if needs_separator:
                stream.write("\n")
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return ""
    return request_id

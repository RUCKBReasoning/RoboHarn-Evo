from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import mimetypes
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib import error, parse, request


ALLOWED_COMMIT_LABELS = {"no_update", "subtask_complete", "state_change"}
ALLOWED_PREFERRED_ARMS = {"left", "right", "either"}
ALLOWED_SELECTED_ARMS = {"left", "right", "none"}
ALLOWED_OOD_SCENARIOS = {
    "none",
    "object_not_visible",
    "motion_blocked",
    "grasp_lost",
    "scene_drift_detected",
    "requires_replan",
}
ALLOWED_RECOVERY_INTENTS = {"retry", "replan", "abort"}
ALLOWED_EFFECT_VALUES = {"true", "false", "unverified"}
ALLOWED_SUBTASK_STATUS_VALUES = {"in_progress", "completed", "failed", "uncertain"}
ALLOWED_VERIFIER_CONTROL_VALUES = {"continue", "retry", "replan"}
ALLOWED_QUERY_ROLES = {"target", "tool", "context"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
TASK_RELATED_OOD_TERMS = {
    "block",
    "object",
    "scene",
    "visual",
    "image",
    "view",
    "motion",
    "grasp",
    "pick",
}
ACTION_EFFECT_SKILL_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "recovery"
    / "effect-verification"
    / "action-effect-verification"
    / "SKILL.md"
)
CONTROL_TURN_PLANNER_SKILL_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "planning"
    / "control-turn-planner"
    / "SKILL.md"
)
PLANNER_PROMPT_MODE = "rendered_system_once"
MAX_PLANNER_TASK_MIN_BYTES = 2_000_000
_NO_PROXY_OPENER = request.build_opener(request.ProxyHandler({}))


class SchemaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ImageInput:
    camera: str
    path: Path
    mime_type: str
    byte_count: int
    sha256: str
    data_b64: str

    @classmethod
    def load(cls, *, camera: str, path: Path) -> "ImageInput":
        resolved = path.expanduser().resolve(strict=True)
        data = resolved.read_bytes()
        if not data:
            raise ValueError(f"{camera} image is empty: {resolved}")
        mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return cls(
            camera=camera,
            path=resolved,
            mime_type=mime_type,
            byte_count=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            data_b64=base64.b64encode(data).decode("ascii"),
        )

    def report_record(self, order: int) -> dict[str, Any]:
        return {
            "order": order,
            "camera": self.camera,
            "path": str(self.path),
            "mime_type": self.mime_type,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    body: bytes
    latency_ms: float
    transport_error: Exception | None = None


@dataclass(frozen=True)
class ProbeSpec:
    endpoint: str
    mode: str
    payload: dict[str, Any]
    validator: Callable[[dict[str, Any]], None]
    request_metadata: dict[str, Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise every RoboHarn-Evo agent gateway role with real local image files."
    )
    parser.add_argument(
        "--gateway-base-url",
        default="http://127.0.0.1:9105",
        help="Agent gateway base URL; endpoint paths are appended by this script.",
    )
    parser.add_argument("--head-image", type=Path, required=True)
    parser.add_argument("--third-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--planner-task-min-bytes",
        "--planner-payload-min-bytes",
        dest="planner_task_min_bytes",
        type=int,
        default=0,
        help=(
            "Minimum UTF-8 byte count for the dynamic /plan task field, excluding images "
            "and the rendered prompt copy. A positive value builds a deterministic "
            "trace-like task and a "
            "canonically rendered control-turn-planner prompt; zero preserves the short "
            "legacy probe. Use 53342 to reproduce the failed rollout control-payload "
            "scale. --planner-payload-min-bytes is retained as an alias."
        ),
    )
    parser.add_argument(
        "--expected-planner-prompt-mode",
        default="",
        help=(
            "When nonempty, require /health planner_prompt_mode to equal this value. "
            f"The local Qwen deduplication contract uses {PLANNER_PROMPT_MODE!r}."
        ),
    )
    parser.add_argument(
        "--expected-planner-max-output-tokens",
        type=int,
        default=None,
        help=(
            "When set, require /health planner_max_output_tokens to equal this value."
        ),
    )
    parser.add_argument(
        "--skip-malformed-json-probe",
        action="store_true",
        help="Skip the final invalid-JSON rejection probe.",
    )
    return parser.parse_args(argv)


def _require_object(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{context} response must be a JSON object")
    return payload


def _require_string(payload: dict[str, Any], key: str, *, nonempty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SchemaValidationError(f"{key} must be a string")
    if nonempty and not value.strip():
        raise SchemaValidationError(f"{key} must not be empty")
    return value


def _require_enum(payload: dict[str, Any], key: str, allowed: set[str]) -> str:
    value = _require_string(payload, key)
    if value not in allowed:
        raise SchemaValidationError(f"{key}={value!r} is not one of {sorted(allowed)}")
    return value


def _require_confidence(payload: dict[str, Any]) -> None:
    value = payload.get("confidence")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaValidationError("confidence must be numeric")
    if not 0.0 <= float(value) <= 1.0:
        raise SchemaValidationError("confidence must be in [0, 1]")


def validate_plan_response(payload: dict[str, Any]) -> None:
    _require_object(payload, context="plan")
    _require_enum(payload, "commit_label", ALLOWED_COMMIT_LABELS)
    _require_string(payload, "memory_text")
    _require_string(payload, "selected_skill", nonempty=True)
    _require_string(payload, "subtask_text", nonempty=True)
    _require_enum(payload, "preferred_arm", ALLOWED_PREFERRED_ARMS)


def validate_ood_response(payload: dict[str, Any]) -> None:
    _require_object(payload, context="ood")
    _require_enum(payload, "OOD_scenario", ALLOWED_OOD_SCENARIOS)
    reason = _require_string(payload, "reason", nonempty=True)
    normalized_reason = reason.casefold()
    if not any(term in normalized_reason for term in TASK_RELATED_OOD_TERMS):
        raise SchemaValidationError(
            "reason must cite task-related scene, object, visual, or manipulation evidence"
        )
    _require_confidence(payload)


def validate_recovery_response(payload: dict[str, Any]) -> None:
    _require_object(payload, context="recover")
    _require_string(payload, "recovery_workflow", nonempty=True)
    _require_enum(payload, "selected_arm", ALLOWED_SELECTED_ARMS)
    tool_calls = payload.get("tool_calls")
    if not isinstance(tool_calls, list):
        raise SchemaValidationError("tool_calls must be a list")
    if not tool_calls:
        raise SchemaValidationError("tool_calls must contain at least one recovery action")
    tool_names: list[str] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            raise SchemaValidationError(f"tool_calls[{index}] must be an object")
        tool_names.append(_require_string(tool_call, "tool_name", nonempty=True))
        if not isinstance(tool_call.get("args"), dict):
            raise SchemaValidationError(f"tool_calls[{index}].args must be an object")
        _require_string(tool_call, "reason", nonempty=True)
    if "reobserve_scene" not in tool_names:
        raise SchemaValidationError("tool_calls must contain reobserve_scene")
    _require_enum(payload, "post_recovery_intent", ALLOWED_RECOVERY_INTENTS)
    _require_string(payload, "reason", nonempty=True)
    _require_string(payload, "stop_condition", nonempty=True)


def validate_action_effect_response(payload: dict[str, Any]) -> None:
    _require_object(payload, context="action-effect verification")
    _require_enum(payload, "effect_verified", ALLOWED_EFFECT_VALUES)
    _require_string(payload, "effect_type", nonempty=True)
    _require_confidence(payload)
    for key in (
        "failure_reason",
        "memory_update",
    ):
        _require_string(payload, key)
    _require_string(payload, "evidence_summary", nonempty=True)
    _require_string(payload, "next_constraint", nonempty=True)
    _require_enum(payload, "subtask_status", ALLOWED_SUBTASK_STATUS_VALUES)
    _require_enum(payload, "recommended_control", ALLOWED_VERIFIER_CONTROL_VALUES)


def _validate_perception_queries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _require_object(payload, context="perception queries")
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise SchemaValidationError("queries must be a list")
    if len(queries) > 3:
        raise SchemaValidationError("queries exceeds the requested maximum of 3")
    for index, query_item in enumerate(queries):
        if not isinstance(query_item, dict):
            raise SchemaValidationError(f"queries[{index}] must be an object")
        _require_string(query_item, "object_id", nonempty=True)
        _require_string(query_item, "text_prompt", nonempty=True)
        _require_enum(query_item, "role", ALLOWED_QUERY_ROLES)
        _require_string(query_item, "instance_hint")
        _require_string(query_item, "reason")
    return queries


def validate_perception_query_response(payload: dict[str, Any]) -> None:
    queries = _validate_perception_queries(payload)
    if not any(query_item.get("role") == "target" for query_item in queries):
        raise SchemaValidationError("queries must contain at least one target")


def validate_perception_normalization_response(payload: dict[str, Any]) -> None:
    queries = _validate_perception_queries(payload)
    if not any(query_item.get("role") == "target" for query_item in queries):
        raise SchemaValidationError(
            "normalized queries must contain at least one target"
        )


def _canonical_action_effect_skill_body() -> str:
    try:
        body = ACTION_EFFECT_SKILL_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot load canonical action-effect skill: {ACTION_EFFECT_SKILL_PATH}"
        ) from exc
    if not body.strip():
        raise ValueError(
            f"canonical action-effect skill is empty: {ACTION_EFFECT_SKILL_PATH}"
        )
    return body


@lru_cache(maxsize=1)
def _canonical_planner_prompt_template() -> str:
    try:
        skill_body = CONTROL_TURN_PLANNER_SKILL_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot load canonical planner skill: {CONTROL_TURN_PLANNER_SKILL_PATH}"
        ) from exc
    marker = "## Prompt Body"
    if marker not in skill_body:
        raise ValueError(
            f"canonical planner skill has no {marker!r}: {CONTROL_TURN_PLANNER_SKILL_PATH}"
        )
    prompt_template = skill_body.split(marker, 1)[1].strip()
    if not prompt_template:
        raise ValueError(
            f"canonical planner prompt is empty: {CONTROL_TURN_PLANNER_SKILL_PATH}"
        )
    return prompt_template


def _render_planner_prompt(
    *,
    task: str,
    previous_memory_text: str,
    planner_state: list[float],
) -> str:
    rendered = _canonical_planner_prompt_template()
    fields = {
        "task": task,
        "previous_memory_text": previous_memory_text,
        "state_summary": json.dumps(planner_state, ensure_ascii=False),
    }
    for key, value in fields.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _json_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _without_inline_image_data(value: Any) -> Any:
    """Return a report-only copy with inline image bytes removed.

    The long-context target deliberately excludes base64 pixels: a large camera
    frame must not accidentally satisfy a textual planner-context preflight.
    """

    if isinstance(value, list):
        return [_without_inline_image_data(item) for item in value]
    if not isinstance(value, dict):
        return value
    scrubbed: dict[str, Any] = {}
    is_image_item = value.get("type") == "image"
    for key, item in value.items():
        if key in {
            "planner_start_image_b64",
            "planner_end_image_b64",
            "image_b64",
            "image_b64_by_camera",
        }:
            scrubbed[key] = {} if isinstance(item, dict) else ""
        elif key == "data" and is_image_item:
            scrubbed[key] = ""
        else:
            scrubbed[key] = _without_inline_image_data(item)
    return scrubbed


def _payload_byte_counts(payload: dict[str, Any]) -> tuple[int, int]:
    return (
        len(_json_body(payload)),
        len(_json_body(_without_inline_image_data(payload))),
    )


def _diagnostic_context_record(index: int) -> dict[str, Any]:
    digest = hashlib.sha256(f"tcm-planner-preflight-{index}".encode("utf-8")).hexdigest()
    return {
        "sequence": index,
        "observation_id": f"diagnostic_observation_{index:05d}",
        "track_id": f"track_{index % 8:04d}",
        "source_camera": "head" if index % 2 == 0 else "third",
        "world_m": [
            round(-0.24 + (index % 31) * 0.0017, 6),
            round(-0.18 + (index % 29) * 0.0019, 6),
            round(0.72 + (index % 7) * 0.0008, 6),
        ],
        "position_validity": "memory_valid",
        "motion_event": "none",
        "evidence_sha256": digest,
    }


def _long_planner_payload(
    *,
    images: list[ImageInput],
    minimum_task_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if minimum_task_bytes <= 0:
        raise ValueError("minimum_task_bytes must be positive")
    head, third = images
    instruction = (
        "Observe the scene and propose one safe next manipulation step toward picking "
        "up the visible movable block."
    )
    previous_memory_text = "No task-relevant state change has been committed yet."
    planner_state = [0.0, 0.0, 0.0, 0.0]
    diagnostic_records: list[dict[str, Any]] = []
    task_envelope: dict[str, Any] = {
        "instruction": instruction,
        "memory_harness": {
            "preflight_only": True,
            "semantic_constraint": (
                "diagnostic_context_records exercise request transport only and do not "
                "assert task progress or physical scene changes"
            ),
            "runtime_evaluation": {
                "global_task_success": False,
                "step_count": 0,
                "step_limit": 150,
            },
            "agent_state": {
                "task": {"completed_skills": [], "failed_skills": []},
                "working": {"manipulation_state": {}, "scene_memory": {}},
            },
            "diagnostic_context_records": diagnostic_records,
        },
    }

    def build_payload() -> dict[str, Any]:
        task = json.dumps(task_envelope, ensure_ascii=False, separators=(",", ":"))
        prompt = _render_planner_prompt(
            task=task,
            previous_memory_text=previous_memory_text,
            planner_state=planner_state,
        )
        return {
            "task": task,
            "previous_memory_text": previous_memory_text,
            "planner_state": planner_state,
            "planner_start_image_b64": head.data_b64,
            "planner_end_image_b64": third.data_b64,
            "prompt": prompt,
        }

    payload = build_payload()
    task_bytes = len(str(payload["task"]).encode("utf-8"))
    while task_bytes < minimum_task_bytes:
        diagnostic_records.append(_diagnostic_context_record(len(diagnostic_records)))
        payload = build_payload()
        task_bytes = len(str(payload["task"]).encode("utf-8"))

    request_bytes, non_image_bytes = _payload_byte_counts(payload)
    prompt_bytes = len(str(payload["prompt"]).encode("utf-8"))
    return payload, {
        "planner_task_requested_min_utf8_byte_count": minimum_task_bytes,
        "planner_task_actual_utf8_byte_count": task_bytes,
        "planner_payload_actual_non_image_byte_count": non_image_bytes,
        "planner_payload_actual_request_body_byte_count": request_bytes,
        "planner_rendered_prompt_byte_count": prompt_bytes,
        "planner_diagnostic_context_record_count": len(diagnostic_records),
        "planner_rendered_prompt_present": True,
    }


def _media(images: list[ImageInput]) -> list[dict[str, Any]]:
    return [
        {
            "type": "image",
            "source": "base64",
            "mime_type": image.mime_type,
            "data": image.data_b64,
            "camera": image.camera,
        }
        for image in images
    ]


def _image_map(images: list[ImageInput]) -> dict[str, str]:
    return {image.camera: image.data_b64 for image in images}


def build_probe_specs(
    images: list[ImageInput],
    *,
    planner_task_min_bytes: int = 0,
) -> list[ProbeSpec]:
    if planner_task_min_bytes < 0:
        raise ValueError("planner_task_min_bytes must be nonnegative")
    if planner_task_min_bytes > MAX_PLANNER_TASK_MIN_BYTES:
        raise ValueError(
            "planner_task_min_bytes exceeds the safety limit of "
            f"{MAX_PLANNER_TASK_MIN_BYTES}"
        )
    head, third = images
    task = "Observe the scene and propose one safe next manipulation step toward picking up the visible movable block."
    observation = "Use the supplied head and third camera images as the current visual evidence."
    media = _media(images)
    image_b64_by_camera = _image_map(images)
    action_effect_skill_body = _canonical_action_effect_skill_body()
    if planner_task_min_bytes:
        planner_payload, planner_request_metadata = _long_planner_payload(
            images=images,
            minimum_task_bytes=planner_task_min_bytes,
        )
    else:
        planner_payload = {
            "task": task,
            "previous_memory_text": "No task-relevant state change has been committed yet.",
            "planner_state": [0.0, 0.0, 0.0, 0.0],
            "planner_start_image_b64": head.data_b64,
            "planner_end_image_b64": third.data_b64,
        }
        request_bytes, non_image_bytes = _payload_byte_counts(planner_payload)
        planner_request_metadata = {
            "planner_task_requested_min_utf8_byte_count": 0,
            "planner_task_actual_utf8_byte_count": len(task.encode("utf-8")),
            "planner_payload_actual_non_image_byte_count": non_image_bytes,
            "planner_payload_actual_request_body_byte_count": request_bytes,
            "planner_rendered_prompt_byte_count": 0,
            "planner_diagnostic_context_record_count": 0,
            "planner_rendered_prompt_present": False,
        }
    return [
        ProbeSpec(
            endpoint="/plan",
            mode="planner",
            payload=planner_payload,
            validator=validate_plan_response,
            request_metadata=planner_request_metadata,
        ),
        ProbeSpec(
            endpoint="/ood",
            mode="ood_detection",
            payload={
                "skill_payload": {
                    "selected_skill": "ood-detection",
                    "current_subtask": "Inspect the visible block before approaching it.",
                    "observation_summary": observation,
                    "monitor_status": "preflight",
                    "recovery_state": {},
                    "execution_context": {"step_count": 0, "step_limit": 150},
                    "step_count": 0,
                    "step_limit": 150,
                },
                "media": media,
            },
            validator=validate_ood_response,
        ),
        ProbeSpec(
            endpoint="/recover",
            mode="recovery_planning",
            payload={
                "recovery_payload": {
                    "signal_name": "object_not_visible",
                    "OOD_scenario": "object_not_visible",
                    "reason": "Preflight asks for a bounded observation-only recovery plan.",
                    "global_task": task,
                    "current_subtask": "Reobserve the scene without moving either arm.",
                    "observation_summary": observation,
                    "robot_state": {},
                    "scene_memory": {"instances": []},
                    "observation_preprocess": {},
                    "preferred_arm": "either",
                    "blocked_grounded_setups": [],
                    "recovery_state": {},
                    "recovery_history": [],
                    "available_tools": ["reobserve_scene"],
                    "semantic_tags": {},
                    "retrieved_experience": {},
                    "router_skill": {
                        "name": "recovery-router",
                        "description": "Choose one bounded recovery workflow.",
                        "body": "Use reobserve_scene when fresh visual evidence is required.",
                    },
                    "workflow_skills": [
                        {
                            "name": "reobserve-scene",
                            "description": "Acquire one fresh observation.",
                            "body": "Call reobserve_scene once and then replan.",
                        }
                    ],
                    "primitive_skills": [],
                    "signal_workflow_candidates": ["reobserve-scene"],
                    "skill_payload_mode": "signal_scoped",
                },
                "media": media,
            },
            validator=validate_recovery_response,
        ),
        ProbeSpec(
            endpoint="/recover",
            mode="action_effect_verification",
            payload={
                "recovery_payload": {
                    "mode": "action_effect_verification",
                    "evidence_payload": {
                        "global_task": task,
                        "current_subtask": "Inspect the visible block.",
                        "intended_effect": "Obtain a fresh view of the scene.",
                        "action_summary": "No physical motion; capture current views.",
                        "before_observation": "Preflight reference frame supplied.",
                        "after_observation": observation,
                    },
                    "verification_skill": {
                        "name": "action-effect-verification",
                        "description": "Judge an action effect from before/after evidence.",
                        "body": action_effect_skill_body,
                    },
                },
                "prompt": (
                    "Return JSON only with keys effect_verified (true, false, or unverified), "
                    "effect_type, confidence, evidence_summary, failure_reason, next_constraint, "
                    "memory_update, subtask_status (in_progress, completed, failed, or uncertain), "
                    "and recommended_control (continue, retry, or replan)."
                ),
                "media": media,
            },
            validator=validate_action_effect_response,
        ),
        ProbeSpec(
            endpoint="/perception_queries",
            mode="perception_query_generation",
            payload={
                "global_task": task,
                "current_subtask": "Locate the visible movable block.",
                "committed_memory": "No object has moved.",
                "observation_summary": observation,
                "robot_state": {},
                "active_skill": "monitored-subtask-execution",
                "max_queries": 3,
                "camera": "head",
                "cameras": ["head", "third"],
                "oracle_objects": [],
                "scene_instances": [],
                "require_instance_binding": False,
                "instance_binding_phase": "candidate_discovery",
                "binding_requirement": {"required_any_roles": ["target"]},
                "image_b64": head.data_b64,
                "image_b64_by_camera": image_b64_by_camera,
            },
            validator=validate_perception_query_response,
        ),
        ProbeSpec(
            endpoint="/normalize_perception_queries",
            mode="perception_query_normalization",
            payload={
                "global_task": task,
                "current_subtask": "Locate the visible movable block.",
                "committed_memory": "No object has moved.",
                "observation_summary": observation,
                "raw_queries": [
                    {
                        "object_id": "movable_block",
                        "text_prompt": "movable block",
                        "role": "target",
                        "instance_hint": "",
                        "reason": "The current subtask refers to the visible movable block.",
                    }
                ],
                "max_queries": 3,
                "oracle_objects": [],
                "scene_instances": [],
                "require_instance_binding": False,
                "instance_binding_phase": "candidate_discovery",
                "binding_requirement": {"required_any_roles": ["target"]},
                "image_b64": head.data_b64,
                "image_b64_by_camera": image_b64_by_camera,
            },
            validator=validate_perception_normalization_response,
        ),
    ]


def _http_request(
    *,
    url: str,
    timeout_sec: float,
    payload: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
) -> HttpResult:
    if (payload is None) == (raw_body is None):
        raise ValueError("exactly one of payload or raw_body must be supplied")
    body = raw_body if raw_body is not None else _json_body(payload)
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with _NO_PROXY_OPENER.open(http_request, timeout=timeout_sec) as response:
            response_body = response.read()
            status = int(response.getcode())
        return HttpResult(
            status=status,
            body=response_body,
            latency_ms=round((time.monotonic() - started) * 1000.0, 3),
        )
    except error.HTTPError as exc:
        return HttpResult(
            status=int(exc.code),
            body=exc.read(),
            latency_ms=round((time.monotonic() - started) * 1000.0, 3),
            transport_error=exc,
        )
    except Exception as exc:
        return HttpResult(
            status=None,
            body=b"",
            latency_ms=round((time.monotonic() - started) * 1000.0, 3),
            transport_error=exc,
        )


def _http_get(*, url: str, timeout_sec: float) -> HttpResult:
    http_request = request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    started = time.monotonic()
    try:
        with _NO_PROXY_OPENER.open(http_request, timeout=timeout_sec) as response:
            response_body = response.read()
            status = int(response.getcode())
        return HttpResult(
            status=status,
            body=response_body,
            latency_ms=round((time.monotonic() - started) * 1000.0, 3),
        )
    except error.HTTPError as exc:
        return HttpResult(
            status=int(exc.code),
            body=exc.read(),
            latency_ms=round((time.monotonic() - started) * 1000.0, 3),
            transport_error=exc,
        )
    except Exception as exc:
        return HttpResult(
            status=None,
            body=b"",
            latency_ms=round((time.monotonic() - started) * 1000.0, 3),
            transport_error=exc,
        )


def _response_metadata(result: HttpResult) -> dict[str, Any]:
    return {
        "http_status": result.status,
        "latency_ms": result.latency_ms,
        "response_byte_count": len(result.body),
        "response_sha256": hashlib.sha256(result.body).hexdigest(),
    }


def _probe_record_base(spec: ProbeSpec, images: list[ImageInput]) -> dict[str, Any]:
    request_bytes, non_image_bytes = _payload_byte_counts(spec.payload)
    record = {
        "endpoint": spec.endpoint,
        "mode": spec.mode,
        "request_image_count": len(images),
        "request_images": [image.report_record(order) for order, image in enumerate(images)],
        "request_payload_byte_count": request_bytes,
        "request_non_image_payload_byte_count": non_image_bytes,
        "fallback_count": 0,
    }
    if spec.request_metadata:
        record.update(spec.request_metadata)
    return record


def run_positive_probe(
    *,
    base_url: str,
    timeout_sec: float,
    spec: ProbeSpec,
    images: list[ImageInput],
) -> dict[str, Any]:
    record = _probe_record_base(spec, images)
    result = _http_request(
        url=f"{base_url}{spec.endpoint}",
        timeout_sec=timeout_sec,
        payload=spec.payload,
    )
    record.update(_response_metadata(result))
    if result.transport_error is not None and result.status is None:
        record.update(
            status="failed",
            error=f"transport error: {type(result.transport_error).__name__}: {result.transport_error}",
        )
        return record
    if result.status is None or not 200 <= result.status < 300:
        record.update(status="failed", error=f"expected HTTP 2xx, received {result.status}")
        return record
    try:
        response_payload = json.loads(result.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        record.update(status="failed", error=f"response is not valid UTF-8 JSON: {exc}")
        return record
    try:
        response_object = _require_object(response_payload, context=spec.mode)
        spec.validator(response_object)
    except (SchemaValidationError, ValueError, TypeError) as exc:
        record.update(status="failed", error=f"response schema validation failed: {exc}")
        return record
    record.update(status="passed", schema_valid=True)
    return record


def run_planner_context_contract_probe(
    *,
    base_url: str,
    timeout_sec: float,
    expected_prompt_mode: str,
    expected_max_output_tokens: int | None,
) -> dict[str, Any]:
    result = _http_get(url=f"{base_url}/health", timeout_sec=timeout_sec)
    record: dict[str, Any] = {
        "endpoint": "/health",
        "mode": "planner_context_contract",
        "request_image_count": 0,
        "request_images": [],
        "fallback_count": 0,
        "expected_planner_prompt_mode": expected_prompt_mode,
        "expected_planner_max_output_tokens": expected_max_output_tokens,
        **_response_metadata(result),
    }
    if result.transport_error is not None and result.status is None:
        record.update(
            status="failed",
            error=f"transport error: {type(result.transport_error).__name__}: {result.transport_error}",
        )
        return record
    if result.status is None or not 200 <= result.status < 300:
        record.update(status="failed", error=f"expected HTTP 2xx, received {result.status}")
        return record
    try:
        health = _require_object(
            json.loads(result.body.decode("utf-8")),
            context="health",
        )
    except (UnicodeDecodeError, json.JSONDecodeError, SchemaValidationError) as exc:
        record.update(status="failed", error=f"health response is not valid JSON: {exc}")
        return record
    actual_prompt_mode = health.get("planner_prompt_mode")
    actual_max_output_tokens = health.get("planner_max_output_tokens")
    record.update(
        planner_prompt_mode=actual_prompt_mode,
        planner_max_output_tokens=actual_max_output_tokens,
    )
    mismatches: list[str] = []
    if expected_prompt_mode and actual_prompt_mode != expected_prompt_mode:
        mismatches.append(
            f"planner_prompt_mode expected {expected_prompt_mode!r}, got {actual_prompt_mode!r}"
        )
    if (
        expected_max_output_tokens is not None
        and actual_max_output_tokens != expected_max_output_tokens
    ):
        mismatches.append(
            "planner_max_output_tokens expected "
            f"{expected_max_output_tokens}, got {actual_max_output_tokens!r}"
        )
    if mismatches:
        record.update(status="failed", error="; ".join(mismatches))
        return record
    record.update(status="passed", schema_valid=True)
    return record


def run_malformed_json_probe(*, base_url: str, timeout_sec: float) -> dict[str, Any]:
    result = _http_request(
        url=f"{base_url}/plan",
        timeout_sec=timeout_sec,
        raw_body=b'{"malformed_json":',
    )
    record: dict[str, Any] = {
        "endpoint": "/plan",
        "mode": "malformed_json_negative",
        "request_image_count": 0,
        "request_images": [],
        "fallback_count": 0,
        **_response_metadata(result),
    }
    if result.status != 400:
        if result.transport_error is not None and result.status is None:
            record.update(
                status="failed",
                error=(
                    "malformed JSON must return HTTP 400 JSON, but the request failed "
                    f"in transport: {type(result.transport_error).__name__}: "
                    f"{result.transport_error}"
                ),
            )
            return record
        record.update(
            status="failed",
            error=f"malformed JSON must return HTTP 400, received {result.status}",
        )
        return record
    try:
        response_payload = json.loads(result.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        record.update(
            status="failed",
            error=f"malformed JSON rejection body is not valid UTF-8 JSON: {exc}",
        )
        return record
    if not isinstance(response_payload, dict):
        record.update(status="failed", error="malformed JSON rejection body must be an object")
        return record
    if response_payload.get("error_category") != "malformed_json":
        record.update(
            status="failed",
            error="malformed JSON rejection must contain error_category=malformed_json",
        )
        return record
    record.update(
        status="passed",
        rejection="http_400_malformed_json",
        error_category="malformed_json",
    )
    return record


def _validate_local_base_url(raw_base_url: str) -> str:
    base_url = raw_base_url.strip().rstrip("/")
    parsed = parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("gateway_base_url must be an absolute http:// or https:// URL")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "?" in base_url
        or "#" in base_url
    ):
        raise ValueError(
            "gateway_base_url must not contain credentials, a query, or a fragment"
        )
    if parsed.hostname.lower() not in LOCAL_HOSTS:
        raise ValueError(
            "gateway_base_url must use localhost, 127.0.0.1, or ::1; remote gateways "
            "are forbidden"
        )
    return base_url


def _write_report(path: Path, report: dict[str, Any]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def run_preflight(
    *,
    gateway_base_url: str,
    head_image_path: Path,
    third_image_path: Path,
    output_json_path: Path,
    timeout_sec: float,
    include_malformed_json_probe: bool = True,
    planner_task_min_bytes: int = 0,
    expected_planner_prompt_mode: str = "",
    expected_planner_max_output_tokens: int | None = None,
) -> dict[str, Any]:
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if (
        expected_planner_max_output_tokens is not None
        and expected_planner_max_output_tokens <= 0
    ):
        raise ValueError("expected_planner_max_output_tokens must be positive when set")
    base_url = _validate_local_base_url(gateway_base_url)
    images = [
        ImageInput.load(camera="head", path=head_image_path),
        ImageInput.load(camera="third", path=third_image_path),
    ]
    started_at = datetime.now(timezone.utc)
    specs = build_probe_specs(
        images,
        planner_task_min_bytes=planner_task_min_bytes,
    )
    records: list[dict[str, Any]] = []
    if expected_planner_prompt_mode or expected_planner_max_output_tokens is not None:
        records.append(
            run_planner_context_contract_probe(
                base_url=base_url,
                timeout_sec=timeout_sec,
                expected_prompt_mode=expected_planner_prompt_mode,
                expected_max_output_tokens=expected_planner_max_output_tokens,
            )
        )
    records.extend(
        run_positive_probe(
            base_url=base_url,
            timeout_sec=timeout_sec,
            spec=spec,
            images=images,
        )
        for spec in specs
    )
    if include_malformed_json_probe:
        records.append(run_malformed_json_probe(base_url=base_url, timeout_sec=timeout_sec))
    finished_at = datetime.now(timezone.utc)
    failures = [
        {"endpoint": item["endpoint"], "mode": item["mode"], "error": item.get("error", "failed")}
        for item in records
        if item.get("status") != "passed"
    ]
    planner_spec = specs[0]
    planner_request_bytes, planner_non_image_bytes = _payload_byte_counts(
        planner_spec.payload
    )
    report = {
        "schema_version": 1,
        "preflight": "tcm_agent_fullstack",
        "gateway_base_url": base_url,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_ms": round((finished_at - started_at).total_seconds() * 1000.0, 3),
        "source_images": [image.report_record(order) for order, image in enumerate(images)],
        "fallback_policy": "disabled",
        "fallback_count": 0,
        "planner_payload": {
            "requested_min_task_utf8_byte_count": planner_task_min_bytes,
            "actual_task_utf8_byte_count": len(
                str(planner_spec.payload["task"]).encode("utf-8")
            ),
            "actual_non_image_byte_count": planner_non_image_bytes,
            "actual_request_body_byte_count": planner_request_bytes,
            "request_image_count": 2,
            "rendered_prompt_present": "prompt" in planner_spec.payload,
        },
        "planner_context_contract": {
            "expected_prompt_mode": expected_planner_prompt_mode or None,
            "expected_max_output_tokens": expected_planner_max_output_tokens,
        },
        "probe_count": len(records),
        "passed_probe_count": len(records) - len(failures),
        "failed_probe_count": len(failures),
        "passed": not failures,
        "failures": failures,
        "probes": records,
    }
    _write_report(output_json_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_preflight(
            gateway_base_url=args.gateway_base_url,
            head_image_path=args.head_image,
            third_image_path=args.third_image,
            output_json_path=args.output_json,
            timeout_sec=args.timeout_sec,
            include_malformed_json_probe=not args.skip_malformed_json_probe,
            planner_task_min_bytes=args.planner_task_min_bytes,
            expected_planner_prompt_mode=args.expected_planner_prompt_mode,
            expected_planner_max_output_tokens=args.expected_planner_max_output_tokens,
        )
    except Exception as exc:
        print(f"[agent-fullstack-preflight] setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not report["passed"]:
        print(
            "[agent-fullstack-preflight] failed: "
            + "; ".join(
                f"{item['endpoint']}[{item['mode']}]: {item['error']}"
                for item in report["failures"]
            ),
            file=sys.stderr,
        )
        return 1
    print(
        f"[agent-fullstack-preflight] passed {report['passed_probe_count']}/{report['probe_count']} "
        f"probes; report={args.output_json.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

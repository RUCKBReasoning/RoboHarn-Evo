from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any
from urllib import request

import numpy as np
from PIL import Image

from roboharn_evo.agent.arm_contract import normalize_preferred_arm
from roboharn_evo.agent.experience import normalize_semantic_tags
from roboharn_evo.models.paths import robobrain_inference_path
from roboharn_evo.models.prompt_rendering import render_known_prompt_fields


ALLOWED_COMMIT_LABELS = {"no_update", "subtask_complete", "state_change"}
DEFAULT_ROBOBRAIN_PLANNER_PROMPT = """You are the planner VLM in a long-horizon robot manipulation system.
You receive:
- the global task instruction
- the previous committed memory
- two images: segment start frame and segment end frame
- a numeric state summary vector

Return JSON only with this schema:
{
  "commit_label": "no_update | subtask_complete | state_change",
  "memory_text": "one concise sentence describing committed task state after this segment",
  "subtask_text": "the next subtask the executor should perform",
  "preferred_arm": "left | right | either",
  "semantic_tags": {
    "task_family": "pick_and_place | open_drawer | close_drawer | articulated_object | tool_use | other",
    "subtask_type": "grasp | place | open | close | move | align | reobserve | recover | other",
    "state_tags": {
      "object_state": "",
      "visibility_state": "",
      "gripper_state": "",
      "motion_state": ""
    }
  }
}

Rules:
- Use exactly one commit label from: no_update, subtask_complete, state_change.
- If there is no task-relevant state update, set commit_label to no_update and keep memory consistent.
- memory_text must describe current committed state, not a future plan.
- subtask_text must be the next action objective for the executor.
- Select preferred_arm from current geometry and robot state; use either only when neither arm has a meaningful advantage.
- semantic_tags is optional. If present, use only the exact taxonomy values shown above; leave uncertain state tag values as empty strings.
- Do not output explanations, markdown, or extra keys.

Task instruction: {task}
Previous committed memory: {previous_memory_text}
State summary vector: {state_summary}
"""


def _load_robobrain_inference_class() -> Any:
    module_path = robobrain_inference_path()
    spec = importlib.util.spec_from_file_location("roboharn_evo_robobrain_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load RoboBrain inference module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UnifiedInference


def _to_png_base64(image: np.ndarray) -> str:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _from_png_base64(encoded: str) -> np.ndarray:
    payload = base64.b64decode(encoded.encode("utf-8"))
    image = Image.open(BytesIO(payload)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        payload = json.loads(stripped[first : last + 1])
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Failed to parse RoboBrain JSON output: {text}")


def _normalize_prediction(payload: dict[str, Any], previous_memory_text: str) -> dict[str, Any]:
    commit_label = str(payload.get("commit_label", "")).strip()
    if not commit_label:
        update_required = payload.get("update_required")
        update_trigger = str(payload.get("update_trigger", "")).strip()
        if update_required is False:
            commit_label = "no_update"
        elif update_trigger:
            commit_label = update_trigger
        else:
            commit_label = "state_change"
    if commit_label not in ALLOWED_COMMIT_LABELS:
        commit_label = "state_change" if commit_label else "no_update"

    memory_text = str(
        payload.get("memory_text", payload.get("memory", previous_memory_text))
    ).strip()
    if commit_label == "no_update" and not memory_text:
        memory_text = previous_memory_text

    subtask_text = str(
        payload.get("subtask_text", payload.get("subtask", payload.get("next_subtask", "")))
    ).strip()

    if not memory_text:
        raise ValueError(f"RoboBrain planner output missing memory_text: {payload}")
    if not subtask_text:
        raise ValueError(f"RoboBrain planner output missing subtask_text: {payload}")

    result = {
        "commit_label": commit_label,
        "memory_text": memory_text,
        "subtask_text": subtask_text,
        "preferred_arm": normalize_preferred_arm(payload.get("preferred_arm")),
        "raw_output": payload,
    }
    if isinstance(payload.get("semantic_tags"), dict):
        result["semantic_tags"] = normalize_semantic_tags(payload.get("semantic_tags"), default_source="planner_vlm")
    return result


def _format_state_summary(planner_state: np.ndarray) -> str:
    values = np.asarray(planner_state, dtype=np.float32).reshape(-1)
    rounded = [round(float(value), 4) for value in values.tolist()]
    return json.dumps(rounded, ensure_ascii=False)


@dataclass(frozen=True)
class RoboBrainPlannerConfig:
    mode: str
    model_id: str
    device_map: str
    server_url: str
    timeout_sec: int
    do_sample: bool
    temperature: float
    prompt_template: str


class RoboBrainPlannerAdapter:
    def __init__(self, config: RoboBrainPlannerConfig) -> None:
        self.config = config
        self._inference_model: Any | None = None

    def reset(self) -> None:
        # RoboBrain planner is stateless across calls.
        return

    def _lazy_load_local_model(self) -> Any:
        if self._inference_model is not None:
            return self._inference_model
        inference_class = _load_robobrain_inference_class()
        self._inference_model = inference_class(
            model_id=self.config.model_id,
            device_map=self.config.device_map,
        )
        return self._inference_model

    def _build_prompt(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_state: np.ndarray,
    ) -> str:
        return render_known_prompt_fields(
            self.config.prompt_template,
            task=task.strip(),
            previous_memory_text=previous_memory_text.strip(),
            state_summary=_format_state_summary(planner_state),
        )

    def predict_planner_step(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: np.ndarray,
        planner_end_image: np.ndarray,
        planner_state: np.ndarray,
    ) -> dict[str, Any]:
        if self.config.mode == "remote":
            return self._predict_remote(
                task=task,
                previous_memory_text=previous_memory_text,
                planner_start_image=planner_start_image,
                planner_end_image=planner_end_image,
                planner_state=planner_state,
            )
        if self.config.mode == "local":
            return self._predict_local(
                task=task,
                previous_memory_text=previous_memory_text,
                planner_start_image=planner_start_image,
                planner_end_image=planner_end_image,
                planner_state=planner_state,
            )
        raise ValueError(f"Unsupported RoboBrain planner mode: {self.config.mode}")

    def _predict_local(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: np.ndarray,
        planner_end_image: np.ndarray,
        planner_state: np.ndarray,
    ) -> dict[str, Any]:
        model = self._lazy_load_local_model()
        prompt = self._build_prompt(
            task=task,
            previous_memory_text=previous_memory_text,
            planner_state=planner_state,
        )
        with tempfile.TemporaryDirectory(prefix="robobrain_planner_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            start_path = tmp_path / "start.png"
            end_path = tmp_path / "end.png"
            Image.fromarray(np.asarray(planner_start_image, dtype=np.uint8)).save(start_path)
            Image.fromarray(np.asarray(planner_end_image, dtype=np.uint8)).save(end_path)
            result = model.inference(
                prompt,
                [str(start_path), str(end_path)],
                task="general",
                plot=False,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
            )
        payload = _extract_json_object(str(result["answer"]))
        normalized = _normalize_prediction(payload, previous_memory_text=previous_memory_text)
        normalized["planner_backend"] = "robobrain_adapter"
        return normalized

    def _predict_remote(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: np.ndarray,
        planner_end_image: np.ndarray,
        planner_state: np.ndarray,
    ) -> dict[str, Any]:
        body = {
            "task": task,
            "previous_memory_text": previous_memory_text,
            "planner_state": np.asarray(planner_state, dtype=np.float32).reshape(-1).tolist(),
            "planner_start_image_b64": _to_png_base64(planner_start_image),
            "planner_end_image_b64": _to_png_base64(planner_end_image),
        }
        payload = json.dumps(body).encode("utf-8")
        http_request = request.Request(
            self.config.server_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.config.timeout_sec) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(response_payload, dict):
            raise ValueError(f"Invalid planner server response: {response_payload!r}")
        normalized = _normalize_prediction(response_payload, previous_memory_text=previous_memory_text)
        normalized["planner_backend"] = "robobrain_adapter"
        return normalized


def decode_remote_request(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": str(payload["task"]),
        "previous_memory_text": str(payload.get("previous_memory_text", "")),
        "planner_state": np.asarray(payload["planner_state"], dtype=np.float32),
        "planner_start_image": _from_png_base64(str(payload["planner_start_image_b64"])),
        "planner_end_image": _from_png_base64(str(payload["planner_end_image_b64"])),
    }

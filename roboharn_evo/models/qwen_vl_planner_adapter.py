from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import base64
import json
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

from roboharn_evo.models.robobrain_adapter import (
    ALLOWED_COMMIT_LABELS,
    DEFAULT_ROBOBRAIN_PLANNER_PROMPT,
    _extract_json_object,
    _normalize_prediction,
)
from roboharn_evo.models.prompt_rendering import render_known_prompt_fields


def _image_to_data_url(image: np.ndarray) -> str:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def _format_state_summary(planner_state: np.ndarray) -> str:
    values = np.asarray(planner_state, dtype=np.float32).reshape(-1)
    rounded = [round(float(value), 4) for value in values.tolist()]
    return json.dumps(rounded, ensure_ascii=False)


@dataclass(frozen=True)
class QwenVlPlannerConfig:
    base_url: str
    api_key: str
    model_path: str
    timeout_sec: int
    max_tokens: int
    temperature: float
    top_p: float
    prompt_template: str


class QwenVlPlannerAdapter:
    """Planner backend using an OpenAI-compatible Qwen-VL server, matching the Mem-0 style."""

    def __init__(self, config: QwenVlPlannerConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_sec,
        )

    def reset(self) -> None:
        return

    def predict_planner_step(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: np.ndarray,
        planner_end_image: np.ndarray,
        planner_state: np.ndarray,
    ) -> dict[str, Any]:
        prompt = render_known_prompt_fields(
            self.config.prompt_template,
            task=task.strip(),
            previous_memory_text=previous_memory_text.strip(),
            state_summary=_format_state_summary(planner_state),
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Return JSON only."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(planner_start_image)}},
                    {"type": "text", "text": "The previous image is the segment start frame."},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(planner_end_image)}},
                    {"type": "text", "text": "The previous image is the segment end frame."},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.config.model_path,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        answer = response.choices[0].message.content
        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", "") for item in answer if isinstance(item, dict) and item.get("text")
            )
        payload = _extract_json_object(str(answer))
        normalized = _normalize_prediction(payload, previous_memory_text=previous_memory_text)
        normalized["planner_backend"] = "qwen_vl_8b"
        return normalized

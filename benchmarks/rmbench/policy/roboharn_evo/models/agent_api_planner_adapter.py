from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import base64
import json
from typing import Any
from urllib import request

import numpy as np
from PIL import Image

from .prompt_rendering import render_known_prompt_fields


def _encode_image_b64(image: np.ndarray) -> str:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _post_json(*, url: str, payload: dict[str, Any], timeout_sec: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with request.urlopen(http_request, timeout=timeout_sec) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(response_payload, dict):
        raise ValueError(f"Agent API planner response must be a JSON object, got {response_payload!r}")
    return response_payload


@dataclass(frozen=True)
class AgentApiPlannerConfig:
    server_url: str
    timeout_sec: int
    prompt_template: str
    auth_token: str
    auth_header: str
    extra_headers: dict[str, str]
    extra_body: dict[str, Any]


class AgentApiPlannerAdapter:
    """Configurable shell for planner backends served behind an HTTP JSON API.

    Expected response schema:
    {
      "commit_label": "no_update | subtask_complete | state_change",
      "memory_text": "...",
      "subtask_text": "...",
      "preferred_arm": "left | right | either"
    }
    """

    def __init__(self, config: AgentApiPlannerConfig) -> None:
        self.config = config

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
            state_summary=json.dumps(np.asarray(planner_state, dtype=np.float32).reshape(-1).tolist(), ensure_ascii=False),
        )
        headers = dict(self.config.extra_headers)
        if self.config.auth_token:
            headers[self.config.auth_header] = self.config.auth_token
        payload = {
            "task": task,
            "previous_memory_text": previous_memory_text,
            "planner_state": np.asarray(planner_state, dtype=np.float32).reshape(-1).tolist(),
            "planner_start_image_b64": _encode_image_b64(planner_start_image),
            "planner_end_image_b64": _encode_image_b64(planner_end_image),
            "prompt": prompt,
            **self.config.extra_body,
        }
        return _post_json(
            url=self.config.server_url,
            payload=payload,
            timeout_sec=self.config.timeout_sec,
            headers=headers,
        )

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import base64
import json
from typing import Any
from urllib import request

import numpy as np
from PIL import Image


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
        raise ValueError(f"World action model response must be a JSON object, got {response_payload!r}")
    return response_payload


@dataclass(frozen=True)
class WorldActionModelExecutorConfig:
    server_url: str
    timeout_sec: int
    auth_token: str
    auth_header: str
    extra_headers: dict[str, str]
    extra_body: dict[str, Any]
    max_chunk_steps: int
    action_dim: int


class WorldActionModelExecutorAdapter:
    """Executor shell for remote world-action-model style policies.

    Expected response schema:
    {
      "actions": [[...], [...], ...]
    }
    """

    def __init__(self, config: WorldActionModelExecutorConfig) -> None:
        self.config = config

    def reset(self) -> None:
        return

    def predict_action_chunk(
        self,
        *,
        observation: dict[str, Any],
        task: str,
        subtask: str,
        memory: str,
    ) -> np.ndarray:
        if "observation" in observation and "joint_action" in observation:
            head_image = np.asarray(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8)
            left_image = np.asarray(observation["observation"]["left_camera"]["rgb"], dtype=np.uint8)
            right_image = np.asarray(observation["observation"]["right_camera"]["rgb"], dtype=np.uint8)
            state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        else:
            executor_images = np.asarray(observation["executor_images"], dtype=np.uint8)
            executor_state = np.asarray(observation["executor_state"], dtype=np.float32)
            current_images = executor_images[-1] if executor_images.ndim == 5 else executor_images
            current_state = executor_state[-1] if executor_state.ndim == 2 else executor_state
            head_image = np.asarray(current_images[0], dtype=np.uint8)
            left_image = np.asarray(current_images[1], dtype=np.uint8)
            right_image = np.asarray(current_images[2], dtype=np.uint8)
            state_vector = np.asarray(current_state, dtype=np.float32)

        headers = dict(self.config.extra_headers)
        if self.config.auth_token:
            headers[self.config.auth_header] = self.config.auth_token
        payload = {
            **self.config.extra_body,
            "task": "",
            "subtask": subtask,
            "memory": "",
            "state": state_vector.reshape(-1).tolist(),
            "images": {
                "head_camera_b64": _encode_image_b64(head_image),
                "left_camera_b64": _encode_image_b64(left_image),
                "right_camera_b64": _encode_image_b64(right_image),
            },
            "max_chunk_steps": self.config.max_chunk_steps,
            "action_dim": self.config.action_dim,
        }
        response_payload = _post_json(
            url=self.config.server_url,
            payload=payload,
            timeout_sec=self.config.timeout_sec,
            headers=headers,
        )
        if "actions" not in response_payload:
            raise ValueError(f"World action model response missing 'actions': {response_payload}")
        actions = np.asarray(response_payload["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected executor actions with shape (T, D), got {actions.shape}")
        if actions.shape[1] != self.config.action_dim:
            raise ValueError(
                f"Executor returned action_dim={actions.shape[1]}, expected {self.config.action_dim}"
            )
        return actions[: self.config.max_chunk_steps]

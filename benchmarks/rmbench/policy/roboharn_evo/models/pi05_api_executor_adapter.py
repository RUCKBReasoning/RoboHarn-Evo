from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request

import numpy as np


@dataclass(frozen=True)
class Pi05ApiExecutorConfig:
    server_url: str
    timeout_sec: int
    auth_token: str
    auth_header: str
    extra_headers: dict[str, str]
    extra_body: dict[str, Any]
    max_chunk_steps: int
    action_dim: int


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
        raise ValueError(f"Pi05 API response must be a JSON object, got {response_payload!r}")
    return response_payload


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class Pi05ApiExecutorAdapter:
    def __init__(self, config: Pi05ApiExecutorConfig) -> None:
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
        headers = dict(self.config.extra_headers)
        if self.config.auth_token:
            headers[self.config.auth_header] = self.config.auth_token
        payload = {
            **self.config.extra_body,
            "task": "",
            "subtask": subtask,
            "memory": "",
            "observation": _to_jsonable(observation),
            "max_chunk_steps": self.config.max_chunk_steps,
            "action_dim": self.config.action_dim,
        }
        response_payload = _post_json(
            url=self.config.server_url,
            payload=payload,
            timeout_sec=self.config.timeout_sec,
            headers=headers,
        )
        if "action_chunk" not in response_payload:
            raise ValueError(f"Pi05 API response missing 'action_chunk': {response_payload}")
        actions = np.asarray(response_payload["action_chunk"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected executor action_chunk with shape (T, D), got {actions.shape}")
        if actions.shape[1] != self.config.action_dim:
            raise ValueError(
                f"Executor returned action_dim={actions.shape[1]}, expected {self.config.action_dim}"
            )
        return actions[: self.config.max_chunk_steps]

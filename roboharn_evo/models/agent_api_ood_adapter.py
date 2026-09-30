from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from roboharn_evo.models.agent_api_planner_adapter import _post_json


@dataclass(frozen=True)
class AgentApiOODConfig:
    server_url: str
    timeout_sec: int
    prompt_template: str
    auth_token: str
    auth_header: str
    extra_headers: dict[str, str]
    extra_body: dict[str, Any]


class AgentApiOODAdapter:
    def __init__(self, config: AgentApiOODConfig) -> None:
        self.config = config

    def evaluate_ood(self, *, skill_payload: str, media: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        try:
            parsed_payload = json.loads(skill_payload)
        except json.JSONDecodeError:
            parsed_payload = {"raw_skill_payload": skill_payload}
        headers = dict(self.config.extra_headers)
        if self.config.auth_token:
            headers[self.config.auth_header] = self.config.auth_token
        prompt = self.config.prompt_template.format(
            skill_payload=json.dumps(parsed_payload, ensure_ascii=False)
        )
        payload = {
            "skill_payload": parsed_payload,
            "prompt": prompt,
            "media": list(media or []),
            **self.config.extra_body,
        }
        return _post_json(
            url=self.config.server_url,
            payload=payload,
            timeout_sec=self.config.timeout_sec,
            headers=headers,
        )

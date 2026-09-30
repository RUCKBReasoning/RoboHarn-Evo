from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDef:
    service_name: str
    tool_name: str
    description: str
    parameters: dict[str, Any]

    @property
    def openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": f"{self.service_name}___{self.tool_name}",
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ServiceRegister:
    service_name: str
    description: str
    is_activation: bool = True
    is_agent_service: bool = True
    tools_list: list[ToolDef] = field(default_factory=list)

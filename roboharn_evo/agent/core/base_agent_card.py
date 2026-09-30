from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseAgentCard:
    config: Any
    control_runtime: Any
    executor_runtime: Any
    ood_backend_config: dict[str, Any] | None
    recovery_backend_config: dict[str, Any] | None
    memory_store: Any
    skill_registry: Any
    control_model_name: str
    executor_name: str
    hpk_runtime: Any = None
    display_trace: list[str] = field(default_factory=list)

"""Public model interfaces.

Training-only policy objects are loaded lazily so importing the deployment
interfaces does not require the optional PyTorch stack.
"""

from __future__ import annotations

from typing import Any

from roboharn_evo.models.backend_factory import (
    build_executor_backend,
    build_ood_backend,
    build_planner_backend,
    build_recovery_backend,
)
from roboharn_evo.models.backend_interfaces import (
    ExecutorBackend,
    OODBackend,
    PlannerBackend,
    RecoveryBackend,
)

_POLICY_EXPORTS = {
    "RoboHarnPolicy",
    "build_policy_from_config",
    "inference",
    "load_policy_from_checkpoint",
}


def __getattr__(name: str) -> Any:
    if name in _POLICY_EXPORTS:
        from roboharn_evo.models import policy

        return getattr(policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _POLICY_EXPORTS)


__all__ = [
    "ExecutorBackend",
    "OODBackend",
    "PlannerBackend",
    "RecoveryBackend",
    "RoboHarnPolicy",
    "build_executor_backend",
    "build_ood_backend",
    "build_planner_backend",
    "build_policy_from_config",
    "build_recovery_backend",
    "inference",
    "load_policy_from_checkpoint",
]

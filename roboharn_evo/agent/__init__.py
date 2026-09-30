from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "AgenticPlanner": ("roboharn_evo.agent.planner", "AgenticPlanner"),
    "AgenticPlannerConfig": ("roboharn_evo.agent.planner", "AgenticPlannerConfig"),
    "AgentRuntimeConfig": ("roboharn_evo.agent.runtime", "AgentRuntimeConfig"),
    "RoboHarnAgentRuntime": ("roboharn_evo.agent.runtime", "RoboHarnAgentRuntime"),
    "build_agent_runtime": ("roboharn_evo.agent.runtime", "build_agent_runtime"),
    "BaseAgent": ("roboharn_evo.agent.core", "BaseAgent"),
    "BaseAgentCard": ("roboharn_evo.agent.core", "BaseAgentCard"),
    "ImgAgent": ("roboharn_evo.agent.core", "ImgAgent"),
    "AgentSession": ("roboharn_evo.agent.core", "AgentSession"),
    "SkillRegistry": ("roboharn_evo.agent.skills", "SkillRegistry"),
    "SkillSpec": ("roboharn_evo.agent.skills", "SkillSpec"),
    "build_default_skill_registry": (
        "roboharn_evo.agent.skills",
        "build_default_skill_registry",
    ),
    "AgentState": ("roboharn_evo.agent.state", "AgentState"),
    "RoleMemory": ("roboharn_evo.agent.state", "RoleMemory"),
    "TaskMemory": ("roboharn_evo.agent.state", "TaskMemory"),
    "TaskPlanItem": ("roboharn_evo.agent.state", "TaskPlanItem"),
    "WorkingMemory": ("roboharn_evo.agent.state", "WorkingMemory"),
    "SkillRunState": ("roboharn_evo.agent.state", "SkillRunState"),
    "ControlSignal": ("roboharn_evo.agent.decisions", "ControlSignal"),
    "VLAExecutionContext": ("roboharn_evo.agent.vla", "VLAExecutionContext"),
    "VLAInstructionPayload": ("roboharn_evo.agent.vla", "VLAInstructionPayload"),
    "VLAInstructionBuilder": ("roboharn_evo.agent.vla", "VLAInstructionBuilder"),
    "VLAExecutionRequest": ("roboharn_evo.agent.vla", "VLAExecutionRequest"),
    "build_vla_execution_request": (
        "roboharn_evo.agent.vla",
        "build_vla_execution_request",
    ),
    "VLAExecutionState": ("roboharn_evo.agent.vla", "VLAExecutionState"),
    "VLAExecutionStateStore": ("roboharn_evo.agent.vla", "VLAExecutionStateStore"),
    "MonitorSignal": ("roboharn_evo.agent.monitoring", "MonitorSignal"),
    "make_signal": ("roboharn_evo.agent.monitoring", "make_signal"),
    "ProgressAssessment": ("roboharn_evo.agent.monitoring", "ProgressAssessment"),
    "ProgressMonitor": ("roboharn_evo.agent.monitoring", "ProgressMonitor"),
    "OODDetector": ("roboharn_evo.agent.monitoring", "OODDetector"),
    "HandoffTarget": ("roboharn_evo.agent.monitoring", "HandoffTarget"),
    "HandoffDecision": ("roboharn_evo.agent.monitoring", "HandoffDecision"),
    "HandoffPolicy": ("roboharn_evo.agent.monitoring", "HandoffPolicy"),
    "RecoveryToolCall": ("roboharn_evo.agent.recovery", "RecoveryToolCall"),
    "RecoveryToolResult": ("roboharn_evo.agent.recovery", "RecoveryToolResult"),
    "RecoveryToolDispatcher": (
        "roboharn_evo.agent.recovery",
        "RecoveryToolDispatcher",
    ),
    "RecoveryPrimitivePlan": (
        "roboharn_evo.agent.recovery",
        "RecoveryPrimitivePlan",
    ),
    "RecoveryPolicyResolver": (
        "roboharn_evo.agent.recovery",
        "RecoveryPolicyResolver",
    ),
    "SkillRecoveryWorkflowLoader": (
        "roboharn_evo.agent.recovery",
        "SkillRecoveryWorkflowLoader",
    ),
    "ResumeMode": ("roboharn_evo.agent.handoff", "ResumeMode"),
    "ResumeDecision": ("roboharn_evo.agent.handoff", "ResumeDecision"),
    "ResumeStrategy": ("roboharn_evo.agent.handoff", "ResumeStrategy"),
    "SubtaskReentry": ("roboharn_evo.agent.handoff", "SubtaskReentry"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one existing public export without eager package side effects."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REOBSERVE_SCENE_TOOL = "reobserve_scene"
REOBSERVE_SCENE_ABLATION_CONFIG_KEY = (
    "agent.recovery.enable_reobserve"
)


@dataclass(slots=True)
class RecoveryToolCall:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecoveryToolResult:
    tool_name: str
    success: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


RECOVERY_TOOLS = {
    "contact_displace",
    "move_ee_to_pose",
    "move_ee_to_grounded_instance",
    "move_to_home",
    "retreat_arm",
    "lift_ee",
    "open_gripper",
    "close_gripper",
    REOBSERVE_SCENE_TOOL,
    "safe_reset_posture",
}

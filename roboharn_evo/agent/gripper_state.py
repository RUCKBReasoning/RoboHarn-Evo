from __future__ import annotations

import math
from typing import Any


def normalized_gripper_target(value: Any) -> float:
    target = float(value)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("Gripper command target must be finite and within [0, 1]")
    return target


def gripper_command_state(arm_state: Any) -> str:
    # 这里只解释已发送的控制目标；实际开度和物体附接证据由各自字段保存。
    if not isinstance(arm_state, dict) or arm_state.get("gripper_command") is None:
        return "unknown"
    target = normalized_gripper_target(arm_state["gripper_command"])
    if math.isclose(target, 0.0, abs_tol=1e-6):
        return "closed"
    if math.isclose(target, 1.0, abs_tol=1e-6):
        return "open"
    return "hold"


def snapshot_gripper_commands(snapshot: Any) -> dict[str, float]:
    return {
        arm: normalized_gripper_target(value)
        for arm, value in snapshot.gripper_command_by_arm.items()
    }

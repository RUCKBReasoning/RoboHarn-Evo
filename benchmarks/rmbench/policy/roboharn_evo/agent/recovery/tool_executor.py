from __future__ import annotations

from typing import Any

from .recovery_adapter import (
    RecoveryCapabilities,
    RecoveryExecutionResult,
)
from .rmbench_recovery_adapter import RMBenchRecoveryAdapter
from .tool_specs import (
    REOBSERVE_SCENE_ABLATION_CONFIG_KEY,
    REOBSERVE_SCENE_TOOL,
    RecoveryToolCall,
    RecoveryToolResult,
)


class RecoveryToolExecutor:
    def __init__(
        self,
        *,
        oracle_objects_enabled: bool = False,
        reobserve_scene_enabled: bool = True,
    ) -> None:
        self.oracle_objects_enabled = bool(oracle_objects_enabled)
        self.reobserve_scene_enabled = bool(
            reobserve_scene_enabled
        )

    def set_reobserve_scene_enabled(self, enabled: bool) -> None:
        self.reobserve_scene_enabled = bool(enabled)

    def build_adapter(self, task_env: Any) -> RMBenchRecoveryAdapter:
        return RMBenchRecoveryAdapter(
            task_env,
            oracle_objects_enabled=self.oracle_objects_enabled,
        )

    def capabilities(self, task_env: Any) -> RecoveryCapabilities:
        capabilities = self.build_adapter(task_env).capabilities()
        if self.reobserve_scene_enabled:
            return capabilities
        capabilities.supported_tools.discard(REOBSERVE_SCENE_TOOL)
        capabilities.notes = {
            **dict(capabilities.notes),
            "disabled_tools_by_ablation": {
                REOBSERVE_SCENE_TOOL: (
                    REOBSERVE_SCENE_ABLATION_CONFIG_KEY
                )
            },
        }
        return capabilities

    def execute(
        self,
        *,
        call: RecoveryToolCall,
        task_env: Any,
        latest_snapshot: Any | None,
    ) -> RecoveryExecutionResult:
        capabilities = self.capabilities(task_env)
        if (
            call.tool_name == REOBSERVE_SCENE_TOOL
            and not self.reobserve_scene_enabled
        ):
            return RecoveryExecutionResult(
                result=RecoveryToolResult(
                    tool_name=call.tool_name,
                    success=False,
                    message=(
                        "recovery tool disabled by ablation "
                        "configuration"
                    ),
                    details={
                        "disabled_by_ablation": True,
                        "disabled_tool": REOBSERVE_SCENE_TOOL,
                        "ablation_config_key": (
                            REOBSERVE_SCENE_ABLATION_CONFIG_KEY
                        ),
                        "available_tools": (
                            capabilities.available_tools()
                        ),
                    },
                )
            )
        if call.tool_name not in capabilities.supported_tools:
            return RecoveryExecutionResult(
                result=RecoveryToolResult(
                    tool_name=call.tool_name,
                    success=False,
                    message="recovery tool not supported by current adapter capabilities",
                    details={
                        "available_tools": capabilities.available_tools(),
                        "motion_modes": sorted(capabilities.motion_modes),
                        "gripper_api": capabilities.gripper_api,
                    },
                )
            )
        adapter = self.build_adapter(task_env)
        return adapter.execute(call, latest_snapshot)

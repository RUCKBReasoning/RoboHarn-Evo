from __future__ import annotations

from typing import Any

from .recovery_adapter import RecoveryCapabilities
from .tool_executor import RecoveryToolExecutor
from .tool_specs import RecoveryToolCall, RecoveryToolResult


class RecoveryToolDispatcher:
    def __init__(
        self,
        executor: RecoveryToolExecutor | None = None,
        *,
        oracle_objects_enabled: bool = False,
        reobserve_scene_enabled: bool = True,
    ) -> None:
        self._executor = executor or RecoveryToolExecutor(
            oracle_objects_enabled=oracle_objects_enabled,
            reobserve_scene_enabled=reobserve_scene_enabled,
        )
        self.latest_snapshot: Any | None = None
        self.latest_environment_success = False

    def capabilities(self, task_env: Any) -> RecoveryCapabilities:
        return self._executor.capabilities(task_env)

    def set_reobserve_scene_enabled(self, enabled: bool) -> None:
        self._executor.set_reobserve_scene_enabled(enabled)

    @property
    def reobserve_scene_enabled(self) -> bool:
        return bool(self._executor.reobserve_scene_enabled)

    def available_tools(self, task_env: Any) -> list[str]:
        return self.capabilities(task_env).available_tools()

    def dispatch(self, call: RecoveryToolCall, *, task_env: Any, latest_snapshot: Any | None) -> RecoveryToolResult:
        execution = self._executor.execute(call=call, task_env=task_env, latest_snapshot=latest_snapshot)
        self.latest_snapshot = execution.latest_snapshot if execution.latest_snapshot is not None else latest_snapshot
        self.latest_environment_success = self._environment_success(task_env, self.latest_snapshot)
        return execution.result

    def dispatch_batch(self, calls: list[RecoveryToolCall], *, task_env: Any, latest_snapshot: Any | None) -> list[RecoveryToolResult]:
        results: list[RecoveryToolResult] = []
        current_snapshot = latest_snapshot
        self.latest_snapshot = current_snapshot
        terminal_success = self._environment_success(task_env, current_snapshot)
        self.latest_environment_success = terminal_success
        halt_reason = "environment_eval_success" if terminal_success else ""
        reobserved_after_halt = False
        for call in calls:
            if halt_reason:
                if not terminal_success and call.tool_name == "reobserve_scene" and not reobserved_after_halt:
                    result = self.dispatch(call, task_env=task_env, latest_snapshot=current_snapshot)
                    current_snapshot = self.latest_snapshot
                    results.append(result)
                    reobserved_after_halt = True
                    continue
                details = {
                    "skipped": True,
                    "batch_halted": True,
                    "batch_halt_reason": halt_reason,
                }
                if terminal_success:
                    details.update(
                        {
                            "skip_reason": "environment_eval_success",
                            "authority": "environment_eval_success",
                            "environment_success": True,
                            "terminal_skip": True,
                        }
                    )
                results.append(
                    RecoveryToolResult(
                        tool_name=call.tool_name,
                        success=False,
                        message=f"skipped because recovery batch halted: {halt_reason}",
                        details=details,
                    )
                )
                continue
            result = self.dispatch(call, task_env=task_env, latest_snapshot=current_snapshot)
            current_snapshot = self.latest_snapshot
            results.append(result)
            terminal_success = self._environment_success(task_env, current_snapshot)
            self.latest_environment_success = terminal_success
            halt_reason = (
                "environment_eval_success"
                if terminal_success
                else self._batch_halt_reason(result)
            )
        return results

    @staticmethod
    def _environment_success(task_env: Any, snapshot: Any | None) -> bool:
        return bool(getattr(snapshot, "eval_success", False)) or bool(getattr(task_env, "eval_success", False))

    @staticmethod
    def _batch_halt_reason(result: RecoveryToolResult) -> str:
        if not result.success:
            return f"{result.tool_name} failed"
        details = result.details if isinstance(result.details, dict) else {}
        if details.get("target_reached") is False:
            error = details.get("target_observation_error_m")
            if isinstance(error, (int, float)):
                return f"{result.tool_name} stopped {float(error):.4g}m from its target"
            return f"{result.tool_name} did not reach its target"
        return ""

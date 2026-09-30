from __future__ import annotations

import json
from typing import Any, Callable

from .signals import MOTION_BLOCKED, OBJECT_NOT_VISIBLE, MonitorSignal, SCENE_DRIFT_DETECTED, make_signal


class OODDetector:
    def __init__(self, evaluate_ood: Callable[..., list[dict[str, Any]]] | None = None) -> None:
        self._evaluate_ood = evaluate_ood or self._default_evaluate_ood

    def _default_evaluate_ood(self, *, skill_payload: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(skill_payload)
        except json.JSONDecodeError:
            return []
        results: list[dict[str, Any]] = []
        observation_summary = str(payload.get("observation_summary", ""))
        observation_summary_lower = observation_summary.lower()
        if "head camera frame unavailable" in observation_summary_lower:
            results.append(
                {
                    "name": OBJECT_NOT_VISIBLE,
                    "level": "warning",
                    "reason": "head camera frame unavailable",
                    "score": 0.7,
                    "details": {
                        "source": "ood-detection-skill-adapter",
                        "mode": "fallback",
                        "evidence": ["observation_summary"],
                    },
                }
            )
        execution_context = payload.get("execution_context", {})
        if execution_context.get("action_chunk_empty", False):
            results.append(
                {
                    "name": MOTION_BLOCKED,
                    "level": "warning",
                    "reason": "executor returned empty action chunk",
                    "score": 0.6,
                    "details": {
                        "source": "ood-detection-skill-adapter",
                        "mode": "fallback",
                        "evidence": ["execution_context.action_chunk_empty"],
                    },
                }
            )
        try:
            step_count = int(payload.get("step_count", execution_context.get("step_count", 0)))
            step_limit = int(payload.get("step_limit", execution_context.get("step_limit", 0)))
        except Exception:
            step_count = 0
            step_limit = 0
        if step_limit and step_count >= step_limit:
            results.append(
                {
                    "name": SCENE_DRIFT_DETECTED,
                    "level": "warning",
                    "reason": "execution reached environment step limit",
                    "score": 0.5,
                    "details": {
                        "source": "ood-detection-skill-adapter",
                        "mode": "fallback",
                        "evidence": ["step_count", "step_limit"],
                    },
                }
            )
        return results

    def build_skill_payload(
        self,
        *,
        snapshot: Any,
        action_chunk: Any | None,
        current_subtask: str,
        observation_summary: str,
        monitor_status: str,
        recovery_state: dict[str, Any],
        execution_context: dict[str, Any],
    ) -> str:
        action_chunk_empty = False
        if action_chunk is not None:
            try:
                action_chunk_empty = len(action_chunk) == 0
            except Exception:
                action_chunk_empty = False
        step_count = getattr(snapshot, "step_count", 0)
        step_limit = getattr(snapshot, "step_limit", 0)
        payload = {
            "selected_skill": "ood-detection",
            "current_subtask": current_subtask,
            "observation_summary": observation_summary,
            "monitor_status": monitor_status,
            "recovery_state": recovery_state,
            "execution_context": {
                **execution_context,
                "step_count": execution_context.get("step_count", step_count),
                "step_limit": execution_context.get("step_limit", step_limit),
                "action_chunk_empty": action_chunk_empty,
            },
            "step_count": step_count,
            "step_limit": step_limit,
        }
        return json.dumps(payload, ensure_ascii=False)

    def detect(
        self,
        *,
        snapshot: Any,
        action_chunk: Any | None = None,
        current_subtask: str = "",
        observation_summary: str = "",
        monitor_status: str = "",
        recovery_state: dict[str, Any] | None = None,
        execution_context: dict[str, Any] | None = None,
    ) -> list[MonitorSignal]:
        payload = self.build_skill_payload(
            snapshot=snapshot,
            action_chunk=action_chunk,
            current_subtask=current_subtask,
            observation_summary=observation_summary,
            monitor_status=monitor_status,
            recovery_state=dict(recovery_state or {}),
            execution_context=dict(execution_context or {}),
        )
        raw_results = self._evaluate_ood(skill_payload=payload)
        signals: list[MonitorSignal] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            signals.append(
                make_signal(
                    name,
                    level=str(item.get("level", "info")),
                    reason=str(item.get("reason", "")),
                    score=float(item.get("score", 0.0)),
                    details=dict(item.get("details", {})),
                )
            )
        return signals

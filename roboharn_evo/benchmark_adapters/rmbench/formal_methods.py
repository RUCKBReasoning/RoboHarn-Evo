
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FLAT_USAGE_SCHEMA = "roboharn_evo/rmbench/hpk_flat_reflection_usage/v1"


class FormalMethodConfigurationError(ValueError):
    """A formal method configuration is incomplete or internally inconsistent."""


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalMethodConfigurationError(f"{label} must be non-empty text")
    return " ".join(value.strip().split())


@dataclass(frozen=True, slots=True)
class FlatReflectionConfig:
    lesson_path: Path
    max_prompt_chars: int = 4000

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FlatReflectionConfig:
        if not isinstance(value, Mapping):
            raise FormalMethodConfigurationError(
                "rmbench_formal.flat_reflection must be an object"
            )
        payload = dict(value)
        if set(payload) != {"lesson_path", "max_prompt_chars"}:
            raise FormalMethodConfigurationError(
                "flat_reflection requires exactly lesson_path and max_prompt_chars"
            )
        path = Path(_text(payload["lesson_path"], label="lesson_path")).resolve(
            strict=True
        )
        limit = payload["max_prompt_chars"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise FormalMethodConfigurationError(
                "flat_reflection.max_prompt_chars must be a positive integer"
            )
        return cls(lesson_path=path, max_prompt_chars=limit)

    def load_lesson(self) -> str:
        lesson = self.lesson_path.read_text(encoding="utf-8").strip()
        if not lesson:
            raise FormalMethodConfigurationError("flat reflection lesson is empty")
        if len(lesson) > self.max_prompt_chars:
            raise FormalMethodConfigurationError(
                "flat reflection lesson exceeds the frozen prompt budget"
            )
        return lesson


class FlatReflectionPlannerBackend:
    """Present one frozen flat lesson to every planner turn.

    This wrapper changes neither the task instruction nor the planner response.
    The downstream RMBench guards and executor remain unchanged. It is local to
    the benchmark because Flat Reflection is an experiment condition, not a new
    HPK knowledge type.
    """

    def __init__(
        self,
        backend: Any,
        *,
        lesson: str,
        audit_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if not callable(getattr(backend, "predict_planner_step", None)):
            raise TypeError("planner backend must expose predict_planner_step")
        self._backend = backend
        self._lesson = _text(lesson, label="flat reflection lesson")
        self._audit_sink = audit_sink
        self._call_index = 0
        self._last_audit: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    @property
    def hpk_runtime(self) -> Any:
        return getattr(self._backend, "hpk_runtime", None)

    @property
    def last_flat_reflection_audit(self) -> dict[str, Any] | None:
        if self._last_audit is None:
            return None
        return json.loads(json.dumps(self._last_audit, ensure_ascii=False))

    def reset(self) -> None:
        self._call_index = 0
        self._last_audit = None
        reset = getattr(self._backend, "reset", None)
        if callable(reset):
            reset()

    def predict_planner_step(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: Any,
        planner_end_image: Any,
        planner_state: Any,
    ) -> dict[str, Any]:
        self._call_index += 1
        context = (
            "Prior flat reflection from matched source trajectories. It is "
            "advisory experience, not a statement about the current scene:\n"
            f"{self._lesson}\n\n"
            "Current committed rollout state:\n"
            f"{str(previous_memory_text).strip()}"
        )
        audit = {
            "schema": FLAT_USAGE_SCHEMA,
            "call_index": self._call_index,
            "context_applied": True,
            "lesson_char_count": len(self._lesson),
            "task_instruction_changed": False,
            "structured_task_knowledge_used": False,
            "structured_action_knowledge_used": False,
        }
        self._last_audit = audit
        if self._audit_sink is not None:
            self._audit_sink("hpk_flat_reflection_usage", dict(audit))
        return self._backend.predict_planner_step(
            task=task,
            previous_memory_text=context,
            planner_start_image=planner_start_image,
            planner_end_image=planner_end_image,
            planner_state=planner_state,
        )


def install_flat_reflection_method(model: Any, config: Mapping[str, Any]) -> None:
    """Install the formal Flat condition on one RMBench Agent instance."""

    if not isinstance(config, Mapping):
        raise FormalMethodConfigurationError("rmbench_formal must be an object")
    payload = dict(config)
    allowed = {"method", "flat_reflection"}
    if set(payload) - allowed:
        raise FormalMethodConfigurationError(
            f"unknown rmbench_formal fields: {sorted(set(payload) - allowed)}"
        )
    method = _text(payload.get("method"), label="rmbench_formal.method")
    if method != "flat":
        if "flat_reflection" in payload:
            raise FormalMethodConfigurationError(
                "flat_reflection config is only valid for method=flat"
            )
        return
    flat = FlatReflectionConfig.from_mapping(payload.get("flat_reflection", {}))
    lesson = flat.load_lesson()
    if getattr(model, "hpk_runtime", None) is not None:
        raise FormalMethodConfigurationError(
            "Flat Reflection cannot run with a structured HPK runtime"
        )
    session = getattr(model, "session", None)
    agent = getattr(session, "agent", None)
    backend = getattr(agent, "control_runtime", None)
    if backend is None:
        raise FormalMethodConfigurationError(
            "Flat Reflection requires the RMBench Agent control backend"
        )
    sink = getattr(agent, "_record_trace_and_rollout_event", None)
    wrapper = FlatReflectionPlannerBackend(
        backend,
        lesson=lesson,
        audit_sink=sink if callable(sink) else None,
    )
    agent._control_runtime = wrapper
    model.control_runtime = wrapper
    model.agent_card.control_runtime = wrapper
    model.rmbench_formal_method = "flat"


__all__ = [
    "FLAT_USAGE_SCHEMA",
    "FlatReflectionConfig",
    "FlatReflectionPlannerBackend",
    "FormalMethodConfigurationError",
    "install_flat_reflection_method",
]

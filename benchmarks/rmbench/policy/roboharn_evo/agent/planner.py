"""AgenticPlanner — a tool-calling VLM planner that implements PlannerBackend.

Runs a multi-turn tool-use loop against an OpenAI-compatible VLM API.
The agent reads/writes structured memory via tools, then calls `commit`
to produce the (commit_label, memory_text, subtask_text) triple.
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable

import numpy as np
from PIL import Image

from .memory import MemoryManager
from .prompts import AGENTIC_SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE
from .tools import TOOL_SCHEMAS, ToolExecutor

logger = logging.getLogger(__name__)


def _image_to_data_url(image: np.ndarray) -> str:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _format_state_summary(state: np.ndarray) -> str:
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    return json.dumps([round(float(v), 4) for v in values], ensure_ascii=False)


@dataclass(frozen=True)
class AgenticPlannerConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "policy/roboharn_evo/checkpoints/qwen_vl/Qwen3-VL-8B-Thinking"
    timeout_sec: int = 3600
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    max_rounds: int = 6
    system_prompt: str = AGENTIC_SYSTEM_PROMPT
    user_message_template: str = USER_MESSAGE_TEMPLATE
    initial_memory_text: str = "The task has started."
    extra_tools: list[dict[str, Any]] = field(default_factory=list)
    service_tool_router: Callable[[str, str, dict[str, Any]], Any] | None = None


class AgenticPlanner:
    """PlannerBackend implementation with agentic tool-calling loop."""

    def __init__(self, config: AgenticPlannerConfig) -> None:
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_sec,
        )
        self.mm = MemoryManager(initial_text=config.initial_memory_text)
        self.tool_executor = ToolExecutor(self.mm, service_tool_router=config.service_tool_router)
        self.tools = TOOL_SCHEMAS + list(config.extra_tools)
        self.last_subtask_text: str = ""

    def reset(self) -> None:
        self.mm.reset()
        self.tool_executor.reset()
        self.last_subtask_text = ""

    def predict_planner_step(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: np.ndarray,
        planner_end_image: np.ndarray,
        planner_state: np.ndarray,
    ) -> dict[str, Any]:
        # The caller-provided previous memory is the only authoritative planner state between turns.
        self.mm.seed_from_previous_memory(previous_memory_text)
        self.tool_executor.reset()

        # ---- build messages --------------------------------------------------
        user_text = self.config.user_message_template.format(
            task=task.strip(),
            previous_memory_text=previous_memory_text.strip(),
            state_summary=_format_state_summary(planner_state),
        )
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(planner_start_image)}},
            {"type": "text", "text": "(Segment start frame)"},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(planner_end_image)}},
            {"type": "text", "text": "(Current frame)"},
        ]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # ---- agentic loop ----------------------------------------------------
        for round_idx in range(self.config.max_rounds):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            choice = response.choices[0]
            assistant_msg = choice.message

            # Append the full assistant message (may contain text + tool_calls)
            messages.append(assistant_msg.model_dump(exclude_none=True))

            # No tool calls → agent finished (maybe without commit)
            if not assistant_msg.tool_calls:
                break

            # Execute each tool call
            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    fn_args = {}

                result = self.tool_executor.execute(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

                logger.debug("round=%d tool=%s args=%s result=%s", round_idx, fn_name, fn_args, result[:200])

                if self.tool_executor.last_commit is not None:
                    break

            # If commit was called, we're done
            if self.tool_executor.last_commit is not None:
                commit = self.tool_executor.last_commit
                self.last_subtask_text = str(commit["subtask_text"])
                commit["planner_backend"] = "agentic"
                return commit

        # ---- fallback: agent exhausted rounds without commit -----------------
        logger.warning("Agentic planner exhausted %d rounds without commit, using fallback.", self.config.max_rounds)
        fallback_subtask = self.mm.peek_next_subtask() or self.last_subtask_text or "continue the current task"
        return {
            "commit_label": "no_update",
            "memory_text": self.mm.read_text(),
            "subtask_text": fallback_subtask,
            "planner_backend": "agentic",
        }

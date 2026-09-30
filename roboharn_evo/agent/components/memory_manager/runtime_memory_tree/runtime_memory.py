from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextBlock:
    block_type: str
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        return {"role": "system", "content": self.text}


@dataclass
class ChatContext:
    role: str
    content: Any
    meta: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self, hide_image: bool = False) -> dict[str, Any]:
        if self.role == "tool":
            payload = {
                "role": "tool",
                "content": self.content,
            }
            if "tool_call_id" in self.meta:
                payload["tool_call_id"] = self.meta["tool_call_id"]
            return payload
        if self.role == "user_image":
            if hide_image:
                return {"role": "user", "content": "[image hidden due to context threshold]"}
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.meta.get("caption", "Robot image context")},
                    {"type": "image_url", "image_url": {"url": self.content}},
                ],
            }
        return {"role": self.role, "content": self.content}


@dataclass
class TaskNode:
    task_brief: str
    assistant_guidance: str
    task_id: str
    task_sys_context: ContextBlock | None = None
    contexts: list[ChatContext] = field(default_factory=list)

    def _cleanup_orphaned_tool_messages(self) -> None:
        valid_tool_ids = {
            tool_call_id
            for context in self.contexts
            if context.role == "assistant" and isinstance(context.meta.get("tool_call_ids"), list)
            for tool_call_id in context.meta.get("tool_call_ids", [])
        }
        if not valid_tool_ids:
            return
        self.contexts = [
            ctx
            for ctx in self.contexts
            if ctx.role != "tool" or ctx.meta.get("tool_call_id") in valid_tool_ids
        ]

    def compress_policy_discard_oldest(self, drop_n: int) -> None:
        if drop_n <= 0:
            return
        self.contexts = self.contexts[drop_n:]


class RuntimeMemoryTree:
    def __init__(self, *, initial_memory_text: str, memory_prompts: dict[str, str], img_threshold: int = 6) -> None:
        self.initial_memory_text = initial_memory_text
        self.memory_prompts = memory_prompts
        self.img_threshold = img_threshold
        self.root_index = 0
        self.self_knowledge = ContextBlock("self_knowledge")
        self.knowledge_graph_caching_block = ContextBlock("knowledge_graph")
        self.service_registry_block = ContextBlock("service_registry")
        self.task_template_block = ContextBlock("task_template")
        self.task_session_block = ContextBlock("task_session")
        self.current_task_node: TaskNode | None = None

    @property
    def auto_index(self) -> int:
        index = self.root_index
        self.root_index += 1
        return index

    def get_full_tree(self) -> str:
        return "RuntimeMemory -> {SelfKnowledge, LongTermMemory, ShortTermMemory -> TaskNode}"

    def init_memory_tree(self, *, control_model_name: str, available_tools: list[str], mode: str, task_brief: str, assistant_guidance: str) -> None:
        self.self_knowledge = ContextBlock(
            "self_knowledge",
            text=self.memory_prompts.get("SELF_KNOWLEDGE_TEMPLATE", "").format(
                memory_tree=self.get_full_tree(),
                prefix="[SELF]",
                emoji="🧠",
                name_cn="SelfKnowledge",
                updated_at="now",
                sn_code=control_model_name,
                ormcp_version="RoboHarn-Evo",
                self_knowledge_extension=self.memory_prompts.get("SELF_KNOWLEDGE_EXTENSION", ""),
            ),
        )
        self.knowledge_graph_caching_block = ContextBlock(
            "knowledge_graph",
            text=self.memory_prompts.get("KNOWLEDGE_GRAPH_CACHING_TEMPLATE", "").format(
                l_prefix="[LONG]",
                l_emoji="📚",
                l_name_cn="LongMemory",
                prefix="[KG]",
                emoji="🗂️",
                name_cn="KnowledgeGraph",
                updated_at="now",
            ),
        )
        self.service_registry_block = ContextBlock(
            "service_registry",
            text=self.memory_prompts.get("SERVER_REGISTRY_TEMPLATE", "").format(
                prefix="[SERVICES]",
                emoji="🧰",
                name_cn="ServiceRegistry",
                updated_at="now",
                services_list="\n".join(f"- {name}" for name in available_tools),
            ),
        )
        self.task_template_block = ContextBlock(
            "task_template",
            text=self.memory_prompts.get("TASK_TEMPLATE_TEMPLATE", "").format(
                prefix="[TASK_TEMPLATE]",
                emoji="📋",
                name_cn="TaskTemplate",
                updated_at="now",
            ),
        )
        self.task_session_block = ContextBlock(
            "task_session",
            text=self.memory_prompts.get("TASK_SESSION_TEMPLATE", "").format(
                s_prefix="[SHORT]",
                s_emoji="🗃️",
                s_name_cn="ShortMemory",
                prefix="[TASK_SESSION]",
                emoji="🧾",
                name_cn="TaskSession",
                updated_at="now",
            ),
        )
        self.current_task_node = TaskNode(
            task_brief=task_brief,
            assistant_guidance=assistant_guidance,
            task_id=f"task_{self.auto_index}",
        )
        self.current_task_node.task_sys_context = ContextBlock(
            "task_node",
            text=self.memory_prompts.get("TASK_NODE_START_TEMPLATE", "").format(
                prefix="[TASK]",
                emoji="🎯",
                name_cn="TaskNode",
                updated_at="now",
                task_id=self.current_task_node.task_id,
                task_brief=task_brief,
                assistant_guidance=assistant_guidance,
            ),
        )

    def current_contexts(self) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        blocks = [
            self.self_knowledge,
            self.knowledge_graph_caching_block,
            self.service_registry_block,
            self.task_template_block,
            self.task_session_block,
        ]
        if self.current_task_node is not None and self.current_task_node.task_sys_context is not None:
            blocks.append(self.current_task_node.task_sys_context)
        for block in blocks:
            if block.text:
                contexts.append(block.to_openai_format())
        if self.current_task_node is None:
            return contexts
        self.current_task_node._cleanup_orphaned_tool_messages()
        img_count = 0
        for chat_ctx in self.current_task_node.contexts:
            hide_image = False
            if chat_ctx.role == "user_image":
                img_count += 1
                hide_image = img_count > self.img_threshold
            contexts.append(chat_ctx.to_openai_format(hide_image=hide_image))
        return contexts

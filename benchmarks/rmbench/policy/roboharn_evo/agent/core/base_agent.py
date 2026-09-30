from __future__ import annotations

from ..components.agent_tools import AgentTools, LocalSkillRegistry
from ..components.memory_manager import MemoryManager
from ..components.service_manager import InternalServiceManager
from .base_agent_card import BaseAgentCard
from ..paths import rmbench_root, roboharn_skills_dir


class BaseAgent:
    def __init__(self, agent_card: BaseAgentCard) -> None:
        self._agent_card = agent_card
        self._control_runtime = agent_card.control_runtime
        self._executor_runtime = agent_card.executor_runtime
        self._control_model_name = agent_card.control_model_name
        self._executor_name = agent_card.executor_name
        self._initialized = False
        self._memory_manager = MemoryManager(initial_memory_text=agent_card.config.initial_memory_text)
        self._service_manager = InternalServiceManager()
        self._skill_registry = LocalSkillRegistry(
            configured_paths=[str(roboharn_skills_dir())],
            workspace_root=str(rmbench_root()),
        )
        self._agent_tools = AgentTools(
            memory_manager=self._memory_manager,
            agent_card=agent_card,
            agent_instance=self,
        )

    @property
    def memory_store(self):
        return self._memory_manager

    @property
    def control_runtime(self):
        return self._control_runtime

    @property
    def executor_runtime(self):
        return self._executor_runtime

    @property
    def skill_registry(self):
        return self._skill_registry

    @property
    def service_manager(self):
        return self._service_manager

    @property
    def agent_tools(self):
        return self._agent_tools

    @property
    def config(self):
        return self._agent_card.config

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def init_agent(self) -> None:
        await self._service_manager.init_services()
        await self._agent_tools.init_agent_tools()
        for service in self._agent_tools.service_registers:
            await self._service_manager.registry_service(service)
        self._initialized = True

    async def shutdown(self) -> None:
        await self._service_manager.shutdown()

    async def terminate(self) -> None:
        await self._service_manager.terminate()

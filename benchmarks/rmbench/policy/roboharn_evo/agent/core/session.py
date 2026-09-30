from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base_agent_card import BaseAgentCard
from .img_agent import ImgAgent


@dataclass
class AgentSession:
    agent_card: BaseAgentCard
    agent: ImgAgent
    running: bool = False

    @classmethod
    def build(cls, *, agent_card: BaseAgentCard) -> "AgentSession":
        return cls(agent_card=agent_card, agent=ImgAgent(agent_card=agent_card), running=False)

    async def init_session(self) -> None:
        await self.agent.init_agent()
        self.running = True

    async def shutdown(self) -> None:
        await self.agent.shutdown()
        self.running = False

    def run_eval_once(self, task_env: Any, observation: dict[str, Any]) -> None:
        self.agent.run_step(task_env, observation)

    def reset(self) -> None:
        self.agent.reset()

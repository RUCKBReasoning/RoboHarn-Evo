from __future__ import annotations

import asyncio
from typing import Any

from .service_types import ServiceRegister


class InternalServiceManager:
    def __init__(self) -> None:
        self._services_register_list: list[ServiceRegister] = []

    @property
    def activate_tools_list(self) -> list[dict[str, Any]]:
        return [
            tool.openai_format
            for service in self._services_register_list
            if service.is_activation
            for tool in service.tools_list
        ]

    async def init_services(self) -> None:
        return

    async def registry_service(self, service: ServiceRegister) -> None:
        self._services_register_list.append(service)

    def check_is_agent_service(self, service_name: str) -> bool:
        for service in self._services_register_list:
            if service.service_name == service_name:
                return service.is_agent_service
        raise RuntimeError(f"could not find service: {service_name}")

    async def shutdown(self) -> None:
        return

    async def terminate(self) -> None:
        return

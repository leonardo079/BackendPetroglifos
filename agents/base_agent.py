"""Contrato común para todos los agentes del sistema."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any

class AgentInput(BaseModel):
    task_id: str
    payload: dict[str, Any]
    context: dict[str, Any] = {}

class AgentOutput(BaseModel):
    task_id: str
    agent_name: str
    result: Any
    status: str = "success"   # success | error | fallback
    metadata: dict[str, Any] = {}

class BaseAgent(ABC):
    name: str = "base"

    def __init__(self, tools: list = [], memory=None) -> None:
        self.tools = tools
        self.memory = memory

    @abstractmethod
    async def run(self, input: AgentInput) -> AgentOutput: ...

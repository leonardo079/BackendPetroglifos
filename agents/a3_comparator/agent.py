"""A3 — Comparador iconográfico (EfficientNet-B0 + pgvector)."""
from agents.base_agent import BaseAgent, AgentInput, AgentOutput

class ComparatorAgent(BaseAgent):
    name = "a3_comparator"

    async def run(self, input: AgentInput) -> AgentOutput:
        # Similitud coseno contra corpus de referencia
        raise NotImplementedError

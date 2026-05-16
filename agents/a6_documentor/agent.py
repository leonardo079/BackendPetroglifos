"""A6 — Documentador (Jinja2 + WeasyPrint → Ficha ICANH)."""
from agents.base_agent import BaseAgent, AgentInput, AgentOutput

class DocumentorAgent(BaseAgent):
    name = "a6_documentor"

    async def run(self, input: AgentInput) -> AgentOutput:
        # Renderiza ficha ICANH en PDF + JSON y persiste en R2/MinIO
        raise NotImplementedError

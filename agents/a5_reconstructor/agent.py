"""A5 — Reconstructor GAN (activo solo cuando A2 detecta deterioro)."""
from agents.base_agent import BaseAgent, AgentInput, AgentOutput

class ReconstructorAgent(BaseAgent):
    name = "a5_reconstructor"

    async def run(self, input: AgentInput) -> AgentOutput:
        # Reconstrucción mediante GAN para petroglifos deteriorados
        raise NotImplementedError

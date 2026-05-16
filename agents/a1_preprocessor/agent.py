"""A1 — Preprocesador de imagen (OpenCV)."""
from agents.base_agent import BaseAgent, AgentInput, AgentOutput

class PreprocessorAgent(BaseAgent):
    name = "a1_preprocessor"

    async def run(self, input: AgentInput) -> AgentOutput:
        # Normalización, eliminación de ruido, realce de contraste
        raise NotImplementedError

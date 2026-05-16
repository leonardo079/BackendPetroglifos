"""A4 — Analista Cultural (RAG + Gemini): núcleo del módulo LLM."""
from agents.base_agent import BaseAgent, AgentInput, AgentOutput

class CulturalAnalystAgent(BaseAgent):
    name = "a4_analyst"

    async def run(self, input: AgentInput) -> AgentOutput:
        # 1. Embedding de consulta
        # 2. Búsqueda vectorial top-k=5
        # 3. Construcción dinámica del prompt
        # 4. Inferencia Gemini 1.5 Flash
        # 5. Postprocesamiento y validación
        # 6. Persistencia en llm_classifications + prompt_logs
        raise NotImplementedError

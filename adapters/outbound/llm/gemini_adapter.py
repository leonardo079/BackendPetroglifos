"""Adaptador concreto para Gemini 1.5 Flash."""
from core.ports.outbound.llm_port import LLMPort

class GeminiAdapter(LLMPort):
    async def generate(self, prompt: str, system: str = "") -> str:
        # google.generativeai / retries con backoff exponencial
        raise NotImplementedError

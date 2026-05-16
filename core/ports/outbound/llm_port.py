"""Puerto de salida hacia cualquier LLM (Gemini, OpenAI, etc.)."""
from abc import ABC, abstractmethod

class LLMPort(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> str: ...

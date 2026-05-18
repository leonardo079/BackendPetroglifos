"""Puerto de salida hacia cualquier LLM (Gemini, OpenAI, etc.)."""
from abc import ABC, abstractmethod

class LLMPort(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> str: ...

    @abstractmethod
    async def generate_json(self, prompt: str, system: str = "") -> dict: ...

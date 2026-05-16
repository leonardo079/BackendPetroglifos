"""Puerto de salida para generación de embeddings."""
from abc import ABC, abstractmethod

class EmbeddingPort(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

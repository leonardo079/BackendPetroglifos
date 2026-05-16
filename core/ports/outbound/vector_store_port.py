"""Puerto de salida hacia la base vectorial (pgvector, Pinecone, etc.)."""
from abc import ABC, abstractmethod

class VectorStorePort(ABC):
    @abstractmethod
    async def similarity_search(self, query_vector: list[float], k: int = 5) -> list[dict]: ...
    @abstractmethod
    async def upsert(self, documents: list[dict]) -> None: ...

"""Value object: resultado de la clasificación taxonómica."""
from __future__ import annotations
from pydantic import BaseModel, Field

class ClassificationResult(BaseModel):
    taxonomy: str
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    retrieved_context: list[dict] = []
    requires_validation: bool = True
    low_context_quality: bool = False
    status: str = "success"

"""Value objects del dominio."""
from __future__ import annotations
from pydantic import BaseModel, Field
from uuid import UUID


class ClassificationResult(BaseModel):
    taxonomy: str
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    retrieved_context: list[dict] = []
    requires_validation: bool = True
    low_context_quality: bool = False
    status: str = "success"

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.70 and not self.low_context_quality


class SimilarityMatch(BaseModel):
    site_id: str
    site_name: str
    reference_name: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    taxonomy: str = ""
    municipality: str = ""
    image_path: str = ""


class DetectionResult(BaseModel):
    motif_description: str
    detected_shapes: list[str] = []
    bounding_boxes: list[dict] = []
    detection_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    motifs_visible: bool = True
    deterioration_detected: bool = False


class ICANHRecord(BaseModel):
    petroglyph_id: str
    site: str
    municipality: str
    department: str
    gps_coordinates: dict = {}
    taxonomy: str
    confidence: float
    justification: str
    detected_shapes: list[str] = []
    similarity_matches: list[dict] = []
    conservation_status: str = "Regular"
    researcher_notes: str = ""
    pdf_url: str = ""
    json_path: str = ""
    generation_time_seconds: float = 0.0
    requires_expert_validation: bool = True
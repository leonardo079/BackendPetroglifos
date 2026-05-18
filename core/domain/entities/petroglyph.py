"""Entidad central del dominio: representa un petroglifo y su clasificación."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class GeoLocation:
    latitude: float
    longitude: float
    altitude_m: float | None = None
    accuracy_m: float | None = None


@dataclass
class Petroglyph:
    id: UUID = field(default_factory=uuid4)
    site: str = ""
    municipality: str = ""
    department: str = ""
    location: GeoLocation | None = None
    # Imagen
    raw_image_path: str = ""
    preprocessed_image_path: str = ""
    reconstructed_image_path: str = ""
    # Detección y clasificación
    motif_description: str = ""
    detected_shapes: list[str] = field(default_factory=list)
    taxonomy: str = ""
    confidence: float = 0.0
    justification: str = ""
    requires_validation: bool = True
    # Similitudes iconográficas (para el grafo social)
    similarity_matches: list[dict] = field(default_factory=list)
    # Metadata
    researcher_notes: str = ""
    conservation_status: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RupestranSite:
    """Nodo del grafo social: sitio arqueológico."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    municipality: str = ""
    department: str = ""
    location: GeoLocation | None = None
    conservation_status: str = ""
    petroglyph_count: int = 0
    dominant_taxonomy: str = ""
    # Relaciones iconográficas (aristas del grafo)
    similar_sites: list[dict] = field(default_factory=list)
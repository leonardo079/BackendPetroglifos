"""Entidad central del dominio: representa un petroglifo y su clasificación."""
from __future__ import annotations
from dataclasses import dataclass, field
from uuid import UUID, uuid4

@dataclass
class Petroglyph:
    id: UUID = field(default_factory=uuid4)
    site: str = ""
    municipality: str = ""
    department: str = ""
    motif_description: str = ""
    detected_shapes: list[str] = field(default_factory=list)
    taxonomy: str = ""
    confidence: float = 0.0
    justification: str = ""
    requires_validation: bool = True

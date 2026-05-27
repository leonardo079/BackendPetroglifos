"""Estado compartido que fluye entre nodos del grafo LangGraph."""
from __future__ import annotations
from typing import TypedDict, Any

class PetroglyphState(TypedDict, total=False):
    petroglyph_id: str
    raw_image_path: str
    site_metadata: dict[str, Any]
    # A1
    preprocessed_image_path: str
    # A2
    motif_description: str
    detected_shapes: list[str]
    bounding_box: dict
    detection_confidence: float
    motifs_visible: bool
    _deterioration_detected: bool   # usado por el router para decidir A3 vs A5
    # A3
    similarity_matches: list[dict]
    # A5
    reconstructed_image_path: str
    # A4
    a4_taxonomy_result: dict
    a4_requires_validation: bool
    a4_petroglyph_description: dict
    a4_rag_feedback: dict
    # A6
    icanh_pdf_url: str
    icanh_json: dict

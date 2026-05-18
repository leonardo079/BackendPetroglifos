"""
Re-exporta PetroglyphSocialGraph desde su ubicación canónica.

El grafo social vive en orchestrator/graph/petroglyph_graph.py.
Este módulo existe para que los agentes (a3_comparator) puedan importar
de `graphs.social_graph` sin romper la arquitectura hexagonal.
"""
from orchestrator.graph.petroglyph_graph import PetroglyphSocialGraph  # noqa: F401

__all__ = ["PetroglyphSocialGraph"]
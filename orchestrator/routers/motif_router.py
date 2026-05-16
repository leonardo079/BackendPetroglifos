"""Enrutador condicional: ¿motivos visibles? → A3  |  deterioro → A5."""
from orchestrator.state.graph_state import PetroglyphState

def route_after_detection(state: PetroglyphState) -> str:
    if state.get("motifs_visible", False):
        return "a3_comparator"
    return "a5_reconstructor"

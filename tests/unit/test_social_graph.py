from __future__ import annotations

from core.domain.site_normalization import normalize_site_metadata
from orchestrator.graph.petroglyph_graph import PetroglyphSocialGraph


def test_graph_accumulates_evidence_and_marks_reliability() -> None:
    graph = PetroglyphSocialGraph()
    graph.add_site("A", municipality="M1", department="D1")
    graph.add_site("B", municipality="M2", department="D2")
    graph.add_site("C", municipality="M3", department="D3")
    graph.add_site("D", municipality="M4", department="D4")

    graph.add_or_update_edge("A", "B", weight=0.80, taxonomy="Geométrico")
    graph.add_or_update_edge("A", "B", weight=0.90, taxonomy="Zoomorfo")
    graph.add_or_update_edge("B", "C", weight=0.88, taxonomy="Lineal")
    graph.add_or_update_edge("B", "C", weight=0.92, taxonomy="Curvilíneo")
    graph.add_or_update_edge("A", "D", weight=0.40, taxonomy="Indeterminado")

    edge_ab = graph._G["A"]["B"]
    assert edge_ab["weight"] == 0.85
    assert edge_ab["evidence_count"] == 2
    assert edge_ab["is_provisional"] is False
    assert edge_ab["shared_taxonomies"] == ["Geométrico", "Zoomorfo"]

    edge_ad = graph._G["A"]["D"]
    assert edge_ad["is_provisional"] is True

    similar = graph.most_similar_sites("A", top_k=5)
    assert similar[0]["site"] == "B"
    assert similar[0]["confidence_level"] == "medium"
    assert similar[1]["site"] == "D"
    assert similar[1]["confidence_level"] == "low"

    pr = graph.pagerank()
    assert set(pr) == {"A", "B", "C"}
    assert pr["B"] == max(pr.values())

    betweenness = graph.betweenness_centrality()
    assert betweenness["B"] == max(betweenness.values())

    summary = graph.summary()
    assert summary["nodes"] == 4
    assert summary["edges"] == 3

    metrics = graph.metrics()
    assert metrics["nodes"] == 4
    assert metrics["edges"] == 3
    top_sites = {item["site"] for item in metrics["degree_distribution"]["top_hubs"][:2]}
    assert top_sites == {"A", "B"}

    payload = graph.to_dict()
    edge_payload = next(edge for edge in payload["edges"] if edge["source"] == "A" and edge["target"] == "B")
    assert edge_payload["confidence_level"] == "medium"


def test_site_normalization_uses_workshop_canonical_names() -> None:
    site, municipality, department = normalize_site_metadata("gameza", "gameza", "boyaca")

    assert site == "Gámeza"
    assert municipality == "Gámeza"
    assert department == "Boyacá"


def test_graph_canonicalizes_site_variants() -> None:
    graph = PetroglyphSocialGraph()
    graph.add_site("gameza", municipality="gameza", department="boyaca")
    graph.add_or_update_edge("gameza", "sachica", weight=0.81, taxonomy="Geométrico")
    graph.add_or_update_edge("Gámeza", "Sáchica", weight=0.87, taxonomy="Zoomorfo")

    assert set(graph._G.nodes) == {"Gámeza", "Sáchica"}
    edge = graph._G["Gámeza"]["Sáchica"]
    assert edge["weight"] == 0.84
    assert edge["evidence_count"] == 2
    assert edge["shared_taxonomies"] == ["Geométrico", "Zoomorfo"]

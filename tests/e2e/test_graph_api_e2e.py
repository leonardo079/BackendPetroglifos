from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from orchestrator.graph import petroglyph_graph as graph_module
from orchestrator.graph.petroglyph_graph import PetroglyphSocialGraph


def _build_sample_graph() -> PetroglyphSocialGraph:
    graph = PetroglyphSocialGraph()
    graph.add_site("A", municipality="M1", department="D1")
    graph.add_site("B", municipality="M2", department="D2")
    graph.add_site("C", municipality="M3", department="D3")

    graph.add_or_update_edge("A", "B", weight=0.80, taxonomy="Geométrico")
    graph.add_or_update_edge("A", "B", weight=0.90, taxonomy="Zoomorfo")
    graph.add_or_update_edge("B", "C", weight=0.88, taxonomy="Lineal")
    graph.add_or_update_edge("B", "C", weight=0.92, taxonomy="Curvilíneo")
    return graph


def test_graph_endpoints_return_expected_payloads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://petro:petro@localhost:5432/petroglifos",
    )

    import config.settings as settings_module

    settings_module.get_settings.cache_clear()
    settings_module.settings = settings_module.get_settings()

    from adapters.inbound.api import main

    graph = _build_sample_graph()

    async def fake_build_graph_from_db(_session):
        return graph

    async def fake_get_session():
        yield object()

    monkeypatch.setattr(main, "_build_graph_from_db", fake_build_graph_from_db)
    monkeypatch.setitem(main.app.dependency_overrides, main.get_session, fake_get_session)
    monkeypatch.setattr(graph_module, "GRAPH_OUTPUT_DIR", tmp_path)

    try:
        with TestClient(main.app) as client:
            response = client.get("/graph")
            assert response.status_code == 200
            payload = response.json()
            assert payload["summary"]["nodes"] == 3
            assert payload["summary"]["edges"] == 2
            assert any(edge["confidence_level"] == "medium" for edge in payload["edges"])

            pagerank = client.get("/graph/pagerank")
            assert pagerank.status_code == 200
            assert pagerank.json()["top_site"] == "B"

            communities = client.get("/graph/communities")
            assert communities.status_code == 200
            assert communities.json()["count"] == 1

            betweenness = client.get("/graph/betweenness")
            assert betweenness.status_code == 200
            assert betweenness.json()["top_bridge_site"] == "B"

            image = client.get("/graph/export/image")
            assert image.status_code == 200
            assert image.headers["content-type"] == "image/png"

            export = client.get("/graph/export")
            assert export.status_code == 200
            exported_files = list(Path(tmp_path).glob("red_rupestre.html"))
            assert exported_files
            assert "text/html" in export.headers["content-type"]
    finally:
        main.app.dependency_overrides.clear()

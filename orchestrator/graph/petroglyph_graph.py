"""
Grafo social de similitud iconográfica entre sitios rupestres.

Nodos  = sitios arqueológicos (Villa de Leyva, Gameza, Facatativá…)
Aristas = similitud coseno entre motivos detectados (peso 0–1)

Análisis disponibles:
- Comunidades (Louvain)
- PageRank (sitios más "centrales" en la red rupestre)
- Centralidad de intermediación
- Exportación HTML interactiva con PyVis
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import structlog
import networkx as nx

log = structlog.get_logger(__name__)

GRAPH_OUTPUT_DIR = Path("storage/graphs")
GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PetroglyphSocialGraph:
    """
    Grafo ponderado no dirigido de similitud iconográfica entre sitios rupestres.

    Uso típico (desde A3):
        graph = PetroglyphSocialGraph()
        graph.add_site("Villa de Leyva", municipality="Villa de Leyva", department="Boyacá")
        graph.add_or_update_edge("Villa de Leyva", "Gámeza", weight=0.83, taxonomy="Geométrico")
        graph.export_html("storage/graphs/red_rupestre.html")
    """

    def __init__(self) -> None:
        self._G: nx.Graph = nx.Graph()

    # ── Construcción del grafo ────────────────────────────────────────────────

    def add_site(
        self,
        site_id: str,
        *,
        municipality: str = "",
        department: str = "",
        dominant_taxonomy: str = "Indeterminado",
        petroglyph_count: int = 0,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> None:
        """Agrega o actualiza un nodo (sitio rupestre)."""
        self._G.add_node(
            site_id,
            municipality=municipality,
            department=department,
            dominant_taxonomy=dominant_taxonomy,
            petroglyph_count=petroglyph_count,
            latitude=latitude,
            longitude=longitude,
        )

    def add_or_update_edge(
        self,
        site_a: str,
        site_b: str,
        weight: float,
        taxonomy: str = "",
    ) -> None:
        """
        Agrega o actualiza una arista entre dos sitios.
        Si ya existe, promedia el peso y acumula evidencia.
        """
        if site_a == site_b:
            return
        # Asegurar que los nodos existen
        for s in (site_a, site_b):
            if s not in self._G:
                self.add_site(s)

        if self._G.has_edge(site_a, site_b):
            data = self._G[site_a][site_b]
            # Promedio acumulativo del peso
            n = data.get("evidence_count", 1)
            new_weight = (data["weight"] * n + weight) / (n + 1)
            data["weight"] = round(new_weight, 4)
            data["evidence_count"] = n + 1
            if taxonomy and taxonomy not in data.get("shared_taxonomies", []):
                data.setdefault("shared_taxonomies", []).append(taxonomy)
        else:
            self._G.add_edge(
                site_a,
                site_b,
                weight=round(weight, 4),
                evidence_count=1,
                shared_taxonomies=[taxonomy] if taxonomy else [],
            )
        log.debug("graph_edge_updated", site_a=site_a, site_b=site_b, weight=weight)

    # ── Análisis ──────────────────────────────────────────────────────────────

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Calcula PageRank — sitios más centrales en la red iconográfica."""
        if len(self._G) == 0:
            return {}
        return nx.pagerank(self._G, alpha=alpha, weight="weight")

    def communities(self) -> list[set[str]]:
        """Detección de comunidades con el algoritmo de Louvain (greedy modularity)."""
        if len(self._G) == 0:
            return []
        try:
            from community import best_partition  # type: ignore
            partition = best_partition(self._G, weight="weight")
            groups: dict[int, set[str]] = {}
            for node, comm_id in partition.items():
                groups.setdefault(comm_id, set()).add(node)
            return list(groups.values())
        except ImportError:
            # Fallback: greedy modularity de networkx
            comms = nx.algorithms.community.greedy_modularity_communities(self._G, weight="weight")
            return [set(c) for c in comms]

    def betweenness_centrality(self) -> dict[str, float]:
        """Centralidad de intermediación — sitios "puente" entre regiones."""
        if len(self._G) == 0:
            return {}
        return nx.betweenness_centrality(self._G, weight="weight", normalized=True)

    def most_similar_sites(self, site_id: str, top_k: int = 5) -> list[dict]:
        """Retorna los sitios más similares a un sitio dado, ordenados por peso."""
        if site_id not in self._G:
            return []
        neighbors = [
            {
                "site": nb,
                "weight": data["weight"],
                "evidence_count": data.get("evidence_count", 1),
                "shared_taxonomies": data.get("shared_taxonomies", []),
            }
            for nb, data in self._G[site_id].items()
        ]
        return sorted(neighbors, key=lambda x: x["weight"], reverse=True)[:top_k]

    def summary(self) -> dict:
        """Resumen estadístico del grafo."""
        if len(self._G) == 0:
            return {"nodes": 0, "edges": 0}
        pr = self.pagerank()
        top_site = max(pr, key=pr.get) if pr else ""
        weights = [d["weight"] for _, _, d in self._G.edges(data=True)]
        return {
            "nodes": self._G.number_of_nodes(),
            "edges": self._G.number_of_edges(),
            "avg_similarity": round(sum(weights) / len(weights), 4) if weights else 0.0,
            "max_similarity": round(max(weights), 4) if weights else 0.0,
            "most_central_site": top_site,
            "communities": len(self.communities()),
            "density": round(nx.density(self._G), 4),
        }

    # ── Serialización ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serializa el grafo a dict (para guardar en BD o API)."""
        return {
            "nodes": [
                {"id": n, **self._G.nodes[n]}
                for n in self._G.nodes
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self._G.edges(data=True)
            ],
            "summary": self.summary(),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def save_json(self, path: str | None = None) -> str:
        """Guarda el grafo como JSON."""
        out = Path(path) if path else GRAPH_OUTPUT_DIR / "social_graph.json"
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("graph_saved_json", path=str(out))
        return str(out)

    def export_html(self, path: str | None = None, height: str = "750px") -> str:
        """
        Exporta visualización interactiva con PyVis.
        El HTML resultante se sirve desde la plataforma web.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            log.error("pyvis_not_installed")
            return ""

        out = Path(path) if path else GRAPH_OUTPUT_DIR / "red_rupestre.html"
        net = Network(height=height, width="100%", bgcolor="#1a1a2e", font_color="white",
                      notebook=False, directed=False)
        net.set_options("""
        {
          "physics": {"solver": "forceAtlas2Based", "stabilization": {"iterations": 150}},
          "edges": {"smooth": {"type": "continuous"}, "color": {"inherit": "both"}},
          "nodes": {"shape": "dot", "scaling": {"min": 10, "max": 40}},
          "interaction": {"hover": true, "tooltipDelay": 200}
        }
        """)

        # Calcular métricas para tamaño y color de nodos
        pr = self.pagerank()
        communities_list = self.communities()
        node_community: dict[str, int] = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                node_community[node] = i

        COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                  "#1abc9c", "#e67e22", "#34495e"]

        for node in self._G.nodes:
            attrs = self._G.nodes[node]
            size = 15 + int(pr.get(node, 0) * 500)
            comm_id = node_community.get(node, 0)
            color = COLORS[comm_id % len(COLORS)]
            taxonomy = attrs.get("dominant_taxonomy", "Indeterminado")
            municipality = attrs.get("municipality", "")
            count = attrs.get("petroglyph_count", 0)
            title = (
                f"<b>{node}</b><br>"
                f"Municipio: {municipality}<br>"
                f"Taxonomía dominante: {taxonomy}<br>"
                f"Petroglifos: {count}<br>"
                f"PageRank: {pr.get(node, 0):.4f}"
            )
            net.add_node(node, label=node, size=size, color=color, title=title)

        for u, v, data in self._G.edges(data=True):
            weight = data.get("weight", 0.5)
            evidence = data.get("evidence_count", 1)
            taxonomies = ", ".join(data.get("shared_taxonomies", []))
            title = (
                f"Similitud: {weight:.2%}<br>"
                f"Evidencias: {evidence}<br>"
                f"Taxonomías compartidas: {taxonomies or 'N/A'}"
            )
            net.add_edge(u, v, value=weight, title=title, width=weight * 5)

        net.save_graph(str(out))
        log.info("graph_exported_html", path=str(out), nodes=len(self._G.nodes))
        return str(out)

    # ── Persistencia en PostgreSQL ────────────────────────────────────────────

    async def sync_to_db(self, session) -> None:
        """
        Sincroniza las aristas del grafo a la tabla site_graph_edges.
        Requiere que los sitios ya existan en rupestrian_sites.
        """
        from infrastructure.database.models.models import SiteGraphEdge, RupestranSiteModel
        from sqlalchemy import select

        # Obtener IDs reales de los sitios por nombre
        result = await session.execute(select(RupestranSiteModel))
        sites_by_name = {s.name: s.id for s in result.scalars()}

        for u, v, data in self._G.edges(data=True):
            id_a = sites_by_name.get(u)
            id_b = sites_by_name.get(v)
            if not id_a or not id_b:
                continue
            # Upsert manual (ON CONFLICT UPDATE)
            existing = await session.execute(
                select(SiteGraphEdge).where(
                    SiteGraphEdge.site_a_id == id_a,
                    SiteGraphEdge.site_b_id == id_b,
                )
            )
            edge = existing.scalar_one_or_none()
            if edge:
                edge.weight = data["weight"]
                edge.evidence_count = data.get("evidence_count", 1)
                edge.shared_taxonomies = data.get("shared_taxonomies", [])
            else:
                session.add(SiteGraphEdge(
                    site_a_id=id_a,
                    site_b_id=id_b,
                    weight=data["weight"],
                    evidence_count=data.get("evidence_count", 1),
                    shared_taxonomies=data.get("shared_taxonomies", []),
                ))
        await session.commit()
        log.info("graph_synced_to_db", edges=self._G.number_of_edges())
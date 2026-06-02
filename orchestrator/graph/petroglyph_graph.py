"""
Grafo social de similitud iconográfica entre sitios rupestres.

Nodos  = sitios arqueológicos
Aristas = similitud coseno entre motivos detectados (peso 0-1)

Análisis disponibles:
- Comunidades (Louvain)
- PageRank (sitios más "centrales" en la red rupestre)
- Centralidad de intermediación
- Métricas de topología
- Exportación HTML interactiva con PyVis
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import networkx as nx
import structlog

from config.settings import settings
from core.domain.site_normalization import canonicalize_site_name, canonicalize_municipality

log = structlog.get_logger(__name__)

GRAPH_OUTPUT_DIR = Path("storage/graphs")
GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _compute_confidence_level(weight: float, evidence_count: int) -> str:
    """Clasifica una arista en low/medium/high según similitud y evidencia acumulada."""
    if weight >= 0.85 and evidence_count >= 3:
        return "high"
    if (
        weight >= settings.edge_reliable_min_similarity
        and evidence_count >= settings.edge_min_evidence
    ):
        return "medium"
    return "low"


class PetroglyphSocialGraph:
    """
    Grafo ponderado no dirigido de similitud iconográfica entre sitios rupestres.

    Uso típico:
        graph = PetroglyphSocialGraph()
        graph.add_site("Villa de Leyva", municipality="Villa de Leyva", department="Boyacá")
        graph.add_or_update_edge("Villa de Leyva", "Gámeza", weight=0.83, taxonomy="Geométrico")
        graph.export_html("storage/graphs/red_rupestre.html")
    """

    def __init__(self) -> None:
        self._G: nx.Graph = nx.Graph()

    # ── Construcción del grafo ──────────────────────────────────────────────

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
        site_id = canonicalize_site_name(site_id)
        municipality = canonicalize_municipality(municipality)
        self._G.add_node(
            site_id,
            municipality=municipality,
            department=department,
            dominant_taxonomy=dominant_taxonomy,
            petroglyph_count=petroglyph_count,
            latitude=latitude,
            longitude=longitude,
        )

    def _is_provisional(self, weight: float, evidence_count: int) -> bool:
        return not (
            weight >= settings.edge_reliable_min_similarity
            and evidence_count >= settings.edge_min_evidence
        )

    def add_or_update_edge(
        self,
        site_a: str,
        site_b: str,
        weight: float,
        taxonomy: str = "",
    ) -> None:
        """Agrega o actualiza una arista. Si existe, promedia el peso y acumula evidencia."""
        site_a = canonicalize_site_name(site_a)
        site_b = canonicalize_site_name(site_b)
        if site_a == site_b:
            return

        for s in (site_a, site_b):
            if s not in self._G:
                self.add_site(s)

        if self._G.has_edge(site_a, site_b):
            data = self._G[site_a][site_b]
            n = data.get("evidence_count", 1)
            data["weight"] = round((data["weight"] * n + weight) / (n + 1), 4)
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
            data = self._G[site_a][site_b]

        data["is_provisional"] = self._is_provisional(
            float(data.get("weight", 0.0)),
            int(data.get("evidence_count", 1)),
        )
        log.debug("graph_edge_updated", site_a=site_a, site_b=site_b, weight=weight)

    def load_persisted_edge(
        self,
        site_a: str,
        site_b: str,
        *,
        weight: float,
        evidence_count: int,
        shared_taxonomies: list[str] | None = None,
        is_provisional: bool | None = None,
    ) -> None:
        """
        Carga una arista persistida sin tratarla como una observación nueva.

        Se usa al reconstruir el grafo desde la BD para conservar weight,
        evidence_count e is_provisional tal como fueron guardados.
        """
        site_a = canonicalize_site_name(site_a)
        site_b = canonicalize_site_name(site_b)
        if site_a == site_b:
            return

        for s in (site_a, site_b):
            if s not in self._G:
                self.add_site(s)

        if is_provisional is None:
            is_provisional = self._is_provisional(weight, evidence_count)

        self._G.add_edge(
            site_a,
            site_b,
            weight=round(weight, 4),
            evidence_count=evidence_count,
            shared_taxonomies=list(shared_taxonomies or []),
            is_provisional=is_provisional,
        )

    # ── Análisis ────────────────────────────────────────────────────────────

    def _reliable_subgraph(self) -> nx.Graph:
        """Subgrafo con solo aristas confiables."""
        reliable = [
            (u, v)
            for u, v, d in self._G.edges(data=True)
            if not d.get("is_provisional", True)
        ]
        return self._G.edge_subgraph(reliable).copy()

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """PageRank usando solo aristas confiables.

        NetworkX 3.x delega en SciPy para este cálculo en muchos entornos.
        Para evitar una dependencia pesada en el entorno del proyecto, usamos
        una implementación por iteración de potencia basada en pesos de arista.
        """
        G = self._reliable_subgraph()
        if len(G) == 0:
            return {}

        nodes = list(G.nodes())
        n = len(nodes)
        ranks = {node: 1.0 / n for node in nodes}
        out_weight = {
            node: sum(float(data.get("weight", 1.0)) for _, _, data in G.edges(node, data=True))
            for node in nodes
        }

        for _ in range(100):
            new_ranks = {node: (1.0 - alpha) / n for node in nodes}
            dangling_mass = sum(ranks[node] for node in nodes if out_weight[node] == 0.0)

            for src in nodes:
                if out_weight[src] == 0.0:
                    continue
                share = ranks[src] / out_weight[src]
                for dst, data in G[src].items():
                    new_ranks[dst] += alpha * share * float(data.get("weight", 1.0))

            if dangling_mass:
                dangling_share = alpha * dangling_mass / n
                for node in nodes:
                    new_ranks[node] += dangling_share

            delta = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
            ranks = new_ranks
            if delta < n * 1e-6:
                break

        total = sum(ranks.values()) or 1.0
        return {node: rank / total for node, rank in ranks.items()}

    def communities(self) -> list[set[str]]:
        """Comunidades Louvain sobre aristas confiables, con fallback greedy."""
        G = self._reliable_subgraph()
        if len(G) == 0:
            return []
        try:
            from community import best_partition  # type: ignore

            partition = best_partition(G, weight="weight", random_state=42)
            groups: dict[int, set[str]] = {}
            for node, comm_id in partition.items():
                groups.setdefault(comm_id, set()).add(node)
            return list(groups.values())
        except ImportError:
            comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
            return [set(c) for c in comms]

    def betweenness_centrality(self) -> dict[str, float]:
        """Centralidad de intermediación sobre aristas confiables."""
        G = self._reliable_subgraph()
        if len(G) == 0:
            return {}

        distance_graph = nx.Graph()
        distance_graph.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            similarity = float(data.get("weight", 0.0))
            distance = max(1e-6, 1.0 - similarity)
            distance_graph.add_edge(u, v, weight=distance)

        return nx.betweenness_centrality(distance_graph, weight="weight", normalized=True)

    def most_similar_sites(self, site_id: str, top_k: int = 5) -> list[dict]:
        """Top-k sitios más similares a uno dado, ordenados por peso de arista."""
        if site_id not in self._G:
            return []

        neighbors = [
            {
                "site": nb,
                "weight": data["weight"],
                "evidence_count": data.get("evidence_count", 1),
                "shared_taxonomies": data.get("shared_taxonomies", []),
                "is_provisional": data.get("is_provisional", True),
                "confidence_level": _compute_confidence_level(
                    float(data["weight"]), int(data.get("evidence_count", 1))
                ),
            }
            for nb, data in self._G[site_id].items()
        ]
        return sorted(neighbors, key=lambda x: x["weight"], reverse=True)[:top_k]

    def metrics(self) -> dict:
        """
        Métricas de topología del grafo.

        Incluye clustering coefficient, componentes conectados,
        distribución de grado (top hubs) y diámetro de la red.
        """
        n = self._G.number_of_nodes()
        e = self._G.number_of_edges()
        if n == 0:
            return {"nodes": 0, "edges": 0}

        clustering = round(nx.average_clustering(self._G, weight="weight"), 4)
        components = list(nx.connected_components(self._G))
        components_sorted = sorted(components, key=len, reverse=True)
        num_components = len(components)
        largest_size = len(components_sorted[0]) if components_sorted else 0

        diameter = None
        if largest_size > 1:
            largest_subgraph = self._G.subgraph(components_sorted[0])
            try:
                diameter = nx.diameter(largest_subgraph)
            except nx.NetworkXError:
                diameter = None

        degrees = dict(self._G.degree())
        avg_degree = round(sum(degrees.values()) / n, 2) if n > 0 else 0.0
        top_hubs = sorted(
            [{"site": s, "degree": d} for s, d in degrees.items()],
            key=lambda x: x["degree"],
            reverse=True,
        )[:5]

        weights = [d["weight"] for _, _, d in self._G.edges(data=True)]
        avg_similarity = round(sum(weights) / len(weights), 4) if weights else 0.0

        return {
            "nodes": n,
            "edges": e,
            "density": round(nx.density(self._G), 4),
            "avg_similarity": avg_similarity,
            "clustering_coefficient": clustering,
            "connected_components": num_components,
            "largest_component_size": largest_size,
            "diameter": diameter,
            "degree_distribution": {
                "avg_degree": avg_degree,
                "top_hubs": top_hubs,
            },
        }

    def summary(self) -> dict:
        """Resumen estadístico básico del grafo."""
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

    # ── Serialización ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serializa el grafo a dict para la API."""
        return {
            "nodes": [{"id": n, **self._G.nodes[n]} for n in self._G.nodes],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **d,
                    "confidence_level": _compute_confidence_level(
                        float(d.get("weight", 0.0)), int(d.get("evidence_count", 1))
                    ),
                }
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
        """Exporta visualización interactiva con PyVis."""
        try:
            from pyvis.network import Network
        except ImportError:
            log.error("pyvis_not_installed")
            return ""

        out = Path(path) if path else GRAPH_OUTPUT_DIR / "red_rupestre.html"
        net = Network(
            height=height,
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            notebook=False,
            directed=False,
            cdn_resources="in_line",
        )
        net.set_options(
            """
        {
          "physics": {"solver": "forceAtlas2Based", "stabilization": {"iterations": 150}},
          "edges": {"smooth": {"type": "continuous"}, "color": {"inherit": "both"}},
          "nodes": {"shape": "dot", "scaling": {"min": 10, "max": 40}},
          "interaction": {"hover": true, "tooltipDelay": 200}
        }
        """
        )

        pr = self.pagerank()
        communities_list = self.communities()
        node_community: dict[str, int] = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                node_community[node] = i

        colors = [
            "#e74c3c",
            "#3498db",
            "#2ecc71",
            "#f39c12",
            "#9b59b6",
            "#1abc9c",
            "#e67e22",
            "#34495e",
        ]

        for node in self._G.nodes:
            attrs = self._G.nodes[node]
            size = 15 + int(pr.get(node, 0) * 500)
            color = colors[node_community.get(node, 0) % len(colors)]
            title = (
                f"<b>{node}</b><br>"
                f"Municipio: {attrs.get('municipality', '')}<br>"
                f"Taxonomía dominante: {attrs.get('dominant_taxonomy', 'Indeterminado')}<br>"
                f"Petroglifos: {attrs.get('petroglyph_count', 0)}<br>"
                f"PageRank: {pr.get(node, 0):.4f}"
            )
            net.add_node(node, label=node, size=size, color=color, title=title)

        for u, v, data in self._G.edges(data=True):
            weight = float(data.get("weight", 0.5))
            title = (
                f"Similitud: {weight:.2%}<br>"
                f"Evidencias: {data.get('evidence_count', 1)}<br>"
                f"Taxonomías compartidas: {', '.join(data.get('shared_taxonomies', [])) or 'N/A'}"
            )
            net.add_edge(u, v, value=weight, title=title, width=weight * 5)

        html = net.generate_html()
        if not html.lstrip().lower().startswith("<!doctype"):
            html = "<!DOCTYPE html>\n" + html

        out.write_text(html, encoding="utf-8")

        log.info("graph_exported_html", path=str(out), nodes=len(self._G.nodes))
        return str(out)

    def export_image(self, path: str | None = None) -> str:
        """Exporta una imagen estática del grafo como PNG."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.error("matplotlib_not_installed")
            return ""

        if len(self._G) == 0:
            return ""

        out = Path(path) if path else GRAPH_OUTPUT_DIR / "red_rupestre.png"
        pr = self.pagerank()
        communities_list = self.communities()
        node_community: dict[str, int] = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                node_community[node] = i

        palette = [
            "#e74c3c",
            "#3498db",
            "#2ecc71",
            "#f39c12",
            "#9b59b6",
            "#1abc9c",
            "#e67e22",
            "#34495e",
        ]

        n_nodes = self._G.number_of_nodes()
        k_spacing = 4.0 / (n_nodes**0.5) if n_nodes else 1.0
        pos = nx.spring_layout(self._G, weight="weight", seed=42, k=k_spacing, iterations=300)

        fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
        ax.set_title("Red Social de Similitud Iconográfica - Sitios Rupestres", fontsize=16, pad=20)
        ax.axis("off")

        reliable_edges = [
            (u, v)
            for u, v, d in self._G.edges(data=True)
            if not d.get("is_provisional", True)
        ]
        provisional_edges = [
            (u, v)
            for u, v, d in self._G.edges(data=True)
            if d.get("is_provisional", True)
        ]

        if provisional_edges:
            nx.draw_networkx_edges(
                self._G,
                pos,
                edgelist=provisional_edges,
                ax=ax,
                width=1.0,
                alpha=0.18,
                edge_color="#a0a0aa",
                style="dashed",
            )
        if reliable_edges:
            nx.draw_networkx_edges(
                self._G,
                pos,
                edgelist=reliable_edges,
                ax=ax,
                width=2.2,
                alpha=0.55,
                edge_color="#5a5f78",
            )

        for comm_id in sorted(set(node_community.values())):
            nodes_in_comm = [n for n in self._G.nodes if node_community.get(n) == comm_id]
            if not nodes_in_comm:
                continue
            color = palette[comm_id % len(palette)]
            sizes = [450 + pr.get(n, 0.0) * 7000 for n in nodes_in_comm]
            nx.draw_networkx_nodes(
                self._G,
                pos,
                nodelist=nodes_in_comm,
                node_color=color,
                node_size=sizes,
                ax=ax,
                linewidths=1.5,
                edgecolors="white",
                alpha=0.95,
            )

        nx.draw_networkx_labels(
            self._G,
            pos,
            font_size=9,
            font_color="#1f1f1f",
            ax=ax,
        )

        plt.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)

        log.info("graph_exported_image", path=str(out), nodes=len(self._G.nodes))
        return str(out)

    def export_plotly(self, path: str | None = None) -> str:
        """Exporta el grafo como HTML interactivo con Plotly."""
        import plotly.graph_objects as go

        if len(self._G) == 0:
            return ""

        n_nodes = self._G.number_of_nodes()
        k_spacing = 4.0 / (n_nodes**0.5) if n_nodes else 1.0
        pos = nx.spring_layout(self._G, weight="weight", seed=42, k=k_spacing, iterations=300)
        pr = self.pagerank()
        communities_list = self.communities()
        node_community: dict[str, int] = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                node_community[node] = i

        palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

        rel_x, rel_y, prov_x, prov_y = [], [], [], []
        hover_x, hover_y, hover_txt = [], [], []
        for u, v, data in self._G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = float(data.get("weight", 0.5))
            if data.get("is_provisional", True):
                prov_x += [x0, x1, None]
                prov_y += [y0, y1, None]
            else:
                rel_x += [x0, x1, None]
                rel_y += [y0, y1, None]
                taxonomies = ", ".join(data.get("shared_taxonomies", [])) or "N/A"
                hover_x.append((x0 + x1) / 2)
                hover_y.append((y0 + y1) / 2)
                hover_txt.append(
                    f"<b>{u} ↔ {v}</b><br>Similitud: {weight:.2%}<br>"
                    f"Evidencias: {data.get('evidence_count', 1)}<br>"
                    f"Taxonomías: {taxonomies}"
                )

        edge_traces = [
            go.Scatter(
                x=rel_x,
                y=rel_y,
                mode="lines",
                line=dict(width=1.4, color="rgba(90,95,120,0.45)"),
                hoverinfo="skip",
                name="Conexiones confiables",
            ),
            go.Scatter(
                x=prov_x,
                y=prov_y,
                mode="lines",
                line=dict(width=0.6, color="rgba(160,160,170,0.18)"),
                hoverinfo="skip",
                name="Provisionales",
                visible="legendonly",
            ),
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker=dict(size=6, color="rgba(0,0,0,0)"),
                hoverinfo="text",
                text=hover_txt,
                showlegend=False,
            ),
        ]

        node_traces = []
        for comm_id in sorted(set(node_community.values())):
            nodes_in_comm = [n for n in self._G.nodes if node_community.get(n) == comm_id]
            xs = [pos[n][0] for n in nodes_in_comm]
            ys = [pos[n][1] for n in nodes_in_comm]
            sizes = [20 + pr.get(n, 0) * 600 for n in nodes_in_comm]
            attrs_list = [self._G.nodes[n] for n in nodes_in_comm]
            hover_texts = [
                f"<b>{n}</b><br>"
                f"Municipio: {a.get('municipality', '')}<br>"
                f"Taxonomía: {a.get('dominant_taxonomy', 'Indeterminado')}<br>"
                f"PageRank: {pr.get(n, 0):.4f}<br>"
                f"Comunidad: {comm_id + 1}"
                for n, a in zip(nodes_in_comm, attrs_list)
            ]
            color = palette[comm_id % len(palette)]
            node_traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    marker=dict(size=sizes, color=color, line=dict(width=2, color="white")),
                    text=nodes_in_comm,
                    textposition="top center",
                    textfont=dict(size=11, color="#333"),
                    hovertext=hover_texts,
                    hoverinfo="text",
                    name=f"Comunidad {comm_id + 1}",
                )
            )

        fig = go.Figure(
            data=edge_traces + node_traces,
            layout=go.Layout(
                title=dict(
                    text="Red Social de Similitud Iconográfica — Sitios Rupestres",
                    font=dict(size=18),
                    x=0.5,
                ),
                showlegend=True,
                hovermode="closest",
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=20, r=20, t=60, b=20),
                height=650,
                legend=dict(title="Comunidades", bordercolor="#ccc", borderwidth=1),
                annotations=[
                    dict(
                        text=(
                            f"Nodos: {self._G.number_of_nodes()} | "
                            f"Aristas: {self._G.number_of_edges()} | "
                            f"Densidad: {nx.density(self._G):.2f}"
                        ),
                        xref="paper",
                        yref="paper",
                        x=0.01,
                        y=0.01,
                        showarrow=False,
                        font=dict(size=11, color="#888"),
                    )
                ],
            ),
        )

        out = Path(path) if path else GRAPH_OUTPUT_DIR / "red_rupestre_plotly.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        log.info("graph_exported_plotly", path=str(out), nodes=len(self._G.nodes))
        return str(out)

    # ── Persistencia en PostgreSQL ──────────────────────────────────────────

    async def sync_to_db(self, session) -> None:
        """Sincroniza todas las aristas del grafo en memoria a site_graph_edges."""
        from sqlalchemy import select

        from infrastructure.database.models.models import RupestranSiteModel, SiteGraphEdge

        result = await session.execute(select(RupestranSiteModel))
        sites_by_name = {s.name: s.id for s in result.scalars()}

        for u, v, data in self._G.edges(data=True):
            id_a = sites_by_name.get(u)
            id_b = sites_by_name.get(v)
            if not id_a or not id_b:
                continue

            site_a_id, site_b_id = sorted([id_a, id_b])
            existing = (
                await session.execute(
                    select(SiteGraphEdge).where(
                        SiteGraphEdge.site_a_id == site_a_id,
                        SiteGraphEdge.site_b_id == site_b_id,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                existing.weight = data["weight"]
                existing.evidence_count = data.get("evidence_count", 1)
                existing.shared_taxonomies = data.get("shared_taxonomies", [])
            else:
                session.add(
                    SiteGraphEdge(
                        site_a_id=site_a_id,
                        site_b_id=site_b_id,
                        weight=data["weight"],
                        evidence_count=data.get("evidence_count", 1),
                        shared_taxonomies=data.get("shared_taxonomies", []),
                    )
                )

        await session.commit()
        log.info("graph_synced_to_db", edges=self._G.number_of_edges())

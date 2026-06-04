"""A6 — Documentador (Jinja2 + WeasyPrint → Ficha ICANH en PDF + JSON)."""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from textwrap import wrap
import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape
from agents.base_agent import BaseAgent, AgentInput, AgentOutput

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("storage/fichas_icanh")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_TEMPLATES_DIR = Path("prompts/templates")
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


class DocumentorAgent(BaseAgent):
    name = "a6_documentor"

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        p = input.payload

        record = {
            "petroglyph_id": p.get("petroglyph_id", input.task_id),
            "site": p.get("site", "Sin nombre"),
            "municipality": p.get("municipality", ""),
            "department": p.get("department", ""),
            "gps_coordinates": p.get("gps_coordinates", {}),
            "taxonomy": p.get("taxonomy", "Indeterminado"),
            "confidence": p.get("confidence", 0.0),
            "justification": p.get("justification", ""),
            "petroglyph_description": p.get("petroglyph_description", {}),
            "rag_feedback": p.get("rag_feedback", {}),
            "segmentation_validation": p.get("segmentation_validation", {}),
            "reconstruction_diagnostics": p.get("reconstruction_diagnostics", {}),
            "reconstruction_assessment": p.get("reconstruction_assessment", {}),
            "detected_shapes": p.get("detected_shapes", []),
            "similarity_matches": p.get("similarity_matches", []),
            "conservation_status": p.get("conservation_status", "Regular"),
            "researcher_notes": p.get("researcher_notes", ""),
            "requires_expert_validation": p.get("requires_validation", True),
            "generation_date": datetime.utcnow().isoformat(),
            "image_path": p.get("preprocessed_image_path", "") or p.get("image_path", ""),
            "reconstructed_image_path": p.get("reconstructed_image_path", ""),
        }

        task_id = input.task_id
        json_path = OUTPUT_DIR / f"{task_id}_ficha.json"
        pdf_path = OUTPUT_DIR / f"{task_id}_ficha.pdf"

        # 1. Guardar JSON
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        # 2. Renderizar HTML y generar PDF
        try:
            html_content = self._render_html(record)
            self._generate_pdf(html_content, pdf_path, record)
            pdf_url = str(pdf_path)
        except Exception as e:
            log.warning("pdf_generation_failed", error=str(e), task_id=task_id)
            pdf_url = ""

        elapsed = round((time.monotonic() - t0) * 1000)
        log.info("a6_documentor_done",
                 task_id=task_id,
                 json_path=str(json_path),
                 pdf_path=pdf_url,
                 latency_ms=elapsed)

        return AgentOutput(
            task_id=task_id,
            agent_name=self.name,
            result={
                "icanh_pdf_url": pdf_url,
                "icanh_json_path": str(json_path),
                "icanh_record": record,
            },
            status="success",
            metadata={"latency_ms": elapsed},
        )

    def _render_html(self, record: dict) -> str:
        try:
            template = _jinja_env.get_template("icanh_record.html")
            return template.render(**record)
        except Exception:
            # Fallback: HTML básico inline
            return self._fallback_html(record)

    def _generate_pdf(self, html: str, output_path: Path, record: dict) -> None:
        try:
            from weasyprint import HTML

            HTML(string=html).write_pdf(str(output_path))
            return
        except Exception as e:
            log.warning("weasyprint_fallback_pdf", error=str(e), task_id=record.get("petroglyph_id"))

        self._generate_pdf_fallback(record, output_path)

    def _generate_pdf_fallback(self, record: dict, output_path: Path) -> None:
        """Genera un PDF simple sin dependencias nativas externas."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        def _safe(value) -> str:
            return "" if value is None else str(value)

        def _add_wrapped(fig, y: float, label: str, value: str, *, value_width: int = 86) -> float:
            fig.text(0.08, y, label, fontsize=11, fontweight="bold", color="#1f2937")
            y -= 0.022
            lines = wrap(value, width=value_width) or [""]
            for line in lines:
                fig.text(0.10, y, line, fontsize=10, color="#334155")
                y -= 0.018
            return y - 0.008

        segmentation_validation = record.get("segmentation_validation", {}) or {}
        reconstruction_diagnostics = record.get("reconstruction_diagnostics", {}) or {}
        reconstruction_assessment = record.get("reconstruction_assessment", {}) or {}
        petroglyph_description = record.get("petroglyph_description", {}) or {}
        rag_feedback = record.get("rag_feedback", {}) or {}

        with PdfPages(str(output_path)) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            y = 0.96
            fig.text(0.08, y, "Ficha de Registro ICANH - Petroglifo", fontsize=18, fontweight="bold")
            y -= 0.04
            fig.text(0.08, y, f"ID: {_safe(record.get('petroglyph_id'))}", fontsize=10, color="#475569")
            y -= 0.03
            fig.text(
                0.08,
                y,
                f"Sitio: {_safe(record.get('site'))}, {_safe(record.get('municipality'))}, {_safe(record.get('department'))}",
                fontsize=10,
                color="#475569",
            )
            y -= 0.05

            y = _add_wrapped(fig, y, "Clasificacion:", _safe(record.get("taxonomy", "Indeterminado")))
            y = _add_wrapped(fig, y, "Confianza:", f"{round(float(record.get('confidence', 0.0)) * 100, 1)}%")
            y = _add_wrapped(fig, y, "Estado de conservacion:", _safe(record.get("conservation_status", "Regular")))
            y = _add_wrapped(fig, y, "Justificacion:", _safe(record.get("justification", "No disponible.")))
            y = _add_wrapped(
                fig,
                y,
                "Descripcion tecnica:",
                _safe(petroglyph_description.get("detailed_description", "No disponible.")),
            )
            y = _add_wrapped(fig, y, "Sitio probable:", _safe(petroglyph_description.get("probable_site", "No definido")))
            y = _add_wrapped(
                fig,
                y,
                "Probabilidad de sitio:",
                f"{round(float(petroglyph_description.get('site_probability', 0.0)) * 100, 1)}%",
            )
            y = _add_wrapped(
                fig,
                y,
                "RAG promedio:",
                f"{round(float(rag_feedback.get('avg_similarity', 0.0)) * 100, 1)}%",
            )
            y = _add_wrapped(
                fig,
                y,
                "Formas detectadas:",
                ", ".join(record.get("detected_shapes", [])) or "No se detectaron formas.",
            )
            y = _add_wrapped(fig, y, "Coincidencias iconograficas:", str(len(record.get("similarity_matches", []))))
            y = _add_wrapped(
                fig,
                y,
                "Segmentacion:",
                f"score={_safe(segmentation_validation.get('validation_score', 'N/A'))}, "
                f"status={_safe(segmentation_validation.get('segmentation_status', 'N/A'))}, "
                f"area={_safe(segmentation_validation.get('area_percent', 'N/A'))}, "
                f"warnings={', '.join(segmentation_validation.get('validation_warnings', [])) or 'N/A'}",
            )
            y = _add_wrapped(
                fig,
                y,
                "Reconstruccion:",
                f"pipeline={_safe(reconstruction_diagnostics.get('pipeline', 'N/A'))}, "
                f"endpoint={_safe(reconstruction_diagnostics.get('endpoint', 'N/A'))}, "
                f"damage_pixels={_safe(reconstruction_diagnostics.get('reconstruction_response', {}).get('damage_pixel_count', 'N/A'))}, "
                f"guide_pixels={_safe(reconstruction_diagnostics.get('reconstruction_response', {}).get('guide_pixel_count', 'N/A'))}",
            )
            y = _add_wrapped(
                fig,
                y,
                "Criterio humano:",
                f"estado={_safe(reconstruction_assessment.get('conservation_status', 'N/A'))}, "
                f"score={_safe(round(float(reconstruction_assessment.get('conservation_score', 0.0)) * 100, 1)) if reconstruction_assessment else 'N/A'}%, "
                f"recomendada={_safe(reconstruction_assessment.get('human_reconstruction_recommended', 'N/A'))}",
            )
            _damage_pct = reconstruction_assessment.get("damage_figure_percent")
            y = _add_wrapped(
                fig,
                y,
                "Dano de la figura (modelo):",
                f"{_damage_pct}%" if _damage_pct is not None else "N/A",
            )

            if record.get("requires_expert_validation", True):
                fig.text(
                    0.08,
                    max(0.04, y),
                    "Esta ficha requiere validacion por un arqueologo experto.",
                    fontsize=10,
                    color="#92400e",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb", edgecolor="#f59e0b"),
                )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    def _fallback_html(self, r: dict) -> str:
        confidence_pct = round(r["confidence"] * 100, 1)
        desc = r.get("petroglyph_description", {})
        rag_feedback = r.get("rag_feedback", {})
        segmentation_validation = r.get("segmentation_validation", {})
        reconstruction_diagnostics = r.get("reconstruction_diagnostics", {})
        reconstruction_assessment = r.get("reconstruction_assessment", {})
        rag_avg = round(float(rag_feedback.get("avg_similarity", 0.0)) * 100, 1)
        shapes_html = "".join(f"<li>{s}</li>" for s in r["detected_shapes"])
        matches_html = "".join(
            f"<li>{m.get('reference_name', '?')} — {m.get('site_name', '?')} "
            f"(similitud: {round(m.get('similarity_score', 0)*100, 1)}%)</li>"
            for m in r["similarity_matches"]
        )
        key_info_html = "".join(
            f"<li>{item}</li>" for item in desc.get("key_figure_info", [])
        )
        rag_top_html = "".join(
            f"<li>{m.get('source', '?')} (similitud: {round(m.get('similarity', 0)*100, 1)}%)</li>"
            for m in rag_feedback.get("top_matches", [])
        )
        return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #5a3e1b; border-bottom: 2px solid #a0522d; padding-bottom: 8px; }}
  h2 {{ color: #7b5927; margin-top: 24px; }}
  .badge {{ background: #ffe4b5; padding: 4px 12px; border-radius: 12px;
             font-weight: bold; display: inline-block; }}
  .meta {{ background: #fdf5e6; padding: 12px 16px; border-radius: 8px; margin: 12px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; }}
  td:first-child {{ color: #666; width: 180px; }}
  .validation-warning {{ background: #fff3cd; border-left: 4px solid #ffc107;
                          padding: 10px 14px; margin: 16px 0; }}
</style>
</head>
<body>
<h1>Ficha de Registro ICANH — Petroglifo</h1>

<div class="meta">
  <strong>Sitio:</strong> {r["site"]}, {r["municipality"]}, {r["department"]}<br>
  <strong>Fecha de generación:</strong> {r["generation_date"]}<br>
  <strong>ID:</strong> {r["petroglyph_id"]}
</div>

<h2>Clasificación taxonómica</h2>
<table>
  <tr><td>Categoría</td><td><span class="badge">{r["taxonomy"]}</span></td></tr>
  <tr><td>Confianza</td><td>{confidence_pct}%</td></tr>
  <tr><td>Estado de conservación</td><td>{r["conservation_status"]}</td></tr>
</table>

<h2>Justificación arqueológica</h2>
<p>{r["justification"] or "No disponible."}</p>

<h2>Descripción técnica del petroglifo (LLM)</h2>
<p>{desc.get("detailed_description", "No disponible.")}</p>
<table>
    <tr><td>Sitio probable</td><td>{desc.get("probable_site", "No definido")}</td></tr>
    <tr><td>Probabilidad de sitio</td><td>{round(float(desc.get("site_probability", 0.0))*100, 1)}%</td></tr>
</table>

<h2>Información clave de la figura</h2>
<ul>{key_info_html or "<li>No disponible.</li>"}</ul>

<h2>Retroalimentacion RAG <-> descripcion</h2>
<table>
    <tr><td>Consistencia promedio</td><td>{rag_avg}%</td></tr>
    <tr><td>Etiqueta</td><td>{rag_feedback.get("consistency_label", "N/A")}</td></tr>
</table>
<ul>{rag_top_html or "<li>Sin evidencia de alineación.</li>"}</ul>

<h2>Diagnostico de segmentacion</h2>
<table>
    <tr><td>Validation score</td><td>{segmentation_validation.get("validation_score", "N/A")}</td></tr>
    <tr><td>Segmentation status</td><td>{segmentation_validation.get("segmentation_status", "N/A")}</td></tr>
    <tr><td>Area percent</td><td>{segmentation_validation.get("area_percent", "N/A")}</td></tr>
    <tr><td>Warnings</td><td>{", ".join(segmentation_validation.get("validation_warnings", [])) or "N/A"}</td></tr>
</table>

<h2>Diagnostico de reconstruccion</h2>
<table>
    <tr><td>Pipeline</td><td>{reconstruction_diagnostics.get("pipeline", "N/A")}</td></tr>
    <tr><td>Endpoint</td><td>{reconstruction_diagnostics.get("endpoint", "N/A")}</td></tr>
    <tr><td>Damage pixels</td><td>{reconstruction_diagnostics.get("reconstruction_response", {}).get("damage_pixel_count", "N/A")}</td></tr>
    <tr><td>Guide pixels</td><td>{reconstruction_diagnostics.get("reconstruction_response", {}).get("guide_pixel_count", "N/A")}</td></tr>
    <tr><td>Estado de conservacion</td><td>{reconstruction_assessment.get("conservation_status", "N/A")}</td></tr>
    <tr><td>Score humano</td><td>{round(float(reconstruction_assessment.get("conservation_score", 0.0)) * 100, 1) if reconstruction_assessment else "N/A"}%</td></tr>
    <tr><td>Daño de la figura (modelo)</td><td>{f'{reconstruction_assessment.get("damage_figure_percent")}%' if reconstruction_assessment.get("damage_figure_percent") is not None else "N/A"}</td></tr>
    <tr><td>Reconstruccion recomendada</td><td>{reconstruction_assessment.get("reconstruction_recommended", "N/A")}</td></tr>
</table>

<h2>Formas detectadas</h2>
<ul>{shapes_html or "<li>No se detectaron formas.</li>"}</ul>

<h2>Coincidencias iconográficas</h2>
<ul>{matches_html or "<li>Sin coincidencias en el corpus de referencia.</li>"}</ul>

{"<div class='validation-warning'>⚠ Esta ficha requiere validación por un arqueólogo experto.</div>" if r["requires_expert_validation"] else ""}

<p style="margin-top: 40px; color: #999; font-size: 12px;">
  Generado automáticamente por el Sistema de IA para Petroglifos Andinos Colombianos (UPTC)
</p>
</body>
</html>"""

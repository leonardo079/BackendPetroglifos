"""A6 — Documentador (Jinja2 + WeasyPrint → Ficha ICANH en PDF + JSON)."""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape
from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from infrastructure.storage.cloudinary_service import upload_pdf

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
            self._generate_pdf(html_content, pdf_path)
            pdf_url = str(pdf_path)
            pdf_cloudinary_url = upload_pdf(pdf_path, public_id=f"{task_id}_ficha")
        except Exception as e:
            log.warning("pdf_generation_failed", error=str(e), task_id=task_id)
            pdf_url = ""
            pdf_cloudinary_url = ""

        elapsed = round((time.monotonic() - t0) * 1000)
        log.info("a6_documentor_done",
                 task_id=task_id,
                 json_path=str(json_path),
                 pdf_path=pdf_url,
                 pdf_cloudinary_url=pdf_cloudinary_url or None,
                 latency_ms=elapsed)

        return AgentOutput(
            task_id=task_id,
            agent_name=self.name,
            result={
                "icanh_pdf_url": pdf_url,
                "icanh_pdf_cloudinary_url": pdf_cloudinary_url,
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

    def _generate_pdf(self, html: str, output_path: Path) -> None:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(output_path))

    def _fallback_html(self, r: dict) -> str:
        confidence_pct = round(r["confidence"] * 100, 1)
        shapes_html = "".join(f"<li>{s}</li>" for s in r["detected_shapes"])
        matches_html = "".join(
            f"<li>{m.get('reference_name', '?')} — {m.get('site_name', '?')} "
            f"(similitud: {round(m.get('similarity_score', 0)*100, 1)}%)</li>"
            for m in r["similarity_matches"]
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

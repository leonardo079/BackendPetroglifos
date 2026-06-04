"""A5 — Reconstructor de petroglifos (activo solo cuando A2 detecta deterioro)."""
from __future__ import annotations
import base64
import time
import shutil
from pathlib import Path
import httpx
import structlog
from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from config.settings import settings

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("storage/reconstructed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ReconstructorAgent(BaseAgent):
    """
    Invoca la API de reconstrucción externa (Keras U-Net + LaMa inpainting)
    para restaurar petroglifos deteriorados.
    Si GAN_MOCK_MODE=true retorna la imagen preprocesada como fallback.
    """
    name = "a5_reconstructor"

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        image_path: str = input.payload.get("preprocessed_image_path", "") or \
                          input.payload.get("image_path", "")
        deterioration: bool = input.payload.get("deterioration_detected", True)
        segmentation_validation: dict = input.payload.get("segmentation_validation", {})
        conservation_status: str = str(input.payload.get("conservation_status", "Regular"))
        reconstruction_assessment: dict = input.payload.get("reconstruction_assessment", {})

        if not deterioration:
            # No hay deterioro detectado: pasar imagen sin reconstruir
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={
                    "reconstructed_image_path": image_path,
                    "reconstruction_diagnostics": {
                        "pipeline": "skipped",
                        "reason": "no_deterioration",
                        "damage_severity": self._damage_severity(segmentation_validation),
                        "segmentation_validation": segmentation_validation,
                        "conservation_status": conservation_status,
                        "reconstruction_assessment": reconstruction_assessment,
                    },
                },
                status="skipped",
                metadata={"reason": "no_deterioration"},
            )

        if settings.gan_mock_mode:
            result_path = await self._mock_reconstruct(image_path, input.task_id)
            elapsed = round((time.monotonic() - t0) * 1000)
            log.info("a5_reconstructor_mock", task_id=input.task_id, latency_ms=elapsed)
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={
                    "reconstructed_image_path": result_path,
                    "reconstruction_diagnostics": {
                        "pipeline": "mock",
                        "damage_severity": self._damage_severity(segmentation_validation),
                        "segmentation_validation": segmentation_validation,
                        "conservation_status": conservation_status,
                        "reconstruction_assessment": reconstruction_assessment,
                    },
                },
                status="fallback",
                metadata={"mode": "mock", "latency_ms": elapsed},
            )

        try:
            result_path, diagnostics = await self._call_reconstruction_pipeline(
                image_path=image_path,
                task_id=input.task_id,
                segmentation_validation=segmentation_validation,
                conservation_status=conservation_status,
            )
            elapsed = round((time.monotonic() - t0) * 1000)
            log.info("a5_reconstructor_done", task_id=input.task_id, latency_ms=elapsed)
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={
                    "reconstructed_image_path": result_path,
                    "reconstruction_diagnostics": diagnostics,
                },
                status="success",
                metadata={"mode": diagnostics.get("pipeline", "api"), "latency_ms": elapsed},
            )
        except Exception as e:
            log.error("a5_reconstructor_error", error=str(e), task_id=input.task_id)
            # Fallback: usar imagen preprocesada
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={
                    "reconstructed_image_path": image_path,
                    "reconstruction_diagnostics": {
                        "pipeline": "fallback_to_preprocessed",
                        "error": str(e),
                        "damage_severity": self._damage_severity(segmentation_validation),
                        "segmentation_validation": segmentation_validation,
                        "conservation_status": conservation_status,
                        "reconstruction_assessment": reconstruction_assessment,
                    },
                },
                status="fallback",
                metadata={"error": str(e), "mode": "fallback_to_preprocessed"},
            )

    async def _call_reconstruction_pipeline(
        self,
        image_path: str,
        task_id: str,
        segmentation_validation: dict,
        conservation_status: str,
    ) -> tuple[str, dict]:
        """
        Intenta primero el pipeline completo de reconstruccion, que incluye la
        calificacion del dano, y cae al endpoint legacy si no esta disponible.
        """
        pipeline_errors: list[str] = []

        for pipeline_name, url, file_field in self._pipeline_candidates(segmentation_validation):
            try:
                return await self._call_reconstruction_endpoint(
                    image_path=image_path,
                    task_id=task_id,
                    url=url,
                    pipeline_name=pipeline_name,
                    file_field=file_field,
                    segmentation_validation=segmentation_validation,
                    conservation_status=conservation_status,
                )
            except Exception as exc:
                pipeline_errors.append(f"{pipeline_name}: {exc}")
                log.warning(
                    "a5_reconstruction_pipeline_failed",
                    task_id=task_id,
                    pipeline=pipeline_name,
                    error=str(exc),
                )

        raise RuntimeError("No se pudo ejecutar ningun pipeline de reconstruccion: " + " | ".join(pipeline_errors))

    def _pipeline_candidates(self, segmentation_validation: dict) -> list[tuple[str, str, str]]:
        """
        Ordena los pipelines segun la severidad de la segmentacion.
        Cuando la mascara es debil o muy fragmentada, priorizamos el modo
        visual asistido; en casos moderados, primero se intenta la ruta completa
        con metrica de validacion.
        """
        score = float(segmentation_validation.get("validation_score", 0.0) or 0.0)
        status = str(segmentation_validation.get("segmentation_status", "unknown"))
        area_percent = float(segmentation_validation.get("area_percent", 0.0) or 0.0)
        warnings = set(segmentation_validation.get("validation_warnings", []) or [])

        severe_damage = (
            score < 0
            or status == "weak_segmentation"
            or area_percent < 6.0
            or "fragmented_mask" in warnings
            or "weak_main_component" in warnings
        )

        visual_first = severe_damage or area_percent < 4.0

        if visual_first:
            return [
                ("reconstructVisualAssisted", settings.reconstruction_visual_assisted_url, "file"),
                ("reconstructFull", f"{settings.reconstruction_api_base_url}/reconstructFull", "file"),
                ("legacy_reconstruct", settings.gan_api_url, "image"),
            ]

        return [
            ("reconstructFull", f"{settings.reconstruction_api_base_url}/reconstructFull", "file"),
            ("reconstructVisualAssisted", settings.reconstruction_visual_assisted_url, "file"),
            ("legacy_reconstruct", settings.gan_api_url, "image"),
        ]

    def _damage_severity(self, segmentation_validation: dict) -> str:
        score = float(segmentation_validation.get("validation_score", 0.0) or 0.0)
        status = str(segmentation_validation.get("segmentation_status", "unknown"))
        area_percent = float(segmentation_validation.get("area_percent", 0.0) or 0.0)
        warnings = set(segmentation_validation.get("validation_warnings", []) or [])

        if (
            score < 0
            or status == "weak_segmentation"
            or area_percent < 6.0
            or "fragmented_mask" in warnings
            or "weak_main_component" in warnings
        ):
            return "severe"
        if area_percent < 6.0 or warnings:
            return "moderate"
        return "mild"

    async def _call_reconstruction_endpoint(
        self,
        image_path: str,
        task_id: str,
        url: str,
        pipeline_name: str,
        file_field: str,
        segmentation_validation: dict,
        conservation_status: str,
    ) -> tuple[str, dict]:
        out_path = OUTPUT_DIR / f"{task_id}_{pipeline_name}.png"
        async with httpx.AsyncClient(timeout=180.0) as client:
            headers = {}
            if settings.gan_api_key:
                headers["Authorization"] = f"Bearer {settings.gan_api_key}"
            with open(image_path, "rb") as f:
                response = await client.post(
                    url,
                    files={file_field: (Path(image_path).name, f, "image/jpeg")},
                    data={"include_previews": "false"} if file_field == "file" else None,
                    headers=headers,
                )
            response.raise_for_status()

        if "application/json" in response.headers.get("content-type", ""):
            payload = response.json()
            image_b64 = (
                payload.get("result_image")
                or payload.get("final_image")
                or payload.get("fused_image")
                or payload.get("composed_image")
            )
            if not image_b64:
                raise RuntimeError(f"{pipeline_name} no devolvio una imagen base64 utilizable")

            out_path.write_bytes(base64.b64decode(image_b64))
            diagnostics = {
                "pipeline": pipeline_name,
                "endpoint": url,
                "damage_severity": self._damage_severity(segmentation_validation),
                "segmentation_validation": segmentation_validation,
                "conservation_status": conservation_status,
                "reconstruction_response": {
                    key: value
                    for key, value in payload.items()
                    if key
                    in {
                        "validation_score",
                        "validation_warnings",
                        "area_percent",
                        "segmentation_status",
                        "damage_pixel_count",
                        "damage_percent",
                        "guide_pixel_count",
                        "selected_guide_threshold",
                        "selected_guide_strategy",
                    }
                },
            }
            return str(out_path), diagnostics

        out_path.write_bytes(response.content)
        diagnostics = {
            "pipeline": pipeline_name,
            "endpoint": url,
            "damage_severity": self._damage_severity(segmentation_validation),
            "segmentation_validation": segmentation_validation,
            "conservation_status": conservation_status,
            "reconstruction_response": {
                "content_type": response.headers.get("content-type", ""),
                "bytes": len(response.content),
            },
        }
        return str(out_path), diagnostics

    async def _mock_reconstruct(self, image_path: str, task_id: str) -> str:
        """Mock: copia la imagen preprocesada como si fuera la reconstruida."""
        if not image_path or not Path(image_path).exists():
            return image_path
        out_path = OUTPUT_DIR / f"{task_id}_mock_reconstructed.png"
        shutil.copy2(image_path, out_path)
        return str(out_path)

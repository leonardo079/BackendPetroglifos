"""A5 — Reconstructor GAN (activo solo cuando A2 detecta deterioro)."""
from __future__ import annotations
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
    Invoca la API GAN externa para reconstruir petroglifos deteriorados.
    Si GAN_MOCK_MODE=true retorna la imagen preprocesada como fallback.
    """
    name = "a5_reconstructor"

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        image_path: str = input.payload.get("preprocessed_image_path", "") or \
                          input.payload.get("image_path", "")
        deterioration: bool = input.payload.get("deterioration_detected", True)

        if not deterioration:
            # No hay deterioro detectado: pasar imagen sin reconstruir
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={"reconstructed_image_path": image_path},
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
                result={"reconstructed_image_path": result_path},
                status="fallback",
                metadata={"mode": "mock", "latency_ms": elapsed},
            )

        try:
            result_path = await self._call_gan_api(image_path, input.task_id)
            elapsed = round((time.monotonic() - t0) * 1000)
            log.info("a5_reconstructor_done", task_id=input.task_id, latency_ms=elapsed)
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={"reconstructed_image_path": result_path},
                status="success",
                metadata={"mode": "api", "latency_ms": elapsed},
            )
        except Exception as e:
            log.error("a5_reconstructor_error", error=str(e), task_id=input.task_id)
            # Fallback: usar imagen preprocesada
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={"reconstructed_image_path": image_path},
                status="fallback",
                metadata={"error": str(e), "mode": "fallback_to_preprocessed"},
            )

    async def _call_gan_api(self, image_path: str, task_id: str) -> str:
        """Envía la imagen a la API GAN y guarda el resultado."""
        out_path = OUTPUT_DIR / f"{task_id}_reconstructed.jpg"
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {}
            if settings.gan_api_key:
                headers["Authorization"] = f"Bearer {settings.gan_api_key}"
            with open(image_path, "rb") as f:
                response = await client.post(
                    settings.gan_api_url,
                    files={"image": (Path(image_path).name, f, "image/jpeg")},
                    headers=headers,
                )
            response.raise_for_status()
            out_path.write_bytes(response.content)
        return str(out_path)

    async def _mock_reconstruct(self, image_path: str, task_id: str) -> str:
        """Mock: copia la imagen preprocesada como si fuera la reconstruida."""
        if not image_path or not Path(image_path).exists():
            return image_path
        out_path = OUTPUT_DIR / f"{task_id}_mock_reconstructed.jpg"
        shutil.copy2(image_path, out_path)
        return str(out_path)
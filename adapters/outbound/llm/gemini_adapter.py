"""Adaptador concreto para Gemini 1.5 Flash."""
from __future__ import annotations
import json
import time
import structlog
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.ports.outbound.llm_port import LLMPort
from config.settings import settings

log = structlog.get_logger(__name__)

_GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    max_output_tokens=2048,
)


class GeminiAdapter(LLMPort):
    """Adaptador para Gemini 1.5 Flash con retry exponencial."""

    def __init__(self, lite: bool = False) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        model_name = self._normalize_model_name(
            settings.gemini_model_lite if lite else settings.gemini_model
        )
        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=_GENERATION_CONFIG,
        )
        self._model_name = model_name

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """
        Gemini espera el nombre del recurso en formato `models/{model}`.
        Aceptamos nombres planos desde `.env` y los convertimos al formato
        canónico que usa la API.
        """
        normalized = model_name.strip()
        if normalized.startswith("models/") or normalized.startswith("tunedModels/"):
            return normalized
        return f"models/{normalized}"

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    async def generate(self, prompt: str, system: str = "") -> str:
        """Genera texto con Gemini. Registra latencia y tokens."""
        t0 = time.monotonic()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = self._model.generate_content(full_prompt)
        elapsed_ms = round((time.monotonic() - t0) * 1000)

        text = response.text.strip()
        log.info(
            "gemini_inference",
            model=self._model_name,
            latency_ms=elapsed_ms,
            prompt_chars=len(full_prompt),
            response_chars=len(text),
        )
        return text

    async def generate_json(self, prompt: str, system: str = "") -> dict:
        """Genera y parsea una respuesta JSON de Gemini."""
        raw = await self.generate(prompt, system)
        # Eliminar posibles bloques markdown ```json ... ```
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            log.error("gemini_json_parse_error", raw=raw[:200], error=str(e))
            raise ValueError(f"Gemini no retornó JSON válido: {e}") from e

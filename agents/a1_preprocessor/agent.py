"""A1 — Preprocesador de imagen (OpenCV)."""
from __future__ import annotations
import os
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
import structlog
from agents.base_agent import BaseAgent, AgentInput, AgentOutput

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("storage/preprocessed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PreprocessorAgent(BaseAgent):
    name = "a1_preprocessor"

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        image_path: str = input.payload.get("image_path", "")

        if not image_path or not os.path.exists(image_path):
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={},
                status="error",
                metadata={"error": f"Imagen no encontrada: {image_path}"},
            )

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"cv2 no pudo leer: {image_path}")

            processed = self._preprocess(img)
            out_path = OUTPUT_DIR / f"{input.task_id}_{uuid.uuid4().hex[:6]}.jpg"
            cv2.imwrite(str(out_path), processed)

            elapsed = round((time.monotonic() - t0) * 1000)
            log.info("a1_preprocessor_done", task_id=input.task_id, latency_ms=elapsed)

            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={"preprocessed_image_path": str(out_path)},
                status="success",
                metadata={"latency_ms": elapsed, "original_shape": img.shape},
            )
        except Exception as e:
            log.error("a1_preprocessor_error", error=str(e), task_id=input.task_id)
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result={},
                status="error",
                metadata={"error": str(e)},
            )

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Pipeline de preprocesamiento:
        1. Corrección de perspectiva (detección de contorno mayor)
        2. Conversión a escala de grises
        3. Realce de contraste CLAHE
        4. Reducción de ruido bilateral
        5. Normalización de iluminación
        """
        # 1. Resize si es muy grande (mantener aspect ratio)
        h, w = img.shape[:2]
        if max(h, w) > 2048:
            scale = 2048 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 2. Corrección de perspectiva automática
        img = self._correct_perspective(img)

        # 3. Escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4. CLAHE — realce de contraste adaptativo
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 5. Filtro bilateral (reduce ruido preservando bordes)
        denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # 6. Normalización de iluminación (equalización local)
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

        # Convertir de vuelta a BGR para compatibilidad
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    def _correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """Intenta corregir perspectiva detectando el contorno mayor."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return img
            largest = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(largest) / (img.shape[0] * img.shape[1])
            # Solo corregir si el contorno ocupa >15% de la imagen
            if area_ratio < 0.15:
                return img
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            if len(approx) == 4:
                return self._four_point_transform(img, approx.reshape(4, 2))
        except Exception:
            pass
        return img

    @staticmethod
    def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))
# Plan (v2) — Detección automática de daño para reconstruir (umbral 70%)

> Reescrito tras los cambios del compañero (commit `e456f6f`). La adaptación YOLO-cls se conservó
> y ahora existe una capa nueva de `conservation_status`. Este plan se integra con ella.

## Estado actual del código (lo que ya hay)

El compañero añadió una capa de **recomendación humana de reconstrucción** basada en el
`conservation_status` que ingresa el investigador:

- `agents/a2_detector/agent.py`:
  - `_CONSERVATION_SCORE_MAP`: bueno=0.0, regular=0.33, malo=0.75, crítico=1.0.
  - `human_reconstruction_recommended = conservation_score >= 0.75` (es decir, **Malo o Crítico**).
  - `reconstruction_recommended = deterioration_detected OR human_reconstruction_recommended`.
  - Devuelve también un dict `reconstruction_assessment`.
- `orchestrator/PetroglyphOrchestrator.py`:
  - El router (`_route_after_detection`) ahora decide con **`_reconstruction_recommended`**.
  - A5 recibe `deterioration_detected = _reconstruction_recommended`.
  - `reconstruction_assessment` fluye hasta A6.
- `agents/a6_documentor/agent.py`: la ficha ya muestra estado de conservación, score humano y
  "reconstrucción recomendada".

**Lo que sigue roto (causa raíz original):** la detección **automática** de deterioro en
`_check_deterioration_api` sigue usando solo **`/segmentPetroglyph`** (presencia/calidad de la
figura: `validation_score`, `area_percent<6.0`, warnings). **No consulta el modelo de daño.**
Por eso, si el usuario NO marca conservación = Malo/Crítico, una figura 70% dañada con surcos
limpios todavía da `deterioration_detected = False` → no se reconstruye sola.

El modelo de daño real existe en el servicio
(`recontrucción/petroglyph-service-reconstruction-api`, endpoint **`/segmentDamagePytorch`**,
peso `unet_danos_petroglifos_v3.pth`) y devuelve la máscara de daño como imagen base64
(`mask_image`). `/segmentPetroglyph` ya devuelve la máscara de la figura (`mask_image`) en su
respuesta base, incluso con `include_previews=false`.

## Decisiones (confirmadas antes)

1. **Daño relativo a la figura**: `daño ∩ figura / área_figura`. 70% = "70% de la figura dañada".
   Se calcula en el backend cruzando las dos máscaras → **no hay que tocar el servicio Keras**.
2. **Criterios actuales se mantienen como respaldo (OR)**.

## Por qué ahora el fix es mínimo

Como el compañero ya cableó `reconstruction_recommended = deterioration_detected OR human_…` y el
router/A5 consumen eso, **basta con que `deterioration_detected` se vuelva `True` cuando el daño de
la figura ≥ umbral**. Todo lo de abajo ya lo aprovecha automáticamente. No se toca el router ni A5.

## Implementación

### 1. `config/settings.py` — umbral configurable
Junto a los campos de reconstrucción:
```python
# Umbral de daño (fracción de la figura dañada) para forzar reconstrucción automática.
damage_reconstruction_threshold: float = 0.70  # env: DAMAGE_RECONSTRUCTION_THRESHOLD
```
Documentar la variable en `.env.example` y `.env`.

### 2. `agents/a2_detector/agent.py` — consultar el modelo de daño (núcleo)
- `import base64` (cv2/numpy/httpx ya están).
- En `_check_deterioration_api`:
  - Mantener la llamada a `/segmentPetroglyph` y los criterios de calidad actuales
    (`quality_deterioration`, incluido `area_percent < 6.0`) **sin cambios**.
  - Leer además `data.get("mask_image")` → máscara de figura (`_decode_mask_b64`).
  - Nueva llamada a `{settings.reconstruction_api_base_url}/segmentDamagePytorch`
    (`save_png=false`) → `mask_image` → máscara de daño.
  - `damage_figure_ratio = _figure_damage_ratio(figura, daño)`.
  - `damage_deterioration = damage_figure_ratio is not None and damage_figure_ratio >= settings.damage_reconstruction_threshold`.
  - **`deterioration_detected = quality_deterioration or damage_deterioration`**.
  - Si `/segmentDamagePytorch` falla → warning y seguir solo con calidad. Si `/segmentPetroglyph`
    falla → fallback conservador actual (`deterioration_detected = True`).
  - Añadir `damage_figure_percent` (= `round(ratio*100, 2)`) al dict y a `segmentation_validation`;
    registrarlo en el log `a2_deterioration_api`.
- Helpers privados nuevos:
  - `_decode_mask_b64(b64) -> np.ndarray | None`.
  - `_figure_damage_ratio(petro_mask, damage_mask) -> float | None` (resize `INTER_NEAREST` si
    difieren; `figure = petro_mask>127`; `figure_area==0` → `None`;
    `(figure & (damage_mask>127)).sum()/figure_area`).
- En `run()`, enriquecer `reconstruction_assessment` con
  `model_damage_recommended` (= `damage_deterioration`) y `damage_figure_percent`, para trazabilidad
  en la ficha. (`reconstruction_recommended` no cambia de fórmula: ya incluye `deterioration_detected`.)

### 3. `agents/a6_documentor/agent.py` (recomendado) — mostrar el % de daño
Añadir `damage_figure_percent` / `model_damage_recommended` (desde `reconstruction_assessment` o
`segmentation_validation`) en la sección de diagnóstico, junto a lo que ya se muestra.

### 4. `orchestrator/state/graph_state.py` (opcional, claridad de tipos)
Añadir las claves que el orquestador ya usa pero no están declaradas:
`reconstruction_assessment: dict` y `_reconstruction_recommended: bool`.

## Casos borde / fallback
- Servicio caído → fallback conservador actual (`deterioration_detected=True`).
- `/segmentPetroglyph` OK pero `/segmentDamagePytorch` falla → decide por calidad (OR sin daño).
- Figura no detectada (`figure_area==0`) → `area_percent<6.0` ya dispara; ratio = `None`.
- Máscaras 512×512 (mismos `IMG_SIZE`); se redimensiona por seguridad.
- El override humano (Malo/Crítico) sigue funcionando igual, en paralelo al automático.

## Verificación (end-to-end)
**Prerrequisito:** servicio en `:8001` con `/segmentDamagePytorch` → 200. Revisar la ruta del peso
de daño: `segmentar_danos_pytorch.py` apunta a `modelo/unet_danos_petroglifos_v3.pth` (minúscula)
pero el archivo está en `Models/` (mayúscula); ajustar ruta/symlink si da 503.

1. `curl -F "file=@imagen.jpg" http://localhost:8001/segmentDamagePytorch` → 200 con `mask_image`.
2. Backend: `pyenv activate ia`; reiniciar `uvicorn adapters.inbound.api.main:app`.
3. Pipeline: `POST /classify/sync` (o `/classify/upload`) con una imagen muy dañada y
   `conservation_status=Regular` (para aislar el camino automático). En logs esperar:
   `a2_deterioration_api` con `damage_figure_percent` alto y `deterioration=true`, luego
   `router_decision route=a5_reconstructor`. Con imagen sana → `route=a3_comparator`.
4. Verificar en la ficha (`/fichas/{id}/json`) que `damage_figure_percent` aparece.
5. Ajustar `DAMAGE_RECONSTRUCTION_THRESHOLD` en `.env` según pruebas reales.

## Archivos a modificar
- `config/settings.py` — umbral configurable.
- `agents/a2_detector/agent.py` — consultar `/segmentDamagePytorch` + daño figura-relativo (núcleo).
- `agents/a6_documentor/agent.py` — (recomendado) mostrar % de daño.
- `orchestrator/state/graph_state.py` — (opcional) declarar claves nuevas.
- `.env.example` / `.env` — documentar la variable.

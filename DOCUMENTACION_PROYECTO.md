# Sistema Multiagente de Petroglifos — Documentación de principio a fin

Clasificación taxonómica y reconstrucción de petroglifos andinos colombianos.
Arquitectura hexagonal + LangGraph (orquestación multiagente) + RAG (Gemini + pgvector) +
servicio externo de visión (Keras/PyTorch/LaMa).

---

## 1. Arquitectura general

El sistema son **dos servicios HTTP separados**:

1. **Backend** (este repo, `BackendPetroglifos/`) — orquesta el pipeline de 6 agentes (A1–A6),
   expone la API REST y el bot de Telegram. Puerto **8000**.
2. **Servicio de reconstrucción** (repo aparte, `recontrucción/petroglyph-service-reconstruction-api/`)
   — modelos de visión pesados (segmentación, detección de daño, inpainting). Puerto **8001**.

Infra externa: **PostgreSQL + pgvector** (Supabase, remoto) y **Redis** (remoto, broker de Celery).

```
Usuario (Telegram / API)
        │
        ▼
  Backend :8000  ── Celery worker ──► Pipeline LangGraph A1→A6
        │                                   │  (llama por HTTP)
        │                                   ▼
        │                          Servicio Keras :8001
        │                          (segmentación / daño / LaMa)
        ▼
  PostgreSQL+pgvector (RAG, grafo, fichas)   Gemini API (LLM + embeddings)
```

---

## 2. El pipeline paso a paso (flujo completo)

Orquestado con LangGraph en `orchestrator/PetroglyphOrchestrator.py`.

```
START
  │
  ▼
A1 Preprocesador ───────────── limpia la imagen (OpenCV)
  │
  ▼
A2 Detector ────────────────── YOLOv8-cls (clase) + deterioro (servicio Keras)
  │
  ▼
[ROUTER condicional]
  ├── motivos visibles Y sin recomendación de reconstrucción ──► A3
  └── deterioro / daño / sin motivos / criterio humano ────────► A5
                                                                  │
A5 Reconstructor ── inpainting LaMa (servicio Keras) ────────────┘
  │
  └──► vuelve a A2 (re-detecta sobre la imagen RECONSTRUIDA)
          │
          ▼
        [ROUTER] (ya hay imagen reconstruida) ──► A3
  │
  ▼
A3 Comparador ──────────────── EfficientNet-B0 + pgvector (similitud iconográfica) + grafo
  │
  ▼
A4 Analista Cultural ───────── RAG (pgvector) + Gemini (clasificación taxonómica)
  │
  ▼
A6 Documentador ────────────── Jinja2 + WeasyPrint → ficha ICANH (PDF + JSON)
  │
  ▼
END  → resultado al usuario (clasificación + ficha + imagen reconstruida si aplica)
```

**Importante (orden con reconstrucción):** cuando hay reconstrucción, el flujo es
`A2 → A5 → A2(re-detecta sobre la reconstruida) → A3 → A4 → A6`. La clasificación (A4)
**siempre corre al final**, sobre el estado posterior a la reconstrucción.

---

## 3. Qué hace cada agente

### A1 — Preprocesador (`agents/a1_preprocessor/agent.py`)
Tecnología: **OpenCV**. Prepara la imagen cruda:
1. Redimensiona si supera 2048 px.
2. Corrección de perspectiva (contorno mayor + transformación de 4 puntos).
3. Escala de grises + **CLAHE** (realce de contraste adaptativo).
4. Filtro bilateral (quita ruido sin perder bordes).
5. Normalización de iluminación.
Salida: `storage/preprocessed/...jpg`.

### A2 — Detector (`agents/a2_detector/agent.py`)
Dos funciones:
- **Detección de motivos** con **YOLOv8s-cls** (clasificación de imagen completa):
  devuelve la clase (`Antropomorfo` / `Zoomorfo` / `Geométrico`) y la confianza.
  Si la confianza supera `CONFIDENCE_THRESHOLD` (0.70) → `motifs_visible=True`.
  Si no hay modelo, cae a un heurístico OpenCV (contornos → formas geométricas).
- **Evaluación de deterioro/daño** llamando al servicio Keras:
  - `/segmentPetroglyph` → calidad de la segmentación de la figura.
  - `/segmentDamagePytorch` → máscara de daño; se cruza con la máscara de la figura para
    calcular `damage_figure_percent` = **daño relativo a la figura** (daño ∩ figura / área figura).
  - `deterioration_detected = (criterios de calidad)  OR  (daño_figura ≥ DAMAGE_RECONSTRUCTION_THRESHOLD)`.

A2 además combina con el **criterio humano** (estado de conservación que ingresa el usuario):
```
reconstruction_recommended = deterioration_detected  OR  human_reconstruction_recommended
```
donde `human_reconstruction_recommended` = conservación **Malo/Crítico** (score ≥ 0.75).

### Router (`_route_after_detection` en el orquestador)
- Si ya hay imagen reconstruida → A3.
- Si hay motivos visibles y NO se recomienda reconstrucción → A3.
- En otro caso (daño, criterio humano, o sin motivos) → A5.

### A3 — Comparador iconográfico (`agents/a3_comparator/agent.py`)
Tecnología: **EfficientNet-B0** (timm) + **pgvector**.
1. Extrae un embedding visual de 1280 dims de la imagen (reconstruida si existe).
2. Busca las 5 más similares en el corpus (umbral 0.60).
3. Actualiza el **grafo social** de sitios (aristas con similitud ≥ 0.70) y lo persiste en
   `site_graph_edges`.
Salida: `similarity_matches`.

### A4 — Analista Cultural (`agents/a4_analyst/agent.py`) — núcleo LLM
Tecnología: **RAG (pgvector) + Gemini**.
1. Recupera fragmentos arqueológicos relevantes del corpus (RAG).
2. Construye prompt (Jinja2) con el contexto + formas detectadas + similitudes.
3. **Gemini** clasifica → `{taxonomy, confidence, justification}`.
4. Segundo llamado a Gemini → descripción técnica detallada del petroglifo.
5. Calcula **RAG feedback** (consistencia descripción ↔ fuentes).
6. Persiste en BD (`llm_classifications`, `petroglyph_description_embeddings`, `prompt_logs`).
Si Gemini falla → **clasificador heurístico de respaldo** (por eso, si la API está caída o sin
cuota, la confianza puede quedar fija ~0.54).

### A5 — Reconstructor (`agents/a5_reconstructor/agent.py`)
Tecnología: servicio Keras externo (**U-Net + LaMa inpainting**).
- Solo actúa si hay deterioro/recomendación de reconstrucción.
- Si `GAN_MOCK_MODE=true` → copia la imagen (no reconstruye de verdad).
- Si `false` → llama al servicio con prioridad según severidad:
  `reconstructVisualAssisted` / `reconstructFull` / `reconstruct` (legacy).
Salida: `storage/reconstructed/...png` + diagnósticos. Tras A5 se vuelve a A2.

### A6 — Documentador (`agents/a6_documentor/agent.py`)
Tecnología: **Jinja2 + WeasyPrint** (fallback a matplotlib).
Consolida todo (taxonomía, confianza, justificación, descripción, diagnóstico de
segmentación/daño, criterio de conservación, similitudes) y genera la **ficha ICANH**:
`storage/fichas_icanh/{id}_ficha.pdf` y `.json`.

---

## 4. Modelos usados (resumen)

| Modelo | Dónde | Para qué | Tipo |
|--------|-------|----------|------|
| **YOLOv8s-cls** (`models/petroglifos_yolov8.pt`) | A2 (backend) | Clasificar la figura (antropomorfo/zoomorfo/geométrico) | Clasificación imagen |
| **EfficientNet-B0** (timm, pretrained) | A3 (backend) | Embedding visual 1280-d para similitud | Extractor de features |
| **Gemini 2.0 Flash** (API) | A4 (backend) | Clasificación taxonómica + descripción | LLM |
| **Gemini embeddings** (API) | A4/RAG (backend) | Embeddings de texto (1280-d) para RAG | Embeddings |
| **U-Net Keras** (`modelo/mejor_modelo.keras`) | servicio :8001 | Segmentar la figura del petroglifo | Segmentación |
| **U-Net PyTorch** (`modelo/unet_danos_petroglifos_v3.pth`) | servicio :8001 | Detectar zonas de **daño** | Segmentación |
| **LaMa** (inpainting) | servicio :8001 | Rellenar/reconstruir zonas dañadas | Inpainting (GAN) |

---

## 5. El subsistema de reconstrucción en detalle

El servicio externo (`recontrucción/petroglyph-service-reconstruction-api/`) expone:

- `POST /segmentPetroglyph` — U-Net Keras: segmenta la figura. Devuelve `area_percent`,
  `validation_score`, `segmentation_status`, `validation_warnings`, `mask_image` (base64).
- `POST /segmentDamagePytorch` — U-Net PyTorch: máscara de **daño** (`mask_image` base64).
- `POST /reconstructFull` — pipeline completo: daño + LaMa inpainting + re-segmentación.
  Devuelve imágenes base64 por etapa + `damage_percent`, `damage_pixel_count`.
- `POST /reconstructVisualAssisted` — variante asistida para daño severo.
- `POST /reconstruct` — legacy (devuelve PNG directo).

**Decisión automática de reconstruir (clave del proyecto):**
A2 NO se basa solo en la calidad de la segmentación; cruza la **máscara de daño** con la
**máscara de la figura** y calcula qué fracción de la figura está dañada
(`damage_figure_percent`). Si supera `DAMAGE_RECONSTRUCTION_THRESHOLD` (default 0.70) → reconstruye.

> Nota empírica: el modelo de daño rara vez marca >30 % de la figura aunque a ojo se vea muy
> deteriorada (los surcos erosionados son finos). Si quieres que la vía automática dispare más
> seguido, baja el umbral en `.env` (p. ej. `DAMAGE_RECONSTRUCTION_THRESHOLD=0.25`).

**Doble vía de reconstrucción:**
- **Automática** — el modelo de daño (`damage_figure_percent ≥ umbral`).
- **Humana** — el usuario marca estado de conservación **Malo/Crítico**.
Cualquiera de las dos dispara A5 (es un OR).

---

## 6. Cómo arrancar TODO el proyecto

Todo corre en el entorno **`pyenv activate ia`** (Python 3.11). DB y Redis son remotos
(no se necesita Docker). Usa **4 terminales** (una por servicio).

> Arranca en orden: **1 → 2 → 3 → 4**.
> Para probar solo por API (sin Telegram) basta con **1 + 2** usando `POST /classify/sync`.

### 1️⃣ Servicio de reconstrucción Keras — puerto 8001
```bash
cd "/home/leonardo/Documentos/Uni/Inteligencia/Proyecto/recontrucción/petroglyph-service-reconstruction-api"
pyenv activate ia
uvicorn service:app --host 0.0.0.0 --port 8001
```
Espera en consola: "Modelo Keras cargado" y "Modelo PyTorch cargado".

### 2️⃣ Backend API — puerto 8000
```bash
cd /home/leonardo/Documentos/Uni/Inteligencia/Proyecto/BackendPetroglifos
pyenv activate ia
uvicorn adapters.inbound.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3️⃣ Worker de Celery (ejecuta el pipeline para el bot)
```bash
cd /home/leonardo/Documentos/Uni/Inteligencia/Proyecto/BackendPetroglifos
pyenv activate ia
celery -A infrastructure.messaging.celery_app worker --loglevel=info -c 2
```
Espera: "celery@... ready."

### 4️⃣ Bot de Telegram
```bash
cd /home/leonardo/Documentos/Uni/Inteligencia/Proyecto/BackendPetroglifos
pyenv activate ia
python -m adapters.inbound.telegram_bot.bot
```
Espera: "Application started".

### Verificación (otra terminal)
```bash
curl -s -o /dev/null -w "recon 8001: %{http_code}\n" http://localhost:8001/docs
curl -s http://localhost:8000/health
```

---

## 7. Cómo probar

### Por Telegram
Envía una foto al bot → responde sitio → municipio → estado de conservación. Recibirás:
- Estado de reconstrucción (si aplica),
- la ficha ICANH (PDF),
- la imagen reconstruida (si aplica),
- el mensaje de clasificación (taxonomía + confianza + justificación).

### Por API (síncrono, sin Celery/Telegram)
```bash
curl -X POST http://localhost:8000/classify/sync \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/ruta/a/imagen.jpg","site":"Piedras del Tunjo","municipality":"Soacha","conservation_status":"Regular"}'
```

### Logs útiles
```
A1→A6, damage_figure_percent, router → worker (terminal 3)
clasificación / PDF / imagen enviada → bot (terminal 4)
segmentación / daño / LaMa → servicio (terminal 1)
```

---

## 8. Configuración relevante (`.env`)

| Variable | Valor actual | Qué hace |
|----------|--------------|----------|
| `GEMINI_MODEL` | `gemini-2.0-flash` | Modelo LLM de A4 (barato, sin "thinking") |
| `GEMINI_MODEL_LITE` | `gemini-2.0-flash-lite` | Variante ligera |
| `EMBEDDING_MODEL` | (embeddings 1280-d) | Embeddings de texto para RAG |
| `CONFIDENCE_THRESHOLD` | `0.70` | Umbral de confianza de YOLO en A2 |
| `DAMAGE_RECONSTRUCTION_THRESHOLD` | `0.70` | Fracción de figura dañada que fuerza reconstrucción automática |
| `GAN_MOCK_MODE` | `false` | `true` = A5 solo copia la imagen; `false` = reconstrucción real |
| `RECONSTRUCTION_API_BASE_URL` | `http://localhost:8001` | URL del servicio Keras |

---

## 9. Problemas conocidos / notas

- **Confianza fija en 54 %**: significa que A4 cayó al heurístico (Gemini no respondió). Causas
  típicas: nombre de modelo inválido, **cuota agotada (HTTP 429)**, o JSON cortado. Con
  `gemini-2.0-flash` + JSON mode esto está mitigado; si ves 429, espera a que reponga la cuota.
- **Modelos 1.5 retirados**: usar IDs `gemini-2.0-flash` / `gemini-2.5-flash` (no "Gemini 1.5 Flash").
- **`basicsr` del servicio**: es dependencia muerta (no se importa); se omite al instalar.
- **GPU**: si el driver NVIDIA es viejo, todo corre en CPU (más lento, sin afectar resultados).
- **Carpeta de modelos del servicio**: debe llamarse `modelo/` (minúscula), que es donde el
  código del servicio busca los pesos.

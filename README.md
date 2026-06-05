# Petroglifos LLM — Módulo de Clasificación Taxonómica

Sistema multiagente para clasificación taxonómica de petroglifos andinos colombianos.
Arquitectura hexagonal + LangGraph + RAG (Gemini + pgvector).

## Agentes
| ID | Nombre | Tecnología |
|----|--------|------------|
| A1 | Preprocesador | OpenCV |
| A2 | Detector de motivos | YOLOv8 |
| A3 | Comparador iconográfico | EfficientNet-B0 + pgvector |
| A4 | Analista Cultural (LLM) | RAG + Gemini 1.5 Flash |
| A5 | Reconstructor GAN | GAN + validacion de dano (solo deterioro) |
| A6 | Documentador ICANH | Jinja2 + WeasyPrint |

## Inicio rápido
```bash
cp .env.example .env
docker compose up -d db
uv sync
uvicorn adapters.inbound.api.main:app --reload
```

## Evaluacion tecnica
El sistema guarda ejecuciones en `storage/metrics/runs.jsonl` y refresca `storage/metrics/report.json` en cada corrida para medir:

- tasa de exito autonomo
- tiempo de produccion de ficha por sitio
- FID de reconstrucciones por lote de corridas
- IoU de deteccion cuando la corrida incluye cajas ground truth

Para evaluar las metricas tecnicas usa:

```bash
python -m scripts.evaluate_technical_metrics \
  --detections-gt data/detections_gt.csv \
  --detections-pred data/detections_pred.csv \
  --real-dir storage/reference_images \
  --generated-dir storage/reconstructed \
  --runs-jsonl storage/metrics/runs.jsonl \
  --output storage/metrics/report.json
```

Formato esperado para deteccion:

- `image_id,x1,y1,x2,y2,score`
- `score` es opcional en ground truth y recomendable en predicciones

Notas:

- IoU objetivo: mayor o igual a `0.82`
- FID objetivo: menor a `45`
- Exito autonomo objetivo: mayor o igual a `0.83`
- Tiempo de ficha objetivo: menor a `45` minutos
- Si no hay `ground_truth_boxes` en las corridas, el reporte marca IoU como omitido en lugar de fallar

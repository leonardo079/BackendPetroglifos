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
| A5 | Reconstructor GAN | GAN (solo deterioro) |
| A6 | Documentador ICANH | Jinja2 + WeasyPrint |

## Inicio rápido
```bash
cp .env.example .env
docker compose up -d db
uv sync
uvicorn adapters.inbound.api.main:app --reload
```

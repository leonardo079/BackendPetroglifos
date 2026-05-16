"""Punto de entrada FastAPI del sistema."""
from fastapi import FastAPI

app = FastAPI(title="Petroglifos LLM API", version="0.1.0")

@app.post("/classify")
async def classify_petroglyph(payload: dict):
    """Recibe datos del agente de visión y retorna clasificación taxonómica."""
    raise NotImplementedError

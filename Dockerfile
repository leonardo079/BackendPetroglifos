FROM python:3.11-slim

# ── Dependencias del sistema ───────────────────────────────────────────────
# poppler-utils  → pdf2image (OCR de PDFs escaneados)
# tesseract-ocr  → pytesseract
# libgomp1       → PyTorch (OpenMP)
# libgl1         → OpenCV
# weasyprint deps: libpango, libcairo, libgdk-pixbuf
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-spa \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ──────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python ────────────────────────────────────────────────────
# Copiar primero solo los archivos de dependencias para aprovechar la caché
COPY pyproject.toml ./

# Instalar pip actualizado y el proyecto con todas sus dependencias
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]"

# ── Código fuente ──────────────────────────────────────────────────────────
COPY . .

# ── Directorios de almacenamiento ─────────────────────────────────────────
RUN mkdir -p storage/uploads storage/fichas_icanh storage/graphs storage/reference_images

# ── Usuario no-root (seguridad) ────────────────────────────────────────────
RUN useradd -m -u 1000 petro && chown -R petro:petro /app
USER petro

# Puerto por defecto (solo para documentación; docker-compose lo mapea)
EXPOSE 8000

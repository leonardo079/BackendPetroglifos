"""
Configuración centralizada de structlog para todo el sistema.

Llamar `configure_logging()` UNA SOLA VEZ al arrancar la aplicación
(en main.py de FastAPI, en el worker Celery y en el bot de Telegram).
"""
from __future__ import annotations
import logging
import sys
import structlog
from config.settings import settings


def configure_logging() -> None:
    """
    Configura structlog con:
    - JSON en producción (parseable por Loki / CloudWatch)
    - Texto con colores en desarrollo
    - Nivel de log tomado de settings.log_level
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configurar logging estándar de Python (para librerías externas)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Silenciar loggers ruidosos
    for noisy in ("uvicorn.access", "httpx", "asyncio", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.env == "development":
        # Texto coloreado para desarrollo local
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON para producción
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
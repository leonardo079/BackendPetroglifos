"""
Aplicación Celery para procesamiento asíncrono de petroglifos.

El worker se lanza con:
    celery -A infrastructure.messaging.celery_app worker --loglevel=info -c 2
"""
from __future__ import annotations
from celery import Celery
from config.settings import settings

celery_app = Celery(
    "petroglifos",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["infrastructure.messaging.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Bogota",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Reintentos automáticos ante fallos de broker
    broker_connection_retry_on_startup=True,
    # Tiempo máximo por tarea (45 min según métricas del proyecto)
    task_soft_time_limit=settings.max_record_sheet_minutes * 60,
    task_time_limit=settings.max_record_sheet_minutes * 60 + 60,
    # Rutas por defecto
    task_default_queue="petroglifos",
    task_queues={
        "petroglifos": {"exchange": "petroglifos", "routing_key": "petroglifos"},
        "ingestion": {"exchange": "ingestion", "routing_key": "ingestion"},
    },
)
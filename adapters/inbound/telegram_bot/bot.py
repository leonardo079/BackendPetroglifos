"""
Bot de Telegram del sistema de Petroglifos — punto de entrada.

Ejecución local:
    python -m adapters.inbound.telegram_bot.bot

Con Docker (docker-compose):
    command: python -m adapters.inbound.telegram_bot.bot

Variables de entorno requeridas:
    TELEGRAM_BOT_TOKEN  — token del bot obtenido desde @BotFather
    API_BASE_URL        — URL base de la API FastAPI (default: http://localhost:8000)
                          En Docker, usar: http://api:8000
"""
from __future__ import annotations

import structlog
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config.settings import settings
from infrastructure.observability.logging_config import configure_logging
from adapters.inbound.telegram_bot.handlers.classify import (
    SITE_NAME,
    MUNICIPALITY,
    CONSERVATION,
    handle_photo,
    handle_site_name,
    handle_municipality,
    handle_conservation,
    cancel,
)
from adapters.inbound.telegram_bot.handlers.commands import (
    start,
    ayuda,
    estado,
    sitios,
    grafo,
)

configure_logging()
log = structlog.get_logger(__name__)


def build_application() -> Application:
    """Construye y configura la aplicación del bot con todos sus handlers."""
    if not settings.telegram_bot_token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN no está configurado. "
            "Añádelo en el archivo .env o como variable de entorno."
        )

    app = Application.builder().token(settings.telegram_bot_token).build()

    # ── Comandos simples ──────────────────────────────────────────────────────
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ayuda", ayuda))
    app.add_handler(CommandHandler("help", ayuda))
    app.add_handler(CommandHandler("estado", estado))
    app.add_handler(CommandHandler("sitios", sitios))
    app.add_handler(CommandHandler("grafo", grafo))

    # ── Conversación: foto → metadatos → lanzar clasificación ─────────────────
    # El ConversationHandler gestiona el estado por usuario+chat para que
    # múltiples usuarios puedan clasificar simultáneamente sin interferencia.
    classify_conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo),
        ],
        states={
            SITE_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_site_name),
            ],
            MUNICIPALITY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_municipality),
            ],
            CONSERVATION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_conservation),
            ],
        },
        fallbacks=[CommandHandler("cancelar", cancel)],
        per_user=True,
        per_chat=True,
        allow_reentry=True,
    )
    app.add_handler(classify_conv)

    log.info(
        "telegram_bot_configured",
        commands=["start", "ayuda", "estado", "sitios", "grafo", "cancelar"],
        api_url=settings.api_base_url,
    )
    return app


def main() -> None:
    """Inicia el bot en modo polling (long-polling de la API de Telegram)."""
    app = build_application()
    log.info("telegram_bot_starting", mode="polling")
    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,  # Ignorar mensajes acumulados mientras el bot estaba offline
    )


if __name__ == "__main__":
    main()

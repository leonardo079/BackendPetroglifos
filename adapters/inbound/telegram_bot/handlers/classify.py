"""
Handler de conversaciÃ³n para clasificaciÃ³n de petroglifos vÃ­a fotografÃ­a.

Flujo de la conversaciÃ³n:
    [foto] â†’ SITE_NAME â†’ MUNICIPALITY â†’ CONSERVATION â†’ [lanza pipeline async]

Una vez recopilados los metadatos, la imagen se encola a travÃ©s de la API REST
(POST /classify/upload) y el bot hace polling del resultado (GET /tasks/{id})
en segundo plano, editando el mensaje de estado cuando termina.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import httpx
import structlog
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes, ConversationHandler

from config.settings import settings
from core.domain.site_normalization import canonicalize_municipality, canonicalize_site_name
from adapters.inbound.telegram_bot.messages import (
    ASK_SITE_NAME,
    ASK_MUNICIPALITY,
    ASK_CONSERVATION,
    PROCESSING,
    RESULT_OK,
    VALIDATION_WARNING,
    SIMILARITY_BLOCK,
    RESULT_PDF_NOTE,
    CANCELLED,
    INVALID_CONSERVATION,
    ERROR_API,
    ERROR_ENQUEUE,
    ERROR_CLASSIFICATION,
    ERROR_TIMEOUT,
)

log = structlog.get_logger(__name__)

# Directorio para fotos descargadas desde Telegram
UPLOAD_DIR = Path("storage/uploads/telegram")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Estados de la conversaciÃ³n (exportados para el ConversationHandler en bot.py)
SITE_NAME, MUNICIPALITY, CONSERVATION = range(3)

_API = settings.api_base_url.rstrip("/")

_CONSERVATION_MAP: dict[str, str] = {
    "bueno": "Bueno",
    "regular": "Regular",
    "malo": "Malo",
    "crÃ­tico": "CrÃ­tico",
    "critico": "CrÃ­tico",
}

# ConfiguraciÃ³n del polling
_POLL_INTERVAL_SECONDS = 10
_MAX_POLL_SECONDS = 300  # 5 minutos


# â”€â”€ Pasos de la conversaciÃ³n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Punto de entrada: el usuario envÃ­a una fotografÃ­a.
    Guarda el file_id en user_data y pide el nombre del sitio.
    """
    photo = update.message.photo[-1]  # Tomar la versiÃ³n de mayor resoluciÃ³n
    context.user_data.clear()
    context.user_data["photo_file_id"] = photo.file_id
    context.user_data["task_id"] = str(uuid.uuid4())

    log.info(
        "bot_photo_received",
        user_id=update.effective_user.id,
        task_id=context.user_data["task_id"],
    )
    await update.message.reply_text(ASK_SITE_NAME, parse_mode=ParseMode.HTML)
    return SITE_NAME


async def handle_site_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Guarda el nombre del sitio y pide el municipio."""
    context.user_data["site"] = canonicalize_site_name(update.message.text.strip())
    await update.message.reply_text(ASK_MUNICIPALITY, parse_mode=ParseMode.HTML)
    return MUNICIPALITY


async def handle_municipality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Guarda el municipio y pide el estado de conservaciÃ³n."""
    context.user_data["municipality"] = canonicalize_municipality(update.message.text.strip())
    await update.message.reply_text(ASK_CONSERVATION, parse_mode=ParseMode.HTML)
    return CONSERVATION


async def handle_conservation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Valida el estado de conservaciÃ³n, descarga la foto de Telegram,
    envÃ­a mensaje de "procesando" y lanza el pipeline en segundo plano.
    """
    raw = update.message.text.strip().lower()
    conservation = _CONSERVATION_MAP.get(raw)
    if not conservation:
        await update.message.reply_text(INVALID_CONSERVATION, parse_mode=ParseMode.HTML)
        return CONSERVATION  # Repetir este estado hasta que sea vÃ¡lido

    task_id: str = context.user_data["task_id"]
    file_id: str = context.user_data["photo_file_id"]
    site: str = context.user_data.get("site", "Sin nombre")
    municipality: str = context.user_data.get("municipality", "")

    # Descargar la foto desde los servidores de Telegram
    tg_file = await context.bot.get_file(file_id)
    dest = UPLOAD_DIR / f"{task_id}.jpg"
    await tg_file.download_to_drive(str(dest))

    # Enviar mensaje inicial de estado
    status_msg = await update.message.reply_text(
        PROCESSING.format(task_id=task_id),
        parse_mode=ParseMode.HTML,
    )

    # Lanzar el pipeline en segundo plano (no bloquea al usuario)
    context.application.create_task(
        _enqueue_and_poll(
            app=context.application,
            chat_id=update.effective_chat.id,
            status_message_id=status_msg.message_id,
            task_id=task_id,
            image_path=str(dest),
            site=site,
            municipality=municipality,
            conservation_status=conservation,
        ),
        update=update,
    )

    context.user_data.clear()
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancela la conversaciÃ³n en curso."""
    context.user_data.clear()
    await update.message.reply_text(CANCELLED, parse_mode=ParseMode.HTML)
    return ConversationHandler.END


# â”€â”€ Pipeline asÃ­ncrono en segundo plano â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def _enqueue_and_poll(
    app,
    chat_id: int,
    status_message_id: int,
    task_id: str,
    image_path: str,
    site: str,
    municipality: str,
    conservation_status: str,
) -> None:
    """
    Encola la clasificaciÃ³n vÃ­a POST /classify/upload y hace polling de
    GET /tasks/{task_id} hasta obtener el resultado o agotar el tiempo.

    Al terminar, edita el mensaje de estado con el resultado y adjunta el PDF.
    """
    # â”€â”€ 1. Encolar tarea â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    api_task_id = task_id
    try:
        img_path = Path(image_path)
        async with httpx.AsyncClient(timeout=30.0) as client:
            with img_path.open("rb") as f:
                resp = await client.post(
                    f"{_API}/classify/upload",
                    data={
                        "site": site,
                        "municipality": municipality,
                        "conservation_status": conservation_status,
                    },
                    files={"file": (img_path.name, f, "image/jpeg")},
                )
            resp.raise_for_status()
            api_task_id = resp.json().get("task_id", task_id)
    except httpx.RequestError:
        log.error("bot_enqueue_connection_error", task_id=task_id)
        await app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text=ERROR_API,
            parse_mode=ParseMode.HTML,
        )
        return
    except httpx.HTTPStatusError as exc:
        log.error("bot_enqueue_http_error", task_id=task_id, status=exc.response.status_code)
        await app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text=ERROR_ENQUEUE,
            parse_mode=ParseMode.HTML,
        )
        return

    log.info("bot_task_enqueued", task_id=api_task_id, site=site)

    # â”€â”€ 2. Polling del resultado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elapsed = 0
    while elapsed < _MAX_POLL_SECONDS:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        elapsed += _POLL_INTERVAL_SECONDS

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                poll_resp = await client.get(f"{_API}/tasks/{api_task_id}")
                poll_resp.raise_for_status()
                data = poll_resp.json()
        except (httpx.RequestError, httpx.HTTPStatusError):
            # Error transitorio â€” continuar intentando
            continue

        celery_state = data.get("celery_state", "")

        if celery_state == "SUCCESS":
            result = data.get("result") or {}
            if not isinstance(result, dict):
                log.error(
                    "bot_task_success_without_payload",
                    task_id=api_task_id,
                    payload_type=type(result).__name__,
                    payload_preview=str(result)[:200],
                )
                await app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text=ERROR_CLASSIFICATION.format(task_id=api_task_id),
                    parse_mode=ParseMode.HTML,
                )
                return

            if not result:
                log.error("bot_task_success_empty_result", task_id=api_task_id)
                await app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text=ERROR_CLASSIFICATION.format(task_id=api_task_id),
                    parse_mode=ParseMode.HTML,
                )
                return

            await _send_result(
                app, chat_id, status_message_id,
                api_task_id, result, site, municipality,
            )
            return

        if celery_state in ("FAILURE", "REVOKED"):
            log.error("bot_task_failed", task_id=api_task_id, state=celery_state)
            await app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text=ERROR_CLASSIFICATION.format(task_id=api_task_id),
                parse_mode=ParseMode.HTML,
            )
            return

    # â”€â”€ Timeout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log.warning("bot_task_timeout", task_id=api_task_id, elapsed_s=elapsed)
    await app.bot.edit_message_text(
        chat_id=chat_id,
        message_id=status_message_id,
        text=ERROR_TIMEOUT.format(task_id=api_task_id),
        parse_mode=ParseMode.HTML,
    )


async def _send_result(
    app,
    chat_id: int,
    status_message_id: int,
    task_id: str,
    result: dict,
    site: str,
    municipality: str,
) -> None:
    """Formatea el resultado de la clasificaciÃ³n y lo envÃ­a al usuario."""
    if not isinstance(result, dict):
        log.error(
            "bot_send_result_invalid_payload",
            task_id=task_id,
            payload_type=type(result).__name__,
            payload_preview=str(result)[:200],
        )
        await app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text=ERROR_CLASSIFICATION.format(task_id=task_id),
            parse_mode=ParseMode.HTML,
        )
        return

    classification = result.get("classification") or {}
    if not isinstance(classification, dict):
        log.error(
            "bot_send_result_invalid_classification",
            task_id=task_id,
            payload_type=type(classification).__name__,
            payload_preview=str(classification)[:200],
        )
        await app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text=ERROR_CLASSIFICATION.format(task_id=task_id),
            parse_mode=ParseMode.HTML,
        )
        return

    taxonomy = classification.get("taxonomy", "Indeterminado")
    confidence_raw = float(classification.get("confidence", 0.0))
    confidence = round(confidence_raw * 100, 1)
    justification = classification.get("justification", "Sin informaciÃ³n.")
    requires_validation = classification.get("requires_validation", True)

    # Truncar justificaciÃ³n muy larga para respetar el lÃ­mite de Telegram (4096 chars)
    if len(justification) > 500:
        justification = justification[:497] + "â€¦"

    validation_flag = VALIDATION_WARNING if requires_validation else ""
    text = RESULT_OK.format(
        site=site or "Sin nombre",
        municipality=municipality or "â€”",
        taxonomy=taxonomy,
        confidence=confidence,
        validation_flag=validation_flag,
        justification=justification,
    )

    # AÃ±adir similitudes iconogrÃ¡ficas (mÃ¡ximo 3 para no exceder el lÃ­mite)
    similarity_matches = classification.get("similarity_matches", [])
    if similarity_matches:
        match_lines = "\n".join(
            f"  â€¢ <b>{m['site_name']}</b> ({m.get('municipality', 'â€”')}) "
            f"â€” {round(m['similarity_score'] * 100, 1)}% [{m.get('taxonomy', '?')}]"
            for m in similarity_matches[:3]
        )
        text += SIMILARITY_BLOCK.format(matches=match_lines)

    # En este punto 'text' = clasificación (RESULT_OK + similitudes iconográficas).
    classification_text = text

    pdf_local_path = result.get("icanh_pdf_url", "")
    reconstructed_image_path = result.get("reconstructed_image_path", "")
    reconstruction_assessment = result.get("reconstruction_assessment", {}) or {}
    conservation_status = reconstruction_assessment.get("conservation_status", "Regular")
    conservation_score = float(reconstruction_assessment.get("conservation_score", 0.33) or 0.33)
    human_reconstruction = bool(reconstruction_assessment.get("human_reconstruction_recommended", False))
    model_reconstruction = bool(reconstruction_assessment.get("model_deterioration_detected", False))

    # Bloque informativo de reconstrucción (estado + señales).
    if reconstructed_image_path:
        recon_text = (
            "🛠️ <b>Reconstrucción realizada</b>\n"
            "Se generó una imagen reconstruida porque la evaluación combinada indicó intervención.\n"
            f"<b>Estado de conservación:</b> {conservation_status} ({round(conservation_score * 100, 1)}%)\n"
            f"<b>Señal humana:</b> {'recomendada' if human_reconstruction else 'no prioritaria'}\n"
            f"<b>Señal automática:</b> {'deterioro detectado' if model_reconstruction else 'sin deterioro crítico'}\n"
            "La imagen reconstruida y la ficha se adjuntan a continuación."
        )
    else:
        recon_text = (
            "<b>Reconstrucción:</b>\n"
            "No se realizó reconstrucción.\n"
            f"<b>Estado de conservación:</b> {conservation_status} ({round(conservation_score * 100, 1)}%)\n"
            f"<b>Señal humana:</b> {'recomendada' if human_reconstruction else 'no prioritaria'}\n"
            f"<b>Señal automática:</b> {'deterioro detectado' if model_reconstruction else 'sin deterioro crítico'}\n"
            "El análisis se trabajó sobre la imagen preprocesada."
        )

    pdf_note = RESULT_PDF_NOTE if pdf_local_path else ""

    if reconstructed_image_path:
        # CON reconstrucción: el mensaje editado muestra el estado de reconstrucción;
        # la clasificación se envía al FINAL, después de la imagen reconstruida.
        await app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text=recon_text,
            parse_mode=ParseMode.HTML,
        )
    else:
        # SIN reconstrucción: clasificación + estado en el mismo mensaje editado.
        await app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text=f"{classification_text}\n\n{recon_text}{pdf_note}",
            parse_mode=ParseMode.HTML,
        )

    # Adjuntar el PDF generado por el documentador
    if pdf_local_path:
        pdf_path = Path(pdf_local_path)
        if pdf_path.exists():
            try:
                with pdf_path.open("rb") as f:
                    await app.bot.send_document(
                        chat_id=chat_id,
                        document=f,
                        filename=f"ficha_icanh_{task_id}.pdf",
                        caption="Ficha ICANH generada automaticamente.",
                )
                log.info("bot_pdf_sent", task_id=task_id)
            except Exception as exc:
                log.warning("bot_pdf_send_error", task_id=task_id, error=str(exc))

    if reconstructed_image_path:
        reconstructed_path = Path(reconstructed_image_path)
        if reconstructed_path.exists():
            try:
                with reconstructed_path.open("rb") as f:
                    await app.bot.send_photo(
                        chat_id=chat_id,
                        photo=f,
                        caption="Imagen reconstruida generada automaticamente.",
                    )
                log.info("bot_reconstructed_image_sent", task_id=task_id)
            except Exception as exc:
                log.warning("bot_reconstructed_image_send_error", task_id=task_id, error=str(exc))

        # AL FINAL: el mensaje de clasificación (el mismo que en el flujo sin reconstrucción).
        try:
            await app.bot.send_message(
                chat_id=chat_id,
                text=f"{classification_text}{pdf_note}",
                parse_mode=ParseMode.HTML,
            )
            log.info("bot_classification_sent", task_id=task_id)
        except Exception as exc:
            log.warning("bot_classification_send_error", task_id=task_id, error=str(exc))

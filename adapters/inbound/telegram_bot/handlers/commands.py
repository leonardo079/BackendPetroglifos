"""
Handlers de comandos generales del bot de Petroglifos.

Comandos implementados:
    /start  — bienvenida
    /ayuda  — instrucciones de uso
    /estado — estado de una tarea Celery
    /sitios — lista de sitios rupestres registrados
    /grafo  — estadísticas y visualización del grafo social
"""
from __future__ import annotations

from io import BytesIO

import httpx
import structlog
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from config.settings import settings
from adapters.inbound.telegram_bot.messages import (
    WELCOME,
    HELP,
    TASK_STATUS,
    TASK_DONE_DETAIL,
    SITE_HEADER,
    SITE_ITEM,
    SITE_MORE,
    NO_SITES,
    GRAPH_SUMMARY,
    GRAPH_EMPTY,
    GRAPH_IMAGE_CAPTION,
    GRAPH_HTML_CAPTION,
    ERROR_API,
    ERROR_TASK_NOT_FOUND,
    ERROR_GRAPH_IMAGE,
    ERROR_GRAPH_EXPORT,
)

log = structlog.get_logger(__name__)

_API = settings.api_base_url.rstrip("/")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envía el mensaje de bienvenida."""
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML)


async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envía las instrucciones de uso."""
    await update.message.reply_text(HELP, parse_mode=ParseMode.HTML)


async def estado(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Consulta el estado de una tarea Celery.
    Uso: /estado <task_id>
    """
    if not context.args:
        await update.message.reply_text(
            "Uso: <code>/estado &lt;task_id&gt;</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    task_id = context.args[0].strip()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_API}/tasks/{task_id}")
            if resp.status_code == 404:
                await update.message.reply_text(
                    ERROR_TASK_NOT_FOUND.format(task_id=task_id),
                    parse_mode=ParseMode.HTML,
                )
                return
            resp.raise_for_status()
            data = resp.json()
    except httpx.RequestError:
        await update.message.reply_text(ERROR_API, parse_mode=ParseMode.HTML)
        return

    state = data.get("celery_state", "UNKNOWN")
    details = ""
    result = data.get("result")
    if result and state == "SUCCESS":
        classification = result.get("classification", {})
        taxonomy = classification.get("taxonomy", "—")
        confidence_raw = float(classification.get("confidence", 0.0))
        details = TASK_DONE_DETAIL.format(
            taxonomy=taxonomy,
            confidence=round(confidence_raw * 100, 1),
        )

    await update.message.reply_text(
        TASK_STATUS.format(task_id=task_id, state=state, details=details),
        parse_mode=ParseMode.HTML,
    )


async def sitios(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Lista los sitios rupestres registrados.
    Uso: /sitios [departamento]  — el departamento es un filtro opcional.
    """
    params: dict[str, str] = {}
    if context.args:
        params["department"] = " ".join(context.args)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_API}/sites", params=params)
            resp.raise_for_status()
            sites = resp.json()
    except httpx.RequestError:
        await update.message.reply_text(ERROR_API, parse_mode=ParseMode.HTML)
        return

    if not sites:
        await update.message.reply_text(NO_SITES, parse_mode=ParseMode.HTML)
        return

    lines = [SITE_HEADER.format(count=len(sites))]
    for s in sites[:20]:  # Telegram limita mensajes a 4096 chars
        lines.append(
            SITE_ITEM.format(
                name=s["name"],
                municipality=s["municipality"],
                department=s["department"],
                taxonomy=s["dominant_taxonomy"],
                count=s["petroglyph_count"],
            )
        )
    if len(sites) > 20:
        lines.append(SITE_MORE.format(extra=len(sites) - 20))

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
    )


async def grafo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Muestra estadísticas del grafo social y envía la imagen estática y el HTML interactivo.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{_API}/graph")
            resp.raise_for_status()
            data = resp.json()
    except httpx.RequestError:
        await update.message.reply_text(ERROR_API, parse_mode=ParseMode.HTML)
        return

    summary = data.get("summary", {})
    nodes = summary.get("nodes", 0)
    edges = summary.get("edges", 0)

    if nodes == 0:
        await update.message.reply_text(GRAPH_EMPTY, parse_mode=ParseMode.HTML)
        return

    text = GRAPH_SUMMARY.format(
        nodes=nodes,
        edges=edges,
        density=f"{summary.get('density', 0.0):.4f}",
        avg_sim=f"{summary.get('avg_similarity', 0.0):.2%}",
        top_site=summary.get("most_central_site") or "—",
        communities=summary.get("communities", 0),
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # Intentar descargar y enviar primero la imagen estática del grafo
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            image_resp = await client.get(f"{_API}/graph/export/image")
            image_resp.raise_for_status()
            image_buf = BytesIO(image_resp.content)
            image_buf.name = "red_rupestre.png"
            await update.message.reply_photo(
                photo=image_buf,
                caption=GRAPH_IMAGE_CAPTION,
            )
    except (httpx.RequestError, httpx.HTTPStatusError):
        await update.message.reply_text(ERROR_GRAPH_IMAGE, parse_mode=ParseMode.HTML)

    # Luego enviar el HTML interactivo
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            html_resp = await client.get(f"{_API}/graph/export")
            html_resp.raise_for_status()
            buf = BytesIO(html_resp.content)
            await update.message.reply_document(
                document=buf,
                filename="red_rupestre.html",
                caption=GRAPH_HTML_CAPTION,
            )
    except (httpx.RequestError, httpx.HTTPStatusError):
        await update.message.reply_text(ERROR_GRAPH_EXPORT, parse_mode=ParseMode.HTML)

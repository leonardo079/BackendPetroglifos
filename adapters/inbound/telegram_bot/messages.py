"""Plantillas de texto para los mensajes del bot de Petroglifos."""
from __future__ import annotations

WELCOME = (
    "🪨 <b>Bot de Petroglifos — UPTC 2026</b>\n\n"
    "Soy el asistente de clasificación de arte rupestre andino colombiano.\n\n"
    "Envíame una <b>fotografía</b> de un petroglifo y recibirás:\n"
    "• Detección de motivos y formas\n"
    "• Clasificación taxonómica con contexto arqueológico\n"
    "• Ficha ICANH preliminar en PDF\n\n"
    "<b>Comandos disponibles:</b>\n"
    "/ayuda — instrucciones de uso\n"
    "/estado &lt;task_id&gt; — estado de una clasificación\n"
    "/sitios — sitios rupestres registrados\n"
    "/grafo — red de similitud iconográfica\n"
    "/cancelar — cancela la operación actual"
)

HELP = (
    "<b>¿Cómo clasificar un petroglifo?</b>\n\n"
    "1. Envía una <b>fotografía</b> del petroglifo\n"
    "2. Escribe el <b>nombre del sitio</b> cuando se te pida\n"
    "3. Escribe el <b>municipio</b>\n"
    "4. Indica el <b>estado de conservación</b>\n"
    "5. Espera la clasificación (~1–3 min)\n\n"
    "<b>Consejos para mejores resultados:</b>\n"
    "• Fotografía en luz natural difusa, evita sombras duras\n"
    "• Encuadra la roca completa cuando sea posible\n"
    "• Alta resolución mejora la detección de motivos\n\n"
    "<b>Comandos:</b>\n"
    "/estado &lt;task_id&gt; — consulta el estado de una tarea\n"
    "/sitios [departamento] — lista de sitios (filtro opcional)\n"
    "/grafo — red de similitud iconográfica entre sitios\n"
    "/cancelar — cancela la conversación actual"
)

ASK_SITE_NAME = (
    "📍 ¿Cuál es el <b>nombre del sitio</b> arqueológico?\n"
    "<i>Ej: Piedras del Tunjo, Villa de Leyva, Gámeza</i>"
)

ASK_MUNICIPALITY = "🏘️ ¿En qué <b>municipio</b> se encuentra el sitio?"

ASK_CONSERVATION = (
    "🪨 ¿Cuál es el <b>estado de conservación</b> del petroglifo?\n\n"
    "Responde con: <b>Bueno</b>, <b>Regular</b>, <b>Malo</b> o <b>Crítico</b>"
)

PROCESSING = (
    "⚙️ <b>Procesando clasificación…</b>\n\n"
    "🆔 Task ID: <code>{task_id}</code>\n\n"
    "El pipeline multiagente está analizando la imagen.\n"
    "Recibirás los resultados en este chat (~1–3 min).\n\n"
    "Consulta el estado en cualquier momento:\n"
    "<code>/estado {task_id}</code>"
)

RESULT_OK = (
    "✅ <b>Clasificación completada</b>\n\n"
    "🏛️ <b>Sitio:</b> {site}\n"
    "📍 <b>Municipio:</b> {municipality}\n"
    "🔬 <b>Taxonomía:</b> {taxonomy}\n"
    "📊 <b>Confianza:</b> {confidence}%\n"
    "{validation_flag}"
    "\n<b>Justificación:</b>\n<i>{justification}</i>"
)

VALIDATION_WARNING = "⚠️ <i>Requiere validación experta (confianza &lt; 70%)</i>\n"

SIMILARITY_BLOCK = "\n\n🔗 <b>Similitudes iconográficas:</b>\n{matches}"

RESULT_PDF_NOTE = "\n\n📄 Ficha ICANH adjunta."

TASK_STATUS = (
    "📋 <b>Estado de la tarea</b>\n\n"
    "🆔 <code>{task_id}</code>\n"
    "🔄 Estado: <b>{state}</b>\n"
    "{details}"
)

TASK_DONE_DETAIL = (
    "\n✅ <b>Completado</b>\n"
    "🔬 Taxonomía: {taxonomy}\n"
    "📊 Confianza: {confidence}%"
)

SITE_HEADER = "🗿 <b>Sitios rupestres registrados ({count})</b>\n"
SITE_ITEM = "• <b>{name}</b> — {municipality}, {department} | {taxonomy} ({count} petroglifos)"
SITE_MORE = "\n<i>... y {extra} sitios más.</i>"
NO_SITES = "ℹ️ No hay sitios rupestres registrados en el sistema aún."

GRAPH_SUMMARY = (
    "🕸️ <b>Red de similitud iconográfica</b>\n\n"
    "📍 <b>Nodos (sitios):</b> {nodes}\n"
    "🔗 <b>Aristas (conexiones):</b> {edges}\n"
    "📐 <b>Densidad:</b> {density}\n"
    "🔬 <b>Similitud promedio:</b> {avg_sim}\n"
    "🏆 <b>Sitio más central:</b> {top_site}\n"
    "🧩 <b>Comunidades iconográficas:</b> {communities}\n"
)

GRAPH_EMPTY = (
    "ℹ️ El grafo aún no tiene datos suficientes.\n"
    "Se construye automáticamente con cada clasificación procesada."
)

GRAPH_IMAGE_CAPTION = "🖼️ Imagen estática del mapa de grafo social."
GRAPH_HTML_CAPTION = "📊 Visualización interactiva del grafo iconográfico (PyVis)."

CANCELLED = "❌ Operación cancelada."

INVALID_CONSERVATION = (
    "❌ Valor no válido. Responde con:\n"
    "<b>Bueno</b>, <b>Regular</b>, <b>Malo</b> o <b>Crítico</b>"
)

ERROR_API = "❌ No se pudo conectar con el servidor. Intenta de nuevo en unos momentos."
ERROR_ENQUEUE = (
    "❌ Error al encolar la clasificación.\n"
    "Verifica que el servidor esté activo e intenta de nuevo."
)
ERROR_CLASSIFICATION = (
    "❌ Ocurrió un error durante la clasificación.\n\n"
    "🆔 <code>{task_id}</code>\n"
    "Contacta al administrador si el problema persiste."
)
ERROR_TIMEOUT = (
    "⏰ El tiempo de espera se agotó (5 min).\n\n"
    "Consulta el estado con:\n<code>/estado {task_id}</code>"
)
ERROR_TASK_NOT_FOUND = "❌ No se encontró la tarea <code>{task_id}</code>. Verifica el ID."
ERROR_GRAPH_EXPORT = "⚠️ No se pudo generar la visualización interactiva del grafo."
ERROR_GRAPH_IMAGE = "⚠️ No se pudo generar la imagen estática del grafo."

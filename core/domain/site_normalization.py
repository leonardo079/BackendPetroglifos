"""Normalización canónica de sitios rupestres para evitar duplicados."""
from __future__ import annotations

import math
from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class CanonicalSiteRecord:
    site_name: str
    municipality: str
    department: str
    # Coordenadas aproximadas del centroide del municipio.
    # Permiten calcular distancia geográfica para el reranking de A3.
    lat: float = 0.0
    lon: float = 0.0


_WORKSHOP_SITE_RECORDS: tuple[CanonicalSiteRecord, ...] = (
    # ── Boyacá ────────────────────────────────────────────────────────────────
    CanonicalSiteRecord("Villa de Leyva",     "Villa de Leyva",  "Boyacá",        5.6340,  -73.5253),
    CanonicalSiteRecord("Sáchica",            "Sáchica",         "Boyacá",        5.6628,  -73.5956),
    CanonicalSiteRecord("Gámeza",             "Gámeza",          "Boyacá",        5.8278,  -72.7833),
    CanonicalSiteRecord("Sogamoso",           "Sogamoso",        "Boyacá",        5.7139,  -72.9236),
    CanonicalSiteRecord("Tunja",              "Tunja",           "Boyacá",        5.5353,  -73.3678),
    CanonicalSiteRecord("Chivor",             "Chivor",          "Boyacá",        4.9972,  -73.2961),
    CanonicalSiteRecord("Soatá",              "Soatá",           "Boyacá",        6.3347,  -72.6875),
    CanonicalSiteRecord("Monguí",             "Monguí",          "Boyacá",        5.7211,  -72.8453),
    CanonicalSiteRecord("Chiquinquirá",       "Chiquinquirá",    "Boyacá",        5.6183,  -73.8169),
    CanonicalSiteRecord("Ráquira",            "Ráquira",         "Boyacá",        5.5333,  -73.6333),
    CanonicalSiteRecord("Tenza",              "Tenza",           "Boyacá",        5.0706,  -73.4117),
    # ── Cundinamarca ──────────────────────────────────────────────────────────
    CanonicalSiteRecord("Piedras del Tunjo",  "Facatativá",      "Cundinamarca",  4.8175,  -74.3567),
    CanonicalSiteRecord("Sutatausa",          "Sutatausa",       "Cundinamarca",  5.0347,  -73.8628),
    CanonicalSiteRecord("Tibacuy",            "Tibacuy",         "Cundinamarca",  4.3492,  -74.4514),
    CanonicalSiteRecord("Soacha",             "Soacha",          "Cundinamarca",  4.5797,  -74.2172),
    CanonicalSiteRecord("Zipaquirá",          "Zipaquirá",       "Cundinamarca",  5.0231,  -74.0058),
    CanonicalSiteRecord("El Colegio",         "El Colegio",      "Cundinamarca",  4.5803,  -74.4397),
    CanonicalSiteRecord("Pandi",              "Pandi",           "Cundinamarca",  4.1928,  -74.4878),
    CanonicalSiteRecord("Supatá",             "Supatá",          "Cundinamarca",  5.0392,  -74.0436),
    CanonicalSiteRecord("Nemocón",            "Nemocón",         "Cundinamarca",  5.0681,  -73.8853),
    CanonicalSiteRecord("La Mesa",            "La Mesa",         "Cundinamarca",  4.6347,  -74.4611),
    CanonicalSiteRecord("Bojacá",             "Bojacá",          "Cundinamarca",  4.7333,  -74.3333),
    # ── Santander ─────────────────────────────────────────────────────────────
    CanonicalSiteRecord("Guane",              "Barichara",       "Santander",     6.6467,  -73.2294),
    CanonicalSiteRecord("Jordán",             "San Gil",         "Santander",     6.5556,  -73.1369),
    CanonicalSiteRecord("Cepitá",             "Cepitá",          "Santander",     6.2833,  -72.9667),
    CanonicalSiteRecord("Charalá",            "Charalá",         "Santander",     6.2894,  -73.1428),
    # ── Huila ─────────────────────────────────────────────────────────────────
    CanonicalSiteRecord("San Agustín",        "San Agustín",     "Huila",         1.8853,  -76.2744),
    CanonicalSiteRecord("Isnos",              "Isnos",           "Huila",         1.9278,  -76.2375),
    CanonicalSiteRecord("La Plata",           "La Plata",        "Huila",         2.3897,  -75.8950),
    # ── Nariño ────────────────────────────────────────────────────────────────
    CanonicalSiteRecord("La Florida",         "La Florida",      "Nariño",        1.3025,  -77.4100),
    CanonicalSiteRecord("Cumbal",             "Cumbal",          "Nariño",        0.9094,  -77.7894),
    CanonicalSiteRecord("Ipiales",            "Ipiales",         "Nariño",        0.8281,  -77.6447),
    # ── Antioquia ─────────────────────────────────────────────────────────────
    CanonicalSiteRecord("Santa Fe de Antioquia", "Santa Fe de Antioquia", "Antioquia", 6.5567, -75.8267),
    CanonicalSiteRecord("Sopetrán",           "Sopetrán",        "Antioquia",     6.5044,  -75.7419),
    CanonicalSiteRecord("Amalfi",             "Amalfi",          "Antioquia",     6.9133,  -75.0706),
)

# Radio de la Tierra en km (para haversine)
_EARTH_RADIUS_KM = 6_371.0

# Distancia máxima para considerar un sitio como "cercano" (km).
# Por debajo → factor 1.0 (sin penalización).
# Por encima → penalización proporcional hasta PENALTY_MAX_DISTANCE_KM.
GEO_NEAR_THRESHOLD_KM: float = 150.0

# A esta distancia o más, el factor geográfico llega a su mínimo.
GEO_FAR_THRESHOLD_KM: float = 800.0

# Factor mínimo que recibe un match muy lejano (nunca se elimina del todo).
GEO_MIN_FACTOR: float = 0.55

# Peso del componente geográfico en el score final (0.0 = sin reranking).
GEO_ALPHA: float = 0.30


def _normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split()).casefold()
    return text


def _compact(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _build_lookup(values: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical in values:
        lookup[_normalize_key(canonical)] = canonical
    return lookup


_SITE_LOOKUP = _build_lookup([record.site_name for record in _WORKSHOP_SITE_RECORDS])
_MUNICIPALITY_LOOKUP = _build_lookup(
    sorted({record.municipality for record in _WORKSHOP_SITE_RECORDS})
)
_DEPARTMENT_LOOKUP = _build_lookup(
    sorted({record.department for record in _WORKSHOP_SITE_RECORDS})
)
_SITE_RECORD_BY_NAME = {record.site_name: record for record in _WORKSHOP_SITE_RECORDS}
_SITE_RECORD_BY_MUNICIPALITY = {record.municipality: record for record in _WORKSHOP_SITE_RECORDS}


def canonicalize_site_name(value: str) -> str:
    """Devuelve el nombre canónico del sitio según el catálogo del taller."""
    key = _normalize_key(value)
    return _SITE_LOOKUP.get(key, _compact(value))


def canonicalize_municipality(value: str) -> str:
    """Normaliza el municipio usando la misma fuente de verdad del taller."""
    key = _normalize_key(value)
    return _MUNICIPALITY_LOOKUP.get(key, _compact(value))


def canonicalize_department(value: str) -> str:
    """Normaliza el departamento usando la misma fuente de verdad del taller."""
    key = _normalize_key(value)
    return _DEPARTMENT_LOOKUP.get(key, _compact(value))


def normalize_site_metadata(
    site_name: str,
    municipality: str = "",
    department: str = "",
) -> tuple[str, str, str]:
    """
    Normaliza el trío sitio/municipio/departamento.

    Si el sitio existe en el catálogo del taller, toma su municipio y
    departamento canónicos para evitar divergencias entre fuentes.
    """
    normalized_site = canonicalize_site_name(site_name)
    canonical = _SITE_RECORD_BY_NAME.get(normalized_site)
    if canonical:
        return canonical.site_name, canonical.municipality, canonical.department

    return (
        normalized_site,
        canonicalize_municipality(municipality),
        canonicalize_department(department),
    )


# ── Utilidades geográficas ────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia entre dos puntos geográficos en kilómetros (fórmula de Haversine)."""
    if not (lat1 or lon1 or lat2 or lon2):
        return float("inf")

    r = _EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def site_coords(site_name: str, municipality: str = "") -> tuple[float, float] | None:
    """
    Devuelve (lat, lon) del sitio si está en el catálogo, o None si no se conoce.

    Busca primero por nombre canónico de sitio, luego por municipio.
    """
    canonical_name = canonicalize_site_name(site_name)
    record = _SITE_RECORD_BY_NAME.get(canonical_name)
    if record and (record.lat or record.lon):
        return record.lat, record.lon

    if municipality:
        canonical_mun = canonicalize_municipality(municipality)
        record_by_mun = _SITE_RECORD_BY_MUNICIPALITY.get(canonical_mun)
        if record_by_mun and (record_by_mun.lat or record_by_mun.lon):
            return record_by_mun.lat, record_by_mun.lon

    return None


def geo_proximity_factor(
    query_lat: float,
    query_lon: float,
    match_lat: float,
    match_lon: float,
) -> float:
    """
    Factor de proximidad geográfica en [GEO_MIN_FACTOR, 1.0].

    - Distancia ≤ GEO_NEAR_THRESHOLD_KM  → 1.0  (sin penalización)
    - Distancia ≥ GEO_FAR_THRESHOLD_KM   → GEO_MIN_FACTOR
    - En el medio                         → interpolación lineal
    """
    dist = haversine_km(query_lat, query_lon, match_lat, match_lon)
    if dist <= GEO_NEAR_THRESHOLD_KM:
        return 1.0
    if dist >= GEO_FAR_THRESHOLD_KM:
        return GEO_MIN_FACTOR
    ratio = (dist - GEO_NEAR_THRESHOLD_KM) / (GEO_FAR_THRESHOLD_KM - GEO_NEAR_THRESHOLD_KM)
    return round(1.0 - ratio * (1.0 - GEO_MIN_FACTOR), 4)


def geo_rerank(
    matches: list[dict],
    query_site: str,
    query_municipality: str = "",
    query_lat: float = 0.0,
    query_lon: float = 0.0,
    alpha: float = GEO_ALPHA,
) -> list[dict]:
    """
    Reordena y ajusta los matches de similitud iconográfica dando preferencia
    a sitios geográficamente cercanos al petroglifo consultado.

    Estrategia:
        geo_score = alpha * proximity_factor + (1 - alpha) * similarity_score

    El campo `similarity_score` original se preserva intacto en el resultado;
    se agrega `geo_adjusted_score` para trazabilidad y la lista se reordena
    por ese score combinado.

    Si no se puede obtener las coordenadas del sitio consultado (no está en el
    catálogo), los matches se devuelven sin modificar.

    Args:
        matches:            Lista de dicts producidos por ImageVectorAdapter.
        query_site:         Nombre del sitio que se está clasificando.
        query_municipality: Municipio del sitio que se está clasificando.
        query_lat/lon:      Coordenadas explícitas (si el usuario las aportó).
        alpha:              Peso del componente geográfico (0 = sin efecto).

    Returns:
        Lista reordenada con el campo `geo_adjusted_score` añadido.
    """
    if not matches or alpha <= 0.0:
        return matches

    # Resolver coordenadas del sitio consultado
    if query_lat and query_lon:
        origin = (query_lat, query_lon)
    else:
        origin = site_coords(query_site, query_municipality)

    # Sin coordenadas de origen → no se puede aplicar reranking; devolver sin cambios
    if not origin:
        return matches

    q_lat, q_lon = origin
    adjusted = []

    for match in matches:
        sim = float(match.get("similarity_score", 0.0))

        # Resolver coordenadas del match
        m_coords = site_coords(
            match.get("site_name", ""),
            match.get("municipality", ""),
        )

        if m_coords:
            prox = geo_proximity_factor(q_lat, q_lon, m_coords[0], m_coords[1])
            geo_adj = round(alpha * prox + (1.0 - alpha) * sim, 4)
            dist_km = round(haversine_km(q_lat, q_lon, m_coords[0], m_coords[1]), 1)
        else:
            # Sitio desconocido → usar proximidad neutral (factor intermedio)
            prox = (1.0 + GEO_MIN_FACTOR) / 2.0
            geo_adj = round(alpha * prox + (1.0 - alpha) * sim, 4)
            dist_km = None

        adjusted.append({
            **match,
            "geo_adjusted_score": geo_adj,
            "geo_proximity_factor": round(prox, 4),
            "distance_km": dist_km,
        })

    # Reordenar por score combinado, preservando similitud como desempate
    adjusted.sort(key=lambda m: (m["geo_adjusted_score"], m["similarity_score"]), reverse=True)
    return adjusted
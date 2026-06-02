"""Normalización canónica de sitios rupestres para evitar duplicados."""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class CanonicalSiteRecord:
    site_name: str
    municipality: str
    department: str


_WORKSHOP_SITE_RECORDS: tuple[CanonicalSiteRecord, ...] = (
    CanonicalSiteRecord("Villa de Leyva", "Villa de Leyva", "Boyacá"),
    CanonicalSiteRecord("Sáchica", "Sáchica", "Boyacá"),
    CanonicalSiteRecord("Gámeza", "Gámeza", "Boyacá"),
    CanonicalSiteRecord("Sogamoso", "Sogamoso", "Boyacá"),
    CanonicalSiteRecord("Tunja", "Tunja", "Boyacá"),
    CanonicalSiteRecord("Piedras del Tunjo", "Facatativá", "Cundinamarca"),
    CanonicalSiteRecord("Sutatausa", "Sutatausa", "Cundinamarca"),
    CanonicalSiteRecord("Tibacuy", "Tibacuy", "Cundinamarca"),
    CanonicalSiteRecord("Soacha", "Soacha", "Cundinamarca"),
    CanonicalSiteRecord("Zipaquirá", "Zipaquirá", "Cundinamarca"),
    CanonicalSiteRecord("El Colegio", "El Colegio", "Cundinamarca"),
    CanonicalSiteRecord("Pandi", "Pandi", "Cundinamarca"),
    CanonicalSiteRecord("Chivor", "Chivor", "Boyacá"),
    CanonicalSiteRecord("Soatá", "Soatá", "Boyacá"),
    CanonicalSiteRecord("Monguí", "Monguí", "Boyacá"),
    CanonicalSiteRecord("Chiquinquirá", "Chiquinquirá", "Boyacá"),
    CanonicalSiteRecord("Ráquira", "Ráquira", "Boyacá"),
    CanonicalSiteRecord("Tenza", "Tenza", "Boyacá"),
    CanonicalSiteRecord("Supatá", "Supatá", "Cundinamarca"),
    CanonicalSiteRecord("Nemocón", "Nemocón", "Cundinamarca"),
    CanonicalSiteRecord("La Mesa", "La Mesa", "Cundinamarca"),
    CanonicalSiteRecord("Bojacá", "Bojacá", "Cundinamarca"),
    CanonicalSiteRecord("Guane", "Barichara", "Santander"),
    CanonicalSiteRecord("Jordán", "San Gil", "Santander"),
    CanonicalSiteRecord("Cepitá", "Cepitá", "Santander"),
    CanonicalSiteRecord("Charalá", "Charalá", "Santander"),
    CanonicalSiteRecord("San Agustín", "San Agustín", "Huila"),
    CanonicalSiteRecord("Isnos", "Isnos", "Huila"),
    CanonicalSiteRecord("La Plata", "La Plata", "Huila"),
    CanonicalSiteRecord("La Florida", "La Florida", "Nariño"),
    CanonicalSiteRecord("Cumbal", "Cumbal", "Nariño"),
    CanonicalSiteRecord("Ipiales", "Ipiales", "Nariño"),
    CanonicalSiteRecord("Santa Fe de Antioquia", "Santa Fe de Antioquia", "Antioquia"),
    CanonicalSiteRecord("Sopetrán", "Sopetrán", "Antioquia"),
    CanonicalSiteRecord("Amalfi", "Amalfi", "Antioquia"),
)


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


"""Utilidades para subir archivos puntuales a Cloudinary."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import structlog

from config.settings import settings

log = structlog.get_logger(__name__)


@lru_cache
def _is_configured() -> bool:
    return all(
        (
            settings.cloudinary_cloud_name,
            settings.cloudinary_api_key,
            settings.cloudinary_api_secret,
        )
    )


@lru_cache
def _configure_cloudinary() -> bool:
    """Configura el SDK una sola vez si hay credenciales disponibles."""
    if not _is_configured():
        return False

    try:
        import cloudinary
    except ModuleNotFoundError:
        log.warning("cloudinary_sdk_missing", hint="Instala la dependencia cloudinary")
        return False

    cloudinary.config(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key,
        api_secret=settings.cloudinary_api_secret,
        secure=True,
    )
    return True


def upload_image(path: str | Path, public_id: str) -> str:
    """Sube una imagen del usuario y devuelve la URL segura resultante."""
    return _upload(
        path=path,
        public_id=public_id,
        folder=settings.cloudinary_image_folder,
        resource_type="image",
    )


def upload_pdf(path: str | Path, public_id: str) -> str:
    """Sube el PDF generado y devuelve la URL segura resultante."""
    return _upload(
        path=path,
        public_id=public_id,
        folder=settings.cloudinary_pdf_folder,
        resource_type="image",
    )


def _upload(
    path: str | Path,
    public_id: str,
    folder: str,
    resource_type: str,
) -> str:
    file_path = Path(path)
    if not file_path.exists():
        log.warning("cloudinary_upload_missing_file", path=str(file_path))
        return ""

    if not _configure_cloudinary():
        return ""

    try:
        import cloudinary.uploader

        result = cloudinary.uploader.upload(
            str(file_path),
            folder=folder,
            public_id=public_id,
            resource_type=resource_type,
            overwrite=True,
            use_filename=False,
            unique_filename=False,
        )
        return result.get("secure_url", "") or result.get("url", "")
    except Exception as exc:
        log.warning(
            "cloudinary_upload_failed",
            path=str(file_path),
            public_id=public_id,
            folder=folder,
            error=str(exc),
        )
        return ""

"""
Normaliza los nombres de sitios rupestres en la BD usando el catálogo del taller.

Qué hace:
    - Unifica variantes como `Gameza` -> `Gámeza`
    - Reasigna `petroglyphs.site_id` a los sitios canónicos
    - Reescribe `image_embeddings.site_name` y `municipality`
    - Reconstruye `site_graph_edges` sin duplicados
    - Recalcula `petroglyph_count`

Uso:
    python -m scripts.normalize_site_names --dry-run
    python -m scripts.normalize_site_names --apply
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import structlog
from sqlalchemy import delete, func, select, update

from core.domain.site_normalization import normalize_site_metadata

log = structlog.get_logger("normalize_site_names")


@dataclass
class SiteMergePlan:
    canonical_name: str
    canonical_municipality: str
    canonical_department: str
    keeper_id: str
    keeper_name: str
    site_ids: list[str]
    site_names: list[str]


def _pick_keeper_id(rows) -> object:
    exact = next((row for row in rows if row.name == normalize_site_metadata(row.name, row.municipality, row.department)[0]), None)
    if exact:
        return exact.id
    prioritized = sorted(
        rows,
        key=lambda row: (
            0 if row.name else 1,
            0 if row.municipality else 1,
            0 if row.department else 1,
            str(getattr(row, "created_at", "")),
            str(row.id),
        ),
    )
    return prioritized[0].id


async def _load_state(session):
    from infrastructure.database.models.models import ImageEmbedding, PetroglyphModel, RupestranSiteModel, SiteGraphEdge

    sites = list((await session.execute(select(RupestranSiteModel))).scalars().all())
    petroglyphs = list((await session.execute(select(PetroglyphModel))).scalars().all())
    embeddings = list((await session.execute(select(ImageEmbedding))).scalars().all())
    edges = list((await session.execute(select(SiteGraphEdge))).scalars().all())
    return sites, petroglyphs, embeddings, edges


def _build_plan(sites) -> tuple[list[SiteMergePlan], dict[str, str], dict[str, tuple[str, str, str]]]:
    grouped: dict[str, list] = defaultdict(list)
    canonical_meta: dict[str, tuple[str, str, str]] = {}

    for site in sites:
        canonical_name, canonical_municipality, canonical_department = normalize_site_metadata(
            site.name,
            site.municipality,
            site.department,
        )
        grouped[canonical_name].append(site)
        canonical_meta[canonical_name] = (
            canonical_name,
            canonical_municipality,
            canonical_department,
        )

    plans: list[SiteMergePlan] = []
    site_id_map: dict[str, str] = {}

    for canonical_name, rows in sorted(grouped.items(), key=lambda item: item[0]):
        keeper_id = _pick_keeper_id(rows)
        keeper = next(row for row in rows if row.id == keeper_id)
        canonical_site_name, canonical_municipality, canonical_department = canonical_meta[canonical_name]
        plans.append(
            SiteMergePlan(
                canonical_name=canonical_site_name,
                canonical_municipality=canonical_municipality,
                canonical_department=canonical_department,
                keeper_id=keeper_id,
                keeper_name=keeper.name,
                site_ids=[row.id for row in rows],
                site_names=[row.name for row in rows],
            )
        )
        for row in rows:
            site_id_map[row.id] = keeper_id

    return plans, site_id_map, canonical_meta


def _aggregate_edges(edges, site_id_map: dict[str, str]) -> list[dict]:
    aggregated: dict[tuple[str, str], dict] = {}

    for edge in edges:
        site_a = site_id_map.get(edge.site_a_id, edge.site_a_id)
        site_b = site_id_map.get(edge.site_b_id, edge.site_b_id)
        if site_a == site_b:
            continue

        pair = tuple(sorted((site_a, site_b)))
        bucket = aggregated.setdefault(
            pair,
            {
                "site_a_id": pair[0],
                "site_b_id": pair[1],
                "weight_total": 0.0,
                "evidence_total": 0,
                "shared_taxonomies": [],
            },
        )
        evidence = max(int(getattr(edge, "evidence_count", 1) or 1), 1)
        bucket["weight_total"] += float(edge.weight or 0.0) * evidence
        bucket["evidence_total"] += evidence
        for taxonomy in list(edge.shared_taxonomies or []):
            if taxonomy and taxonomy not in bucket["shared_taxonomies"]:
                bucket["shared_taxonomies"].append(taxonomy)

    merged = []
    for pair, bucket in sorted(aggregated.items(), key=lambda item: item[0]):
        evidence_total = max(bucket["evidence_total"], 1)
        merged.append(
            {
                "site_a_id": bucket["site_a_id"],
                "site_b_id": bucket["site_b_id"],
                "weight": round(bucket["weight_total"] / evidence_total, 4),
                "evidence_count": evidence_total,
                "shared_taxonomies": bucket["shared_taxonomies"],
            }
        )
    return merged


async def _apply_plan(session, plans: list[SiteMergePlan], site_id_map: dict[str, str], edges) -> None:
    from infrastructure.database.models.models import ImageEmbedding, PetroglyphModel, RupestranSiteModel, SiteGraphEdge

    # Reasignar petroglifos a los sitios canónicos
    for plan in plans:
        for old_site_id in plan.site_ids:
            if old_site_id == plan.keeper_id:
                continue
            await session.execute(
                update(PetroglyphModel)
                .where(PetroglyphModel.site_id == old_site_id)
                .values(site_id=plan.keeper_id)
            )

    # Normalizar los sitios canónicos y eliminar duplicados
    for plan in plans:
        await session.execute(
            update(RupestranSiteModel)
            .where(RupestranSiteModel.id == plan.keeper_id)
            .values(
                name=plan.canonical_name,
                municipality=plan.canonical_municipality,
                department=plan.canonical_department,
            )
        )
        duplicate_ids = [site_id for site_id in plan.site_ids if site_id != plan.keeper_id]
        if duplicate_ids:
            await session.execute(
                delete(RupestranSiteModel).where(RupestranSiteModel.id.in_(duplicate_ids))
            )

    # Normalizar embeddings de referencia, incluso cuando el nombre no existe
    # literalmente en rupestrian_sites pero sí es una variante del taller.
    for embedding in session.identity_map.values():
        if isinstance(embedding, ImageEmbedding):
            site_name, municipality, _department = normalize_site_metadata(
                embedding.site_name,
                embedding.municipality,
                "",
            )
            embedding.site_name = site_name
            embedding.municipality = municipality

    # Recalcular conteos por sitio
    counts = dict(
        (await session.execute(
            select(PetroglyphModel.site_id, func.count(PetroglyphModel.id))
            .where(PetroglyphModel.site_id.is_not(None))
            .group_by(PetroglyphModel.site_id)
        )).all()
    )
    for plan in plans:
        await session.execute(
            update(RupestranSiteModel)
            .where(RupestranSiteModel.id == plan.keeper_id)
            .values(petroglyph_count=int(counts.get(plan.keeper_id, 0)))
        )

    # Reconstruir aristas para evitar duplicados tras la fusión de sitios
    await session.execute(delete(SiteGraphEdge))
    merged_edges = _aggregate_edges(edges, site_id_map)
    for edge in merged_edges:
        session.add(
            SiteGraphEdge(
                site_a_id=edge["site_a_id"],
                site_b_id=edge["site_b_id"],
                weight=edge["weight"],
                evidence_count=edge["evidence_count"],
                shared_taxonomies=edge["shared_taxonomies"],
            )
        )

    await session.flush()
    await session.commit()


async def _dry_run(session) -> None:
    sites, petroglyphs, embeddings, edges = await _load_state(session)
    plans, site_id_map, _ = _build_plan(sites)
    duplicate_groups = [plan for plan in plans if len(plan.site_ids) > 1]

    log.info(
        "normalization_preview",
        total_sites=len(sites),
        groups=len(plans),
        duplicate_groups=len(duplicate_groups),
        petroglyphs=len(petroglyphs),
        embeddings=len(embeddings),
        edges=len(edges),
    )

    for plan in duplicate_groups[:20]:
        log.info(
            "duplicate_site_group",
            canonical=plan.canonical_name,
            keeper_id=plan.keeper_id,
            names=plan.site_names,
        )


async def _run(apply_changes: bool) -> None:
    from infrastructure.database.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        sites, petroglyphs, embeddings, edges = await _load_state(session)
        plans, site_id_map, _ = _build_plan(sites)

        if not apply_changes:
            await _dry_run(session)
            return

        log.info(
            "normalization_apply_start",
            total_sites=len(sites),
            duplicate_groups=sum(1 for plan in plans if len(plan.site_ids) > 1),
            petroglyphs=len(petroglyphs),
            embeddings=len(embeddings),
            edges=len(edges),
        )
        await _apply_plan(session, plans, site_id_map, edges)
        log.info("normalization_apply_complete", groups=len(plans))


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza nombres de sitios rupestres en la BD")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Aplica cambios a la base de datos")
    mode.add_argument("--dry-run", action="store_true", help="Solo muestra el plan de normalización")
    args = parser.parse_args()
    asyncio.run(_run(apply_changes=args.apply and not args.dry_run))


if __name__ == "__main__":
    main()

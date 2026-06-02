"""Motor y sesión async de SQLAlchemy."""
from __future__ import annotations
from collections.abc import AsyncGenerator
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from config.settings import settings


class Base(DeclarativeBase):
    pass


def _normalize_database_url(raw_url: str) -> str:
    """
    Accept a plain PostgreSQL URL in .env and upgrade it to the async driver
    used by this app.
    """
    url = make_url(raw_url)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+asyncpg")
    return url.render_as_string(hide_password=False)


engine = create_async_engine(
    _normalize_database_url(settings.database_url),
    echo=settings.env == "development",
    connect_args={"statement_cache_size": 0},
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Crea todas las tablas (usar solo en dev; en prod usar Alembic)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

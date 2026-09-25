"""Alembic environment (async, asyncpg). The URL comes from settings, never from alembic.ini."""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from autotrader.core.settings import Settings


def _url() -> str:
    return Settings().database_url.get_secret_value()


def run_offline() -> None:
    context.configure(url=_url(), literal_binds=True, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection: object) -> None:
    context.configure(connection=connection, target_metadata=None)  # type: ignore[arg-type]
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    engine = create_async_engine(_url())
    async with engine.connect() as conn:
        await conn.run_sync(_do_run)
        await conn.commit()
    await engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    asyncio.run(run_online())

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.core.db import get_engine
from app.main import create_app

ADMIN = {"Authorization": f"Bearer {os.environ['ADMIN_API_KEY']}"}


async def _create_database_if_missing() -> None:
    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    base, db_name = url.rsplit("/", 1)
    connection = await asyncpg.connect(f"{base}/postgres")
    try:
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        if not exists:
            await connection.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await connection.close()


@pytest.fixture(scope="session", autouse=True)
async def database() -> AsyncIterator[None]:
    """Creates the test database if needed and applies migrations once per session."""
    await _create_database_if_missing()
    await asyncio.to_thread(
        subprocess.run,
        ["alembic", "upgrade", "head"],
        check=True,
        capture_output=True,
        env=os.environ,
    )
    yield
    await get_engine().dispose()


@pytest.fixture(autouse=True)
async def clean_tables() -> None:
    """Every test starts empty. Model versions, training runs and predictions are all removed."""
    async with get_engine().begin() as connection:
        await connection.execute(
            text("TRUNCATE model_versions, training_runs, predictions CASCADE")
        )


@pytest.fixture(scope="session")
def app() -> Any:
    return create_app()


@pytest.fixture
async def client(app: Any) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


API = {"Authorization": f"Bearer {os.environ['API_KEY']}"}


async def train(client: AsyncClient, **body: Any) -> dict[str, Any]:
    """Trains a model through the API and returns {"run": ..., "modelVersion": ...}."""
    response = await client.post("/training/runs", headers=ADMIN, json=body)
    assert response.status_code == 201, response.text
    return response.json()["data"]


async def promote(client: AsyncClient, version_id: str, stage: str = "production") -> dict:
    response = await client.post(
        f"/models/{version_id}/promote", headers=ADMIN, json={"stage": stage}
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]


def sample_instances(count: int = 1) -> list[dict[str, float]]:
    """Real rows from the bundled dataset, as request instances."""
    from app.modules.training.trainer import load_dataset

    features, _, _ = load_dataset()
    return [{k: float(v) for k, v in row.items()} for _, row in features.head(count).iterrows()]

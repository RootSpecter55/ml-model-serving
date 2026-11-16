from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.base_repository import BaseRepository
from app.modules.models.models import ModelVersion, Stage


class ModelVersionRepository(BaseRepository[ModelVersion]):
    model = ModelVersion

    async def next_version_number(self, session: AsyncSession, name: str) -> int:
        current = await session.scalar(
            select(func.max(ModelVersion.version)).where(ModelVersion.name == name)
        )
        return (current or 0) + 1

    async def get_production(self, session: AsyncSession, name: str) -> ModelVersion | None:
        return await session.scalar(
            select(ModelVersion).where(
                ModelVersion.name == name, ModelVersion.stage == Stage.PRODUCTION
            )
        )

    async def lock_production(self, session: AsyncSession, name: str) -> ModelVersion | None:
        """The current production version, row-locked until the transaction ends."""
        return await session.scalar(
            select(ModelVersion)
            .where(ModelVersion.name == name, ModelVersion.stage == Stage.PRODUCTION)
            .with_for_update()
        )

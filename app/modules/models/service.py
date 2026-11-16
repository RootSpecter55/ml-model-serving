from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import Database
from app.core.errors import ConflictError, NotFoundError
from app.core.pagination import PageParams
from app.modules.models.models import ModelVersion, Stage
from app.modules.models.repository import ModelVersionRepository
from app.modules.models.schemas import ModelVersionOut

OnPromoted = Callable[[ModelVersionOut], Awaitable[None]]


class ModelRegistryService:
    def __init__(
        self, db: Database, versions: ModelVersionRepository, on_promoted: OnPromoted
    ) -> None:
        self._db = db
        self._versions = versions
        # Called after a production promotion commits (the prediction cache loads the model).
        self._on_promoted = on_promoted

    async def list(
        self, name: str | None, stage: str | None, page: PageParams
    ) -> tuple[list[ModelVersionOut], int]:
        filters = []
        if name:
            filters.append(ModelVersion.name == name)
        if stage:
            filters.append(ModelVersion.stage == stage)
        async with self._db.transaction() as session:
            rows, total = await self._versions.list(
                session,
                *filters,
                order_by=ModelVersion.created_at.desc(),
                offset=page.offset,
                limit=page.page_size,
            )
        return [ModelVersionOut.model_validate(v) for v in rows], total

    async def get(self, version_id: UUID) -> ModelVersionOut:
        async with self._db.transaction() as session:
            return ModelVersionOut.model_validate(await self._require(session, version_id))

    async def get_production(self, name: str) -> ModelVersionOut | None:
        async with self._db.transaction() as session:
            version = await self._versions.get_production(session, name)
        return ModelVersionOut.model_validate(version) if version else None

    async def update_description(
        self, version_id: UUID, description: str | None
    ) -> ModelVersionOut:
        async with self._db.transaction() as session:
            version = await self._require(session, version_id)
            version.description = description
        return ModelVersionOut.model_validate(version)

    async def delete(self, version_id: UUID) -> None:
        async with self._db.transaction() as session:
            version = await self._require(session, version_id)
            if version.stage == Stage.PRODUCTION:
                raise ConflictError("Cannot delete the production version; promote another first")
            await self._versions.delete(session, version)

    async def promote(self, version_id: UUID, stage: str) -> ModelVersionOut:
        """Moves a version to staging or production.

        Production: the current production version of the same model is locked and archived in
        the same transaction, so there is never a moment with two (a partial unique index in the
        database enforces this too, including against concurrent promotions).
        """
        try:
            async with self._db.transaction() as session:
                version = await self._require(session, version_id)
                if stage == Stage.PRODUCTION:
                    current = await self._versions.lock_production(session, version.name)
                    if current and current.id != version.id:
                        self._set_stage(current, Stage.ARCHIVED)
                        await session.flush()  # archive first, so the unique index allows the swap
                self._set_stage(version, Stage(stage))
        except IntegrityError as exc:
            raise ConflictError(
                "Another promotion of this model happened at the same time"
            ) from exc

        promoted = ModelVersionOut.model_validate(version)
        if stage == Stage.PRODUCTION:
            await self._on_promoted(promoted)
        return promoted

    async def archive(self, version_id: UUID) -> ModelVersionOut:
        """Archiving the production version leaves nothing to serve: predict then returns 409."""
        async with self._db.transaction() as session:
            version = await self._require(session, version_id)
            self._set_stage(version, Stage.ARCHIVED)
        return ModelVersionOut.model_validate(version)

    async def register(self, session: AsyncSession, version: ModelVersion) -> ModelVersion:
        """Used by training, inside its transaction: stores a new version at the next number.

        Two trainings of the same model finishing at the same moment could pick the same number;
        the (name, version) unique constraint then rejects the second instead of duplicating it.
        """
        version.version = await self._versions.next_version_number(session, version.name)
        return await self._versions.add(session, version)

    @staticmethod
    def _set_stage(version: ModelVersion, stage: Stage) -> None:
        version.stage = stage
        version.stage_changed_at = datetime.now(UTC)

    async def _require(self, session: AsyncSession, version_id: UUID) -> ModelVersion:
        version = await self._versions.get(session, version_id)
        if version is None:
            raise NotFoundError("Model version")
        return version

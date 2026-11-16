from typing import Any
from uuid import UUID

from app.core.pagination import PageParams, data_envelope, page_envelope
from app.modules.models.schemas import ModelVersionUpdate, PromoteRequest
from app.modules.models.service import ModelRegistryService


class ModelsController:
    def __init__(self, service: ModelRegistryService) -> None:
        self._service = service

    async def list(self, name: str | None, stage: str | None, page: PageParams) -> dict[str, Any]:
        items, total = await self._service.list(name, stage, page)
        return page_envelope([v.to_json() for v in items], total, page)

    async def get(self, version_id: UUID) -> dict[str, Any]:
        return data_envelope((await self._service.get(version_id)).to_json())

    async def update(self, version_id: UUID, body: ModelVersionUpdate) -> dict[str, Any]:
        version = await self._service.update_description(version_id, body.description)
        return data_envelope(version.to_json())

    async def delete(self, version_id: UUID) -> None:
        await self._service.delete(version_id)

    async def promote(self, version_id: UUID, body: PromoteRequest) -> dict[str, Any]:
        return data_envelope((await self._service.promote(version_id, body.stage)).to_json())

    async def archive(self, version_id: UUID) -> dict[str, Any]:
        return data_envelope((await self._service.archive(version_id)).to_json())

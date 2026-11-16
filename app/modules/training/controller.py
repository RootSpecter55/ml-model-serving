from typing import Any
from uuid import UUID

from app.core.pagination import PageParams, data_envelope, page_envelope
from app.modules.training.schemas import TrainingRequest
from app.modules.training.service import TrainingService


class TrainingController:
    def __init__(self, service: TrainingService) -> None:
        self._service = service

    async def train(self, body: TrainingRequest) -> dict[str, Any]:
        return data_envelope((await self._service.train(body)).to_json())

    async def list(self, page: PageParams) -> dict[str, Any]:
        items, total = await self._service.list(page)
        return page_envelope([r.to_json() for r in items], total, page)

    async def get(self, run_id: UUID) -> dict[str, Any]:
        return data_envelope((await self._service.get(run_id)).to_json())

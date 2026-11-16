from typing import Any
from uuid import UUID

from app.core.pagination import PageParams, data_envelope, page_envelope
from app.modules.predictions.schemas import PredictRequest
from app.modules.predictions.service import PredictionsService


class PredictionsController:
    def __init__(self, service: PredictionsService) -> None:
        self._service = service

    async def predict(self, body: PredictRequest) -> dict[str, Any]:
        instances = [{k: float(v) for k, v in row.items()} for row in body.instances]
        return data_envelope((await self._service.predict(body.model, instances)).to_json())

    async def list(self, model_version_id: UUID | None, page: PageParams) -> dict[str, Any]:
        items, total = await self._service.list(model_version_id, page)
        return page_envelope([p.to_json() for p in items], total, page)

    async def get(self, prediction_id: UUID) -> dict[str, Any]:
        return data_envelope((await self._service.get(prediction_id)).to_json())

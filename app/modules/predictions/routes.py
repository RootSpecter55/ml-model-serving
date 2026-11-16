from collections.abc import Callable
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.core.pagination import PageParams, page_params
from app.modules.predictions.controller import PredictionsController
from app.modules.predictions.schemas import PredictRequest


def predictions_router(
    controller: PredictionsController, require_api_key: Callable[..., Any]
) -> APIRouter:
    router = APIRouter(tags=["predictions"], dependencies=[Depends(require_api_key)])

    @router.post("/predict", summary="Predict with the model's production version")
    async def predict(body: PredictRequest) -> dict[str, Any]:
        return await controller.predict(body)

    @router.get("/predictions", summary="Logged predictions, newest first")
    async def list_predictions(
        page: Annotated[PageParams, Depends(page_params)],
        model_version_id: Annotated[UUID | None, Query(alias="modelVersionId")] = None,
    ) -> dict[str, Any]:
        return await controller.list(model_version_id, page)

    @router.get("/predictions/{prediction_id}", summary="One logged prediction")
    async def get(prediction_id: UUID) -> dict[str, Any]:
        return await controller.get(prediction_id)

    return router

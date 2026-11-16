from collections.abc import Callable
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends

from app.core.pagination import PageParams, page_params
from app.modules.training.controller import TrainingController
from app.modules.training.schemas import TrainingRequest


def training_router(controller: TrainingController, require_admin: Callable[..., Any]) -> APIRouter:
    router = APIRouter(
        prefix="/training/runs", tags=["training"], dependencies=[Depends(require_admin)]
    )

    @router.post("", status_code=201, summary="Train, log to MLflow and register a new version")
    async def train(body: TrainingRequest) -> dict[str, Any]:
        return await controller.train(body)

    @router.get("", summary="Training run history, newest first")
    async def list_runs(page: Annotated[PageParams, Depends(page_params)]) -> dict[str, Any]:
        return await controller.list(page)

    @router.get("/{run_id}", summary="One training run")
    async def get(run_id: UUID) -> dict[str, Any]:
        return await controller.get(run_id)

    return router

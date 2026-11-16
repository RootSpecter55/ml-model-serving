from collections.abc import Callable
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response

from app.core.pagination import PageParams, page_params
from app.modules.models.controller import ModelsController
from app.modules.models.schemas import ModelVersionUpdate, PromoteRequest

StageFilter = Literal["none", "staging", "production", "archived"]


def models_router(controller: ModelsController, require_admin: Callable[..., Any]) -> APIRouter:
    router = APIRouter(
        prefix="/models", tags=["model registry"], dependencies=[Depends(require_admin)]
    )

    @router.get("", summary="List model versions, newest first")
    async def list_versions(
        page: Annotated[PageParams, Depends(page_params)],
        name: Annotated[str | None, Query(max_length=100)] = None,
        stage: StageFilter | None = None,
    ) -> dict[str, Any]:
        return await controller.list(name, stage, page)

    @router.get("/{version_id}", summary="Get a model version with its metrics")
    async def get(version_id: UUID) -> dict[str, Any]:
        return await controller.get(version_id)

    @router.patch("/{version_id}", summary="Update a version's description")
    async def update(version_id: UUID, body: ModelVersionUpdate) -> dict[str, Any]:
        return await controller.update(version_id, body)

    @router.delete("/{version_id}", status_code=204, summary="Delete a non-production version")
    async def delete(version_id: UUID) -> Response:
        await controller.delete(version_id)
        return Response(status_code=204)

    @router.post(
        "/{version_id}/promote",
        summary="Move to staging or production (production archives the previous one)",
    )
    async def promote(version_id: UUID, body: PromoteRequest) -> dict[str, Any]:
        return await controller.promote(version_id, body)

    @router.post("/{version_id}/archive", summary="Archive a version")
    async def archive(version_id: UUID) -> dict[str, Any]:
        return await controller.archive(version_id)

    return router

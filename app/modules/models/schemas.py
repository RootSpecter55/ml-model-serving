from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from app.core.schemas import ApiModel


class ModelVersionOut(ApiModel):
    id: UUID
    name: str
    version: int
    algorithm: str
    params: dict[str, Any]
    metrics: dict[str, float]
    feature_names: list[str]
    target_names: list[str]
    model_uri: str
    mlflow_run_id: str
    stage: str
    description: str | None
    created_at: datetime
    stage_changed_at: datetime


class ModelVersionUpdate(ApiModel):
    description: str | None = Field(max_length=2000)


class PromoteRequest(ApiModel):
    stage: Literal["staging", "production"]

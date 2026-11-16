from datetime import datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from app.core.schemas import ApiModel
from app.modules.models.schemas import ModelVersionOut
from app.modules.training.trainer import validate_params


class TrainingRequest(ApiModel):
    name: str | None = Field(default=None, pattern=r"^[a-z0-9-]{2,100}$")
    algorithm: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    params: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    random_state: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def check_params(self) -> Self:
        # Validates the hyperparameters for the chosen algorithm and fills in defaults.
        self.params = validate_params(self.algorithm, self.params)
        return self


class TrainingRunOut(ApiModel):
    id: UUID
    model_name: str
    algorithm: str
    params: dict[str, Any]
    test_size: float
    random_state: int
    status: str
    error: str | None
    mlflow_run_id: str | None
    model_version_id: UUID | None
    started_at: datetime
    finished_at: datetime | None


class TrainingResult(ApiModel):
    run: TrainingRunOut
    model_version: ModelVersionOut

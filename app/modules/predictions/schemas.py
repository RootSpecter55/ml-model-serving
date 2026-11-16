from datetime import datetime
from uuid import UUID

from pydantic import Field, StrictFloat, StrictInt

from app.core.schemas import ApiModel

# Strict numbers: "1.5" as a string is rejected rather than silently converted.
Number = StrictFloat | StrictInt


class PredictRequest(ApiModel):
    model: str | None = Field(default=None, pattern=r"^[a-z0-9-]{2,100}$")
    instances: list[dict[str, Number]] = Field(min_length=1)


class PredictionResult(ApiModel):
    prediction_id: UUID
    label: str
    probability: float
    probabilities: dict[str, float]


class ServedModel(ApiModel):
    id: UUID
    name: str
    version: int


class PredictResponse(ApiModel):
    model: ServedModel
    predictions: list[PredictionResult]


class PredictionOut(ApiModel):
    id: UUID
    model_version_id: UUID
    features: dict[str, float]
    label: str
    probabilities: dict[str, float]
    latency_ms: float
    created_at: datetime

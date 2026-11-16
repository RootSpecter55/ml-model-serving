import math
import time
import uuid
from typing import Any
from uuid import UUID

import pandas as pd

from app.core.db import Database
from app.core.errors import AppError, BadRequestError, NotFoundError
from app.core.pagination import PageParams
from app.modules.models.service import ModelRegistryService
from app.modules.predictions.model_cache import ModelCache
from app.modules.predictions.models import Prediction
from app.modules.predictions.repository import PredictionRepository
from app.modules.predictions.schemas import (
    PredictionOut,
    PredictionResult,
    PredictResponse,
    ServedModel,
)


def check_instances(instances: list[dict[str, float]], feature_names: list[str]) -> None:
    """Every instance must have exactly the model's features, each a finite number."""
    expected = set(feature_names)
    problems: list[dict[str, Any]] = []
    for index, instance in enumerate(instances):
        missing = sorted(expected - instance.keys())
        unexpected = sorted(instance.keys() - expected)
        not_finite = sorted(k for k, v in instance.items() if not math.isfinite(v))
        if missing or unexpected or not_finite:
            problems.append(
                {
                    "instance": index,
                    "missing": missing,
                    "unexpected": unexpected,
                    "notFinite": not_finite,
                }
            )
    if problems:
        raise AppError(
            400, "INVALID_FEATURES", "Instances do not match the model's features", problems
        )


class PredictionsService:
    def __init__(
        self,
        db: Database,
        predictions: PredictionRepository,
        registry: ModelRegistryService,
        cache: ModelCache,
        default_model_name: str,
        max_batch_size: int,
    ) -> None:
        self._db = db
        self._predictions = predictions
        self._registry = registry
        self._cache = cache
        self._default_name = default_model_name
        self._max_batch = max_batch_size

    async def predict(
        self, model_name: str | None, instances: list[dict[str, float]]
    ) -> PredictResponse:
        if len(instances) > self._max_batch:
            raise BadRequestError(
                "BATCH_TOO_LARGE", f"At most {self._max_batch} instances per request"
            )

        name = model_name or self._default_name
        version = await self._registry.get_production(name)
        if version is None:
            raise AppError(409, "NO_PRODUCTION_MODEL", f"No version of {name!r} is in production")
        check_instances(instances, version.feature_names)

        pipeline = await self._cache.get(version.name, version.id, version.model_uri)
        # Columns in the order the pipeline was trained on, whatever order the request used.
        frame = pd.DataFrame(
            [[row[f] for f in version.feature_names] for row in instances],
            columns=version.feature_names,
        )
        started = time.perf_counter()
        probabilities = pipeline.predict_proba(frame)
        latency_ms = (time.perf_counter() - started) * 1000 / len(instances)

        rows: list[Prediction] = []
        results: list[PredictionResult] = []
        for features, probs in zip(instances, probabilities, strict=True):
            by_label = {
                label: float(p) for label, p in zip(version.target_names, probs, strict=True)
            }
            label = max(by_label, key=by_label.__getitem__)
            row = Prediction(
                id=uuid.uuid4(),
                model_version_id=version.id,
                features=features,
                label=label,
                probabilities=by_label,
                latency_ms=latency_ms,
            )
            rows.append(row)
            results.append(
                PredictionResult(
                    prediction_id=row.id,
                    label=label,
                    probability=by_label[label],
                    probabilities=by_label,
                )
            )

        # Every prediction is logged with its exact input and the version that produced it.
        async with self._db.transaction() as session:
            await self._predictions.add_all(session, rows)

        return PredictResponse(
            model=ServedModel(id=version.id, name=version.name, version=version.version),
            predictions=results,
        )

    async def list(
        self, model_version_id: UUID | None, page: PageParams
    ) -> tuple[list[PredictionOut], int]:
        filters = [Prediction.model_version_id == model_version_id] if model_version_id else []
        async with self._db.transaction() as session:
            rows, total = await self._predictions.list(
                session,
                *filters,
                order_by=Prediction.created_at.desc(),
                offset=page.offset,
                limit=page.page_size,
            )
        return [PredictionOut.model_validate(p) for p in rows], total

    async def get(self, prediction_id: UUID) -> PredictionOut:
        async with self._db.transaction() as session:
            prediction = await self._predictions.get(session, prediction_id)
        if prediction is None:
            raise NotFoundError("Prediction")
        return PredictionOut.model_validate(prediction)

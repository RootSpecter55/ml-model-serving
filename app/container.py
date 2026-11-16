from dataclasses import dataclass
from typing import Any

from app.core.auth import make_key_auth
from app.core.config import Settings
from app.core.db import Database, get_engine
from app.modules.models.controller import ModelsController
from app.modules.models.repository import ModelVersionRepository
from app.modules.models.schemas import ModelVersionOut
from app.modules.models.service import ModelRegistryService
from app.modules.predictions.controller import PredictionsController
from app.modules.predictions.model_cache import ModelCache
from app.modules.predictions.repository import PredictionRepository
from app.modules.predictions.service import PredictionsService
from app.modules.training.controller import TrainingController
from app.modules.training.repository import TrainingRunRepository
from app.modules.training.service import TrainingService
from app.modules.training.tracker import ExperimentTracker


@dataclass
class Container:
    require_admin: Any
    require_api_key: Any
    cache: ModelCache
    training_service: TrainingService
    registry_service: ModelRegistryService
    training: TrainingController
    models: ModelsController
    predictions: PredictionsController


def build_container(settings: Settings) -> Container:
    """Composition root: wires every repository, service and controller in one place."""
    db = Database(get_engine())
    tracker = ExperimentTracker(
        settings.mlflow_tracking_uri, settings.mlflow_experiment, settings.mlflow_artifact_location
    )
    cache = ModelCache(tracker.load_model)

    async def warm_cache(version: ModelVersionOut) -> None:
        # Reload on promotion: load the new production model now, not on the next prediction.
        await cache.get(version.name, version.id, version.model_uri)

    registry = ModelRegistryService(db, ModelVersionRepository(), on_promoted=warm_cache)
    training = TrainingService(
        db, TrainingRunRepository(), registry, tracker, settings.default_model_name
    )
    predictions = PredictionsService(
        db,
        PredictionRepository(),
        registry,
        cache,
        settings.default_model_name,
        settings.max_batch_size,
    )

    return Container(
        require_admin=make_key_auth(settings.admin_api_key, "admin key"),
        require_api_key=make_key_auth(settings.api_key, "API key"),
        cache=cache,
        training_service=training,
        registry_service=registry,
        training=TrainingController(training),
        models=ModelsController(registry),
        predictions=PredictionsController(predictions),
    )

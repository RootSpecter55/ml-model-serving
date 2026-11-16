"""Imports every ORM model so Base.metadata is complete (Alembic and tests use this)."""

from app.core.db import Base
from app.modules.models.models import ModelVersion
from app.modules.predictions.models import Prediction
from app.modules.training.models import TrainingRun

__all__ = ["Base", "ModelVersion", "Prediction", "TrainingRun"]

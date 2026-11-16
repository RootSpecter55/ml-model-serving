from app.core.base_repository import BaseRepository
from app.modules.training.models import TrainingRun


class TrainingRunRepository(BaseRepository[TrainingRun]):
    model = TrainingRun

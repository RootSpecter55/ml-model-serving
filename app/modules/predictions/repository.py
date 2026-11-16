from sqlalchemy.ext.asyncio import AsyncSession

from app.core.base_repository import BaseRepository
from app.modules.predictions.models import Prediction


class PredictionRepository(BaseRepository[Prediction]):
    model = Prediction

    async def add_all(self, session: AsyncSession, predictions: list[Prediction]) -> None:
        session.add_all(predictions)
        await session.flush()

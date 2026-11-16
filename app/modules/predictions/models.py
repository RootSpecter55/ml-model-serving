import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Prediction(Base):
    """One logged prediction: the exact input, the output and which model version produced it."""

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_version_created", "model_version_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE")
    )
    features: Mapped[dict[str, Any]] = mapped_column(JSON)
    label: Mapped[str] = mapped_column(String(100))
    probabilities: Mapped[dict[str, float]] = mapped_column(JSON)
    latency_ms: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

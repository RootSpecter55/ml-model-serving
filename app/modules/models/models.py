import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, Index, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Stage(StrEnum):
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        # The database itself refuses a second production version of the same model.
        Index(
            "uq_model_versions_one_production",
            "name",
            unique=True,
            postgresql_where=text("stage = 'production'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[int]
    algorithm: Mapped[str] = mapped_column(String(50))
    params: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics: Mapped[dict[str, float]] = mapped_column(JSON)
    feature_names: Mapped[list[str]] = mapped_column(JSON)
    target_names: Mapped[list[str]] = mapped_column(JSON)
    model_uri: Mapped[str] = mapped_column(String(500))
    mlflow_run_id: Mapped[str] = mapped_column(String(100))
    stage: Mapped[str] = mapped_column(String(20), default=Stage.NONE)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    stage_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

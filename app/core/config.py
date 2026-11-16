from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every setting comes from environment variables and is validated at startup."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = Field(pattern=r"^postgresql\+asyncpg://")
    # Admin key: training and model registry changes. API key: predictions.
    admin_api_key: str = Field(min_length=32)
    api_key: str = Field(min_length=32)

    mlflow_tracking_uri: str = "sqlite:///./data/mlflow.db"
    mlflow_experiment: str = "model-serving"
    # Where artifacts go when the experiment is first created (ignored by a tracking server
    # that manages its own artifact store).
    mlflow_artifact_location: str | None = None

    default_model_name: str = "breast-cancer"
    max_batch_size: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def keys_differ(self) -> Self:
        if self.admin_api_key == self.api_key:
            raise ValueError("ADMIN_API_KEY and API_KEY must be different")
        return self


@lru_cache
def get_settings() -> Settings:
    """Singleton: settings are parsed and validated once, then shared."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment

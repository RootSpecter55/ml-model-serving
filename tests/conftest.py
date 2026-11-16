"""Test configuration. The environment is set before any app module reads settings."""

import os
import tempfile

_mlflow_dir = tempfile.mkdtemp(prefix="ml-serving-test-mlflow-")

os.environ.update(
    {
        "APP_ENV": "test",
        "LOG_LEVEL": "WARNING",
        "DATABASE_URL": os.environ.get(
            "TEST_DATABASE_URL", "postgresql+asyncpg://app:app@localhost:5444/ml_test"
        ),
        "ADMIN_API_KEY": "test-admin-key-that-is-at-least-32-characters",
        "API_KEY": "test-client-key-that-is-at-least-32-characters",
        # A throwaway local MLflow store per test session: no tracking server needed.
        "MLFLOW_TRACKING_URI": f"sqlite:///{_mlflow_dir}/mlflow.db",
        "MLFLOW_ARTIFACT_LOCATION": f"{_mlflow_dir}/artifacts",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
)

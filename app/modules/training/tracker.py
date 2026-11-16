from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class LoggedRun:
    run_id: str
    model_uri: str


class ExperimentTracker:
    """The only code that talks to MLflow: logs training runs and loads logged models.

    MLflow's client is synchronous; callers run these methods in a worker thread.
    """

    def __init__(self, tracking_uri: str, experiment: str, artifact_location: str | None) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._experiment = experiment
        self._artifact_location = artifact_location
        self._experiment_id: str | None = None

    def _get_experiment_id(self) -> str:
        if self._experiment_id is None:
            existing = mlflow.get_experiment_by_name(self._experiment)
            self._experiment_id = (
                existing.experiment_id
                if existing
                else mlflow.create_experiment(
                    self._experiment, artifact_location=self._artifact_location
                )
            )
        return self._experiment_id

    def log_training(
        self,
        *,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        pipeline: Pipeline,
        input_example: Any,
        tags: dict[str, str],
        trusted_types: list[str],
    ) -> LoggedRun:
        with mlflow.start_run(experiment_id=self._get_experiment_id(), run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags)
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=input_example,
                # Stored with the model and reused when loading it.
                skops_trusted_types=trusted_types or None,
            )
        return LoggedRun(run_id=run.info.run_id, model_uri=info.model_uri)

    def load_model(self, model_uri: str) -> Pipeline:
        # Only types declared trusted at save time can be deserialised (skops format).
        return mlflow.sklearn.load_model(model_uri)

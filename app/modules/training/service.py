import asyncio
import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import Database
from app.core.errors import AppError, NotFoundError
from app.core.pagination import PageParams
from app.modules.models.models import ModelVersion
from app.modules.models.schemas import ModelVersionOut
from app.modules.models.service import ModelRegistryService
from app.modules.training import trainer
from app.modules.training.models import RunStatus, TrainingRun
from app.modules.training.repository import TrainingRunRepository
from app.modules.training.schemas import TrainingRequest, TrainingResult, TrainingRunOut
from app.modules.training.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(
        self,
        db: Database,
        runs: TrainingRunRepository,
        registry: ModelRegistryService,
        tracker: ExperimentTracker,
        default_model_name: str,
    ) -> None:
        self._db = db
        self._runs = runs
        self._registry = registry
        self._tracker = tracker
        self._default_name = default_model_name

    async def train(self, request: TrainingRequest) -> TrainingResult:
        """Trains, logs to MLflow and registers a new version (stage "none").

        Runs inside the request: this dataset trains in seconds. CPU-bound work runs in a worker
        thread so the event loop keeps serving other requests.
        """
        name = request.name or self._default_name
        async with self._db.transaction() as session:
            run = await self._runs.add(
                session,
                TrainingRun(
                    model_name=name,
                    algorithm=request.algorithm,
                    params=request.params,
                    test_size=request.test_size,
                    random_state=request.random_state,
                    status=RunStatus.RUNNING,
                ),
            )

        try:
            result = await asyncio.to_thread(
                trainer.train,
                request.algorithm,
                request.params,
                request.test_size,
                request.random_state,
            )
            logged = await asyncio.to_thread(
                self._tracker.log_training,
                run_name=f"{name}-{request.algorithm}",
                params={
                    **request.params,
                    "algorithm": request.algorithm,
                    "test_size": request.test_size,
                    "random_state": request.random_state,
                },
                metrics=result.metrics,
                pipeline=result.pipeline,
                input_example=result.input_example,
                tags={"model_name": name, "training_run_id": str(run.id)},
            )
            async with self._db.transaction() as session:
                version = await self._registry.register(
                    session,
                    ModelVersion(
                        name=name,
                        algorithm=request.algorithm,
                        params=request.params,
                        metrics=result.metrics,
                        feature_names=result.feature_names,
                        target_names=result.target_names,
                        model_uri=logged.model_uri,
                        mlflow_run_id=logged.run_id,
                    ),
                )
                run = await self._finish(session, run.id, RunStatus.SUCCEEDED, version.id)
                run.mlflow_run_id = logged.run_id
        except Exception as exc:
            logger.exception("Training run %s failed", run.id)
            async with self._db.transaction() as session:
                await self._finish(session, run.id, RunStatus.FAILED, None, error=str(exc)[:2000])
            raise AppError(
                500, "TRAINING_FAILED", "Training failed", {"trainingRunId": str(run.id)}
            ) from exc

        return TrainingResult(
            run=TrainingRunOut.model_validate(run),
            model_version=ModelVersionOut.model_validate(version),
        )

    async def list(self, page: PageParams) -> tuple[list[TrainingRunOut], int]:
        async with self._db.transaction() as session:
            rows, total = await self._runs.list(
                session,
                order_by=TrainingRun.started_at.desc(),
                offset=page.offset,
                limit=page.page_size,
            )
        return [TrainingRunOut.model_validate(r) for r in rows], total

    async def get(self, run_id: UUID) -> TrainingRunOut:
        async with self._db.transaction() as session:
            run = await self._runs.get(session, run_id)
        if run is None:
            raise NotFoundError("Training run")
        return TrainingRunOut.model_validate(run)

    async def _finish(
        self,
        session: AsyncSession,
        run_id: UUID,
        status: RunStatus,
        version_id: UUID | None,
        error: str | None = None,
    ) -> TrainingRun:
        run = await self._runs.get(session, run_id)
        assert run is not None  # created at the start of this training
        run.status = status
        run.model_version_id = version_id
        run.error = error
        run.finished_at = datetime.now(UTC)
        return run

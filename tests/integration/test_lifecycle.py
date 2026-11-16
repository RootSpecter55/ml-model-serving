"""The full lifecycle: train -> register -> promote -> predict -> promote again -> swap."""

import os
from typing import Any

import mlflow
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.container import build_container
from app.core.config import get_settings
from app.core.db import get_engine
from app.main import create_app
from app.modules.training.tracker import ExperimentTracker
from tests.integration.conftest import ADMIN, API, promote, sample_instances, train


async def predict(client: AsyncClient, instances: list[dict[str, Any]], **extra: Any):
    return await client.post("/predict", headers=API, json={"instances": instances, **extra})


async def test_auth_keys_are_separate(client: AsyncClient) -> None:
    assert (await client.post("/training/runs", headers=API, json={})).status_code == 401
    assert (await client.get("/models")).status_code == 401
    response = await client.post("/predict", headers=ADMIN, json={"instances": [{}]})
    assert response.status_code == 401


async def test_training_registers_a_version_with_metrics_logged_to_mlflow(
    client: AsyncClient,
) -> None:
    result = await train(client, algorithm="logistic_regression", params={"C": 0.5})
    run, version = result["run"], result["modelVersion"]

    assert run["status"] == "succeeded" and run["modelVersionId"] == version["id"]
    assert version["version"] == 1 and version["stage"] == "none"
    assert version["params"] == {"C": 0.5, "max_iter": 1000}  # defaults filled in
    assert len(version["featureNames"]) == 30
    assert version["targetNames"] == ["malignant", "benign"]
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"):
        assert 0.0 <= version["metrics"][metric] <= 1.0

    logged = mlflow.get_run(version["mlflowRunId"])
    assert logged.data.metrics["accuracy"] == pytest.approx(version["metrics"]["accuracy"])
    assert logged.data.params["algorithm"] == "logistic_regression"

    second = await train(client, algorithm="random_forest", params={"n_estimators": 20})
    assert second["modelVersion"]["version"] == 2


async def test_rejects_invalid_hyperparameters(client: AsyncClient) -> None:
    bad = [
        {"algorithm": "logistic_regression", "params": {"C": 0}},
        {"algorithm": "logistic_regression", "params": {"n_estimators": 10}},
        {"algorithm": "gradient_boosting"},
    ]
    for body in bad:
        response = await client.post("/training/runs", headers=ADMIN, json=body)
        assert response.status_code == 400, body


async def test_predict_needs_a_production_version(client: AsyncClient) -> None:
    await train(client)
    response = await predict(client, sample_instances())
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NO_PRODUCTION_MODEL"


async def test_predicts_with_the_promoted_version_and_logs_every_prediction(
    client: AsyncClient,
) -> None:
    version = (await train(client))["modelVersion"]
    await promote(client, version["id"])

    response = await predict(client, sample_instances(3))
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["model"] == {"id": version["id"], "name": "breast-cancer", "version": 1}
    for prediction in data["predictions"]:
        assert prediction["label"] in ("malignant", "benign")
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
        assert prediction["probability"] == prediction["probabilities"][prediction["label"]]

    logged = await client.get(f"/predictions?modelVersionId={version['id']}", headers=API)
    assert logged.json()["meta"]["total"] == 3
    one = await client.get(f"/predictions/{data['predictions'][0]['predictionId']}", headers=API)
    assert one.json()["data"]["features"] == sample_instances(1)[0]


async def test_promoting_a_new_version_archives_the_old_one_and_swaps_the_served_model(
    app: Any, client: AsyncClient
) -> None:
    v1 = (await train(client, algorithm="logistic_regression"))["modelVersion"]
    v2 = (await train(client, algorithm="random_forest", params={"n_estimators": 10}))[
        "modelVersion"
    ]
    await promote(client, v1["id"])
    assert (await predict(client, sample_instances())).json()["data"]["model"]["version"] == 1

    await promote(client, v2["id"])
    assert (await client.get(f"/models/{v1['id']}", headers=ADMIN)).json()["data"][
        "stage"
    ] == "archived"
    assert (await predict(client, sample_instances())).json()["data"]["model"]["version"] == 2

    production = await client.get("/models?stage=production", headers=ADMIN)
    assert [v["version"] for v in production.json()["data"]] == [2]


async def test_another_instance_promoting_is_picked_up_on_the_next_prediction(
    client: AsyncClient,
) -> None:
    v1 = (await train(client))["modelVersion"]
    v2 = (await train(client))["modelVersion"]
    await promote(client, v1["id"])
    await predict(client, sample_instances())

    # Simulate another API instance promoting v2: change the database directly, so this
    # instance's cache is not told about it.
    async with get_engine().begin() as connection:
        await connection.execute(
            text("UPDATE model_versions SET stage = 'archived' WHERE id = :id"), {"id": v1["id"]}
        )
        await connection.execute(
            text("UPDATE model_versions SET stage = 'production' WHERE id = :id"), {"id": v2["id"]}
        )

    assert (await predict(client, sample_instances())).json()["data"]["model"]["version"] == 2


async def test_the_database_refuses_two_production_versions(client: AsyncClient) -> None:
    v1 = (await train(client))["modelVersion"]
    v2 = (await train(client))["modelVersion"]
    await promote(client, v1["id"])

    with pytest.raises(IntegrityError):
        async with get_engine().begin() as connection:
            await connection.execute(
                text("UPDATE model_versions SET stage = 'production' WHERE id = :id"),
                {"id": v2["id"]},
            )


async def test_production_versions_cannot_be_deleted(client: AsyncClient) -> None:
    v1 = (await train(client))["modelVersion"]
    v2 = (await train(client))["modelVersion"]
    await promote(client, v1["id"])

    assert (await client.delete(f"/models/{v1['id']}", headers=ADMIN)).status_code == 409
    assert (await client.delete(f"/models/{v2['id']}", headers=ADMIN)).status_code == 204


async def test_rejects_instances_that_do_not_match_the_model(client: AsyncClient) -> None:
    version = (await train(client))["modelVersion"]
    await promote(client, version["id"])
    [good] = sample_instances()

    missing = {k: v for k, v in good.items() if k != "mean radius"}
    extra = {**good, "favourite colour": 1.0}
    response = await predict(client, [good, missing, extra])
    assert response.status_code == 400
    details = response.json()["error"]["details"]
    assert details == [
        {"instance": 1, "missing": ["mean radius"], "unexpected": [], "notFinite": []},
        {"instance": 2, "missing": [], "unexpected": ["favourite colour"], "notFinite": []},
    ]

    as_string = {**good, "mean radius": "17.99"}
    assert (await predict(client, [as_string])).status_code == 400
    assert (await predict(client, [good] * 101)).status_code == 400


class FailingTracker(ExperimentTracker):
    def log_training(self, **kwargs: Any) -> Any:
        raise RuntimeError("tracking server unavailable")


async def test_a_failed_training_run_is_recorded() -> None:
    settings = get_settings()
    tracker = FailingTracker(
        settings.mlflow_tracking_uri, settings.mlflow_experiment, settings.mlflow_artifact_location
    )
    app = create_app(build_container(settings, tracker=tracker))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/training/runs", headers=ADMIN, json={})
        assert response.status_code == 500
        run_id = response.json()["error"]["details"]["trainingRunId"]
        run = (await c.get(f"/training/runs/{run_id}", headers=ADMIN)).json()["data"]
    assert run["status"] == "failed"
    assert "tracking server unavailable" in run["error"]
    assert os.environ["MLFLOW_TRACKING_URI"].startswith("sqlite:")

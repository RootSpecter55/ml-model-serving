import asyncio
import json
import uuid
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import AppError
from app.modules.predictions.model_cache import ModelCache
from app.modules.predictions.service import check_instances
from app.modules.training import trainer
from scripts.start_with_secrets import SecretsError, load_secrets


class TestTrainer:
    def test_fills_in_default_hyperparameters(self) -> None:
        assert trainer.validate_params("logistic_regression", {}) == {"C": 1.0, "max_iter": 1000}
        assert trainer.validate_params("random_forest", {"n_estimators": 5}) == {
            "n_estimators": 5,
            "max_depth": None,
        }

    @pytest.mark.parametrize(
        ("algorithm", "params"),
        [
            ("logistic_regression", {"C": -1}),
            ("logistic_regression", {"unknown": 1}),
            ("random_forest", {"n_estimators": 0}),
            ("svm", {}),
        ],
    )
    def test_rejects_invalid_hyperparameters(self, algorithm: str, params: dict) -> None:
        with pytest.raises(ValueError):
            trainer.validate_params(algorithm, params)

    def test_every_algorithm_declares_its_trusted_types(self) -> None:
        assert set(trainer.TRUSTED_TYPES) == set(trainer.ALGORITHMS)

    def test_trains_on_a_stratified_split_and_scores_the_held_out_part(self) -> None:
        result = trainer.train("logistic_regression", {"C": 1.0, "max_iter": 1000}, 0.25, 7)
        total = result.metrics["train_rows"] + result.metrics["test_rows"]
        assert total == 569  # every row of the bundled dataset
        assert result.metrics["test_rows"] == pytest.approx(569 * 0.25, abs=1)
        assert len(result.feature_names) == 30
        for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"):
            assert 0.0 <= result.metrics[name] <= 1.0

    def test_the_same_seed_gives_the_same_metrics(self) -> None:
        params = {"n_estimators": 10, "max_depth": 3}
        first = trainer.train("random_forest", params, 0.2, 1).metrics
        second = trainer.train("random_forest", params, 0.2, 1).metrics
        assert first == second


class TestCheckInstances:
    features = ("a", "b")

    def test_accepts_exact_features(self) -> None:
        check_instances([{"a": 1.0, "b": 2.0}, {"b": 0.0, "a": 3.0}], list(self.features))

    def test_reports_missing_unexpected_and_non_finite_values_per_instance(self) -> None:
        with pytest.raises(AppError) as error:
            check_instances(
                [{"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 3.0}, {"a": float("nan"), "b": 1.0}],
                list(self.features),
            )
        assert error.value.details == [
            {"instance": 1, "missing": ["b"], "unexpected": ["c"], "notFinite": []},
            {"instance": 2, "missing": [], "unexpected": [], "notFinite": ["a"]},
        ]


class TestModelCache:
    async def test_loads_each_version_once_and_reloads_when_the_version_changes(self) -> None:
        loads: list[str] = []
        cache = ModelCache(lambda uri: loads.append(uri) or f"model from {uri}")
        v1, v2 = uuid.uuid4(), uuid.uuid4()

        assert await cache.get("m", v1, "uri-1") == "model from uri-1"
        assert await cache.get("m", v1, "uri-1") == "model from uri-1"
        assert await cache.get("m", v2, "uri-2") == "model from uri-2"
        assert loads == ["uri-1", "uri-2"]
        assert cache.cached_version("m") == v2

    async def test_concurrent_first_requests_share_one_load(self) -> None:
        loads: list[str] = []
        cache = ModelCache(lambda uri: loads.append(uri) or object())
        version = uuid.uuid4()
        results = await asyncio.gather(*(cache.get("m", version, "uri") for _ in range(20)))
        assert loads == ["uri"]
        assert all(r is results[0] for r in results)


class TestSettings:
    def test_admin_and_client_keys_must_differ(self) -> None:
        with pytest.raises(ValidationError, match="must be different"):
            Settings(
                database_url="postgresql+asyncpg://u:p@localhost/db",
                admin_api_key="k" * 32,
                api_key="k" * 32,
            )


class TestStartWithSecrets:
    def client(self, secret: str | None) -> MagicMock:
        client = MagicMock()
        client.get_secret_value.return_value = {"SecretString": secret}
        return client

    def test_copies_the_secret_into_the_environment(self) -> None:
        env = {"SECRETS_SOURCE": "aws", "SECRET_ID": "ml/prod"}
        secret = {"DATABASE_URL": "postgresql+asyncpg://x", "ADMIN_API_KEY": "a", "API_KEY": "b"}
        load_secrets(env, self.client(json.dumps(secret)))
        assert env["API_KEY"] == "b"

    def test_names_missing_required_keys(self) -> None:
        with pytest.raises(SecretsError, match="ADMIN_API_KEY, API_KEY"):
            load_secrets(
                {"SECRETS_SOURCE": "aws", "SECRET_ID": "s"},
                self.client(json.dumps({"DATABASE_URL": "x"})),
            )

    def test_requires_a_secret_id(self) -> None:
        with pytest.raises(SecretsError, match="SECRET_ID is required"):
            load_secrets({"SECRETS_SOURCE": "aws"}, self.client("{}"))

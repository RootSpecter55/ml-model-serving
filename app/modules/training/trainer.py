"""Pure machine-learning code: dataset, pipeline construction, training and evaluation.

Nothing here knows about HTTP, the database or MLflow, so it is easy to test on its own.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    C: float = Field(default=1.0, gt=0, le=1000)
    max_iter: int = Field(default=1000, ge=50, le=10_000)


class RandomForestParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = Field(default=100, ge=1, le=500)
    max_depth: int | None = Field(default=None, ge=1, le=50)


def _logistic_regression(params: LogisticRegressionParams, seed: int) -> Pipeline:
    # Scaling matters for logistic regression: the features have very different ranges.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=params.C, max_iter=params.max_iter, random_state=seed)),
        ]
    )


def _random_forest(params: RandomForestParams, seed: int) -> Pipeline:
    forest = RandomForestClassifier(
        n_estimators=params.n_estimators, max_depth=params.max_depth, random_state=seed
    )
    return Pipeline([("model", forest)])


# Factory: maps an algorithm name to its parameter schema and pipeline builder, so adding an
# algorithm is one entry here and the rest of the app only passes a name around.
ALGORITHMS: dict[str, tuple[type[BaseModel], Callable[[Any, int], Pipeline]]] = {
    "logistic_regression": (LogisticRegressionParams, _logistic_regression),
    "random_forest": (RandomForestParams, _random_forest),
}


# Models are saved with skops, which refuses to load internal scikit-learn types unless they are
# listed as trusted. Tree models need sklearn's Tree type. Only this service creates these files.
TRUSTED_TYPES: dict[str, list[str]] = {
    "logistic_regression": [],
    "random_forest": ["sklearn.tree._tree.Tree"],
}


def validate_params(algorithm: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validates hyperparameters for the algorithm and fills in defaults. Raises ValueError."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {sorted(ALGORITHMS)}")
    schema, _ = ALGORITHMS[algorithm]
    return schema.model_validate(params).model_dump()


def build_pipeline(algorithm: str, params: dict[str, Any], seed: int) -> Pipeline:
    schema, builder = ALGORITHMS[algorithm]
    return builder(schema.model_validate(params), seed)


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """The Wisconsin breast cancer dataset bundled with scikit-learn (no download)."""
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data, bunch.target, [str(name) for name in bunch.target_names]


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_names: list[str]
    target_names: list[str]
    input_example: pd.DataFrame


def train(algorithm: str, params: dict[str, Any], test_size: float, seed: int) -> TrainResult:
    """Fits on a stratified training split and scores on the held-out test split.

    Precision, recall and F1 are macro-averaged over both classes, so neither class is treated
    as "the positive one". ROC AUC uses the predicted probability of class 1.
    """
    features, target, target_names = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=seed, stratify=target
    )
    pipeline = build_pipeline(algorithm, params, seed).fit(x_train, y_train)

    predicted = pipeline.predict(x_test)
    probability_of_class_1 = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_macro": float(precision_score(y_test, predicted, average="macro")),
        "recall_macro": float(recall_score(y_test, predicted, average="macro")),
        "f1_macro": float(f1_score(y_test, predicted, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, probability_of_class_1)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
    }
    return TrainResult(
        pipeline=pipeline,
        metrics=metrics,
        feature_names=[str(c) for c in features.columns],
        target_names=target_names,
        input_example=x_train.head(2),
    )

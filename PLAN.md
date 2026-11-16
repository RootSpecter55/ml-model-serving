# PLAN — ml-model-serving

Train, version, promote and serve a scikit-learn classifier behind an API.

- **Data:** the Wisconsin breast cancer dataset that ships inside scikit-learn (`load_breast_cancer`), so there are no downloads.
- **Tracking:** every training run is recorded in MLflow (params, metrics, model artifact).
- **Registry:** a model registry in Postgres decides which version serves predictions.
- **Serving:** the promoted version is cached in memory and swapped when a new version is promoted.

**Stack:** Python 3.12, FastAPI, SQLAlchemy 2 (async), Alembic, Pydantic v2, scikit-learn, MLflow, pytest, ruff, uv.

## Modules

| Module | Responsibility |
|---|---|
| training | `POST /training/runs` (and the CLI `scripts/train.py`): split the data, fit a pipeline, compute test-set metrics, log to MLflow, register a new model version. Run history is in `GET /training/runs`. |
| models | Registry: list and get versions (filter by name or stage), update the description, delete (never the production version). `POST /models/{id}/promote` with `{stage: staging\|production}` and `POST /models/{id}/archive`. Promoting to production archives the previous production version in the same transaction. |
| predictions | `POST /predict`: validate the input against the promoted version's feature names, predict, return labels and probabilities, and log every prediction. `GET /predictions` lists and filters them. |

**Algorithms:** `logistic_regression` (StandardScaler + LogisticRegression) and `random_forest`, each with its own validated hyperparameters. A small factory maps the algorithm name to a pipeline builder.

**Singletons:** settings, the database engine, and `ModelCache`, the loaded model.

## Data model

```
training_runs   id, model_name, algorithm, params JSON, test_size, random_state, status
                (running|succeeded|failed), error?, mlflow_run_id?, model_version_id?, started_at, finished_at?
model_versions  id, name, version (unique per name), algorithm, params JSON, metrics JSON,
                feature_names JSON, target_names JSON, model_uri (MLflow), mlflow_run_id,
                stage (none|staging|production|archived), description?, created_at, stage_changed_at
                partial unique index: one production version per name
predictions     id, model_version_id → model_versions, features JSON, label, probabilities JSON,
                latency_ms, created_at
```

## Key mechanisms

- **One production version per model:** promotion archives the old production version and promotes the new one in a single transaction. A partial unique index (`WHERE stage = 'production'`) makes the database itself refuse two.
- **Reload on promotion:** `ModelCache` holds `(version id, pipeline)` per model name.
  - Each prediction reads the current production version id. That's a tiny indexed query, so every API instance notices a promotion made through any other instance.
  - The cache reloads from MLflow when the id changes.
  - Promoting through this instance also loads the new model straight away, so the first prediction after it doesn't wait.
- **Input validation:** request instances must have exactly the model's feature names, and every value must be a finite number. Batches can hold up to `MAX_BATCH_SIZE` instances.
- **Metrics:** accuracy, precision, recall, F1 and ROC AUC on a held-out test split (stratified, with a seeded `random_state`), stored on the version and logged to MLflow. These are computed by the code; the README quotes none of them.

## Auth

`ADMIN_API_KEY` covers training and the registry; `API_KEY` covers predictions. Both are bearer tokens from config.

## Tests

- **Unit:** pipeline factory and params validation, the trainer's metrics and split, input validation, cache reload logic (with a fake loader), `start_with_secrets`.
- **Integration** (Postgres + MLflow on local SQLite): train → registry, promote/archive rules, the unique-production guarantee, predict with the promoted model, swapping the served model after promotion, prediction logging, and auth.

## Deployment files

The same set as the other repos. Docker-compose adds an MLflow tracking server (UI at :5001). Postgres host port: **5444**.

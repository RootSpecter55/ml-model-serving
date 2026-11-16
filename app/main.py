from fastapi import FastAPI

from app.container import Container, build_container
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.pagination import data_envelope
from app.modules.models.routes import models_router
from app.modules.predictions.routes import predictions_router
from app.modules.training.routes import training_router


def create_app(container: Container | None = None) -> FastAPI:
    configure_logging()
    c = container or build_container(get_settings())

    app = FastAPI(
        title="ML model serving",
        version="1.0.0",
        description="Train, version, promote and serve a scikit-learn classifier.",
    )
    register_error_handlers(app)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, object]:
        return data_envelope({"status": "ok"})

    app.include_router(training_router(c.training, c.require_admin))
    app.include_router(models_router(c.models, c.require_admin))
    app.include_router(predictions_router(c.predictions, c.require_api_key))
    return app

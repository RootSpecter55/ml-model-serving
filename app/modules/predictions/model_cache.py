import asyncio
import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

ModelLoader = Callable[[str], Any]  # model URI -> fitted pipeline (synchronous)


class ModelCache:
    """Singleton: keeps the loaded production pipeline in memory, one per model name.

    Loading from MLflow takes a while, so it happens once per version, not per request. Every
    prediction passes the id of the version that is currently in production; when that id
    differs from the cached one (a promotion happened, through this instance or any other),
    the new version is loaded and swapped in.
    """

    def __init__(self, loader: ModelLoader) -> None:
        self._loader = loader
        self._loaded: dict[str, tuple[UUID, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, name: str, version_id: UUID, model_uri: str) -> Any:
        cached = self._loaded.get(name)
        if cached and cached[0] == version_id:
            return cached[1]
        # One load at a time: concurrent first requests wait for the same load, not repeat it.
        async with self._lock:
            cached = self._loaded.get(name)
            if cached and cached[0] == version_id:
                return cached[1]
            logger.info("Loading %s version %s from %s", name, version_id, model_uri)
            pipeline = await asyncio.to_thread(self._loader, model_uri)
            self._loaded[name] = (version_id, pipeline)
            return pipeline

    def cached_version(self, name: str) -> UUID | None:
        cached = self._loaded.get(name)
        return cached[0] if cached else None

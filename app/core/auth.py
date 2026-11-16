from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Header

from app.core.errors import UnauthorizedError
from app.core.security import bearer_token, keys_match


def make_key_auth(expected_key: str, label: str) -> Callable[..., Awaitable[None]]:
    """FastAPI dependency that only lets requests with `Authorization: Bearer <key>` through."""

    async def require_key(authorization: Annotated[str | None, Header()] = None) -> None:
        token = bearer_token(authorization)
        if not token or not keys_match(token, expected_key):
            raise UnauthorizedError(f"Invalid {label}")

    return require_key

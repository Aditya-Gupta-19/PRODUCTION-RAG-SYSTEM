import secrets

from fastapi import Header, HTTPException, status

from src.config import settings


async def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    """FastAPI dependency: reject the request unless the X-API-Key header matches.

    ``secrets.compare_digest`` is constant-time, so a wrong key cannot be
    recovered byte-by-byte from response-timing differences.
    """
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

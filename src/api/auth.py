import secrets

from fastapi import Header, HTTPException, status

from src.config import settings


async def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    """FastAPI dependency: reject the request unless the X-API-Key header matches.

    ``secrets.compare_digest`` is constant-time, so a wrong key cannot be
    recovered byte-by-byte from response-timing differences. Compared as bytes
    so a non-ASCII header value returns 401, not a 500 from ``compare_digest``.
    """
    if not secrets.compare_digest(x_api_key.encode("utf-8", "ignore"), settings.api_key.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

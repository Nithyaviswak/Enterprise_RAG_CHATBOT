"""Rate Limiting Middleware using SlowAPI."""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.config import Config
from starlette.requests import Request
from starlette.responses import JSONResponse


# SlowAPI reads ".env" via starlette Config using the platform default encoding,
# which crashes on Windows when .env contains UTF-8 emoji. Point it at a
# non-existent file (starlette Config then loads nothing), so env loading is
# handled solely by pydantic-settings (config.py).
limiter = Limiter(
    key_func=get_remote_address,
    headers_enabled=True,
    config_filename=".env.ignore",
)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit exceeded handler."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Too many requests. {str(exc.detail)}",
        },
    )

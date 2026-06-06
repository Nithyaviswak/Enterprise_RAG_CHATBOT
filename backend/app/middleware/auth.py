"""Firebase Authentication Middleware (Optional)."""

import logging
from typing import Optional

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

logger = logging.getLogger(__name__)

# Routes that don't require authentication
PUBLIC_ROUTES = {"/api/health", "/docs", "/redoc", "/openapi.json"}


class FirebaseAuthMiddleware(BaseHTTPMiddleware):
    """Optional Firebase Auth middleware.

    Only active when firebase_credentials_path is configured.
    Verifies Firebase ID tokens from the Authorization header.
    """

    def __init__(self, app):
        super().__init__(app)
        self.enabled = False
        settings = get_settings()

        if settings.firebase_credentials_path:
            try:
                import firebase_admin
                from firebase_admin import credentials

                cred = credentials.Certificate(settings.firebase_credentials_path)
                firebase_admin.initialize_app(cred)
                self.enabled = True
                logger.info("Firebase Auth middleware enabled")
            except Exception as e:
                logger.warning(f"Firebase Auth setup failed: {e}. Auth is disabled.")

    async def dispatch(self, request: Request, call_next):
        # Skip auth for preflight and public routes
        if request.method == "OPTIONS":
            return await call_next(request)

        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(route) for route in PUBLIC_ROUTES):
            return await call_next(request)

        # Verify token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

        token = auth_header.split("Bearer ")[1]
        try:
            from firebase_admin import auth

            decoded = auth.verify_id_token(token)
            request.state.user_id = decoded.get("uid")
            request.state.user_email = decoded.get("email")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

        return await call_next(request)

"""API Key authentication middleware."""

import time
import hashlib
import secrets
from typing import Optional, Set, Callable
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse

from backend.config import config


class APIKeyMiddleware(BaseHTTPMiddleware):
    EXEMPT_PATHS = {"/", "/health", "/health/llm", "/docs", "/redoc", "/openapi.json", "/llm-models", "/metrics"}

    def __init__(
        self,
        app,
        api_keys: Set[str] = None,
        header_name: str = None,
        exempt_paths: Set[str] = None,
        enabled: bool = None
    ):
        super().__init__(app)
        self.enabled = enabled if enabled is not None else config.API_KEY_ENABLED
        self.api_keys = api_keys or config.API_KEYS
        self.header_name = header_name or config.API_KEY_HEADER
        self.exempt_paths = exempt_paths or self.EXEMPT_PATHS
        self._key_hashes = {self._hash_key(k) for k in self.api_keys if k}

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def _verify_key(self, key: str) -> bool:
        if not key:
            return False
        key_hash = self._hash_key(key)
        return secrets.compare_digest(key_hash, key_hash) and key_hash in self._key_hashes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        if path in self.exempt_paths:
            return await call_next(request)

        api_key = request.headers.get(self.header_name)
        
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key required", "header": self.header_name}
            )

        if not self._verify_key(api_key):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Invalid API key"}
            )

        return await call_next(request)


api_key_header = APIKeyHeader(name=config.API_KEY_HEADER, auto_error=False)


async def verify_api_key(request: Request, api_key: Optional[str] = None) -> bool:
    if not config.API_KEY_ENABLED:
        return True

    if api_key is None:
        api_key = request.headers.get(config.API_KEY_HEADER)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": f"ApiKey header={config.API_KEY_HEADER}"}
        )

    if api_key not in config.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return True


def generate_api_key(prefix: str = "aud") -> str:
    random_bytes = secrets.token_bytes(32)
    key = secrets.token_urlsafe(32)
    return f"{prefix}_{key}"


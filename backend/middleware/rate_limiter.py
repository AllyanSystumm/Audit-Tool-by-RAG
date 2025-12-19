"""Rate limiting middleware using token bucket algorithm."""

import time
import threading
from typing import Dict, Callable, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse

from backend.config import config


@dataclass
class TokenBucket:
    capacity: float
    tokens: float
    refill_rate: float
    last_refill: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def consume(self, tokens: float = 1.0) -> bool:
        with self.lock:
            now = time.time()
            time_passed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: float = 1.0) -> float:
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate


class TokenBucketRateLimiter:
    def __init__(
        self,
        requests_per_window: int = None,
        window_seconds: int = None,
        burst: int = None,
        cleanup_interval: int = 300
    ):
        self.requests_per_window = requests_per_window or config.RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or config.RATE_LIMIT_WINDOW_SECONDS
        self.burst = burst or config.RATE_LIMIT_BURST
        
        self.refill_rate = self.requests_per_window / self.window_seconds
        self.capacity = self.burst
        
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._cleanup_interval = cleanup_interval

    def _get_bucket(self, key: str) -> TokenBucket:
        with self._lock:
            self._maybe_cleanup()
            
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    capacity=self.capacity,
                    tokens=self.capacity,
                    refill_rate=self.refill_rate
                )
            return self._buckets[key]

    def _maybe_cleanup(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        stale_keys = []
        
        for key, bucket in self._buckets.items():
            if now - bucket.last_refill > self._cleanup_interval * 2:
                stale_keys.append(key)
        
        for key in stale_keys:
            del self._buckets[key]

    def is_allowed(self, key: str, tokens: float = 1.0) -> bool:
        bucket = self._get_bucket(key)
        return bucket.consume(tokens)

    def get_wait_time(self, key: str, tokens: float = 1.0) -> float:
        bucket = self._get_bucket(key)
        return bucket.get_wait_time(tokens)

    def get_remaining(self, key: str) -> int:
        bucket = self._get_bucket(key)
        with bucket.lock:
            now = time.time()
            time_passed = now - bucket.last_refill
            current_tokens = min(bucket.capacity, bucket.tokens + time_passed * bucket.refill_rate)
            return int(current_tokens)

    def reset(self, key: str):
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]


class RateLimiterMiddleware(BaseHTTPMiddleware):
    EXEMPT_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}

    def __init__(
        self,
        app,
        limiter: TokenBucketRateLimiter = None,
        key_func: Callable[[Request], str] = None,
        exempt_paths: set = None,
        enabled: bool = None
    ):
        super().__init__(app)
        self.enabled = enabled if enabled is not None else config.RATE_LIMIT_ENABLED
        self.limiter = limiter or TokenBucketRateLimiter()
        self.key_func = key_func or self._default_key_func
        self.exempt_paths = exempt_paths or self.EXEMPT_PATHS

    def _default_key_func(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"{client_ip}:{request.url.path}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        if path in self.exempt_paths:
            return await call_next(request)

        key = self.key_func(request)
        
        if not self.limiter.is_allowed(key):
            wait_time = self.limiter.get_wait_time(key)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": round(wait_time, 2)
                },
                headers={
                    "Retry-After": str(int(wait_time) + 1),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + wait_time))
                }
            )

        response = await call_next(request)
        
        remaining = self.limiter.get_remaining(key)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.capacity)
        
        return response


def check_rate_limit(request: Request, limiter: Optional[TokenBucketRateLimiter] = None) -> bool:
    if not config.RATE_LIMIT_ENABLED:
        return True

    if limiter is None:
        limiter = TokenBucketRateLimiter()

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    key = f"{client_ip}:{request.url.path}"

    if not limiter.is_allowed(key):
        wait_time = limiter.get_wait_time(key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {wait_time:.1f} seconds",
            headers={"Retry-After": str(int(wait_time) + 1)}
        )

    return True


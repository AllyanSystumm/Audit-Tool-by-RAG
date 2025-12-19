"""Structured logging and metrics utilities."""

import json
import time
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.config import config


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class StructuredLogger:
    def __init__(self, name: str, use_structured: bool = None):
        self._logger = logging.getLogger(name)
        self._use_structured = use_structured if use_structured is not None else config.STRUCTURED_LOGGING
        
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            
            if self._use_structured:
                handler.setFormatter(StructuredFormatter())
            else:
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ))
            
            self._logger.addHandler(handler)
            self._logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    def _log(self, level: int, message: str, **kwargs):
        extra = {"extra_fields": kwargs} if kwargs else {}
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        self._logger.exception(message, extra={"extra_fields": kwargs} if kwargs else {})


_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


@dataclass
class RequestMetrics:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    method: str = ""
    path: str = ""
    status_code: int = 0
    duration_ms: float = 0.0
    client_ip: str = ""
    user_agent: str = ""
    content_length: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}


class MetricsCollector:
    def __init__(self):
        self._request_count = 0
        self._error_count = 0
        self._total_duration_ms = 0.0
        self._status_counts: Dict[int, int] = {}
        self._endpoint_counts: Dict[str, int] = {}
        self._start_time = time.time()

    def record_request(self, metrics: RequestMetrics):
        self._request_count += 1
        self._total_duration_ms += metrics.duration_ms
        
        status_group = (metrics.status_code // 100) * 100
        self._status_counts[status_group] = self._status_counts.get(status_group, 0) + 1
        
        if metrics.status_code >= 400:
            self._error_count += 1
        
        self._endpoint_counts[metrics.path] = self._endpoint_counts.get(metrics.path, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time
        avg_duration = self._total_duration_ms / self._request_count if self._request_count > 0 else 0
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self._request_count,
            "error_count": self._error_count,
            "error_rate": round(self._error_count / self._request_count * 100, 2) if self._request_count > 0 else 0,
            "avg_duration_ms": round(avg_duration, 2),
            "status_counts": dict(self._status_counts),
            "top_endpoints": dict(sorted(self._endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }

    def reset(self):
        self._request_count = 0
        self._error_count = 0
        self._total_duration_ms = 0.0
        self._status_counts.clear()
        self._endpoint_counts.clear()
        self._start_time = time.time()


_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _metrics_collector


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app, logger: StructuredLogger = None, skip_paths: set = None, collect_metrics: bool = True):
        super().__init__(app)
        self.logger = logger or get_logger("api.requests")
        self.skip_paths = skip_paths or self.SKIP_PATHS
        self.collect_metrics = collect_metrics

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.skip_paths:
            return await call_next(request)

        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        forwarded = request.headers.get("X-Forwarded-For")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")

        metrics = RequestMetrics(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            user_agent=request.headers.get("User-Agent", "")[:100]
        )

        try:
            response = await call_next(request)
            metrics.status_code = response.status_code
            
            content_length = response.headers.get("Content-Length")
            if content_length:
                metrics.content_length = int(content_length)
            
        except Exception as e:
            metrics.status_code = 500
            metrics.error = str(e)[:200]
            raise
        finally:
            metrics.duration_ms = round((time.time() - start_time) * 1000, 2)
            
            if self.collect_metrics:
                _metrics_collector.record_request(metrics)

            log_level = "info"
            if metrics.status_code >= 500:
                log_level = "error"
            elif metrics.status_code >= 400:
                log_level = "warning"

            getattr(self.logger, log_level)(
                f"{metrics.method} {metrics.path} -> {metrics.status_code} ({metrics.duration_ms}ms)",
                **metrics.to_dict()
            )

        response.headers["X-Request-ID"] = request_id
        return response


@contextmanager
def log_operation(logger: StructuredLogger, operation: str, **extra):
    start_time = time.time()
    logger.info(f"Starting: {operation}", operation=operation, **extra)
    
    try:
        yield
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Completed: {operation}", operation=operation, duration_ms=duration_ms, success=True, **extra)
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Failed: {operation}", operation=operation, duration_ms=duration_ms, success=False, error=str(e), **extra)
        raise


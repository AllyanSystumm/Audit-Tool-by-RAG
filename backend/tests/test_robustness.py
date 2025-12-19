import time
import threading
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

from backend.config import config
from backend.ingestion.embedder import LRUCache, EmbeddingGenerator
from backend.middleware.auth import APIKeyMiddleware, generate_api_key
from backend.middleware.rate_limiter import TokenBucketRateLimiter, RateLimiterMiddleware
from backend.middleware.logging_utils import StructuredLogger, MetricsCollector, RequestMetrics, get_logger


class TestLRUCache:
    def test_basic_put_get(self):
        cache = LRUCache(max_size=3)
        cache.put("a", np.array([1, 2, 3]))
        result = cache.get("a")
        assert result is not None
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_eviction_order(self):
        cache = LRUCache(max_size=3)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.put("c", np.array([3]))
        
        cache.get("a")
        
        cache.put("d", np.array([4]))
        
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None
        assert cache.get("d") is not None

    def test_update_moves_to_end(self):
        cache = LRUCache(max_size=2)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.put("a", np.array([10]))
        cache.put("c", np.array([3]))
        
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_stats(self):
        cache = LRUCache(max_size=2)
        cache.put("a", np.array([1]))
        cache.get("a")
        cache.get("b")
        
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0
        assert stats["size"] == 1

    def test_thread_safety(self):
        cache = LRUCache(max_size=100)
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    cache.put(f"{thread_id}_{i}", np.array([i]))
            except Exception as e:
                errors.append(e)

        def reader(thread_id):
            try:
                for i in range(50):
                    cache.get(f"{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestFilenameValidation:
    def test_valid_filename(self):
        valid, error = config.validate_filename("test_document.docx")
        assert valid is True
        assert error is None

    def test_path_traversal_rejected(self):
        valid, error = config.validate_filename("../../../etc/passwd")
        assert valid is False
        assert "invalid" in error.lower()

    def test_absolute_path_rejected(self):
        valid, error = config.validate_filename("/etc/passwd")
        assert valid is False

    def test_invalid_extension_rejected(self):
        valid, error = config.validate_filename("malware.exe")
        assert valid is False
        assert "extension" in error.lower()

    def test_special_characters_rejected(self):
        valid, error = config.validate_filename("file<script>.docx")
        assert valid is False

    def test_empty_filename_rejected(self):
        valid, error = config.validate_filename("")
        assert valid is False

    def test_long_filename_rejected(self):
        long_name = "a" * 300 + ".docx"
        valid, error = config.validate_filename(long_name)
        assert valid is False
        assert "length" in error.lower()


class TestFilenameSanitization:
    def test_removes_path_traversal(self):
        result = config.sanitize_filename("../../evil.docx")
        assert ".." not in result

    def test_removes_special_characters(self):
        result = config.sanitize_filename("file<script>alert('xss').docx")
        assert "<" not in result
        assert ">" not in result
        assert "'" not in result

    def test_preserves_extension(self):
        result = config.sanitize_filename("test.docx")
        assert result.endswith(".docx")

    def test_handles_empty_name(self):
        result = config.sanitize_filename(".docx")
        assert result == "unnamed.docx"

    def test_truncates_long_names(self):
        long_name = "a" * 300 + ".docx"
        result = config.sanitize_filename(long_name)
        assert len(result) <= config.FILENAME_MAX_LENGTH


class TestTokenBucketRateLimiter:
    def test_allows_within_limit(self):
        limiter = TokenBucketRateLimiter(requests_per_window=10, window_seconds=1, burst=5)
        
        for _ in range(5):
            assert limiter.is_allowed("test_key") is True

    def test_blocks_over_limit(self):
        limiter = TokenBucketRateLimiter(requests_per_window=10, window_seconds=60, burst=3)
        
        for _ in range(3):
            limiter.is_allowed("test_key")
        
        assert limiter.is_allowed("test_key") is False

    def test_refills_over_time(self):
        limiter = TokenBucketRateLimiter(requests_per_window=100, window_seconds=1, burst=2)
        
        limiter.is_allowed("test_key")
        limiter.is_allowed("test_key")
        assert limiter.is_allowed("test_key") is False
        
        time.sleep(0.05)
        assert limiter.is_allowed("test_key") is True

    def test_different_keys_independent(self):
        limiter = TokenBucketRateLimiter(requests_per_window=10, window_seconds=60, burst=2)
        
        limiter.is_allowed("key1")
        limiter.is_allowed("key1")
        assert limiter.is_allowed("key1") is False
        
        assert limiter.is_allowed("key2") is True

    def test_get_wait_time(self):
        limiter = TokenBucketRateLimiter(requests_per_window=10, window_seconds=1, burst=1)
        
        limiter.is_allowed("test_key")
        wait_time = limiter.get_wait_time("test_key")
        
        assert wait_time > 0
        assert wait_time <= 0.2

    def test_get_remaining(self):
        limiter = TokenBucketRateLimiter(requests_per_window=10, window_seconds=60, burst=5)
        
        assert limiter.get_remaining("test_key") == 5
        limiter.is_allowed("test_key")
        assert limiter.get_remaining("test_key") == 4


class TestAPIKeyGeneration:
    def test_generates_unique_keys(self):
        key1 = generate_api_key()
        key2 = generate_api_key()
        assert key1 != key2

    def test_has_prefix(self):
        key = generate_api_key(prefix="test")
        assert key.startswith("test_")

    def test_sufficient_length(self):
        key = generate_api_key()
        assert len(key) >= 40


class TestMetricsCollector:
    def test_records_request(self):
        collector = MetricsCollector()
        
        metrics = RequestMetrics(
            method="GET",
            path="/test",
            status_code=200,
            duration_ms=50.0
        )
        collector.record_request(metrics)
        
        stats = collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["error_count"] == 0

    def test_counts_errors(self):
        collector = MetricsCollector()
        
        for status in [200, 200, 404, 500, 500]:
            metrics = RequestMetrics(
                method="GET",
                path="/test",
                status_code=status,
                duration_ms=10.0
            )
            collector.record_request(metrics)
        
        stats = collector.get_stats()
        assert stats["total_requests"] == 5
        assert stats["error_count"] == 3

    def test_calculates_avg_duration(self):
        collector = MetricsCollector()
        
        for duration in [100.0, 200.0, 300.0]:
            metrics = RequestMetrics(
                method="GET",
                path="/test",
                status_code=200,
                duration_ms=duration
            )
            collector.record_request(metrics)
        
        stats = collector.get_stats()
        assert stats["avg_duration_ms"] == 200.0

    def test_tracks_endpoints(self):
        collector = MetricsCollector()
        
        for path in ["/a", "/a", "/b"]:
            metrics = RequestMetrics(
                method="GET",
                path=path,
                status_code=200,
                duration_ms=10.0
            )
            collector.record_request(metrics)
        
        stats = collector.get_stats()
        assert stats["top_endpoints"]["/a"] == 2
        assert stats["top_endpoints"]["/b"] == 1

    def test_reset(self):
        collector = MetricsCollector()
        
        metrics = RequestMetrics(method="GET", path="/test", status_code=200, duration_ms=10.0)
        collector.record_request(metrics)
        
        collector.reset()
        stats = collector.get_stats()
        assert stats["total_requests"] == 0


class TestStructuredLogger:
    def test_creates_logger(self):
        logger = get_logger("test.module")
        assert logger is not None

    def test_logs_without_error(self):
        logger = StructuredLogger("test", use_structured=False)
        logger.info("Test message")
        logger.warning("Warning message")
        logger.error("Error message")


class TestLLMClientRetry:
    def test_retry_decorator_exists(self):
        from backend.generation.llm_client import retry_with_exponential_backoff
        assert callable(retry_with_exponential_backoff)

    def test_custom_exceptions(self):
        from backend.generation.llm_client import (
            LLMClientError,
            LLMConnectionError,
            LLMRateLimitError,
            LLMTimeoutError
        )
        
        assert issubclass(LLMConnectionError, LLMClientError)
        assert issubclass(LLMRateLimitError, LLMClientError)
        assert issubclass(LLMTimeoutError, LLMClientError)


class TestHybridSearchTimeout:
    def test_timeout_exception_exists(self):
        from backend.retrieval.hybrid_search import HybridSearchTimeoutError
        assert issubclass(HybridSearchTimeoutError, Exception)


class TestConfigDefaults:
    def test_llm_retry_settings(self):
        assert config.LLM_MAX_RETRIES >= 1
        assert config.LLM_INITIAL_RETRY_DELAY > 0
        assert config.LLM_MAX_RETRY_DELAY > config.LLM_INITIAL_RETRY_DELAY
        assert config.LLM_REQUEST_TIMEOUT > 0

    def test_upload_settings(self):
        assert config.MAX_UPLOAD_SIZE_MB > 0
        assert config.MAX_UPLOAD_SIZE_BYTES == config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        assert len(config.ALLOWED_EXTENSIONS) > 0

    def test_security_settings(self):
        assert isinstance(config.API_KEY_ENABLED, bool)
        assert isinstance(config.RATE_LIMIT_ENABLED, bool)
        assert config.RATE_LIMIT_REQUESTS > 0
        assert config.RATE_LIMIT_WINDOW_SECONDS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


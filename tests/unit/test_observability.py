"""Tests for observability utilities."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from memory_layer.core.observability import (
    HealthCheckResult,
    HealthChecker,
    HealthReport,
    HealthStatus,
    JsonFormatter,
    MetricsCollector,
    StructuredLogger,
    check_api_key,
    counted,
    get_health_checker,
    get_metrics_collector,
    get_structured_logger,
    setup_structured_logging,
    timed,
    timed_sync,
)


class TestJsonFormatter:
    """Tests for JSON log formatter."""

    def test_format_basic_message(self):
        """Test formatting a basic log message."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JsonFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_timestamp_format(self):
        """Test timestamp is ISO 8601 format."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)

        # Should parse as ISO 8601
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_log_with_extra_fields(self):
        """Test logging with extra fields."""
        with patch.object(logging.Logger, "log") as mock_log:
            logger = StructuredLogger("test")
            logger.info("Test message", user_id="123", action="login")

            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert args[0] == logging.INFO
            assert args[1] == "Test message"
            assert kwargs["extra"]["user_id"] == "123"
            assert kwargs["extra"]["action"] == "login"

    def test_all_log_levels(self):
        """Test all log level methods."""
        logger = StructuredLogger("test")

        with patch.object(logging.Logger, "log") as mock_log:
            logger.debug("debug")
            logger.info("info")
            logger.warning("warning")
            logger.error("error")

            assert mock_log.call_count == 4


class TestSetupStructuredLogging:
    """Tests for setup_structured_logging."""

    def test_setup_text_format(self):
        """Test setting up text format logging."""
        setup_structured_logging(level="INFO", format="text")
        logger = logging.getLogger("memory_layer")
        assert logger.level == logging.INFO

    def test_setup_json_format(self):
        """Test setting up JSON format logging."""
        setup_structured_logging(level="DEBUG", format="json")
        logger = logging.getLogger("memory_layer")
        assert logger.level == logging.DEBUG

        # Check handler uses JsonFormatter
        assert any(
            isinstance(h.formatter, JsonFormatter) for h in logger.handlers
        )

    def test_get_structured_logger(self):
        """Test getting a structured logger."""
        logger = get_structured_logger("test_module")
        assert isinstance(logger, StructuredLogger)


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_increment_counter(self):
        """Test incrementing a counter."""
        collector = MetricsCollector()
        collector.increment("requests_total", 1)
        collector.increment("requests_total", 2)

        assert collector.get_counter("requests_total") == 3

    def test_counter_with_labels(self):
        """Test counter with labels."""
        collector = MetricsCollector()
        collector.increment("requests_total", labels={"method": "GET"})
        collector.increment("requests_total", labels={"method": "POST"})

        assert collector.get_counter("requests_total", {"method": "GET"}) == 1
        assert collector.get_counter("requests_total", {"method": "POST"}) == 1

    def test_set_gauge(self):
        """Test setting a gauge."""
        collector = MetricsCollector()
        collector.set_gauge("active_connections", 10)
        collector.set_gauge("active_connections", 15)

        assert collector.get_gauge("active_connections") == 15

    def test_observe_histogram(self):
        """Test observing histogram values."""
        collector = MetricsCollector()
        collector.observe_histogram("request_duration", 0.1)
        collector.observe_histogram("request_duration", 0.2)
        collector.observe_histogram("request_duration", 0.3)

        # Check it's recorded
        assert "request_duration" in collector.to_prometheus()

    def test_to_prometheus_format(self):
        """Test Prometheus format output."""
        collector = MetricsCollector()
        collector.increment(
            "http_requests_total",
            labels={"method": "GET"},
            description="Total HTTP requests",
        )
        collector.set_gauge(
            "memory_usage_bytes",
            1024,
            description="Memory usage in bytes",
        )

        output = collector.to_prometheus()

        assert "http_requests_total" in output
        assert "memory_usage_bytes" in output
        assert '# TYPE' in output

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        collector.increment("counter", 10)
        collector.set_gauge("gauge", 20)
        collector.reset()

        assert collector.get_counter("counter") == 0
        assert collector.get_gauge("gauge") == 0

    def test_get_global_collector(self):
        """Test getting global collector."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2


class TestHealthChecker:
    """Tests for HealthChecker."""

    @pytest.mark.asyncio
    async def test_register_and_run_check(self):
        """Test registering and running a health check."""
        checker = HealthChecker()

        async def healthy_check():
            return HealthCheckResult(
                name="test",
                status=HealthStatus.HEALTHY,
                message="All good",
            )

        checker.register("test", healthy_check)
        result = await checker.run_check("test")

        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_run_check_with_exception(self):
        """Test health check that raises exception."""
        checker = HealthChecker()

        async def failing_check():
            raise RuntimeError("Service unavailable")

        checker.register("failing", failing_check)
        result = await checker.run_check("failing")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Service unavailable" in result.message

    @pytest.mark.asyncio
    async def test_run_unknown_check(self):
        """Test running unknown health check."""
        checker = HealthChecker()
        result = await checker.run_check("unknown")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Unknown" in result.message

    @pytest.mark.asyncio
    async def test_run_all_checks(self):
        """Test running all health checks."""
        checker = HealthChecker()

        async def healthy():
            return HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY)

        async def degraded():
            return HealthCheckResult(name="degraded", status=HealthStatus.DEGRADED)

        checker.register("healthy", healthy)
        checker.register("degraded", degraded)

        report = await checker.run_all()

        assert report.status == HealthStatus.DEGRADED
        assert len(report.checks) == 2

    @pytest.mark.asyncio
    async def test_all_healthy_report(self):
        """Test report when all checks healthy."""
        checker = HealthChecker()

        async def healthy():
            return HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY)

        checker.register("db", healthy)
        checker.register("cache", healthy)

        report = await checker.run_all()
        assert report.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_any_unhealthy_report(self):
        """Test report when any check unhealthy."""
        checker = HealthChecker()

        async def healthy():
            return HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY)

        async def unhealthy():
            return HealthCheckResult(name="unhealthy", status=HealthStatus.UNHEALTHY)

        checker.register("db", healthy)
        checker.register("external", unhealthy)

        report = await checker.run_all()
        assert report.status == HealthStatus.UNHEALTHY

    def test_unregister_check(self):
        """Test unregistering a check."""
        checker = HealthChecker()

        async def check():
            return HealthCheckResult(name="test", status=HealthStatus.HEALTHY)

        checker.register("test", check)
        checker.unregister("test")

        # Should not raise
        checker.unregister("nonexistent")

    def test_get_global_checker(self):
        """Test getting global health checker."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        assert checker1 is checker2


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Connected",
            duration_ms=5.5,
            details={"connections": 10},
        )
        data = result.to_dict()

        assert data["name"] == "database"
        assert data["status"] == "healthy"
        assert data["message"] == "Connected"
        assert data["duration_ms"] == 5.5
        assert data["details"] == {"connections": 10}


class TestHealthReport:
    """Tests for HealthReport."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            checks=[
                HealthCheckResult(name="db", status=HealthStatus.HEALTHY),
            ],
        )
        data = report.to_dict()

        assert data["status"] == "healthy"
        assert len(data["checks"]) == 1
        assert "timestamp" in data


class TestCommonHealthChecks:
    """Tests for common health check functions."""

    @pytest.mark.asyncio
    async def test_check_api_key_present(self):
        """Test API key check when key is present."""
        result = await check_api_key("ANTHROPIC", "sk-ant-1234567890abcdef")

        assert result.status == HealthStatus.HEALTHY
        assert "configured" in result.message.lower()
        assert "key_preview" in result.details

    @pytest.mark.asyncio
    async def test_check_api_key_missing(self):
        """Test API key check when key is missing."""
        result = await check_api_key("ANTHROPIC", None)

        assert result.status == HealthStatus.DEGRADED
        assert "not configured" in result.message.lower()


class TestTimedDecorator:
    """Tests for timed decorator."""

    @pytest.mark.asyncio
    async def test_timed_async_function(self):
        """Test timing an async function."""
        collector = get_metrics_collector()
        collector.reset()

        @timed("test_operation")
        async def slow_operation():
            await asyncio.sleep(0.01)
            return "done"

        result = await slow_operation()

        assert result == "done"
        # Check metric was recorded
        output = collector.to_prometheus()
        assert "test_operation_seconds" in output

    def test_timed_sync_function(self):
        """Test timing a sync function."""
        collector = get_metrics_collector()
        collector.reset()

        @timed_sync("sync_operation")
        def sync_operation():
            return "done"

        result = sync_operation()

        assert result == "done"
        output = collector.to_prometheus()
        assert "sync_operation_seconds" in output


class TestCountedDecorator:
    """Tests for counted decorator."""

    @pytest.mark.asyncio
    async def test_counted_async_function(self):
        """Test counting an async function."""
        collector = get_metrics_collector()
        collector.reset()

        @counted("api_calls", labels={"endpoint": "search"})
        async def api_call():
            return "response"

        await api_call()
        await api_call()

        assert collector.get_counter("api_calls", {"endpoint": "search"}) == 2

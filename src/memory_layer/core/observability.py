"""
Observability utilities for Memory Layer.

Provides:
- Structured logging (JSON format)
- Metrics collection (Prometheus format)
- Health check utilities
- Tracing helpers
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# Structured Logging
# =============================================================================


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Outputs log records as JSON objects with consistent fields:
    - timestamp: ISO 8601 format
    - level: Log level name
    - logger: Logger name
    - message: Log message
    - extra: Any additional fields passed to the logger
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            }:
                extra_fields[key] = value

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str)


class StructuredLogger:
    """Logger wrapper that supports structured fields.

    Example:
        logger = StructuredLogger("memory_layer")
        logger.info("Memory created", memory_id="mem-123", category="pattern")
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log with extra fields."""
        self._logger.log(level, message, extra=kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with optional fields."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with optional fields."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with optional fields."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with optional fields."""
        self._log(logging.ERROR, message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, extra=kwargs)


def setup_structured_logging(
    level: str = "INFO",
    format: str = "text",
    log_file: Path | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Log format ("text" or "json")
        log_file: Optional log file path
    """
    root_logger = logging.getLogger("memory_layer")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if format.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually module name)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(f"memory_layer.{name}")


# =============================================================================
# Metrics Collection
# =============================================================================


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """A single metric value."""

    name: str
    type: MetricType
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    description: str = ""


class MetricsCollector:
    """Collects and exports application metrics.

    Metrics are stored in memory and can be exported in Prometheus format.

    Example:
        collector = MetricsCollector()
        collector.increment("requests_total", labels={"method": "GET"})
        collector.set_gauge("active_connections", 42)
        collector.observe_histogram("request_duration", 0.123)
    """

    def __init__(self):
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._descriptions: dict[str, str] = {}

    def _key(self, name: str, labels: dict[str, str] | None = None) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def increment(
        self,
        name: str,
        value: float = 1,
        labels: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Amount to increment (default 1)
            labels: Optional labels
            description: Metric description
        """
        key = self._key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
        if description:
            self._descriptions[name] = description

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
            description: Metric description
        """
        key = self._key(name, labels)
        self._gauges[key] = value
        if description:
            self._descriptions[name] = description

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        """Observe a histogram value.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
            description: Metric description
        """
        key = self._key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        if description:
            self._descriptions[name] = description

    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get a counter value."""
        key = self._key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get a gauge value."""
        key = self._key(name, labels)
        return self._gauges.get(key, 0)

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Export counters
        for key, value in sorted(self._counters.items()):
            name = key.split("{")[0]
            desc = self._descriptions.get(name, "")
            if desc and f"# HELP {name}" not in "\n".join(lines):
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} counter")
            lines.append(f"{key} {value}")

        # Export gauges
        for key, value in sorted(self._gauges.items()):
            name = key.split("{")[0]
            desc = self._descriptions.get(name, "")
            if desc and f"# HELP {name}" not in "\n".join(lines):
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} gauge")
            lines.append(f"{key} {value}")

        # Export histogram summaries
        for key, values in sorted(self._histograms.items()):
            name = key.split("{")[0]
            desc = self._descriptions.get(name, "")
            if desc and f"# HELP {name}" not in "\n".join(lines):
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} histogram")
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global metrics collector
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# =============================================================================
# Health Checks
# =============================================================================


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 2),
            "details": self.details,
        }


@dataclass
class HealthReport:
    """Aggregated health report."""

    status: HealthStatus
    checks: list[HealthCheckResult]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }


HealthCheckFn = Callable[[], Awaitable[HealthCheckResult]]


class HealthChecker:
    """Manages and runs health checks.

    Example:
        checker = HealthChecker()
        checker.register("database", check_database)
        checker.register("embedding", check_embedding_model)
        report = await checker.run_all()
    """

    def __init__(self):
        self._checks: dict[str, HealthCheckFn] = {}

    def register(self, name: str, check: HealthCheckFn) -> None:
        """Register a health check.

        Args:
            name: Check name
            check: Async function that returns HealthCheckResult
        """
        self._checks[name] = check

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)

    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a single health check.

        Args:
            name: Check name

        Returns:
            Health check result
        """
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Unknown health check: {name}",
            )

        check = self._checks[name]
        start = time.monotonic()
        try:
            result = await check()
            result.duration_ms = (time.monotonic() - start) * 1000
            return result
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

    async def run_all(self) -> HealthReport:
        """Run all health checks.

        Returns:
            Aggregated health report
        """
        results = await asyncio.gather(
            *[self.run_check(name) for name in self._checks],
            return_exceptions=False,
        )

        # Determine overall status
        if all(r.status == HealthStatus.HEALTHY for r in results):
            overall = HealthStatus.HEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return HealthReport(status=overall, checks=results)


# Global health checker
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# =============================================================================
# Common Health Checks
# =============================================================================


async def check_database(db_path: Path) -> HealthCheckResult:
    """Check database connectivity.

    Args:
        db_path: Path to SQLite database

    Returns:
        Health check result
    """
    import aiosqlite

    try:
        if not db_path.exists() and str(db_path) != ":memory:":
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database file not found: {db_path}",
            )

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM memories")
            count = (await cursor.fetchone())[0]
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connected",
                details={"memory_count": count},
            )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database error: {e}",
        )


async def check_embedding_model() -> HealthCheckResult:
    """Check embedding model availability.

    Returns:
        Health check result
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Quick test embedding
        _ = model.encode("test")
        return HealthCheckResult(
            name="embedding",
            status=HealthStatus.HEALTHY,
            message="Embedding model loaded",
        )
    except ImportError:
        return HealthCheckResult(
            name="embedding",
            status=HealthStatus.DEGRADED,
            message="sentence-transformers not installed",
        )
    except Exception as e:
        return HealthCheckResult(
            name="embedding",
            status=HealthStatus.UNHEALTHY,
            message=f"Embedding model error: {e}",
        )


async def check_api_key(key_name: str, key_value: str | None) -> HealthCheckResult:
    """Check if API key is configured.

    Args:
        key_name: Name of the API key (for display)
        key_value: The key value (or None if not set)

    Returns:
        Health check result
    """
    if key_value:
        # Mask the key for display
        masked = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
        return HealthCheckResult(
            name=f"api_key_{key_name.lower()}",
            status=HealthStatus.HEALTHY,
            message=f"{key_name} configured",
            details={"key_preview": masked},
        )
    return HealthCheckResult(
        name=f"api_key_{key_name.lower()}",
        status=HealthStatus.DEGRADED,
        message=f"{key_name} not configured",
    )


# =============================================================================
# Timing Decorators
# =============================================================================


def timed(
    metric_name: str | None = None,
    log_level: int = logging.DEBUG,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to time async function execution.

    Records duration as a histogram metric and optionally logs it.

    Args:
        metric_name: Optional metric name (defaults to function name)
        log_level: Logging level for timing output

    Example:
        @timed("search_duration")
        async def search(query: str):
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        name = metric_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.monotonic()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.monotonic() - start
                get_metrics_collector().observe_histogram(
                    f"{name}_seconds",
                    duration,
                    description=f"Duration of {func.__name__}",
                )
                logging.getLogger("memory_layer.timing").log(
                    log_level,
                    f"{func.__name__} completed in {duration*1000:.2f}ms",
                )

        return wrapper

    return decorator


def timed_sync(
    metric_name: str | None = None,
    log_level: int = logging.DEBUG,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to time sync function execution."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        name = metric_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.monotonic() - start
                get_metrics_collector().observe_histogram(
                    f"{name}_seconds",
                    duration,
                    description=f"Duration of {func.__name__}",
                )
                logging.getLogger("memory_layer.timing").log(
                    log_level,
                    f"{func.__name__} completed in {duration*1000:.2f}ms",
                )

        return wrapper

    return decorator


def counted(
    metric_name: str | None = None,
    labels: dict[str, str] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to count function calls.

    Args:
        metric_name: Optional metric name (defaults to function name)
        labels: Optional labels for the counter

    Example:
        @counted("api_requests", labels={"endpoint": "search"})
        async def search(query: str):
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        name = metric_name or f"{func.__module__}.{func.__name__}_total"

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            get_metrics_collector().increment(
                name,
                labels=labels,
                description=f"Total calls to {func.__name__}",
            )
            return await func(*args, **kwargs)

        return wrapper

    return decorator

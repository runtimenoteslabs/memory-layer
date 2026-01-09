"""Logging infrastructure for Memory Layer.

Provides structured logging with support for both console and JSON output formats.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, ClassVar


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Console formatter with color support."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        """Initialize formatter.

        Args:
            use_colors: Whether to use ANSI color codes in output.
        """
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        message = record.getMessage()
        formatted = f"{timestamp} | {level_str} | {record.name} | {message}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logging(
    level: int | str = logging.INFO,
    json_output: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """Configure logging for Memory Layer.

    Args:
        level: Logging level (e.g., logging.INFO, "DEBUG").
        json_output: If True, use JSON format for output.
        log_file: Optional path to log file.

    Returns:
        Configured root logger for memory_layer.
    """
    logger = logging.getLogger("memory_layer")

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"memory_layer.{name}")


class _DefaultLoggerHolder:
    """Holder for the default logger to avoid global statement."""

    logger: logging.Logger | None = None


def get_default_logger() -> logging.Logger:
    """Get or create the default Memory Layer logger.

    Returns:
        Default logger instance.
    """
    if _DefaultLoggerHolder.logger is None:
        _DefaultLoggerHolder.logger = setup_logging()
    return _DefaultLoggerHolder.logger

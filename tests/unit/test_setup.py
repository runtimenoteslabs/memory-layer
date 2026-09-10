"""Tests for project setup and logging infrastructure."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest


class TestProjectSetup:
    """Tests for basic project setup."""

    def test_package_import(self) -> None:
        """Test that the main package can be imported."""
        import runtime_memory

        # Assert a valid semver string rather than a literal, so version
        # bumps don't break this test (version is sourced from __init__.py).
        assert re.match(r"^\d+\.\d+\.\d+", runtime_memory.__version__)
        assert runtime_memory.__author__ == "exitcode42"

    def test_core_import(self) -> None:
        """Test that core module can be imported."""
        from runtime_memory import core

        assert hasattr(core, "get_logger")
        assert hasattr(core, "setup_logging")


class TestLogging:
    """Tests for logging infrastructure."""

    def test_setup_logging_default(self) -> None:
        """Test default logging setup."""
        from runtime_memory.core.logging import setup_logging

        logger = setup_logging()
        assert logger.name == "runtime_memory"
        assert logger.level == logging.INFO

    def test_setup_logging_debug_level(self) -> None:
        """Test logging setup with debug level."""
        from runtime_memory.core.logging import setup_logging

        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logging_string_level(self) -> None:
        """Test logging setup with string level."""
        from runtime_memory.core.logging import setup_logging

        logger = setup_logging(level="WARNING")
        assert logger.level == logging.WARNING

    def test_get_logger(self) -> None:
        """Test getting a module-specific logger."""
        from runtime_memory.core.logging import get_logger

        logger = get_logger("test_module")
        assert logger.name == "runtime_memory.test_module"

    def test_get_logger_does_not_double_the_package_prefix(self) -> None:
        """Callers pass __name__, which already starts with the package name."""
        from runtime_memory.core.logging import get_logger

        logger = get_logger("runtime_memory.core.embeddings")
        assert logger.name == "runtime_memory.core.embeddings"
        assert get_logger("runtime_memory").name == "runtime_memory"

    def test_get_logger_keeps_names_under_the_package(self) -> None:
        """setup_logging attaches handlers to `runtime_memory`, so names must sit under it."""
        from runtime_memory.core.logging import get_logger

        assert get_logger("runtime_memory.core.engine").name.startswith("runtime_memory")
        assert get_logger("outside_caller").name.startswith("runtime_memory.")

    def test_setup_logging_json_output(self) -> None:
        """Test logging setup with JSON output."""
        from runtime_memory.core.logging import JSONFormatter, setup_logging

        logger = setup_logging(json_output=True)
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_json_formatter(self) -> None:
        """Test JSON formatter produces valid output."""
        import json

        from runtime_memory.core.logging import JSONFormatter

        formatter = JSONFormatter()
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
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_setup_logging_with_file(self, temp_dir: Path) -> None:
        """Test logging setup with file output."""
        from runtime_memory.core.logging import setup_logging

        log_file = temp_dir / "test.log"
        logger = setup_logging(log_file=str(log_file))

        # Log a message
        logger.info("Test file logging")

        # Check that file handler was added
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_console_formatter_no_colors(self) -> None:
        """Test console formatter without colors."""
        from runtime_memory.core.logging import ConsoleFormatter

        formatter = ConsoleFormatter(use_colors=False)
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

        assert "INFO" in output
        assert "Test message" in output
        assert "\033[" not in output  # No ANSI codes


class TestDirectoryStructure:
    """Tests for project directory structure."""

    def test_src_structure(self) -> None:
        """Test that source directory structure exists."""
        src_root = Path(__file__).parent.parent.parent / "src" / "runtime_memory"

        expected_dirs = [
            "core",
            "extraction",
            "claude_code",
            "server",
            "cli",
            "sdk",
            "tasks",
        ]

        for dir_name in expected_dirs:
            dir_path = src_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"
            assert (dir_path / "__init__.py").exists(), f"{dir_name}/__init__.py should exist"

    def test_pyproject_exists(self) -> None:
        """Test that pyproject.toml exists."""
        project_root = Path(__file__).parent.parent.parent
        pyproject = project_root / "pyproject.toml"
        assert pyproject.exists()

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
        import memory_layer

        # Assert a valid semver string rather than a literal, so version
        # bumps don't break this test (version is sourced from __init__.py).
        assert re.match(r"^\d+\.\d+\.\d+", memory_layer.__version__)
        assert memory_layer.__author__ == "Memory Layer Team"

    def test_core_import(self) -> None:
        """Test that core module can be imported."""
        from memory_layer import core

        assert hasattr(core, "get_logger")
        assert hasattr(core, "setup_logging")


class TestLogging:
    """Tests for logging infrastructure."""

    def test_setup_logging_default(self) -> None:
        """Test default logging setup."""
        from memory_layer.core.logging import setup_logging

        logger = setup_logging()
        assert logger.name == "memory_layer"
        assert logger.level == logging.INFO

    def test_setup_logging_debug_level(self) -> None:
        """Test logging setup with debug level."""
        from memory_layer.core.logging import setup_logging

        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logging_string_level(self) -> None:
        """Test logging setup with string level."""
        from memory_layer.core.logging import setup_logging

        logger = setup_logging(level="WARNING")
        assert logger.level == logging.WARNING

    def test_get_logger(self) -> None:
        """Test getting a module-specific logger."""
        from memory_layer.core.logging import get_logger

        logger = get_logger("test_module")
        assert logger.name == "memory_layer.test_module"

    def test_setup_logging_json_output(self) -> None:
        """Test logging setup with JSON output."""
        from memory_layer.core.logging import JSONFormatter, setup_logging

        logger = setup_logging(json_output=True)
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_json_formatter(self) -> None:
        """Test JSON formatter produces valid output."""
        import json

        from memory_layer.core.logging import JSONFormatter

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
        from memory_layer.core.logging import setup_logging

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
        from memory_layer.core.logging import ConsoleFormatter

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
        src_root = Path(__file__).parent.parent.parent / "src" / "memory_layer"

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

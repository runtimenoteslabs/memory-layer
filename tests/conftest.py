"""Pytest configuration and fixtures for Memory Layer tests."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path(temp_dir: Path) -> Path:
    """Create a path for a temporary database.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Path for temporary database file.
    """
    return temp_dir / "test_memory.db"


@pytest.fixture
async def async_temp_dir() -> AsyncGenerator[Path, None]:
    """Async version of temp_dir fixture.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

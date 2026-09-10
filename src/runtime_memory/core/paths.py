"""Where Runtime Memory keeps its files."""

from __future__ import annotations

from pathlib import Path

STORE_DIR_NAME = ".runtime-memory"
LEGACY_STORE_DIR_NAME = ".memory-layer"
"""Store directory used before the 3.0 rename."""


def store_dir() -> Path:
    """The directory holding the database, cache and extracted context.

    Falls back to the pre-3.0 directory when it exists and the current one does
    not, so an install that predates the rename keeps reading the memories it
    already has instead of quietly starting an empty store.

    Returns:
        Directory path. It is not created here.
    """
    current = Path.home() / STORE_DIR_NAME
    if not current.exists() and (Path.home() / LEGACY_STORE_DIR_NAME).exists():
        return Path.home() / LEGACY_STORE_DIR_NAME
    return current


def default_db_path() -> Path:
    """The database used when nothing names another one."""
    return store_dir() / "memories.db"

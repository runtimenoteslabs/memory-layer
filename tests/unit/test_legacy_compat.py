"""Tests for the 3.0 rename compatibility layer.

The package moved from `memory_layer` to `runtime_memory`, which renamed the
environment prefix and the store directory. Both keep working from their old
names so an install that predates the rename does not quietly lose its data.
"""

from __future__ import annotations

from pathlib import Path

from runtime_memory.core.legacy_env import apply_legacy_env
from runtime_memory.core.paths import (
    LEGACY_STORE_DIR_NAME,
    STORE_DIR_NAME,
    default_db_path,
    store_dir,
)


class TestLegacyEnv:
    """MEMORY_LAYER_* is carried onto RUNTIME_MEMORY_*."""

    def test_carries_a_legacy_variable(self) -> None:
        env = {"MEMORY_LAYER_DB": "/tmp/old.db"}

        carried = apply_legacy_env(env)

        assert env["RUNTIME_MEMORY_DB"] == "/tmp/old.db"
        assert carried == ["MEMORY_LAYER_DB"]

    def test_explicit_setting_wins(self) -> None:
        """A current name already set is never overwritten by a stale one."""
        env = {"MEMORY_LAYER_DB": "/tmp/old.db", "RUNTIME_MEMORY_DB": "/tmp/new.db"}

        carried = apply_legacy_env(env)

        assert env["RUNTIME_MEMORY_DB"] == "/tmp/new.db"
        assert carried == []

    def test_leaves_unrelated_variables_alone(self) -> None:
        env = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-test"}

        apply_legacy_env(env)

        assert env == {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-test"}

    def test_is_idempotent(self) -> None:
        env = {"MEMORY_LAYER_EMBEDDING": "mock"}

        apply_legacy_env(env)
        second = apply_legacy_env(env)

        assert second == []
        assert env["RUNTIME_MEMORY_EMBEDDING"] == "mock"


class TestStoreDir:
    """The store falls back to the pre-3.0 directory rather than starting empty."""

    def test_uses_the_current_directory(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / STORE_DIR_NAME).mkdir()

        assert store_dir() == tmp_path / STORE_DIR_NAME

    def test_falls_back_to_the_legacy_directory(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / LEGACY_STORE_DIR_NAME).mkdir()

        assert store_dir() == tmp_path / LEGACY_STORE_DIR_NAME

    def test_current_wins_when_both_exist(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / STORE_DIR_NAME).mkdir()
        (tmp_path / LEGACY_STORE_DIR_NAME).mkdir()

        assert store_dir() == tmp_path / STORE_DIR_NAME

    def test_defaults_to_the_current_directory_on_a_fresh_machine(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert store_dir() == tmp_path / STORE_DIR_NAME

    def test_database_sits_in_the_store(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert default_db_path() == tmp_path / STORE_DIR_NAME / "memories.db"

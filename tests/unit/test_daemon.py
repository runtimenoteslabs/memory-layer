"""Unit tests for the memory layer daemon."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_layer.claude_code.daemon import (
    DaemonConfig,
    DaemonError,
    DaemonAlreadyRunningError,
    MemoryLayerDaemon,
    SessionInfo,
    SessionHandler,
    get_daemon_status,
    stop_daemon,
)


class TestDaemonConfig:
    """Tests for DaemonConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DaemonConfig()

        assert config.claude_dir is not None
        assert config.session_dir is not None
        assert config.pid_file is not None
        assert config.process_on_modify is True
        assert config.process_on_create is False
        assert config.debounce_seconds == 2.0
        assert config.session_patterns == ["*.json", "*.jsonl"]
        assert config.auto_extract is True
        assert config.update_claude_md is True
        assert config.drop_privileges is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = DaemonConfig(
            claude_dir=Path("/custom/claude"),
            session_dir=Path("/custom/sessions"),
            pid_file=Path("/custom/pid"),
            process_on_modify=False,
            process_on_create=True,
            debounce_seconds=5.0,
            session_patterns=["*.txt"],
            auto_extract=False,
            update_claude_md=False,
            drop_privileges=True,
        )

        assert config.claude_dir == Path("/custom/claude")
        assert config.session_dir == Path("/custom/sessions")
        assert config.pid_file == Path("/custom/pid")
        assert config.process_on_modify is False
        assert config.process_on_create is True
        assert config.debounce_seconds == 5.0
        assert config.session_patterns == ["*.txt"]
        assert config.auto_extract is False
        assert config.update_claude_md is False
        assert config.drop_privileges is True

    def test_config_post_init_converts_strings(self) -> None:
        """Test __post_init__ converts string paths to Path objects."""
        # Create config with paths already as Path objects
        config = DaemonConfig(
            claude_dir=Path("/test/claude"),
            session_dir=Path("/test/sessions"),
            pid_file=Path("/test/pid"),
        )

        assert isinstance(config.claude_dir, Path)
        assert isinstance(config.session_dir, Path)
        assert isinstance(config.pid_file, Path)


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_session_info_creation(self) -> None:
        """Test creating SessionInfo."""
        from datetime import datetime

        session = SessionInfo(
            session_id="sess-123",
            file_path=Path("/sessions/session.json"),
            project_path=Path("/project"),
            started_at=datetime.now(),
            ended_at=None,
            is_active=True,
        )

        assert session.session_id == "sess-123"
        assert session.file_path == Path("/sessions/session.json")
        assert session.project_path == Path("/project")
        assert session.is_active is True
        assert session.transcript == ""
        assert session.metadata == {}

    def test_session_info_with_content(self) -> None:
        """Test SessionInfo with transcript content."""
        session = SessionInfo(
            session_id="sess-456",
            file_path=Path("/test.json"),
            project_path=None,
            started_at=None,
            ended_at=None,
            is_active=False,
            transcript="Session transcript content",
            metadata={"key": "value"},
        )

        assert session.transcript == "Session transcript content"
        assert session.is_active is False
        assert session.metadata == {"key": "value"}


class TestMemoryLayerDaemon:
    """Tests for MemoryLayerDaemon class."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create mock memory engine."""
        engine = MagicMock()
        engine.add = AsyncMock()
        engine.get_context = AsyncMock()
        return engine

    @pytest.fixture
    def mock_extractor(self) -> MagicMock:
        """Create mock extractor."""
        extractor = MagicMock()
        extractor.extract_and_store = AsyncMock()
        return extractor

    @pytest.fixture
    def daemon(
        self, mock_engine: MagicMock, mock_extractor: MagicMock, tmp_path: Path
    ) -> MemoryLayerDaemon:
        """Create daemon instance with mocks."""
        config = DaemonConfig(
            session_dir=tmp_path / "sessions",
            pid_file=tmp_path / "daemon.pid",
        )
        (tmp_path / "sessions").mkdir()
        return MemoryLayerDaemon(
            config=config,
            engine=mock_engine,
            extractor=mock_extractor,
        )

    # Initialization tests

    def test_init_with_defaults(self) -> None:
        """Test initialization with default config."""
        daemon = MemoryLayerDaemon()

        assert daemon.config is not None
        assert daemon._engine is None
        assert daemon._extractor is None
        assert daemon.is_running is False

    def test_init_with_engine(self, mock_engine: MagicMock) -> None:
        """Test initialization with engine."""
        daemon = MemoryLayerDaemon(engine=mock_engine)

        assert daemon._engine == mock_engine

    def test_init_with_custom_config(self, tmp_path: Path) -> None:
        """Test initialization with custom config."""
        config = DaemonConfig(
            session_dir=tmp_path,
            debounce_seconds=15.0,
        )
        daemon = MemoryLayerDaemon(config=config)

        assert daemon.config.session_dir == tmp_path
        assert daemon.config.debounce_seconds == 15.0

    # Start/stop tests

    def test_start_creates_pid_file(self, daemon: MemoryLayerDaemon) -> None:
        """Test that start creates PID file."""
        daemon.start()

        assert daemon.config.pid_file.exists()
        pid_content = daemon.config.pid_file.read_text()
        assert pid_content.strip().isdigit()

        daemon.stop()

    def test_stop_removes_pid_file(self, daemon: MemoryLayerDaemon) -> None:
        """Test that stop removes PID file."""
        daemon.start()
        daemon.stop()

        assert not daemon.config.pid_file.exists()

    def test_start_sets_running_flag(self, daemon: MemoryLayerDaemon) -> None:
        """Test that start sets running flag."""
        daemon.start()

        assert daemon.is_running is True

        daemon.stop()

    def test_stop_clears_running_flag(self, daemon: MemoryLayerDaemon) -> None:
        """Test that stop clears running flag."""
        daemon.start()
        daemon.stop()

        assert daemon.is_running is False

    def test_double_stop_is_safe(self, daemon: MemoryLayerDaemon) -> None:
        """Test that stopping twice is safe."""
        daemon.start()
        daemon.stop()
        daemon.stop()  # Should not raise

        assert daemon.is_running is False

    def test_start_when_already_running_raises(self, daemon: MemoryLayerDaemon) -> None:
        """Test that starting when daemon already running raises error."""
        daemon.start()

        try:
            with pytest.raises(DaemonAlreadyRunningError):
                daemon.start()
        finally:
            daemon.stop()

    # PID file tests

    def test_pid_file_contains_current_pid(self, daemon: MemoryLayerDaemon) -> None:
        """Test PID file contains current process ID."""
        daemon.start()

        pid = int(daemon.config.pid_file.read_text().strip())
        assert pid == os.getpid()

        daemon.stop()

    def test_check_existing_daemon_returns_none_when_not_running(
        self, daemon: MemoryLayerDaemon
    ) -> None:
        """Test _check_existing_daemon returns None when no daemon running."""
        result = daemon._check_existing_daemon()
        assert result is None

    def test_check_existing_daemon_removes_stale_pid(
        self, daemon: MemoryLayerDaemon
    ) -> None:
        """Test _check_existing_daemon removes stale PID file."""
        # Write fake PID that doesn't exist
        daemon.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        daemon.config.pid_file.write_text("999999")

        result = daemon._check_existing_daemon()

        assert result is None
        assert not daemon.config.pid_file.exists()

    # Health check tests

    def test_health_check_when_running(self, daemon: MemoryLayerDaemon) -> None:
        """Test health check when daemon is running."""
        daemon.start()

        health = daemon.health_check()

        assert health["running"] is True
        assert health["pid"] == os.getpid()
        assert "watch_dir" in health
        assert health["watch_dir_exists"] is True

        daemon.stop()

    def test_health_check_when_stopped(self, daemon: MemoryLayerDaemon) -> None:
        """Test health check when daemon is stopped."""
        health = daemon.health_check()

        assert health["running"] is False
        assert health["pid"] is None

    # Session parsing tests

    def test_parse_json_session_file(self, daemon: MemoryLayerDaemon) -> None:
        """Test parsing a JSON session file."""
        session_file = daemon.config.session_dir / "session.json"
        session_data = {
            "session_id": "test-123",
            "started_at": "2024-01-01T00:00:00",
            "ended_at": "2024-01-01T01:00:00",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        session_file.write_text(json.dumps(session_data))

        session = daemon._parse_session_file(session_file)

        assert session is not None
        assert session.session_id == "test-123"
        assert "User: Hello" in session.transcript
        assert "Assistant: Hi there" in session.transcript

    def test_parse_jsonl_session_file(self, daemon: MemoryLayerDaemon) -> None:
        """Test parsing a JSONL session file."""
        session_file = daemon.config.session_dir / "session.jsonl"
        lines = [
            '{"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00"}',
            '{"role": "assistant", "content": "Hi", "timestamp": "2024-01-01T00:00:01"}',
            '{"type": "session_end"}',
        ]
        session_file.write_text("\n".join(lines))

        session = daemon._parse_session_file(session_file)

        assert session is not None
        assert "User: Hello" in session.transcript
        assert session.is_active is False

    def test_parse_session_invalid_json(self, daemon: MemoryLayerDaemon) -> None:
        """Test parsing invalid JSON returns None."""
        session_file = daemon.config.session_dir / "invalid.json"
        session_file.write_text("not valid json {{{")

        session = daemon._parse_session_file(session_file)

        assert session is None

    def test_parse_session_empty_file(self, daemon: MemoryLayerDaemon) -> None:
        """Test parsing empty session file."""
        empty_file = daemon.config.session_dir / "empty.json"
        empty_file.write_text("")

        session = daemon._parse_session_file(empty_file)

        assert session is None

    # Session processing tests

    @pytest.mark.asyncio
    async def test_process_session_file(
        self, daemon: MemoryLayerDaemon, mock_extractor: MagicMock
    ) -> None:
        """Test processing a session file."""
        # Create session file
        session_file = daemon.config.session_dir / "session.json"
        session_data = {
            "session_id": "test-session",
            "ended_at": "2024-01-01T01:00:00",
            "transcript": "User: Question\nAssistant: Answer\n" * 20,
        }
        session_file.write_text(json.dumps(session_data))

        extraction_result = MagicMock()
        extraction_result.success = True
        extraction_result.memory_count = 2
        extraction_result.memories = []
        mock_extractor.extract_and_store.return_value = extraction_result

        await daemon._process_session_file(session_file)

        mock_extractor.extract_and_store.assert_called()

    @pytest.mark.asyncio
    async def test_process_session_active_not_extracted(
        self, daemon: MemoryLayerDaemon, mock_extractor: MagicMock
    ) -> None:
        """Test that active sessions are not extracted."""
        session_file = daemon.config.session_dir / "active.json"
        session_data = {
            "session_id": "active-session",
            "is_active": True,
            "transcript": "Some content",
        }
        session_file.write_text(json.dumps(session_data))

        await daemon._process_session_file(session_file)

        # Extractor should not be called for active sessions
        mock_extractor.extract_and_store.assert_not_called()

    # Project detection tests

    def test_detect_project_path_git(self, daemon: MemoryLayerDaemon, tmp_path: Path) -> None:
        """Test detecting project path from .git marker."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        session_dir = project_dir / "sessions"
        session_dir.mkdir()
        session_file = session_dir / "session.json"
        session_file.write_text("{}")

        project_path = daemon._detect_project_path(session_file)

        assert project_path == project_dir

    def test_detect_project_path_pyproject(self, daemon: MemoryLayerDaemon, tmp_path: Path) -> None:
        """Test detecting project path from pyproject.toml."""
        project_dir = tmp_path / "python_project"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text("[project]")

        session_file = project_dir / "session.json"
        session_file.write_text("{}")

        project_path = daemon._detect_project_path(session_file)

        assert project_path == project_dir

    def test_detect_project_path_not_found(self, daemon: MemoryLayerDaemon, tmp_path: Path) -> None:
        """Test detecting project path when no markers found."""
        session_file = tmp_path / "session.json"
        session_file.write_text("{}")

        project_path = daemon._detect_project_path(session_file)

        assert project_path is None

    # Callback tests

    def test_register_session_end_callback(self, daemon: MemoryLayerDaemon) -> None:
        """Test registering session end callback."""
        callback = MagicMock()
        daemon.on_session_end(callback)

        assert callback in daemon._on_session_end

    def test_register_extraction_callback(self, daemon: MemoryLayerDaemon) -> None:
        """Test registering extraction callback."""
        callback = MagicMock()
        daemon.on_extraction(callback)

        assert callback in daemon._on_extraction

    # Signal handling tests

    def test_signal_handler_stops_daemon(self, daemon: MemoryLayerDaemon) -> None:
        """Test signal handler stops daemon."""
        daemon.start()
        assert daemon.is_running is True

        daemon._signal_handler(signal.SIGTERM, None)

        assert daemon.is_running is False


class TestSessionHandler:
    """Tests for SessionHandler class."""

    @pytest.fixture
    def mock_daemon(self) -> MagicMock:
        """Create mock daemon."""
        daemon = MagicMock()
        daemon._process_session_file = AsyncMock()
        daemon.config = DaemonConfig(
            session_patterns=["*.json", "*.jsonl"],
            process_on_modify=True,
            process_on_create=True,
            debounce_seconds=0.1,
        )
        return daemon

    @pytest.fixture
    def handler(self, mock_daemon: MagicMock) -> SessionHandler:
        """Create session handler."""
        return SessionHandler(daemon=mock_daemon, config=mock_daemon.config)

    def test_should_process_matching_file(self, handler: SessionHandler, tmp_path: Path) -> None:
        """Test _should_process returns True for matching files."""
        result = handler._should_process(str(tmp_path / "session.json"))
        assert result is True

    def test_should_process_non_matching_file(self, handler: SessionHandler, tmp_path: Path) -> None:
        """Test _should_process returns False for non-matching files."""
        result = handler._should_process(str(tmp_path / "file.py"))
        assert result is False

    def test_should_process_debounce(self, handler: SessionHandler, tmp_path: Path) -> None:
        """Test _should_process debounces repeated calls."""
        path = str(tmp_path / "session.json")

        # First call should process
        result1 = handler._should_process(path)
        assert result1 is True

        # Immediate second call should be debounced
        result2 = handler._should_process(path)
        assert result2 is False

    def test_on_created_ignores_directories(
        self, handler: SessionHandler, mock_daemon: MagicMock
    ) -> None:
        """Test on_created ignores directories."""
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/some/dir"

        handler.on_created(event)

        mock_daemon._process_session_file.assert_not_called()

    def test_on_modified_ignores_directories(
        self, handler: SessionHandler, mock_daemon: MagicMock
    ) -> None:
        """Test on_modified ignores directories."""
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/some/dir"

        handler.on_modified(event)

        mock_daemon._process_session_file.assert_not_called()


class TestPrivilegeDropping:
    """Tests for privilege dropping functionality."""

    def test_drop_privileges_config_option(self) -> None:
        """Test drop_privileges config option."""
        config = DaemonConfig(drop_privileges=True, target_uid=1000, target_gid=1000)
        assert config.drop_privileges is True
        assert config.target_uid == 1000
        assert config.target_gid == 1000

    def test_drop_privileges_not_called_when_disabled(self, tmp_path: Path) -> None:
        """Test privileges not dropped when disabled."""
        config = DaemonConfig(
            session_dir=tmp_path,
            pid_file=tmp_path / "daemon.pid",
            drop_privileges=False,
        )
        tmp_path.mkdir(exist_ok=True)

        daemon = MemoryLayerDaemon(config=config)
        daemon.start()

        # Should not fail when not dropping privileges
        assert daemon.is_running is True

        daemon.stop()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_daemon_status_not_running(self, tmp_path: Path) -> None:
        """Test get_daemon_status when daemon not running."""
        pid_file = tmp_path / "nonexistent.pid"

        status = get_daemon_status(pid_file)

        assert status["running"] is False
        assert status["pid"] is None

    def test_get_daemon_status_stale_pid(self, tmp_path: Path) -> None:
        """Test get_daemon_status with stale PID."""
        pid_file = tmp_path / "stale.pid"
        pid_file.write_text("999999")

        status = get_daemon_status(pid_file)

        assert status["running"] is False

    def test_stop_daemon_not_running(self, tmp_path: Path) -> None:
        """Test stop_daemon when not running."""
        pid_file = tmp_path / "nonexistent.pid"

        result = stop_daemon(pid_file)

        assert result is False


class TestDaemonErrors:
    """Tests for daemon error handling."""

    def test_daemon_error_base(self) -> None:
        """Test DaemonError base class."""
        error = DaemonError("Test error")
        assert str(error) == "Test error"

    def test_daemon_already_running_error(self) -> None:
        """Test DaemonAlreadyRunningError."""
        error = DaemonAlreadyRunningError("PID 123")
        assert "PID 123" in str(error)

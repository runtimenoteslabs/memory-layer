"""Unit tests for Claude Code lifecycle hooks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_layer.claude_code.hooks import (
    HookConfig,
    HookError,
    HookNotInstalledError,
    HookResult,
    HookState,
    HookType,
    MemoryLayerHooks,
    get_hook_status,
    install_all_hooks,
    uninstall_all_hooks,
)


class TestHookType:
    """Tests for HookType enum."""

    def test_hook_type_values(self) -> None:
        """Test hook type string values."""
        assert HookType.PRE_SESSION.value == "pre-session"
        assert HookType.POST_SESSION.value == "post-session"
        assert HookType.PRE_COMPACT.value == "pre-compact"

    def test_all_hook_types(self) -> None:
        """Test all hook types are defined."""
        types = list(HookType)
        assert len(types) == 3
        assert HookType.PRE_SESSION in types
        assert HookType.POST_SESSION in types
        assert HookType.PRE_COMPACT in types


class TestHookConfig:
    """Tests for HookConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = HookConfig()

        assert config.hooks_dir is not None
        assert config.state_file is not None
        assert config.inject_context is True
        assert config.max_context_memories == 15
        assert config.context_query is None
        assert config.extract_memories is True
        assert config.save_before_compact is True
        assert config.output_format == "markdown"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = HookConfig(
            hooks_dir=Path("/custom/hooks"),
            state_file=Path("/custom/state.json"),
            inject_context=False,
            max_context_memories=20,
            context_query="test query",
            extract_memories=False,
            save_before_compact=False,
            output_format="json",
        )

        assert config.hooks_dir == Path("/custom/hooks")
        assert config.state_file == Path("/custom/state.json")
        assert config.inject_context is False
        assert config.max_context_memories == 20
        assert config.context_query == "test query"
        assert config.extract_memories is False
        assert config.save_before_compact is False
        assert config.output_format == "json"

    def test_config_post_init_converts_strings(self) -> None:
        """Test __post_init__ converts string paths."""
        config = HookConfig(
            hooks_dir=Path("/test/hooks"),
            state_file=Path("/test/state.json"),
        )

        assert isinstance(config.hooks_dir, Path)
        assert isinstance(config.state_file, Path)


class TestHookState:
    """Tests for HookState dataclass."""

    def test_hook_state_defaults(self) -> None:
        """Test default state values."""
        state = HookState()

        assert state.last_session_id is None
        assert state.last_session_start is None
        assert state.last_extraction is None
        assert state.last_context_injection is None
        assert state.memories_extracted_total == 0
        assert state.sessions_processed == 0

    def test_hook_state_to_dict(self) -> None:
        """Test converting state to dict."""
        now = datetime.now()
        state = HookState(
            last_session_id="sess-123",
            last_session_start=now,
            memories_extracted_total=5,
            sessions_processed=3,
        )

        d = state.to_dict()

        assert d["last_session_id"] == "sess-123"
        assert d["last_session_start"] == now.isoformat()
        assert d["memories_extracted_total"] == 5
        assert d["sessions_processed"] == 3

    def test_hook_state_from_dict(self) -> None:
        """Test creating state from dict."""
        data = {
            "last_session_id": "sess-456",
            "last_session_start": "2024-01-01T00:00:00",
            "last_extraction": "2024-01-01T01:00:00",
            "memories_extracted_total": 10,
            "sessions_processed": 5,
        }

        state = HookState.from_dict(data)

        assert state.last_session_id == "sess-456"
        assert state.last_session_start == datetime.fromisoformat("2024-01-01T00:00:00")
        assert state.memories_extracted_total == 10
        assert state.sessions_processed == 5

    def test_hook_state_from_dict_missing_keys(self) -> None:
        """Test from_dict with missing keys uses defaults."""
        data = {"last_session_id": "test"}

        state = HookState.from_dict(data)

        assert state.last_session_id == "test"
        assert state.memories_extracted_total == 0
        assert state.sessions_processed == 0


class TestHookResult:
    """Tests for HookResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating success result."""
        result = HookResult(
            success=True,
            hook_type=HookType.PRE_SESSION,
            output="## Memories\n- Item 1",
            memories_processed=5,
        )

        assert result.success is True
        assert result.hook_type == HookType.PRE_SESSION
        assert result.output == "## Memories\n- Item 1"
        assert result.memories_processed == 5
        assert result.error is None

    def test_error_result(self) -> None:
        """Test creating error result."""
        result = HookResult(
            success=False,
            hook_type=HookType.POST_SESSION,
            output="",
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"

    def test_result_with_duration(self) -> None:
        """Test result with duration."""
        result = HookResult(
            success=True,
            hook_type=HookType.PRE_COMPACT,
            output="Saved",
            duration_ms=150.5,
        )

        assert result.duration_ms == 150.5


class TestMemoryLayerHooks:
    """Tests for MemoryLayerHooks class."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create mock memory engine."""
        engine = MagicMock()
        engine.get_context = AsyncMock()
        engine.add = AsyncMock()
        return engine

    @pytest.fixture
    def mock_extractor(self) -> MagicMock:
        """Create mock extractor."""
        extractor = MagicMock()
        extractor.extract_and_store = AsyncMock()
        return extractor

    @pytest.fixture
    def hooks(self, mock_engine: MagicMock, mock_extractor: MagicMock, tmp_path: Path) -> MemoryLayerHooks:
        """Create hooks instance with mocks."""
        config = HookConfig(
            hooks_dir=tmp_path / "hooks",
            state_file=tmp_path / "state.json",
        )
        return MemoryLayerHooks(
            config=config,
            engine=mock_engine,
            extractor=mock_extractor,
        )

    # Initialization tests

    def test_init_with_defaults(self) -> None:
        """Test initialization with defaults."""
        hooks = MemoryLayerHooks()

        assert hooks.config is not None
        assert hooks._engine is None
        assert hooks._extractor is None

    def test_init_with_engine(self, mock_engine: MagicMock) -> None:
        """Test initialization with engine."""
        hooks = MemoryLayerHooks(engine=mock_engine)

        assert hooks._engine == mock_engine

    def test_init_with_custom_config(self, tmp_path: Path) -> None:
        """Test initialization with custom config."""
        config = HookConfig(hooks_dir=tmp_path, max_context_memories=25)
        hooks = MemoryLayerHooks(config=config)

        assert hooks.config.hooks_dir == tmp_path
        assert hooks.config.max_context_memories == 25

    # Hook installation tests

    def test_install_all_hooks(self, hooks: MemoryLayerHooks) -> None:
        """Test installing all hooks."""
        installed = hooks.install_hooks()

        assert HookType.PRE_SESSION in installed
        assert HookType.POST_SESSION in installed
        assert HookType.PRE_COMPACT in installed

        # Check files were created
        for hook_type, path in installed.items():
            assert path.exists()
            assert path.stat().st_mode & 0o111  # Executable

    def test_install_specific_hooks(self, hooks: MemoryLayerHooks) -> None:
        """Test installing specific hooks only."""
        installed = hooks.install_hooks([HookType.PRE_SESSION, HookType.POST_SESSION])

        assert HookType.PRE_SESSION in installed
        assert HookType.POST_SESSION in installed
        assert HookType.PRE_COMPACT not in installed

    def test_hook_script_content(self, hooks: MemoryLayerHooks) -> None:
        """Test that hook scripts have correct content."""
        installed = hooks.install_hooks([HookType.PRE_SESSION])

        script_path = installed[HookType.PRE_SESSION]
        content = script_path.read_text()

        assert "#!/usr/bin/env python3" in content
        assert "memory_layer" in content
        assert "pre_session" in content

    def test_hook_script_is_executable(self, hooks: MemoryLayerHooks) -> None:
        """Test that hook scripts are executable."""
        installed = hooks.install_hooks([HookType.PRE_SESSION])
        script_path = installed[HookType.PRE_SESSION]

        # Check executable bit
        assert script_path.stat().st_mode & 0o100

    # Hook uninstallation tests

    def test_uninstall_hooks(self, hooks: MemoryLayerHooks) -> None:
        """Test uninstalling hooks."""
        hooks.install_hooks()
        uninstalled = hooks.uninstall_hooks()

        assert HookType.PRE_SESSION in uninstalled
        assert HookType.POST_SESSION in uninstalled
        assert HookType.PRE_COMPACT in uninstalled

    def test_uninstall_specific_hooks(self, hooks: MemoryLayerHooks) -> None:
        """Test uninstalling specific hooks."""
        hooks.install_hooks()
        uninstalled = hooks.uninstall_hooks([HookType.PRE_SESSION])

        assert HookType.PRE_SESSION in uninstalled
        assert HookType.POST_SESSION not in uninstalled

    def test_uninstall_nonexistent_hooks(self, hooks: MemoryLayerHooks) -> None:
        """Test uninstalling hooks that don't exist."""
        uninstalled = hooks.uninstall_hooks()

        # Should return empty list
        assert uninstalled == []

    # Hook status tests

    def test_is_installed_true(self, hooks: MemoryLayerHooks) -> None:
        """Test is_installed returns True when installed."""
        hooks.install_hooks([HookType.PRE_SESSION])

        assert hooks.is_installed(HookType.PRE_SESSION) is True

    def test_is_installed_false(self, hooks: MemoryLayerHooks) -> None:
        """Test is_installed returns False when not installed."""
        assert hooks.is_installed(HookType.PRE_SESSION) is False

    def test_get_installed_hooks(self, hooks: MemoryLayerHooks) -> None:
        """Test getting list of installed hooks."""
        hooks.install_hooks([HookType.PRE_SESSION, HookType.POST_SESSION])

        installed = hooks.get_installed_hooks()

        assert HookType.PRE_SESSION in installed
        assert HookType.POST_SESSION in installed
        assert HookType.PRE_COMPACT not in installed

    # Pre-session hook tests

    def test_execute_pre_session_with_context(
        self, hooks: MemoryLayerHooks, mock_engine: MagicMock
    ) -> None:
        """Test pre-session hook with context injection."""
        context_mock = MagicMock()
        context_mock.included_count = 3
        context_mock.to_markdown.return_value = "## Memories\n- Memory 1"
        context_mock.memories = []
        mock_engine.get_context.return_value = context_mock

        result = hooks.execute_pre_session()

        assert result.success is True
        assert result.hook_type == HookType.PRE_SESSION
        assert result.memories_processed == 3
        mock_engine.get_context.assert_called()

    def test_execute_pre_session_no_engine(self, tmp_path: Path) -> None:
        """Test pre-session without engine returns empty context."""
        config = HookConfig(hooks_dir=tmp_path)
        hooks = MemoryLayerHooks(config=config)

        result = hooks.execute_pre_session()

        assert result.success is True
        assert result.output == ""
        assert result.memories_processed == 0

    def test_execute_pre_session_with_input_data(
        self, hooks: MemoryLayerHooks, mock_engine: MagicMock
    ) -> None:
        """Test pre-session parses input JSON."""
        context_mock = MagicMock()
        context_mock.included_count = 0
        context_mock.to_markdown.return_value = ""
        context_mock.memories = []
        mock_engine.get_context.return_value = context_mock

        input_data = json.dumps({"session_id": "test-123", "project": "my-project"})
        result = hooks.execute_pre_session(input_data)

        assert result.success is True
        # State should be updated
        assert hooks.state.last_session_id == "test-123"

    def test_execute_pre_session_updates_state(
        self, hooks: MemoryLayerHooks, mock_engine: MagicMock
    ) -> None:
        """Test pre-session updates state."""
        context_mock = MagicMock()
        context_mock.included_count = 1
        context_mock.to_markdown.return_value = "test"
        context_mock.memories = []
        mock_engine.get_context.return_value = context_mock

        hooks.execute_pre_session()

        assert hooks.state.last_session_start is not None
        assert hooks.state.last_context_injection is not None

    # Post-session hook tests

    def test_execute_post_session_with_extraction(
        self, hooks: MemoryLayerHooks, mock_extractor: MagicMock, mock_engine: MagicMock
    ) -> None:
        """Test post-session hook with extraction."""
        mock_memory = MagicMock()
        mock_memory.category.value = "gotcha"
        mock_memory.content = "Test memory content"

        extraction_result = MagicMock()
        extraction_result.success = True
        extraction_result.memory_count = 2
        extraction_result.memories = [mock_memory]
        mock_extractor.extract_and_store.return_value = extraction_result

        transcript = "User: Question\nAssistant: Answer" * 10
        result = hooks.execute_post_session(input_data=transcript)

        assert result.success is True
        assert result.hook_type == HookType.POST_SESSION
        assert result.memories_processed == 2

    def test_execute_post_session_no_extractor(self, mock_engine: MagicMock, tmp_path: Path) -> None:
        """Test post-session without extractor."""
        config = HookConfig(hooks_dir=tmp_path)
        hooks = MemoryLayerHooks(engine=mock_engine, config=config)

        result = hooks.execute_post_session(input_data="Long transcript " * 50)

        assert result.success is True
        assert result.memories_processed == 0

    def test_execute_post_session_updates_state(
        self, hooks: MemoryLayerHooks, mock_extractor: MagicMock
    ) -> None:
        """Test post-session updates state."""
        extraction_result = MagicMock()
        extraction_result.success = True
        extraction_result.memory_count = 3
        extraction_result.memories = []
        mock_extractor.extract_and_store.return_value = extraction_result

        initial_total = hooks.state.memories_extracted_total

        hooks.execute_post_session(input_data="Long transcript " * 50)

        assert hooks.state.last_extraction is not None
        assert hooks.state.memories_extracted_total == initial_total + 3
        assert hooks.state.sessions_processed >= 1

    def test_execute_post_session_parses_json_input(
        self, hooks: MemoryLayerHooks, mock_extractor: MagicMock
    ) -> None:
        """Test post-session parses JSON input."""
        extraction_result = MagicMock()
        extraction_result.success = True
        extraction_result.memory_count = 0
        extraction_result.memories = []
        mock_extractor.extract_and_store.return_value = extraction_result

        input_data = json.dumps({
            "transcript": "Some content " * 50,
            "project": "test-project",
        })

        result = hooks.execute_post_session(input_data=input_data)

        assert result.success is True

    # Pre-compact hook tests

    def test_execute_pre_compact(
        self, hooks: MemoryLayerHooks, mock_extractor: MagicMock
    ) -> None:
        """Test pre-compact hook execution."""
        extraction_result = MagicMock()
        extraction_result.success = True
        extraction_result.memory_count = 5
        extraction_result.memories = []
        mock_extractor.extract_and_store.return_value = extraction_result

        result = hooks.execute_pre_compact(input_data="Context to save " * 50)

        assert result.success is True
        assert result.hook_type == HookType.PRE_COMPACT
        assert result.memories_processed == 5

    def test_execute_pre_compact_updates_state(
        self, hooks: MemoryLayerHooks, mock_extractor: MagicMock
    ) -> None:
        """Test pre-compact updates state."""
        extraction_result = MagicMock()
        extraction_result.success = True
        extraction_result.memory_count = 2
        extraction_result.memories = []
        mock_extractor.extract_and_store.return_value = extraction_result

        initial_total = hooks.state.memories_extracted_total

        hooks.execute_pre_compact(input_data="Context " * 100)

        assert hooks.state.last_extraction is not None
        assert hooks.state.memories_extracted_total == initial_total + 2

    # State persistence tests

    def test_save_and_load_state(self, hooks: MemoryLayerHooks, tmp_path: Path) -> None:
        """Test saving and loading state."""
        hooks.state.last_session_id = "test-session"
        hooks.state.memories_extracted_total = 10
        hooks._save_state()

        # Create new hooks instance to test loading
        config = HookConfig(
            hooks_dir=tmp_path / "hooks",
            state_file=tmp_path / "state.json",
        )
        hooks2 = MemoryLayerHooks(config=config)

        assert hooks2.state.last_session_id == "test-session"
        assert hooks2.state.memories_extracted_total == 10

    def test_load_state_missing_file(self, tmp_path: Path) -> None:
        """Test loading state when file doesn't exist."""
        config = HookConfig(
            hooks_dir=tmp_path / "hooks",
            state_file=tmp_path / "nonexistent.json",
        )
        hooks = MemoryLayerHooks(config=config)

        # Should not raise, should use defaults
        state = hooks.state
        assert state is not None
        assert state.memories_extracted_total == 0

    def test_load_state_invalid_json(self, tmp_path: Path) -> None:
        """Test loading state with invalid JSON."""
        state_file = tmp_path / "invalid_state.json"
        state_file.write_text("invalid json {{{")

        config = HookConfig(
            hooks_dir=tmp_path / "hooks",
            state_file=state_file,
        )
        hooks = MemoryLayerHooks(config=config)

        # Should not raise, should use defaults
        state = hooks.state
        assert state is not None

    # Error handling tests

    def test_pre_session_engine_error(
        self, hooks: MemoryLayerHooks, mock_engine: MagicMock
    ) -> None:
        """Test pre-session handles engine errors."""
        mock_engine.get_context.side_effect = Exception("Database error")

        result = hooks.execute_pre_session()

        assert result.success is False
        assert result.error is not None
        assert "Database error" in result.error

    def test_post_session_extractor_exception(
        self, hooks: MemoryLayerHooks, mock_extractor: MagicMock
    ) -> None:
        """Test post-session handles extractor exceptions."""
        mock_extractor.extract_and_store.side_effect = Exception("Network error")

        result = hooks.execute_post_session(input_data="Long transcript " * 50)

        assert result.success is False
        assert result.error is not None
        assert "Network error" in result.error

    # Configuration output tests

    def test_get_hook_config_json(self, hooks: MemoryLayerHooks) -> None:
        """Test getting hook configuration as JSON."""
        hooks.install_hooks()

        config_json = hooks.get_hook_config_json()
        config = json.loads(config_json)

        assert "hooks" in config
        assert "pre-session" in config["hooks"]
        assert "post-session" in config["hooks"]
        assert "pre-compact" in config["hooks"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_install_all_hooks_function(self, tmp_path: Path) -> None:
        """Test install_all_hooks convenience function."""
        installed = install_all_hooks(hooks_dir=tmp_path)

        assert HookType.PRE_SESSION in installed
        assert HookType.POST_SESSION in installed
        assert HookType.PRE_COMPACT in installed

    def test_uninstall_all_hooks_function(self, tmp_path: Path) -> None:
        """Test uninstall_all_hooks convenience function."""
        install_all_hooks(hooks_dir=tmp_path)
        uninstalled = uninstall_all_hooks(hooks_dir=tmp_path)

        assert HookType.PRE_SESSION in uninstalled
        assert HookType.POST_SESSION in uninstalled
        assert HookType.PRE_COMPACT in uninstalled

    def test_get_hook_status_function(self, tmp_path: Path) -> None:
        """Test get_hook_status convenience function."""
        install_all_hooks(hooks_dir=tmp_path)
        status = get_hook_status(hooks_dir=tmp_path)

        assert "installed_hooks" in status
        assert "hooks_dir" in status
        assert "state" in status
        assert len(status["installed_hooks"]) == 3


class TestHookErrors:
    """Tests for hook error classes."""

    def test_hook_error_base(self) -> None:
        """Test HookError base class."""
        error = HookError("Test error")
        assert str(error) == "Test error"

    def test_hook_not_installed_error(self) -> None:
        """Test HookNotInstalledError."""
        error = HookNotInstalledError("pre-session not installed")
        assert "pre-session" in str(error)

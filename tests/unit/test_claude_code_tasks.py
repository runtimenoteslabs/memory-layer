"""Tests for Claude Code tasks integration (Phase 7).

Tests cover:
- ClaudeCodeTask model
- ClaudeCodeParser
- ClaudeCodeAdapter
- UnifiedTaskAdapter
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_layer.tasks.models import (
    ClaudeCodeTask,
    ClaudeCodeTaskStatus,
    TaskSource,
    TaskSyncResult,
)
from memory_layer.tasks.claude_code_parser import ClaudeCodeParser


# =============================================================================
# ClaudeCodeTask Model Tests
# =============================================================================


class TestClaudeCodeTaskStatus:
    """Tests for ClaudeCodeTaskStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert ClaudeCodeTaskStatus.PENDING.value == "pending"
        assert ClaudeCodeTaskStatus.IN_PROGRESS.value == "in_progress"
        assert ClaudeCodeTaskStatus.COMPLETED.value == "completed"

    def test_status_is_string_enum(self):
        """Test that status values are strings."""
        assert isinstance(ClaudeCodeTaskStatus.PENDING.value, str)
        assert str(ClaudeCodeTaskStatus.PENDING) == "ClaudeCodeTaskStatus.PENDING"


class TestClaudeCodeTask:
    """Tests for ClaudeCodeTask dataclass."""

    def test_create_task(self):
        """Test creating a task with minimal data."""
        task = ClaudeCodeTask(id="cc-test-0", content="Test task")
        assert task.id == "cc-test-0"
        assert task.content == "Test task"
        assert task.status == ClaudeCodeTaskStatus.PENDING

    def test_create_task_with_all_fields(self):
        """Test creating a task with all fields."""
        now = datetime.now(UTC)
        task = ClaudeCodeTask(
            id="cc-abc123-5",
            content="Complete the implementation",
            status=ClaudeCodeTaskStatus.IN_PROGRESS,
            active_form="Completing implementation",
            session_id="abc12345-1234-5678-9abc-def012345678",
            agent_id="agent-123",
            index=5,
            file_path="/home/user/.claude/todos/abc.json",
            created_at=now,
            updated_at=now,
        )

        assert task.id == "cc-abc123-5"
        assert task.content == "Complete the implementation"
        assert task.status == ClaudeCodeTaskStatus.IN_PROGRESS
        assert task.active_form == "Completing implementation"
        assert task.session_id == "abc12345-1234-5678-9abc-def012345678"
        assert task.agent_id == "agent-123"
        assert task.index == 5

    def test_title_property(self):
        """Test the title property truncates content."""
        short_task = ClaudeCodeTask(id="cc-0", content="Short content")
        assert short_task.title == "Short content"

        long_content = "A" * 100
        long_task = ClaudeCodeTask(id="cc-1", content=long_content)
        assert len(long_task.title) == 83  # 80 + "..."
        assert long_task.title.endswith("...")

    def test_description_property(self):
        """Test the description property returns content."""
        task = ClaudeCodeTask(id="cc-0", content="Full description here")
        assert task.description == "Full description here"

    def test_is_completed_property(self):
        """Test the is_completed property."""
        pending = ClaudeCodeTask(id="cc-0", content="t", status=ClaudeCodeTaskStatus.PENDING)
        assert not pending.is_completed

        in_progress = ClaudeCodeTask(id="cc-1", content="t", status=ClaudeCodeTaskStatus.IN_PROGRESS)
        assert not in_progress.is_completed

        completed = ClaudeCodeTask(id="cc-2", content="t", status=ClaudeCodeTaskStatus.COMPLETED)
        assert completed.is_completed

    def test_is_ready_property(self):
        """Test the is_ready property."""
        pending = ClaudeCodeTask(id="cc-0", content="t", status=ClaudeCodeTaskStatus.PENDING)
        assert pending.is_ready

        in_progress = ClaudeCodeTask(id="cc-1", content="t", status=ClaudeCodeTaskStatus.IN_PROGRESS)
        assert not in_progress.is_ready

        completed = ClaudeCodeTask(id="cc-2", content="t", status=ClaudeCodeTaskStatus.COMPLETED)
        assert not completed.is_ready

    def test_source_property(self):
        """Test the source property returns CLAUDE_CODE."""
        task = ClaudeCodeTask(id="cc-0", content="Test")
        assert task.source == TaskSource.CLAUDE_CODE

    def test_to_dict(self):
        """Test serialization to dictionary."""
        task = ClaudeCodeTask(
            id="cc-test-0",
            content="Test content",
            status=ClaudeCodeTaskStatus.IN_PROGRESS,
            active_form="Testing",
            session_id="session-123",
            agent_id="agent-456",
            index=0,
        )
        d = task.to_dict()

        assert d["id"] == "cc-test-0"
        assert d["content"] == "Test content"
        assert d["status"] == "in_progress"
        assert d["activeForm"] == "Testing"
        assert d["session_id"] == "session-123"
        assert d["agent_id"] == "agent-456"
        assert d["index"] == 0
        assert d["source"] == "claude_code"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "content": "Do something",
            "status": "completed",
            "activeForm": "Doing something",
        }
        task = ClaudeCodeTask.from_dict(
            data,
            session_id="sess-abc",
            agent_id="agent-def",
            index=3,
            file_path="/path/to/file.json",
        )

        assert task.content == "Do something"
        assert task.status == ClaudeCodeTaskStatus.COMPLETED
        assert task.active_form == "Doing something"
        assert task.session_id == "sess-abc"
        assert task.agent_id == "agent-def"
        assert task.index == 3
        assert task.file_path == "/path/to/file.json"
        assert task.id == "cc-sess-abc-3"

    def test_from_dict_default_status(self):
        """Test from_dict uses default status for unknown values."""
        data = {"content": "Test"}
        task = ClaudeCodeTask.from_dict(data, session_id="s", index=0)
        assert task.status == ClaudeCodeTaskStatus.PENDING

    def test_from_dict_no_session_id(self):
        """Test from_dict generates ID without session_id."""
        data = {"content": "Test"}
        task = ClaudeCodeTask.from_dict(data, index=5)
        assert task.id == "cc-5"


class TestTaskSource:
    """Tests for TaskSource enum."""

    def test_source_values(self):
        """Test that all expected source values exist."""
        assert TaskSource.BEADS.value == "beads"
        assert TaskSource.CLAUDE_CODE.value == "claude_code"


class TestTaskSyncResult:
    """Tests for TaskSyncResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = TaskSyncResult()
        assert result.source == TaskSource.BEADS
        assert result.tasks_found == 0
        assert result.tasks_synced == 0
        assert result.outcomes_recorded == 0
        assert result.success is True

    def test_with_source(self):
        """Test creating result with specific source."""
        result = TaskSyncResult(source=TaskSource.CLAUDE_CODE)
        assert result.source == TaskSource.CLAUDE_CODE

    def test_success_with_errors(self):
        """Test success property with errors."""
        result = TaskSyncResult(errors=["Something went wrong"])
        assert result.success is False

    def test_to_dict(self):
        """Test serialization."""
        result = TaskSyncResult(
            source=TaskSource.CLAUDE_CODE,
            tasks_found=10,
            tasks_synced=5,
            outcomes_recorded=3,
        )
        d = result.to_dict()

        assert d["source"] == "claude_code"
        assert d["tasks_found"] == 10
        assert d["tasks_synced"] == 5
        assert d["outcomes_recorded"] == 3
        assert d["success"] is True


# =============================================================================
# ClaudeCodeParser Tests
# =============================================================================


class TestClaudeCodeParser:
    """Tests for ClaudeCodeParser."""

    def test_parser_creation(self):
        """Test creating a parser."""
        parser = ClaudeCodeParser()
        assert parser is not None

    def test_parser_with_explicit_dir(self):
        """Test creating parser with explicit directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = ClaudeCodeParser(todos_dir=tmpdir)
            assert parser.todos_dir == Path(tmpdir)
            assert parser.is_available() is True

    def test_parser_nonexistent_dir(self):
        """Test parser with nonexistent directory."""
        parser = ClaudeCodeParser(todos_dir="/nonexistent/path")
        assert parser.is_available() is False

    def test_parse_empty_directory(self):
        """Test parsing empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = ClaudeCodeParser(todos_dir=tmpdir)
            tasks = parser.list_tasks()
            assert tasks == []

    def test_parse_json_file(self):
        """Test parsing a JSON file with tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a task file
            task_file = Path(tmpdir) / "abc12345-agent-def67890.json"
            tasks_data = [
                {"content": "Task 1", "status": "pending", "activeForm": "Working on task 1"},
                {"content": "Task 2", "status": "completed", "activeForm": "Completed task 2"},
                {"content": "Task 3", "status": "in_progress", "activeForm": "Doing task 3"},
            ]
            task_file.write_text(json.dumps(tasks_data))

            parser = ClaudeCodeParser(todos_dir=tmpdir)
            tasks = parser.list_tasks()

            assert len(tasks) == 3
            assert tasks[0].content == "Task 1"
            assert tasks[0].status == ClaudeCodeTaskStatus.PENDING
            assert tasks[0].session_id == "abc12345"
            assert tasks[0].agent_id == "def67890"
            assert tasks[0].index == 0

            assert tasks[1].content == "Task 2"
            assert tasks[1].status == ClaudeCodeTaskStatus.COMPLETED
            assert tasks[1].index == 1

            assert tasks[2].content == "Task 3"
            assert tasks[2].status == ClaudeCodeTaskStatus.IN_PROGRESS
            assert tasks[2].index == 2

    def test_parse_empty_json_file(self):
        """Test parsing empty JSON array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            task_file.write_text("[]")

            parser = ClaudeCodeParser(todos_dir=tmpdir)
            tasks = parser.list_tasks()
            assert tasks == []

    def test_parse_malformed_json(self):
        """Test parsing malformed JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            task_file.write_text("not valid json")

            parser = ClaudeCodeParser(todos_dir=tmpdir)
            tasks = parser.list_tasks()
            assert tasks == []

    def test_filter_by_status(self):
        """Test filtering tasks by status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            tasks_data = [
                {"content": "Pending 1", "status": "pending"},
                {"content": "Pending 2", "status": "pending"},
                {"content": "Done", "status": "completed"},
            ]
            task_file.write_text(json.dumps(tasks_data))

            parser = ClaudeCodeParser(todos_dir=tmpdir)

            pending = parser.list_tasks(status=ClaudeCodeTaskStatus.PENDING)
            assert len(pending) == 2

            completed = parser.list_tasks(status=ClaudeCodeTaskStatus.COMPLETED)
            assert len(completed) == 1

    def test_get_ready_tasks(self):
        """Test getting ready (pending) tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            tasks_data = [
                {"content": "Pending", "status": "pending"},
                {"content": "In Progress", "status": "in_progress"},
            ]
            task_file.write_text(json.dumps(tasks_data))

            parser = ClaudeCodeParser(todos_dir=tmpdir)
            ready = parser.get_ready_tasks()

            assert len(ready) == 1
            assert ready[0].content == "Pending"

    def test_get_completed_tasks(self):
        """Test getting completed tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            tasks_data = [
                {"content": "Pending", "status": "pending"},
                {"content": "Done", "status": "completed"},
            ]
            task_file.write_text(json.dumps(tasks_data))

            parser = ClaudeCodeParser(todos_dir=tmpdir)
            completed = parser.get_completed_tasks()

            assert len(completed) == 1
            assert completed[0].content == "Done"

    def test_get_task_by_id(self):
        """Test getting task by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "abc12345-agent-def67890.json"
            tasks_data = [
                {"content": "First", "status": "pending"},
                {"content": "Second", "status": "pending"},
            ]
            task_file.write_text(json.dumps(tasks_data))

            parser = ClaudeCodeParser(todos_dir=tmpdir)

            task = parser.get_task("cc-abc12345-0")
            assert task is not None
            assert task.content == "First"

            task = parser.get_task("cc-abc12345-1")
            assert task is not None
            assert task.content == "Second"

            task = parser.get_task("nonexistent")
            assert task is None

    def test_get_sessions(self):
        """Test getting unique session IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files for two sessions
            (Path(tmpdir) / "session1-agent-a.json").write_text('[{"content": "t1", "status": "pending"}]')
            (Path(tmpdir) / "session2-agent-b.json").write_text('[{"content": "t2", "status": "pending"}]')

            parser = ClaudeCodeParser(todos_dir=tmpdir)
            sessions = parser.get_sessions()

            assert len(sessions) == 2
            assert "session1" in sessions
            assert "session2" in sessions

    def test_refresh_cache(self):
        """Test refreshing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            task_file.write_text('[{"content": "Original", "status": "pending"}]')

            parser = ClaudeCodeParser(todos_dir=tmpdir)
            assert len(parser.list_tasks()) == 1

            # Modify file
            task_file.write_text('[{"content": "New", "status": "pending"}, {"content": "Another", "status": "pending"}]')

            # Cache should still have old data
            assert len(parser.list_tasks()) == 1

            # After refresh, should see new data
            count = parser.refresh()
            assert count == 2
            assert len(parser.list_tasks()) == 2

    def test_get_stats(self):
        """Test getting statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            tasks_data = [
                {"content": "t1", "status": "pending"},
                {"content": "t2", "status": "in_progress"},
                {"content": "t3", "status": "completed"},
            ]
            task_file.write_text(json.dumps(tasks_data))

            parser = ClaudeCodeParser(todos_dir=tmpdir)
            stats = parser.get_stats()

            assert stats["total_tasks"] == 3
            assert stats["by_status"]["pending"] == 1
            assert stats["by_status"]["in_progress"] == 1
            assert stats["by_status"]["completed"] == 1
            assert stats["sessions"] == 1

    def test_task_list_id_filter(self):
        """Test filtering by task list ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two session files
            (Path(tmpdir) / "target-session-agent-a.json").write_text('[{"content": "target", "status": "pending"}]')
            (Path(tmpdir) / "other-session-agent-b.json").write_text('[{"content": "other", "status": "pending"}]')

            # Parser without filter
            parser_all = ClaudeCodeParser(todos_dir=tmpdir)
            assert len(parser_all.list_tasks()) == 2

            # Parser with filter
            parser_filtered = ClaudeCodeParser(todos_dir=tmpdir, task_list_id="target")
            assert len(parser_filtered.list_tasks()) == 1
            assert parser_filtered.list_tasks()[0].content == "target"


# =============================================================================
# ClaudeCodeAdapter Tests
# =============================================================================


class TestClaudeCodeAdapter:
    """Tests for ClaudeCodeAdapter."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine."""
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.db_path = ":memory:"
        engine.search = AsyncMock(return_value=[])
        engine.get = AsyncMock()
        engine.record_outcome = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_adapter_creation(self, mock_engine):
        """Test creating an adapter."""
        from memory_layer.tasks.claude_code_adapter import ClaudeCodeAdapter

        adapter = ClaudeCodeAdapter(mock_engine)
        assert adapter is not None
        assert adapter.auto_outcome_enabled is True

    @pytest.mark.asyncio
    async def test_adapter_with_explicit_dir(self, mock_engine):
        """Test adapter with explicit directory."""
        from memory_layer.tasks.claude_code_adapter import ClaudeCodeAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = ClaudeCodeAdapter(mock_engine, todos_dir=tmpdir)
            assert adapter.is_available is True

    @pytest.mark.asyncio
    async def test_adapter_initialize(self, mock_engine):
        """Test initializing the adapter."""
        from memory_layer.tasks.claude_code_adapter import ClaudeCodeAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = ClaudeCodeAdapter(mock_engine, todos_dir=tmpdir)
            await adapter.initialize()
            assert adapter._initialized is True

    @pytest.mark.asyncio
    async def test_list_tasks(self, mock_engine):
        """Test listing tasks through adapter."""
        from memory_layer.tasks.claude_code_adapter import ClaudeCodeAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            task_file.write_text('[{"content": "Test task", "status": "pending"}]')

            adapter = ClaudeCodeAdapter(mock_engine, todos_dir=tmpdir)
            tasks = adapter.list_tasks()

            assert len(tasks) == 1
            assert tasks[0].content == "Test task"

    @pytest.mark.asyncio
    async def test_get_current_task(self, mock_engine):
        """Test getting current (in progress) task."""
        from memory_layer.tasks.claude_code_adapter import ClaudeCodeAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            tasks_data = [
                {"content": "Pending", "status": "pending"},
                {"content": "Current", "status": "in_progress"},
            ]
            task_file.write_text(json.dumps(tasks_data))

            adapter = ClaudeCodeAdapter(mock_engine, todos_dir=tmpdir)
            current = adapter.get_current_task()

            assert current is not None
            assert current.content == "Current"

    @pytest.mark.asyncio
    async def test_null_adapter(self, mock_engine):
        """Test null adapter for unavailable Claude Code."""
        from memory_layer.tasks.claude_code_adapter import NullClaudeCodeAdapter

        adapter = NullClaudeCodeAdapter()
        assert adapter.is_available is False
        assert adapter.list_tasks() == []
        assert adapter.get_current_task() is None
        assert await adapter.get_task_memories("any") == []

        result = await adapter.sync()
        assert result.success is True  # No errors is success
        assert "not available" in result.warnings[0]


# =============================================================================
# UnifiedTaskAdapter Tests
# =============================================================================


class TestUnifiedTaskAdapter:
    """Tests for UnifiedTaskAdapter."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine."""
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.db_path = ":memory:"
        engine.search = AsyncMock(return_value=[])
        engine.get = AsyncMock()
        engine.record_outcome = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_adapter_creation(self, mock_engine):
        """Test creating unified adapter."""
        from memory_layer.tasks.unified_adapter import UnifiedTaskAdapter

        adapter = UnifiedTaskAdapter(mock_engine)
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_adapter_initialize(self, mock_engine):
        """Test initializing unified adapter."""
        from memory_layer.tasks.unified_adapter import UnifiedTaskAdapter

        adapter = UnifiedTaskAdapter(mock_engine)
        await adapter.initialize()
        assert adapter._initialized is True

    @pytest.mark.asyncio
    async def test_available_sources(self, mock_engine):
        """Test checking available sources."""
        from memory_layer.tasks.unified_adapter import UnifiedTaskAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = UnifiedTaskAdapter(mock_engine, todos_dir=tmpdir)
            await adapter.initialize()

            # Claude Code should be available (we created the dir)
            assert adapter.claude_code_available is True
            # Beads likely not available
            assert TaskSource.CLAUDE_CODE in adapter.available_sources

    @pytest.mark.asyncio
    async def test_list_tasks_unified(self, mock_engine):
        """Test listing tasks from all sources."""
        from memory_layer.tasks.unified_adapter import UnifiedTaskAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            task_file.write_text('[{"content": "Claude task", "status": "pending"}]')

            adapter = UnifiedTaskAdapter(mock_engine, todos_dir=tmpdir)
            await adapter.initialize()

            tasks = adapter.list_tasks()
            assert len(tasks) >= 1

            # Filter by source
            cc_tasks = adapter.list_tasks(source=TaskSource.CLAUDE_CODE)
            assert len(cc_tasks) == 1
            assert cc_tasks[0].source == TaskSource.CLAUDE_CODE

    @pytest.mark.asyncio
    async def test_get_task_auto_detect_source(self, mock_engine):
        """Test getting task with auto-detected source."""
        from memory_layer.tasks.unified_adapter import UnifiedTaskAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "abc12345-agent-def.json"
            task_file.write_text('[{"content": "Test", "status": "pending"}]')

            adapter = UnifiedTaskAdapter(mock_engine, todos_dir=tmpdir)
            await adapter.initialize()

            # Get by ID - should auto-detect source from prefix
            task = adapter.get_task("cc-abc12345-0")
            assert task is not None
            assert task.source == TaskSource.CLAUDE_CODE

    @pytest.mark.asyncio
    async def test_unified_task_properties(self, mock_engine):
        """Test UnifiedTask wrapper properties."""
        from memory_layer.tasks.unified_adapter import UnifiedTask

        cc_task = ClaudeCodeTask(
            id="cc-test-0",
            content="Test content here",
            status=ClaudeCodeTaskStatus.PENDING,
        )

        unified = UnifiedTask(task=cc_task, source=TaskSource.CLAUDE_CODE)

        assert unified.id == "cc-test-0"
        assert unified.title == "Test content here"
        assert unified.description == "Test content here"
        assert unified.status == "pending"
        assert unified.is_ready is True
        assert unified.is_completed is False

    @pytest.mark.asyncio
    async def test_sync_all(self, mock_engine):
        """Test syncing all sources."""
        from memory_layer.tasks.unified_adapter import UnifiedTaskAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "session-agent.json"
            task_file.write_text('[{"content": "Done", "status": "completed"}]')

            adapter = UnifiedTaskAdapter(mock_engine, todos_dir=tmpdir)
            await adapter.initialize()

            result = await adapter.sync_all()

            assert result is not None
            # Should have at least attempted to sync
            assert hasattr(result, 'results') or hasattr(result, 'tasks_found')


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for Claude Code tasks."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full workflow: parse -> list -> get context."""
        from memory_layer.tasks import ClaudeCodeParser, ClaudeCodeTaskStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            task_file = Path(tmpdir) / "session123-agent-456.json"
            tasks_data = [
                {"content": "First task", "status": "completed", "activeForm": "Completing first"},
                {"content": "Second task", "status": "in_progress", "activeForm": "Working on second"},
                {"content": "Third task", "status": "pending", "activeForm": "Waiting for third"},
            ]
            task_file.write_text(json.dumps(tasks_data))

            # Parse
            parser = ClaudeCodeParser(todos_dir=tmpdir)
            assert parser.is_available()

            # List all
            all_tasks = parser.list_tasks()
            assert len(all_tasks) == 3

            # Filter by status
            completed = parser.get_completed_tasks()
            assert len(completed) == 1
            assert completed[0].content == "First task"

            in_progress = parser.get_in_progress_tasks()
            assert len(in_progress) == 1
            assert in_progress[0].content == "Second task"

            ready = parser.get_ready_tasks()
            assert len(ready) == 1
            assert ready[0].content == "Third task"

            # Get by ID (session_id is truncated to first 8 chars)
            task = parser.get_task("cc-session1-0")
            assert task is not None
            assert task.content == "First task"

            # Stats
            stats = parser.get_stats()
            assert stats["total_tasks"] == 3
            assert stats["sessions"] == 1

"""Tests for Beads task tracker integration.

Tests cover:
- Data models (BeadsTask, TaskMemoryLink, etc.)
- File parser (BeadsParser)
- Task-memory linking (TaskMemoryLinker)
- Automatic outcome capture (OutcomeCapture)
- Unified adapter (BeadsAdapter)
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_layer.tasks import (
    BeadsAdapter,
    BeadsParser,
    BeadsSyncResult,
    BeadsTask,
    BeadsTaskStatus,
    CANCELLED_TASK_PENALTY,
    NullBeadsAdapter,
    OutcomeCapture,
    TaskContext,
    TaskMemoryLink,
    TaskMemoryLinker,
    TASK_STATUS_TO_OUTCOME,
    create_adapter,
)


# =============================================================================
# Test Data Models
# =============================================================================


class TestBeadsTaskStatus:
    """Tests for BeadsTaskStatus enum."""

    def test_status_values(self) -> None:
        """Test all status values exist."""
        assert BeadsTaskStatus.PENDING.value == "pending"
        assert BeadsTaskStatus.IN_PROGRESS.value == "in_progress"
        assert BeadsTaskStatus.DONE.value == "done"
        assert BeadsTaskStatus.BLOCKED.value == "blocked"
        assert BeadsTaskStatus.CANCELLED.value == "cancelled"

    def test_status_count(self) -> None:
        """Test we have exactly 5 statuses."""
        assert len(BeadsTaskStatus) == 5


class TestTaskStatusToOutcome:
    """Tests for TASK_STATUS_TO_OUTCOME mapping."""

    def test_done_maps_to_worked(self) -> None:
        assert TASK_STATUS_TO_OUTCOME[BeadsTaskStatus.DONE] == "worked"

    def test_cancelled_maps_to_failed(self) -> None:
        assert TASK_STATUS_TO_OUTCOME[BeadsTaskStatus.CANCELLED] == "failed"

    def test_blocked_maps_to_partial(self) -> None:
        assert TASK_STATUS_TO_OUTCOME[BeadsTaskStatus.BLOCKED] == "partial"


class TestBeadsTask:
    """Tests for BeadsTask dataclass."""

    def test_create_minimal_task(self) -> None:
        """Test creating task with minimal fields."""
        task = BeadsTask(id="bd-a3f8", title="Test task")
        assert task.id == "bd-a3f8"
        assert task.title == "Test task"
        assert task.status == BeadsTaskStatus.PENDING
        assert task.description == ""
        assert task.parent_id is None
        assert task.dependencies == []
        assert task.tags == []

    def test_create_full_task(self) -> None:
        """Test creating task with all fields."""
        now = datetime.now(UTC)
        task = BeadsTask(
            id="bd-a3f8",
            title="Full task",
            description="A complete task",
            status=BeadsTaskStatus.IN_PROGRESS,
            parent_id="bd-parent",
            dependencies=["bd-dep1", "bd-dep2"],
            tags=["feature", "urgent"],
            created_at=now,
            updated_at=now,
            metadata={"priority": "high"},
        )
        assert task.description == "A complete task"
        assert task.status == BeadsTaskStatus.IN_PROGRESS
        assert task.parent_id == "bd-parent"
        assert len(task.dependencies) == 2
        assert len(task.tags) == 2
        assert task.metadata["priority"] == "high"

    def test_is_subtask_property(self) -> None:
        """Test is_subtask property."""
        parent_task = BeadsTask(id="bd-parent", title="Parent")
        child_task = BeadsTask(id="bd-parent.1", title="Child", parent_id="bd-parent")

        assert not parent_task.is_subtask
        assert child_task.is_subtask

    def test_is_ready_property(self) -> None:
        """Test is_ready property."""
        ready_task = BeadsTask(id="bd-1", title="Ready", status=BeadsTaskStatus.PENDING)
        blocked_task = BeadsTask(
            id="bd-2",
            title="Blocked",
            status=BeadsTaskStatus.PENDING,
            dependencies=["bd-1"],
        )
        in_progress_task = BeadsTask(
            id="bd-3", title="In Progress", status=BeadsTaskStatus.IN_PROGRESS
        )

        assert ready_task.is_ready
        assert not blocked_task.is_ready  # Has dependencies
        assert not in_progress_task.is_ready  # Wrong status

    def test_is_completed_property(self) -> None:
        """Test is_completed property."""
        done_task = BeadsTask(id="bd-1", title="Done", status=BeadsTaskStatus.DONE)
        cancelled_task = BeadsTask(
            id="bd-2", title="Cancelled", status=BeadsTaskStatus.CANCELLED
        )
        pending_task = BeadsTask(
            id="bd-3", title="Pending", status=BeadsTaskStatus.PENDING
        )

        assert done_task.is_completed
        assert cancelled_task.is_completed
        assert not pending_task.is_completed

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        task = BeadsTask(
            id="bd-a3f8",
            title="Test task",
            description="Description",
            status=BeadsTaskStatus.IN_PROGRESS,
        )
        data = task.to_dict()

        assert data["id"] == "bd-a3f8"
        assert data["title"] == "Test task"
        assert data["description"] == "Description"
        assert data["status"] == "in_progress"
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "id": "bd-a3f8",
            "title": "Test task",
            "status": "done",
            "dependencies": ["bd-1", "bd-2"],
            "tags": ["feature"],
        }
        task = BeadsTask.from_dict(data)

        assert task.id == "bd-a3f8"
        assert task.title == "Test task"
        assert task.status == BeadsTaskStatus.DONE
        assert task.dependencies == ["bd-1", "bd-2"]
        assert task.tags == ["feature"]

    def test_from_dict_with_timestamps(self) -> None:
        """Test deserialization with ISO timestamps."""
        data = {
            "id": "bd-a3f8",
            "title": "Test task",
            "created_at": "2024-01-15T10:30:00+00:00",
            "updated_at": "2024-01-15T11:00:00Z",
        }
        task = BeadsTask.from_dict(data)

        assert task.created_at.year == 2024
        assert task.created_at.month == 1
        assert task.created_at.day == 15


class TestTaskMemoryLink:
    """Tests for TaskMemoryLink dataclass."""

    def test_create_link(self) -> None:
        """Test creating a link."""
        link = TaskMemoryLink(task_id="bd-a3f8", memory_id="mem-123")

        assert link.task_id == "bd-a3f8"
        assert link.memory_id == "mem-123"
        assert link.outcome is None
        assert link.context is None

    def test_link_with_outcome(self) -> None:
        """Test link with recorded outcome."""
        link = TaskMemoryLink(
            task_id="bd-a3f8",
            memory_id="mem-123",
            outcome="worked",
            context="Used for authentication",
        )

        assert link.outcome == "worked"
        assert link.context == "Used for authentication"

    def test_to_dict(self) -> None:
        """Test serialization."""
        link = TaskMemoryLink(task_id="bd-a3f8", memory_id="mem-123", outcome="worked")
        data = link.to_dict()

        assert data["task_id"] == "bd-a3f8"
        assert data["memory_id"] == "mem-123"
        assert data["outcome"] == "worked"

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "task_id": "bd-a3f8",
            "memory_id": "mem-123",
            "used_at": "2024-01-15T10:30:00+00:00",
            "outcome": "failed",
        }
        link = TaskMemoryLink.from_dict(data)

        assert link.task_id == "bd-a3f8"
        assert link.memory_id == "mem-123"
        assert link.outcome == "failed"


class TestBeadsSyncResult:
    """Tests for BeadsSyncResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = BeadsSyncResult()

        assert result.tasks_found == 0
        assert result.tasks_synced == 0
        assert result.outcomes_recorded == 0
        assert result.memories_linked == 0
        assert result.errors == []
        assert result.warnings == []

    def test_success_property(self) -> None:
        """Test success property."""
        success_result = BeadsSyncResult(tasks_synced=5, outcomes_recorded=10)
        failure_result = BeadsSyncResult(errors=["Something went wrong"])

        assert success_result.success
        assert not failure_result.success

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = BeadsSyncResult(
            tasks_found=10, tasks_synced=5, outcomes_recorded=15, errors=["Error 1"]
        )
        data = result.to_dict()

        assert data["tasks_found"] == 10
        assert data["tasks_synced"] == 5
        assert data["outcomes_recorded"] == 15
        assert data["errors"] == ["Error 1"]
        assert data["success"] is False


class TestTaskContext:
    """Tests for TaskContext dataclass."""

    def test_to_markdown(self) -> None:
        """Test markdown formatting."""
        task = BeadsTask(
            id="bd-a3f8",
            title="Implement auth",
            description="Add user authentication",
            status=BeadsTaskStatus.IN_PROGRESS,
            dependencies=["bd-setup"],
        )

        # Create mock memories
        mock_memory = MagicMock()
        mock_memory.content = "Use JWT tokens for authentication"
        mock_memory.category = MagicMock()
        mock_memory.category.value = "decision"

        context = TaskContext(task=task, memories=[mock_memory])
        markdown = context.to_markdown()

        assert "## Current Task: Implement auth" in markdown
        assert "**Status:** in_progress" in markdown
        assert "**ID:** bd-a3f8" in markdown
        assert "### Description" in markdown
        assert "Add user authentication" in markdown
        assert "### Blocked By" in markdown
        assert "- bd-setup" in markdown
        assert "### Relevant Memories" in markdown
        assert "[decision]" in markdown


# =============================================================================
# Test File Parser
# =============================================================================


class TestBeadsParser:
    """Tests for BeadsParser."""

    def test_parser_without_beads_dir(self) -> None:
        """Test parser when no .beads/ exists."""
        parser = BeadsParser(beads_dir="/nonexistent/path")
        assert not parser.is_available()

    def test_parser_with_beads_dir(self) -> None:
        """Test parser with valid .beads/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            parser = BeadsParser(beads_dir=beads_dir)
            assert parser.is_available()

    def test_parse_jsonl_file(self) -> None:
        """Test parsing JSONL task files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            # Create a JSONL file with tasks
            tasks_file = beads_dir / "tasks.jsonl"
            tasks = [
                {"id": "bd-001", "title": "Task 1", "status": "pending"},
                {"id": "bd-002", "title": "Task 2", "status": "done"},
                {"id": "bd-003", "title": "Task 3", "status": "in_progress"},
            ]
            with open(tasks_file, "w") as f:
                for task in tasks:
                    f.write(json.dumps(task) + "\n")

            parser = BeadsParser(beads_dir=beads_dir)
            all_tasks = parser.list_tasks()

            assert len(all_tasks) == 3

    def test_list_tasks_by_status(self) -> None:
        """Test filtering tasks by status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            tasks_file = beads_dir / "tasks.jsonl"
            tasks = [
                {"id": "bd-001", "title": "Task 1", "status": "pending"},
                {"id": "bd-002", "title": "Task 2", "status": "done"},
                {"id": "bd-003", "title": "Task 3", "status": "pending"},
            ]
            with open(tasks_file, "w") as f:
                for task in tasks:
                    f.write(json.dumps(task) + "\n")

            parser = BeadsParser(beads_dir=beads_dir)
            pending_tasks = parser.list_tasks(status=BeadsTaskStatus.PENDING)
            done_tasks = parser.list_tasks(status=BeadsTaskStatus.DONE)

            assert len(pending_tasks) == 2
            assert len(done_tasks) == 1

    def test_get_ready_tasks(self) -> None:
        """Test getting ready tasks (no blockers)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            tasks_file = beads_dir / "tasks.jsonl"
            tasks = [
                {"id": "bd-001", "title": "Ready", "status": "pending"},
                {
                    "id": "bd-002",
                    "title": "Blocked",
                    "status": "pending",
                    "dependencies": ["bd-001"],
                },
                {"id": "bd-003", "title": "Done", "status": "done"},
            ]
            with open(tasks_file, "w") as f:
                for task in tasks:
                    f.write(json.dumps(task) + "\n")

            parser = BeadsParser(beads_dir=beads_dir)
            ready = parser.get_ready_tasks()

            assert len(ready) == 1
            assert ready[0].id == "bd-001"

    def test_get_task_by_id(self) -> None:
        """Test getting a specific task by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            tasks_file = beads_dir / "tasks.jsonl"
            with open(tasks_file, "w") as f:
                f.write(json.dumps({"id": "bd-abc", "title": "Target"}) + "\n")

            parser = BeadsParser(beads_dir=beads_dir)
            task = parser.get_task("bd-abc")

            assert task is not None
            assert task.title == "Target"

    def test_get_task_not_found(self) -> None:
        """Test getting a non-existent task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            parser = BeadsParser(beads_dir=beads_dir)
            task = parser.get_task("nonexistent")

            assert task is None

    def test_handle_malformed_json(self) -> None:
        """Test graceful handling of malformed JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            tasks_file = beads_dir / "tasks.jsonl"
            with open(tasks_file, "w") as f:
                f.write(json.dumps({"id": "bd-001", "title": "Valid"}) + "\n")
                f.write("not valid json\n")  # Malformed
                f.write(json.dumps({"id": "bd-002", "title": "Also Valid"}) + "\n")

            parser = BeadsParser(beads_dir=beads_dir)
            tasks = parser.list_tasks()

            # Should only get the valid tasks
            assert len(tasks) == 2

    def test_refresh_cache(self) -> None:
        """Test refreshing the task cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            tasks_file = beads_dir / "tasks.jsonl"

            # Initial state
            with open(tasks_file, "w") as f:
                f.write(json.dumps({"id": "bd-001", "title": "Initial"}) + "\n")

            parser = BeadsParser(beads_dir=beads_dir)
            assert len(parser.list_tasks()) == 1

            # Add more tasks
            with open(tasks_file, "a") as f:
                f.write(json.dumps({"id": "bd-002", "title": "New"}) + "\n")

            # Cache should still show 1
            assert len(parser.list_tasks()) == 1

            # After refresh, should show 2
            parser.refresh()
            assert len(parser.list_tasks()) == 2


# =============================================================================
# Test Task-Memory Linking
# =============================================================================


class TestTaskMemoryLinker:
    """Tests for TaskMemoryLinker."""

    @pytest.fixture
    def db_path(self) -> Path:
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test.db"

    @pytest.mark.asyncio
    async def test_initialize(self, db_path: Path) -> None:
        """Test initializing the linker creates the table."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        # Should not raise on second init
        await linker.initialize()

    @pytest.mark.asyncio
    async def test_link_memory_to_task(self, db_path: Path) -> None:
        """Test linking a memory to a task."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        link = await linker.link("bd-a3f8", "mem-123", context="test context")

        assert link.task_id == "bd-a3f8"
        assert link.memory_id == "mem-123"
        assert link.context == "test context"
        assert link.outcome is None

    @pytest.mark.asyncio
    async def test_link_many(self, db_path: Path) -> None:
        """Test linking multiple memories to a task."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        links = await linker.link_many("bd-a3f8", ["mem-1", "mem-2", "mem-3"])

        assert len(links) == 3
        assert all(link.task_id == "bd-a3f8" for link in links)

    @pytest.mark.asyncio
    async def test_link_idempotent(self, db_path: Path) -> None:
        """Test that linking the same pair twice doesn't create duplicates."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        await linker.link("bd-a3f8", "mem-123")
        await linker.link("bd-a3f8", "mem-123")  # Same link again

        links = await linker.get_memories_for_task("bd-a3f8")
        assert len(links) == 1

    @pytest.mark.asyncio
    async def test_get_memories_for_task(self, db_path: Path) -> None:
        """Test getting all memories linked to a task."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        await linker.link("bd-a3f8", "mem-1")
        await linker.link("bd-a3f8", "mem-2")
        await linker.link("bd-other", "mem-3")

        links = await linker.get_memories_for_task("bd-a3f8")

        assert len(links) == 2
        memory_ids = {link.memory_id for link in links}
        assert memory_ids == {"mem-1", "mem-2"}

    @pytest.mark.asyncio
    async def test_get_tasks_for_memory(self, db_path: Path) -> None:
        """Test getting all tasks using a memory."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        await linker.link("bd-1", "mem-shared")
        await linker.link("bd-2", "mem-shared")
        await linker.link("bd-1", "mem-other")

        links = await linker.get_tasks_for_memory("mem-shared")

        assert len(links) == 2
        task_ids = {link.task_id for link in links}
        assert task_ids == {"bd-1", "bd-2"}

    @pytest.mark.asyncio
    async def test_get_unresolved_links(self, db_path: Path) -> None:
        """Test getting links without recorded outcomes."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        await linker.link("bd-a3f8", "mem-1")
        await linker.link("bd-a3f8", "mem-2")

        # Record outcome for one
        await linker.record_task_outcome("bd-a3f8", "worked")

        # Now add another link (after recording)
        await linker.link("bd-a3f8", "mem-3")

        unresolved = await linker.get_unresolved_links("bd-a3f8")

        # Only mem-3 should be unresolved
        assert len(unresolved) == 1
        assert unresolved[0].memory_id == "mem-3"

    @pytest.mark.asyncio
    async def test_record_task_outcome(self, db_path: Path) -> None:
        """Test recording outcome for all task links."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        await linker.link("bd-a3f8", "mem-1")
        await linker.link("bd-a3f8", "mem-2")

        count = await linker.record_task_outcome("bd-a3f8", "worked")

        assert count == 2

        # All links should now have outcomes
        unresolved = await linker.get_unresolved_links("bd-a3f8")
        assert len(unresolved) == 0

    @pytest.mark.asyncio
    async def test_unlink(self, db_path: Path) -> None:
        """Test removing a link."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        await linker.link("bd-a3f8", "mem-123")
        removed = await linker.unlink("bd-a3f8", "mem-123")

        assert removed is True

        links = await linker.get_memories_for_task("bd-a3f8")
        assert len(links) == 0

    @pytest.mark.asyncio
    async def test_unlink_nonexistent(self, db_path: Path) -> None:
        """Test removing a non-existent link."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        removed = await linker.unlink("bd-none", "mem-none")
        assert removed is False

    @pytest.mark.asyncio
    async def test_get_stats(self, db_path: Path) -> None:
        """Test getting link statistics."""
        linker = TaskMemoryLinker(db_path)
        await linker.initialize()

        await linker.link("bd-1", "mem-1")
        await linker.link("bd-1", "mem-2")
        await linker.link("bd-2", "mem-1")
        await linker.record_task_outcome("bd-1", "worked")

        stats = await linker.get_stats()

        assert stats["total_links"] == 3
        assert stats["unique_tasks"] == 2
        assert stats["unique_memories"] == 2
        assert "worked" in stats["by_outcome"]


# =============================================================================
# Test Outcome Capture
# =============================================================================


class TestOutcomeCapture:
    """Tests for OutcomeCapture."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create a mock MemoryEngine."""
        engine = MagicMock()
        engine.record_outcome = AsyncMock(return_value=[])
        engine.get = AsyncMock()
        return engine

    @pytest.fixture
    def mock_linker(self) -> MagicMock:
        """Create a mock TaskMemoryLinker."""
        linker = MagicMock()
        linker.get_unresolved_links = AsyncMock(return_value=[])
        linker.record_task_outcome = AsyncMock(return_value=0)
        return linker

    @pytest.mark.asyncio
    async def test_on_task_completed(
        self, mock_engine: MagicMock, mock_linker: MagicMock
    ) -> None:
        """Test handling task completion."""
        # Setup mock links
        link1 = TaskMemoryLink(task_id="bd-a3f8", memory_id="mem-1")
        link2 = TaskMemoryLink(task_id="bd-a3f8", memory_id="mem-2")
        mock_linker.get_unresolved_links = AsyncMock(return_value=[link1, link2])

        capture = OutcomeCapture(mock_engine, mock_linker)
        count = await capture.on_task_completed("bd-a3f8")

        assert count == 2
        mock_engine.record_outcome.assert_called_once()
        mock_linker.record_task_outcome.assert_called_once_with("bd-a3f8", "worked")

    @pytest.mark.asyncio
    async def test_on_task_completed_no_links(
        self, mock_engine: MagicMock, mock_linker: MagicMock
    ) -> None:
        """Test handling task completion with no linked memories."""
        mock_linker.get_unresolved_links = AsyncMock(return_value=[])

        capture = OutcomeCapture(mock_engine, mock_linker)
        count = await capture.on_task_completed("bd-a3f8")

        assert count == 0
        mock_engine.record_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_task_failed_disabled(
        self, mock_engine: MagicMock, mock_linker: MagicMock
    ) -> None:
        """Test that task failure doesn't record outcome by default."""
        link = TaskMemoryLink(task_id="bd-a3f8", memory_id="mem-1")
        mock_linker.get_unresolved_links = AsyncMock(return_value=[link])

        capture = OutcomeCapture(
            mock_engine, mock_linker, outcome_on_cancel=False  # Default
        )
        count = await capture.on_task_failed("bd-a3f8")

        assert count == 0
        mock_engine.record_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_task_failed_enabled(
        self, mock_engine: MagicMock, mock_linker: MagicMock
    ) -> None:
        """Test that task failure records outcome when enabled."""
        link = TaskMemoryLink(task_id="bd-a3f8", memory_id="mem-1")
        mock_linker.get_unresolved_links = AsyncMock(return_value=[link])

        capture = OutcomeCapture(mock_engine, mock_linker, outcome_on_cancel=True)
        count = await capture.on_task_failed("bd-a3f8")

        assert count == 1
        mock_engine.record_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_outcome_disabled(
        self, mock_engine: MagicMock, mock_linker: MagicMock
    ) -> None:
        """Test that auto-outcome can be disabled."""
        link = TaskMemoryLink(task_id="bd-a3f8", memory_id="mem-1")
        mock_linker.get_unresolved_links = AsyncMock(return_value=[link])

        capture = OutcomeCapture(
            mock_engine, mock_linker, auto_outcome_enabled=False
        )
        count = await capture.on_task_completed("bd-a3f8")

        assert count == 0
        mock_engine.record_outcome.assert_not_called()


# =============================================================================
# Test Beads Adapter
# =============================================================================


class TestBeadsAdapter:
    """Tests for BeadsAdapter."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create a mock MemoryEngine."""
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.db_path = Path("/tmp/test.db")
        engine.record_outcome = AsyncMock(return_value=[])
        engine.get = AsyncMock()
        engine.search = AsyncMock(return_value=[])
        return engine

    def test_is_available_no_beads(self, mock_engine: MagicMock) -> None:
        """Test is_available when no .beads/ exists."""
        adapter = BeadsAdapter(mock_engine, beads_dir="/nonexistent")
        assert not adapter.is_available

    def test_is_available_with_beads(self, mock_engine: MagicMock) -> None:
        """Test is_available when .beads/ exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            adapter = BeadsAdapter(mock_engine, beads_dir=beads_dir)
            assert adapter.is_available

    @pytest.mark.asyncio
    async def test_initialize(self, mock_engine: MagicMock) -> None:
        """Test adapter initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()
            mock_engine._storage.db_path = Path(tmpdir) / "test.db"

            adapter = BeadsAdapter(mock_engine, beads_dir=beads_dir)
            await adapter.initialize()

            assert adapter._initialized
            assert adapter._linker is not None
            assert adapter._outcome_capture is not None

    def test_get_task(self, mock_engine: MagicMock) -> None:
        """Test getting a task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            tasks_file = beads_dir / "tasks.jsonl"
            with open(tasks_file, "w") as f:
                f.write(json.dumps({"id": "bd-test", "title": "Test"}) + "\n")

            adapter = BeadsAdapter(mock_engine, beads_dir=beads_dir)
            task = adapter.get_task("bd-test")

            assert task is not None
            assert task.title == "Test"

    def test_get_current_task(self, mock_engine: MagicMock) -> None:
        """Test getting current (in_progress) task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            tasks_file = beads_dir / "tasks.jsonl"
            with open(tasks_file, "w") as f:
                f.write(json.dumps({"id": "bd-1", "title": "Pending", "status": "pending"}) + "\n")
                f.write(json.dumps({"id": "bd-2", "title": "Current", "status": "in_progress"}) + "\n")

            adapter = BeadsAdapter(mock_engine, beads_dir=beads_dir)
            current = adapter.get_current_task()

            assert current is not None
            assert current.id == "bd-2"
            assert current.title == "Current"


class TestNullBeadsAdapter:
    """Tests for NullBeadsAdapter."""

    def test_is_available(self) -> None:
        """Test that null adapter is never available."""
        adapter = NullBeadsAdapter()
        assert not adapter.is_available

    @pytest.mark.asyncio
    async def test_operations_return_empty(self) -> None:
        """Test that operations return empty/None values."""
        adapter = NullBeadsAdapter()

        assert adapter.get_task("any") is None
        assert adapter.list_tasks() == []
        assert adapter.get_ready_tasks() == []
        assert adapter.get_current_task() is None

        await adapter.link_memory_to_task("task", "mem")  # Should not raise
        assert await adapter.get_task_memories("task") == []
        assert await adapter.on_task_done("task") == 0
        assert await adapter.get_context_for_injection() == ""

    @pytest.mark.asyncio
    async def test_sync_returns_warning(self) -> None:
        """Test that sync returns a result with warning."""
        adapter = NullBeadsAdapter()
        result = await adapter.sync()

        assert not result.success or len(result.warnings) > 0


class TestCreateAdapter:
    """Tests for create_adapter factory function."""

    def test_creates_null_adapter_when_unavailable(self) -> None:
        """Test that factory returns NullBeadsAdapter when Beads unavailable."""
        mock_engine = MagicMock()
        mock_engine._storage = MagicMock()
        mock_engine._storage.db_path = Path("/tmp/test.db")

        adapter = create_adapter(mock_engine, beads_dir="/nonexistent")
        assert isinstance(adapter, NullBeadsAdapter)

    def test_creates_real_adapter_when_available(self) -> None:
        """Test that factory returns BeadsAdapter when Beads available."""
        mock_engine = MagicMock()
        mock_engine._storage = MagicMock()
        mock_engine._storage.db_path = Path("/tmp/test.db")

        with tempfile.TemporaryDirectory() as tmpdir:
            beads_dir = Path(tmpdir) / ".beads"
            beads_dir.mkdir()

            adapter = create_adapter(mock_engine, beads_dir=beads_dir)
            assert isinstance(adapter, BeadsAdapter)

"""Data models for task tracker integration.

This module defines:
- Enums for task statuses (Beads and Claude Code)
- Dataclasses for tasks, links, and sync results
- Task source enum for unified tracking
- Pydantic models for validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class TaskSource(str, Enum):
    """Source of task data.

    Used to identify which system a task originated from.
    """

    BEADS = "beads"
    """Task from Beads task tracker (.beads/ directory)."""

    CLAUDE_CODE = "claude_code"
    """Task from Claude Code todos (~/.claude/todos/)."""


class BeadsTaskStatus(str, Enum):
    """Task statuses matching Beads conventions.

    These map to the standard Beads task lifecycle.
    """

    PENDING = "pending"
    """Task not yet started."""

    IN_PROGRESS = "in_progress"
    """Task is actively being worked on."""

    DONE = "done"
    """Task completed successfully."""

    BLOCKED = "blocked"
    """Task waiting on a dependency or blocker."""

    CANCELLED = "cancelled"
    """Task was abandoned or no longer needed."""


# Mapping from Beads status to Memory Layer outcome
# Used when auto-recording outcomes on task completion
TASK_STATUS_TO_OUTCOME: dict[BeadsTaskStatus, str] = {
    BeadsTaskStatus.DONE: "worked",
    BeadsTaskStatus.CANCELLED: "failed",
    BeadsTaskStatus.BLOCKED: "partial",
}

# Score adjustment for cancelled tasks (less severe than "failed")
CANCELLED_TASK_PENALTY = -0.1


@dataclass
class BeadsTask:
    """Represents a task from the Beads task tracker.

    This mirrors the Beads JSONL schema for task storage.

    Attributes:
        id: Hash-based task ID (e.g., "bd-a3f8")
        title: Short task title
        description: Full task description
        status: Current task status
        parent_id: Parent task ID for subtasks (e.g., "bd-a3f8" for "bd-a3f8.1")
        dependencies: List of task IDs this task is blocked by
        tags: List of tags for categorization
        created_at: When the task was created
        updated_at: When the task was last modified
        metadata: Additional custom fields from Beads
    """

    id: str
    title: str
    status: BeadsTaskStatus = BeadsTaskStatus.PENDING
    description: str = ""
    parent_id: str | None = None
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_subtask(self) -> bool:
        """Check if this is a subtask (has a parent)."""
        return self.parent_id is not None

    @property
    def is_ready(self) -> bool:
        """Check if task is ready to work on (no blockers, pending status)."""
        return self.status == BeadsTaskStatus.PENDING and len(self.dependencies) == 0

    @property
    def is_completed(self) -> bool:
        """Check if task has reached a terminal state."""
        return self.status in (BeadsTaskStatus.DONE, BeadsTaskStatus.CANCELLED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeadsTask:
        """Create a BeadsTask from a dictionary (e.g., parsed JSONL)."""
        # Handle status as string or enum
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = BeadsTaskStatus(status)

        # Parse timestamps
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif created_at is None:
            created_at = datetime.now(UTC)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        elif updated_at is None:
            updated_at = datetime.now(UTC)

        return cls(
            id=data["id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=status,
            parent_id=data.get("parent_id"),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Claude Code Task Models
# =============================================================================


class ClaudeCodeTaskStatus(str, Enum):
    """Task statuses for Claude Code todos.

    Claude Code uses a simpler status system than Beads.
    """

    PENDING = "pending"
    """Task not yet started."""

    IN_PROGRESS = "in_progress"
    """Task is actively being worked on."""

    COMPLETED = "completed"
    """Task completed."""


# Mapping from Claude Code status to Memory Layer outcome
CLAUDE_CODE_STATUS_TO_OUTCOME: dict[ClaudeCodeTaskStatus, str] = {
    ClaudeCodeTaskStatus.COMPLETED: "worked",
}


@dataclass
class ClaudeCodeTask:
    """Represents a task from Claude Code todos.

    Claude Code stores todos in ~/.claude/todos/ as JSON files.
    Each file is a JSON array of todo objects.

    Attributes:
        id: Generated task ID (session_id:index or content hash)
        content: Task content/description
        status: Current task status
        active_form: Present continuous form shown in spinner
        session_id: Claude session ID the task belongs to
        agent_id: Agent ID (if task is from a sub-agent)
        index: Index within the session's task list
        file_path: Path to the source file
        created_at: When the task was first seen
        updated_at: When the task was last modified
    """

    id: str
    content: str
    status: ClaudeCodeTaskStatus = ClaudeCodeTaskStatus.PENDING
    active_form: str = ""
    session_id: str = ""
    agent_id: str = ""
    index: int = 0
    file_path: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def title(self) -> str:
        """Get task title (first 80 chars of content for compatibility)."""
        return self.content[:80] + ("..." if len(self.content) > 80 else "")

    @property
    def description(self) -> str:
        """Get full task description (alias for content)."""
        return self.content

    @property
    def is_completed(self) -> bool:
        """Check if task has been completed."""
        return self.status == ClaudeCodeTaskStatus.COMPLETED

    @property
    def is_ready(self) -> bool:
        """Check if task is ready to work on."""
        return self.status == ClaudeCodeTaskStatus.PENDING

    @property
    def source(self) -> TaskSource:
        """Get the task source."""
        return TaskSource.CLAUDE_CODE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "activeForm": self.active_form,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "index": self.index,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source.value,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        session_id: str = "",
        agent_id: str = "",
        index: int = 0,
        file_path: str = "",
    ) -> ClaudeCodeTask:
        """Create a ClaudeCodeTask from a dictionary.

        Args:
            data: Dictionary with 'content', 'status', 'activeForm' keys.
            session_id: The Claude session ID.
            agent_id: The agent ID.
            index: Index in the task list.
            file_path: Path to the source file.

        Returns:
            ClaudeCodeTask instance.
        """
        # Handle status mapping
        raw_status = data.get("status", "pending")
        if isinstance(raw_status, str):
            # Map Claude Code statuses to our enum
            status_map = {
                "pending": ClaudeCodeTaskStatus.PENDING,
                "in_progress": ClaudeCodeTaskStatus.IN_PROGRESS,
                "completed": ClaudeCodeTaskStatus.COMPLETED,
            }
            status = status_map.get(raw_status, ClaudeCodeTaskStatus.PENDING)
        else:
            status = raw_status

        content = data.get("content", "")

        # Generate task ID from session_id and index
        task_id = f"cc-{session_id[:8]}-{index}" if session_id else f"cc-{index}"

        # Parse timestamps if present
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif created_at is None:
            created_at = datetime.now(UTC)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        elif updated_at is None:
            updated_at = datetime.now(UTC)

        return cls(
            id=task_id,
            content=content,
            status=status,
            active_form=data.get("activeForm", ""),
            session_id=session_id,
            agent_id=agent_id,
            index=index,
            file_path=file_path,
            created_at=created_at,
            updated_at=updated_at,
        )


# =============================================================================
# Unified Task Type
# =============================================================================

# Type alias for any task type
Task = BeadsTask | ClaudeCodeTask


@dataclass
class TaskMemoryLink:
    """Links a memory to a task it was used for.

    This enables automatic outcome capture: when a task completes,
    all linked memories can have their outcomes recorded.

    Attributes:
        task_id: Beads task ID
        memory_id: Memory Layer memory ID
        used_at: When the memory was surfaced for this task
        outcome: Outcome recorded when task completed (None until then)
        context: Optional context about how memory was used
    """

    task_id: str
    memory_id: str
    used_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    outcome: str | None = None
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "task_id": self.task_id,
            "memory_id": self.memory_id,
            "used_at": self.used_at.isoformat(),
            "outcome": self.outcome,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskMemoryLink:
        """Create from dictionary."""
        used_at = data.get("used_at")
        if isinstance(used_at, str):
            used_at = datetime.fromisoformat(used_at.replace("Z", "+00:00"))
        elif used_at is None:
            used_at = datetime.now(UTC)

        return cls(
            task_id=data["task_id"],
            memory_id=data["memory_id"],
            used_at=used_at,
            outcome=data.get("outcome"),
            context=data.get("context"),
        )


@dataclass
class BeadsSyncResult:
    """Result of syncing with Beads task tracker.

    Attributes:
        tasks_found: Total tasks discovered in .beads/
        tasks_synced: Tasks that were processed
        outcomes_recorded: Number of automatic outcome recordings
        memories_linked: Number of new task-memory links created
        errors: List of error messages encountered
        warnings: List of warning messages
    """

    tasks_found: int = 0
    tasks_synced: int = 0
    outcomes_recorded: int = 0
    memories_linked: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if sync completed without errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tasks_found": self.tasks_found,
            "tasks_synced": self.tasks_synced,
            "outcomes_recorded": self.outcomes_recorded,
            "memories_linked": self.memories_linked,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success,
        }


@dataclass
class TaskSyncResult:
    """Result of syncing with any task tracker.

    Generic version that works with both Beads and Claude Code tasks.

    Attributes:
        source: Which task system was synced
        tasks_found: Total tasks discovered
        tasks_synced: Tasks that were processed
        outcomes_recorded: Number of automatic outcome recordings
        memories_linked: Number of new task-memory links created
        errors: List of error messages encountered
        warnings: List of warning messages
    """

    source: TaskSource = TaskSource.BEADS
    tasks_found: int = 0
    tasks_synced: int = 0
    outcomes_recorded: int = 0
    memories_linked: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if sync completed without errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source.value,
            "tasks_found": self.tasks_found,
            "tasks_synced": self.tasks_synced,
            "outcomes_recorded": self.outcomes_recorded,
            "memories_linked": self.memories_linked,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success,
        }


@dataclass
class TaskContext:
    """Unified context combining task info and relevant memories.

    Used for context injection at the start of a task-focused session.
    Works with both Beads and Claude Code tasks.

    Attributes:
        task: The current task (Beads or Claude Code)
        memories: Relevant memories for this task
        formatted: Pre-formatted string for injection
        source: Which task system the task is from
    """

    task: Task  # BeadsTask | ClaudeCodeTask
    memories: list[Any]  # list[Memory] - Any to avoid circular import
    formatted: str = ""
    source: TaskSource = TaskSource.BEADS

    def to_markdown(self) -> str:
        """Format as markdown for context injection."""
        # Get title - works for both task types
        title = self.task.title if hasattr(self.task, "title") else str(self.task)

        lines = [
            f"## Current Task: {title}",
            "",
            f"**Status:** {self.task.status.value}",
            f"**ID:** {self.task.id}",
            f"**Source:** {self.source.value}",
        ]

        # Get description - works differently for each type
        description = ""
        if isinstance(self.task, BeadsTask):
            description = self.task.description
        elif isinstance(self.task, ClaudeCodeTask):
            description = self.task.content

        if description:
            lines.extend(["", "### Description", description])

        # Dependencies - only for Beads tasks
        if isinstance(self.task, BeadsTask) and self.task.dependencies:
            lines.extend(
                ["", "### Blocked By", *[f"- {dep}" for dep in self.task.dependencies]]
            )

        if self.memories:
            lines.extend(["", "### Relevant Memories", ""])
            for mem in self.memories:
                content = getattr(mem, "content", str(mem))
                category = getattr(mem, "category", "unknown")
                if hasattr(category, "value"):
                    category = category.value
                lines.append(f"- **[{category}]** {content[:200]}...")

        return "\n".join(lines)

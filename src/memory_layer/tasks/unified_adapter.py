"""Unified task adapter combining Beads and Claude Code task systems.

This module provides a single interface that works with both task systems,
auto-detecting which systems are available and providing unified access.

Example:
    >>> from memory_layer.tasks import UnifiedTaskAdapter
    >>> adapter = UnifiedTaskAdapter(engine)
    >>> await adapter.initialize()
    >>>
    >>> # List all tasks from all sources
    >>> tasks = adapter.list_tasks()
    >>>
    >>> # Sync outcomes for all completed tasks
    >>> results = await adapter.sync_all()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from memory_layer.tasks.adapter import BeadsAdapter, NullBeadsAdapter, create_adapter
from memory_layer.tasks.claude_code_adapter import (
    ClaudeCodeAdapter,
    NullClaudeCodeAdapter,
    create_claude_code_adapter,
)
from memory_layer.tasks.models import (
    BeadsTask,
    BeadsTaskStatus,
    ClaudeCodeTask,
    ClaudeCodeTaskStatus,
    Task,
    TaskContext,
    TaskSource,
    TaskSyncResult,
)

if TYPE_CHECKING:
    from memory_layer.core.engine import MemoryEngine
    from memory_layer.core.models import Memory

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTask:
    """Wrapper providing unified interface for any task type.

    Normalizes differences between Beads and Claude Code tasks.
    """

    task: Task  # BeadsTask | ClaudeCodeTask
    source: TaskSource

    @property
    def id(self) -> str:
        return self.task.id

    @property
    def title(self) -> str:
        return self.task.title

    @property
    def description(self) -> str:
        if isinstance(self.task, BeadsTask):
            return self.task.description
        return self.task.content

    @property
    def status(self) -> str:
        """Get normalized status string."""
        return self.task.status.value

    @property
    def is_completed(self) -> bool:
        return self.task.is_completed

    @property
    def is_ready(self) -> bool:
        return self.task.is_ready

    def to_dict(self) -> dict:
        """Convert to dictionary with source info."""
        data = self.task.to_dict()
        data["source"] = self.source.value
        return data


@dataclass
class UnifiedSyncResult:
    """Combined sync results from all task sources.

    Attributes:
        results: Individual results by source
        total_tasks_found: Total across all sources
        total_outcomes_recorded: Total outcomes recorded
        errors: All errors from all sources
    """

    results: dict[TaskSource, TaskSyncResult] = field(default_factory=dict)

    @property
    def total_tasks_found(self) -> int:
        return sum(r.tasks_found for r in self.results.values())

    @property
    def total_tasks_synced(self) -> int:
        return sum(r.tasks_synced for r in self.results.values())

    @property
    def total_outcomes_recorded(self) -> int:
        return sum(r.outcomes_recorded for r in self.results.values())

    @property
    def errors(self) -> list[str]:
        errors = []
        for source, result in self.results.items():
            for error in result.errors:
                errors.append(f"[{source.value}] {error}")
        return errors

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "results": {k.value: v.to_dict() for k, v in self.results.items()},
            "total_tasks_found": self.total_tasks_found,
            "total_tasks_synced": self.total_tasks_synced,
            "total_outcomes_recorded": self.total_outcomes_recorded,
            "errors": self.errors,
            "success": self.success,
        }


class UnifiedTaskAdapter:
    """Unified adapter that works with both Beads and Claude Code tasks.

    Provides a single interface for:
    - Listing tasks from all available sources
    - Linking memories to tasks
    - Recording outcomes
    - Generating unified context

    Auto-detects which task systems are available.
    """

    def __init__(
        self,
        engine: MemoryEngine,
        beads_dir: Path | str | None = None,
        todos_dir: Path | str | None = None,
        auto_outcome_enabled: bool = True,
    ) -> None:
        """Initialize the unified adapter.

        Args:
            engine: The Memory Layer engine.
            beads_dir: Explicit path to .beads/ directory.
            todos_dir: Explicit path to ~/.claude/todos/ directory.
            auto_outcome_enabled: Whether to auto-record outcomes.
        """
        self._engine = engine
        self._beads_dir = beads_dir
        self._todos_dir = todos_dir
        self._auto_outcome_enabled = auto_outcome_enabled

        # Adapters will be created on initialize()
        self._beads_adapter: BeadsAdapter | NullBeadsAdapter | None = None
        self._claude_adapter: ClaudeCodeAdapter | NullClaudeCodeAdapter | None = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all adapters.

        Creates adapters for available task systems.
        Must be called before using the unified adapter.
        """
        if self._initialized:
            return

        # Create Beads adapter
        self._beads_adapter = create_adapter(
            self._engine,
            self._beads_dir,
            auto_outcome_enabled=self._auto_outcome_enabled,
        )
        if isinstance(self._beads_adapter, BeadsAdapter):
            await self._beads_adapter.initialize()

        # Create Claude Code adapter
        self._claude_adapter = create_claude_code_adapter(
            self._engine,
            self._todos_dir,
            auto_outcome_enabled=self._auto_outcome_enabled,
        )
        if isinstance(self._claude_adapter, ClaudeCodeAdapter):
            await self._claude_adapter.initialize()

        self._initialized = True
        logger.debug(
            f"UnifiedTaskAdapter initialized: beads={self._beads_adapter.is_available}, "
            f"claude_code={self._claude_adapter.is_available}"
        )

    def _ensure_initialized(self) -> None:
        """Ensure the adapter is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "UnifiedTaskAdapter not initialized. Call initialize() first."
            )

    @property
    def beads_available(self) -> bool:
        """Check if Beads is available."""
        return self._beads_adapter is not None and self._beads_adapter.is_available

    @property
    def claude_code_available(self) -> bool:
        """Check if Claude Code todos are available."""
        return self._claude_adapter is not None and self._claude_adapter.is_available

    @property
    def available_sources(self) -> list[TaskSource]:
        """Get list of available task sources."""
        sources = []
        if self.beads_available:
            sources.append(TaskSource.BEADS)
        if self.claude_code_available:
            sources.append(TaskSource.CLAUDE_CODE)
        return sources

    # =========================================================================
    # Task Operations
    # =========================================================================

    def get_task(
        self,
        task_id: str,
        source: TaskSource | None = None,
    ) -> UnifiedTask | None:
        """Get a task by ID.

        Args:
            task_id: The task ID.
            source: Which source to check. If None, checks all sources.

        Returns:
            UnifiedTask wrapper or None if not found.
        """
        self._ensure_initialized()

        # Check Beads
        if source in (None, TaskSource.BEADS) and self.beads_available:
            task = self._beads_adapter.get_task(task_id)
            if task:
                return UnifiedTask(task=task, source=TaskSource.BEADS)

        # Check Claude Code
        if source in (None, TaskSource.CLAUDE_CODE) and self.claude_code_available:
            task = self._claude_adapter.get_task(task_id)
            if task:
                return UnifiedTask(task=task, source=TaskSource.CLAUDE_CODE)

        return None

    def list_tasks(
        self,
        source: TaskSource | None = None,
        status: str | None = None,
    ) -> list[UnifiedTask]:
        """List all tasks, optionally filtered.

        Args:
            source: Filter by source. If None, includes all sources.
            status: Filter by status (normalized: "pending", "in_progress", "done"/"completed").

        Returns:
            List of UnifiedTask wrappers.
        """
        self._ensure_initialized()

        tasks = []

        # Get Beads tasks
        if source in (None, TaskSource.BEADS) and self.beads_available:
            beads_status = None
            if status:
                # Map normalized status to Beads status
                status_map = {
                    "pending": BeadsTaskStatus.PENDING,
                    "in_progress": BeadsTaskStatus.IN_PROGRESS,
                    "done": BeadsTaskStatus.DONE,
                    "completed": BeadsTaskStatus.DONE,
                    "blocked": BeadsTaskStatus.BLOCKED,
                    "cancelled": BeadsTaskStatus.CANCELLED,
                }
                beads_status = status_map.get(status.lower())

            for task in self._beads_adapter.list_tasks(status=beads_status):
                tasks.append(UnifiedTask(task=task, source=TaskSource.BEADS))

        # Get Claude Code tasks
        if source in (None, TaskSource.CLAUDE_CODE) and self.claude_code_available:
            cc_status = None
            if status:
                # Map normalized status to Claude Code status
                status_map = {
                    "pending": ClaudeCodeTaskStatus.PENDING,
                    "in_progress": ClaudeCodeTaskStatus.IN_PROGRESS,
                    "done": ClaudeCodeTaskStatus.COMPLETED,
                    "completed": ClaudeCodeTaskStatus.COMPLETED,
                }
                cc_status = status_map.get(status.lower())

            for task in self._claude_adapter.list_tasks(status=cc_status):
                tasks.append(UnifiedTask(task=task, source=TaskSource.CLAUDE_CODE))

        return tasks

    def get_ready_tasks(
        self,
        source: TaskSource | None = None,
    ) -> list[UnifiedTask]:
        """Get tasks that are ready to work on.

        Args:
            source: Filter by source. If None, includes all sources.

        Returns:
            List of ready tasks.
        """
        self._ensure_initialized()

        tasks = []

        if source in (None, TaskSource.BEADS) and self.beads_available:
            for task in self._beads_adapter.get_ready_tasks():
                tasks.append(UnifiedTask(task=task, source=TaskSource.BEADS))

        if source in (None, TaskSource.CLAUDE_CODE) and self.claude_code_available:
            for task in self._claude_adapter.get_ready_tasks():
                tasks.append(UnifiedTask(task=task, source=TaskSource.CLAUDE_CODE))

        return tasks

    def get_current_task(
        self,
        source: TaskSource | None = None,
    ) -> UnifiedTask | None:
        """Get the currently active task (in_progress).

        Args:
            source: Which source to check first. If None, prefers Claude Code.

        Returns:
            The in-progress task, or None if no task is active.
        """
        self._ensure_initialized()

        # Check Claude Code first (more likely to be active during Claude sessions)
        if source in (None, TaskSource.CLAUDE_CODE) and self.claude_code_available:
            task = self._claude_adapter.get_current_task()
            if task:
                return UnifiedTask(task=task, source=TaskSource.CLAUDE_CODE)

        # Check Beads
        if source in (None, TaskSource.BEADS) and self.beads_available:
            task = self._beads_adapter.get_current_task()
            if task:
                return UnifiedTask(task=task, source=TaskSource.BEADS)

        return None

    # =========================================================================
    # Memory Linking
    # =========================================================================

    async def link_memory_to_task(
        self,
        task_id: str,
        memory_id: str,
        source: TaskSource | None = None,
        context: str | None = None,
    ) -> None:
        """Link a memory to a task.

        Args:
            task_id: The task ID.
            memory_id: The Memory Layer memory ID.
            source: Which adapter to use. Auto-detects if None.
            context: Optional context about how memory was used.
        """
        self._ensure_initialized()

        # Auto-detect source from task ID prefix
        if source is None:
            if task_id.startswith("cc-"):
                source = TaskSource.CLAUDE_CODE
            elif task_id.startswith("bd-"):
                source = TaskSource.BEADS
            else:
                # Try both
                source = TaskSource.BEADS if self.beads_available else TaskSource.CLAUDE_CODE

        if source == TaskSource.BEADS and self.beads_available:
            await self._beads_adapter.link_memory_to_task(task_id, memory_id, context)
        elif source == TaskSource.CLAUDE_CODE and self.claude_code_available:
            await self._claude_adapter.link_memory_to_task(task_id, memory_id, context)

    async def get_task_memories(
        self,
        task_id: str,
        source: TaskSource | None = None,
    ) -> list[Memory]:
        """Get all memories linked to a task.

        Args:
            task_id: The task ID.
            source: Which adapter to use. Auto-detects if None.

        Returns:
            List of Memory objects linked to the task.
        """
        self._ensure_initialized()

        # Auto-detect source from task ID prefix
        if source is None:
            if task_id.startswith("cc-"):
                source = TaskSource.CLAUDE_CODE
            elif task_id.startswith("bd-"):
                source = TaskSource.BEADS

        if source == TaskSource.BEADS and self.beads_available:
            return await self._beads_adapter.get_task_memories(task_id)
        elif source == TaskSource.CLAUDE_CODE and self.claude_code_available:
            return await self._claude_adapter.get_task_memories(task_id)

        return []

    # =========================================================================
    # Outcome Capture
    # =========================================================================

    async def on_task_completed(
        self,
        task_id: str,
        source: TaskSource | None = None,
    ) -> int:
        """Handle a task being marked as completed.

        Args:
            task_id: The task ID.
            source: Which adapter to use. Auto-detects if None.

        Returns:
            Number of memories that had outcomes recorded.
        """
        self._ensure_initialized()

        # Auto-detect source from task ID prefix
        if source is None:
            if task_id.startswith("cc-"):
                source = TaskSource.CLAUDE_CODE
            elif task_id.startswith("bd-"):
                source = TaskSource.BEADS

        if source == TaskSource.BEADS and self.beads_available:
            return await self._beads_adapter.on_task_done(task_id)
        elif source == TaskSource.CLAUDE_CODE and self.claude_code_available:
            return await self._claude_adapter.on_task_completed(task_id)

        return 0

    async def sync(
        self,
        source: TaskSource | None = None,
    ) -> TaskSyncResult | UnifiedSyncResult:
        """Sync outcomes for completed tasks.

        Args:
            source: Which source to sync. If None, syncs all sources.

        Returns:
            TaskSyncResult if single source, UnifiedSyncResult if all sources.
        """
        self._ensure_initialized()

        if source == TaskSource.BEADS:
            return await self._beads_adapter.sync()
        elif source == TaskSource.CLAUDE_CODE:
            return await self._claude_adapter.sync()

        # Sync all sources
        result = UnifiedSyncResult()

        if self.beads_available:
            result.results[TaskSource.BEADS] = await self._beads_adapter.sync()

        if self.claude_code_available:
            result.results[TaskSource.CLAUDE_CODE] = await self._claude_adapter.sync()

        return result

    async def sync_all(self) -> UnifiedSyncResult:
        """Sync outcomes for all completed tasks from all sources.

        Returns:
            UnifiedSyncResult with combined statistics.
        """
        result = await self.sync(source=None)
        if isinstance(result, UnifiedSyncResult):
            return result
        # Convert single result to unified
        unified = UnifiedSyncResult()
        unified.results[result.source] = result
        return unified

    # =========================================================================
    # Context Generation
    # =========================================================================

    async def get_unified_context(
        self,
        task_id: str | None = None,
        source: TaskSource | None = None,
        max_memories: int = 10,
    ) -> TaskContext | None:
        """Get unified context combining task info and relevant memories.

        Args:
            task_id: The task ID. If None, uses current task.
            source: Which source. Auto-detects if None.
            max_memories: Maximum number of memories to include.

        Returns:
            TaskContext object or None if no task found.
        """
        self._ensure_initialized()

        # Auto-detect source from task ID or find current task
        if task_id:
            if task_id.startswith("cc-"):
                source = TaskSource.CLAUDE_CODE
            elif task_id.startswith("bd-"):
                source = TaskSource.BEADS

        if source == TaskSource.BEADS and self.beads_available:
            return await self._beads_adapter.get_unified_context(task_id, max_memories)
        elif source == TaskSource.CLAUDE_CODE and self.claude_code_available:
            return await self._claude_adapter.get_unified_context(task_id, max_memories)

        # No source specified, try current task from any source
        current = self.get_current_task()
        if current:
            if current.source == TaskSource.BEADS:
                return await self._beads_adapter.get_unified_context(
                    current.id, max_memories
                )
            else:
                return await self._claude_adapter.get_unified_context(
                    current.id, max_memories
                )

        return None

    async def get_context_for_injection(
        self,
        task_id: str | None = None,
        source: TaskSource | None = None,
        max_memories: int = 10,
    ) -> str:
        """Get formatted context string for injection into prompts.

        Args:
            task_id: The task ID. If None, uses current task.
            source: Which source. Auto-detects if None.
            max_memories: Maximum number of memories to include.

        Returns:
            Formatted markdown string for context injection.
        """
        context = await self.get_unified_context(task_id, source, max_memories)
        if not context:
            return ""
        return context.formatted

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict:
        """Get statistics about all task integrations.

        Returns:
            Dict with statistics from all sources.
        """
        self._ensure_initialized()

        stats = {
            "available_sources": [s.value for s in self.available_sources],
            "beads": {},
            "claude_code": {},
        }

        if self.beads_available:
            stats["beads"] = await self._beads_adapter.get_stats()

        if self.claude_code_available:
            stats["claude_code"] = await self._claude_adapter.get_stats()

        return stats

    def refresh(self) -> dict[TaskSource, int]:
        """Refresh task caches from disk.

        Returns:
            Dict mapping source to number of tasks loaded.
        """
        self._ensure_initialized()

        counts = {}

        if self.beads_available:
            counts[TaskSource.BEADS] = self._beads_adapter.refresh()

        if self.claude_code_available:
            counts[TaskSource.CLAUDE_CODE] = self._claude_adapter.refresh()

        return counts


def create_unified_adapter(
    engine: MemoryEngine,
    beads_dir: Path | str | None = None,
    todos_dir: Path | str | None = None,
    **kwargs,
) -> UnifiedTaskAdapter:
    """Factory function to create a unified adapter.

    Args:
        engine: The Memory Layer engine.
        beads_dir: Explicit path to .beads/ directory.
        todos_dir: Explicit path to ~/.claude/todos/ directory.
        **kwargs: Additional arguments.

    Returns:
        UnifiedTaskAdapter instance.
    """
    return UnifiedTaskAdapter(
        engine,
        beads_dir=beads_dir,
        todos_dir=todos_dir,
        **kwargs,
    )

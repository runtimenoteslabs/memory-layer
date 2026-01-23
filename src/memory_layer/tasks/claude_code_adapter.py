"""Claude Code tasks adapter for Memory Layer integration.

This module provides integration with Claude Code's native todos system,
enabling automatic outcome capture when tasks complete.

Example:
    >>> from memory_layer.tasks import ClaudeCodeAdapter
    >>> adapter = ClaudeCodeAdapter(engine)
    >>> await adapter.initialize()
    >>>
    >>> # List tasks
    >>> tasks = adapter.list_tasks()
    >>>
    >>> # Get context for current task
    >>> context = await adapter.get_unified_context()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from memory_layer.core.models import Memory
from memory_layer.tasks.claude_code_parser import ClaudeCodeParser
from memory_layer.tasks.linking import TaskMemoryLinker
from memory_layer.tasks.models import (
    CLAUDE_CODE_STATUS_TO_OUTCOME,
    ClaudeCodeTask,
    ClaudeCodeTaskStatus,
    TaskContext,
    TaskSource,
    TaskSyncResult,
)

if TYPE_CHECKING:
    from memory_layer.core.engine import MemoryEngine

logger = logging.getLogger(__name__)


class ClaudeCodeAdapter:
    """Adapter for Claude Code todos integration.

    Provides similar interface to BeadsAdapter but for Claude Code's
    native task system at ~/.claude/todos/.

    Key differences from Beads:
    - Simpler status model (pending, in_progress, completed)
    - No dependencies/blocking
    - Tasks organized by session/agent
    """

    def __init__(
        self,
        engine: MemoryEngine,
        todos_dir: Path | str | None = None,
        task_list_id: str | None = None,
        auto_outcome_enabled: bool = True,
    ) -> None:
        """Initialize the Claude Code adapter.

        Args:
            engine: The Memory Layer engine.
            todos_dir: Explicit path to todos directory (auto-discovers if None).
            task_list_id: Filter to specific task list (session/agent).
            auto_outcome_enabled: Whether to auto-record outcomes on task completion.
        """
        self._engine = engine
        self._todos_dir = Path(todos_dir) if todos_dir else None
        self._task_list_id = task_list_id

        # Initialize components
        self._parser = ClaudeCodeParser(self._todos_dir, task_list_id)
        self._linker: TaskMemoryLinker | None = None

        # Configuration
        self.auto_outcome_enabled = auto_outcome_enabled

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components.

        Creates the linker table and sets up components.
        Must be called before using the adapter.
        """
        if self._initialized:
            return

        # Get db_path from engine's storage
        db_path = self._engine._storage.db_path

        # Initialize linker
        self._linker = TaskMemoryLinker(db_path)
        await self._linker.initialize()

        self._initialized = True
        logger.debug("ClaudeCodeAdapter initialized")

    def _ensure_initialized(self) -> None:
        """Ensure the adapter is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "ClaudeCodeAdapter not initialized. Call initialize() first."
            )

    @property
    def is_available(self) -> bool:
        """Check if Claude Code todos are available."""
        return self._parser.is_available()

    # =========================================================================
    # Task Operations
    # =========================================================================

    def get_task(self, task_id: str) -> ClaudeCodeTask | None:
        """Get a task by ID.

        Args:
            task_id: The task ID (e.g., "cc-abc12345-0").

        Returns:
            The task or None if not found.
        """
        return self._parser.get_task(task_id)

    def list_tasks(
        self,
        status: ClaudeCodeTaskStatus | None = None,
        session_id: str | None = None,
    ) -> list[ClaudeCodeTask]:
        """List all tasks, optionally filtered.

        Args:
            status: Optional status filter.
            session_id: Optional session ID filter.

        Returns:
            List of matching tasks.
        """
        return self._parser.list_tasks(status=status, session_id=session_id)

    def get_ready_tasks(self) -> list[ClaudeCodeTask]:
        """Get tasks that are ready to work on.

        Returns:
            List of tasks with pending status.
        """
        return self._parser.get_ready_tasks()

    def get_current_task(self) -> ClaudeCodeTask | None:
        """Get the currently active task (in_progress).

        Returns:
            The in-progress task, or None if no task is active.
        """
        in_progress = self._parser.get_in_progress_tasks()
        return in_progress[0] if in_progress else None

    def get_sessions(self) -> list[str]:
        """Get list of unique session IDs.

        Returns:
            List of session IDs that have tasks.
        """
        return self._parser.get_sessions()

    # =========================================================================
    # Memory Linking
    # =========================================================================

    async def link_memory_to_task(
        self,
        task_id: str,
        memory_id: str,
        context: str | None = None,
    ) -> None:
        """Link a memory to a task.

        Args:
            task_id: The task ID.
            memory_id: The Memory Layer memory ID.
            context: Optional context about how memory was used.
        """
        self._ensure_initialized()
        await self._linker.link(task_id, memory_id, context)

    async def link_memories_to_task(
        self,
        task_id: str,
        memory_ids: list[str],
        context: str | None = None,
    ) -> None:
        """Link multiple memories to a task.

        Args:
            task_id: The task ID.
            memory_ids: List of Memory Layer memory IDs.
            context: Optional context about how memories were used.
        """
        self._ensure_initialized()
        await self._linker.link_many(task_id, memory_ids, context)

    async def get_task_memories(self, task_id: str) -> list[Memory]:
        """Get all memories linked to a task.

        Args:
            task_id: The task ID.

        Returns:
            List of Memory objects linked to the task.
        """
        self._ensure_initialized()
        links = await self._linker.get_memories_for_task(task_id)
        memories = []
        for link in links:
            try:
                memory = await self._engine.get(link.memory_id)
                memories.append(memory)
            except Exception:
                logger.debug(f"Memory {link.memory_id} not found, skipping")
        return memories

    async def auto_link_search_results(
        self,
        task_id: str,
        memory_ids: list[str],
    ) -> None:
        """Automatically link search results to the current task.

        Args:
            task_id: The task ID.
            memory_ids: Memory IDs from search results.
        """
        self._ensure_initialized()
        await self._linker.link_many(task_id, memory_ids, context="search_result")

    # =========================================================================
    # Outcome Capture
    # =========================================================================

    async def on_task_completed(self, task_id: str) -> int:
        """Handle a task being marked as completed.

        Records "worked" outcome for all linked memories.

        Args:
            task_id: The task ID.

        Returns:
            Number of memories that had outcomes recorded.
        """
        self._ensure_initialized()

        if not self.auto_outcome_enabled:
            return 0

        # Get unresolved links for this task
        links = await self._linker.get_unresolved_links(task_id)
        if not links:
            return 0

        # Record outcome for all linked memories
        memory_ids = [link.memory_id for link in links]
        from memory_layer.core.models import Outcome

        await self._engine.record_outcome(memory_ids, Outcome.WORKED)

        # Mark links as resolved
        await self._linker.record_task_outcome(task_id, "worked")

        logger.info(f"Recorded 'worked' outcome for {len(links)} memories on task {task_id}")
        return len(links)

    async def sync(self) -> TaskSyncResult:
        """Sync outcomes for all completed tasks.

        Scans completed tasks and records outcomes for any that
        have unresolved memory links.

        Returns:
            TaskSyncResult with sync statistics.
        """
        self._ensure_initialized()

        result = TaskSyncResult(source=TaskSource.CLAUDE_CODE)
        completed_tasks = self._parser.get_completed_tasks()
        result.tasks_found = len(self._parser.list_tasks())

        for task in completed_tasks:
            try:
                count = await self.on_task_completed(task.id)
                if count > 0:
                    result.tasks_synced += 1
                    result.outcomes_recorded += count
            except Exception as e:
                result.errors.append(f"Failed to sync task {task.id}: {e}")

        return result

    # =========================================================================
    # Context Generation
    # =========================================================================

    async def get_unified_context(
        self,
        task_id: str | None = None,
        max_memories: int = 10,
    ) -> TaskContext | None:
        """Get unified context combining task info and relevant memories.

        Args:
            task_id: The task ID to get context for. If None, uses current task.
            max_memories: Maximum number of memories to include.

        Returns:
            TaskContext object or None if no task found.
        """
        self._ensure_initialized()

        # Get task
        if task_id:
            task = self.get_task(task_id)
        else:
            task = self.get_current_task()

        if not task:
            return None

        # Get linked memories
        memories = await self.get_task_memories(task.id)

        # If no linked memories, search for relevant ones
        if not memories and task.content:
            search_query = task.content[:200]
            results = await self._engine.search(
                search_query,
                limit=max_memories,
                min_score=0.5,
                track_usage=False,
            )
            memories = [r.memory for r in results if r.score >= 0.5]

        # Limit memories
        memories = memories[:max_memories]

        # Create context
        context = TaskContext(
            task=task,
            memories=memories,
            source=TaskSource.CLAUDE_CODE,
        )
        context.formatted = context.to_markdown()

        return context

    async def get_context_for_injection(
        self,
        task_id: str | None = None,
        max_memories: int = 10,
    ) -> str:
        """Get formatted context string for injection into prompts.

        Args:
            task_id: The task ID to get context for. If None, uses current task.
            max_memories: Maximum number of memories to include.

        Returns:
            Formatted markdown string for context injection.
        """
        context = await self.get_unified_context(task_id, max_memories)
        if not context:
            return ""
        return context.formatted

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict:
        """Get statistics about Claude Code integration.

        Returns:
            Dict with statistics.
        """
        self._ensure_initialized()

        # Parser stats
        parser_stats = self._parser.get_stats()

        # Link stats
        link_stats = await self._linker.get_stats()

        return {
            "claude_code_available": self.is_available,
            "todos_dir": str(self._parser.todos_dir) if self._parser.todos_dir else None,
            "tasks": parser_stats,
            "links": link_stats,
            "auto_outcome_enabled": self.auto_outcome_enabled,
        }

    def refresh(self) -> int:
        """Refresh task cache from disk.

        Returns:
            Number of tasks loaded.
        """
        return self._parser.refresh()


class NullClaudeCodeAdapter:
    """Null adapter for when Claude Code todos are not available.

    Provides the same interface but does nothing, allowing graceful
    degradation when Claude Code integration is not configured.
    """

    @property
    def is_available(self) -> bool:
        return False

    async def initialize(self) -> None:
        pass

    def get_task(self, task_id: str) -> None:
        return None

    def list_tasks(
        self,
        status: ClaudeCodeTaskStatus | None = None,
        session_id: str | None = None,
    ) -> list:
        return []

    def get_ready_tasks(self) -> list:
        return []

    def get_current_task(self) -> None:
        return None

    def get_sessions(self) -> list:
        return []

    async def link_memory_to_task(self, *args, **kwargs) -> None:
        pass

    async def link_memories_to_task(self, *args, **kwargs) -> None:
        pass

    async def get_task_memories(self, task_id: str) -> list:
        return []

    async def on_task_completed(self, task_id: str) -> int:
        return 0

    async def sync(self) -> TaskSyncResult:
        result = TaskSyncResult(source=TaskSource.CLAUDE_CODE)
        result.warnings.append("Claude Code integration not available")
        return result

    async def get_unified_context(self, *args, **kwargs) -> None:
        return None

    async def get_context_for_injection(self, *args, **kwargs) -> str:
        return ""

    async def get_stats(self) -> dict:
        return {"claude_code_available": False}

    def refresh(self) -> int:
        return 0


def create_claude_code_adapter(
    engine: MemoryEngine,
    todos_dir: Path | str | None = None,
    **kwargs,
) -> ClaudeCodeAdapter | NullClaudeCodeAdapter:
    """Factory function to create the appropriate adapter.

    Returns a NullClaudeCodeAdapter if Claude Code is not available.

    Args:
        engine: The Memory Layer engine.
        todos_dir: Explicit path to todos directory.
        **kwargs: Additional arguments for ClaudeCodeAdapter.

    Returns:
        ClaudeCodeAdapter if available, NullClaudeCodeAdapter otherwise.
    """
    adapter = ClaudeCodeAdapter(engine, todos_dir, **kwargs)
    if adapter.is_available:
        return adapter
    return NullClaudeCodeAdapter()

"""Unified Beads adapter combining all task integration components.

This module provides a single interface for all Beads task tracker
integration functionality, combining:
- Task parsing from .beads/ directory
- Task-memory linking
- Automatic outcome capture
- Context generation

Example:
    >>> from memory_layer.tasks import BeadsAdapter
    >>> adapter = BeadsAdapter(engine)
    >>> await adapter.initialize()
    >>>
    >>> # Sync outcomes for completed tasks
    >>> result = await adapter.sync()
    >>> print(f"Recorded {result.outcomes_recorded} outcomes")
    >>>
    >>> # Get context for current task
    >>> context = await adapter.get_unified_context()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from memory_layer.core.models import Memory
from memory_layer.tasks.linking import TaskMemoryLinker
from memory_layer.tasks.models import (
    BeadsSyncResult,
    BeadsTask,
    BeadsTaskStatus,
    TaskContext,
)
from memory_layer.tasks.outcomes import OutcomeCapture
from memory_layer.tasks.parser import BeadsParser

if TYPE_CHECKING:
    from memory_layer.core.engine import MemoryEngine

logger = logging.getLogger(__name__)


class BeadsAdapter:
    """Unified adapter for Beads task tracker integration.

    Combines:
    - BeadsParser: Read tasks from .beads/ directory
    - TaskMemoryLinker: Track which memories are used per task
    - OutcomeCapture: Auto-record outcomes when tasks complete

    This is the main entry point for all Beads integration functionality.
    """

    def __init__(
        self,
        engine: MemoryEngine,
        beads_dir: Path | str | None = None,
        auto_outcome_enabled: bool = True,
        outcome_on_cancel: bool = False,
    ) -> None:
        """Initialize the Beads adapter.

        Args:
            engine: The Memory Layer engine.
            beads_dir: Explicit path to .beads/ directory (auto-discovers if None).
            auto_outcome_enabled: Whether to auto-record outcomes on task completion.
            outcome_on_cancel: Whether to record "failed" when tasks are cancelled.
        """
        self._engine = engine
        self._beads_dir = Path(beads_dir) if beads_dir else None

        # Initialize components
        self._parser = BeadsParser(self._beads_dir)
        self._linker: TaskMemoryLinker | None = None
        self._outcome_capture: OutcomeCapture | None = None

        # Configuration
        self.auto_outcome_enabled = auto_outcome_enabled
        self.outcome_on_cancel = outcome_on_cancel

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components.

        Creates the linker table and sets up outcome capture.
        Must be called before using the adapter.
        """
        if self._initialized:
            return

        # Get db_path from engine's storage
        db_path = self._engine._storage.db_path

        # Initialize linker
        self._linker = TaskMemoryLinker(db_path)
        await self._linker.initialize()

        # Initialize outcome capture
        self._outcome_capture = OutcomeCapture(
            engine=self._engine,
            linker=self._linker,
            parser=self._parser,
            auto_outcome_enabled=self.auto_outcome_enabled,
            outcome_on_cancel=self.outcome_on_cancel,
        )

        self._initialized = True
        logger.debug("BeadsAdapter initialized")

    def _ensure_initialized(self) -> None:
        """Ensure the adapter is initialized."""
        if not self._initialized:
            raise RuntimeError("BeadsAdapter not initialized. Call initialize() first.")

    @property
    def is_available(self) -> bool:
        """Check if Beads is available (directory exists)."""
        return self._parser.is_available()

    # =========================================================================
    # Task Operations
    # =========================================================================

    def get_task(self, task_id: str) -> BeadsTask | None:
        """Get a task by ID.

        Args:
            task_id: The Beads task ID (e.g., "bd-a3f8").

        Returns:
            The task or None if not found.
        """
        return self._parser.get_task(task_id)

    def list_tasks(
        self,
        status: BeadsTaskStatus | None = None,
    ) -> list[BeadsTask]:
        """List all tasks, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            List of matching tasks.
        """
        return self._parser.list_tasks(status=status)

    def get_ready_tasks(self) -> list[BeadsTask]:
        """Get tasks that are ready to work on.

        Returns:
            List of tasks with no blockers and pending status.
        """
        return self._parser.get_ready_tasks()

    def get_current_task(self) -> BeadsTask | None:
        """Get the currently active task (in_progress).

        Returns:
            The in-progress task, or None if no task is active.
        """
        in_progress = self._parser.get_in_progress_tasks()
        return in_progress[0] if in_progress else None

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
            task_id: The Beads task ID.
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
            task_id: The Beads task ID.
            memory_ids: List of Memory Layer memory IDs.
            context: Optional context about how memories were used.
        """
        self._ensure_initialized()
        await self._linker.link_many(task_id, memory_ids, context)

    async def get_task_memories(self, task_id: str) -> list[Memory]:
        """Get all memories linked to a task.

        Args:
            task_id: The Beads task ID.

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
                # Memory might not exist anymore
                logger.debug(f"Memory {link.memory_id} not found, skipping")
        return memories

    async def auto_link_search_results(
        self,
        task_id: str,
        memory_ids: list[str],
    ) -> None:
        """Automatically link search results to the current task.

        Called after a search to track which memories were surfaced.

        Args:
            task_id: The Beads task ID.
            memory_ids: Memory IDs from search results.
        """
        self._ensure_initialized()
        await self._linker.link_many(task_id, memory_ids, context="search_result")

    # =========================================================================
    # Outcome Capture
    # =========================================================================

    async def on_task_done(self, task_id: str) -> int:
        """Handle a task being marked as done.

        Records "worked" outcome for all linked memories.

        Args:
            task_id: The Beads task ID.

        Returns:
            Number of memories that had outcomes recorded.
        """
        self._ensure_initialized()
        return await self._outcome_capture.on_task_completed(task_id)

    async def on_task_cancelled(self, task_id: str) -> int:
        """Handle a task being cancelled.

        Records "failed" outcome for linked memories (if enabled).

        Args:
            task_id: The Beads task ID.

        Returns:
            Number of memories that had outcomes recorded.
        """
        self._ensure_initialized()
        return await self._outcome_capture.on_task_failed(task_id)

    async def on_task_blocked(self, task_id: str) -> int:
        """Handle a task being blocked.

        Records "partial" outcome for linked memories.

        Args:
            task_id: The Beads task ID.

        Returns:
            Number of memories that had outcomes recorded.
        """
        self._ensure_initialized()
        return await self._outcome_capture.on_task_blocked(task_id)

    async def sync(self) -> BeadsSyncResult:
        """Sync outcomes for all completed tasks.

        Scans completed tasks and records outcomes for any that
        have unresolved memory links.

        Returns:
            BeadsSyncResult with sync statistics.
        """
        self._ensure_initialized()
        return await self._outcome_capture.sync_completed_tasks()

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
        if not memories and task.title:
            search_query = f"{task.title} {task.description[:100] if task.description else ''}"
            results = await self._engine.search(
                search_query,
                limit=max_memories,
                min_score=0.5,  # Only include memories with >50% relevance
                track_usage=False,  # Don't track this as usage
            )
            # Filter to only highly relevant results
            memories = [r.memory for r in results if r.score >= 0.5]

        # Limit memories
        memories = memories[:max_memories]

        # Create context
        context = TaskContext(
            task=task,
            memories=memories,
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
        """Get statistics about Beads integration.

        Returns:
            Dict with statistics.
        """
        self._ensure_initialized()

        # Task stats
        all_tasks = self.list_tasks()
        task_stats = {
            "total_tasks": len(all_tasks),
            "by_status": {},
        }
        for task in all_tasks:
            status = task.status.value
            task_stats["by_status"][status] = task_stats["by_status"].get(status, 0) + 1

        # Link stats
        link_stats = await self._linker.get_stats()

        return {
            "beads_available": self.is_available,
            "beads_dir": str(self._parser.beads_dir) if self._parser.beads_dir else None,
            "tasks": task_stats,
            "links": link_stats,
            "auto_outcome_enabled": self.auto_outcome_enabled,
            "outcome_on_cancel": self.outcome_on_cancel,
        }

    def refresh(self) -> int:
        """Refresh task cache from disk.

        Returns:
            Number of tasks loaded.
        """
        return self._parser.refresh()


class NullBeadsAdapter:
    """Null adapter for when Beads is not available.

    Provides the same interface but does nothing, allowing graceful
    degradation when Beads integration is not configured.
    """

    @property
    def is_available(self) -> bool:
        return False

    async def initialize(self) -> None:
        pass

    def get_task(self, task_id: str) -> None:
        return None

    def list_tasks(self, status: BeadsTaskStatus | None = None) -> list:
        return []

    def get_ready_tasks(self) -> list:
        return []

    def get_current_task(self) -> None:
        return None

    async def link_memory_to_task(self, *args, **kwargs) -> None:
        pass

    async def link_memories_to_task(self, *args, **kwargs) -> None:
        pass

    async def get_task_memories(self, task_id: str) -> list:
        return []

    async def on_task_done(self, task_id: str) -> int:
        return 0

    async def on_task_cancelled(self, task_id: str) -> int:
        return 0

    async def on_task_blocked(self, task_id: str) -> int:
        return 0

    async def sync(self) -> BeadsSyncResult:
        result = BeadsSyncResult()
        result.warnings.append("Beads integration not available")
        return result

    async def get_unified_context(self, *args, **kwargs) -> None:
        return None

    async def get_context_for_injection(self, *args, **kwargs) -> str:
        return ""

    async def get_stats(self) -> dict:
        return {"beads_available": False}

    def refresh(self) -> int:
        return 0


def create_adapter(
    engine: MemoryEngine,
    beads_dir: Path | str | None = None,
    **kwargs,
) -> BeadsAdapter | NullBeadsAdapter:
    """Factory function to create the appropriate adapter.

    Returns a NullBeadsAdapter if Beads is not available.

    Args:
        engine: The Memory Layer engine.
        beads_dir: Explicit path to .beads/ directory.
        **kwargs: Additional arguments for BeadsAdapter.

    Returns:
        BeadsAdapter if Beads is available, NullBeadsAdapter otherwise.
    """
    adapter = BeadsAdapter(engine, beads_dir, **kwargs)
    if adapter.is_available:
        return adapter
    return NullBeadsAdapter()

"""Automatic outcome capture for Beads task integration.

This module is THE key feature of Beads integration. It automatically
records outcomes for memories when tasks complete, closing the feedback
loop without requiring manual /outcome calls.

Flow:
    Task "bd-a3f8" marked "done"
        → Find all memories linked to this task
        → Record "worked" outcome for each memory
        → Score boost (+0.2) applied automatically
        → Bad advice naturally sinks, good advice rises
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memory_layer.core.models import Outcome
from memory_layer.tasks.linking import TaskMemoryLinker
from memory_layer.tasks.models import (
    BeadsSyncResult,
    BeadsTask,
    BeadsTaskStatus,
    CANCELLED_TASK_PENALTY,
    TASK_STATUS_TO_OUTCOME,
)
from memory_layer.tasks.parser import BeadsParser

if TYPE_CHECKING:
    from memory_layer.core.engine import MemoryEngine

logger = logging.getLogger(__name__)


class OutcomeCapture:
    """Automatically captures outcomes when Beads tasks complete.

    This class monitors task status and records outcomes for linked
    memories when tasks reach terminal states (done, cancelled).

    Example:
        >>> capture = OutcomeCapture(engine, linker, parser)
        >>> result = await capture.on_task_completed("bd-a3f8")
        >>> print(f"Updated {result} memories")
    """

    def __init__(
        self,
        engine: MemoryEngine,
        linker: TaskMemoryLinker,
        parser: BeadsParser | None = None,
        auto_outcome_enabled: bool = True,
        outcome_on_cancel: bool = False,
        min_confidence_for_outcome: float = 0.0,
    ) -> None:
        """Initialize the outcome capture system.

        Args:
            engine: The Memory Layer engine for recording outcomes.
            linker: The task-memory linker for finding linked memories.
            parser: Optional Beads parser for task status lookups.
            auto_outcome_enabled: Whether to automatically record outcomes.
            outcome_on_cancel: Whether to record "failed" on cancelled tasks.
            min_confidence_for_outcome: Minimum memory confidence to auto-record.
        """
        self._engine = engine
        self._linker = linker
        self._parser = parser
        self.auto_outcome_enabled = auto_outcome_enabled
        self.outcome_on_cancel = outcome_on_cancel
        self.min_confidence_for_outcome = min_confidence_for_outcome

    async def on_task_completed(self, task_id: str) -> int:
        """Handle a task being marked as completed (done).

        Records "worked" outcome for all unresolved linked memories.

        Args:
            task_id: The Beads task ID that was completed.

        Returns:
            Number of memories that had outcomes recorded.
        """
        if not self.auto_outcome_enabled:
            logger.debug(f"Auto-outcome disabled, skipping task {task_id}")
            return 0

        return await self._record_outcome_for_task(task_id, Outcome.WORKED)

    async def on_task_failed(self, task_id: str) -> int:
        """Handle a task being marked as failed/cancelled.

        Records "failed" outcome for all unresolved linked memories
        (if outcome_on_cancel is enabled).

        Args:
            task_id: The Beads task ID that failed.

        Returns:
            Number of memories that had outcomes recorded.
        """
        if not self.auto_outcome_enabled:
            logger.debug(f"Auto-outcome disabled, skipping task {task_id}")
            return 0

        if not self.outcome_on_cancel:
            logger.debug(f"Outcome on cancel disabled, skipping task {task_id}")
            return 0

        return await self._record_outcome_for_task(task_id, Outcome.FAILED)

    async def on_task_blocked(self, task_id: str) -> int:
        """Handle a task being marked as blocked.

        Records "partial" outcome for all unresolved linked memories.
        This indicates the advice was on the right track but couldn't
        fully solve the problem.

        Args:
            task_id: The Beads task ID that was blocked.

        Returns:
            Number of memories that had outcomes recorded.
        """
        if not self.auto_outcome_enabled:
            logger.debug(f"Auto-outcome disabled, skipping task {task_id}")
            return 0

        return await self._record_outcome_for_task(task_id, Outcome.PARTIAL)

    async def _record_outcome_for_task(
        self,
        task_id: str,
        outcome: Outcome,
    ) -> int:
        """Record outcome for all unresolved memories linked to a task.

        Args:
            task_id: The Beads task ID.
            outcome: The outcome to record.

        Returns:
            Number of memories updated.
        """
        # Get unresolved links (memories without outcomes yet)
        links = await self._linker.get_unresolved_links(task_id)

        if not links:
            logger.debug(f"No unresolved links for task {task_id}")
            return 0

        # Filter by minimum confidence if specified
        memory_ids = []
        for link in links:
            if self.min_confidence_for_outcome > 0:
                try:
                    memory = await self._engine.get(link.memory_id)
                    if memory.confidence < self.min_confidence_for_outcome:
                        logger.debug(
                            f"Skipping memory {link.memory_id} "
                            f"(confidence {memory.confidence} < {self.min_confidence_for_outcome})"
                        )
                        continue
                except Exception:
                    # Memory might not exist anymore, skip it
                    continue
            memory_ids.append(link.memory_id)

        if not memory_ids:
            logger.debug(f"No memories passed confidence filter for task {task_id}")
            return 0

        # Record outcomes in the engine
        try:
            await self._engine.record_outcome(memory_ids, outcome)
            logger.info(
                f"Recorded {outcome.value} for {len(memory_ids)} memories "
                f"linked to task {task_id}"
            )
        except Exception as e:
            logger.error(f"Failed to record outcomes for task {task_id}: {e}")
            return 0

        # Mark links as having outcomes recorded
        await self._linker.record_task_outcome(task_id, outcome.value)

        return len(memory_ids)

    async def process_task_status_change(
        self,
        task_id: str,
        old_status: BeadsTaskStatus,
        new_status: BeadsTaskStatus,
    ) -> int:
        """Process a task status change and record outcomes if appropriate.

        Args:
            task_id: The Beads task ID.
            old_status: Previous task status.
            new_status: New task status.

        Returns:
            Number of memories that had outcomes recorded.
        """
        # Only process transitions to terminal states
        if new_status == BeadsTaskStatus.DONE:
            return await self.on_task_completed(task_id)
        elif new_status == BeadsTaskStatus.CANCELLED:
            return await self.on_task_failed(task_id)
        elif new_status == BeadsTaskStatus.BLOCKED:
            return await self.on_task_blocked(task_id)

        return 0

    async def sync_completed_tasks(self) -> BeadsSyncResult:
        """Sync outcomes for all completed tasks that have unresolved links.

        Scans all completed tasks in Beads and records outcomes for any
        that have memories without recorded outcomes.

        Returns:
            BeadsSyncResult with sync statistics.
        """
        result = BeadsSyncResult()

        if not self._parser:
            result.errors.append("No parser available for task sync")
            return result

        if not self._parser.is_available():
            result.warnings.append("Beads directory not found")
            return result

        # Get all completed tasks
        completed_tasks = self._parser.get_completed_tasks()
        result.tasks_found = len(completed_tasks)

        for task in completed_tasks:
            try:
                # Check if task has unresolved links
                unresolved = await self._linker.get_unresolved_links(task.id)
                if not unresolved:
                    continue

                # Record outcome based on task status
                if task.status == BeadsTaskStatus.DONE:
                    count = await self.on_task_completed(task.id)
                elif task.status == BeadsTaskStatus.CANCELLED:
                    count = await self.on_task_failed(task.id)
                else:
                    count = 0

                result.outcomes_recorded += count
                result.tasks_synced += 1

            except Exception as e:
                result.errors.append(f"Error processing task {task.id}: {e}")

        return result

    async def check_and_record(self, task_id: str) -> int:
        """Check task status and record outcome if completed.

        Convenience method that looks up the task status and calls
        the appropriate outcome handler.

        Args:
            task_id: The Beads task ID to check.

        Returns:
            Number of memories that had outcomes recorded.
        """
        if not self._parser:
            logger.warning("No parser available, cannot check task status")
            return 0

        task = self._parser.get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found")
            return 0

        if task.status == BeadsTaskStatus.DONE:
            return await self.on_task_completed(task_id)
        elif task.status == BeadsTaskStatus.CANCELLED:
            return await self.on_task_failed(task_id)
        elif task.status == BeadsTaskStatus.BLOCKED:
            return await self.on_task_blocked(task_id)

        return 0


async def auto_capture_outcome(
    engine: MemoryEngine,
    linker: TaskMemoryLinker,
    task_id: str,
    status: BeadsTaskStatus,
) -> int:
    """Convenience function to capture outcome for a task.

    Args:
        engine: The Memory Layer engine.
        linker: The task-memory linker.
        task_id: The Beads task ID.
        status: The task's new status.

    Returns:
        Number of memories that had outcomes recorded.
    """
    capture = OutcomeCapture(engine, linker)

    if status == BeadsTaskStatus.DONE:
        return await capture.on_task_completed(task_id)
    elif status == BeadsTaskStatus.CANCELLED:
        return await capture.on_task_failed(task_id)
    elif status == BeadsTaskStatus.BLOCKED:
        return await capture.on_task_blocked(task_id)

    return 0

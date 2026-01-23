"""Task-Memory linking for tracking which memories are used per task.

This module manages the relationship between Beads tasks and Memory Layer
memories, enabling automatic outcome capture when tasks complete.

Schema:
    task_memory_links - Links memories to tasks they were used for
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from memory_layer.tasks.models import TaskMemoryLink

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Schema for task-memory links table
LINKS_SCHEMA_SQL = """
-- Task-memory links table
CREATE TABLE IF NOT EXISTS task_memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    used_at TEXT NOT NULL,
    outcome TEXT,
    context TEXT,
    UNIQUE(task_id, memory_id)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_links_task_id ON task_memory_links(task_id);
CREATE INDEX IF NOT EXISTS idx_links_memory_id ON task_memory_links(memory_id);
CREATE INDEX IF NOT EXISTS idx_links_outcome ON task_memory_links(outcome);
CREATE INDEX IF NOT EXISTS idx_links_task_outcome ON task_memory_links(task_id, outcome);
"""


class TaskMemoryLinker:
    """Manages links between tasks and memories.

    This class tracks which memories are used for which tasks,
    enabling automatic outcome recording when tasks complete.

    Example:
        >>> linker = TaskMemoryLinker(db_path)
        >>> await linker.initialize()
        >>> await linker.link("bd-a3f8", "mem-123")
        >>> links = await linker.get_memories_for_task("bd-a3f8")
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the linker.

        Args:
            db_path: Path to SQLite database (same as MemoryStorage).
        """
        self.db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the links table in the database."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.executescript(LINKS_SCHEMA_SQL)
            await conn.commit()

        self._initialized = True
        logger.debug("Task-memory links table initialized")

    async def link(
        self,
        task_id: str,
        memory_id: str,
        context: str | None = None,
    ) -> TaskMemoryLink:
        """Link a memory to a task.

        If the link already exists, updates the used_at timestamp.

        Args:
            task_id: The Beads task ID.
            memory_id: The Memory Layer memory ID.
            context: Optional context about how the memory was used.

        Returns:
            The created or updated TaskMemoryLink.
        """
        now = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(self.db_path) as conn:
            # Use INSERT OR REPLACE to handle duplicates
            await conn.execute(
                """
                INSERT INTO task_memory_links (task_id, memory_id, used_at, context)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id, memory_id) DO UPDATE SET
                    used_at = excluded.used_at,
                    context = COALESCE(excluded.context, context)
                """,
                (task_id, memory_id, now, context),
            )
            await conn.commit()

        logger.debug(f"Linked memory {memory_id} to task {task_id}")
        return TaskMemoryLink(
            task_id=task_id,
            memory_id=memory_id,
            used_at=datetime.fromisoformat(now),
            context=context,
        )

    async def link_many(
        self,
        task_id: str,
        memory_ids: Sequence[str],
        context: str | None = None,
    ) -> list[TaskMemoryLink]:
        """Link multiple memories to a task.

        Args:
            task_id: The Beads task ID.
            memory_ids: List of Memory Layer memory IDs.
            context: Optional context about how the memories were used.

        Returns:
            List of created TaskMemoryLink objects.
        """
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        links = []

        async with aiosqlite.connect(self.db_path) as conn:
            for memory_id in memory_ids:
                await conn.execute(
                    """
                    INSERT INTO task_memory_links (task_id, memory_id, used_at, context)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(task_id, memory_id) DO UPDATE SET
                        used_at = excluded.used_at
                    """,
                    (task_id, memory_id, now_iso, context),
                )
                links.append(
                    TaskMemoryLink(
                        task_id=task_id,
                        memory_id=memory_id,
                        used_at=now,
                        context=context,
                    )
                )
            await conn.commit()

        logger.debug(f"Linked {len(memory_ids)} memories to task {task_id}")
        return links

    async def unlink(self, task_id: str, memory_id: str) -> bool:
        """Remove a link between a task and memory.

        Args:
            task_id: The Beads task ID.
            memory_id: The Memory Layer memory ID.

        Returns:
            True if a link was removed, False if it didn't exist.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "DELETE FROM task_memory_links WHERE task_id = ? AND memory_id = ?",
                (task_id, memory_id),
            )
            await conn.commit()
            removed = cursor.rowcount > 0

        if removed:
            logger.debug(f"Unlinked memory {memory_id} from task {task_id}")
        return removed

    async def get_memories_for_task(
        self,
        task_id: str,
        include_with_outcome: bool = True,
    ) -> list[TaskMemoryLink]:
        """Get all memory links for a task.

        Args:
            task_id: The Beads task ID.
            include_with_outcome: If False, only returns links without outcomes.

        Returns:
            List of TaskMemoryLink objects.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            if include_with_outcome:
                cursor = await conn.execute(
                    "SELECT * FROM task_memory_links WHERE task_id = ?",
                    (task_id,),
                )
            else:
                cursor = await conn.execute(
                    "SELECT * FROM task_memory_links WHERE task_id = ? AND outcome IS NULL",
                    (task_id,),
                )
            rows = await cursor.fetchall()

        return [self._row_to_link(row) for row in rows]

    async def get_unresolved_links(self, task_id: str) -> list[TaskMemoryLink]:
        """Get links for a task that don't have outcomes recorded yet.

        This is used to find which memories need outcome feedback
        when a task completes.

        Args:
            task_id: The Beads task ID.

        Returns:
            List of TaskMemoryLink objects without outcomes.
        """
        return await self.get_memories_for_task(task_id, include_with_outcome=False)

    async def get_tasks_for_memory(self, memory_id: str) -> list[TaskMemoryLink]:
        """Get all task links for a memory.

        Args:
            memory_id: The Memory Layer memory ID.

        Returns:
            List of TaskMemoryLink objects.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM task_memory_links WHERE memory_id = ?",
                (memory_id,),
            )
            rows = await cursor.fetchall()

        return [self._row_to_link(row) for row in rows]

    async def record_task_outcome(
        self,
        task_id: str,
        outcome: str,
    ) -> int:
        """Record outcome for all unresolved links of a task.

        This is called when a task completes to mark all linked
        memories with the appropriate outcome.

        Args:
            task_id: The Beads task ID.
            outcome: The outcome string ("worked", "failed", "partial").

        Returns:
            Number of links updated.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                """
                UPDATE task_memory_links
                SET outcome = ?
                WHERE task_id = ? AND outcome IS NULL
                """,
                (outcome, task_id),
            )
            await conn.commit()
            count = cursor.rowcount

        logger.debug(f"Recorded outcome '{outcome}' for {count} links of task {task_id}")
        return count

    async def get_link(self, task_id: str, memory_id: str) -> TaskMemoryLink | None:
        """Get a specific task-memory link.

        Args:
            task_id: The Beads task ID.
            memory_id: The Memory Layer memory ID.

        Returns:
            The link or None if not found.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM task_memory_links WHERE task_id = ? AND memory_id = ?",
                (task_id, memory_id),
            )
            row = await cursor.fetchone()

        return self._row_to_link(row) if row else None

    async def count_links(self, task_id: str | None = None) -> int:
        """Count total links, optionally for a specific task.

        Args:
            task_id: Optional task ID to filter by.

        Returns:
            Number of links.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            if task_id:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM task_memory_links WHERE task_id = ?",
                    (task_id,),
                )
            else:
                cursor = await conn.execute("SELECT COUNT(*) FROM task_memory_links")
            row = await cursor.fetchone()

        return row[0] if row else 0

    async def get_stats(self) -> dict:
        """Get statistics about task-memory links.

        Returns:
            Dict with link statistics.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            # Total links
            cursor = await conn.execute("SELECT COUNT(*) FROM task_memory_links")
            total = (await cursor.fetchone())[0]

            # Links by outcome
            cursor = await conn.execute(
                """
                SELECT outcome, COUNT(*) as count
                FROM task_memory_links
                GROUP BY outcome
                """
            )
            by_outcome = {row[0] or "unresolved": row[1] for row in await cursor.fetchall()}

            # Unique tasks
            cursor = await conn.execute(
                "SELECT COUNT(DISTINCT task_id) FROM task_memory_links"
            )
            unique_tasks = (await cursor.fetchone())[0]

            # Unique memories
            cursor = await conn.execute(
                "SELECT COUNT(DISTINCT memory_id) FROM task_memory_links"
            )
            unique_memories = (await cursor.fetchone())[0]

        return {
            "total_links": total,
            "by_outcome": by_outcome,
            "unique_tasks": unique_tasks,
            "unique_memories": unique_memories,
        }

    async def clear_task_links(self, task_id: str) -> int:
        """Remove all links for a task.

        Args:
            task_id: The Beads task ID.

        Returns:
            Number of links removed.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "DELETE FROM task_memory_links WHERE task_id = ?",
                (task_id,),
            )
            await conn.commit()
            count = cursor.rowcount

        logger.debug(f"Cleared {count} links for task {task_id}")
        return count

    def _row_to_link(self, row: aiosqlite.Row) -> TaskMemoryLink:
        """Convert a database row to a TaskMemoryLink."""
        used_at = row["used_at"]
        if isinstance(used_at, str):
            used_at = datetime.fromisoformat(used_at.replace("Z", "+00:00"))

        return TaskMemoryLink(
            task_id=row["task_id"],
            memory_id=row["memory_id"],
            used_at=used_at,
            outcome=row["outcome"],
            context=row["context"],
        )

"""Beads file parser for reading task data from .beads/ directory.

This module handles:
- Discovery of .beads/ directory
- Parsing JSONL task files
- Building task index with parent-child relationships
- Graceful handling of malformed data
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from memory_layer.tasks.models import BeadsTask, BeadsTaskStatus

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class BeadsDirectoryNotFoundError(Exception):
    """Raised when .beads/ directory cannot be found."""

    pass


class BeadsParser:
    """Parses Beads task data from .beads/ directory.

    Supports:
    - JSONL files (one JSON object per line)
    - Directory discovery (walks up from current dir)
    - Environment variable override ($BEADS_DIR)
    - Caching of parsed tasks

    Example:
        >>> parser = BeadsParser()
        >>> tasks = parser.list_tasks()
        >>> ready = parser.get_ready_tasks()
    """

    def __init__(self, beads_dir: Path | str | None = None):
        """Initialize the parser.

        Args:
            beads_dir: Explicit path to .beads/ directory.
                       If None, will auto-discover.
        """
        self._beads_dir: Path | None = None
        self._explicit_dir = Path(beads_dir) if beads_dir else None
        self._task_cache: dict[str, BeadsTask] = {}
        self._cache_valid = False

    @property
    def beads_dir(self) -> Path | None:
        """Get the .beads/ directory path, discovering if needed."""
        if self._beads_dir is None:
            self._beads_dir = self._discover_beads_dir()
        return self._beads_dir

    def _discover_beads_dir(self) -> Path | None:
        """Discover the .beads/ directory.

        Search order:
        1. Explicit path provided to constructor
        2. $BEADS_DIR environment variable
        3. Walk up from current directory

        Returns:
            Path to .beads/ directory or None if not found.
        """
        # 1. Explicit path
        if self._explicit_dir:
            if self._explicit_dir.is_dir():
                logger.debug(f"Using explicit beads dir: {self._explicit_dir}")
                return self._explicit_dir
            logger.warning(f"Explicit beads dir not found: {self._explicit_dir}")
            return None

        # 2. Environment variable
        env_dir = os.environ.get("BEADS_DIR")
        if env_dir:
            env_path = Path(env_dir)
            if env_path.is_dir():
                logger.debug(f"Using $BEADS_DIR: {env_path}")
                return env_path
            logger.warning(f"$BEADS_DIR not found: {env_path}")

        # 3. Walk up from current directory
        current = Path.cwd()
        for parent in [current, *current.parents]:
            beads_path = parent / ".beads"
            if beads_path.is_dir():
                logger.debug(f"Found .beads/ at: {beads_path}")
                return beads_path

        logger.debug("No .beads/ directory found")
        return None

    def is_available(self) -> bool:
        """Check if Beads is available (directory exists)."""
        return self.beads_dir is not None

    def invalidate_cache(self) -> None:
        """Invalidate the task cache, forcing a re-parse on next access."""
        self._task_cache.clear()
        self._cache_valid = False

    def _ensure_cache(self) -> None:
        """Ensure the task cache is populated."""
        if self._cache_valid:
            return

        self._task_cache.clear()

        if not self.beads_dir:
            return

        # Parse all JSONL files in .beads/
        for task in self._parse_all_files():
            self._task_cache[task.id] = task

        self._cache_valid = True
        logger.debug(f"Cached {len(self._task_cache)} tasks")

    def _parse_all_files(self) -> Iterator[BeadsTask]:
        """Parse all JSONL files in the .beads/ directory.

        Yields:
            BeadsTask objects for each valid task entry.
        """
        if not self.beads_dir:
            return

        # Find all .jsonl files
        jsonl_files = list(self.beads_dir.glob("*.jsonl"))
        logger.debug(f"Found {len(jsonl_files)} JSONL files")

        for jsonl_file in jsonl_files:
            yield from self._parse_jsonl_file(jsonl_file)

    def _parse_jsonl_file(self, file_path: Path) -> Iterator[BeadsTask]:
        """Parse a single JSONL file.

        Args:
            file_path: Path to the JSONL file.

        Yields:
            BeadsTask objects for each valid line.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        task = BeadsTask.from_dict(data)
                        yield task
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Malformed JSON at {file_path}:{line_num}: {e}"
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning(
                            f"Invalid task data at {file_path}:{line_num}: {e}"
                        )
        except OSError as e:
            logger.error(f"Failed to read {file_path}: {e}")

    def get_task(self, task_id: str) -> BeadsTask | None:
        """Get a task by ID.

        Args:
            task_id: The task ID (e.g., "bd-a3f8").

        Returns:
            The task or None if not found.
        """
        self._ensure_cache()
        return self._task_cache.get(task_id)

    def list_tasks(
        self,
        status: BeadsTaskStatus | None = None,
        parent_id: str | None = None,
    ) -> list[BeadsTask]:
        """List all tasks, optionally filtered.

        Args:
            status: Filter by status (e.g., IN_PROGRESS).
            parent_id: Filter by parent task (for subtasks).

        Returns:
            List of matching tasks.
        """
        self._ensure_cache()

        tasks = list(self._task_cache.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        if parent_id is not None:
            tasks = [t for t in tasks if t.parent_id == parent_id]

        return tasks

    def get_ready_tasks(self) -> list[BeadsTask]:
        """Get tasks that are ready to work on.

        A task is ready if:
        - Status is PENDING
        - No unresolved dependencies

        Returns:
            List of ready tasks.
        """
        self._ensure_cache()

        ready = []
        for task in self._task_cache.values():
            if task.status != BeadsTaskStatus.PENDING:
                continue

            # Check if all dependencies are done
            deps_resolved = all(
                self._task_cache.get(dep_id, BeadsTask(id=dep_id, title="")).status
                == BeadsTaskStatus.DONE
                for dep_id in task.dependencies
            )

            if deps_resolved:
                ready.append(task)

        return ready

    def get_in_progress_tasks(self) -> list[BeadsTask]:
        """Get tasks currently being worked on.

        Returns:
            List of in-progress tasks.
        """
        return self.list_tasks(status=BeadsTaskStatus.IN_PROGRESS)

    def get_subtasks(self, parent_id: str) -> list[BeadsTask]:
        """Get subtasks for a parent task.

        Args:
            parent_id: The parent task ID.

        Returns:
            List of subtasks.
        """
        return self.list_tasks(parent_id=parent_id)

    def get_task_hierarchy(self, task_id: str) -> dict[str, BeadsTask | list]:
        """Get a task with its subtasks in a hierarchical structure.

        Args:
            task_id: The task ID.

        Returns:
            Dict with 'task' and 'subtasks' keys.
        """
        task = self.get_task(task_id)
        if not task:
            return {"task": None, "subtasks": []}

        subtasks = self.get_subtasks(task_id)
        return {"task": task, "subtasks": subtasks}

    def get_completed_tasks(self) -> list[BeadsTask]:
        """Get tasks that have been completed (done or cancelled).

        Returns:
            List of completed tasks.
        """
        self._ensure_cache()
        return [t for t in self._task_cache.values() if t.is_completed]

    def get_blocked_tasks(self) -> list[BeadsTask]:
        """Get tasks that are blocked.

        Returns:
            List of blocked tasks.
        """
        return self.list_tasks(status=BeadsTaskStatus.BLOCKED)

    def refresh(self) -> int:
        """Refresh the task cache from disk.

        Returns:
            Number of tasks loaded.
        """
        self.invalidate_cache()
        self._ensure_cache()
        return len(self._task_cache)

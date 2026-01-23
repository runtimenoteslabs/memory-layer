"""Claude Code todos parser for reading task data from ~/.claude/todos/.

This module handles:
- Discovery of ~/.claude/todos/ directory
- Parsing JSON array task files
- Building task index with session/agent relationships
- Graceful handling of malformed data
- Support for CLAUDE_CODE_TASK_LIST_ID environment variable
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from memory_layer.tasks.models import ClaudeCodeTask, ClaudeCodeTaskStatus

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class ClaudeCodeDirectoryNotFoundError(Exception):
    """Raised when ~/.claude/todos/ directory cannot be found."""

    pass


class ClaudeCodeParser:
    """Parses Claude Code task data from ~/.claude/todos/ directory.

    Supports:
    - JSON array files (each file is a session's task list)
    - CLAUDE_CODE_TASK_LIST_ID environment variable for filtering
    - Caching of parsed tasks

    File naming convention:
        {session-uuid}-agent-{agent-uuid}.json

    Example:
        >>> parser = ClaudeCodeParser()
        >>> tasks = parser.list_tasks()
        >>> ready = parser.get_ready_tasks()
    """

    # Default location for Claude Code todos
    DEFAULT_TODOS_DIR = Path.home() / ".claude" / "todos"

    def __init__(
        self,
        todos_dir: Path | str | None = None,
        task_list_id: str | None = None,
    ):
        """Initialize the parser.

        Args:
            todos_dir: Explicit path to todos directory.
                       If None, uses ~/.claude/todos/.
            task_list_id: Filter to specific task list ID (session/agent).
                          If None, checks CLAUDE_CODE_TASK_LIST_ID env var.
        """
        self._todos_dir: Path | None = None
        self._explicit_dir = Path(todos_dir) if todos_dir else None
        self._task_list_id = task_list_id or os.environ.get("CLAUDE_CODE_TASK_LIST_ID")
        self._task_cache: dict[str, ClaudeCodeTask] = {}
        self._cache_valid = False

    @property
    def todos_dir(self) -> Path | None:
        """Get the todos directory path, discovering if needed."""
        if self._todos_dir is None:
            self._todos_dir = self._discover_todos_dir()
        return self._todos_dir

    def _discover_todos_dir(self) -> Path | None:
        """Discover the todos directory.

        Search order:
        1. Explicit path provided to constructor
        2. $CLAUDE_CODE_TODOS_DIR environment variable
        3. Default ~/.claude/todos/

        Returns:
            Path to todos directory or None if not found.
        """
        # 1. Explicit path
        if self._explicit_dir:
            if self._explicit_dir.is_dir():
                logger.debug(f"Using explicit todos dir: {self._explicit_dir}")
                return self._explicit_dir
            logger.warning(f"Explicit todos dir not found: {self._explicit_dir}")
            return None

        # 2. Environment variable
        env_dir = os.environ.get("CLAUDE_CODE_TODOS_DIR")
        if env_dir:
            env_path = Path(env_dir)
            if env_path.is_dir():
                logger.debug(f"Using $CLAUDE_CODE_TODOS_DIR: {env_path}")
                return env_path
            logger.warning(f"$CLAUDE_CODE_TODOS_DIR not found: {env_path}")

        # 3. Default location
        default_path = self.DEFAULT_TODOS_DIR
        if default_path.is_dir():
            logger.debug(f"Using default todos dir: {default_path}")
            return default_path

        logger.debug("No Claude Code todos directory found")
        return None

    def is_available(self) -> bool:
        """Check if Claude Code todos are available (directory exists)."""
        return self.todos_dir is not None

    def invalidate_cache(self) -> None:
        """Invalidate the task cache, forcing a re-parse on next access."""
        self._task_cache.clear()
        self._cache_valid = False

    def _ensure_cache(self) -> None:
        """Ensure the task cache is populated."""
        if self._cache_valid:
            return

        self._task_cache.clear()

        if not self.todos_dir:
            return

        # Parse all JSON files in todos dir
        for task in self._parse_all_files():
            self._task_cache[task.id] = task

        self._cache_valid = True
        logger.debug(f"Cached {len(self._task_cache)} Claude Code tasks")

    def _parse_all_files(self) -> Iterator[ClaudeCodeTask]:
        """Parse all JSON files in the todos directory.

        Yields:
            ClaudeCodeTask objects for each valid task entry.
        """
        if not self.todos_dir:
            return

        # Find all .json files
        json_files = list(self.todos_dir.glob("*.json"))
        logger.debug(f"Found {len(json_files)} JSON files in todos dir")

        for json_file in json_files:
            # Filter by task list ID if specified
            if self._task_list_id and self._task_list_id not in json_file.name:
                continue

            yield from self._parse_json_file(json_file)

    def _parse_json_file(self, file_path: Path) -> Iterator[ClaudeCodeTask]:
        """Parse a single JSON file containing tasks.

        Args:
            file_path: Path to the JSON file.

        Yields:
            ClaudeCodeTask objects for each valid task.
        """
        try:
            # Extract session and agent IDs from filename
            # Format: {session-uuid}-agent-{agent-uuid}.json
            filename = file_path.stem  # Remove .json
            session_id = ""
            agent_id = ""

            if "-agent-" in filename:
                parts = filename.split("-agent-")
                session_id = parts[0]
                agent_id = parts[1] if len(parts) > 1 else ""
            else:
                session_id = filename

            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Handle empty files or empty arrays
            if not data:
                return

            # File should be a JSON array
            if not isinstance(data, list):
                logger.warning(f"Expected JSON array in {file_path}, got {type(data)}")
                return

            for index, task_data in enumerate(data):
                try:
                    if not isinstance(task_data, dict):
                        logger.warning(
                            f"Expected dict at index {index} in {file_path}"
                        )
                        continue

                    task = ClaudeCodeTask.from_dict(
                        task_data,
                        session_id=session_id,
                        agent_id=agent_id,
                        index=index,
                        file_path=str(file_path),
                    )
                    yield task

                except (KeyError, ValueError) as e:
                    logger.warning(
                        f"Invalid task data at {file_path}[{index}]: {e}"
                    )

        except json.JSONDecodeError as e:
            logger.warning(f"Malformed JSON in {file_path}: {e}")
        except OSError as e:
            logger.error(f"Failed to read {file_path}: {e}")

    def get_task(self, task_id: str) -> ClaudeCodeTask | None:
        """Get a task by ID.

        Args:
            task_id: The task ID (e.g., "cc-abc12345-0").

        Returns:
            The task or None if not found.
        """
        self._ensure_cache()
        return self._task_cache.get(task_id)

    def list_tasks(
        self,
        status: ClaudeCodeTaskStatus | None = None,
        session_id: str | None = None,
    ) -> list[ClaudeCodeTask]:
        """List all tasks, optionally filtered.

        Args:
            status: Filter by status (e.g., COMPLETED).
            session_id: Filter by session ID.

        Returns:
            List of matching tasks.
        """
        self._ensure_cache()

        tasks = list(self._task_cache.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        if session_id is not None:
            tasks = [t for t in tasks if t.session_id == session_id]

        return tasks

    def get_ready_tasks(self) -> list[ClaudeCodeTask]:
        """Get tasks that are ready to work on (pending status).

        Returns:
            List of ready tasks.
        """
        return self.list_tasks(status=ClaudeCodeTaskStatus.PENDING)

    def get_in_progress_tasks(self) -> list[ClaudeCodeTask]:
        """Get tasks currently being worked on.

        Returns:
            List of in-progress tasks.
        """
        return self.list_tasks(status=ClaudeCodeTaskStatus.IN_PROGRESS)

    def get_completed_tasks(self) -> list[ClaudeCodeTask]:
        """Get tasks that have been completed.

        Returns:
            List of completed tasks.
        """
        return self.list_tasks(status=ClaudeCodeTaskStatus.COMPLETED)

    def get_tasks_by_session(self, session_id: str) -> list[ClaudeCodeTask]:
        """Get all tasks for a specific session.

        Args:
            session_id: The Claude session ID.

        Returns:
            List of tasks for that session.
        """
        return self.list_tasks(session_id=session_id)

    def get_sessions(self) -> list[str]:
        """Get list of unique session IDs.

        Returns:
            List of session IDs that have tasks.
        """
        self._ensure_cache()
        sessions = set()
        for task in self._task_cache.values():
            if task.session_id:
                sessions.add(task.session_id)
        return sorted(sessions)

    def refresh(self) -> int:
        """Refresh the task cache from disk.

        Returns:
            Number of tasks loaded.
        """
        self.invalidate_cache()
        self._ensure_cache()
        return len(self._task_cache)

    def get_stats(self) -> dict:
        """Get statistics about Claude Code tasks.

        Returns:
            Dict with task statistics.
        """
        self._ensure_cache()

        tasks = list(self._task_cache.values())
        by_status = {}
        for task in tasks:
            status = task.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_tasks": len(tasks),
            "by_status": by_status,
            "sessions": len(self.get_sessions()),
            "todos_dir": str(self.todos_dir) if self.todos_dir else None,
        }

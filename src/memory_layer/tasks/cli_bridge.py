"""Beads CLI bridge for interacting with the bd command.

This module provides a wrapper around the Beads CLI (`bd` command),
offering an alternative to direct file parsing. Useful when:
- You want real-time task status (CLI may have fresher data)
- You need to leverage bd's dependency resolution
- File parsing is insufficient for complex operations

Falls back to file parsing if bd command is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memory_layer.tasks.models import BeadsTask, BeadsTaskStatus

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Cache duration for CLI results (seconds)
CLI_CACHE_DURATION = 30

# Command timeout (seconds)
CLI_TIMEOUT = 5


@dataclass
class CLIResult:
    """Result from a CLI command execution."""

    success: bool
    stdout: str
    stderr: str
    return_code: int


class BeadsCLI:
    """Wrapper for the Beads CLI (bd command).

    Provides methods to interact with Beads via its command-line interface.
    Results are cached for performance.

    Example:
        >>> cli = BeadsCLI()
        >>> if cli.is_available():
        ...     tasks = await cli.ready()
        ...     for task in tasks:
        ...         print(task.title)
    """

    def __init__(self, timeout: float = CLI_TIMEOUT) -> None:
        """Initialize the CLI bridge.

        Args:
            timeout: Command timeout in seconds.
        """
        self._timeout = timeout
        self._bd_path: str | None = None
        self._available: bool | None = None
        self._version: str | None = None

        # Cache
        self._cache: dict[str, tuple[datetime, any]] = {}
        self._cache_duration = CLI_CACHE_DURATION

    def is_available(self) -> bool:
        """Check if the bd command is available.

        Returns:
            True if bd command exists and is executable.
        """
        if self._available is None:
            self._bd_path = shutil.which("bd")
            self._available = self._bd_path is not None
            if self._available:
                logger.debug(f"Found bd at: {self._bd_path}")
            else:
                logger.debug("bd command not found in PATH")
        return self._available

    async def get_version(self) -> str | None:
        """Get the Beads version.

        Returns:
            Version string or None if unavailable.
        """
        if self._version is not None:
            return self._version

        if not self.is_available():
            return None

        result = await self._run_command(["bd", "--version"])
        if result.success:
            self._version = result.stdout.strip()
            return self._version
        return None

    async def ready(self) -> list[BeadsTask]:
        """Get tasks that are ready to work on.

        Calls `bd ready --json` to get unblocked tasks.

        Returns:
            List of ready BeadsTask objects.
        """
        return await self._cached_command("ready", ["bd", "ready", "--json"])

    async def show(self, task_id: str) -> BeadsTask | None:
        """Get details for a specific task.

        Calls `bd show <id> --json`.

        Args:
            task_id: The task ID to look up.

        Returns:
            BeadsTask or None if not found.
        """
        cache_key = f"show:{task_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if not self.is_available():
            return None

        result = await self._run_command(["bd", "show", task_id, "--json"])
        if not result.success:
            return None

        try:
            data = json.loads(result.stdout)
            task = self._parse_task(data)
            self._set_cached(cache_key, task)
            return task
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse bd show output: {e}")
            return None

    async def list_all(self) -> list[BeadsTask]:
        """Get all tasks.

        Calls `bd export --json` to get complete task list.

        Returns:
            List of all BeadsTask objects.
        """
        return await self._cached_command("list_all", ["bd", "export", "--json"])

    async def get_current(self) -> BeadsTask | None:
        """Get the currently active task.

        Attempts to detect the current task from:
        1. Tasks with in_progress status
        2. Git branch naming convention (if applicable)

        Returns:
            Current task or None.
        """
        if not self.is_available():
            return None

        # Get all in-progress tasks
        all_tasks = await self.list_all()
        in_progress = [
            t for t in all_tasks if t.status == BeadsTaskStatus.IN_PROGRESS
        ]

        if in_progress:
            return in_progress[0]

        return None

    async def _cached_command(
        self,
        cache_key: str,
        command: list[str],
    ) -> list[BeadsTask]:
        """Run a command with caching.

        Args:
            cache_key: Key for cache lookup.
            command: Command to execute.

        Returns:
            List of parsed tasks.
        """
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if not self.is_available():
            return []

        result = await self._run_command(command)
        if not result.success:
            logger.warning(f"Command failed: {' '.join(command)}")
            return []

        tasks = self._parse_tasks_output(result.stdout)
        self._set_cached(cache_key, tasks)
        return tasks

    async def _run_command(self, command: list[str]) -> CLIResult:
        """Run a shell command asynchronously.

        Args:
            command: Command and arguments to execute.

        Returns:
            CLIResult with output and status.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CLIResult(
                    success=False,
                    stdout="",
                    stderr=f"Command timed out after {self._timeout}s",
                    return_code=-1,
                )

            return CLIResult(
                success=process.returncode == 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                return_code=process.returncode or 0,
            )

        except FileNotFoundError:
            return CLIResult(
                success=False,
                stdout="",
                stderr="Command not found",
                return_code=-1,
            )
        except Exception as e:
            return CLIResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
            )

    def _parse_tasks_output(self, output: str) -> list[BeadsTask]:
        """Parse JSON output containing multiple tasks.

        Handles both JSON array and JSONL formats.

        Args:
            output: Raw command output.

        Returns:
            List of parsed tasks.
        """
        output = output.strip()
        if not output:
            return []

        tasks = []

        # Try JSON array first
        try:
            data = json.loads(output)
            if isinstance(data, list):
                for item in data:
                    try:
                        tasks.append(self._parse_task(item))
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse task: {e}")
                return tasks
            elif isinstance(data, dict):
                # Single task wrapped in object
                if "tasks" in data:
                    for item in data["tasks"]:
                        try:
                            tasks.append(self._parse_task(item))
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Failed to parse task: {e}")
                    return tasks
                else:
                    return [self._parse_task(data)]
        except json.JSONDecodeError:
            pass

        # Try JSONL (one JSON object per line)
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tasks.append(self._parse_task(data))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.debug(f"Failed to parse line: {e}")

        return tasks

    def _parse_task(self, data: dict) -> BeadsTask:
        """Parse a single task from JSON data.

        Args:
            data: Task data dictionary.

        Returns:
            BeadsTask object.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If data is invalid.
        """
        # Map common field names from Beads output
        task_id = data.get("id") or data.get("task_id") or data.get("ID")
        if not task_id:
            raise KeyError("Task ID not found")

        title = data.get("title") or data.get("name") or data.get("summary") or ""

        # Map status
        status_str = data.get("status", "pending").lower()
        status_map = {
            "pending": BeadsTaskStatus.PENDING,
            "todo": BeadsTaskStatus.PENDING,
            "ready": BeadsTaskStatus.PENDING,
            "in_progress": BeadsTaskStatus.IN_PROGRESS,
            "in-progress": BeadsTaskStatus.IN_PROGRESS,
            "active": BeadsTaskStatus.IN_PROGRESS,
            "working": BeadsTaskStatus.IN_PROGRESS,
            "done": BeadsTaskStatus.DONE,
            "completed": BeadsTaskStatus.DONE,
            "finished": BeadsTaskStatus.DONE,
            "blocked": BeadsTaskStatus.BLOCKED,
            "waiting": BeadsTaskStatus.BLOCKED,
            "cancelled": BeadsTaskStatus.CANCELLED,
            "canceled": BeadsTaskStatus.CANCELLED,
            "abandoned": BeadsTaskStatus.CANCELLED,
        }
        status = status_map.get(status_str, BeadsTaskStatus.PENDING)

        return BeadsTask(
            id=task_id,
            title=title,
            description=data.get("description", ""),
            status=status,
            parent_id=data.get("parent_id") or data.get("parent"),
            dependencies=data.get("dependencies", []) or data.get("blocked_by", []),
            tags=data.get("tags", []) or data.get("labels", []),
            metadata=data.get("metadata", {}),
        )

    def _get_cached(self, key: str) -> any:
        """Get a cached value if still valid.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if expired/missing.
        """
        if key not in self._cache:
            return None

        cached_time, value = self._cache[key]
        age = (datetime.now(UTC) - cached_time).total_seconds()

        if age > self._cache_duration:
            del self._cache[key]
            return None

        return value

    def _set_cached(self, key: str, value: any) -> None:
        """Set a cached value.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        self._cache[key] = (datetime.now(UTC), value)

    def invalidate_cache(self) -> None:
        """Clear all cached values."""
        self._cache.clear()


# Convenience function
def get_beads_cli() -> BeadsCLI:
    """Get a BeadsCLI instance.

    Returns:
        BeadsCLI instance.
    """
    return BeadsCLI()

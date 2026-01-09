"""Daemon for automatic Claude Code session monitoring.

.. deprecated:: 2.0.0
    The daemon-based architecture is deprecated in favor of the native
    Claude Code 2.1.1+ plugin system. Use native hooks (hooks/hooks.json)
    and Agent Skills (skills/*.md) instead.

    Migration guide:
    - SessionStart hook: Use hooks/hooks.json SessionStart
    - SessionEnd hook: Use hooks/hooks.json SessionEnd
    - PreCompact hook: Use hooks/hooks.json PreCompact
    - Auto-extraction: Use PreCompact hook with 'mem extract --auto'
    - CLAUDE.md updates: Use 'mem context --inject' in SessionStart hook

Watches Claude Code session files and automatically:
- Extracts learnings from completed sessions
- Updates CLAUDE.md with relevant context
- Triggers hooks at appropriate lifecycle points
"""

from __future__ import annotations

import warnings

# Emit deprecation warning when module is imported
warnings.warn(
    "The daemon module is deprecated as of v2.0.0. "
    "Use the native plugin system with hooks/hooks.json instead. "
    "See memory-layer/hooks/hooks.json for the v2 hook configuration.",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import json
import os
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from memory_layer.core.logging import get_logger

if TYPE_CHECKING:
    from memory_layer.core.engine import MemoryEngine
    from memory_layer.extraction.extractor import MemoryExtractor

logger = get_logger(__name__)

# Default paths
DEFAULT_CLAUDE_DIR = Path.home() / ".claude"
DEFAULT_SESSION_DIR = DEFAULT_CLAUDE_DIR / "session-memory"
DEFAULT_PID_FILE = Path(tempfile.gettempdir()) / "memory-layer-daemon.pid"


@dataclass
class DaemonConfig:
    """Configuration for the daemon."""

    # Watch directories
    claude_dir: Path = field(default_factory=lambda: DEFAULT_CLAUDE_DIR)
    """Claude Code configuration directory."""

    session_dir: Path = field(default_factory=lambda: DEFAULT_SESSION_DIR)
    """Directory containing session files."""

    # PID file
    pid_file: Path = field(default_factory=lambda: DEFAULT_PID_FILE)
    """Path to PID file for daemon management."""

    # Processing settings
    process_on_modify: bool = True
    """Process sessions on file modification."""

    process_on_create: bool = False
    """Process sessions on file creation."""

    debounce_seconds: float = 2.0
    """Debounce time to avoid duplicate processing."""

    # Session file patterns
    session_patterns: list[str] = field(default_factory=lambda: ["*.json", "*.jsonl"])
    """File patterns to watch for session data."""

    # Auto-extraction
    auto_extract: bool = True
    """Automatically extract memories from sessions."""

    # CLAUDE.md management
    update_claude_md: bool = True
    """Automatically update CLAUDE.md with context."""

    claude_md_path: Path | None = None
    """Path to CLAUDE.md (default: project root)."""

    # Privilege settings
    drop_privileges: bool = False
    """Drop to unprivileged user after binding."""

    target_uid: int | None = None
    """UID to drop to (if drop_privileges is True)."""

    target_gid: int | None = None
    """GID to drop to (if drop_privileges is True)."""

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        if isinstance(self.claude_dir, str):
            self.claude_dir = Path(self.claude_dir)
        if isinstance(self.session_dir, str):
            self.session_dir = Path(self.session_dir)
        if isinstance(self.pid_file, str):
            self.pid_file = Path(self.pid_file)


@dataclass
class SessionInfo:
    """Information about a Claude Code session."""

    session_id: str
    """Unique session identifier."""

    file_path: Path
    """Path to the session file."""

    project_path: Path | None
    """Path to the project directory."""

    started_at: datetime | None
    """When the session started."""

    ended_at: datetime | None
    """When the session ended."""

    is_active: bool
    """Whether the session is currently active."""

    transcript: str = ""
    """Session transcript content."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional session metadata."""


class DaemonError(Exception):
    """Base exception for daemon errors."""

    pass


class DaemonAlreadyRunningError(DaemonError):
    """Raised when daemon is already running."""

    pass


class SessionHandler(FileSystemEventHandler):
    """Handles file system events for Claude Code sessions."""

    def __init__(
        self,
        daemon: MemoryLayerDaemon,
        config: DaemonConfig,
    ) -> None:
        """Initialize the session handler.

        Args:
            daemon: Parent daemon instance.
            config: Daemon configuration.
        """
        super().__init__()
        self.daemon = daemon
        self.config = config
        self._last_processed: dict[str, float] = {}
        self._processing_lock = asyncio.Lock()

    def _should_process(self, path: str) -> bool:
        """Check if a path should be processed.

        Args:
            path: File path to check.

        Returns:
            True if file should be processed.
        """
        path_obj = Path(path)

        # Check if file matches patterns
        matched = any(
            path_obj.match(pattern)
            for pattern in self.config.session_patterns
        )
        if not matched:
            return False

        # Debounce
        now = datetime.now(UTC).timestamp()
        last = self._last_processed.get(path, 0)
        if now - last < self.config.debounce_seconds:
            return False

        self._last_processed[path] = now
        return True

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: The file system event.
        """
        if event.is_directory:
            return

        if not self.config.process_on_modify:
            return

        if self._should_process(event.src_path):
            logger.debug(f"Session file modified: {event.src_path}")
            asyncio.run(self.daemon._process_session_file(Path(event.src_path)))

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: The file system event.
        """
        if event.is_directory:
            return

        if not self.config.process_on_create:
            return

        if self._should_process(event.src_path):
            logger.debug(f"Session file created: {event.src_path}")
            asyncio.run(self.daemon._process_session_file(Path(event.src_path)))


class MemoryLayerDaemon:
    """Daemon for automatic memory extraction from Claude Code sessions."""

    def __init__(
        self,
        config: DaemonConfig | None = None,
        engine: MemoryEngine | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        """Initialize the daemon.

        Args:
            config: Daemon configuration.
            engine: Memory engine instance.
            extractor: Memory extractor instance.
        """
        self.config = config or DaemonConfig()
        self._engine = engine
        self._extractor = extractor
        self._observer: Observer | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Callbacks
        self._on_session_start: list[Callable[[SessionInfo], None]] = []
        self._on_session_end: list[Callable[[SessionInfo], None]] = []
        self._on_extraction: list[Callable[[SessionInfo, int], None]] = []

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    # =========================================================================
    # PID File Management
    # =========================================================================

    def _write_pid_file(self) -> None:
        """Write current process PID to file."""
        pid = os.getpid()
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.write_text(str(pid))
        logger.debug(f"Wrote PID {pid} to {self.config.pid_file}")

    def _remove_pid_file(self) -> None:
        """Remove PID file."""
        if self.config.pid_file.exists():
            self.config.pid_file.unlink()
            logger.debug(f"Removed PID file {self.config.pid_file}")

    def _check_existing_daemon(self) -> int | None:
        """Check if another daemon is running.

        Returns:
            PID of existing daemon, or None if not running.
        """
        if not self.config.pid_file.exists():
            return None

        try:
            pid = int(self.config.pid_file.read_text().strip())
            # Check if process is running
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            # Invalid PID or process not running
            self._remove_pid_file()
            return None

    # =========================================================================
    # Privilege Management
    # =========================================================================

    def _drop_privileges(self) -> None:
        """Drop to unprivileged user/group."""
        if not self.config.drop_privileges:
            return

        if os.name != "posix":
            logger.warning("Privilege dropping only supported on POSIX systems")
            return

        try:
            if self.config.target_gid is not None:
                os.setgid(self.config.target_gid)
                logger.info(f"Dropped to GID {self.config.target_gid}")

            if self.config.target_uid is not None:
                os.setuid(self.config.target_uid)
                logger.info(f"Dropped to UID {self.config.target_uid}")
        except PermissionError as e:
            logger.error(f"Failed to drop privileges: {e}")
            raise DaemonError(f"Cannot drop privileges: {e}") from e

    # =========================================================================
    # Session Processing
    # =========================================================================

    async def _process_session_file(self, file_path: Path) -> None:
        """Process a session file.

        Args:
            file_path: Path to the session file.
        """
        try:
            session = self._parse_session_file(file_path)
            if session is None:
                return

            logger.info(f"Processing session: {session.session_id}")

            # Check if session ended
            if not session.is_active and session.transcript:
                # Extract memories if enabled
                if self.config.auto_extract and self._extractor:
                    await self._extract_from_session(session)

                # Notify callbacks
                for callback in self._on_session_end:
                    try:
                        callback(session)
                    except Exception as e:
                        logger.error(f"Session end callback failed: {e}")

        except Exception as e:
            logger.error(f"Failed to process session file {file_path}: {e}")

    def _parse_session_file(self, file_path: Path) -> SessionInfo | None:
        """Parse a session file.

        Args:
            file_path: Path to the session file.

        Returns:
            SessionInfo or None if parsing fails.
        """
        try:
            content = file_path.read_text()

            # Try JSON format
            if file_path.suffix == ".json":
                data = json.loads(content)
                return self._parse_json_session(file_path, data)

            # Try JSONL format (multiple JSON objects per line)
            if file_path.suffix == ".jsonl":
                lines = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
                return self._parse_jsonl_session(file_path, lines)

            # Plain text transcript
            return SessionInfo(
                session_id=file_path.stem,
                file_path=file_path,
                project_path=self._detect_project_path(file_path),
                started_at=None,
                ended_at=None,
                is_active=False,
                transcript=content,
            )

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse session file {file_path}: {e}")
            return None

    def _parse_json_session(self, file_path: Path, data: dict[str, Any]) -> SessionInfo:
        """Parse JSON format session data.

        Args:
            file_path: Path to the session file.
            data: Parsed JSON data.

        Returns:
            SessionInfo instance.
        """
        # Extract session ID
        session_id = data.get("session_id", data.get("id", file_path.stem))

        # Extract timestamps
        started_at = None
        ended_at = None
        if "started_at" in data:
            started_at = datetime.fromisoformat(data["started_at"])
        if "ended_at" in data:
            ended_at = datetime.fromisoformat(data["ended_at"])

        # Extract transcript
        transcript = ""
        if "transcript" in data:
            transcript = data["transcript"]
        elif "messages" in data:
            # Build transcript from messages
            messages = data["messages"]
            parts = []
            for msg in messages:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            transcript = "\n\n".join(parts)

        return SessionInfo(
            session_id=str(session_id),
            file_path=file_path,
            project_path=self._detect_project_path(file_path),
            started_at=started_at,
            ended_at=ended_at,
            is_active=data.get("is_active", ended_at is None),
            transcript=transcript,
            metadata=data.get("metadata", {}),
        )

    def _parse_jsonl_session(
        self, file_path: Path, lines: list[dict[str, Any]]
    ) -> SessionInfo:
        """Parse JSONL format session data.

        Args:
            file_path: Path to the session file.
            lines: List of parsed JSON lines.

        Returns:
            SessionInfo instance.
        """
        # Build transcript from lines
        parts = []
        started_at = None
        ended_at = None
        session_id = file_path.stem
        is_active = True

        for line in lines:
            # Extract timestamps
            if "timestamp" in line:
                ts = datetime.fromisoformat(line["timestamp"])
                if started_at is None:
                    started_at = ts
                ended_at = ts

            # Extract messages
            if "role" in line and "content" in line:
                role = line["role"].capitalize()
                content = line["content"]
                parts.append(f"{role}: {content}")

            # Check for session end marker
            if line.get("type") == "session_end":
                is_active = False

            # Extract session ID if present
            if "session_id" in line:
                session_id = line["session_id"]

        return SessionInfo(
            session_id=str(session_id),
            file_path=file_path,
            project_path=self._detect_project_path(file_path),
            started_at=started_at,
            ended_at=ended_at,
            is_active=is_active,
            transcript="\n\n".join(parts),
        )

    def _detect_project_path(self, session_file: Path) -> Path | None:
        """Detect project path from session file location.

        Args:
            session_file: Path to the session file.

        Returns:
            Project path or None.
        """
        # Session files might contain project path in name or location
        # For now, try to find a project marker
        current = session_file.parent
        while current != current.parent:
            # Look for common project markers
            markers = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent
        return None

    async def _extract_from_session(self, session: SessionInfo) -> int:
        """Extract memories from a session.

        Args:
            session: Session information.

        Returns:
            Number of memories extracted.
        """
        if not self._extractor or not self._engine:
            return 0

        project = session.project_path.name if session.project_path else None

        result = await self._extractor.extract_and_store(
            transcript=session.transcript,
            engine=self._engine,
            project=project,
        )

        if result.success:
            logger.info(f"Extracted {result.memory_count} memories from session {session.session_id}")

            # Notify callbacks
            for callback in self._on_extraction:
                try:
                    callback(session, result.memory_count)
                except Exception as e:
                    logger.error(f"Extraction callback failed: {e}")

            # Update CLAUDE.md if enabled
            if self.config.update_claude_md and session.project_path:
                await self._update_claude_md(session.project_path, project)

            return result.memory_count

        logger.warning(f"Extraction failed for session {session.session_id}: {result.error}")
        return 0

    # =========================================================================
    # CLAUDE.md Management
    # =========================================================================

    async def _update_claude_md(self, project_path: Path, project: str | None) -> None:
        """Update CLAUDE.md with memory context.

        Args:
            project_path: Path to the project directory.
            project: Project name for memory filtering.
        """
        if not self._engine:
            return

        claude_md_path = self.config.claude_md_path or (project_path / "CLAUDE.md")

        # Get context from engine
        context = await self._engine.get_context(project=project, max_memories=20)

        # Build memory section
        memory_section = self._build_claude_md_section(context)

        # Read existing file or create new
        existing_content = ""
        if claude_md_path.exists():
            existing_content = claude_md_path.read_text()

        # Update or append memory section
        new_content = self._merge_claude_md_content(existing_content, memory_section)
        claude_md_path.write_text(new_content)

        logger.info(f"Updated {claude_md_path} with {context.included_count} memories")

    def _build_claude_md_section(self, context: Any) -> str:
        """Build CLAUDE.md memory section.

        Args:
            context: ContextResponse from engine.

        Returns:
            Formatted markdown section.
        """
        lines = [
            "<!-- MEMORY-LAYER-START -->",
            "## Project Knowledge (Auto-Generated)",
            "",
        ]

        # Group by category
        by_category: dict[str, list[Any]] = {}
        for memory in context.memories:
            cat = memory.category.value.title()
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(memory)

        # Format each category
        for category, memories in sorted(by_category.items()):
            lines.append(f"### {category}")
            for memory in memories:
                # Use checkmark for high-confidence, question mark for low
                if memory.outcome_score > 0.3:
                    prefix = "- ✓"
                elif memory.outcome_score < -0.2:
                    prefix = "- ?"
                else:
                    prefix = "-"
                lines.append(f"{prefix} {memory.content}")
            lines.append("")

        lines.append(f"*Last updated: {datetime.now(UTC).isoformat()}*")
        lines.append("<!-- MEMORY-LAYER-END -->")

        return "\n".join(lines)

    def _merge_claude_md_content(self, existing: str, memory_section: str) -> str:
        """Merge memory section into existing CLAUDE.md content.

        Args:
            existing: Existing file content.
            memory_section: New memory section.

        Returns:
            Merged content.
        """
        start_marker = "<!-- MEMORY-LAYER-START -->"
        end_marker = "<!-- MEMORY-LAYER-END -->"

        # Check if markers exist
        if start_marker in existing and end_marker in existing:
            # Replace existing section
            start_idx = existing.index(start_marker)
            end_idx = existing.index(end_marker) + len(end_marker)
            return existing[:start_idx] + memory_section + existing[end_idx:]

        # Append to end
        if existing and not existing.endswith("\n"):
            existing += "\n"
        return existing + "\n" + memory_section

    # =========================================================================
    # Daemon Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the daemon.

        Raises:
            DaemonAlreadyRunningError: If daemon is already running.
            DaemonError: If daemon cannot start.
        """
        # Check for existing daemon
        existing_pid = self._check_existing_daemon()
        if existing_pid:
            raise DaemonAlreadyRunningError(f"Daemon already running with PID {existing_pid}")

        # Ensure session directory exists
        if not self.config.session_dir.exists():
            logger.warning(f"Session directory does not exist: {self.config.session_dir}")
            self.config.session_dir.mkdir(parents=True, exist_ok=True)

        # Write PID file
        self._write_pid_file()

        # Drop privileges if configured
        self._drop_privileges()

        # Set up signal handlers
        self._setup_signal_handlers()

        # Create and start observer
        self._observer = Observer()
        handler = SessionHandler(self, self.config)
        self._observer.schedule(handler, str(self.config.session_dir), recursive=True)
        self._observer.start()

        self._running = True
        logger.info(f"Daemon started, watching {self.config.session_dir}")

    def stop(self) -> None:
        """Stop the daemon gracefully."""
        if not self._running:
            return

        logger.info("Stopping daemon...")
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        self._remove_pid_file()
        self._shutdown_event.set()
        logger.info("Daemon stopped")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        if os.name == "posix":
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGHUP, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    async def run_forever(self) -> None:
        """Run the daemon until stopped."""
        self.start()
        try:
            await self._shutdown_event.wait()
        finally:
            self.stop()

    def run(self) -> None:
        """Run the daemon synchronously."""
        asyncio.run(self.run_forever())

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Check daemon health.

        Returns:
            Health status dictionary.
        """
        return {
            "status": "healthy" if self._running else "stopped",
            "running": self._running,
            "pid": os.getpid() if self._running else None,
            "watch_dir": str(self.config.session_dir),
            "watch_dir_exists": self.config.session_dir.exists(),
            "observer_alive": self._observer.is_alive() if self._observer else False,
        }

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_session_start(self, callback: Callable[[SessionInfo], None]) -> None:
        """Register callback for session start.

        Args:
            callback: Function to call when session starts.
        """
        self._on_session_start.append(callback)

    def on_session_end(self, callback: Callable[[SessionInfo], None]) -> None:
        """Register callback for session end.

        Args:
            callback: Function to call when session ends.
        """
        self._on_session_end.append(callback)

    def on_extraction(self, callback: Callable[[SessionInfo, int], None]) -> None:
        """Register callback for memory extraction.

        Args:
            callback: Function to call after extraction (session, count).
        """
        self._on_extraction.append(callback)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_daemon_status(pid_file: Path | None = None) -> dict[str, Any]:
    """Get status of the daemon.

    Args:
        pid_file: Path to PID file.

    Returns:
        Status dictionary.
    """
    pid_file = pid_file or DEFAULT_PID_FILE

    if not pid_file.exists():
        return {"running": False, "pid": None}

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # Check if process exists
        return {"running": True, "pid": pid}
    except (ValueError, ProcessLookupError, PermissionError):
        return {"running": False, "pid": None}


def stop_daemon(pid_file: Path | None = None) -> bool:
    """Stop a running daemon.

    Args:
        pid_file: Path to PID file.

    Returns:
        True if daemon was stopped, False if not running.
    """
    pid_file = pid_file or DEFAULT_PID_FILE
    status = get_daemon_status(pid_file)

    if not status["running"]:
        return False

    pid = status["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to stop
        import time
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        return True
    except (ProcessLookupError, PermissionError):
        return False

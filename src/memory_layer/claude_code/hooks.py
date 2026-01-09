"""Lifecycle hooks for Claude Code integration.

.. deprecated:: 2.0.0
    This module is deprecated in favor of the native Claude Code 2.1.1+
    hook system defined in hooks/hooks.json. The native hook system:
    - Doesn't require external scripts
    - Is managed by Claude Code directly
    - Supports environment variables like $CLAUDE_SESSION_ID

    Migration guide:
    - pre-session → hooks.json SessionStart
    - post-session → hooks.json SessionEnd
    - pre-compact → hooks.json PreCompact

Provides hooks that integrate with Claude Code's lifecycle:
- pre-session: Inject relevant context before starting
- post-session: Extract learnings after ending
- pre-compact: Extract learnings before context window compaction
"""

from __future__ import annotations

import warnings

# Emit deprecation warning when module is imported
warnings.warn(
    "The hooks module is deprecated as of v2.0.0. "
    "Use the native hook system in hooks/hooks.json instead.",
    DeprecationWarning,
    stacklevel=2,
)

import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory_layer.core.logging import get_logger

if TYPE_CHECKING:
    from memory_layer.core.engine import MemoryEngine
    from memory_layer.extraction.extractor import MemoryExtractor

logger = get_logger(__name__)

# Default paths
DEFAULT_HOOKS_DIR = Path.home() / ".claude" / "hooks"
DEFAULT_STATE_FILE = Path.home() / ".claude" / "memory-layer-state.json"


class HookType(str, Enum):
    """Types of lifecycle hooks."""

    PRE_SESSION = "pre-session"
    """Hook that runs before a session starts."""

    POST_SESSION = "post-session"
    """Hook that runs after a session ends."""

    PRE_COMPACT = "pre-compact"
    """Hook that runs before context window compaction."""


@dataclass
class HookConfig:
    """Configuration for hooks."""

    # Hook installation
    hooks_dir: Path = field(default_factory=lambda: DEFAULT_HOOKS_DIR)
    """Directory where hooks are installed."""

    state_file: Path = field(default_factory=lambda: DEFAULT_STATE_FILE)
    """File for tracking hook state."""

    # Pre-session settings
    inject_context: bool = True
    """Whether to inject context in pre-session hook."""

    max_context_memories: int = 15
    """Maximum memories to include in context."""

    context_query: str | None = None
    """Optional query to filter context memories."""

    # Post-session settings
    extract_memories: bool = True
    """Whether to extract memories in post-session hook."""

    # Pre-compact settings
    save_before_compact: bool = True
    """Whether to extract memories before compaction."""

    # Output settings
    output_format: str = "markdown"
    """Output format for context: 'markdown', 'json', 'text'."""

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        if isinstance(self.hooks_dir, str):
            self.hooks_dir = Path(self.hooks_dir)
        if isinstance(self.state_file, str):
            self.state_file = Path(self.state_file)


@dataclass
class HookState:
    """State tracking for hooks."""

    last_session_id: str | None = None
    """ID of the last session."""

    last_session_start: datetime | None = None
    """When the last session started."""

    last_extraction: datetime | None = None
    """When memories were last extracted."""

    last_context_injection: datetime | None = None
    """When context was last injected."""

    memories_extracted_total: int = 0
    """Total memories extracted across all sessions."""

    sessions_processed: int = 0
    """Number of sessions processed."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "last_session_id": self.last_session_id,
            "last_session_start": self.last_session_start.isoformat() if self.last_session_start else None,
            "last_extraction": self.last_extraction.isoformat() if self.last_extraction else None,
            "last_context_injection": self.last_context_injection.isoformat() if self.last_context_injection else None,
            "memories_extracted_total": self.memories_extracted_total,
            "sessions_processed": self.sessions_processed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HookState:
        """Create from dictionary."""
        return cls(
            last_session_id=data.get("last_session_id"),
            last_session_start=datetime.fromisoformat(data["last_session_start"]) if data.get("last_session_start") else None,
            last_extraction=datetime.fromisoformat(data["last_extraction"]) if data.get("last_extraction") else None,
            last_context_injection=datetime.fromisoformat(data["last_context_injection"]) if data.get("last_context_injection") else None,
            memories_extracted_total=data.get("memories_extracted_total", 0),
            sessions_processed=data.get("sessions_processed", 0),
        )


@dataclass
class HookResult:
    """Result of hook execution."""

    success: bool
    """Whether the hook executed successfully."""

    hook_type: HookType
    """Type of hook that was executed."""

    output: str
    """Output from the hook (e.g., context for pre-session)."""

    memories_processed: int = 0
    """Number of memories processed."""

    error: str | None = None
    """Error message if hook failed."""

    duration_ms: float = 0.0
    """Execution duration in milliseconds."""


class HookError(Exception):
    """Base exception for hook errors."""

    pass


class HookNotInstalledError(HookError):
    """Raised when hook is not installed."""

    pass


class MemoryLayerHooks:
    """Manages lifecycle hooks for Claude Code integration."""

    # Hook script templates
    HOOK_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Memory Layer {hook_type} hook.

Auto-generated by memory-layer. Do not edit directly.
"""

import sys
import json

def main():
    # Read input from stdin
    input_data = sys.stdin.read()

    try:
        # Import memory layer
        from memory_layer.claude_code.hooks import MemoryLayerHooks

        # Execute hook
        hooks = MemoryLayerHooks()
        result = hooks.execute_{hook_func}(input_data)

        # Output result
        if result.output:
            print(result.output)

        sys.exit(0 if result.success else 1)

    except ImportError as e:
        print(f"Error: memory-layer not installed: {{e}}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    def __init__(
        self,
        config: HookConfig | None = None,
        engine: MemoryEngine | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        """Initialize hooks manager.

        Args:
            config: Hook configuration.
            engine: Memory engine instance.
            extractor: Memory extractor instance.
        """
        self.config = config or HookConfig()
        self._engine = engine
        self._extractor = extractor
        self._state: HookState | None = None

    # =========================================================================
    # State Management
    # =========================================================================

    def _load_state(self) -> HookState:
        """Load state from file."""
        if self._state is not None:
            return self._state

        if self.config.state_file.exists():
            try:
                data = json.loads(self.config.state_file.read_text())
                self._state = HookState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load hook state: {e}")
                self._state = HookState()
        else:
            self._state = HookState()

        return self._state

    def _save_state(self) -> None:
        """Save state to file."""
        if self._state is None:
            return

        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.state_file.write_text(
            json.dumps(self._state.to_dict(), indent=2)
        )

    @property
    def state(self) -> HookState:
        """Get current state."""
        return self._load_state()

    # =========================================================================
    # Hook Installation
    # =========================================================================

    def install_hooks(self, hook_types: list[HookType] | None = None) -> dict[HookType, Path]:
        """Install hooks to Claude Code hooks directory.

        Args:
            hook_types: Types of hooks to install (default: all).

        Returns:
            Dictionary mapping hook types to installed paths.
        """
        if hook_types is None:
            hook_types = list(HookType)

        # Ensure hooks directory exists
        self.config.hooks_dir.mkdir(parents=True, exist_ok=True)

        installed: dict[HookType, Path] = {}

        for hook_type in hook_types:
            hook_path = self._install_hook(hook_type)
            installed[hook_type] = hook_path
            logger.info(f"Installed {hook_type.value} hook at {hook_path}")

        return installed

    def _install_hook(self, hook_type: HookType) -> Path:
        """Install a single hook.

        Args:
            hook_type: Type of hook to install.

        Returns:
            Path to installed hook.
        """
        # Determine function name
        hook_func_map = {
            HookType.PRE_SESSION: "pre_session",
            HookType.POST_SESSION: "post_session",
            HookType.PRE_COMPACT: "pre_compact",
        }
        hook_func = hook_func_map[hook_type]

        # Generate script
        script = self.HOOK_SCRIPT_TEMPLATE.format(
            hook_type=hook_type.value,
            hook_func=hook_func,
        )

        # Write hook file
        hook_path = self.config.hooks_dir / f"memory-layer-{hook_type.value}.py"
        hook_path.write_text(script)

        # Make executable
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        return hook_path

    def uninstall_hooks(self, hook_types: list[HookType] | None = None) -> list[HookType]:
        """Uninstall hooks.

        Args:
            hook_types: Types of hooks to uninstall (default: all).

        Returns:
            List of uninstalled hook types.
        """
        if hook_types is None:
            hook_types = list(HookType)

        uninstalled: list[HookType] = []

        for hook_type in hook_types:
            hook_path = self.config.hooks_dir / f"memory-layer-{hook_type.value}.py"
            if hook_path.exists():
                hook_path.unlink()
                uninstalled.append(hook_type)
                logger.info(f"Uninstalled {hook_type.value} hook")

        return uninstalled

    def is_installed(self, hook_type: HookType) -> bool:
        """Check if a hook is installed.

        Args:
            hook_type: Type of hook to check.

        Returns:
            True if hook is installed.
        """
        hook_path = self.config.hooks_dir / f"memory-layer-{hook_type.value}.py"
        return hook_path.exists()

    def get_installed_hooks(self) -> list[HookType]:
        """Get list of installed hooks.

        Returns:
            List of installed hook types.
        """
        return [ht for ht in HookType if self.is_installed(ht)]

    # =========================================================================
    # Hook Execution
    # =========================================================================

    def execute_pre_session(self, input_data: str = "") -> HookResult:
        """Execute pre-session hook.

        Injects relevant context before a session starts.

        Args:
            input_data: Input data (e.g., session info JSON).

        Returns:
            HookResult with context to inject.
        """
        import time
        start_time = time.time()

        try:
            # Parse input data if provided
            session_info: dict[str, Any] = {}
            if input_data.strip():
                try:
                    session_info = json.loads(input_data)
                except json.JSONDecodeError:
                    pass

            # Get project from session info
            project = session_info.get("project")

            # Update state
            state = self._load_state()
            state.last_session_id = session_info.get("session_id")
            state.last_session_start = datetime.now(UTC)
            self._save_state()

            # Get context if enabled
            output = ""
            memories_processed = 0

            if self.config.inject_context and self._engine:
                import asyncio
                context = asyncio.run(
                    self._engine.get_context(
                        query=self.config.context_query,
                        project=project,
                        max_memories=self.config.max_context_memories,
                    )
                )
                memories_processed = context.included_count

                # Format output
                if self.config.output_format == "markdown":
                    output = context.to_markdown()
                elif self.config.output_format == "json":
                    output = json.dumps(context.to_dict(), indent=2)
                else:
                    output = "\n".join(m.content for m in context.memories)

                # Update state
                state.last_context_injection = datetime.now(UTC)
                self._save_state()

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Pre-session hook: injected {memories_processed} memories")

            return HookResult(
                success=True,
                hook_type=HookType.PRE_SESSION,
                output=output,
                memories_processed=memories_processed,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Pre-session hook failed: {e}")
            return HookResult(
                success=False,
                hook_type=HookType.PRE_SESSION,
                output="",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    def execute_post_session(self, input_data: str = "") -> HookResult:
        """Execute post-session hook.

        Extracts learnings after a session ends.

        Args:
            input_data: Session transcript or info JSON.

        Returns:
            HookResult with extraction summary.
        """
        import time
        start_time = time.time()

        try:
            # Parse input - could be JSON or plain transcript
            transcript = input_data
            project: str | None = None

            if input_data.strip().startswith("{"):
                try:
                    session_info = json.loads(input_data)
                    transcript = session_info.get("transcript", "")
                    project = session_info.get("project")
                except json.JSONDecodeError:
                    pass

            # Extract memories if enabled
            memories_processed = 0
            output = ""

            if self.config.extract_memories and self._extractor and self._engine:
                import asyncio
                result = asyncio.run(
                    self._extractor.extract_and_store(
                        transcript=transcript,
                        engine=self._engine,
                        project=project,
                    )
                )

                if result.success:
                    memories_processed = result.memory_count
                    output = f"Extracted {memories_processed} memories:\n"
                    for memory in result.memories:
                        output += f"- [{memory.category.value}] {memory.content[:100]}...\n"

                    # Update state
                    state = self._load_state()
                    state.last_extraction = datetime.now(UTC)
                    state.memories_extracted_total += memories_processed
                    state.sessions_processed += 1
                    self._save_state()
                else:
                    output = f"Extraction failed: {result.error}"

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Post-session hook: extracted {memories_processed} memories")

            return HookResult(
                success=True,
                hook_type=HookType.POST_SESSION,
                output=output,
                memories_processed=memories_processed,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Post-session hook failed: {e}")
            return HookResult(
                success=False,
                hook_type=HookType.POST_SESSION,
                output="",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    def execute_pre_compact(self, input_data: str = "") -> HookResult:
        """Execute pre-compact hook.

        Saves context before context window compaction.

        Args:
            input_data: Current context/transcript to save.

        Returns:
            HookResult with save summary.
        """
        import time
        start_time = time.time()

        try:
            # Parse input
            transcript = input_data
            project: str | None = None

            if input_data.strip().startswith("{"):
                try:
                    compact_info = json.loads(input_data)
                    transcript = compact_info.get("context", compact_info.get("transcript", ""))
                    project = compact_info.get("project")
                except json.JSONDecodeError:
                    pass

            # Extract memories before compact if enabled
            memories_processed = 0
            output = ""

            if self.config.save_before_compact and self._extractor and self._engine:
                import asyncio
                result = asyncio.run(
                    self._extractor.extract_and_store(
                        transcript=transcript,
                        engine=self._engine,
                        project=project,
                    )
                )

                if result.success:
                    memories_processed = result.memory_count
                    output = f"Saved {memories_processed} memories before compaction"

                    # Update state
                    state = self._load_state()
                    state.last_extraction = datetime.now(UTC)
                    state.memories_extracted_total += memories_processed
                    self._save_state()
                else:
                    output = f"Pre-compact extraction failed: {result.error}"

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Pre-compact hook: saved {memories_processed} memories")

            return HookResult(
                success=True,
                hook_type=HookType.PRE_COMPACT,
                output=output,
                memories_processed=memories_processed,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Pre-compact hook failed: {e}")
            return HookResult(
                success=False,
                hook_type=HookType.PRE_COMPACT,
                output="",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    # =========================================================================
    # Configuration Management
    # =========================================================================

    def get_hook_config_json(self) -> str:
        """Get hook configuration as JSON for Claude Code settings.

        Returns:
            JSON configuration string.
        """
        config = {
            "hooks": {
                "pre-session": {
                    "command": f"python3 {self.config.hooks_dir / 'memory-layer-pre-session.py'}",
                    "timeout": 10000,
                    "enabled": self.is_installed(HookType.PRE_SESSION),
                },
                "post-session": {
                    "command": f"python3 {self.config.hooks_dir / 'memory-layer-post-session.py'}",
                    "timeout": 30000,
                    "enabled": self.is_installed(HookType.POST_SESSION),
                },
                "pre-compact": {
                    "command": f"python3 {self.config.hooks_dir / 'memory-layer-pre-compact.py'}",
                    "timeout": 30000,
                    "enabled": self.is_installed(HookType.PRE_COMPACT),
                },
            }
        }
        return json.dumps(config, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================

def install_all_hooks(
    hooks_dir: Path | None = None,
    engine: MemoryEngine | None = None,
    extractor: MemoryExtractor | None = None,
) -> dict[HookType, Path]:
    """Install all hooks.

    Args:
        hooks_dir: Directory to install hooks.
        engine: Memory engine instance.
        extractor: Memory extractor instance.

    Returns:
        Dictionary of installed hooks.
    """
    config = HookConfig()
    if hooks_dir:
        config.hooks_dir = hooks_dir

    hooks = MemoryLayerHooks(config=config, engine=engine, extractor=extractor)
    return hooks.install_hooks()


def uninstall_all_hooks(hooks_dir: Path | None = None) -> list[HookType]:
    """Uninstall all hooks.

    Args:
        hooks_dir: Directory where hooks are installed.

    Returns:
        List of uninstalled hook types.
    """
    config = HookConfig()
    if hooks_dir:
        config.hooks_dir = hooks_dir

    hooks = MemoryLayerHooks(config=config)
    return hooks.uninstall_hooks()


def get_hook_status(hooks_dir: Path | None = None) -> dict[str, Any]:
    """Get status of all hooks.

    Args:
        hooks_dir: Directory where hooks are installed.

    Returns:
        Status dictionary.
    """
    config = HookConfig()
    if hooks_dir:
        config.hooks_dir = hooks_dir

    hooks = MemoryLayerHooks(config=config)
    state = hooks.state

    return {
        "installed_hooks": [ht.value for ht in hooks.get_installed_hooks()],
        "hooks_dir": str(config.hooks_dir),
        "state": state.to_dict(),
    }

"""Task tracker integration for Memory Layer.

This module provides integration with task tracking systems,
enabling automatic outcome capture when tasks complete.

Supported task systems:
- Beads task tracker (.beads/ directory)
- Claude Code todos (~/.claude/todos/)

Key Features:
- Parse task data from multiple sources
- Link memories to tasks they're used for
- Auto-record outcomes when tasks complete
- Unified context combining tasks + memories
- Unified adapter for all task sources

Example:
    >>> from memory_layer.tasks import UnifiedTaskAdapter
    >>> adapter = UnifiedTaskAdapter(engine)
    >>> await adapter.initialize()
    >>> tasks = adapter.list_tasks()  # From all sources
    >>> result = await adapter.sync_all()
    >>> print(f"Recorded {result.total_outcomes_recorded} outcomes")

    # Or use specific adapters:
    >>> from memory_layer.tasks import BeadsAdapter, ClaudeCodeAdapter
"""

from memory_layer.tasks.models import (
    # Enums
    BeadsTaskStatus,
    ClaudeCodeTaskStatus,
    TaskSource,
    # Beads models
    BeadsTask,
    BeadsSyncResult,
    # Claude Code models
    ClaudeCodeTask,
    CLAUDE_CODE_STATUS_TO_OUTCOME,
    # Shared models
    Task,
    TaskMemoryLink,
    TaskContext,
    TaskSyncResult,
    # Constants
    TASK_STATUS_TO_OUTCOME,
    CANCELLED_TASK_PENALTY,
)
from memory_layer.tasks.parser import BeadsDirectoryNotFoundError, BeadsParser
from memory_layer.tasks.claude_code_parser import (
    ClaudeCodeDirectoryNotFoundError,
    ClaudeCodeParser,
)
from memory_layer.tasks.linking import TaskMemoryLinker
from memory_layer.tasks.outcomes import OutcomeCapture, auto_capture_outcome
from memory_layer.tasks.adapter import BeadsAdapter, NullBeadsAdapter, create_adapter
from memory_layer.tasks.claude_code_adapter import (
    ClaudeCodeAdapter,
    NullClaudeCodeAdapter,
    create_claude_code_adapter,
)
from memory_layer.tasks.unified_adapter import (
    UnifiedTaskAdapter,
    UnifiedTask,
    UnifiedSyncResult,
    create_unified_adapter,
)
from memory_layer.tasks.cli_bridge import BeadsCLI, get_beads_cli

__all__ = [
    # === Enums ===
    "BeadsTaskStatus",
    "ClaudeCodeTaskStatus",
    "TaskSource",
    # === Task Models ===
    "BeadsTask",
    "ClaudeCodeTask",
    "Task",  # Type alias for BeadsTask | ClaudeCodeTask
    "TaskMemoryLink",
    "TaskContext",
    # === Sync Results ===
    "BeadsSyncResult",
    "TaskSyncResult",
    "UnifiedSyncResult",
    # === Parsers ===
    "BeadsParser",
    "BeadsDirectoryNotFoundError",
    "ClaudeCodeParser",
    "ClaudeCodeDirectoryNotFoundError",
    # === CLI Bridge ===
    "BeadsCLI",
    "get_beads_cli",
    # === Linking ===
    "TaskMemoryLinker",
    # === Outcome Capture ===
    "OutcomeCapture",
    "auto_capture_outcome",
    # === Beads Adapter ===
    "BeadsAdapter",
    "NullBeadsAdapter",
    "create_adapter",
    # === Claude Code Adapter ===
    "ClaudeCodeAdapter",
    "NullClaudeCodeAdapter",
    "create_claude_code_adapter",
    # === Unified Adapter (main entry point) ===
    "UnifiedTaskAdapter",
    "UnifiedTask",
    "create_unified_adapter",
    # === Constants ===
    "TASK_STATUS_TO_OUTCOME",
    "CLAUDE_CODE_STATUS_TO_OUTCOME",
    "CANCELLED_TASK_PENALTY",
]

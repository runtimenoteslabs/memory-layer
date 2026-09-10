"""Task tracker integration for Runtime Memory.

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
    >>> from runtime_memory.tasks import UnifiedTaskAdapter
    >>> adapter = UnifiedTaskAdapter(engine)
    >>> await adapter.initialize()
    >>> tasks = adapter.list_tasks()  # From all sources
    >>> result = await adapter.sync_all()
    >>> print(f"Recorded {result.total_outcomes_recorded} outcomes")

    # Or use specific adapters:
    >>> from runtime_memory.tasks import BeadsAdapter, ClaudeCodeAdapter
"""

from runtime_memory.tasks.adapter import BeadsAdapter, NullBeadsAdapter, create_adapter
from runtime_memory.tasks.claude_code_adapter import (
    ClaudeCodeAdapter,
    NullClaudeCodeAdapter,
    create_claude_code_adapter,
)
from runtime_memory.tasks.claude_code_parser import (
    ClaudeCodeDirectoryNotFoundError,
    ClaudeCodeParser,
)
from runtime_memory.tasks.cli_bridge import BeadsCLI, get_beads_cli
from runtime_memory.tasks.linking import TaskMemoryLinker
from runtime_memory.tasks.models import (
    CANCELLED_TASK_PENALTY,
    CLAUDE_CODE_STATUS_TO_OUTCOME,
    # Constants
    TASK_STATUS_TO_OUTCOME,
    BeadsSyncResult,
    # Beads models
    BeadsTask,
    # Enums
    BeadsTaskStatus,
    # Claude Code models
    ClaudeCodeTask,
    ClaudeCodeTaskStatus,
    # Shared models
    Task,
    TaskContext,
    TaskMemoryLink,
    TaskSource,
    TaskSyncResult,
)
from runtime_memory.tasks.outcomes import OutcomeCapture, auto_capture_outcome
from runtime_memory.tasks.parser import BeadsDirectoryNotFoundError, BeadsParser
from runtime_memory.tasks.unified_adapter import (
    UnifiedSyncResult,
    UnifiedTask,
    UnifiedTaskAdapter,
    create_unified_adapter,
)

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

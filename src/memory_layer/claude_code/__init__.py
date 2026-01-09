"""Claude Code integration module.

This module provides integration with Claude Code through:
- File watching daemon for automatic session processing
- Lifecycle hooks for context injection and extraction
- Custom slash commands for memory operations
"""

from memory_layer.claude_code.commands import (
    CommandConfig,
    CommandHandler,
    CommandResult,
    CommandType,
    export_command_schemas,
    get_command_schemas,
)
from memory_layer.claude_code.daemon import (
    DaemonConfig,
    MemoryLayerDaemon,
    SessionInfo,
)
from memory_layer.claude_code.hooks import (
    HookConfig,
    HookResult,
    HookState,
    HookType,
    MemoryLayerHooks,
)

__all__ = [
    # Commands
    "CommandConfig",
    "CommandHandler",
    "CommandResult",
    "CommandType",
    "export_command_schemas",
    "get_command_schemas",
    # Daemon
    "DaemonConfig",
    "MemoryLayerDaemon",
    "SessionInfo",
    # Hooks
    "HookConfig",
    "HookResult",
    "HookState",
    "HookType",
    "MemoryLayerHooks",
]

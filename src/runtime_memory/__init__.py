"""Runtime Memory - Persistent memory for AI coding agents with outcome-based learning.

This package provides:
- Core memory storage and retrieval with SQLite
- Hybrid search using BM25 and vector embeddings
- Outcome-based learning (advice that works gets boosted, failures get penalized)
- Multi-agent access via MCP, REST API, CLI, and SDK
- Task integration with Beads and other task systems
"""

from __future__ import annotations

__version__ = "3.1.0"
__author__ = "exitcode42"

from runtime_memory.core.legacy_env import apply_legacy_env
from runtime_memory.core.logging import get_logger, setup_logging

# Carry pre-3.0 MEMORY_LAYER_* settings onto the current names before anything
# reads them, so a config written before the rename still works.
apply_legacy_env()

__all__ = [
    "__version__",
    "apply_legacy_env",
    "get_logger",
    "setup_logging",
]

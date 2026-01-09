"""Memory Layer - Persistent memory for AI coding agents with outcome-based learning.

This package provides:
- Core memory storage and retrieval with SQLite
- Hybrid search using BM25 and vector embeddings
- Outcome-based learning (advice that works gets boosted, failures get penalized)
- Multi-agent access via MCP, REST API, CLI, and SDK
- Task integration with Beads and other task systems
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Memory Layer Team"

from memory_layer.core.logging import get_logger, setup_logging

__all__ = [
    "__version__",
    "get_logger",
    "setup_logging",
]

"""Custom commands for Claude Code integration.

This module implements slash commands that can be used within Claude Code
to interact with the memory layer.

Commands:
- /remember: Store a new memory
- /recall: Search for memories
- /forget: Delete a memory
- /outcome: Record outcome feedback
- /context: Get formatted context
- /memories: List memories
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from memory_layer.core.engine import MemoryEngine
from memory_layer.core.models import (
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
    SearchResult,
)


class CommandType(str, Enum):
    """Types of custom commands."""

    REMEMBER = "remember"
    RECALL = "recall"
    FORGET = "forget"
    OUTCOME = "outcome"
    CONTEXT = "context"
    MEMORIES = "memories"


@dataclass
class CommandResult:
    """Result from executing a command."""

    success: bool
    command: CommandType
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "success": self.success,
            "command": self.command.value,
            "message": self.message,
        }
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Convert to markdown format for display."""
        if not self.success:
            return f"❌ **Error**: {self.error or self.message}"

        lines = [f"✅ **{self.command.value.title()}**: {self.message}"]

        if self.data:
            if "memories" in self.data:
                lines.append("")
                for mem in self.data["memories"]:
                    score_str = f" (score: {mem.get('score', 0):.2f})" if "score" in mem else ""
                    lines.append(f"- **{mem.get('category', 'unknown')}**{score_str}: {mem.get('content', '')[:100]}...")
            elif "memory" in self.data:
                mem = self.data["memory"]
                lines.append(f"\n**ID**: `{mem.get('id', 'unknown')}`")
                lines.append(f"**Category**: {mem.get('category', 'unknown')}")
                lines.append(f"**Content**: {mem.get('content', '')}")
            elif "context" in self.data:
                lines.append("")
                lines.append(self.data["context"])
            elif "stats" in self.data:
                stats = self.data["stats"]
                lines.append(f"\n**Total**: {stats.get('total', 0)} memories")
                lines.append(f"**Active**: {stats.get('active', 0)}")
                lines.append(f"**Archived**: {stats.get('archived', 0)}")

        return "\n".join(lines)


@dataclass
class CommandConfig:
    """Configuration for command handler."""

    default_project: str | None = None
    default_scope: MemoryScope = MemoryScope.PROJECT
    max_results: int = 10
    include_archived: bool = False
    output_format: str = "markdown"  # markdown, json, plain


class CommandHandler:
    """Handler for custom slash commands.

    This class processes slash commands and interacts with the memory engine
    to perform operations like storing, searching, and managing memories.
    """

    def __init__(
        self,
        engine: MemoryEngine,
        config: CommandConfig | None = None,
    ) -> None:
        """Initialize command handler.

        Args:
            engine: Memory engine instance
            config: Command configuration
        """
        self.engine = engine
        self.config = config or CommandConfig()

    async def execute(self, command: str, args: str = "", project: str | None = None) -> CommandResult:
        """Execute a slash command.

        Args:
            command: Command name (with or without leading slash)
            args: Command arguments as string
            project: Project context (optional)

        Returns:
            CommandResult with operation outcome
        """
        # Normalize command name
        cmd = command.lstrip("/").lower().strip()

        # Use project from config if not specified
        project = project or self.config.default_project

        try:
            if cmd == CommandType.REMEMBER.value:
                return await self._remember(args, project)
            elif cmd == CommandType.RECALL.value:
                return await self._recall(args, project)
            elif cmd == CommandType.FORGET.value:
                return await self._forget(args, project)
            elif cmd == CommandType.OUTCOME.value:
                return await self._outcome(args, project)
            elif cmd == CommandType.CONTEXT.value:
                return await self._context(args, project)
            elif cmd == CommandType.MEMORIES.value:
                return await self._memories(args, project)
            else:
                return CommandResult(
                    success=False,
                    command=CommandType.REMEMBER,  # Default
                    message="Unknown command",
                    error=f"Unknown command: /{cmd}. Available: /remember, /recall, /forget, /outcome, /context, /memories",
                )
        except Exception as e:
            return CommandResult(
                success=False,
                command=CommandType(cmd) if cmd in [c.value for c in CommandType] else CommandType.REMEMBER,
                message="Command execution failed",
                error=str(e),
            )

    async def _remember(self, args: str, project: str | None) -> CommandResult:
        """Store a new memory.

        Format: /remember [category:CATEGORY] content
        Example: /remember category:gotcha Always use --no-cache with pip install
        """
        # Parse category from args
        category = MemoryCategory.PATTERN
        content = args.strip()

        # Check for category prefix
        category_match = re.match(r"category:(\w+)\s+(.+)", content, re.IGNORECASE | re.DOTALL)
        if category_match:
            cat_str = category_match.group(1).upper()
            content = category_match.group(2).strip()
            try:
                category = MemoryCategory(cat_str.lower())
            except ValueError:
                # Try to match partial category name
                for cat in MemoryCategory:
                    if cat.value.upper().startswith(cat_str):
                        category = cat
                        break

        if not content:
            return CommandResult(
                success=False,
                command=CommandType.REMEMBER,
                message="No content provided",
                error="Usage: /remember [category:CATEGORY] content",
            )

        # Add the memory
        memory = await self.engine.add(
            content=content,
            category=category,
            scope=self.config.default_scope,
            source=MemorySource.EXPLICIT,
            project=project,
        )

        return CommandResult(
            success=True,
            command=CommandType.REMEMBER,
            message=f"Memory stored with ID: {memory.id}",
            data={
                "memory": {
                    "id": memory.id,
                    "content": memory.content,
                    "category": memory.category.value,
                    "scope": memory.scope.value,
                    "project": memory.project,
                }
            },
        )

    async def _recall(self, args: str, project: str | None) -> CommandResult:
        """Search for memories.

        Format: /recall query [limit:N]
        Example: /recall docker memory issues limit:5
        """
        query = args.strip()
        limit = self.config.max_results

        # Check for limit suffix
        limit_match = re.search(r"\s+limit:(\d+)$", query)
        if limit_match:
            limit = min(int(limit_match.group(1)), 50)
            query = query[: limit_match.start()].strip()

        if not query:
            return CommandResult(
                success=False,
                command=CommandType.RECALL,
                message="No search query provided",
                error="Usage: /recall query [limit:N]",
            )

        # Search memories
        results = await self.engine.search(
            query=query,
            project=project,
            limit=limit,
        )

        if not results:
            return CommandResult(
                success=True,
                command=CommandType.RECALL,
                message="No memories found matching your query",
                data={"memories": [], "query": query},
            )

        memories_data = []
        for result in results:
            memories_data.append({
                "id": result.memory.id,
                "content": result.memory.content,
                "category": result.memory.category.value,
                "score": result.score,
                "project": result.memory.project,
            })

        return CommandResult(
            success=True,
            command=CommandType.RECALL,
            message=f"Found {len(results)} memories",
            data={"memories": memories_data, "query": query},
        )

    async def _forget(self, args: str, project: str | None) -> CommandResult:
        """Delete/archive a memory.

        Format: /forget memory_id [--permanent]
        Example: /forget abc123
        """
        args = args.strip()
        permanent = False

        if "--permanent" in args:
            permanent = True
            args = args.replace("--permanent", "").strip()

        memory_id = args.strip()

        if not memory_id:
            return CommandResult(
                success=False,
                command=CommandType.FORGET,
                message="No memory ID provided",
                error="Usage: /forget memory_id [--permanent]",
            )

        # Check if memory exists
        memory = await self.engine.get(memory_id)
        if not memory:
            return CommandResult(
                success=False,
                command=CommandType.FORGET,
                message="Memory not found",
                error=f"No memory found with ID: {memory_id}",
            )

        # Delete/archive the memory
        await self.engine.delete(memory_id, permanent=permanent)

        action = "deleted" if permanent else "archived"
        return CommandResult(
            success=True,
            command=CommandType.FORGET,
            message=f"Memory {action}: {memory_id}",
            data={"memory_id": memory_id, "permanent": permanent},
        )

    async def _outcome(self, args: str, project: str | None) -> CommandResult:
        """Record outcome feedback for a memory.

        Format: /outcome memory_id worked|failed|partial [notes]
        Example: /outcome abc123 worked The solution fixed the issue
        """
        parts = args.strip().split(None, 2)

        if len(parts) < 2:
            return CommandResult(
                success=False,
                command=CommandType.OUTCOME,
                message="Invalid arguments",
                error="Usage: /outcome memory_id worked|failed|partial [notes]",
            )

        memory_id = parts[0]
        outcome_str = parts[1].lower()
        notes = parts[2] if len(parts) > 2 else None

        # Parse outcome
        try:
            outcome = Outcome(outcome_str)
        except ValueError:
            return CommandResult(
                success=False,
                command=CommandType.OUTCOME,
                message="Invalid outcome value",
                error=f"Outcome must be one of: worked, failed, partial. Got: {outcome_str}",
            )

        # Check if memory exists
        memory = await self.engine.get(memory_id)
        if not memory:
            return CommandResult(
                success=False,
                command=CommandType.OUTCOME,
                message="Memory not found",
                error=f"No memory found with ID: {memory_id}",
            )

        # Record the outcome
        old_score = memory.outcome_score
        updated_memory = await self.engine.record_outcome(
            memory_id=memory_id,
            outcome=outcome,
            context=notes,
        )

        return CommandResult(
            success=True,
            command=CommandType.OUTCOME,
            message=f"Outcome recorded for memory {memory_id}",
            data={
                "memory_id": memory_id,
                "outcome": outcome.value,
                "old_score": old_score,
                "new_score": updated_memory.outcome_score,
                "notes": notes,
            },
        )

    async def _context(self, args: str, project: str | None) -> CommandResult:
        """Get formatted context for current project.

        Format: /context [project_name] [limit:N]
        Example: /context my-project limit:10
        """
        args = args.strip()
        limit = self.config.max_results

        # Check for limit suffix
        limit_match = re.search(r"\s+limit:(\d+)$", args)
        if limit_match:
            limit = min(int(limit_match.group(1)), 50)
            args = args[: limit_match.start()].strip()

        # Use args as project name if provided
        target_project = args or project

        # Get context
        context = await self.engine.get_context(
            project=target_project,
            limit=limit,
        )

        return CommandResult(
            success=True,
            command=CommandType.CONTEXT,
            message=f"Context for project: {target_project or 'all'}",
            data={
                "context": context.formatted,
                "memory_count": context.memory_count,
                "token_estimate": context.token_estimate,
                "project": target_project,
            },
        )

    async def _memories(self, args: str, project: str | None) -> CommandResult:
        """List memories with optional filtering.

        Format: /memories [category:CATEGORY] [scope:SCOPE] [limit:N]
        Example: /memories category:gotcha limit:20
        """
        category: MemoryCategory | None = None
        scope: MemoryScope | None = None
        limit = self.config.max_results
        include_archived = self.config.include_archived

        # Parse arguments
        args_str = args.strip()

        # Category filter
        cat_match = re.search(r"category:(\w+)", args_str, re.IGNORECASE)
        if cat_match:
            cat_str = cat_match.group(1).lower()
            try:
                category = MemoryCategory(cat_str)
            except ValueError:
                for cat in MemoryCategory:
                    if cat.value.startswith(cat_str):
                        category = cat
                        break

        # Scope filter
        scope_match = re.search(r"scope:(\w+)", args_str, re.IGNORECASE)
        if scope_match:
            scope_str = scope_match.group(1).lower()
            try:
                scope = MemoryScope(scope_str)
            except ValueError:
                pass

        # Limit
        limit_match = re.search(r"limit:(\d+)", args_str)
        if limit_match:
            limit = min(int(limit_match.group(1)), 100)

        # Include archived flag
        if "--archived" in args_str or "-a" in args_str:
            include_archived = True

        # List memories
        memories = await self.engine.list(
            project=project,
            category=category,
            scope=scope,
            include_archived=include_archived,
            limit=limit,
        )

        # Get stats
        stats = await self.engine.stats(project=project)

        memories_data = []
        for memory in memories:
            memories_data.append({
                "id": memory.id,
                "content": memory.content[:100] + ("..." if len(memory.content) > 100 else ""),
                "category": memory.category.value,
                "scope": memory.scope.value,
                "outcome_score": memory.outcome_score,
                "use_count": memory.use_count,
                "archived": memory.archived,
            })

        return CommandResult(
            success=True,
            command=CommandType.MEMORIES,
            message=f"Listing {len(memories)} memories",
            data={
                "memories": memories_data,
                "stats": {
                    "total": stats.total_memories,
                    "active": stats.active_memories,
                    "archived": stats.archived_memories,
                    "by_category": {k.value: v for k, v in stats.by_category.items()},
                },
                "filters": {
                    "category": category.value if category else None,
                    "scope": scope.value if scope else None,
                    "project": project,
                    "include_archived": include_archived,
                },
            },
        )


def get_command_schemas() -> dict[str, dict[str, Any]]:
    """Generate JSON schemas for all commands.

    Returns:
        Dictionary mapping command names to their JSON schemas
    """
    schemas = {
        "remember": {
            "name": "remember",
            "description": "Store a new memory in the memory layer",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember",
                    },
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in MemoryCategory],
                        "description": "Category of the memory",
                        "default": "pattern",
                    },
                },
                "required": ["content"],
            },
            "examples": [
                "/remember Always run migrations before deploying",
                "/remember category:gotcha Docker needs --shm-size for PyTorch",
            ],
        },
        "recall": {
            "name": "recall",
            "description": "Search for memories matching a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            "examples": [
                "/recall docker memory issues",
                "/recall database connection limit:5",
            ],
        },
        "forget": {
            "name": "forget",
            "description": "Delete or archive a memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to forget",
                    },
                    "permanent": {
                        "type": "boolean",
                        "description": "Permanently delete instead of archive",
                        "default": False,
                    },
                },
                "required": ["memory_id"],
            },
            "examples": [
                "/forget abc123",
                "/forget abc123 --permanent",
            ],
        },
        "outcome": {
            "name": "outcome",
            "description": "Record whether a memory was helpful",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["worked", "failed", "partial"],
                        "description": "The outcome of using this memory",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional notes about the outcome",
                    },
                },
                "required": ["memory_id", "outcome"],
            },
            "examples": [
                "/outcome abc123 worked",
                "/outcome abc123 failed The solution was outdated",
            ],
        },
        "context": {
            "name": "context",
            "description": "Get formatted context for the current project",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name (optional, uses current project)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to include",
                        "default": 10,
                    },
                },
                "required": [],
            },
            "examples": [
                "/context",
                "/context my-project limit:20",
            ],
        },
        "memories": {
            "name": "memories",
            "description": "List memories with optional filtering",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in MemoryCategory],
                        "description": "Filter by category",
                    },
                    "scope": {
                        "type": "string",
                        "enum": [s.value for s in MemoryScope],
                        "description": "Filter by scope",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived memories",
                        "default": False,
                    },
                },
                "required": [],
            },
            "examples": [
                "/memories",
                "/memories category:gotcha limit:20",
                "/memories scope:global --archived",
            ],
        },
    }

    return schemas


def export_command_schemas(output_path: Path) -> None:
    """Export command schemas to a JSON file.

    Args:
        output_path: Path to write the JSON schema file
    """
    schemas = get_command_schemas()
    with open(output_path, "w") as f:
        json.dump(schemas, f, indent=2)

"""MCP (Model Context Protocol) Server for Memory Layer.

This module implements an MCP server that exposes memory operations
as tools for AI agents. It uses JSON-RPC over stdio for communication.

Usage:
    mem serve --mcp

Protocol:
    The server communicates via JSON-RPC 2.0 over stdin/stdout.
    Each message is a newline-delimited JSON object.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from memory_layer.core.engine import MemoryEngine
from memory_layer.core.logging import get_logger
from memory_layer.core.models import (
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
)

logger = get_logger(__name__)


# =============================================================================
# MCP Protocol Types
# =============================================================================


class MCPErrorCode(int, Enum):
    """JSON-RPC error codes for MCP."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom error codes
    TOOL_NOT_FOUND = -32001
    VALIDATION_ERROR = -32002
    RATE_LIMITED = -32003


@dataclass
class MCPError:
    """MCP error response."""

    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass
class MCPRequest:
    """MCP JSON-RPC request."""

    jsonrpc: str
    method: str
    id: Optional[str | int] = None
    params: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPRequest:
        """Create from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data["method"],
            id=data.get("id"),
            params=data.get("params"),
        )


@dataclass
class MCPResponse:
    """MCP JSON-RPC response."""

    jsonrpc: str = "2.0"
    id: Optional[str | int] = None
    result: Optional[Any] = None
    error: Optional[MCPError] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        response: dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            response["id"] = self.id
        if self.error is not None:
            response["error"] = self.error.to_dict()
        else:
            response["result"] = self.result
        return response


@dataclass
class MCPToolSchema:
    """Schema definition for an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class RateLimiter:
    """Simple rate limiter for MCP requests."""

    max_requests: int = 100
    window_seconds: float = 60.0
    _requests: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def is_allowed(self, client_id: str = "default") -> bool:
        """Check if a request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        # Check limit
        if len(self._requests[client_id]) >= self.max_requests:
            return False

        # Record request
        self._requests[client_id].append(now)
        return True

    def reset(self, client_id: str = "default") -> None:
        """Reset rate limit for a client."""
        self._requests[client_id] = []


# =============================================================================
# Tool Schemas
# =============================================================================


TOOL_SCHEMAS: list[MCPToolSchema] = [
    MCPToolSchema(
        name="search_memories",
        description="Search for relevant memories by query. Returns memories ranked by relevance and outcome score.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                    "minLength": 1,
                    "maxLength": 1000,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (1-100)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
                "categories": {
                    "type": "array",
                    "description": "Filter by categories",
                    "items": {
                        "type": "string",
                        "enum": [c.value for c in MemoryCategory],
                    },
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project name",
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum outcome score (-1.0 to 1.0)",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
            },
            "required": ["query"],
        },
    ),
    MCPToolSchema(
        name="add_memory",
        description="Store a new memory with category. Categories: architecture, convention, decision, pattern, gotcha, workaround, troubleshooting, command, preference, dependency, environment, coding_style, tool_preference, context, todo, general.",
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory content text",
                    "minLength": 1,
                    "maxLength": 10000,
                },
                "category": {
                    "type": "string",
                    "description": "Memory category",
                    "enum": [c.value for c in MemoryCategory],
                    "default": "general",
                },
                "project": {
                    "type": "string",
                    "description": "Project name for scoping",
                },
                "tags": {
                    "type": "array",
                    "description": "Optional tags",
                    "items": {"type": "string"},
                    "maxItems": 20,
                },
                "importance": {
                    "type": "number",
                    "description": "Importance weight (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                },
            },
            "required": ["content"],
        },
    ),
    MCPToolSchema(
        name="record_outcome",
        description="Record worked/failed/partial feedback for memories. This adjusts the outcome score: worked +0.2, failed -0.3, partial +0.05.",
        input_schema={
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "description": "IDs of memories to apply outcome to",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 50,
                },
                "outcome": {
                    "type": "string",
                    "description": "Outcome result",
                    "enum": ["worked", "failed", "partial"],
                },
            },
            "required": ["memory_ids", "outcome"],
        },
    ),
    MCPToolSchema(
        name="get_context",
        description="Get memories relevant to the current project context, formatted for injection into the conversation.",
        input_schema={
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Project name",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum memories to include",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
                "format": {
                    "type": "string",
                    "description": "Output format",
                    "enum": ["brief", "detailed", "markdown"],
                    "default": "markdown",
                },
            },
        },
    ),
    MCPToolSchema(
        name="update_memory",
        description="Update an existing memory's content or category.",
        input_schema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Memory ID to update",
                },
                "content": {
                    "type": "string",
                    "description": "New content (optional)",
                    "minLength": 1,
                    "maxLength": 10000,
                },
                "category": {
                    "type": "string",
                    "description": "New category (optional)",
                    "enum": [c.value for c in MemoryCategory],
                },
                "tags": {
                    "type": "array",
                    "description": "New tags (optional)",
                    "items": {"type": "string"},
                },
            },
            "required": ["id"],
        },
    ),
    MCPToolSchema(
        name="delete_memory",
        description="Archive (soft delete) a memory by ID.",
        input_schema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Memory ID to delete",
                },
            },
            "required": ["id"],
        },
    ),
    MCPToolSchema(
        name="list_memories",
        description="List memories with optional filters by category, project, or minimum score.",
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category",
                    "enum": [c.value for c in MemoryCategory],
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived memories",
                    "default": False,
                },
            },
        },
    ),
    MCPToolSchema(
        name="get_stats",
        description="Get memory statistics including total count, category distribution, and average outcome scores.",
        input_schema={
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Filter stats by project",
                },
            },
        },
    ),
]


# =============================================================================
# Input Validators
# =============================================================================


def validate_string(
    value: Any,
    name: str,
    min_length: int = 0,
    max_length: int = 10000,
    required: bool = True,
) -> Optional[str]:
    """Validate a string parameter."""
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")

    if len(value) < min_length:
        raise ValueError(f"{name} must be at least {min_length} characters")

    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")

    return value


def validate_integer(
    value: Any,
    name: str,
    minimum: int = 0,
    maximum: int = 100,
    default: Optional[int] = None,
) -> Optional[int]:
    """Validate an integer parameter."""
    if value is None:
        return default

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")

    if value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")

    return value


def validate_float(
    value: Any,
    name: str,
    minimum: float = 0.0,
    maximum: float = 1.0,
    default: Optional[float] = None,
) -> Optional[float]:
    """Validate a float parameter."""
    if value is None:
        return default

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")

    if value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")

    return float(value)


def validate_enum(
    value: Any,
    name: str,
    enum_class: type[Enum],
    required: bool = True,
    default: Optional[Enum] = None,
) -> Optional[Enum]:
    """Validate an enum parameter."""
    if value is None:
        if required and default is None:
            raise ValueError(f"{name} is required")
        return default

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")

    try:
        return enum_class(value)
    except ValueError:
        valid_values = [e.value for e in enum_class]
        raise ValueError(f"{name} must be one of: {valid_values}")


def validate_list(
    value: Any,
    name: str,
    item_type: type = str,
    min_items: int = 0,
    max_items: int = 100,
    required: bool = False,
) -> Optional[list]:
    """Validate a list parameter."""
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None

    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")

    if len(value) < min_items:
        raise ValueError(f"{name} must have at least {min_items} items")

    if len(value) > max_items:
        raise ValueError(f"{name} must have at most {max_items} items")

    for i, item in enumerate(value):
        if not isinstance(item, item_type):
            raise ValueError(f"{name}[{i}] must be a {item_type.__name__}")

    return value


# =============================================================================
# MCP Server
# =============================================================================


class MCPServer:
    """MCP Server for Memory Layer.

    Exposes memory operations as MCP tools via JSON-RPC over stdio.
    """

    def __init__(
        self,
        engine: Optional[MemoryEngine] = None,
        rate_limit: int = 100,
        rate_window: float = 60.0,
    ):
        """Initialize the MCP server.

        Args:
            engine: MemoryEngine instance (created if not provided)
            rate_limit: Maximum requests per window
            rate_window: Rate limit window in seconds
        """
        self.engine = engine
        self._rate_limiter = RateLimiter(
            max_requests=rate_limit,
            window_seconds=rate_window,
        )
        self._running = False
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "notifications/initialized": self._handle_initialized,
        }
        self._tool_handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            "search_memories": self._handle_search_memories,
            "add_memory": self._handle_add_memory,
            "record_outcome": self._handle_record_outcome,
            "get_context": self._handle_get_context,
            "update_memory": self._handle_update_memory,
            "delete_memory": self._handle_delete_memory,
            "list_memories": self._handle_list_memories,
            "get_stats": self._handle_get_stats,
        }

    def _ensure_engine(self) -> MemoryEngine:
        """Get or create the engine."""
        if self.engine is None:
            import os
            from pathlib import Path

            db_path = os.environ.get(
                "MEMORY_LAYER_DB",
                str(Path.home() / ".memory-layer" / "memories.db"),
            )
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.engine = MemoryEngine(db_path=db_path)
        return self.engine

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP request.

        Args:
            request: The MCP request to handle

        Returns:
            MCP response
        """
        # Check rate limit
        if not self._rate_limiter.is_allowed():
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCode.RATE_LIMITED,
                    message="Rate limit exceeded",
                ),
            )

        # Find handler
        handler = self._handlers.get(request.method)
        if handler is None:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCode.METHOD_NOT_FOUND,
                    message=f"Method not found: {request.method}",
                ),
            )

        try:
            result = await handler(request.params or {})
            return MCPResponse(id=request.id, result=result)
        except ValueError as e:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCode.VALIDATION_ERROR,
                    message=str(e),
                ),
            )
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCode.INTERNAL_ERROR,
                    message=str(e),
                ),
            )

    # -------------------------------------------------------------------------
    # Protocol Handlers
    # -------------------------------------------------------------------------

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "memory-layer",
                "version": "2.0.0",
            },
        }

    async def _handle_initialized(self, params: dict[str, Any]) -> None:
        """Handle initialized notification."""
        logger.info("MCP client initialized")
        return None

    async def _handle_list_tools(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": [schema.to_dict() for schema in TOOL_SCHEMAS],
        }

    async def _handle_call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        if not tool_name:
            raise ValueError("Tool name is required")

        tool_handler = self._tool_handlers.get(tool_name)
        if tool_handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        arguments = params.get("arguments", {})
        result = await tool_handler(arguments)

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2, default=str),
                }
            ],
        }

    # -------------------------------------------------------------------------
    # Tool Handlers
    # -------------------------------------------------------------------------

    async def _handle_search_memories(
        self, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle search_memories tool."""
        engine = self._ensure_engine()

        # Validate inputs
        query = validate_string(args.get("query"), "query", min_length=1)
        limit = validate_integer(args.get("limit"), "limit", 1, 100, default=10)
        min_score = validate_float(
            args.get("min_score"), "min_score", -1.0, 1.0, default=-1.0
        )
        project = validate_string(
            args.get("project"), "project", required=False
        )

        # Parse categories
        categories = None
        if args.get("categories"):
            cat_list = validate_list(
                args.get("categories"), "categories", str, max_items=16
            )
            if cat_list:
                categories = [MemoryCategory(c) for c in cat_list]

        # Execute search - engine only supports single category
        category = categories[0] if categories else None
        results = await engine.search(
            query=query,
            limit=limit,
            category=category,
            project=project,
            min_score=min_score,
        )

        return {
            "count": len(results),
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "category": r.memory.category.value,
                    "score": r.score,
                    "outcome_score": r.memory.outcome_score,
                    "use_count": r.memory.use_count,
                }
                for r in results
            ],
        }

    async def _handle_add_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle add_memory tool."""
        engine = self._ensure_engine()

        # Validate inputs
        content = validate_string(args.get("content"), "content", min_length=1)
        category = validate_enum(
            args.get("category"),
            "category",
            MemoryCategory,
            required=False,
            default=MemoryCategory.GENERAL,
        )
        project = validate_string(args.get("project"), "project", required=False)
        tags = validate_list(args.get("tags"), "tags", str, max_items=20) or []
        importance = validate_float(
            args.get("importance"), "importance", 0.0, 1.0, default=0.5
        )

        # Add memory
        memory = await engine.add(
            content=content,
            category=category,
            project=project,
            tags=tags,
            importance=importance,
            source=MemorySource.EXPLICIT,
        )

        return {
            "id": memory.id,
            "content": memory.content,
            "category": memory.category.value,
            "created_at": memory.created_at.isoformat(),
        }

    async def _handle_record_outcome(
        self, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle record_outcome tool."""
        engine = self._ensure_engine()

        # Validate inputs
        memory_ids = validate_list(
            args.get("memory_ids"), "memory_ids", str, min_items=1, max_items=50
        )
        outcome = validate_enum(args.get("outcome"), "outcome", Outcome)

        # Record outcome
        success = await engine.record_outcome(
            memory_ids=memory_ids,
            outcome=outcome,
        )

        adjustment = {
            Outcome.WORKED: "+0.2",
            Outcome.FAILED: "-0.3",
            Outcome.PARTIAL: "+0.05",
        }[outcome]

        return {
            "success": success,
            "memory_ids": memory_ids,
            "outcome": outcome.value,
            "adjustment": adjustment,
        }

    async def _handle_get_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle get_context tool."""
        engine = self._ensure_engine()

        # Validate inputs
        project = validate_string(args.get("project"), "project", required=False)
        limit = validate_integer(args.get("limit"), "limit", 1, 50, default=10)
        format_style = args.get("format", "markdown")

        # Get context
        context = await engine.get_context(project=project, max_memories=limit)

        # Format output
        from memory_layer.plugin import ContextFormatter

        formatted = ContextFormatter.format_for_injection(
            context.memories,
            style=format_style if format_style in ["brief", "detailed", "markdown"] else "markdown",
        )

        return {
            "project": project,
            "total_count": context.total_count,
            "included_count": context.included_count,
            "formatted": formatted,
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "category": m.category.value,
                    "outcome_score": m.outcome_score,
                }
                for m in context.memories
            ],
        }

    async def _handle_update_memory(
        self, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle update_memory tool."""
        engine = self._ensure_engine()

        # Validate inputs
        memory_id = validate_string(args.get("id"), "id", min_length=1)
        content = validate_string(
            args.get("content"), "content", required=False
        )
        category = validate_enum(
            args.get("category"),
            "category",
            MemoryCategory,
            required=False,
        )
        tags = validate_list(args.get("tags"), "tags", str, max_items=20)

        # Check memory exists
        existing = await engine.get(memory_id)
        if existing is None:
            raise ValueError(f"Memory not found: {memory_id}")

        # Update memory with provided fields
        updated = await engine.update(
            memory_id=memory_id,
            content=content,
            category=category,
            tags=tags,
        )

        return {
            "id": updated.id,
            "content": updated.content,
            "category": updated.category.value,
            "updated_at": updated.updated_at.isoformat(),
        }

    async def _handle_delete_memory(
        self, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle delete_memory tool."""
        from memory_layer.core.engine import MemoryNotFoundError

        engine = self._ensure_engine()

        # Validate inputs
        memory_id = validate_string(args.get("id"), "id", min_length=1)

        # Delete (archive) memory
        try:
            await engine.delete(memory_id)
        except MemoryNotFoundError:
            raise ValueError(f"Memory not found: {memory_id}")

        return {
            "success": True,
            "id": memory_id,
            "message": "Memory archived",
        }

    async def _handle_list_memories(
        self, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle list_memories tool."""
        engine = self._ensure_engine()

        # Validate inputs
        category = validate_enum(
            args.get("category"),
            "category",
            MemoryCategory,
            required=False,
        )
        project = validate_string(args.get("project"), "project", required=False)
        limit = validate_integer(args.get("limit"), "limit", 1, 100, default=20)
        include_archived = args.get("include_archived", False)

        # List memories
        memories = await engine.list(
            category=category,
            project=project,
            limit=limit,
            include_archived=include_archived,
        )

        return {
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                    "category": m.category.value,
                    "outcome_score": m.outcome_score,
                    "use_count": m.use_count,
                    "archived": m.archived,
                }
                for m in memories
            ],
        }

    async def _handle_get_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle get_stats tool."""
        engine = self._ensure_engine()

        # Validate inputs
        project = validate_string(args.get("project"), "project", required=False)

        # Get stats
        stats = await engine.stats(project=project)

        return stats

    # -------------------------------------------------------------------------
    # Transport
    # -------------------------------------------------------------------------

    async def run_stdio(self) -> None:
        """Run the server using stdio transport."""
        self._running = True
        logger.info("MCP server starting on stdio")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        loop = asyncio.get_event_loop()
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, reader, loop
        )

        try:
            while self._running:
                try:
                    line = await reader.readline()
                    if not line:
                        break

                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    # Parse request
                    try:
                        data = json.loads(line_str)
                        request = MCPRequest.from_dict(data)
                    except json.JSONDecodeError as e:
                        response = MCPResponse(
                            error=MCPError(
                                code=MCPErrorCode.PARSE_ERROR,
                                message=f"Parse error: {e}",
                            )
                        )
                        await self._write_response(writer, response)
                        continue
                    except KeyError as e:
                        response = MCPResponse(
                            error=MCPError(
                                code=MCPErrorCode.INVALID_REQUEST,
                                message=f"Invalid request: missing {e}",
                            )
                        )
                        await self._write_response(writer, response)
                        continue

                    # Handle request
                    response = await self.handle_request(request)

                    # Send response (only for requests with id)
                    if request.id is not None:
                        await self._write_response(writer, response)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in stdio loop: {e}")

        finally:
            self._running = False
            writer.close()
            logger.info("MCP server stopped")

    async def _write_response(
        self, writer: asyncio.StreamWriter, response: MCPResponse
    ) -> None:
        """Write a response to the output stream."""
        data = json.dumps(response.to_dict()) + "\n"
        writer.write(data.encode("utf-8"))
        await writer.drain()

    def stop(self) -> None:
        """Stop the server."""
        self._running = False


# =============================================================================
# Entry Point
# =============================================================================


async def run_mcp_server(
    engine: Optional[MemoryEngine] = None,
    rate_limit: int = 100,
) -> None:
    """Run the MCP server.

    Args:
        engine: Optional MemoryEngine instance
        rate_limit: Maximum requests per minute
    """
    server = MCPServer(engine=engine, rate_limit=rate_limit)
    await server.run_stdio()


def main() -> None:
    """Main entry point for MCP server."""
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()

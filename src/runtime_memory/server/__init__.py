"""Runtime Memory Server Module.

This module provides server implementations for multi-agent access:
- MCP (Model Context Protocol) server for AI agent communication
- REST API server for HTTP access
"""

from runtime_memory.server.api import (
    APIConfig,
    ContextResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    # Request models
    MemoryCreateRequest,
    MemoryListResponse,
    # Response models
    MemoryResponse,
    MemoryUpdateRequest,
    OutcomeRequest,
    OutcomeResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    StatsResponse,
    create_app,
    run_server,
)
from runtime_memory.server.mcp import (
    TOOL_SCHEMAS,
    MCPError,
    MCPErrorCode,
    MCPRequest,
    MCPResponse,
    MCPServer,
    MCPToolSchema,
    run_mcp_server,
)
from runtime_memory.server.mcp import (
    RateLimiter as MCPRateLimiter,
)

__all__ = [
    # MCP Types
    "MCPError",
    "MCPErrorCode",
    "MCPRequest",
    "MCPResponse",
    "MCPToolSchema",
    # MCP Server
    "MCPServer",
    "MCPRateLimiter",
    "TOOL_SCHEMAS",
    "run_mcp_server",
    # REST API
    "APIConfig",
    "create_app",
    "run_server",
    # Request models
    "MemoryCreateRequest",
    "MemoryUpdateRequest",
    "SearchRequest",
    "OutcomeRequest",
    "IngestRequest",
    # Response models
    "MemoryResponse",
    "SearchResultResponse",
    "SearchResponse",
    "MemoryListResponse",
    "ContextResponse",
    "StatsResponse",
    "HealthResponse",
    "ErrorResponse",
    "OutcomeResponse",
]

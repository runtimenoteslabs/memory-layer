"""Memory Layer Server Module.

This module provides server implementations for multi-agent access:
- MCP (Model Context Protocol) server for AI agent communication
- REST API server for HTTP access
"""

from memory_layer.server.mcp import (
    MCPError,
    MCPErrorCode,
    MCPRequest,
    MCPResponse,
    MCPServer,
    MCPToolSchema,
    RateLimiter as MCPRateLimiter,
    TOOL_SCHEMAS,
    run_mcp_server,
)

from memory_layer.server.api import (
    APIConfig,
    create_app,
    run_server,
    # Request models
    MemoryCreateRequest,
    MemoryUpdateRequest,
    SearchRequest,
    OutcomeRequest,
    IngestRequest,
    # Response models
    MemoryResponse,
    SearchResultResponse,
    SearchResponse,
    MemoryListResponse,
    ContextResponse,
    StatsResponse,
    HealthResponse,
    ErrorResponse,
    OutcomeResponse,
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

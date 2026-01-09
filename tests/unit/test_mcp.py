"""Unit tests for MCP Server.

Tests for:
- MCP protocol types (MCPRequest, MCPResponse, MCPError)
- Tool schemas
- Input validators
- Rate limiter
- MCPServer and all tool handlers
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_layer.core.models import (
    ContextResponse,
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
    SearchResult,
)
from memory_layer.server.mcp import (
    MCPError,
    MCPErrorCode,
    MCPRequest,
    MCPResponse,
    MCPServer,
    MCPToolSchema,
    RateLimiter,
    TOOL_SCHEMAS,
    validate_enum,
    validate_float,
    validate_integer,
    validate_list,
    validate_string,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        id="test-mem-001",
        content="Use snake_case for Python variables",
        category=MemoryCategory.CONVENTION,
        outcome_score=0.5,
        use_count=3,
        confidence=0.9,
        project="test-project",
        scope=MemoryScope.PROJECT,
        source=MemorySource.EXPLICIT,
        tags=["python", "naming"],
    )


@pytest.fixture
def sample_memories():
    """Create a list of sample memories for testing."""
    return [
        Memory(
            id="mem-001",
            content="Use snake_case for Python variables",
            category=MemoryCategory.CONVENTION,
            outcome_score=0.5,
            use_count=3,
        ),
        Memory(
            id="mem-002",
            content="PostgreSQL for database",
            category=MemoryCategory.ARCHITECTURE,
            outcome_score=0.8,
            use_count=5,
        ),
    ]


@pytest.fixture
def mock_engine(sample_memory, sample_memories):
    """Create a mock engine with common methods."""
    engine = MagicMock()

    # Configure async methods
    engine.add = AsyncMock(return_value=sample_memory)
    engine.get = AsyncMock(return_value=sample_memory)
    engine.update = AsyncMock(return_value=sample_memory)
    engine.delete = AsyncMock(return_value=True)
    engine.search = AsyncMock(return_value=[
        SearchResult(memory=m, score=0.9 - i * 0.1)
        for i, m in enumerate(sample_memories)
    ])
    engine.list = AsyncMock(return_value=sample_memories)
    engine.record_outcome = AsyncMock(return_value=True)
    engine.get_context = AsyncMock(return_value=ContextResponse(
        memories=sample_memories,
        project="test-project",
        total_count=2,
        included_count=2,
    ))
    engine.stats = AsyncMock(return_value={
        "total_memories": 10,
        "active_memories": 8,
        "archived_memories": 2,
        "avg_outcome_score": 0.35,
        "total_uses": 25,
        "by_category": {"convention": 3, "architecture": 2},
    })

    return engine


@pytest.fixture
def mcp_server(mock_engine):
    """Create an MCP server with mock engine."""
    return MCPServer(engine=mock_engine)


# =============================================================================
# MCP Protocol Types Tests
# =============================================================================


class TestMCPError:
    """Tests for MCPError."""

    def test_create_error(self):
        """Test creating an MCP error."""
        error = MCPError(
            code=MCPErrorCode.INTERNAL_ERROR,
            message="Something went wrong",
        )

        assert error.code == MCPErrorCode.INTERNAL_ERROR
        assert error.message == "Something went wrong"
        assert error.data is None

    def test_create_error_with_data(self):
        """Test creating an MCP error with data."""
        error = MCPError(
            code=MCPErrorCode.VALIDATION_ERROR,
            message="Invalid input",
            data={"field": "query", "reason": "too short"},
        )

        assert error.data["field"] == "query"

    def test_error_to_dict(self):
        """Test converting error to dictionary."""
        error = MCPError(
            code=-32600,
            message="Invalid request",
        )

        result = error.to_dict()

        assert result["code"] == -32600
        assert result["message"] == "Invalid request"
        assert "data" not in result

    def test_error_to_dict_with_data(self):
        """Test converting error with data to dictionary."""
        error = MCPError(
            code=-32602,
            message="Invalid params",
            data={"details": "missing query"},
        )

        result = error.to_dict()

        assert result["data"]["details"] == "missing query"


class TestMCPRequest:
    """Tests for MCPRequest."""

    def test_create_request(self):
        """Test creating an MCP request."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/list",
            id=1,
        )

        assert request.jsonrpc == "2.0"
        assert request.method == "tools/list"
        assert request.id == 1
        assert request.params is None

    def test_create_request_with_params(self):
        """Test creating an MCP request with params."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id="req-123",
            params={"name": "search_memories", "arguments": {"query": "test"}},
        )

        assert request.params["name"] == "search_memories"

    def test_request_from_dict(self):
        """Test creating request from dictionary."""
        data = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1,
            "params": {"protocolVersion": "2024-11-05"},
        }

        request = MCPRequest.from_dict(data)

        assert request.method == "initialize"
        assert request.id == 1
        assert request.params["protocolVersion"] == "2024-11-05"

    def test_request_from_dict_minimal(self):
        """Test creating request from minimal dictionary."""
        data = {"method": "tools/list"}

        request = MCPRequest.from_dict(data)

        assert request.method == "tools/list"
        assert request.jsonrpc == "2.0"
        assert request.id is None


class TestMCPResponse:
    """Tests for MCPResponse."""

    def test_create_success_response(self):
        """Test creating a success response."""
        response = MCPResponse(
            id=1,
            result={"tools": []},
        )

        assert response.id == 1
        assert response.result["tools"] == []
        assert response.error is None

    def test_create_error_response(self):
        """Test creating an error response."""
        response = MCPResponse(
            id=1,
            error=MCPError(code=-32600, message="Invalid request"),
        )

        assert response.error is not None
        assert response.result is None

    def test_response_to_dict_success(self):
        """Test converting success response to dictionary."""
        response = MCPResponse(
            id=1,
            result={"data": "test"},
        )

        result = response.to_dict()

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["result"]["data"] == "test"
        assert "error" not in result

    def test_response_to_dict_error(self):
        """Test converting error response to dictionary."""
        response = MCPResponse(
            id=1,
            error=MCPError(code=-32600, message="Error"),
        )

        result = response.to_dict()

        assert "error" in result
        assert result["error"]["code"] == -32600
        assert "result" not in result


class TestMCPToolSchema:
    """Tests for MCPToolSchema."""

    def test_create_schema(self):
        """Test creating a tool schema."""
        schema = MCPToolSchema(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )

        assert schema.name == "test_tool"
        assert schema.description == "A test tool"

    def test_schema_to_dict(self):
        """Test converting schema to dictionary."""
        schema = MCPToolSchema(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
        )

        result = schema.to_dict()

        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool"
        assert result["inputSchema"]["type"] == "object"


# =============================================================================
# Tool Schemas Tests
# =============================================================================


class TestToolSchemas:
    """Tests for predefined tool schemas."""

    def test_all_tools_defined(self):
        """Test that all required tools are defined."""
        tool_names = {s.name for s in TOOL_SCHEMAS}

        required_tools = {
            "search_memories",
            "add_memory",
            "record_outcome",
            "get_context",
            "update_memory",
            "delete_memory",
            "list_memories",
            "get_stats",
        }

        assert required_tools.issubset(tool_names)

    def test_search_memories_schema(self):
        """Test search_memories schema."""
        schema = next(s for s in TOOL_SCHEMAS if s.name == "search_memories")

        assert "query" in schema.input_schema["properties"]
        assert "query" in schema.input_schema["required"]

    def test_add_memory_schema(self):
        """Test add_memory schema."""
        schema = next(s for s in TOOL_SCHEMAS if s.name == "add_memory")

        assert "content" in schema.input_schema["properties"]
        assert "category" in schema.input_schema["properties"]
        assert "content" in schema.input_schema["required"]

    def test_record_outcome_schema(self):
        """Test record_outcome schema."""
        schema = next(s for s in TOOL_SCHEMAS if s.name == "record_outcome")

        assert "memory_ids" in schema.input_schema["properties"]
        assert "outcome" in schema.input_schema["properties"]
        outcome_enum = schema.input_schema["properties"]["outcome"]["enum"]
        assert "worked" in outcome_enum
        assert "failed" in outcome_enum
        assert "partial" in outcome_enum

    def test_all_schemas_have_description(self):
        """Test that all schemas have descriptions."""
        for schema in TOOL_SCHEMAS:
            assert schema.description, f"{schema.name} missing description"
            assert len(schema.description) > 10

    def test_all_schemas_have_input_schema(self):
        """Test that all schemas have input schemas."""
        for schema in TOOL_SCHEMAS:
            assert schema.input_schema, f"{schema.name} missing input_schema"
            assert schema.input_schema.get("type") == "object"


# =============================================================================
# Input Validator Tests
# =============================================================================


class TestValidateString:
    """Tests for validate_string."""

    def test_valid_string(self):
        """Test valid string."""
        result = validate_string("hello", "test")
        assert result == "hello"

    def test_required_missing(self):
        """Test required string missing."""
        with pytest.raises(ValueError, match="is required"):
            validate_string(None, "test", required=True)

    def test_optional_missing(self):
        """Test optional string missing."""
        result = validate_string(None, "test", required=False)
        assert result is None

    def test_not_string(self):
        """Test non-string value."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_string(123, "test")

    def test_too_short(self):
        """Test string too short."""
        with pytest.raises(ValueError, match="at least"):
            validate_string("ab", "test", min_length=5)

    def test_too_long(self):
        """Test string too long."""
        with pytest.raises(ValueError, match="at most"):
            validate_string("a" * 100, "test", max_length=50)


class TestValidateInteger:
    """Tests for validate_integer."""

    def test_valid_integer(self):
        """Test valid integer."""
        result = validate_integer(10, "test", minimum=1, maximum=100)
        assert result == 10

    def test_default_value(self):
        """Test default value."""
        result = validate_integer(None, "test", default=5)
        assert result == 5

    def test_not_integer(self):
        """Test non-integer value."""
        with pytest.raises(ValueError, match="must be an integer"):
            validate_integer("10", "test")

    def test_below_minimum(self):
        """Test value below minimum."""
        with pytest.raises(ValueError, match="at least"):
            validate_integer(0, "test", minimum=1)

    def test_above_maximum(self):
        """Test value above maximum."""
        with pytest.raises(ValueError, match="at most"):
            validate_integer(200, "test", maximum=100)

    def test_boolean_rejected(self):
        """Test that booleans are rejected."""
        with pytest.raises(ValueError, match="must be an integer"):
            validate_integer(True, "test")


class TestValidateFloat:
    """Tests for validate_float."""

    def test_valid_float(self):
        """Test valid float."""
        result = validate_float(0.5, "test", minimum=0.0, maximum=1.0)
        assert result == 0.5

    def test_valid_integer_as_float(self):
        """Test integer as float."""
        result = validate_float(1, "test", minimum=0.0, maximum=1.0)
        assert result == 1.0

    def test_default_value(self):
        """Test default value."""
        result = validate_float(None, "test", default=0.5)
        assert result == 0.5

    def test_below_minimum(self):
        """Test value below minimum."""
        with pytest.raises(ValueError, match="at least"):
            validate_float(-0.5, "test", minimum=0.0)

    def test_above_maximum(self):
        """Test value above maximum."""
        with pytest.raises(ValueError, match="at most"):
            validate_float(1.5, "test", maximum=1.0)


class TestValidateEnum:
    """Tests for validate_enum."""

    def test_valid_enum(self):
        """Test valid enum value."""
        result = validate_enum("convention", "test", MemoryCategory)
        assert result == MemoryCategory.CONVENTION

    def test_required_missing(self):
        """Test required enum missing."""
        with pytest.raises(ValueError, match="is required"):
            validate_enum(None, "test", MemoryCategory, required=True)

    def test_optional_with_default(self):
        """Test optional enum with default."""
        result = validate_enum(
            None, "test", MemoryCategory,
            required=False, default=MemoryCategory.GENERAL
        )
        assert result == MemoryCategory.GENERAL

    def test_invalid_value(self):
        """Test invalid enum value."""
        with pytest.raises(ValueError, match="must be one of"):
            validate_enum("invalid", "test", MemoryCategory)


class TestValidateList:
    """Tests for validate_list."""

    def test_valid_list(self):
        """Test valid list."""
        result = validate_list(["a", "b", "c"], "test", str)
        assert result == ["a", "b", "c"]

    def test_optional_missing(self):
        """Test optional list missing."""
        result = validate_list(None, "test", required=False)
        assert result is None

    def test_not_list(self):
        """Test non-list value."""
        with pytest.raises(ValueError, match="must be a list"):
            validate_list("not a list", "test")

    def test_too_few_items(self):
        """Test list with too few items."""
        with pytest.raises(ValueError, match="at least"):
            validate_list([], "test", min_items=1)

    def test_too_many_items(self):
        """Test list with too many items."""
        with pytest.raises(ValueError, match="at most"):
            validate_list(list(range(100)), "test", max_items=10)

    def test_wrong_item_type(self):
        """Test list with wrong item type."""
        with pytest.raises(ValueError, match="must be a str"):
            validate_list([1, 2, 3], "test", item_type=str)


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allows_initial_requests(self):
        """Test that initial requests are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            assert limiter.is_allowed() is True

    def test_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        for _ in range(3):
            limiter.is_allowed()

        assert limiter.is_allowed() is False

    def test_different_clients(self):
        """Test rate limiting per client."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # Different client should have its own limit
        assert limiter.is_allowed("client2") is True

    def test_reset(self):
        """Test resetting rate limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.is_allowed()
        limiter.is_allowed()
        assert limiter.is_allowed() is False

        limiter.reset()
        assert limiter.is_allowed() is True

    def test_window_expiry(self):
        """Test that old requests expire."""
        limiter = RateLimiter(max_requests=2, window_seconds=0.1)

        limiter.is_allowed()
        limiter.is_allowed()
        assert limiter.is_allowed() is False

        # Wait for window to expire
        time.sleep(0.15)
        assert limiter.is_allowed() is True


# =============================================================================
# MCP Server Tests
# =============================================================================


class TestMCPServer:
    """Tests for MCPServer."""

    def test_create_server(self, mock_engine):
        """Test creating an MCP server."""
        server = MCPServer(engine=mock_engine)

        assert server.engine == mock_engine

    def test_create_server_without_engine(self):
        """Test creating server without engine."""
        server = MCPServer()

        assert server.engine is None

    @pytest.mark.asyncio
    async def test_handle_initialize(self, mcp_server):
        """Test handling initialize request."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="initialize",
            id=1,
            params={"protocolVersion": "2024-11-05"},
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        assert response.result["protocolVersion"] == "2024-11-05"
        assert response.result["serverInfo"]["name"] == "memory-layer"

    @pytest.mark.asyncio
    async def test_handle_list_tools(self, mcp_server):
        """Test handling tools/list request."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/list",
            id=1,
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        assert "tools" in response.result
        tool_names = {t["name"] for t in response.result["tools"]}
        assert "search_memories" in tool_names
        assert "add_memory" in tool_names

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, mcp_server):
        """Test handling unknown method."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="unknown/method",
            id=1,
        )

        response = await mcp_server.handle_request(request)

        assert response.error is not None
        assert response.error.code == MCPErrorCode.METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_engine):
        """Test rate limiting."""
        server = MCPServer(engine=mock_engine, rate_limit=2, rate_window=60)

        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/list",
            id=1,
        )

        # First two should succeed
        await server.handle_request(request)
        await server.handle_request(request)

        # Third should be rate limited
        response = await server.handle_request(request)
        assert response.error is not None
        assert response.error.code == MCPErrorCode.RATE_LIMITED


# =============================================================================
# Tool Handler Tests
# =============================================================================


class TestSearchMemoriesTool:
    """Tests for search_memories tool handler."""

    @pytest.mark.asyncio
    async def test_search_basic(self, mcp_server, mock_engine):
        """Test basic search."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "search_memories",
                "arguments": {"query": "python variables"},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        mock_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_options(self, mcp_server, mock_engine):
        """Test search with all options."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "search_memories",
                "arguments": {
                    "query": "test",
                    "limit": 5,
                    "categories": ["convention", "pattern"],
                    "project": "myproject",
                    "min_score": 0.5,
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["project"] == "myproject"

    @pytest.mark.asyncio
    async def test_search_missing_query(self, mcp_server):
        """Test search with missing query."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "search_memories",
                "arguments": {},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is not None
        assert response.error.code == MCPErrorCode.VALIDATION_ERROR


class TestAddMemoryTool:
    """Tests for add_memory tool handler."""

    @pytest.mark.asyncio
    async def test_add_basic(self, mcp_server, mock_engine):
        """Test basic memory addition."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "add_memory",
                "arguments": {
                    "content": "Test memory content",
                    "category": "convention",
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        mock_engine.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_with_options(self, mcp_server, mock_engine):
        """Test memory addition with all options."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "add_memory",
                "arguments": {
                    "content": "Test content",
                    "category": "architecture",
                    "project": "myproject",
                    "tags": ["tag1", "tag2"],
                    "importance": 0.8,
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        call_kwargs = mock_engine.add.call_args[1]
        assert call_kwargs["project"] == "myproject"
        assert call_kwargs["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_add_missing_content(self, mcp_server):
        """Test add with missing content."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "add_memory",
                "arguments": {"category": "general"},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is not None


class TestRecordOutcomeTool:
    """Tests for record_outcome tool handler."""

    @pytest.mark.asyncio
    async def test_record_worked(self, mcp_server, mock_engine):
        """Test recording worked outcome."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "record_outcome",
                "arguments": {
                    "memory_ids": ["mem-001"],
                    "outcome": "worked",
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        mock_engine.record_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_multiple_memories(self, mcp_server, mock_engine):
        """Test recording outcome for multiple memories."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "record_outcome",
                "arguments": {
                    "memory_ids": ["mem-001", "mem-002", "mem-003"],
                    "outcome": "partial",
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None

    @pytest.mark.asyncio
    async def test_record_invalid_outcome(self, mcp_server):
        """Test recording invalid outcome."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "record_outcome",
                "arguments": {
                    "memory_ids": ["mem-001"],
                    "outcome": "invalid",
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is not None


class TestGetContextTool:
    """Tests for get_context tool handler."""

    @pytest.mark.asyncio
    async def test_get_context_basic(self, mcp_server, mock_engine):
        """Test basic context retrieval."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "get_context",
                "arguments": {},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        mock_engine.get_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_with_project(self, mcp_server, mock_engine):
        """Test context with project."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "get_context",
                "arguments": {
                    "project": "myproject",
                    "limit": 5,
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None


class TestUpdateMemoryTool:
    """Tests for update_memory tool handler."""

    @pytest.mark.asyncio
    async def test_update_content(self, mcp_server, mock_engine):
        """Test updating memory content."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "update_memory",
                "arguments": {
                    "id": "test-mem-001",
                    "content": "Updated content",
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        mock_engine.get.assert_called_once_with("test-mem-001")
        mock_engine.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_not_found(self, mcp_server, mock_engine):
        """Test updating non-existent memory."""
        mock_engine.get = AsyncMock(return_value=None)

        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "update_memory",
                "arguments": {"id": "nonexistent"},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is not None
        assert "not found" in response.error.message.lower()


class TestDeleteMemoryTool:
    """Tests for delete_memory tool handler."""

    @pytest.mark.asyncio
    async def test_delete_memory(self, mcp_server, mock_engine):
        """Test deleting a memory."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "delete_memory",
                "arguments": {"id": "test-mem-001"},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        mock_engine.delete.assert_called_once_with("test-mem-001")

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mcp_server, mock_engine):
        """Test deleting non-existent memory."""
        mock_engine.delete = AsyncMock(return_value=False)

        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "delete_memory",
                "arguments": {"id": "nonexistent"},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is not None


class TestListMemoriesTool:
    """Tests for list_memories tool handler."""

    @pytest.mark.asyncio
    async def test_list_basic(self, mcp_server, mock_engine):
        """Test basic listing."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "list_memories",
                "arguments": {},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        mock_engine.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_with_filters(self, mcp_server, mock_engine):
        """Test listing with filters."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "list_memories",
                "arguments": {
                    "category": "convention",
                    "project": "myproject",
                    "limit": 10,
                },
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        call_kwargs = mock_engine.list.call_args[1]
        assert call_kwargs["category"] == MemoryCategory.CONVENTION


class TestGetStatsTool:
    """Tests for get_stats tool handler."""

    @pytest.mark.asyncio
    async def test_get_stats(self, mcp_server, mock_engine):
        """Test getting stats."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "get_stats",
                "arguments": {},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        mock_engine.stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats_with_project(self, mcp_server, mock_engine):
        """Test getting stats with project filter."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "get_stats",
                "arguments": {"project": "myproject"},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is None
        call_kwargs = mock_engine.stats.call_args[1]
        assert call_kwargs["project"] == "myproject"


# =============================================================================
# Unknown Tool Tests
# =============================================================================


class TestUnknownTool:
    """Tests for unknown tool handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool(self, mcp_server):
        """Test calling unknown tool."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            id=1,
            params={
                "name": "unknown_tool",
                "arguments": {},
            },
        )

        response = await mcp_server.handle_request(request)

        assert response.error is not None
        assert "unknown" in response.error.message.lower()

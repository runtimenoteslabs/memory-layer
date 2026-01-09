"""Unit tests for custom commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_layer.claude_code.commands import (
    CommandConfig,
    CommandHandler,
    CommandResult,
    CommandType,
    export_command_schemas,
    get_command_schemas,
)
from memory_layer.core.models import (
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
    SearchResult,
)


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful result."""
        result = CommandResult(
            success=True,
            command=CommandType.REMEMBER,
            message="Memory stored",
            data={"memory_id": "abc123"},
        )

        assert result.success is True
        assert result.command == CommandType.REMEMBER
        assert result.message == "Memory stored"
        assert result.data == {"memory_id": "abc123"}
        assert result.error is None

    def test_create_error_result(self) -> None:
        """Test creating an error result."""
        result = CommandResult(
            success=False,
            command=CommandType.RECALL,
            message="Search failed",
            error="Database connection error",
        )

        assert result.success is False
        assert result.command == CommandType.RECALL
        assert result.error == "Database connection error"

    def test_to_dict_success(self) -> None:
        """Test converting success result to dict."""
        result = CommandResult(
            success=True,
            command=CommandType.CONTEXT,
            message="Context retrieved",
            data={"context": "test content"},
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["command"] == "context"
        assert d["message"] == "Context retrieved"
        assert d["data"] == {"context": "test content"}
        assert "error" not in d

    def test_to_dict_error(self) -> None:
        """Test converting error result to dict."""
        result = CommandResult(
            success=False,
            command=CommandType.FORGET,
            message="Delete failed",
            error="Memory not found",
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "Memory not found"

    def test_to_json(self) -> None:
        """Test converting to JSON."""
        result = CommandResult(
            success=True,
            command=CommandType.MEMORIES,
            message="Listed",
            data={"count": 5},
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert parsed["command"] == "memories"
        assert parsed["data"]["count"] == 5

    def test_to_markdown_success(self) -> None:
        """Test converting success to markdown."""
        result = CommandResult(
            success=True,
            command=CommandType.REMEMBER,
            message="Memory stored",
        )

        md = result.to_markdown()

        assert "✅" in md
        assert "Remember" in md
        assert "Memory stored" in md

    def test_to_markdown_error(self) -> None:
        """Test converting error to markdown."""
        result = CommandResult(
            success=False,
            command=CommandType.RECALL,
            message="Failed",
            error="Something went wrong",
        )

        md = result.to_markdown()

        assert "❌" in md
        assert "Error" in md
        assert "Something went wrong" in md

    def test_to_markdown_with_memories(self) -> None:
        """Test markdown with memories list."""
        result = CommandResult(
            success=True,
            command=CommandType.RECALL,
            message="Found 2 memories",
            data={
                "memories": [
                    {"category": "gotcha", "content": "First memory", "score": 0.95},
                    {"category": "pattern", "content": "Second memory", "score": 0.80},
                ]
            },
        )

        md = result.to_markdown()

        assert "gotcha" in md
        assert "0.95" in md
        assert "pattern" in md

    def test_to_markdown_with_context(self) -> None:
        """Test markdown with context data."""
        result = CommandResult(
            success=True,
            command=CommandType.CONTEXT,
            message="Context for project",
            data={"context": "## Memories\n- Item 1\n- Item 2"},
        )

        md = result.to_markdown()

        assert "## Memories" in md


class TestCommandConfig:
    """Tests for CommandConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CommandConfig()

        assert config.default_project is None
        assert config.default_scope == MemoryScope.PROJECT
        assert config.max_results == 10
        assert config.include_archived is False
        assert config.output_format == "markdown"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CommandConfig(
            default_project="my-project",
            default_scope=MemoryScope.GLOBAL,
            max_results=20,
            include_archived=True,
            output_format="json",
        )

        assert config.default_project == "my-project"
        assert config.default_scope == MemoryScope.GLOBAL
        assert config.max_results == 20
        assert config.include_archived is True
        assert config.output_format == "json"


class TestCommandHandler:
    """Tests for CommandHandler."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create a mock memory engine."""
        engine = MagicMock()
        engine.add = AsyncMock()
        engine.get = AsyncMock()
        engine.update = AsyncMock()
        engine.delete = AsyncMock()
        engine.search = AsyncMock()
        engine.list = AsyncMock()
        engine.record_outcome = AsyncMock()
        engine.get_context = AsyncMock()
        engine.stats = AsyncMock()
        return engine

    @pytest.fixture
    def handler(self, mock_engine: MagicMock) -> CommandHandler:
        """Create a command handler with mock engine."""
        return CommandHandler(engine=mock_engine)

    @pytest.fixture
    def sample_memory(self) -> Memory:
        """Create a sample memory."""
        return Memory(
            id="test-123",
            content="Test memory content",
            category=MemoryCategory.PATTERN,
            scope=MemoryScope.PROJECT,
            source=MemorySource.EXPLICIT,
            project="test-project",
            outcome_score=0.5,
            use_count=5,
        )

    # Remember command tests

    @pytest.mark.asyncio
    async def test_remember_basic(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test basic remember command."""
        mock_engine.add.return_value = sample_memory

        result = await handler.execute("remember", "Test memory content", "test-project")

        assert result.success is True
        assert result.command == CommandType.REMEMBER
        mock_engine.add.assert_called_once()
        call_args = mock_engine.add.call_args
        assert call_args.kwargs["content"] == "Test memory content"

    @pytest.mark.asyncio
    async def test_remember_with_category(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test remember with category prefix."""
        sample_memory.category = MemoryCategory.GOTCHA
        mock_engine.add.return_value = sample_memory

        result = await handler.execute(
            "/remember", "category:gotcha Docker needs more memory", "test-project"
        )

        assert result.success is True
        call_args = mock_engine.add.call_args
        assert call_args.kwargs["category"] == MemoryCategory.GOTCHA
        assert call_args.kwargs["content"] == "Docker needs more memory"

    @pytest.mark.asyncio
    async def test_remember_category_partial_match(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test remember with partial category name."""
        sample_memory.category = MemoryCategory.TROUBLESHOOTING
        mock_engine.add.return_value = sample_memory

        result = await handler.execute(
            "remember", "category:TROUBLE Something went wrong", "test-project"
        )

        assert result.success is True
        call_args = mock_engine.add.call_args
        assert call_args.kwargs["category"] == MemoryCategory.TROUBLESHOOTING

    @pytest.mark.asyncio
    async def test_remember_empty_content(self, handler: CommandHandler) -> None:
        """Test remember with no content."""
        result = await handler.execute("remember", "", "test-project")

        assert result.success is False
        assert "Usage" in result.error

    @pytest.mark.asyncio
    async def test_remember_with_leading_slash(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test command with leading slash."""
        mock_engine.add.return_value = sample_memory

        result = await handler.execute("/remember", "Test content", "test-project")

        assert result.success is True

    # Recall command tests

    @pytest.mark.asyncio
    async def test_recall_basic(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test basic recall/search."""
        search_result = SearchResult(memory=sample_memory, score=0.95)
        mock_engine.search.return_value = [search_result]

        result = await handler.execute("recall", "docker memory", "test-project")

        assert result.success is True
        assert result.command == CommandType.RECALL
        assert len(result.data["memories"]) == 1
        mock_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_with_limit(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test recall with limit parameter."""
        mock_engine.search.return_value = []

        result = await handler.execute("recall", "test query limit:5", "test-project")

        assert result.success is True
        call_args = mock_engine.search.call_args
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["query"] == "test query"

    @pytest.mark.asyncio
    async def test_recall_no_results(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test recall with no results."""
        mock_engine.search.return_value = []

        result = await handler.execute("recall", "nonexistent", "test-project")

        assert result.success is True
        assert "No memories found" in result.message
        assert result.data["memories"] == []

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, handler: CommandHandler) -> None:
        """Test recall with empty query."""
        result = await handler.execute("recall", "", "test-project")

        assert result.success is False
        assert "Usage" in result.error

    @pytest.mark.asyncio
    async def test_recall_limit_capped(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test that limit is capped at 50."""
        mock_engine.search.return_value = []

        result = await handler.execute("recall", "test limit:100", "test-project")

        call_args = mock_engine.search.call_args
        assert call_args.kwargs["limit"] == 50

    # Forget command tests

    @pytest.mark.asyncio
    async def test_forget_archive(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test forget (archive) command."""
        mock_engine.get.return_value = sample_memory

        result = await handler.execute("forget", "test-123", "test-project")

        assert result.success is True
        assert result.command == CommandType.FORGET
        mock_engine.delete.assert_called_once_with("test-123", permanent=False)

    @pytest.mark.asyncio
    async def test_forget_permanent(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test permanent delete."""
        mock_engine.get.return_value = sample_memory

        result = await handler.execute("forget", "test-123 --permanent", "test-project")

        assert result.success is True
        assert result.data["permanent"] is True
        mock_engine.delete.assert_called_once_with("test-123", permanent=True)

    @pytest.mark.asyncio
    async def test_forget_not_found(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test forget with non-existent memory."""
        mock_engine.get.return_value = None

        result = await handler.execute("forget", "nonexistent", "test-project")

        assert result.success is False
        assert "no memory found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_forget_no_id(self, handler: CommandHandler) -> None:
        """Test forget without memory ID."""
        result = await handler.execute("forget", "", "test-project")

        assert result.success is False
        assert "Usage" in result.error

    # Outcome command tests

    @pytest.mark.asyncio
    async def test_outcome_worked(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test recording successful outcome."""
        mock_engine.get.return_value = sample_memory
        updated_memory = Memory(
            id=sample_memory.id,
            content=sample_memory.content,
            category=sample_memory.category,
            scope=sample_memory.scope,
            source=sample_memory.source,
            outcome_score=0.7,
        )
        mock_engine.record_outcome.return_value = updated_memory

        result = await handler.execute("outcome", "test-123 worked", "test-project")

        assert result.success is True
        assert result.command == CommandType.OUTCOME
        mock_engine.record_outcome.assert_called_once()
        call_args = mock_engine.record_outcome.call_args
        assert call_args.kwargs["outcome"] == Outcome.WORKED

    @pytest.mark.asyncio
    async def test_outcome_failed(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test recording failed outcome."""
        mock_engine.get.return_value = sample_memory
        mock_engine.record_outcome.return_value = sample_memory

        result = await handler.execute("outcome", "test-123 failed", "test-project")

        assert result.success is True
        call_args = mock_engine.record_outcome.call_args
        assert call_args.kwargs["outcome"] == Outcome.FAILED

    @pytest.mark.asyncio
    async def test_outcome_with_notes(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test outcome with notes."""
        mock_engine.get.return_value = sample_memory
        mock_engine.record_outcome.return_value = sample_memory

        result = await handler.execute(
            "outcome", "test-123 partial The solution was outdated", "test-project"
        )

        assert result.success is True
        call_args = mock_engine.record_outcome.call_args
        assert call_args.kwargs["outcome"] == Outcome.PARTIAL
        assert call_args.kwargs["context"] == "The solution was outdated"

    @pytest.mark.asyncio
    async def test_outcome_invalid(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test invalid outcome value."""
        result = await handler.execute("outcome", "test-123 invalid", "test-project")

        assert result.success is False
        assert "worked, failed, partial" in result.error

    @pytest.mark.asyncio
    async def test_outcome_missing_args(self, handler: CommandHandler) -> None:
        """Test outcome with missing arguments."""
        result = await handler.execute("outcome", "test-123", "test-project")

        assert result.success is False
        assert "Usage" in result.error

    @pytest.mark.asyncio
    async def test_outcome_memory_not_found(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test outcome for non-existent memory."""
        mock_engine.get.return_value = None

        result = await handler.execute("outcome", "nonexistent worked", "test-project")

        assert result.success is False
        assert "no memory found" in result.error.lower()

    # Context command tests

    @pytest.mark.asyncio
    async def test_context_basic(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test basic context retrieval."""
        context_response = MagicMock()
        context_response.formatted = "## Memories\n- Memory 1\n- Memory 2"
        context_response.included_count = 2
        context_response.total_count = 2
        mock_engine.get_context.return_value = context_response

        result = await handler.execute("context", "", "test-project")

        assert result.success is True
        assert result.command == CommandType.CONTEXT
        assert "## Memories" in result.data["context"]

    @pytest.mark.asyncio
    async def test_context_with_project(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test context with specific project."""
        context_response = MagicMock()
        context_response.formatted = ""
        context_response.included_count = 0
        context_response.total_count = 0
        mock_engine.get_context.return_value = context_response

        result = await handler.execute("context", "other-project", "test-project")

        assert result.success is True
        call_args = mock_engine.get_context.call_args
        assert call_args.kwargs["project"] == "other-project"

    @pytest.mark.asyncio
    async def test_context_with_limit(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test context with limit."""
        context_response = MagicMock()
        context_response.formatted = ""
        context_response.included_count = 0
        context_response.total_count = 0
        mock_engine.get_context.return_value = context_response

        result = await handler.execute("context", "my-project limit:20", "test-project")

        call_args = mock_engine.get_context.call_args
        assert call_args.kwargs["limit"] == 20

    # Memories command tests

    @pytest.mark.asyncio
    async def test_memories_list(
        self, handler: CommandHandler, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test listing memories."""
        mock_engine.list.return_value = [sample_memory]
        stats_mock = MagicMock()
        stats_mock.total_memories = 10
        stats_mock.active_memories = 8
        stats_mock.archived_memories = 2
        stats_mock.by_category = {MemoryCategory.PATTERN: 5, MemoryCategory.GOTCHA: 3}
        mock_engine.stats.return_value = stats_mock

        result = await handler.execute("memories", "", "test-project")

        assert result.success is True
        assert result.command == CommandType.MEMORIES
        assert len(result.data["memories"]) == 1
        assert result.data["stats"]["total"] == 10

    @pytest.mark.asyncio
    async def test_memories_with_category_filter(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test memories with category filter."""
        mock_engine.list.return_value = []
        stats_mock = MagicMock()
        stats_mock.total_memories = 0
        stats_mock.active_memories = 0
        stats_mock.archived_memories = 0
        stats_mock.by_category = {}
        mock_engine.stats.return_value = stats_mock

        result = await handler.execute("memories", "category:gotcha", "test-project")

        assert result.success is True
        call_args = mock_engine.list.call_args
        assert call_args.kwargs["category"] == MemoryCategory.GOTCHA

    @pytest.mark.asyncio
    async def test_memories_with_scope_filter(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test memories with scope filter."""
        mock_engine.list.return_value = []
        stats_mock = MagicMock()
        stats_mock.total_memories = 0
        stats_mock.active_memories = 0
        stats_mock.archived_memories = 0
        stats_mock.by_category = {}
        mock_engine.stats.return_value = stats_mock

        result = await handler.execute("memories", "scope:global", "test-project")

        call_args = mock_engine.list.call_args
        assert call_args.kwargs["scope"] == MemoryScope.GLOBAL

    @pytest.mark.asyncio
    async def test_memories_with_archived_flag(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test memories with archived flag."""
        mock_engine.list.return_value = []
        stats_mock = MagicMock()
        stats_mock.total_memories = 0
        stats_mock.active_memories = 0
        stats_mock.archived_memories = 0
        stats_mock.by_category = {}
        mock_engine.stats.return_value = stats_mock

        result = await handler.execute("memories", "--archived", "test-project")

        call_args = mock_engine.list.call_args
        assert call_args.kwargs["include_archived"] is True

    @pytest.mark.asyncio
    async def test_memories_with_limit(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test memories with limit."""
        mock_engine.list.return_value = []
        stats_mock = MagicMock()
        stats_mock.total_memories = 0
        stats_mock.active_memories = 0
        stats_mock.archived_memories = 0
        stats_mock.by_category = {}
        mock_engine.stats.return_value = stats_mock

        result = await handler.execute("memories", "limit:25", "test-project")

        call_args = mock_engine.list.call_args
        assert call_args.kwargs["limit"] == 25

    # Unknown command test

    @pytest.mark.asyncio
    async def test_unknown_command(self, handler: CommandHandler) -> None:
        """Test unknown command."""
        result = await handler.execute("unknown", "args", "test-project")

        assert result.success is False
        assert "Unknown command" in result.error
        assert "/remember" in result.error

    # Error handling tests

    @pytest.mark.asyncio
    async def test_exception_handling(
        self, handler: CommandHandler, mock_engine: MagicMock
    ) -> None:
        """Test exception handling in command execution."""
        mock_engine.add.side_effect = Exception("Database error")

        result = await handler.execute("remember", "test content", "test-project")

        assert result.success is False
        assert "Database error" in result.error

    # Config usage tests

    @pytest.mark.asyncio
    async def test_uses_config_default_project(
        self, mock_engine: MagicMock, sample_memory: Memory
    ) -> None:
        """Test that handler uses config default project."""
        config = CommandConfig(default_project="default-proj")
        handler = CommandHandler(engine=mock_engine, config=config)
        mock_engine.add.return_value = sample_memory

        result = await handler.execute("remember", "test content")

        call_args = mock_engine.add.call_args
        assert call_args.kwargs["project"] == "default-proj"


class TestCommandSchemas:
    """Tests for command schema generation."""

    def test_get_command_schemas(self) -> None:
        """Test getting all command schemas."""
        schemas = get_command_schemas()

        assert "remember" in schemas
        assert "recall" in schemas
        assert "forget" in schemas
        assert "outcome" in schemas
        assert "context" in schemas
        assert "memories" in schemas

    def test_remember_schema(self) -> None:
        """Test remember command schema."""
        schemas = get_command_schemas()
        remember = schemas["remember"]

        assert remember["name"] == "remember"
        assert "content" in remember["parameters"]["properties"]
        assert "category" in remember["parameters"]["properties"]
        assert "content" in remember["parameters"]["required"]

    def test_recall_schema(self) -> None:
        """Test recall command schema."""
        schemas = get_command_schemas()
        recall = schemas["recall"]

        assert recall["name"] == "recall"
        assert "query" in recall["parameters"]["properties"]
        assert "limit" in recall["parameters"]["properties"]

    def test_outcome_schema(self) -> None:
        """Test outcome command schema."""
        schemas = get_command_schemas()
        outcome = schemas["outcome"]

        assert outcome["name"] == "outcome"
        params = outcome["parameters"]["properties"]
        assert "memory_id" in params
        assert "outcome" in params
        assert params["outcome"]["enum"] == ["worked", "failed", "partial"]

    def test_category_enum_in_schema(self) -> None:
        """Test that category enum values are correct."""
        schemas = get_command_schemas()
        categories = schemas["remember"]["parameters"]["properties"]["category"]["enum"]

        for cat in MemoryCategory:
            assert cat.value in categories

    def test_schema_examples(self) -> None:
        """Test that schemas have examples."""
        schemas = get_command_schemas()

        for name, schema in schemas.items():
            assert "examples" in schema, f"{name} schema should have examples"
            assert len(schema["examples"]) > 0

    def test_export_command_schemas(self, tmp_path: Path) -> None:
        """Test exporting schemas to file."""
        output_path = tmp_path / "schemas.json"

        export_command_schemas(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert "remember" in loaded
        assert "recall" in loaded


class TestCommandType:
    """Tests for CommandType enum."""

    def test_all_command_types(self) -> None:
        """Test all command types are defined."""
        expected = ["remember", "recall", "forget", "outcome", "context", "memories"]

        for cmd in expected:
            assert CommandType(cmd) is not None

    def test_command_type_values(self) -> None:
        """Test command type string values."""
        assert CommandType.REMEMBER.value == "remember"
        assert CommandType.RECALL.value == "recall"
        assert CommandType.FORGET.value == "forget"
        assert CommandType.OUTCOME.value == "outcome"
        assert CommandType.CONTEXT.value == "context"
        assert CommandType.MEMORIES.value == "memories"

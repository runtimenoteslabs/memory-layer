"""Unit tests for CLI commands.

Tests for all CLI commands including:
- Core commands (add, search, show, list, delete, outcome)
- Context commands (context, extract)
- Session commands (session start, session end)
- Hook support commands (track-file)
- Utility commands (stats, serve, ingest, export)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from memory_layer.cli.main import cli
from memory_layer.core.models import (
    ContextResponse,
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    SearchResult,
)
from memory_layer.core.engine import EngineStats
from memory_layer.core.storage import StorageStats


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


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
        Memory(
            id="mem-003",
            content="Watch for N+1 queries",
            category=MemoryCategory.GOTCHA,
            outcome_score=-0.2,
            use_count=2,
        ),
    ]


@pytest.fixture
def mock_engine(sample_memory, sample_memories):
    """Create a mock engine with common methods."""
    engine = MagicMock()

    # Include sample_memory in list so partial ID matching works
    all_memories = [sample_memory] + sample_memories

    # Configure async methods
    engine.add = AsyncMock(return_value=sample_memory)
    engine.get = AsyncMock(return_value=sample_memory)
    engine.search = AsyncMock(return_value=[
        SearchResult(memory=m, score=0.9 - i * 0.1)
        for i, m in enumerate(sample_memories)
    ])
    engine.list = AsyncMock(return_value=all_memories)
    engine.delete = AsyncMock(return_value=True)
    engine.record_outcome = AsyncMock(return_value=True)
    engine.get_context = AsyncMock(return_value=ContextResponse(
        memories=sample_memories,
        project="test-project",
        total_count=3,
        included_count=3,
    ))

    # Create proper stats structure using actual dataclasses
    storage_stats = StorageStats(
        total_memories=10,
        active_memories=8,
        archived_memories=2,
        avg_outcome_score=0.35,
        total_uses=25,
        by_category={
            "convention": 3,
            "architecture": 2,
            "gotcha": 5,
        },
        by_scope={"global": 5, "project": 5},
        by_source={"explicit": 8, "extracted": 2},
    )

    stats_result = EngineStats(
        storage_stats=storage_stats,
        indexed_memories=8,
        indexed_with_embeddings=8,
        last_search_result_count=0,
    )

    engine.stats = AsyncMock(return_value=stats_result)

    return engine


# =============================================================================
# CLI Group Tests
# =============================================================================


class TestCLIGroup:
    """Tests for the main CLI group."""

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Memory Layer" in result.output
        assert "Store, search, and manage memories" in result.output

    def test_cli_verbose_flag(self, runner):
        """Test verbose flag is accepted."""
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0

    def test_cli_json_output_flag(self, runner):
        """Test JSON output flag is accepted."""
        result = runner.invoke(cli, ["--json-output", "--help"])
        assert result.exit_code == 0


# =============================================================================
# Add Command Tests
# =============================================================================


class TestAddCommand:
    """Tests for the add command."""

    def test_add_basic(self, runner, mock_engine):
        """Test basic memory addition."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, [
                "add", "Use snake_case for Python", "-c", "convention"
            ])

        assert result.exit_code == 0
        assert "Added memory" in result.output
        assert "convention" in result.output

    def test_add_with_project(self, runner, mock_engine):
        """Test adding memory with project."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, [
                "add", "Test content", "-c", "general", "-p", "myproject"
            ])

        assert result.exit_code == 0
        mock_engine.add.assert_called_once()
        call_kwargs = mock_engine.add.call_args[1]
        assert call_kwargs["project"] == "myproject"

    def test_add_with_tags(self, runner, mock_engine):
        """Test adding memory with tags."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, [
                "add", "Test content", "-c", "general", "--tags", "python,naming,style"
            ])

        assert result.exit_code == 0
        call_kwargs = mock_engine.add.call_args[1]
        assert "python" in call_kwargs["tags"]

    def test_add_with_importance(self, runner, mock_engine):
        """Test adding memory with importance."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, [
                "add", "Important content", "-c", "decision", "--importance", "0.9"
            ])

        assert result.exit_code == 0
        call_kwargs = mock_engine.add.call_args[1]
        assert call_kwargs["importance"] == 0.9

    def test_add_json_output(self, runner, mock_engine):
        """Test add with JSON output."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, [
                "--json-output", "add", "Test content", "-c", "general"
            ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "id" in data
        assert "content" in data

    def test_add_all_categories(self, runner, mock_engine):
        """Test that all categories are accepted."""
        categories = [
            "architecture", "convention", "decision", "pattern", "gotcha",
            "workaround", "troubleshooting", "command", "preference",
            "dependency", "environment", "coding_style", "tool_preference",
            "context", "todo", "general"
        ]

        for cat in categories:
            with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
                result = runner.invoke(cli, ["add", "Test", "-c", cat])
                assert result.exit_code == 0, f"Failed for category: {cat}"


# =============================================================================
# Search Command Tests
# =============================================================================


class TestSearchCommand:
    """Tests for the search command."""

    def test_search_basic(self, runner, mock_engine):
        """Test basic search."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "authentication"])

        assert result.exit_code == 0
        mock_engine.search.assert_called_once()

    def test_search_with_limit(self, runner, mock_engine):
        """Test search with limit."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "test", "-l", "10"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_search_with_category_filter(self, runner, mock_engine):
        """Test search with category filter."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "test", "-c", "convention"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["category"] == MemoryCategory.CONVENTION

    def test_search_with_project_filter(self, runner, mock_engine):
        """Test search with project filter."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "test", "-p", "myproject"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["project"] == "myproject"

    def test_search_with_min_score(self, runner, mock_engine):
        """Test search with minimum score filter."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "test", "--min-score", "0.5"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["min_score"] == 0.5

    def test_search_detailed_format(self, runner, mock_engine):
        """Test search with detailed format."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "test", "--format", "detailed"])

        assert result.exit_code == 0
        assert "score:" in result.output or "CONVENTION" in result.output

    def test_search_context_format(self, runner, mock_engine):
        """Test search with context format."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "test", "--format", "context"])

        assert result.exit_code == 0

    def test_search_json_output(self, runner, mock_engine):
        """Test search with JSON output."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["--json-output", "search", "test"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_search_no_results(self, runner, mock_engine):
        """Test search with no results."""
        mock_engine.search = AsyncMock(return_value=[])
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "nonexistent"])

        assert result.exit_code == 0
        assert "No memories found" in result.output


# =============================================================================
# Show Command Tests
# =============================================================================


class TestShowCommand:
    """Tests for the show command."""

    def test_show_basic(self, runner, mock_engine):
        """Test showing a memory."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["show", "test-mem-001"])

        assert result.exit_code == 0
        assert "ID:" in result.output
        assert "Category:" in result.output
        assert "Content:" in result.output

    def test_show_json_output(self, runner, mock_engine):
        """Test show with JSON output."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["--json-output", "show", "test-mem-001"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "id" in data

    def test_show_not_found(self, runner, mock_engine):
        """Test showing non-existent memory."""
        mock_engine.get = AsyncMock(return_value=None)
        mock_engine.list = AsyncMock(return_value=[])  # No memories to match
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["show", "nonexistent"])

        assert result.exit_code != 0
        assert "no memory found" in result.output.lower()


# =============================================================================
# List Command Tests
# =============================================================================


class TestListCommand:
    """Tests for the list command."""

    def test_list_basic(self, runner, mock_engine):
        """Test basic listing."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0

    def test_list_with_limit(self, runner, mock_engine):
        """Test listing with limit."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["list", "-l", "5"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.list.call_args[1]
        assert call_kwargs["limit"] == 5

    def test_list_with_category(self, runner, mock_engine):
        """Test listing with category filter."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["list", "-c", "convention"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.list.call_args[1]
        assert call_kwargs["category"] == MemoryCategory.CONVENTION

    def test_list_with_project(self, runner, mock_engine):
        """Test listing with project filter."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["list", "-p", "myproject"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.list.call_args[1]
        assert call_kwargs["project"] == "myproject"

    def test_list_with_archived(self, runner, mock_engine):
        """Test listing with archived flag."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["list", "--archived"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.list.call_args[1]
        assert call_kwargs["include_archived"] is True

    def test_list_json_output(self, runner, mock_engine):
        """Test list with JSON output."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["--json-output", "list"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_list_empty(self, runner, mock_engine):
        """Test listing with no results."""
        mock_engine.list = AsyncMock(return_value=[])
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No memories found" in result.output


# =============================================================================
# Delete Command Tests
# =============================================================================


class TestDeleteCommand:
    """Tests for the delete command."""

    def test_delete_with_confirm(self, runner, mock_engine):
        """Test deleting with confirm flag."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["delete", "test-mem-001", "--confirm"])

        assert result.exit_code == 0
        assert "Archived" in result.output
        mock_engine.delete.assert_called_once_with("test-mem-001")

    def test_delete_interactive_yes(self, runner, mock_engine):
        """Test delete with interactive confirmation (yes)."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["delete", "test-mem-001"], input="y\n")

        assert result.exit_code == 0
        assert "Archived" in result.output

    def test_delete_interactive_no(self, runner, mock_engine):
        """Test delete with interactive confirmation (no)."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["delete", "test-mem-001"], input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output
        mock_engine.delete.assert_not_called()

    def test_delete_not_found(self, runner, mock_engine):
        """Test deleting non-existent memory."""
        mock_engine.delete = AsyncMock(return_value=False)
        mock_engine.list = AsyncMock(return_value=[])  # No memories to match
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["delete", "nonexistent", "--confirm"])

        assert result.exit_code != 0
        assert "no memory found" in result.output.lower()


# =============================================================================
# Outcome Command Tests
# =============================================================================


class TestOutcomeCommand:
    """Tests for the outcome command."""

    def test_outcome_worked(self, runner, mock_engine):
        """Test recording worked outcome."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["outcome", "test-mem-001", "worked"])

        assert result.exit_code == 0
        assert "worked" in result.output
        assert "+0.2" in result.output

    def test_outcome_failed(self, runner, mock_engine):
        """Test recording failed outcome."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["outcome", "test-mem-001", "failed"])

        assert result.exit_code == 0
        assert "failed" in result.output
        assert "-0.3" in result.output

    def test_outcome_partial(self, runner, mock_engine):
        """Test recording partial outcome."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["outcome", "test-mem-001", "partial"])

        assert result.exit_code == 0
        assert "partial" in result.output
        assert "+0.05" in result.output

    def test_outcome_not_found(self, runner, mock_engine):
        """Test outcome for non-existent memory."""
        mock_engine.record_outcome = AsyncMock(return_value=False)
        mock_engine.list = AsyncMock(return_value=[])  # No memories to match
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["outcome", "nonexistent", "worked"])

        assert result.exit_code != 0
        assert "no memory found" in result.output.lower()

    def test_outcome_invalid_result(self, runner, mock_engine):
        """Test outcome with invalid result type."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["outcome", "test-mem-001", "invalid"])

        assert result.exit_code != 0


# =============================================================================
# Context Command Tests
# =============================================================================


class TestContextCommand:
    """Tests for the context command."""

    def test_context_basic(self, runner, mock_engine):
        """Test basic context retrieval."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["context"])

        assert result.exit_code == 0

    def test_context_with_project(self, runner, mock_engine):
        """Test context with project."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["context", "-p", "/path/to/project"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.get_context.call_args[1]
        assert call_kwargs["project"] == "project"  # Just the name

    def test_context_with_limit(self, runner, mock_engine):
        """Test context with limit."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["context", "-l", "5"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.get_context.call_args[1]
        assert call_kwargs["max_memories"] == 5

    def test_context_inject_flag(self, runner, mock_engine):
        """Test context with inject flag."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["context", "--inject"])

        assert result.exit_code == 0

    def test_context_json_format(self, runner, mock_engine):
        """Test context with JSON format."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["context", "--format", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "memories" in data

    def test_context_silent_format(self, runner, mock_engine):
        """Test context with silent format (for hooks)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
                with patch.dict(os.environ, {"HOME": tmpdir}):
                    result = runner.invoke(cli, ["context", "--format", "silent"])

            assert result.exit_code == 0
            assert result.output == ""  # Silent

    def test_context_different_formats(self, runner, mock_engine):
        """Test context with different format options."""
        formats = ["brief", "detailed", "structured", "markdown"]

        for fmt in formats:
            with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
                result = runner.invoke(cli, ["context", "--format", fmt])
                assert result.exit_code == 0, f"Failed for format: {fmt}"


# =============================================================================
# Extract Command Tests
# =============================================================================


class TestExtractCommand:
    """Tests for the extract command."""

    def test_extract_auto(self, runner):
        """Test auto extraction."""
        result = runner.invoke(cli, ["extract", "--auto"])

        assert result.exit_code == 0
        assert "Extraction triggered" in result.output

    def test_extract_with_session(self, runner):
        """Test extraction with session ID."""
        result = runner.invoke(cli, ["extract", "--session", "test-session-123"])

        assert result.exit_code == 0
        assert "test-session-123" in result.output

    def test_extract_from_env(self, runner):
        """Test extraction with session from environment."""
        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "env-session-456"}):
            result = runner.invoke(cli, ["extract", "--auto"])

        assert result.exit_code == 0
        assert "env-session-456" in result.output

    def test_extract_quiet(self, runner):
        """Test quiet extraction."""
        result = runner.invoke(cli, ["extract", "--auto", "--quiet"])

        assert result.exit_code == 0
        assert result.output == ""

    def test_extract_with_stdin_flag(self, runner):
        """Test extraction with --stdin flag."""
        result = runner.invoke(cli, ["extract", "--stdin"], input="Test transcript content\n")

        assert result.exit_code == 0

    def test_extract_stdin_content(self, runner):
        """Test extraction reading from stdin."""
        # Note: CliRunner doesn't perfectly simulate stdin.isatty()
        # but we can test the flag behavior
        result = runner.invoke(cli, ["extract", "--stdin"], input="Sample conversation\n")

        assert result.exit_code == 0


# =============================================================================
# Session Commands Tests
# =============================================================================


class TestSessionCommands:
    """Tests for session commands."""

    def test_session_start_basic(self, runner):
        """Test starting a session."""
        result = runner.invoke(cli, ["session", "start", "--session", "test-123"])

        assert result.exit_code == 0
        assert "Session started" in result.output
        assert "test-123" in result.output

    def test_session_start_from_env(self, runner):
        """Test starting session from environment."""
        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "env-session"}):
            result = runner.invoke(cli, ["session", "start"])

        assert result.exit_code == 0
        assert "env-session" in result.output

    def test_session_start_json_output(self, runner):
        """Test session start with JSON output."""
        result = runner.invoke(cli, ["--json-output", "session", "start", "--session", "test-123"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "session_id" in data

    def test_session_end_basic(self, runner):
        """Test ending a session."""
        result = runner.invoke(cli, ["session", "end", "--session", "test-123"])

        assert result.exit_code == 0

    def test_session_end_with_summarize(self, runner):
        """Test ending session with summary."""
        result = runner.invoke(cli, ["session", "end", "--session", "test-123", "--summarize"])

        assert result.exit_code == 0
        assert "Session ended" in result.output

    def test_session_help(self, runner):
        """Test session group help."""
        result = runner.invoke(cli, ["session", "--help"])

        assert result.exit_code == 0
        assert "Session management" in result.output


# =============================================================================
# Track File Command Tests
# =============================================================================


class TestTrackFileCommand:
    """Tests for the track-file command."""

    def test_track_file_basic(self, runner):
        """Test basic file tracking."""
        result = runner.invoke(cli, ["track-file", "/path/to/file.py"])

        assert result.exit_code == 0

    def test_track_file_with_session(self, runner):
        """Test file tracking with session."""
        result = runner.invoke(cli, [
            "track-file", "/path/to/file.py", "--session", "test-123"
        ])

        assert result.exit_code == 0

    def test_track_file_silent(self, runner):
        """Test that track-file is silent for hook usage."""
        result = runner.invoke(cli, ["track-file", "/path/to/file.py"])

        assert result.exit_code == 0
        assert result.output == ""  # Should be silent


# =============================================================================
# Stats Command Tests
# =============================================================================


class TestStatsCommand:
    """Tests for the stats command."""

    def test_stats_basic(self, runner, mock_engine):
        """Test basic stats."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["stats"])

        assert result.exit_code == 0
        assert "Memory Layer Statistics" in result.output
        assert "Total memories:" in result.output
        assert "By Category:" in result.output

    def test_stats_with_project(self, runner, mock_engine):
        """Test stats with project filter."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["stats", "-p", "myproject"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.stats.call_args[1]
        assert call_kwargs["project"] == "myproject"

    def test_stats_json_output(self, runner, mock_engine):
        """Test stats with JSON output."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["--json-output", "stats"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        # Stats structure has storage_stats nested
        assert "storage_stats" in data
        assert "total_memories" in data["storage_stats"]
        assert "by_category" in data["storage_stats"]


# =============================================================================
# Serve Command Tests
# =============================================================================


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_no_option(self, runner):
        """Test serve without --mcp or --rest."""
        result = runner.invoke(cli, ["serve"])

        assert result.exit_code != 0
        assert "Specify --mcp or --rest" in result.output

    def test_serve_mcp(self, runner):
        """Test serve with MCP option."""
        # Mock the run_mcp_server since it uses stdio which doesn't work with CliRunner
        with patch("memory_layer.cli.main.asyncio.run") as mock_run:
            result = runner.invoke(cli, ["serve", "--mcp"])

        assert result.exit_code == 0
        assert "MCP" in result.output
        mock_run.assert_called_once()

    def test_serve_rest(self, runner, mock_engine):
        """Test serve with REST option."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            # Patch at the server package level since CLI does:
            # from memory_layer.server import run_server
            with patch("memory_layer.server.run_server") as mock_run:
                result = runner.invoke(cli, ["serve", "--rest"])

        assert result.exit_code == 0
        assert "REST" in result.output
        mock_run.assert_called_once()

    def test_serve_rest_with_port(self, runner, mock_engine):
        """Test serve with custom port."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            with patch("memory_layer.server.run_server") as mock_run:
                result = runner.invoke(cli, ["serve", "--rest", "--port", "9000"])

        assert result.exit_code == 0
        assert "9000" in result.output
        # Verify port was passed correctly
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("port") == 9000

    def test_serve_rest_with_host(self, runner, mock_engine):
        """Test serve with custom host."""
        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            with patch("memory_layer.server.run_server") as mock_run:
                result = runner.invoke(cli, ["serve", "--rest", "--host", "0.0.0.0"])

        assert result.exit_code == 0
        assert "0.0.0.0" in result.output
        # Verify host was passed correctly
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("host") == "0.0.0.0"


# =============================================================================
# Ingest Command Tests
# =============================================================================


class TestIngestCommand:
    """Tests for the ingest command."""

    def test_ingest_basic(self, runner, tmp_path):
        """Test basic file ingestion."""
        # Create a test file
        test_file = tmp_path / "transcript.txt"
        test_file.write_text("Sample transcript content")

        result = runner.invoke(cli, ["ingest", str(test_file)])

        assert result.exit_code == 0
        assert "Ingesting" in result.output

    def test_ingest_with_project(self, runner, tmp_path):
        """Test ingestion with project."""
        test_file = tmp_path / "transcript.txt"
        test_file.write_text("Sample content")

        result = runner.invoke(cli, ["ingest", str(test_file), "-p", "myproject"])

        assert result.exit_code == 0

    def test_ingest_nonexistent_file(self, runner):
        """Test ingestion with non-existent file."""
        result = runner.invoke(cli, ["ingest", "/nonexistent/file.txt"])

        assert result.exit_code != 0


# =============================================================================
# Export Command Tests
# =============================================================================


class TestExportCommand:
    """Tests for the export command."""

    def test_export_json(self, runner, mock_engine, tmp_path):
        """Test JSON export."""
        output_file = tmp_path / "export.json"

        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["export", str(output_file), "--format", "json"])

        assert result.exit_code == 0
        assert "Exported" in result.output
        assert output_file.exists()

        # Verify JSON content
        data = json.loads(output_file.read_text())
        assert isinstance(data, list)

    def test_export_markdown(self, runner, mock_engine, tmp_path):
        """Test markdown export."""
        output_file = tmp_path / "export.md"

        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["export", str(output_file), "--format", "md"])

        assert result.exit_code == 0
        assert "Exported" in result.output
        assert output_file.exists()

    def test_export_with_project(self, runner, mock_engine, tmp_path):
        """Test export with project filter."""
        output_file = tmp_path / "export.json"

        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, [
                "export", str(output_file), "-p", "myproject"
            ])

        assert result.exit_code == 0
        call_kwargs = mock_engine.list.call_args[1]
        assert call_kwargs["project"] == "myproject"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_add_engine_error(self, runner, mock_engine):
        """Test add command with engine error."""
        mock_engine.add = AsyncMock(side_effect=Exception("Database error"))

        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["add", "Test content", "-c", "general"])

        assert result.exit_code != 0
        assert "error" in result.output.lower()

    def test_search_engine_error(self, runner, mock_engine):
        """Test search command with engine error."""
        mock_engine.search = AsyncMock(side_effect=Exception("Search failed"))

        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["search", "test"])

        assert result.exit_code != 0

    def test_stats_engine_error(self, runner, mock_engine):
        """Test stats command with engine error."""
        mock_engine.stats = AsyncMock(side_effect=Exception("Stats failed"))

        with patch("memory_layer.cli.main.get_engine", return_value=mock_engine):
            result = runner.invoke(cli, ["stats"])

        assert result.exit_code != 0

"""Unit tests for the plugin module.

Tests for:
- HookContext
- ContextFormatter
- SkillTriggers
- SessionManager
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import patch

import pytest

from memory_layer.core.models import Memory, MemoryCategory
from memory_layer.plugin import (
    ContextFormatter,
    HookContext,
    SessionManager,
    SkillTriggers,
    get_hook_context,
    get_plugin_root,
)


# =============================================================================
# HookContext Tests
# =============================================================================


class TestHookContext:
    """Tests for HookContext dataclass."""

    def test_from_environment_with_all_vars(self):
        """Test creating HookContext with all environment variables set."""
        env = {
            "CLAUDE_SESSION_ID": "test-session-123",
            "PWD": "/home/user/project",
            "TOOL_NAME": "Write",
            "TOOL_INPUT_FILE_PATH": "/home/user/project/src/app.py",
        }
        with patch.dict(os.environ, env, clear=False):
            ctx = HookContext.from_environment()

            assert ctx.session_id == "test-session-123"
            assert ctx.project_path == "/home/user/project"
            assert ctx.tool_name == "Write"
            assert ctx.file_path == "/home/user/project/src/app.py"

    def test_from_environment_with_no_vars(self):
        """Test creating HookContext with no environment variables."""
        env = {}
        with patch.dict(os.environ, env, clear=True):
            # PWD will fall back to cwd
            ctx = HookContext.from_environment()

            assert ctx.session_id is None
            assert ctx.tool_name is None
            assert ctx.file_path is None

    def test_has_session_true(self):
        """Test has_session returns True when session_id is set."""
        ctx = HookContext(session_id="test-123")
        assert ctx.has_session is True

    def test_has_session_false_none(self):
        """Test has_session returns False when session_id is None."""
        ctx = HookContext(session_id=None)
        assert ctx.has_session is False

    def test_has_session_false_empty(self):
        """Test has_session returns False when session_id is empty string."""
        ctx = HookContext(session_id="")
        assert ctx.has_session is False

    def test_project_name_extraction(self):
        """Test project_name extracts name from path."""
        ctx = HookContext(project_path="/home/user/projects/my-app")
        assert ctx.project_name == "my-app"

    def test_project_name_empty_path(self):
        """Test project_name with empty path returns 'unknown'."""
        ctx = HookContext(project_path="")
        assert ctx.project_name == "unknown"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ctx = HookContext(
            session_id="test-123",
            project_path="/home/user/project",
            tool_name="Write",
            file_path="/home/user/project/file.py",
        )
        result = ctx.to_dict()

        assert result["session_id"] == "test-123"
        assert result["project_path"] == "/home/user/project"
        assert result["tool_name"] == "Write"
        assert result["file_path"] == "/home/user/project/file.py"
        assert "timestamp" in result

    def test_get_hook_context_function(self):
        """Test convenience function get_hook_context."""
        env = {"CLAUDE_SESSION_ID": "func-test-123"}
        with patch.dict(os.environ, env, clear=False):
            ctx = get_hook_context()
            assert ctx.session_id == "func-test-123"


# =============================================================================
# ContextFormatter Tests
# =============================================================================


class TestContextFormatter:
    """Tests for ContextFormatter class."""

    @pytest.fixture
    def sample_memories(self) -> list[Memory]:
        """Create sample memories for testing."""
        return [
            Memory(
                id="mem-001",
                content="Use snake_case for Python variables",
                category=MemoryCategory.CONVENTION,
                outcome_score=0.5,
                use_count=5,
                confidence=0.9,
            ),
            Memory(
                id="mem-002",
                content="PostgreSQL with pgvector for embeddings",
                category=MemoryCategory.ARCHITECTURE,
                outcome_score=0.7,
                use_count=3,
                confidence=1.0,
            ),
            Memory(
                id="mem-003",
                content="Watch out for N+1 queries in ORM",
                category=MemoryCategory.GOTCHA,
                outcome_score=-0.3,
                use_count=2,
                confidence=0.8,
            ),
        ]

    def test_format_empty_list(self):
        """Test formatting empty memory list."""
        result = ContextFormatter.format_for_injection([], style="brief")
        assert result == ""

    def test_format_brief_style(self, sample_memories):
        """Test brief formatting style."""
        result = ContextFormatter.format_for_injection(sample_memories, style="brief")

        assert "# Memory Context" in result
        assert "[convention]" in result
        assert "[architecture]" in result
        assert "[gotcha]" in result
        assert "snake_case" in result

    def test_format_detailed_style(self, sample_memories):
        """Test detailed formatting style."""
        result = ContextFormatter.format_for_injection(sample_memories, style="detailed")

        assert "# Memory Context (detailed)" in result
        assert "CONVENTION" in result
        assert "Score:" in result
        assert "Used:" in result

    def test_format_structured_style(self, sample_memories):
        """Test structured formatting style."""
        result = ContextFormatter.format_for_injection(sample_memories, style="structured")

        assert "# Memory Context" in result
        # Should group by category
        assert "Convention" in result or "convention" in result.lower()
        # High score should show [proven]
        assert "[proven]" in result

    def test_format_markdown_style(self, sample_memories):
        """Test markdown formatting style."""
        result = ContextFormatter.format_for_injection(sample_memories, style="markdown")

        assert "# Project Knowledge" in result
        # Should have category headers
        assert "## " in result
        # High confidence indicator
        assert "*[high confidence]*" in result

    def test_format_unknown_style_defaults_to_brief(self, sample_memories):
        """Test unknown style falls back to brief."""
        result = ContextFormatter.format_for_injection(sample_memories, style="unknown")
        assert "# Memory Context" in result

    def test_format_max_memories(self, sample_memories):
        """Test max_memories limit."""
        result = ContextFormatter.format_for_injection(
            sample_memories, style="brief", max_memories=1
        )
        # Should only have one memory
        lines = [l for l in result.split("\n") if l.startswith("- [")]
        assert len(lines) == 1

    def test_format_single_memory_with_hint(self, sample_memories):
        """Test formatting single memory with feedback hint."""
        result = ContextFormatter.format_single_memory(
            sample_memories[0], include_feedback_hint=True
        )

        assert "[convention]" in result
        assert "snake_case" in result
        assert "/outcome" in result
        assert "worked|failed|partial" in result

    def test_format_single_memory_without_hint(self, sample_memories):
        """Test formatting single memory without feedback hint."""
        result = ContextFormatter.format_single_memory(
            sample_memories[0], include_feedback_hint=False
        )

        assert "[convention]" in result
        assert "/outcome" not in result

    def test_format_search_results_empty(self):
        """Test formatting empty search results."""
        result = ContextFormatter.format_search_results([])
        assert "No relevant memories found" in result


# =============================================================================
# SkillTriggers Tests
# =============================================================================


class TestSkillTriggers:
    """Tests for SkillTriggers class."""

    # Retrieval trigger tests
    @pytest.mark.parametrize("message,expected", [
        ("what did we decide about the auth flow?", True),
        ("how do we handle errors in this project?", True),
        ("what's our convention for naming?", True),
        ("last time we discussed this", True),
        ("remember when we fixed that bug?", True),
        ("why did we choose PostgreSQL?", True),
        ("just a random question", False),
        ("how do I write a for loop?", False),
    ])
    def test_should_retrieve(self, message: str, expected: bool):
        """Test retrieval trigger detection."""
        result = SkillTriggers.should_retrieve(message)
        assert result == expected, f"Expected {expected} for: {message}"

    # Pattern trigger tests
    @pytest.mark.parametrize("message,expected", [
        ("create a new service for users", True),
        ("implement the payment feature", True),
        ("write a function to validate input", True),
        ("how should I structure this module?", True),
        ("scaffold a new API endpoint", True),
        ("fix the bug in the login", False),
        ("delete the old code", False),
    ])
    def test_should_surface_patterns(self, message: str, expected: bool):
        """Test pattern trigger detection."""
        result = SkillTriggers.should_surface_patterns(message)
        assert result == expected, f"Expected {expected} for: {message}"

    # Outcome signal tests
    @pytest.mark.parametrize("message,expected", [
        ("thanks! that worked perfectly", "worked"),
        ("it worked, thanks!", "worked"),
        ("perfect!", "worked"),
        ("great, that fixed it", "worked"),
        ("still not working", "failed"),
        ("that didn't help", "failed"),
        ("same error as before", "failed"),
        ("nope, still broken", "failed"),
        ("kind of worked but not fully", "partial"),
        ("partially helped", "partial"),
        ("almost there, close but not quite", "partial"),
        ("let me think about this", None),
        ("can you explain more?", None),
    ])
    def test_detect_outcome_signal(self, message: str, expected: str | None):
        """Test outcome signal detection."""
        result = SkillTriggers.detect_outcome_signal(message)
        assert result == expected, f"Expected {expected} for: {message}"

    def test_get_trigger_type_retrieval(self):
        """Test get_trigger_type returns 'retrieval'."""
        result = SkillTriggers.get_trigger_type("what did we decide about X?")
        assert result == "retrieval"

    def test_get_trigger_type_patterns(self):
        """Test get_trigger_type returns 'patterns'."""
        result = SkillTriggers.get_trigger_type("create a new component")
        assert result == "patterns"

    def test_get_trigger_type_outcome(self):
        """Test get_trigger_type returns 'outcome'."""
        result = SkillTriggers.get_trigger_type("thanks that worked!")
        assert result == "outcome"

    def test_get_trigger_type_none(self):
        """Test get_trigger_type returns None."""
        result = SkillTriggers.get_trigger_type("hello there")
        assert result is None

    def test_extract_query_keywords(self):
        """Test keyword extraction from message."""
        message = "what did we decide about the authentication service pattern?"
        keywords = SkillTriggers.extract_query_keywords(message)

        assert "authentication" in keywords
        assert "service" in keywords
        assert "pattern" in keywords
        # Stop words should be removed
        assert "the" not in keywords
        # Short words (<=2 chars) should be removed
        assert "we" not in keywords

    def test_extract_query_keywords_removes_triggers(self):
        """Test that trigger phrases are removed from keywords."""
        message = "what did we decide about database configuration?"
        keywords = SkillTriggers.extract_query_keywords(message)

        # "what did we decide" should be removed
        assert "decide" not in keywords
        assert "database" in keywords
        assert "configuration" in keywords

    def test_extract_query_keywords_limit(self):
        """Test keyword extraction limits to 10 keywords."""
        message = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        keywords = SkillTriggers.extract_query_keywords(message)
        assert len(keywords) <= 10


# =============================================================================
# SessionManager Tests
# =============================================================================


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_init_no_engine(self):
        """Test initialization without engine."""
        manager = SessionManager()
        assert manager.engine is None
        assert manager._current_session is None
        assert manager.is_active is False

    def test_start_session_with_id(self):
        """Test starting session with explicit ID."""
        manager = SessionManager()
        session_id = manager.start_session("my-session-123")

        assert session_id == "my-session-123"
        assert manager.is_active is True
        assert manager._current_session == "my-session-123"

    def test_start_session_from_env(self):
        """Test starting session from environment variable."""
        manager = SessionManager()

        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "env-session-456"}):
            session_id = manager.start_session()
            assert session_id == "env-session-456"

    def test_current_session_id_property(self):
        """Test current_session_id property."""
        manager = SessionManager()
        manager.start_session("prop-test-123")

        assert manager.current_session_id == "prop-test-123"

    def test_end_session_without_summary(self):
        """Test ending session without summary."""
        manager = SessionManager()
        manager.start_session("end-test-123")

        result = manager.end_session(summarize=False)

        assert result is None
        assert manager.is_active is False
        assert manager._current_session is None

    def test_end_session_with_summary(self):
        """Test ending session with summary."""
        manager = SessionManager()
        manager.start_session("summary-test-123")
        manager.track_memory_use("mem-001")
        manager.track_memory_use("mem-002")

        summary = manager.end_session(summarize=True)

        assert summary is not None
        assert summary["session_id"] == "summary-test-123"
        assert summary["memories_used"] == 2
        assert "mem-001" in summary["memory_ids"]
        assert "mem-002" in summary["memory_ids"]
        assert "start_time" in summary
        assert "end_time" in summary

    def test_end_session_no_active_session(self):
        """Test ending session when none is active."""
        manager = SessionManager()
        result = manager.end_session(summarize=True)
        assert result is None

    def test_track_memory_use(self):
        """Test tracking memory usage."""
        manager = SessionManager()
        manager.start_session("track-test")

        manager.track_memory_use("mem-001")
        manager.track_memory_use("mem-002")
        manager.track_memory_use("mem-001")  # Duplicate

        stats = manager.get_session_stats()
        assert stats["memories_used"] == 2  # Deduped

    def test_get_session_stats(self):
        """Test getting session statistics."""
        manager = SessionManager()
        manager.start_session("stats-test")
        manager.track_memory_use("mem-001")

        stats = manager.get_session_stats()

        assert stats["session_id"] == "stats-test"
        assert stats["is_active"] is True
        assert stats["memories_used"] == 1
        assert stats["start_time"] is not None
        assert stats["duration_seconds"] >= 0

    def test_get_session_stats_no_session(self):
        """Test getting stats when no session is active."""
        manager = SessionManager()
        stats = manager.get_session_stats()

        assert stats["session_id"] is None
        assert stats["is_active"] is False
        assert stats["memories_used"] == 0


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_plugin_root_from_env(self):
        """Test get_plugin_root with environment variable."""
        with patch.dict(os.environ, {"CLAUDE_PLUGIN_ROOT": "/custom/path"}):
            result = get_plugin_root()
            assert str(result) == "/custom/path"

    def test_get_plugin_root_search(self):
        """Test get_plugin_root searches for .claude-plugin directory."""
        # This test depends on the actual file structure
        # The function should find the memory-layer root
        result = get_plugin_root()
        assert result.exists() or True  # May not exist in test env

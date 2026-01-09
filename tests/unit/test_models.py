"""Tests for data models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from memory_layer.core.models import (
    ContextResponse,
    ContextResponseModel,
    Memory,
    MemoryCategory,
    MemoryCreate,
    MemoryResponse,
    MemoryScope,
    MemorySource,
    MemoryUpdate,
    Outcome,
    OutcomeRecord,
    SearchQuery,
    SearchResult,
    SearchResultResponse,
    StatsResponse,
    OUTCOME_SCORE_ADJUSTMENTS,
    generate_id,
    utc_now,
)


class TestEnums:
    """Tests for enum types."""

    def test_memory_category_values(self) -> None:
        """Test MemoryCategory has all 16 categories (v1 + v2)."""
        assert len(MemoryCategory) == 16
        # v1 categories
        expected = [
            "architecture",
            "convention",
            "decision",
            "pattern",
            "gotcha",
            "workaround",
            "troubleshooting",
            "command",
            "preference",
            # v2 additions
            "dependency",
            "environment",
            "coding_style",
            "tool_preference",
            "context",
            "todo",
            "general",
        ]
        for value in expected:
            assert MemoryCategory(value) is not None

    def test_memory_category_is_string(self) -> None:
        """Test MemoryCategory values are strings."""
        for category in MemoryCategory:
            assert isinstance(category.value, str)
            assert category == category.value

    def test_memory_scope_values(self) -> None:
        """Test MemoryScope has all scope levels."""
        assert len(MemoryScope) == 3
        assert MemoryScope.GLOBAL.value == "global"
        assert MemoryScope.PROJECT.value == "project"
        assert MemoryScope.SESSION.value == "session"

    def test_memory_source_values(self) -> None:
        """Test MemorySource has all source types."""
        assert len(MemorySource) == 3
        assert MemorySource.EXPLICIT.value == "explicit"
        assert MemorySource.EXTRACTED.value == "extracted"
        assert MemorySource.IMPORTED.value == "imported"

    def test_outcome_values(self) -> None:
        """Test Outcome has all outcome types."""
        assert len(Outcome) == 3
        assert Outcome.WORKED.value == "worked"
        assert Outcome.FAILED.value == "failed"
        assert Outcome.PARTIAL.value == "partial"

    def test_outcome_score_adjustments(self) -> None:
        """Test outcome score adjustments are correct."""
        assert OUTCOME_SCORE_ADJUSTMENTS[Outcome.WORKED] == 0.2
        assert OUTCOME_SCORE_ADJUSTMENTS[Outcome.FAILED] == -0.3
        assert OUTCOME_SCORE_ADJUSTMENTS[Outcome.PARTIAL] == 0.05


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_id_is_uuid(self) -> None:
        """Test generate_id returns valid UUID string."""
        id1 = generate_id()
        id2 = generate_id()
        assert isinstance(id1, str)
        assert len(id1) == 36  # UUID format
        assert id1 != id2  # Should be unique

    def test_utc_now_returns_utc(self) -> None:
        """Test utc_now returns UTC datetime."""
        now = utc_now()
        assert isinstance(now, datetime)
        assert now.tzinfo == UTC


class TestMemory:
    """Tests for Memory dataclass."""

    def test_memory_creation_minimal(self) -> None:
        """Test creating memory with minimal required fields."""
        memory = Memory(
            content="Test content",
            category=MemoryCategory.DECISION,
        )
        assert memory.content == "Test content"
        assert memory.category == MemoryCategory.DECISION
        assert memory.outcome_score == 0.0
        assert memory.confidence == 1.0
        assert memory.importance == 0.5
        assert memory.use_count == 0
        assert memory.project is None
        assert memory.scope == MemoryScope.PROJECT
        assert memory.source == MemorySource.EXPLICIT
        assert memory.tags == []
        assert memory.entities == []
        assert memory.supersedes is None
        assert memory.archived is False
        assert memory.embedding is None
        assert memory.metadata == {}
        assert isinstance(memory.id, str)
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)

    def test_memory_creation_full(self) -> None:
        """Test creating memory with all fields."""
        now = utc_now()
        memory = Memory(
            id="test-id",
            content="Full test",
            category=MemoryCategory.ARCHITECTURE,
            outcome_score=0.5,
            confidence=0.9,
            importance=0.8,
            use_count=5,
            project="my-project",
            scope=MemoryScope.GLOBAL,
            source=MemorySource.EXTRACTED,
            tags=["test", "example"],
            entities=["file.py", "MyClass"],
            supersedes="old-id",
            archived=False,
            created_at=now,
            updated_at=now,
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
        )
        assert memory.id == "test-id"
        assert memory.outcome_score == 0.5
        assert memory.project == "my-project"
        assert memory.scope == MemoryScope.GLOBAL
        assert memory.source == MemorySource.EXTRACTED
        assert memory.tags == ["test", "example"]
        assert memory.entities == ["file.py", "MyClass"]
        assert memory.supersedes == "old-id"
        assert memory.embedding == [0.1, 0.2, 0.3]
        assert memory.metadata == {"key": "value"}

    def test_apply_outcome_worked(self) -> None:
        """Test applying WORKED outcome."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        memory.apply_outcome(Outcome.WORKED)
        assert memory.outcome_score == 0.2

    def test_apply_outcome_failed(self) -> None:
        """Test applying FAILED outcome."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        memory.apply_outcome(Outcome.FAILED)
        assert memory.outcome_score == -0.3

    def test_apply_outcome_partial(self) -> None:
        """Test applying PARTIAL outcome."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        memory.apply_outcome(Outcome.PARTIAL)
        assert memory.outcome_score == 0.05

    def test_apply_outcome_clamps_max(self) -> None:
        """Test outcome score clamps at 1.0."""
        memory = Memory(
            content="Test",
            category=MemoryCategory.DECISION,
            outcome_score=0.95,
        )
        memory.apply_outcome(Outcome.WORKED)
        assert memory.outcome_score == 1.0

    def test_apply_outcome_clamps_min(self) -> None:
        """Test outcome score clamps at -1.0."""
        memory = Memory(
            content="Test",
            category=MemoryCategory.DECISION,
            outcome_score=-0.9,
        )
        memory.apply_outcome(Outcome.FAILED)
        assert memory.outcome_score == -1.0

    def test_apply_outcome_updates_timestamp(self) -> None:
        """Test applying outcome updates timestamp."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        original_updated = memory.updated_at
        memory.apply_outcome(Outcome.WORKED)
        assert memory.updated_at >= original_updated

    def test_increment_use_count(self) -> None:
        """Test incrementing use count."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        assert memory.use_count == 0
        memory.increment_use_count()
        assert memory.use_count == 1
        memory.increment_use_count()
        assert memory.use_count == 2

    def test_archive(self) -> None:
        """Test archiving memory."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        assert memory.archived is False
        memory.archive()
        assert memory.archived is True

    def test_to_dict(self) -> None:
        """Test converting memory to dictionary."""
        memory = Memory(
            id="test-id",
            content="Test content",
            category=MemoryCategory.GOTCHA,
            project="my-project",
            tags=["tag1"],
        )
        data = memory.to_dict()
        assert data["id"] == "test-id"
        assert data["content"] == "Test content"
        assert data["category"] == "gotcha"
        assert data["project"] == "my-project"
        assert data["tags"] == ["tag1"]
        assert data["scope"] == "project"
        assert data["source"] == "explicit"
        assert isinstance(data["created_at"], str)

    def test_from_dict(self) -> None:
        """Test creating memory from dictionary."""
        data = {
            "id": "test-id",
            "content": "Test content",
            "category": "architecture",
            "outcome_score": 0.5,
            "confidence": 0.9,
            "importance": 0.8,
            "use_count": 3,
            "project": "my-project",
            "scope": "global",
            "source": "extracted",
            "tags": ["tag1", "tag2"],
            "entities": ["file.py"],
            "supersedes": "old-id",
            "archived": False,
            "created_at": "2024-01-15T10:00:00+00:00",
            "updated_at": "2024-01-15T11:00:00+00:00",
            "embedding": [0.1, 0.2],
            "metadata": {"key": "value"},
        }
        memory = Memory.from_dict(data)
        assert memory.id == "test-id"
        assert memory.content == "Test content"
        assert memory.category == MemoryCategory.ARCHITECTURE
        assert memory.outcome_score == 0.5
        assert memory.scope == MemoryScope.GLOBAL
        assert memory.source == MemorySource.EXTRACTED
        assert memory.tags == ["tag1", "tag2"]
        assert memory.embedding == [0.1, 0.2]

    def test_from_dict_minimal(self) -> None:
        """Test creating memory from minimal dictionary."""
        data = {
            "content": "Test",
            "category": "decision",
        }
        memory = Memory.from_dict(data)
        assert memory.content == "Test"
        assert memory.category == MemoryCategory.DECISION
        assert isinstance(memory.id, str)

    def test_roundtrip_dict(self) -> None:
        """Test roundtrip to/from dictionary."""
        original = Memory(
            content="Test",
            category=MemoryCategory.PATTERN,
            project="my-project",
            tags=["a", "b"],
            metadata={"x": 1},
        )
        data = original.to_dict()
        restored = Memory.from_dict(data)
        assert restored.content == original.content
        assert restored.category == original.category
        assert restored.project == original.project
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test creating search result."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        result = SearchResult(
            memory=memory,
            score=0.85,
            semantic_score=0.7,
            recency_score=0.9,
            frequency_score=0.5,
            category_boost=1.2,
        )
        assert result.memory == memory
        assert result.score == 0.85
        assert result.semantic_score == 0.7
        assert result.recency_score == 0.9
        assert result.frequency_score == 0.5
        assert result.category_boost == 1.2

    def test_search_result_to_dict(self) -> None:
        """Test converting search result to dictionary."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        result = SearchResult(memory=memory, score=0.85)
        data = result.to_dict()
        assert data["score"] == 0.85
        assert data["memory"]["content"] == "Test"


class TestContextResponse:
    """Tests for ContextResponse dataclass."""

    def test_context_response_creation(self) -> None:
        """Test creating context response."""
        memories = [
            Memory(content="Test 1", category=MemoryCategory.ARCHITECTURE),
            Memory(content="Test 2", category=MemoryCategory.GOTCHA),
        ]
        response = ContextResponse(
            memories=memories,
            project="my-project",
            total_count=10,
            included_count=2,
            categories={"architecture": 1, "gotcha": 1},
        )
        assert len(response.memories) == 2
        assert response.project == "my-project"
        assert response.total_count == 10
        assert response.included_count == 2

    def test_context_response_to_markdown(self) -> None:
        """Test formatting context as markdown."""
        memories = [
            Memory(
                content="Use microservices",
                category=MemoryCategory.ARCHITECTURE,
                outcome_score=0.5,
            ),
            Memory(
                content="Watch out for null",
                category=MemoryCategory.GOTCHA,
                outcome_score=-0.5,
            ),
        ]
        response = ContextResponse(
            memories=memories,
            project="test",
            total_count=2,
            included_count=2,
        )
        markdown = response.to_markdown()
        assert "# Project Knowledge" in markdown
        assert "## Architecture" in markdown
        assert "## Gotcha" in markdown
        assert "Use microservices" in markdown
        assert "[high confidence]" in markdown
        assert "[low confidence]" in markdown

    def test_context_response_empty_memories(self) -> None:
        """Test markdown with no memories."""
        response = ContextResponse(
            memories=[],
            project=None,
            total_count=0,
            included_count=0,
        )
        markdown = response.to_markdown()
        assert "No relevant memories found" in markdown

    def test_context_response_to_dict(self) -> None:
        """Test converting context response to dictionary."""
        memories = [Memory(content="Test", category=MemoryCategory.DECISION)]
        response = ContextResponse(
            memories=memories,
            project="test",
            total_count=1,
            included_count=1,
        )
        data = response.to_dict()
        assert len(data["memories"]) == 1
        assert data["project"] == "test"
        assert "formatted" in data


class TestMemoryCreate:
    """Tests for MemoryCreate Pydantic model."""

    def test_memory_create_valid(self) -> None:
        """Test creating valid memory input."""
        create = MemoryCreate(
            content="Test content",
            category=MemoryCategory.DECISION,
        )
        assert create.content == "Test content"
        assert create.category == MemoryCategory.DECISION

    def test_memory_create_full(self) -> None:
        """Test creating memory input with all fields."""
        create = MemoryCreate(
            content="Full test",
            category=MemoryCategory.ARCHITECTURE,
            project="my-project",
            scope=MemoryScope.GLOBAL,
            source=MemorySource.IMPORTED,
            confidence=0.9,
            importance=0.8,
            tags=["Tag1", " Tag2 "],
            entities=["file.py"],
            supersedes="old-id",
            metadata={"key": "value"},
        )
        assert create.project == "my-project"
        assert create.scope == MemoryScope.GLOBAL
        assert create.source == MemorySource.IMPORTED
        assert create.confidence == 0.9
        assert create.tags == ["tag1", "tag2"]  # Normalized

    def test_memory_create_empty_content_fails(self) -> None:
        """Test that empty content fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryCreate(content="", category=MemoryCategory.DECISION)
        # Pydantic min_length validation catches empty string
        assert "content" in str(exc_info.value).lower()

    def test_memory_create_whitespace_content_fails(self) -> None:
        """Test that whitespace-only content fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryCreate(content="   ", category=MemoryCategory.DECISION)
        # After strip, content is empty and fails min_length
        assert "content" in str(exc_info.value).lower()

    def test_memory_create_content_stripped(self) -> None:
        """Test that content is stripped of whitespace."""
        create = MemoryCreate(
            content="  Test content  ",
            category=MemoryCategory.DECISION,
        )
        assert create.content == "Test content"

    def test_memory_create_confidence_out_of_range(self) -> None:
        """Test that confidence outside 0-1 fails."""
        with pytest.raises(ValidationError):
            MemoryCreate(
                content="Test",
                category=MemoryCategory.DECISION,
                confidence=1.5,
            )

    def test_memory_create_importance_out_of_range(self) -> None:
        """Test that importance outside 0-1 fails."""
        with pytest.raises(ValidationError):
            MemoryCreate(
                content="Test",
                category=MemoryCategory.DECISION,
                importance=-0.1,
            )

    def test_memory_create_to_memory(self) -> None:
        """Test converting to Memory dataclass."""
        create = MemoryCreate(
            content="Test",
            category=MemoryCategory.PATTERN,
            project="my-project",
            tags=["test"],
        )
        memory = create.to_memory()
        assert isinstance(memory, Memory)
        assert memory.content == "Test"
        assert memory.category == MemoryCategory.PATTERN
        assert memory.project == "my-project"
        assert memory.tags == ["test"]


class TestMemoryUpdate:
    """Tests for MemoryUpdate Pydantic model."""

    def test_memory_update_partial(self) -> None:
        """Test updating with partial fields."""
        update = MemoryUpdate(content="New content")
        assert update.content == "New content"
        assert update.category is None
        assert update.confidence is None

    def test_memory_update_apply_to(self) -> None:
        """Test applying update to memory."""
        memory = Memory(
            content="Old content",
            category=MemoryCategory.DECISION,
            confidence=0.5,
        )
        update = MemoryUpdate(
            content="New content",
            confidence=0.9,
            tags=["new"],
        )
        updated = update.apply_to(memory)
        assert updated.content == "New content"
        assert updated.confidence == 0.9
        assert updated.tags == ["new"]
        assert updated.category == MemoryCategory.DECISION  # Unchanged

    def test_memory_update_empty_content_fails(self) -> None:
        """Test that empty content fails validation."""
        with pytest.raises(ValidationError):
            MemoryUpdate(content="")

    def test_memory_update_none_content_ok(self) -> None:
        """Test that None content is valid (no update)."""
        update = MemoryUpdate(confidence=0.8)
        assert update.content is None


class TestSearchQuery:
    """Tests for SearchQuery Pydantic model."""

    def test_search_query_minimal(self) -> None:
        """Test creating search query with minimal fields."""
        query = SearchQuery(query="test search")
        assert query.query == "test search"
        assert query.limit == 10
        assert query.offset == 0
        assert query.include_archived is False

    def test_search_query_full(self) -> None:
        """Test creating search query with all fields."""
        query = SearchQuery(
            query="test search",
            project="my-project",
            categories=[MemoryCategory.DECISION, MemoryCategory.GOTCHA],
            scope=MemoryScope.PROJECT,
            include_archived=True,
            min_score=0.5,
            limit=20,
            offset=10,
        )
        assert query.project == "my-project"
        assert len(query.categories) == 2
        assert query.scope == MemoryScope.PROJECT
        assert query.include_archived is True
        assert query.min_score == 0.5
        assert query.limit == 20
        assert query.offset == 10

    def test_search_query_limit_bounds(self) -> None:
        """Test limit bounds validation."""
        with pytest.raises(ValidationError):
            SearchQuery(query="test", limit=0)
        with pytest.raises(ValidationError):
            SearchQuery(query="test", limit=101)


class TestOutcomeRecord:
    """Tests for OutcomeRecord Pydantic model."""

    def test_outcome_record_valid(self) -> None:
        """Test creating valid outcome record."""
        record = OutcomeRecord(
            memory_ids=["id1", "id2"],
            outcome=Outcome.WORKED,
        )
        assert len(record.memory_ids) == 2
        assert record.outcome == Outcome.WORKED

    def test_outcome_record_empty_ids_fails(self) -> None:
        """Test that empty memory_ids fails."""
        with pytest.raises(ValidationError):
            OutcomeRecord(memory_ids=[], outcome=Outcome.WORKED)


class TestMemoryResponse:
    """Tests for MemoryResponse Pydantic model."""

    def test_memory_response_from_memory(self) -> None:
        """Test creating response from Memory."""
        memory = Memory(
            id="test-id",
            content="Test",
            category=MemoryCategory.DECISION,
            project="my-project",
            tags=["test"],
        )
        response = MemoryResponse.from_memory(memory)
        assert response.id == "test-id"
        assert response.content == "Test"
        assert response.category == MemoryCategory.DECISION
        assert response.project == "my-project"
        assert response.tags == ["test"]


class TestSearchResultResponse:
    """Tests for SearchResultResponse Pydantic model."""

    def test_search_result_response_from_result(self) -> None:
        """Test creating response from SearchResult."""
        memory = Memory(content="Test", category=MemoryCategory.DECISION)
        result = SearchResult(
            memory=memory,
            score=0.85,
            semantic_score=0.7,
            recency_score=0.9,
            frequency_score=0.5,
            category_boost=1.2,
        )
        response = SearchResultResponse.from_search_result(result)
        assert response.score == 0.85
        assert response.semantic_score == 0.7
        assert response.memory.content == "Test"


class TestContextResponseModel:
    """Tests for ContextResponseModel Pydantic model."""

    def test_context_response_model_from_response(self) -> None:
        """Test creating model from ContextResponse."""
        memories = [Memory(content="Test", category=MemoryCategory.DECISION)]
        response = ContextResponse(
            memories=memories,
            project="test",
            total_count=1,
            included_count=1,
            categories={"decision": 1},
        )
        model = ContextResponseModel.from_context_response(response)
        assert model.project == "test"
        assert model.total_count == 1
        assert len(model.memories) == 1
        assert "formatted" in model.model_dump()


class TestStatsResponse:
    """Tests for StatsResponse Pydantic model."""

    def test_stats_response_creation(self) -> None:
        """Test creating stats response."""
        stats = StatsResponse(
            total_memories=100,
            active_memories=90,
            archived_memories=10,
            by_category={"decision": 50, "gotcha": 30, "architecture": 20},
            by_scope={"project": 80, "global": 20},
            by_source={"explicit": 60, "extracted": 40},
            avg_outcome_score=0.15,
            total_uses=500,
        )
        assert stats.total_memories == 100
        assert stats.active_memories == 90
        assert stats.archived_memories == 10
        assert stats.avg_outcome_score == 0.15
        assert stats.total_uses == 500

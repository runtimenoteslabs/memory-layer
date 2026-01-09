"""Tests for Memory Engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_layer.core.embeddings import MockEmbeddingProvider
from memory_layer.core.engine import (
    EngineConfig,
    EngineError,
    EngineNotInitializedError,
    EngineStats,
    LastSearchInfo,
    MemoryEngine,
    create_engine,
)
from memory_layer.core.models import (
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
    SearchResult,
)
from memory_layer.core.retrieval import HybridRetriever
from memory_layer.core.storage import MemoryNotFoundError, MemoryStorage


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Create a mock embedding provider for testing."""
    return MockEmbeddingProvider()


@pytest.fixture
async def storage(temp_db_path: Path) -> MemoryStorage:
    """Create a storage instance for testing."""
    store = MemoryStorage(temp_db_path, pool_size=2, secure_permissions=False)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def retriever(mock_embedding_provider: MockEmbeddingProvider) -> HybridRetriever:
    """Create a retriever for testing."""
    return HybridRetriever(embedding_provider=mock_embedding_provider)


@pytest.fixture
async def engine(
    temp_db_path: Path,
    mock_embedding_provider: MockEmbeddingProvider,
) -> MemoryEngine:
    """Create an initialized engine for testing."""
    config = EngineConfig(
        db_path=temp_db_path,
        secure_permissions=False,
        embedding_provider="mock",
    )
    eng = MemoryEngine(
        config=config,
        embedding_provider=mock_embedding_provider,
    )
    await eng.initialize()
    yield eng
    await eng.close()


@pytest.fixture
def sample_memory() -> Memory:
    """Create a sample memory for testing."""
    return Memory(
        content="Use dependency injection for testability",
        category=MemoryCategory.PATTERN,
        project="test-project",
        tags=["testing", "di"],
        entities=["service.py"],
    )


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EngineConfig()
        assert config.embedding_provider == "local"
        assert config.pool_size == 5
        assert config.secure_permissions is True
        assert config.auto_archive_enabled is True
        assert config.auto_archive_threshold == -0.5
        assert config.track_last_search is True

    def test_path_expansion(self) -> None:
        """Test that paths are expanded."""
        config = EngineConfig(db_path="~/.memory-layer/test.db")
        assert str(config.db_path).startswith(str(Path.home()))

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = EngineConfig(
            db_path="/tmp/test.db",
            pool_size=10,
            embedding_provider="mock",
            auto_archive_threshold=-0.3,
        )
        assert config.db_path == Path("/tmp/test.db")
        assert config.pool_size == 10
        assert config.embedding_provider == "mock"
        assert config.auto_archive_threshold == -0.3


class TestEngineInitialization:
    """Tests for engine initialization."""

    async def test_initialize_creates_storage(self, temp_db_path: Path) -> None:
        """Test that initialization creates storage."""
        config = EngineConfig(
            db_path=temp_db_path,
            secure_permissions=False,
            embedding_provider="mock",
        )
        engine = MemoryEngine(config=config)
        await engine.initialize()
        assert temp_db_path.exists()
        assert engine._initialized
        await engine.close()

    async def test_initialize_idempotent(self, engine: MemoryEngine) -> None:
        """Test that multiple initializations are safe."""
        await engine.initialize()  # Already initialized
        await engine.initialize()  # Should not fail
        assert engine._initialized

    async def test_close_engine(self, temp_db_path: Path) -> None:
        """Test closing the engine."""
        config = EngineConfig(
            db_path=temp_db_path,
            secure_permissions=False,
            embedding_provider="mock",
        )
        engine = MemoryEngine(config=config)
        await engine.initialize()
        await engine.close()
        assert not engine._initialized

    async def test_context_manager(self, temp_db_path: Path) -> None:
        """Test using engine as async context manager."""
        config = EngineConfig(
            db_path=temp_db_path,
            secure_permissions=False,
            embedding_provider="mock",
        )
        async with MemoryEngine(config=config) as engine:
            assert engine._initialized
            memory = await engine.add(
                content="Test memory",
                category=MemoryCategory.DECISION,
            )
            assert memory.id

    async def test_not_initialized_error(self, temp_db_path: Path) -> None:
        """Test that using uninitalized engine raises error."""
        config = EngineConfig(
            db_path=temp_db_path,
            secure_permissions=False,
            embedding_provider="mock",
        )
        engine = MemoryEngine(config=config)
        with pytest.raises(EngineNotInitializedError):
            await engine.add(content="Test", category=MemoryCategory.DECISION)


class TestAddOperations:
    """Tests for add operations."""

    async def test_add_memory(self, engine: MemoryEngine) -> None:
        """Test adding a memory."""
        memory = await engine.add(
            content="Use async/await for I/O operations",
            category=MemoryCategory.PATTERN,
            project="test-project",
        )
        assert memory.id
        assert memory.content == "Use async/await for I/O operations"
        assert memory.category == MemoryCategory.PATTERN
        assert memory.project == "test-project"
        assert memory.embedding is not None

    async def test_add_memory_with_all_fields(self, engine: MemoryEngine) -> None:
        """Test adding a memory with all optional fields."""
        memory = await engine.add(
            content="Test content",
            category=MemoryCategory.GOTCHA,
            project="my-project",
            scope=MemoryScope.GLOBAL,
            source=MemorySource.EXTRACTED,
            confidence=0.8,
            importance=0.9,
            tags=["tag1", "tag2"],
            entities=["file.py", "function_name"],
            metadata={"key": "value"},
        )
        assert memory.scope == MemoryScope.GLOBAL
        assert memory.source == MemorySource.EXTRACTED
        assert memory.confidence == 0.8
        assert memory.importance == 0.9
        assert memory.tags == ["tag1", "tag2"]
        assert memory.entities == ["file.py", "function_name"]
        assert memory.metadata == {"key": "value"}

    async def test_add_memory_object(self, engine: MemoryEngine, sample_memory: Memory) -> None:
        """Test adding an existing Memory object."""
        result = await engine.add_memory(sample_memory)
        assert result.id == sample_memory.id
        assert result.content == sample_memory.content
        assert result.embedding is not None

    async def test_add_many(self, engine: MemoryEngine) -> None:
        """Test adding multiple memories at once."""
        memories = [
            Memory(content=f"Memory {i}", category=MemoryCategory.DECISION)
            for i in range(5)
        ]
        results = await engine.add_many(memories)
        assert len(results) == 5
        for memory in results:
            assert memory.embedding is not None

    async def test_add_with_supersedes(self, engine: MemoryEngine) -> None:
        """Test adding a memory that supersedes another."""
        # Add original memory
        original = await engine.add(
            content="Original content",
            category=MemoryCategory.DECISION,
        )

        # Add superseding memory
        new_memory = await engine.add(
            content="Updated content",
            category=MemoryCategory.DECISION,
            supersedes=original.id,
        )

        # Original should be archived
        original_updated = await engine.get(original.id)
        assert original_updated.archived is True


class TestGetOperations:
    """Tests for get operations."""

    async def test_get_memory(self, engine: MemoryEngine) -> None:
        """Test getting a memory by ID."""
        memory = await engine.add(
            content="Test content",
            category=MemoryCategory.PATTERN,
        )
        retrieved = await engine.get(memory.id)
        assert retrieved.id == memory.id
        assert retrieved.content == memory.content

    async def test_get_memory_not_found(self, engine: MemoryEngine) -> None:
        """Test getting a non-existent memory."""
        with pytest.raises(MemoryNotFoundError):
            await engine.get("non-existent-id")

    async def test_get_many(self, engine: MemoryEngine) -> None:
        """Test getting multiple memories."""
        memories = await engine.add_many([
            Memory(content=f"Memory {i}", category=MemoryCategory.DECISION)
            for i in range(5)
        ])
        ids = [m.id for m in memories[:3]]
        retrieved = await engine.get_many(ids)
        assert len(retrieved) == 3
        assert all(m.id in ids for m in retrieved)

    async def test_get_many_preserves_order(self, engine: MemoryEngine) -> None:
        """Test that get_many preserves order."""
        memories = await engine.add_many([
            Memory(content=f"Memory {i}", category=MemoryCategory.DECISION)
            for i in range(3)
        ])
        ids = [memories[2].id, memories[0].id, memories[1].id]
        retrieved = await engine.get_many(ids)
        assert [m.id for m in retrieved] == ids


class TestUpdateOperations:
    """Tests for update operations."""

    async def test_update_content(self, engine: MemoryEngine) -> None:
        """Test updating memory content."""
        memory = await engine.add(
            content="Original content",
            category=MemoryCategory.PATTERN,
        )
        original_embedding = memory.embedding

        updated = await engine.update(
            memory.id,
            content="Updated content",
        )
        assert updated.content == "Updated content"
        # Embedding should be regenerated
        assert updated.embedding != original_embedding

    async def test_update_category(self, engine: MemoryEngine) -> None:
        """Test updating memory category."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        updated = await engine.update(
            memory.id,
            category=MemoryCategory.GOTCHA,
        )
        assert updated.category == MemoryCategory.GOTCHA

    async def test_update_multiple_fields(self, engine: MemoryEngine) -> None:
        """Test updating multiple fields at once."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
            tags=["old-tag"],
        )
        updated = await engine.update(
            memory.id,
            confidence=0.5,
            importance=0.8,
            tags=["new-tag"],
        )
        assert updated.confidence == 0.5
        assert updated.importance == 0.8
        assert updated.tags == ["new-tag"]

    async def test_update_not_found(self, engine: MemoryEngine) -> None:
        """Test updating non-existent memory."""
        with pytest.raises(MemoryNotFoundError):
            await engine.update("non-existent-id", content="New")


class TestDeleteOperations:
    """Tests for delete operations."""

    async def test_soft_delete(self, engine: MemoryEngine) -> None:
        """Test soft deleting (archiving) a memory."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        await engine.delete(memory.id)

        # Memory should still exist but be archived
        retrieved = await engine.get(memory.id)
        assert retrieved.archived is True

    async def test_hard_delete(self, engine: MemoryEngine) -> None:
        """Test hard deleting a memory."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        await engine.delete(memory.id, hard_delete=True)

        # Memory should be gone
        with pytest.raises(MemoryNotFoundError):
            await engine.get(memory.id)

    async def test_archive(self, engine: MemoryEngine) -> None:
        """Test archiving a memory."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        archived = await engine.archive(memory.id)
        assert archived.archived is True

    async def test_unarchive(self, engine: MemoryEngine) -> None:
        """Test unarchiving a memory."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        await engine.archive(memory.id)
        unarchived = await engine.unarchive(memory.id)
        assert unarchived.archived is False


class TestSearchOperations:
    """Tests for search operations."""

    async def test_search_basic(self, engine: MemoryEngine) -> None:
        """Test basic search."""
        await engine.add(
            content="Use async/await for I/O operations",
            category=MemoryCategory.PATTERN,
        )
        await engine.add(
            content="Always validate user input",
            category=MemoryCategory.GOTCHA,
        )

        results = await engine.search("async await")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    async def test_search_with_category_filter(self, engine: MemoryEngine) -> None:
        """Test search with category filter."""
        await engine.add(
            content="Pattern content",
            category=MemoryCategory.PATTERN,
        )
        await engine.add(
            content="Pattern content two",
            category=MemoryCategory.GOTCHA,
        )

        results = await engine.search(
            "pattern content",
            category=MemoryCategory.PATTERN,
        )
        assert all(r.memory.category == MemoryCategory.PATTERN for r in results)

    async def test_search_with_project_filter(self, engine: MemoryEngine) -> None:
        """Test search with project filter."""
        await engine.add(
            content="Project A content",
            category=MemoryCategory.PATTERN,
            project="project-a",
        )
        await engine.add(
            content="Project B content",
            category=MemoryCategory.PATTERN,
            project="project-b",
        )

        results = await engine.search(
            "project content",
            project="project-a",
        )
        assert all(r.memory.project == "project-a" for r in results)

    async def test_search_tracks_last_search(self, engine: MemoryEngine) -> None:
        """Test that search tracks last search info."""
        await engine.add(
            content="Test content",
            category=MemoryCategory.PATTERN,
        )

        await engine.search("test")
        assert engine.last_search is not None
        assert engine.last_search.query == "test"
        assert isinstance(engine.last_search.results, list)

    async def test_search_increments_use_count(self, engine: MemoryEngine) -> None:
        """Test that search increments use counts."""
        memory = await engine.add(
            content="Test content",
            category=MemoryCategory.PATTERN,
        )
        assert memory.use_count == 0

        await engine.search("test")

        updated = await engine.get(memory.id)
        assert updated.use_count > 0

    async def test_search_does_not_return_archived(self, engine: MemoryEngine) -> None:
        """Test that search excludes archived memories by default."""
        memory = await engine.add(
            content="Archived content",
            category=MemoryCategory.PATTERN,
        )
        await engine.archive(memory.id)

        results = await engine.search("archived content")
        assert all(r.memory.id != memory.id for r in results)


class TestListOperations:
    """Tests for list operations."""

    async def test_list_all(self, engine: MemoryEngine) -> None:
        """Test listing all memories."""
        for i in range(5):
            await engine.add(
                content=f"Memory {i}",
                category=MemoryCategory.DECISION,
            )

        memories = await engine.list()
        assert len(memories) == 5

    async def test_list_with_filters(self, engine: MemoryEngine) -> None:
        """Test listing with filters."""
        await engine.add(
            content="Pattern 1",
            category=MemoryCategory.PATTERN,
            project="project-a",
        )
        await engine.add(
            content="Pattern 2",
            category=MemoryCategory.PATTERN,
            project="project-b",
        )
        await engine.add(
            content="Decision 1",
            category=MemoryCategory.DECISION,
            project="project-a",
        )

        results = await engine.list(
            project="project-a",
            category=MemoryCategory.PATTERN,
        )
        assert len(results) == 1
        assert results[0].content == "Pattern 1"

    async def test_list_excludes_archived(self, engine: MemoryEngine) -> None:
        """Test that list excludes archived by default."""
        memory = await engine.add(
            content="To be archived",
            category=MemoryCategory.PATTERN,
        )
        await engine.archive(memory.id)

        results = await engine.list()
        assert all(m.id != memory.id for m in results)

    async def test_list_includes_archived(self, engine: MemoryEngine) -> None:
        """Test listing including archived memories."""
        memory = await engine.add(
            content="To be archived",
            category=MemoryCategory.PATTERN,
        )
        await engine.archive(memory.id)

        results = await engine.list(include_archived=True)
        assert any(m.id == memory.id for m in results)

    async def test_list_pagination(self, engine: MemoryEngine) -> None:
        """Test list pagination."""
        for i in range(10):
            await engine.add(
                content=f"Memory {i}",
                category=MemoryCategory.DECISION,
            )

        page1 = await engine.list(limit=5, offset=0)
        page2 = await engine.list(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        assert set(m.id for m in page1).isdisjoint(set(m.id for m in page2))


class TestOutcomeOperations:
    """Tests for outcome recording."""

    async def test_record_outcome_single(self, engine: MemoryEngine) -> None:
        """Test recording outcome for single memory."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        assert memory.outcome_score == 0.0

        updated = await engine.record_outcome(memory.id, Outcome.WORKED)
        assert updated[0].outcome_score == 0.2

    async def test_record_outcome_multiple(self, engine: MemoryEngine) -> None:
        """Test recording outcome for multiple memories."""
        memories = await engine.add_many([
            Memory(content=f"Memory {i}", category=MemoryCategory.DECISION)
            for i in range(3)
        ])
        ids = [m.id for m in memories]

        updated = await engine.record_outcome(ids, Outcome.WORKED)
        assert len(updated) == 3
        assert all(m.outcome_score == 0.2 for m in updated)

    async def test_record_outcome_failed(self, engine: MemoryEngine) -> None:
        """Test recording failed outcome."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        updated = await engine.record_outcome(memory.id, Outcome.FAILED)
        assert updated[0].outcome_score == -0.3

    async def test_record_outcome_partial(self, engine: MemoryEngine) -> None:
        """Test recording partial outcome."""
        memory = await engine.add(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        updated = await engine.record_outcome(memory.id, Outcome.PARTIAL)
        assert updated[0].outcome_score == 0.05

    async def test_record_outcome_for_last_search(self, engine: MemoryEngine) -> None:
        """Test recording outcome for last search results."""
        await engine.add(
            content="Test content",
            category=MemoryCategory.PATTERN,
        )
        await engine.search("test")

        updated = await engine.record_outcome_for_last_search(Outcome.WORKED)
        assert len(updated) > 0

    async def test_record_outcome_for_last_search_no_results(self, engine: MemoryEngine) -> None:
        """Test error when no last search exists."""
        with pytest.raises(EngineError):
            await engine.record_outcome_for_last_search(Outcome.WORKED)


class TestAutoArchive:
    """Tests for auto-archival."""

    async def test_auto_archive(self, engine: MemoryEngine) -> None:
        """Test auto-archiving low score memories."""
        # Create memories with different scores
        memory1 = await engine.add(
            content="Good memory",
            category=MemoryCategory.PATTERN,
        )
        memory2 = await engine.add(
            content="Bad memory",
            category=MemoryCategory.PATTERN,
        )

        # Lower the score of memory2 below threshold
        for _ in range(3):  # -0.3 * 3 = -0.9
            await engine.record_outcome(memory2.id, Outcome.FAILED)

        count = await engine.auto_archive(threshold=-0.5)
        assert count == 1

        # Check that bad memory is archived
        bad = await engine.get(memory2.id)
        assert bad.archived is True

        # Good memory should still be active
        good = await engine.get(memory1.id)
        assert good.archived is False


class TestContextGeneration:
    """Tests for context generation."""

    async def test_get_context_basic(self, engine: MemoryEngine) -> None:
        """Test basic context generation."""
        await engine.add(
            content="Use dependency injection",
            category=MemoryCategory.PATTERN,
            project="test-project",
        )
        await engine.add(
            content="Watch out for race conditions",
            category=MemoryCategory.GOTCHA,
            project="test-project",
        )

        context = await engine.get_context(
            query="best practices",
            project="test-project",
        )
        assert len(context.memories) > 0
        assert context.project == "test-project"
        assert context.included_count <= context.total_count

    async def test_get_context_formats_markdown(self, engine: MemoryEngine) -> None:
        """Test that context generates markdown."""
        await engine.add(
            content="Use dependency injection",
            category=MemoryCategory.PATTERN,
        )

        context = await engine.get_context(query="best practices")
        markdown = context.to_markdown()
        assert "# Project Knowledge" in markdown
        assert "Pattern" in markdown

    async def test_get_context_without_query(self, engine: MemoryEngine) -> None:
        """Test context generation without query."""
        await engine.add(
            content="Test content",
            category=MemoryCategory.PATTERN,
        )

        context = await engine.get_context()
        assert context.included_count > 0


class TestStatistics:
    """Tests for statistics."""

    async def test_stats_basic(self, engine: MemoryEngine) -> None:
        """Test basic statistics."""
        await engine.add(
            content="Test 1",
            category=MemoryCategory.PATTERN,
        )
        await engine.add(
            content="Test 2",
            category=MemoryCategory.DECISION,
        )

        stats = await engine.stats()
        assert isinstance(stats, EngineStats)
        assert stats.storage_stats.total_memories == 2
        assert stats.indexed_memories == 2

    async def test_stats_with_project_filter(self, engine: MemoryEngine) -> None:
        """Test statistics with project filter."""
        await engine.add(
            content="Project A",
            category=MemoryCategory.PATTERN,
            project="project-a",
        )
        await engine.add(
            content="Project B",
            category=MemoryCategory.PATTERN,
            project="project-b",
        )

        stats = await engine.stats(project="project-a")
        assert stats.storage_stats.total_memories == 1


class TestHealthCheck:
    """Tests for health check."""

    async def test_health_check_healthy(self, engine: MemoryEngine) -> None:
        """Test health check on healthy engine."""
        health = await engine.health_check()
        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert health["storage"]["status"] == "healthy"
        assert health["embedding_provider"]["status"] == "healthy"

    async def test_health_check_not_initialized(self, temp_db_path: Path) -> None:
        """Test health check on uninitialized engine."""
        config = EngineConfig(
            db_path=temp_db_path,
            secure_permissions=False,
        )
        engine = MemoryEngine(config=config)
        health = await engine.health_check()
        assert health["status"] == "not_initialized"
        assert health["initialized"] is False


class TestLastSearchInfo:
    """Tests for LastSearchInfo."""

    def test_last_search_info(self) -> None:
        """Test LastSearchInfo dataclass."""
        memory = Memory(
            content="Test",
            category=MemoryCategory.PATTERN,
        )
        result = SearchResult(memory=memory, score=0.9)
        info = LastSearchInfo(query="test query", results=[result])

        assert info.query == "test query"
        assert len(info.results) == 1
        assert len(info.memory_ids) == 1
        assert info.memory_ids[0] == memory.id


class TestCreateEngineFactory:
    """Tests for create_engine factory function."""

    async def test_create_engine(self, temp_db_path: Path) -> None:
        """Test create_engine factory."""
        engine = await create_engine(
            db_path=temp_db_path,
            embedding_provider="mock",
        )
        assert engine._initialized
        await engine.close()

    async def test_create_engine_with_kwargs(self, temp_db_path: Path) -> None:
        """Test create_engine with additional kwargs."""
        engine = await create_engine(
            db_path=temp_db_path,
            embedding_provider="mock",
            pool_size=3,
            auto_archive_enabled=False,
        )
        assert engine.config.pool_size == 3
        assert engine.config.auto_archive_enabled is False
        await engine.close()

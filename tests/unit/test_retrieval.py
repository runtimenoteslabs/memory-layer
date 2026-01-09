"""Tests for retrieval system."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from memory_layer.core.embeddings import MockEmbeddingProvider
from memory_layer.core.models import Memory, MemoryCategory, SearchResult
from memory_layer.core.retrieval import (
    BM25Index,
    CategoryRouter,
    HybridRetriever,
    RetrievalConfig,
)


@pytest.fixture
def mock_provider() -> MockEmbeddingProvider:
    """Create a mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def retrieval_config() -> RetrievalConfig:
    """Create a retrieval config for testing."""
    return RetrievalConfig(
        semantic_weight=0.5,
        recency_weight=0.25,
        frequency_weight=0.15,
        outcome_weight=0.1,
        recency_half_life_days=30.0,
    )


@pytest.fixture
def retriever(
    mock_provider: MockEmbeddingProvider,
    retrieval_config: RetrievalConfig,
) -> HybridRetriever:
    """Create a hybrid retriever for testing."""
    return HybridRetriever(mock_provider, retrieval_config)


def create_memory(
    content: str,
    category: MemoryCategory = MemoryCategory.PATTERN,
    project: str | None = None,
    outcome_score: float = 0.0,
    use_count: int = 0,
    days_old: float = 0.0,
    archived: bool = False,
) -> Memory:
    """Helper to create test memories."""
    created_at = datetime.now(UTC) - timedelta(days=days_old)
    return Memory(
        content=content,
        category=category,
        project=project,
        outcome_score=outcome_score,
        use_count=use_count,
        created_at=created_at,
        updated_at=created_at,
        archived=archived,
    )


class TestBM25Index:
    """Tests for BM25Index."""

    def test_add_and_search(self) -> None:
        """Test adding documents and searching."""
        index = BM25Index()
        index.add_document("doc1", "python programming language")
        index.add_document("doc2", "javascript programming language")
        index.add_document("doc3", "python web framework django")

        results = index.search("python", top_k=10)

        assert len(results) == 2
        # doc1 and doc3 contain "python"
        doc_ids = [r[0] for r in results]
        assert "doc1" in doc_ids
        assert "doc3" in doc_ids
        assert "doc2" not in doc_ids

    def test_search_ranking(self) -> None:
        """Test that more relevant documents rank higher."""
        index = BM25Index()
        index.add_document("doc1", "python")
        index.add_document("doc2", "python python python")  # More term frequency
        index.add_document("doc3", "java")

        results = index.search("python", top_k=10)

        # doc2 should rank higher due to more term frequency
        assert len(results) == 2
        assert results[0][0] == "doc2"
        assert results[0][1] > results[1][1]  # Higher score

    def test_search_idf(self) -> None:
        """Test that rare terms have higher weight."""
        index = BM25Index()
        index.add_document("doc1", "common rare")
        index.add_document("doc2", "common common")
        index.add_document("doc3", "common common common")

        # Search for the rare term
        results = index.search("rare", top_k=10)

        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_remove_document(self) -> None:
        """Test removing a document."""
        index = BM25Index()
        index.add_document("doc1", "python")
        index.add_document("doc2", "python")

        index.remove_document("doc1")

        results = index.search("python", top_k=10)
        assert len(results) == 1
        assert results[0][0] == "doc2"

    def test_clear(self) -> None:
        """Test clearing the index."""
        index = BM25Index()
        index.add_document("doc1", "python")
        index.add_document("doc2", "javascript")

        index.clear()

        assert index.document_count == 0
        assert index.search("python", top_k=10) == []

    def test_score_document(self) -> None:
        """Test scoring a specific document."""
        index = BM25Index()
        index.add_document("doc1", "python programming")
        index.add_document("doc2", "javascript programming")

        score = index.score_document("doc1", "python")
        assert score > 0

        score_no_match = index.score_document("doc2", "python")
        assert score_no_match == 0

    def test_empty_query(self) -> None:
        """Test searching with empty query."""
        index = BM25Index()
        index.add_document("doc1", "python")

        results = index.search("", top_k=10)
        assert results == []

    def test_empty_index(self) -> None:
        """Test searching empty index."""
        index = BM25Index()
        results = index.search("python", top_k=10)
        assert results == []

    def test_tokenization(self) -> None:
        """Test that tokenization handles special characters."""
        index = BM25Index()
        index.add_document("doc1", "Hello, World! Python-Flask")
        index.add_document("doc2", "hello world python flask")

        # Both should match "hello"
        results = index.search("hello", top_k=10)
        assert len(results) == 2


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RetrievalConfig()
        assert config.semantic_weight == 0.5
        assert config.recency_weight == 0.25
        assert config.recency_half_life_days == 30.0
        assert config.default_limit == 10

    def test_default_category_boosts(self) -> None:
        """Test default category boosts are set."""
        config = RetrievalConfig()
        assert MemoryCategory.GOTCHA in config.category_boosts
        assert config.category_boosts[MemoryCategory.GOTCHA] == 1.3

    def test_custom_category_boosts(self) -> None:
        """Test custom category boosts."""
        custom_boosts = {MemoryCategory.PATTERN: 2.0}
        config = RetrievalConfig(category_boosts=custom_boosts)
        assert config.category_boosts == custom_boosts


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    async def test_add_memory(self, retriever: HybridRetriever) -> None:
        """Test adding a memory to the retriever."""
        memory = create_memory("Use dependency injection")

        retriever.add_memory(memory)

        assert retriever.memory_count == 1

    async def test_add_memory_with_embedding(
        self, retriever: HybridRetriever
    ) -> None:
        """Test adding a memory with pre-computed embedding."""
        memory = create_memory("Use dependency injection")
        embedding = [0.1, 0.2, 0.3]

        retriever.add_memory(memory, embedding=embedding)

        assert retriever.memory_count == 1
        assert retriever.indexed_with_embeddings == 1

    async def test_remove_memory(self, retriever: HybridRetriever) -> None:
        """Test removing a memory."""
        memory = create_memory("Use dependency injection")
        retriever.add_memory(memory)

        retriever.remove_memory(memory.id)

        assert retriever.memory_count == 0

    async def test_update_memory(self, retriever: HybridRetriever) -> None:
        """Test updating a memory."""
        memory = create_memory("Old content")
        retriever.add_memory(memory)

        memory.content = "New content"
        retriever.update_memory(memory)

        # Should still have one memory
        assert retriever.memory_count == 1

    async def test_clear(self, retriever: HybridRetriever) -> None:
        """Test clearing all memories."""
        for i in range(5):
            retriever.add_memory(create_memory(f"Memory {i}"))

        retriever.clear()

        assert retriever.memory_count == 0

    async def test_search_empty(self, retriever: HybridRetriever) -> None:
        """Test searching with no memories."""
        results = await retriever.search("test query")
        assert results == []

    async def test_search_basic(self, retriever: HybridRetriever) -> None:
        """Test basic search functionality."""
        memory1 = create_memory("Use dependency injection for testing")
        memory2 = create_memory("Always write unit tests")

        retriever.add_memory(memory1)
        retriever.add_memory(memory2)

        results = await retriever.search("dependency injection")

        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        # Memory1 should be more relevant
        assert results[0].memory.id == memory1.id

    async def test_search_with_embeddings(
        self, retriever: HybridRetriever, mock_provider: MockEmbeddingProvider
    ) -> None:
        """Test search uses embeddings when available."""
        memory = create_memory("Use dependency injection")
        result = await mock_provider.embed(memory.content)
        retriever.add_memory(memory, embedding=result.embedding)

        results = await retriever.search("dependency injection")

        assert len(results) == 1
        assert results[0].semantic_score > 0

    async def test_search_filter_by_category(
        self, retriever: HybridRetriever
    ) -> None:
        """Test filtering search by category."""
        memory1 = create_memory("Pattern 1", category=MemoryCategory.PATTERN)
        memory2 = create_memory("Decision 1", category=MemoryCategory.DECISION)

        retriever.add_memory(memory1)
        retriever.add_memory(memory2)

        results = await retriever.search(
            "test", category=MemoryCategory.PATTERN
        )

        assert len(results) == 1
        assert results[0].memory.category == MemoryCategory.PATTERN

    async def test_search_filter_by_project(
        self, retriever: HybridRetriever
    ) -> None:
        """Test filtering search by project."""
        memory1 = create_memory("Memory 1", project="project-a")
        memory2 = create_memory("Memory 2", project="project-b")

        retriever.add_memory(memory1)
        retriever.add_memory(memory2)

        results = await retriever.search("memory", project="project-a")

        assert len(results) == 1
        assert results[0].memory.project == "project-a"

    async def test_search_excludes_archived(
        self, retriever: HybridRetriever
    ) -> None:
        """Test that archived memories are excluded by default."""
        memory1 = create_memory("Active memory")
        memory2 = create_memory("Archived memory", archived=True)

        retriever.add_memory(memory1)
        retriever.add_memory(memory2)

        results = await retriever.search("memory")

        assert len(results) == 1
        assert results[0].memory.archived is False

    async def test_search_includes_archived(
        self, retriever: HybridRetriever
    ) -> None:
        """Test including archived memories in search."""
        memory1 = create_memory("Active memory")
        memory2 = create_memory("Archived memory", archived=True)

        retriever.add_memory(memory1)
        retriever.add_memory(memory2)

        results = await retriever.search("memory", include_archived=True)

        assert len(results) == 2

    async def test_search_limit(self, retriever: HybridRetriever) -> None:
        """Test search result limit."""
        for i in range(10):
            retriever.add_memory(create_memory(f"Memory {i}"))

        results = await retriever.search("memory", limit=3)

        assert len(results) == 3

    async def test_search_min_score(self, retriever: HybridRetriever) -> None:
        """Test minimum score filtering."""
        memory1 = create_memory("Highly relevant exact match")
        memory2 = create_memory("Something completely different")

        retriever.add_memory(memory1)
        retriever.add_memory(memory2)

        # With a high min_score, only very relevant results should appear
        results = await retriever.search(
            "highly relevant exact match", min_score=0.5
        )

        # Only the matching memory should pass the threshold
        assert all(r.score >= 0.5 for r in results)


class TestScoringComponents:
    """Tests for individual scoring components."""

    async def test_recency_score_new(self, retriever: HybridRetriever) -> None:
        """Test recency score for new memory."""
        memory = create_memory("Fresh memory", days_old=0)
        retriever.add_memory(memory)

        results = await retriever.search("memory")

        assert len(results) == 1
        # New memory should have high recency score
        assert results[0].recency_score > 0.9

    async def test_recency_score_old(self, retriever: HybridRetriever) -> None:
        """Test recency score for old memory."""
        # 60 days old = 2 half-lives, should be ~0.25
        memory = create_memory("Old memory", days_old=60)
        retriever.add_memory(memory)

        results = await retriever.search("memory")

        assert len(results) == 1
        assert results[0].recency_score < 0.3

    async def test_frequency_score_unused(
        self, retriever: HybridRetriever
    ) -> None:
        """Test frequency score for unused memory."""
        memory = create_memory("Unused memory", use_count=0)
        retriever.add_memory(memory)

        results = await retriever.search("memory")

        assert len(results) == 1
        assert results[0].frequency_score == 0.0

    async def test_frequency_score_used(
        self, retriever: HybridRetriever
    ) -> None:
        """Test frequency score for frequently used memory."""
        memory = create_memory("Popular memory", use_count=50)
        retriever.add_memory(memory)

        results = await retriever.search("memory")

        assert len(results) == 1
        assert results[0].frequency_score > 0.5

    async def test_category_boost(self, retriever: HybridRetriever) -> None:
        """Test category boost affects scores."""
        # Gotcha has boost of 1.3
        memory_gotcha = create_memory("Gotcha warning", category=MemoryCategory.GOTCHA)
        # Convention has boost of 0.9
        memory_conv = create_memory("Convention rule", category=MemoryCategory.CONVENTION)

        retriever.add_memory(memory_gotcha)
        retriever.add_memory(memory_conv)

        results_gotcha = await retriever.search(
            "warning", category=MemoryCategory.GOTCHA
        )
        results_conv = await retriever.search(
            "rule", category=MemoryCategory.CONVENTION
        )

        # Both should have different category boosts
        assert results_gotcha[0].category_boost == 1.3
        assert results_conv[0].category_boost == 0.9

    async def test_outcome_score_positive(
        self, retriever: HybridRetriever
    ) -> None:
        """Test that positive outcome boosts score."""
        memory_good = create_memory("Good pattern", outcome_score=0.8)
        memory_bad = create_memory("Bad pattern", outcome_score=-0.8)

        retriever.add_memory(memory_good)
        retriever.add_memory(memory_bad)

        results = await retriever.search("pattern")

        assert len(results) == 2
        # Good memory should rank higher
        assert results[0].memory.outcome_score > results[1].memory.outcome_score


class TestSearchByCategory:
    """Tests for search_by_category."""

    async def test_search_by_category(self, retriever: HybridRetriever) -> None:
        """Test searching across multiple categories."""
        retriever.add_memory(
            create_memory("Pattern 1", category=MemoryCategory.PATTERN)
        )
        retriever.add_memory(
            create_memory("Pattern 2", category=MemoryCategory.PATTERN)
        )
        retriever.add_memory(
            create_memory("Decision 1", category=MemoryCategory.DECISION)
        )

        results = await retriever.search_by_category(
            "test",
            categories=[MemoryCategory.PATTERN, MemoryCategory.DECISION],
            limit_per_category=2,
        )

        assert MemoryCategory.PATTERN in results
        assert MemoryCategory.DECISION in results
        assert len(results[MemoryCategory.PATTERN]) <= 2
        assert len(results[MemoryCategory.DECISION]) <= 2


class TestGetContextMemories:
    """Tests for get_context_memories."""

    async def test_get_context_memories(
        self, retriever: HybridRetriever
    ) -> None:
        """Test getting context memories."""
        # Add memories of different categories
        retriever.add_memory(
            create_memory("Gotcha 1", category=MemoryCategory.GOTCHA)
        )
        retriever.add_memory(
            create_memory("Pattern 1", category=MemoryCategory.PATTERN)
        )
        retriever.add_memory(
            create_memory("Decision 1", category=MemoryCategory.DECISION)
        )

        results = await retriever.get_context_memories("test query")

        assert len(results) <= 10  # Default max

    async def test_get_context_memories_custom_distribution(
        self, retriever: HybridRetriever
    ) -> None:
        """Test context memories with custom distribution."""
        for i in range(5):
            retriever.add_memory(
                create_memory(f"Pattern {i}", category=MemoryCategory.PATTERN)
            )
            retriever.add_memory(
                create_memory(f"Gotcha {i}", category=MemoryCategory.GOTCHA)
            )

        distribution = {
            MemoryCategory.PATTERN: 3,
            MemoryCategory.GOTCHA: 2,
        }

        results = await retriever.get_context_memories(
            "test",
            category_distribution=distribution,
        )

        # Should get at most 5 (3 + 2)
        assert len(results) <= 5


class TestCategoryRouter:
    """Tests for CategoryRouter."""

    def test_route_query_gotcha(self) -> None:
        """Test routing query with gotcha keywords."""
        router = CategoryRouter()
        results = router.route_query("watch out for this trap")

        categories = [cat for cat, _ in results]
        assert MemoryCategory.GOTCHA in categories

    def test_route_query_troubleshooting(self) -> None:
        """Test routing query with troubleshooting keywords."""
        router = CategoryRouter()
        results = router.route_query("how to fix this error")

        categories = [cat for cat, _ in results]
        assert MemoryCategory.TROUBLESHOOTING in categories

    def test_route_query_pattern(self) -> None:
        """Test routing query with pattern keywords."""
        router = CategoryRouter()
        results = router.route_query("best practice for this approach")

        categories = [cat for cat, _ in results]
        assert MemoryCategory.PATTERN in categories

    def test_route_query_no_match(self) -> None:
        """Test routing query with no keyword matches."""
        router = CategoryRouter()
        results = router.route_query("xyz abc 123")

        # Should return default categories
        assert len(results) == 3
        categories = [cat for cat, _ in results]
        assert MemoryCategory.PATTERN in categories
        assert MemoryCategory.DECISION in categories
        assert MemoryCategory.GOTCHA in categories

    def test_route_query_confidence(self) -> None:
        """Test that confidence scores are normalized."""
        router = CategoryRouter()
        results = router.route_query("error fix debug")

        # All confidences should be <= 1.0
        for _, confidence in results:
            assert 0 <= confidence <= 1.0

    def test_get_boost_for_query(self) -> None:
        """Test getting category boost for a query."""
        router = CategoryRouter()
        boost = router.get_boost_for_query("fix this error", MemoryCategory.TROUBLESHOOTING)

        # Should have some boost for troubleshooting on error query
        assert boost >= 1.0

    def test_get_boost_for_query_no_match(self) -> None:
        """Test boost for non-matching category."""
        router = CategoryRouter()
        boost = router.get_boost_for_query("random query", MemoryCategory.COMMAND)

        # Should return base boost (1.0)
        assert boost == 1.0


class TestDeduplication:
    """Tests for result deduplication."""

    async def test_dedup_similar_memories(
        self, retriever: HybridRetriever, mock_provider: MockEmbeddingProvider
    ) -> None:
        """Test that similar memories are deduplicated."""
        # Create two nearly identical memories
        memory1 = create_memory("Use dependency injection for testing")
        memory2 = create_memory("Use dependency injection for testing purposes")

        # Use same fixed embedding to simulate high similarity
        fixed_embedding = [0.5] * 384
        retriever.add_memory(memory1, embedding=fixed_embedding)
        retriever.add_memory(memory2, embedding=fixed_embedding)

        # With dedup, should get fewer results
        results = await retriever.search("dependency injection")

        # Since embeddings are identical, one should be deduplicated
        assert len(results) == 1


class TestEdgeCases:
    """Tests for edge cases."""

    async def test_search_special_characters(
        self, retriever: HybridRetriever
    ) -> None:
        """Test search with special characters."""
        memory = create_memory("Handle error: NullPointerException!")
        retriever.add_memory(memory)

        results = await retriever.search("NullPointerException")
        assert len(results) == 1

    async def test_search_unicode(self, retriever: HybridRetriever) -> None:
        """Test search with unicode content."""
        memory = create_memory("处理中文内容")
        retriever.add_memory(memory)

        results = await retriever.search("中文")
        assert len(results) == 1

    async def test_very_long_query(self, retriever: HybridRetriever) -> None:
        """Test search with very long query."""
        memory = create_memory("Short memory")
        retriever.add_memory(memory)

        long_query = " ".join(["word"] * 1000)
        results = await retriever.search(long_query)
        # Should not crash
        assert isinstance(results, list)

    async def test_search_with_tags_and_entities(
        self, retriever: HybridRetriever
    ) -> None:
        """Test that tags and entities are included in search."""
        memory = Memory(
            content="Base content",
            category=MemoryCategory.PATTERN,
            tags=["python", "testing"],
            entities=["service.py", "TestClass"],
        )
        retriever.add_memory(memory)

        # Search for a tag
        results = await retriever.search("python")
        assert len(results) == 1

        # Search for an entity
        results = await retriever.search("service.py")
        assert len(results) == 1


class TestPerformance:
    """Performance-related tests."""

    async def test_search_latency(
        self, retriever: HybridRetriever, mock_provider: MockEmbeddingProvider
    ) -> None:
        """Test that search completes within reasonable time."""
        # Add 100 memories
        for i in range(100):
            memory = create_memory(f"Memory content number {i}")
            result = await mock_provider.embed(memory.content)
            retriever.add_memory(memory, embedding=result.embedding)

        start = time.time()
        results = await retriever.search("content number")
        elapsed = time.time() - start

        # Should complete in under 1 second (generous for test environment)
        assert elapsed < 1.0
        assert len(results) > 0

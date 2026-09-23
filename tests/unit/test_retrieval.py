"""Tests for retrieval system."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from runtime_memory.core.embeddings import MockEmbeddingProvider
from runtime_memory.core.models import Memory, MemoryCategory, SearchResult
from runtime_memory.core.retrieval import (
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
        semantic_weight=0.35,
        outcome_weight=0.25,
        recency_weight=0.15,
        frequency_weight=0.15,
        confidence_weight=0.10,
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
    confidence: float = 1.0,
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
        confidence=confidence,
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


class TestConfigFromEnv:
    """An evaluation arm needs to ablate one signal without touching the code."""

    def test_defaults_when_nothing_is_set(self, monkeypatch) -> None:
        for signal in ("SEMANTIC", "OUTCOME", "RECENCY", "FREQUENCY", "CONFIDENCE"):
            monkeypatch.delenv(f"RUNTIME_MEMORY_{signal}_WEIGHT", raising=False)

        config = RetrievalConfig.from_env()

        assert config.outcome_weight == 0.25
        assert config.semantic_weight == 0.55

    def test_env_ablates_a_signal(self, monkeypatch) -> None:
        monkeypatch.setenv("RUNTIME_MEMORY_OUTCOME_WEIGHT", "0")

        config = RetrievalConfig.from_env()

        assert config.outcome_weight == 0.0
        # The others keep their weights: dropping a signal must not silently
        # reweight the rest, or the ablation measures two changes at once.
        assert config.semantic_weight == 0.55
        assert config.recency_weight == 0.10
        assert config.frequency_weight == 0.0
        assert config.confidence_weight == 0.10

    def test_explicit_argument_beats_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("RUNTIME_MEMORY_OUTCOME_WEIGHT", "0")

        config = RetrievalConfig.from_env(outcome_weight=0.25)

        assert config.outcome_weight == 0.25


class TestConfidenceSignal:
    """Extraction confidence is one of the five weighted signals.

    It was documented as 10% of the score but was absent from the formula
    entirely, contributing only a category-router boost.
    """

    @pytest.mark.asyncio
    async def test_confidence_separates_otherwise_equal_memories(
        self, retriever: HybridRetriever
    ) -> None:
        sure = create_memory("pytest runs from the repo root", confidence=1.0)
        unsure = create_memory("pytest runs from the repo root", confidence=0.2)
        retriever.add_memory(sure)
        retriever.add_memory(unsure)

        results = await retriever.search("pytest repo root")

        ranked = {r.memory.id: r.score for r in results}
        assert ranked[sure.id] > ranked[unsure.id]

    @pytest.mark.asyncio
    async def test_confidence_contributes_its_documented_share(
        self, retriever: HybridRetriever
    ) -> None:
        """A full-confidence memory carries the whole 0.10, a zero one none."""
        sure = create_memory("ruff replaced flake8", confidence=1.0)
        unsure = create_memory("ruff replaced flake8", confidence=0.0)
        retriever.add_memory(sure)
        retriever.add_memory(unsure)

        results = await retriever.search("ruff flake8")

        ranked = {r.memory.id: r.score for r in results}
        gap = ranked[sure.id] - ranked[unsure.id]
        assert abs(gap - 0.10) < 1e-6


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RetrievalConfig()
        assert config.semantic_weight == 0.55
        assert config.outcome_weight == 0.25
        assert config.recency_weight == 0.10
        # Frequency left the default score in 4.0.0; the option stays.
        assert config.frequency_weight == 0.0
        assert config.confidence_weight == 0.10
        assert config.recency_half_life_days == 30.0
        assert config.default_limit == 10
        assert config.recency_from_created is True

    def test_recency_decays_from_age_not_from_last_touch(self) -> None:
        """A retrieval or an outcome must not make an old memory look new."""
        config = RetrievalConfig()
        retriever = HybridRetriever(MockEmbeddingProvider(), config)
        old = create_memory("Clear the pytest cache when tests fail randomly")
        old.created_at = datetime.now(UTC) - timedelta(days=60)
        old.updated_at = datetime.now(UTC)

        fresh = retriever._calculate_recency_score(old)
        legacy = HybridRetriever(
            MockEmbeddingProvider(), RetrievalConfig.legacy_3x()
        )._calculate_recency_score(old)

        assert fresh == pytest.approx(0.25, abs=0.01)  # two half-lives of age
        assert legacy == pytest.approx(1.0, abs=0.01)  # touched just now

    def test_legacy_3x_restores_the_shipped_scoring(self) -> None:
        """Runs made against 3.x have to stay reproducible."""
        config = RetrievalConfig.legacy_3x()
        assert config.semantic_weight == 0.35
        assert config.recency_weight == 0.15
        assert config.frequency_weight == 0.15
        assert config.relevance_pool_factor is None
        assert config.recency_from_created is False
        assert config.category_boosts[MemoryCategory.GOTCHA] == 1.3
        assert config.category_boosts[MemoryCategory.CONVENTION] == 0.9
        assert RetrievalConfig.legacy_3x(outcome_weight=0.0).outcome_weight == 0.0

    def test_weights_sum_to_one(self) -> None:
        """The score is a weighted average, so the weights must normalise."""
        config = RetrievalConfig()
        total = (
            config.semantic_weight
            + config.outcome_weight
            + config.recency_weight
            + config.frequency_weight
            + config.confidence_weight
        )

        assert abs(total - 1.0) < 1e-9

    def test_matches_the_settings_class(self) -> None:
        """The two classes must agree.

        They are separate objects with the same five weights, and they drifted
        apart once: the retriever scored outcome at 0.1 while every document and
        the settings model said 0.25.
        """
        from runtime_memory.core.config import RetrievalSettings

        scoring = RetrievalConfig()
        settings = RetrievalSettings()

        for weight in (
            "semantic_weight",
            "outcome_weight",
            "recency_weight",
            "frequency_weight",
            "confidence_weight",
        ):
            assert getattr(scoring, weight) == getattr(settings, weight), weight

    def test_outcome_gates_frequency_is_off_by_default(self) -> None:
        """Default scoring must not change for anyone who has not opted in."""
        assert RetrievalConfig().outcome_gates_frequency is False

    def test_default_category_boosts(self) -> None:
        """Test default category boosts are set."""
        config = RetrievalConfig()
        # Neutral since 4.0.0: a category no longer multiplies the whole score.
        assert config.category_boosts == {}

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

        # The query has to match something: since 4.0.0 a search returns only
        # memories with some relevance to it.
        results = await retriever.search(
            "Pattern", category=MemoryCategory.PATTERN
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

    async def test_frequency_ignores_outcome_by_default(
        self, retriever: HybridRetriever
    ) -> None:
        """Without the gate, a memory at the outcome floor keeps its full boost.

        This is the shipped behaviour the gate exists to change: retrieval counts
        as use, and use pays whatever the outcome record says.
        """
        memory = create_memory("Popular but failing memory", use_count=50, outcome_score=-1.0)
        retriever.add_memory(memory)

        results = await retriever.search("memory")

        assert results[0].frequency_score > 0.5

    @pytest.mark.parametrize(
        ("outcome_score", "factor"),
        [(0.6, 1.0), (0.0, 1.0), (-0.5, 0.5), (-1.0, 0.0)],
    )
    async def test_gated_frequency_scales_with_a_negative_outcome(
        self,
        mock_provider: MockEmbeddingProvider,
        outcome_score: float,
        factor: float,
    ) -> None:
        """With the gate on, the boost is untouched at or above zero and gone at -1."""
        ungated = HybridRetriever(mock_provider, RetrievalConfig())
        gated = HybridRetriever(mock_provider, RetrievalConfig(outcome_gates_frequency=True))
        ungated.add_memory(create_memory("Popular memory", use_count=50, outcome_score=outcome_score))
        gated.add_memory(create_memory("Popular memory", use_count=50, outcome_score=outcome_score))

        base = (await ungated.search("memory"))[0].frequency_score
        result = (await gated.search("memory"))[0].frequency_score

        assert base > 0.5
        assert result == pytest.approx(base * factor)

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

        # Neutral by default since 4.0.0.
        assert results_gotcha[0].category_boost == 1.0
        assert results_conv[0].category_boost == 1.0

    async def test_category_boost_when_configured(self) -> None:
        """The boosts still apply when a caller asks for them."""
        retriever = HybridRetriever(MockEmbeddingProvider(), RetrievalConfig.legacy_3x())
        memory_gotcha = create_memory("Gotcha warning", category=MemoryCategory.GOTCHA)
        memory_conv = create_memory("Convention rule", category=MemoryCategory.CONVENTION)
        retriever.add_memory(memory_gotcha)
        retriever.add_memory(memory_conv)

        results_gotcha = await retriever.search("warning", category=MemoryCategory.GOTCHA)
        results_conv = await retriever.search("rule", category=MemoryCategory.CONVENTION)

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

        results = await retriever.search("处理中文内容")
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


class TestSemanticScoreWithoutVectors:
    """Retrieval must fall back to BM25 when either vector is missing."""

    def test_empty_query_embedding_uses_bm25_only(
        self, retriever: HybridRetriever
    ) -> None:
        memory = create_memory("clear the pytest cache when tests fail")
        retriever.add_memory(memory, [0.1] * 384)

        # An empty query vector is what NullEmbeddingProvider returns.
        score = retriever._calculate_semantic_score(memory, "pytest cache", [])
        expected = retriever._bm25.score_document(memory.id, "pytest cache")
        assert score == pytest.approx(expected / (expected + 1.0))

    def test_unembedded_memory_uses_bm25_only(self, retriever: HybridRetriever) -> None:
        memory = create_memory("use snake_case for python identifiers")
        retriever.add_memory(memory)  # no embedding stored

        score = retriever._calculate_semantic_score(memory, "snake_case", [0.1] * 384)
        expected = retriever._bm25.score_document(memory.id, "snake_case")
        assert score == pytest.approx(expected / (expected + 1.0))

    def test_bm25_still_discriminates_without_vectors(
        self, retriever: HybridRetriever
    ) -> None:
        hit = create_memory("clear the pytest cache when tests fail randomly")
        miss = create_memory("use snake_case for python identifiers")
        for m in (hit, miss):
            retriever.add_memory(m)

        hit_score = retriever._calculate_semantic_score(hit, "pytest cache", [])
        miss_score = retriever._calculate_semantic_score(miss, "pytest cache", [])
        assert hit_score > miss_score


class TestRelevancePool:
    """Two-stage retrieval: relevance picks the pool, the full score picks from it.

    Memories are added without vectors, so relevance is BM25 alone and each case
    can state exactly which words a memory shares with the query.
    """

    QUERY = "clear the pytest cache when tests fail randomly"

    @staticmethod
    def retriever(factor: float | None) -> HybridRetriever:
        return HybridRetriever(
            MockEmbeddingProvider(), RetrievalConfig(relevance_pool_factor=factor)
        )

    @staticmethod
    def legacy_retriever() -> HybridRetriever:
        """3.x scoring: no pool, category boosts, frequency in the score."""
        return HybridRetriever(MockEmbeddingProvider(), RetrievalConfig.legacy_3x())

    @staticmethod
    def unrelated_and_relevant() -> tuple[Memory, Memory]:
        """A well-used gotcha that shares no word with the query, and a match."""
        unrelated = create_memory(
            "Never use mutable default arguments in Python",
            category=MemoryCategory.GOTCHA,
            use_count=50,
            outcome_score=0.6,
        )
        relevant = create_memory(
            "Clear the pytest cache when tests fail randomly",
            category=MemoryCategory.CONVENTION,
        )
        return unrelated, relevant

    @staticmethod
    def graded() -> list[Memory]:
        """Memories from most to least relevant, the most relevant with a failing record."""
        return [
            create_memory("Clear the pytest cache when tests fail randomly", outcome_score=-1.0),
            create_memory("Tests fail randomly when the cache is stale", outcome_score=1.0),
            create_memory("Randomly failing tests usually share state"),
            # Shares only "the" with the query, and is boosted and heavily used.
            create_memory(
                "Pin dependency versions in the lockfile",
                category=MemoryCategory.GOTCHA,
                outcome_score=1.0,
                use_count=80,
            ),
        ]

    async def test_single_stage_can_rank_an_unrelated_memory_first(self) -> None:
        """The 3.x behaviour this option exists to change."""
        retriever = self.legacy_retriever()
        unrelated, relevant = self.unrelated_and_relevant()
        for memory in (unrelated, relevant):
            retriever.add_memory(memory)

        results = await retriever.search(self.QUERY, limit=1)

        assert results[0].memory.id == unrelated.id
        assert results[0].semantic_score == 0.0

    async def test_pool_keeps_an_unrelated_memory_out(self) -> None:
        retriever = self.retriever(1.0)
        unrelated, relevant = self.unrelated_and_relevant()
        for memory in (unrelated, relevant):
            retriever.add_memory(memory)

        results = await retriever.search(self.QUERY, limit=1)

        assert results[0].memory.id == relevant.id

    async def test_memory_with_no_relevance_is_never_returned(self) -> None:
        """Single-stage fills the limit whatever matches; the pool does not."""
        unrelated, relevant = self.unrelated_and_relevant()
        single, pooled = self.retriever(None), self.retriever(2.0)
        for retriever in (single, pooled):
            retriever.add_memory(unrelated)
            retriever.add_memory(relevant)

        assert len(await single.search(self.QUERY, limit=5)) == 2
        results = await pooled.search(self.QUERY, limit=5)
        assert [r.memory.id for r in results] == [relevant.id]

    @pytest.mark.parametrize(
        ("factor", "expected"),
        [
            # A pool the size of the limit: the other signals can only reorder,
            # so the most relevant memory stays despite failing.
            (1.0, "Clear the pytest cache when tests fail randomly"),
            # ceil(1 x 1.5) = 2: the next most relevant can replace it.
            (1.5, "Tests fail randomly when the cache is stale"),
            (2.0, "Tests fail randomly when the cache is stale"),
        ],
    )
    async def test_pool_size_decides_what_outcome_can_replace(
        self, factor: float, expected: str
    ) -> None:
        retriever = self.retriever(factor)
        for memory in self.graded():
            retriever.add_memory(memory)

        results = await retriever.search(self.QUERY, limit=1)

        assert results[0].memory.content == expected

    async def test_other_signals_still_order_the_pool(self) -> None:
        """Within the pool the full score decides, so the most relevant memory,
        which keeps failing, ranks below the next one, which keeps working."""
        retriever = self.retriever(1.0)
        memories = self.graded()
        for memory in memories:
            retriever.add_memory(memory)

        results = await retriever.search(self.QUERY, limit=3)

        assert {r.memory.id for r in results} == {m.id for m in memories[:3]}
        assert results[0].semantic_score < results[1].semantic_score
        assert [r.memory.id for r in results[:2]] == [memories[1].id, memories[0].id]

    async def test_without_the_pool_a_weak_match_wins_on_boost_and_use(self) -> None:
        """3.x scoring: category boost and use count outrank relevance."""
        retriever = self.legacy_retriever()
        memories = self.graded()
        for memory in memories:
            retriever.add_memory(memory)

        results = await retriever.search(self.QUERY, limit=1)

        assert results[0].memory.id == memories[3].id

    def test_on_by_default(self) -> None:
        assert RetrievalConfig().relevance_pool_factor == 2.0
        assert RetrievalConfig.legacy_3x().relevance_pool_factor is None

    def test_factor_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"at least 1\.0"):
            RetrievalConfig(relevance_pool_factor=0.5)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(None, 2.0), ("", 2.0), ("2", 2.0), ("3", 3.0), ("off", None), ("None", None)],
    )
    def test_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float | None
    ) -> None:
        if raw is None:
            monkeypatch.delenv("RUNTIME_MEMORY_RELEVANCE_POOL_FACTOR", raising=False)
        else:
            monkeypatch.setenv("RUNTIME_MEMORY_RELEVANCE_POOL_FACTOR", raw)

        assert RetrievalConfig.from_env().relevance_pool_factor == expected


class TestSemanticScaling:
    """Relative scaling: the best match among a query's candidates scores 1.0."""

    @staticmethod
    def keyword_retriever(
        raw: dict[str, float], scaling: str = "relative"
    ) -> HybridRetriever:
        """A retriever whose BM25 returns given raw scores, with no vectors.

        A long task prompt gives raw BM25 in the tens for most memories, which is
        tedious to build from real text; fixing the raw scores states the case.
        """
        retriever = HybridRetriever(
            MockEmbeddingProvider(), RetrievalConfig(semantic_scaling=scaling)
        )
        retriever._bm25.score_document = lambda doc_id, _query: raw.get(doc_id, 0.0)  # type: ignore[method-assign]
        return retriever

    def test_relative_is_the_default(self) -> None:
        assert RetrievalConfig().semantic_scaling == "relative"
        assert RetrievalConfig.legacy_3x().semantic_scaling == "fixed"

    def test_unknown_scaling_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="semantic_scaling"):
            RetrievalConfig(semantic_scaling="sigmoid")

    def test_best_match_scores_one_and_no_match_zero(self) -> None:
        best, weaker, none = (create_memory(text) for text in ("a", "b", "c"))
        retriever = self.keyword_retriever({best.id: 40.0, weaker.id: 10.0})

        scores = retriever._semantic_scores([best, weaker, none], "query", [])

        assert scores == pytest.approx([1.0, 0.25, 0.0])

    @pytest.mark.parametrize(("scaling", "winner"), [("relative", 0), ("fixed", 1)])
    async def test_relevance_outranks_extraction_confidence(
        self, scaling: str, winner: int
    ) -> None:
        """Without vectors, fixed scaling let confidence order saturated matches.

        Raw BM25 of 30 and 15 become 0.968 and 0.938 under fixed scaling, a gap
        the 0.55 semantic weight turns into 0.017, while confidence of 0.95
        against 0.70 moves the score by 0.025.
        """
        memories = [
            create_memory("the closer match", confidence=0.70),
            create_memory("the weaker match", confidence=0.95),
        ]
        retriever = self.keyword_retriever(
            {memories[0].id: 30.0, memories[1].id: 15.0}, scaling
        )
        for memory in memories:
            retriever.add_memory(memory)

        results = await retriever.search("a long task prompt", limit=1)

        assert results[0].memory.id == memories[winner].id

    def test_vectors_are_scaled_by_the_best_match(self) -> None:
        """Cosine is clipped at zero and divided by the best, keywords add nothing."""
        retriever = HybridRetriever(MockEmbeddingProvider(), RetrievalConfig())
        close, halfway, opposite = (create_memory(text) for text in ("x", "y", "z"))
        retriever.add_memory(close, [1.0, 0.0])
        retriever.add_memory(halfway, [0.5, 0.866])
        retriever.add_memory(opposite, [-1.0, 0.0])

        scores = retriever._semantic_scores(
            [close, halfway, opposite], "no shared words", [1.0, 0.0]
        )

        assert scores == pytest.approx([0.6, 0.3, 0.0], abs=1e-3)

    def test_a_memory_without_a_vector_is_scored_on_keywords(self) -> None:
        """As under fixed scaling, a missing vector is not counted against it."""
        retriever = HybridRetriever(MockEmbeddingProvider(), RetrievalConfig())
        embedded = create_memory("clear the pytest cache")
        bare = create_memory("clear the pytest cache")
        retriever.add_memory(embedded, [1.0, 0.0])
        retriever.add_memory(bare)

        embedded_score, bare_score = retriever._semantic_scores(
            [embedded, bare], "pytest cache", [1.0, 0.0]
        )

        assert bare_score == pytest.approx(1.0)
        assert embedded_score == pytest.approx(1.0)

    def test_fixed_scaling_is_the_3x_formula(self) -> None:
        retriever = HybridRetriever(
            MockEmbeddingProvider(), RetrievalConfig(semantic_scaling="fixed")
        )
        memory = create_memory("clear the pytest cache when tests fail")
        retriever.add_memory(memory)

        (score,) = retriever._semantic_scores([memory], "pytest cache", [])

        assert score == pytest.approx(
            retriever._calculate_semantic_score(memory, "pytest cache", [])
        )

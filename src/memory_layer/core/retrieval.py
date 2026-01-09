"""Hybrid retrieval system for Memory Layer.

Provides intelligent memory retrieval with:
- BM25 text search
- Vector similarity search
- Hybrid scoring combining multiple signals
- Recency decay
- Frequency boosting
- Category routing
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar

from memory_layer.core.embeddings import EmbeddingProvider  # noqa: TC001
from memory_layer.core.logging import get_logger
from memory_layer.core.models import Memory, MemoryCategory, SearchResult

logger = get_logger(__name__)

# Type alias for embeddings (list of floats)
EmbeddingVector = list[float]


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval system."""

    # Scoring weights (should sum to 1.0 for normalized scoring)
    semantic_weight: float = 0.5
    recency_weight: float = 0.25
    frequency_weight: float = 0.15
    outcome_weight: float = 0.1

    # BM25 parameters
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization

    # Vector search parameters
    vector_weight: float = 0.6  # Weight of vector vs BM25 in semantic score
    min_vector_similarity: float = 0.3  # Minimum similarity threshold

    # Recency decay parameters
    recency_half_life_days: float = 30.0  # Half-life for recency decay

    # Frequency boosting parameters
    frequency_log_base: float = 2.0  # Log base for frequency scaling
    max_frequency_boost: float = 2.0  # Maximum frequency boost

    # Category boosting
    category_boosts: dict[MemoryCategory, float] = field(default_factory=dict)

    # Result settings
    default_limit: int = 10
    max_limit: int = 100
    dedup_threshold: float = 0.95  # Similarity threshold for deduplication

    def __post_init__(self) -> None:
        """Set default category boosts if not provided."""
        if not self.category_boosts:
            self.category_boosts = {
                MemoryCategory.GOTCHA: 1.3,  # Boost gotchas (important warnings)
                MemoryCategory.TROUBLESHOOTING: 1.2,  # Boost troubleshooting
                MemoryCategory.DECISION: 1.1,  # Slight boost for decisions
                MemoryCategory.WORKAROUND: 1.1,  # Boost workarounds
                MemoryCategory.PREFERENCE: 1.0,
                MemoryCategory.PATTERN: 1.0,
                MemoryCategory.ARCHITECTURE: 1.0,
                MemoryCategory.CONVENTION: 0.9,
                MemoryCategory.COMMAND: 0.9,
            }


class BM25Index:
    """BM25 index for text search.

    Implements the Okapi BM25 ranking function for text retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b

        # Document data
        self._docs: dict[str, list[str]] = {}  # doc_id -> tokens
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_length: float = 0.0

        # Term statistics
        self._doc_freqs: Counter[str] = Counter()  # term -> document frequency
        self._term_freqs: dict[str, Counter[str]] = {}  # doc_id -> term -> frequency

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index.

        Args:
            doc_id: Document identifier.
            text: Document text.
        """
        tokens = self._tokenize(text)
        self._docs[doc_id] = tokens
        self._doc_lengths[doc_id] = len(tokens)

        # Update term frequencies
        term_freq: Counter[str] = Counter(tokens)
        self._term_freqs[doc_id] = term_freq

        # Update document frequencies (count each term once per doc)
        for term in set(tokens):
            self._doc_freqs[term] += 1

        # Update average document length
        self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index.

        Args:
            doc_id: Document identifier.
        """
        if doc_id not in self._docs:
            return

        # Update document frequencies
        for term in set(self._docs[doc_id]):
            self._doc_freqs[term] -= 1
            if self._doc_freqs[term] <= 0:
                del self._doc_freqs[term]

        # Remove document data
        del self._docs[doc_id]
        del self._doc_lengths[doc_id]
        del self._term_freqs[doc_id]

        # Update average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 0.0

    def clear(self) -> None:
        """Clear all documents from the index."""
        self._docs.clear()
        self._doc_lengths.clear()
        self._term_freqs.clear()
        self._doc_freqs.clear()
        self._avg_doc_length = 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for documents matching query.

        Args:
            query: Search query.
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self._docs)
        scores: dict[str, float] = {}

        for doc_id in self._docs:
            score = 0.0
            doc_length = self._doc_lengths[doc_id]
            term_freqs = self._term_freqs[doc_id]

            for term in query_tokens:
                if term not in self._doc_freqs:
                    continue

                # Document frequency
                df = self._doc_freqs[term]

                # Inverse document frequency
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

                # Term frequency in document
                tf = term_freqs.get(term, 0)

                # BM25 score component
                length_norm = 1 - self.b + self.b * (doc_length / self._avg_doc_length)
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
                score += idf * tf_component

            if score > 0:
                scores[doc_id] = score

        # Sort by score descending and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def score_document(self, doc_id: str, query: str) -> float:
        """Score a specific document against a query.

        Args:
            doc_id: Document identifier.
            query: Search query.

        Returns:
            BM25 score for the document.
        """
        if doc_id not in self._docs or not self._docs:
            return 0.0

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        n_docs = len(self._docs)
        score = 0.0
        doc_length = self._doc_lengths[doc_id]
        term_freqs = self._term_freqs[doc_id]

        for term in query_tokens:
            if term not in self._doc_freqs:
                continue

            df = self._doc_freqs[term]
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf = term_freqs.get(term, 0)
            length_norm = 1 - self.b + self.b * (doc_length / self._avg_doc_length)
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
            score += idf * tf_component

        return score

    @property
    def document_count(self) -> int:
        """Get the number of documents in the index."""
        return len(self._docs)


class HybridRetriever:
    """Hybrid retrieval system combining BM25 and vector search.

    Implements the scoring formula:
    final_score = (semantic_weight * semantic_score +
                   recency_weight * recency_score +
                   frequency_weight * frequency_score +
                   outcome_weight * outcome_score) * category_boost
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: RetrievalConfig | None = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            embedding_provider: Provider for vector embeddings.
            config: Retrieval configuration.
        """
        self.embedding_provider = embedding_provider
        self.config = config or RetrievalConfig()

        # BM25 index
        self._bm25 = BM25Index(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

        # Memory storage for retrieval
        self._memories: dict[str, Memory] = {}
        self._embeddings: dict[str, EmbeddingVector] = {}

    def add_memory(self, memory: Memory, embedding: EmbeddingVector | None = None) -> None:
        """Add a memory to the retrieval index.

        Args:
            memory: Memory to add.
            embedding: Pre-computed embedding (optional).
        """
        self._memories[memory.id] = memory
        self._bm25.add_document(memory.id, self._get_searchable_text(memory))

        if embedding:
            self._embeddings[memory.id] = embedding
        elif memory.embedding:
            self._embeddings[memory.id] = memory.embedding

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from the retrieval index.

        Args:
            memory_id: ID of memory to remove.
        """
        self._bm25.remove_document(memory_id)
        self._memories.pop(memory_id, None)
        self._embeddings.pop(memory_id, None)

    def update_memory(self, memory: Memory, embedding: EmbeddingVector | None = None) -> None:
        """Update a memory in the retrieval index.

        Args:
            memory: Updated memory.
            embedding: Updated embedding (optional).
        """
        self.remove_memory(memory.id)
        self.add_memory(memory, embedding)

    def clear(self) -> None:
        """Clear all memories from the index."""
        self._bm25.clear()
        self._memories.clear()
        self._embeddings.clear()

    def _get_searchable_text(self, memory: Memory) -> str:
        """Get searchable text from a memory.

        Args:
            memory: Memory to extract text from.

        Returns:
            Concatenated searchable text.
        """
        parts = [memory.content]
        if memory.tags:
            parts.extend(memory.tags)
        if memory.entities:
            parts.extend(memory.entities)
        return " ".join(parts)

    async def search(
        self,
        query: str,
        limit: int | None = None,
        category: MemoryCategory | None = None,
        project: str | None = None,
        include_archived: bool = False,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for relevant memories.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            category: Filter by category.
            project: Filter by project.
            include_archived: Whether to include archived memories.
            min_score: Minimum score threshold.

        Returns:
            List of search results sorted by relevance.
        """
        if not self._memories:
            return []

        limit = min(limit or self.config.default_limit, self.config.max_limit)

        # Get query embedding
        query_result = await self.embedding_provider.embed(query)
        query_embedding = query_result.embedding

        # Get candidate memories with filters
        candidates = self._get_candidates(
            category=category,
            project=project,
            include_archived=include_archived,
        )

        if not candidates:
            return []

        # Score all candidates
        results: list[SearchResult] = []
        for memory in candidates:
            result = self._score_memory(memory, query, query_embedding)
            if result.score >= min_score:
                results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate similar results
        results = self._deduplicate(results)

        return results[:limit]

    def _get_candidates(
        self,
        category: MemoryCategory | None = None,
        project: str | None = None,
        include_archived: bool = False,
    ) -> list[Memory]:
        """Get candidate memories for search.

        Args:
            category: Filter by category.
            project: Filter by project.
            include_archived: Whether to include archived memories.

        Returns:
            List of candidate memories.
        """
        candidates = []
        for memory in self._memories.values():
            # Apply filters
            if not include_archived and memory.archived:
                continue
            if category and memory.category != category:
                continue
            if project and memory.project != project:
                continue
            candidates.append(memory)
        return candidates

    def _score_memory(
        self,
        memory: Memory,
        query: str,
        query_embedding: EmbeddingVector,
    ) -> SearchResult:
        """Score a memory against a query.

        Args:
            memory: Memory to score.
            query: Search query text.
            query_embedding: Query embedding vector.

        Returns:
            SearchResult with scoring breakdown.
        """
        # Calculate semantic score (BM25 + vector)
        semantic_score = self._calculate_semantic_score(
            memory, query, query_embedding
        )

        # Calculate recency score
        recency_score = self._calculate_recency_score(memory)

        # Calculate frequency score
        frequency_score = self._calculate_frequency_score(memory)

        # Get outcome score (already in -1 to 1 range, normalize to 0-1)
        outcome_score = (memory.outcome_score + 1.0) / 2.0

        # Get category boost
        category_boost = self.config.category_boosts.get(memory.category, 1.0)

        # Calculate final weighted score
        final_score = (
            self.config.semantic_weight * semantic_score
            + self.config.recency_weight * recency_score
            + self.config.frequency_weight * frequency_score
            + self.config.outcome_weight * outcome_score
        ) * category_boost

        return SearchResult(
            memory=memory,
            score=final_score,
            semantic_score=semantic_score,
            recency_score=recency_score,
            frequency_score=frequency_score,
            category_boost=category_boost,
        )

    def _calculate_semantic_score(
        self,
        memory: Memory,
        query: str,
        query_embedding: EmbeddingVector,
    ) -> float:
        """Calculate semantic relevance score.

        Combines BM25 text matching with vector similarity.

        Args:
            memory: Memory to score.
            query: Search query text.
            query_embedding: Query embedding vector.

        Returns:
            Semantic score between 0 and 1.
        """
        # BM25 score (normalize to 0-1 range approximately)
        bm25_score = self._bm25.score_document(memory.id, query)
        # Normalize BM25 score using sigmoid-like function
        normalized_bm25 = bm25_score / (bm25_score + 1.0) if bm25_score > 0 else 0.0

        # Vector similarity score
        vector_score = 0.0
        if memory.id in self._embeddings:
            memory_embedding = self._embeddings[memory.id]
            similarity = self.embedding_provider.cosine_similarity(
                query_embedding, memory_embedding
            )
            # Convert from [-1, 1] to [0, 1] and apply threshold
            vector_score = max(0, (similarity + 1) / 2)
            if similarity < self.config.min_vector_similarity:
                vector_score *= 0.5  # Penalize low similarity

        # Combine BM25 and vector scores
        vector_weight = self.config.vector_weight
        bm25_weight = 1.0 - vector_weight

        # If no embedding available, use only BM25
        if memory.id not in self._embeddings:
            return normalized_bm25

        return bm25_weight * normalized_bm25 + vector_weight * vector_score

    def _calculate_recency_score(self, memory: Memory) -> float:
        """Calculate recency score with exponential decay.

        Uses half-life decay: score = 0.5 ^ (days_old / half_life)

        Args:
            memory: Memory to score.

        Returns:
            Recency score between 0 and 1.
        """
        now = datetime.now(UTC)
        age = now - memory.updated_at
        days_old = age.total_seconds() / 86400  # Convert to days

        half_life = self.config.recency_half_life_days
        decay = math.pow(0.5, days_old / half_life)

        return decay

    def _calculate_frequency_score(self, memory: Memory) -> float:
        """Calculate frequency boost score.

        Uses logarithmic scaling: score = log(1 + use_count) / log(1 + max_count)

        Args:
            memory: Memory to score.

        Returns:
            Frequency score between 0 and 1.
        """
        if memory.use_count == 0:
            return 0.0

        # Logarithmic scaling capped at max boost
        log_base = self.config.frequency_log_base
        score = math.log(1 + memory.use_count, log_base)

        # Normalize to 0-1 range with max boost consideration
        # Assume use_count of ~100 gives max score
        max_expected = math.log(1 + 100, log_base)
        normalized = min(score / max_expected, 1.0)

        return normalized

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove near-duplicate results.

        Args:
            results: List of search results.

        Returns:
            Deduplicated list.
        """
        if len(results) <= 1:
            return results

        threshold = self.config.dedup_threshold
        deduplicated: list[SearchResult] = []

        for result in results:
            is_duplicate = False

            # Check against already selected results
            if result.memory.id in self._embeddings:
                result_embedding = self._embeddings[result.memory.id]

                for kept in deduplicated:
                    if kept.memory.id in self._embeddings:
                        kept_embedding = self._embeddings[kept.memory.id]
                        similarity = self.embedding_provider.cosine_similarity(
                            result_embedding, kept_embedding
                        )
                        if similarity >= threshold:
                            is_duplicate = True
                            break

            if not is_duplicate:
                deduplicated.append(result)

        return deduplicated

    async def search_by_category(
        self,
        query: str,
        categories: list[MemoryCategory],
        limit_per_category: int = 3,
    ) -> dict[MemoryCategory, list[SearchResult]]:
        """Search across specific categories.

        Args:
            query: Search query.
            categories: Categories to search.
            limit_per_category: Max results per category.

        Returns:
            Dictionary mapping categories to results.
        """
        results: dict[MemoryCategory, list[SearchResult]] = {}

        for category in categories:
            category_results = await self.search(
                query=query,
                limit=limit_per_category,
                category=category,
            )
            results[category] = category_results

        return results

    async def get_context_memories(
        self,
        query: str,
        project: str | None = None,
        max_memories: int = 10,
        category_distribution: dict[MemoryCategory, int] | None = None,
    ) -> list[SearchResult]:
        """Get memories for context injection.

        Retrieves a balanced set of memories across categories.

        Args:
            query: Context query (e.g., task description).
            project: Project filter.
            max_memories: Maximum total memories.
            category_distribution: Optional category -> count mapping.

        Returns:
            List of context-relevant memories.
        """
        if category_distribution is None:
            # Default distribution prioritizing important categories
            category_distribution = {
                MemoryCategory.GOTCHA: 2,
                MemoryCategory.TROUBLESHOOTING: 2,
                MemoryCategory.DECISION: 2,
                MemoryCategory.PATTERN: 2,
                MemoryCategory.PREFERENCE: 1,
                MemoryCategory.ARCHITECTURE: 1,
            }

        all_results: list[SearchResult] = []

        for category, count in category_distribution.items():
            if count <= 0:
                continue

            results = await self.search(
                query=query,
                limit=count,
                category=category,
                project=project,
            )
            all_results.extend(results)

        # Sort by score and limit
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:max_memories]

    @property
    def memory_count(self) -> int:
        """Get the number of indexed memories."""
        return len(self._memories)

    @property
    def indexed_with_embeddings(self) -> int:
        """Get the number of memories with embeddings."""
        return len(self._embeddings)


class CategoryRouter:
    """Routes queries to relevant categories based on content analysis."""

    # Keywords that suggest specific categories
    CATEGORY_KEYWORDS: ClassVar[dict[MemoryCategory, set[str]]] = {
        MemoryCategory.GOTCHA: {
            "gotcha", "watch out", "careful", "warning", "trap", "pitfall",
            "don't", "avoid", "never", "beware", "caution",
        },
        MemoryCategory.TROUBLESHOOTING: {
            "error", "exception", "bug", "fix", "crash", "fail", "issue",
            "traceback", "stack", "debug", "broken", "troubleshoot",
        },
        MemoryCategory.DECISION: {
            "decided", "chose", "decision", "why", "because", "rationale",
            "trade-off", "tradeoff", "alternative", "option",
        },
        MemoryCategory.PREFERENCE: {
            "prefer", "like", "want", "style", "always",
            "usually", "habit", "favorite",
        },
        MemoryCategory.PATTERN: {
            "pattern", "approach", "method", "technique", "way to",
            "how to", "best practice", "idiom",
        },
        MemoryCategory.ARCHITECTURE: {
            "architecture", "design", "structure", "component", "module",
            "layer", "system", "interface", "api",
        },
        MemoryCategory.CONVENTION: {
            "convention", "standard", "naming", "format", "rule",
            "guideline", "code style",
        },
        MemoryCategory.COMMAND: {
            "command", "cli", "terminal", "shell", "npm", "script",
            "run", "execute", "install",
        },
        MemoryCategory.WORKAROUND: {
            "workaround", "hack", "temporary", "quick fix", "bypass",
            "monkey patch", "until",
        },
    }

    def __init__(self) -> None:
        """Initialize the category router."""
        # Build reverse lookup for efficiency
        self._keyword_to_categories: dict[str, list[MemoryCategory]] = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword not in self._keyword_to_categories:
                    self._keyword_to_categories[keyword] = []
                self._keyword_to_categories[keyword].append(category)

    def route_query(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[MemoryCategory, float]]:
        """Route a query to likely relevant categories.

        Args:
            query: Search query.
            top_k: Number of top categories to return.

        Returns:
            List of (category, confidence) tuples.
        """
        query_lower = query.lower()
        category_scores: Counter[MemoryCategory] = Counter()

        # Check for keyword matches
        for keyword, categories in self._keyword_to_categories.items():
            if keyword in query_lower:
                for category in categories:
                    category_scores[category] += 1

        if not category_scores:
            # No matches, return general categories
            return [
                (MemoryCategory.PATTERN, 0.5),
                (MemoryCategory.DECISION, 0.5),
                (MemoryCategory.GOTCHA, 0.5),
            ]

        # Normalize scores
        max_score = max(category_scores.values())
        results = [
            (cat, score / max_score)
            for cat, score in category_scores.most_common(top_k)
        ]

        return results

    def get_boost_for_query(
        self,
        query: str,
        category: MemoryCategory,
    ) -> float:
        """Get category boost based on query relevance.

        Args:
            query: Search query.
            category: Category to check.

        Returns:
            Boost factor (1.0 = no boost).
        """
        routed = self.route_query(query, top_k=9)
        for cat, confidence in routed:
            if cat == category:
                # Scale confidence to boost (1.0 to 1.5)
                return 1.0 + (confidence * 0.5)
        return 1.0

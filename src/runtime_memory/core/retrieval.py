"""Hybrid retrieval system for Runtime Memory.

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
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from runtime_memory.core.embeddings import EmbeddingProvider  # noqa: TC001
from runtime_memory.core.logging import get_logger
from runtime_memory.core.models import Memory, MemoryCategory, SearchResult
from runtime_memory.core.outcomes import OutcomeModel

logger = get_logger(__name__)

# Type alias for embeddings (list of floats)
EmbeddingVector = list[float]


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval system."""

    # Scoring weights. These are the five signals the README and the design
    # notes document, and they sum to 1.0. RetrievalSettings in core/config.py
    # carries the same numbers for settings-file configuration; a test asserts
    # the two agree, because they silently disagreed once.
    # 4.0.0 changed these. Frequency left the default score: it rewards having
    # been retrieved, which is not evidence of having helped, and the Tier 2
    # evaluation caught it holding a failing memory in the top 8 through a whole
    # sequence and starving a correct one that had taken no outcome. Its weight
    # went to semantic, which is the only signal the query touches.
    # RetrievalConfig.legacy_3x() restores the 3.x numbers and boosts.
    semantic_weight: float = 0.55
    outcome_weight: float = 0.25
    recency_weight: float = 0.10
    frequency_weight: float = 0.0
    confidence_weight: float = 0.10

    # BM25 parameters
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization

    # Vector search parameters
    vector_weight: float = 0.6  # Weight of vector vs BM25 in semantic score
    # Cosine below this is halved. Fixed scaling only; relative scaling divides
    # by the best match instead.
    min_vector_similarity: float = 0.3

    # How the two semantic parts are brought to 0-1 before they are combined.
    # "relative", the default since 4.0.0, divides each by the best match among
    # the query's candidates: BM25 by the highest BM25, cosine (clipped at zero)
    # by the highest cosine. The best match scores 1.0 and no match still scores
    # 0.0. "fixed" is the 3.x scaling, BM25 / (BM25 + 1) and (cos + 1) / 2. A
    # task-length query gives raw BM25 of 2 to 69, so the first sits near 1 for
    # nearly every memory, and the second halves the spread of cosine: the
    # semantic score of the top 16 candidates then spans about 0.04 without
    # vectors, which is less than confidence's 0.10 weight moves a memory, so
    # the extractor's self-reported confidence ordered the pool, not relevance.
    # The order by semantic score alone is the same either way; what changes is
    # how much it counts against the other signals.
    semantic_scaling: str = "relative"

    # How a memory's outcome record becomes its outcome signal. Since 4.0.0 an
    # OutcomeModel reads the decayed worked and failed counts at search time, so
    # old evidence fades between outcomes and one observation counts for less
    # than ten. None reads the stored outcome score, which is what 3.x ranked on.
    outcome_model: OutcomeModel | None = field(default_factory=OutcomeModel)

    # A memory whose outcome record reads this or worse is not retrieved. Under
    # the default model that is two failures and no successes: one failure, at
    # -0.43, may be a misattribution and does not gate. Ranking alone cannot
    # drop such a memory once relevance is scaled to the query's best match,
    # because a lead in relevance outweighs any outcome record, so a memory the
    # query matches best would keep being injected however often it failed. As
    # its failures decay the record climbs back past the gate and the memory
    # can be retrieved again. None turns the gate off; it applies only with an
    # outcome_model and an outcome_weight above zero, so an evaluation arm that
    # sets the weight to 0 to switch outcome learning off switches this off too.
    failure_gate: float | None = -0.5

    # Recency decay parameters
    recency_half_life_days: float = 30.0  # Half-life for recency decay

    # Which timestamp recency decays from. Since 4.0.0 it is the memory's age,
    # from created_at. Before, it decayed from updated_at, which a retrieval and
    # every recorded outcome moved, so the signal read "last touched" and a
    # memory refreshed itself by being retrieved. False restores that.
    recency_from_created: bool = True

    # Frequency boosting parameters
    frequency_log_base: float = 2.0  # Log base for frequency scaling
    max_frequency_boost: float = 2.0  # Maximum frequency boost

    # Whether a memory's outcome record limits its frequency boost. Off by
    # default, which scores exactly as before. Retrieval counts as use, so a
    # memory that keeps being retrieved keeps gaining frequency even while every
    # outcome recorded against it is a failure, and once its outcome score is at
    # the floor of -1.0 further failures cost it nothing. With this on, the
    # frequency score is scaled by 1 + outcome_score, clamped to [0, 1]: a memory
    # at or above zero is untouched and a memory at the floor gets no boost.
    outcome_gates_frequency: bool = False

    # Two-stage retrieval. A search first keeps the ceil(limit x factor)
    # candidates with the highest semantic score, dropping any with none at all,
    # and only that pool competes on the full score. The other signals then
    # choose among relevant memories rather than decide whether an unrelated one
    # gets in. At 1.0 they can only reorder; above it they can replace a
    # relevant memory that keeps failing with the next one. None, the 3.x
    # behaviour, lets every candidate compete on the full score, of which only
    # the semantic signal depends on the query, so category boost, outcome and
    # frequency can seat a memory that barely matches the query above one that
    # matches it well: the Tier 2 evaluation found a gotcha ranked 25th of 38 on
    # relevance injected first. On by default since 4.0.0.
    relevance_pool_factor: float | None = 2.0

    # Category boosting. Empty means every category scores at 1.0, the default
    # since 4.0.0. The 3.x boosts decided which memories were injected in two
    # Tier 2 runs: once seating a wrong memory at rank 1 that ranked about 25th
    # on relevance, once keeping the one memory that would have held a rule out
    # of every prompt because its category was multiplied by 0.9.
    category_boosts: dict[MemoryCategory, float] = field(default_factory=dict)

    # Result settings
    default_limit: int = 10
    max_limit: int = 100
    dedup_threshold: float = 0.95  # Similarity threshold for deduplication

    def __post_init__(self) -> None:
        """Check the pool factor and the semantic scaling.

        Raises:
            ValueError: If ``relevance_pool_factor`` is below 1.0, which would make
                the pool smaller than the limit it is meant to fill, or if
                ``semantic_scaling`` is neither ``relative`` nor ``fixed``.
        """
        if self.relevance_pool_factor is not None and self.relevance_pool_factor < 1.0:
            raise ValueError(
                f"relevance_pool_factor must be at least 1.0, got {self.relevance_pool_factor}"
            )
        if self.semantic_scaling not in ("relative", "fixed"):
            raise ValueError(
                f"semantic_scaling must be 'relative' or 'fixed', got {self.semantic_scaling!r}"
            )

    @classmethod
    def legacy_3x(cls, **overrides: Any) -> RetrievalConfig:
        """The scoring 3.x shipped: category boosts, no pool, fixed semantic scaling.

        Kept so evaluations run against 3.x remain reproducible, and so a caller
        who tuned for those numbers can ask for them by name.

        Args:
            **overrides: Any field to set instead of the 3.x value.

        Returns:
            A config scoring as 3.1.0 did.
        """
        defaults: dict[str, Any] = {
            "semantic_weight": 0.35,
            "outcome_weight": 0.25,
            "recency_weight": 0.15,
            "frequency_weight": 0.15,
            "confidence_weight": 0.10,
            "relevance_pool_factor": None,
            "recency_from_created": False,
            "semantic_scaling": "fixed",
            "outcome_model": None,
            "failure_gate": None,
            "category_boosts": {
                MemoryCategory.GOTCHA: 1.3,
                MemoryCategory.TROUBLESHOOTING: 1.2,
                MemoryCategory.DECISION: 1.1,
                MemoryCategory.WORKAROUND: 1.1,
                MemoryCategory.PREFERENCE: 1.0,
                MemoryCategory.PATTERN: 1.0,
                MemoryCategory.ARCHITECTURE: 1.0,
                MemoryCategory.CONVENTION: 0.9,
                MemoryCategory.COMMAND: 0.9,
            },
        }
        return cls(**{**defaults, **overrides})

    @classmethod
    def from_env(cls, **overrides: Any) -> RetrievalConfig:
        """Build a config, letting the environment override any signal weight.

        Reads ``RUNTIME_MEMORY_SEMANTIC_WEIGHT`` and the matching names for
        outcome, recency, frequency and confidence. Weights are applied as
        given, never renormalised: an ablation that zeroes one signal should
        leave the others exactly where they were, not silently reweight them.
        ``RUNTIME_MEMORY_RELEVANCE_POOL_FACTOR`` sets the two-stage pool's
        factor; ``off`` or ``none`` turns the pool off, and unset or empty keeps
        the default.

        Args:
            **overrides: Field values that win over both defaults and env.

        Returns:
            A config with any environment overrides applied.
        """
        values: dict[str, Any] = {}
        for signal in ("semantic", "outcome", "recency", "frequency", "confidence"):
            raw = os.environ.get(f"RUNTIME_MEMORY_{signal.upper()}_WEIGHT")
            if raw is not None:
                values[f"{signal}_weight"] = float(raw)
        pool = os.environ.get("RUNTIME_MEMORY_RELEVANCE_POOL_FACTOR", "").strip()
        if pool.lower() in ("off", "none"):
            values["relevance_pool_factor"] = None
        elif pool:
            values["relevance_pool_factor"] = float(pool)
        values.update(overrides)
        return cls(**values)


def _normalized_text(text: str) -> str:
    """Text with case, runs of whitespace and end punctuation set aside."""
    return " ".join(text.lower().split()).strip(" .!?;:,")


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
                   outcome_weight * outcome_score +
                   recency_weight * recency_score +
                   frequency_weight * frequency_score +
                   confidence_weight * confidence) * category_boost
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

    def find_same_content(self, content: str, project: str | None) -> Memory | None:
        """Find a live memory in the same project whose text is the same as this.

        The same once case, runs of whitespace and end punctuation are set aside,
        and nothing looser: see ``EngineConfig.skip_exact_duplicates``.

        Args:
            content: Text about to be stored.
            project: Its project, None for global.

        Returns:
            The first such memory indexed, or None.
        """
        wanted = _normalized_text(content)
        for memory in self._memories.values():
            if (
                not memory.archived
                and memory.project == project
                and _normalized_text(memory.content) == wanted
            ):
                return memory
        return None

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

        # See RetrievalConfig.failure_gate. Gated before relevance is decided, so
        # the next most relevant memory takes the gated one's place in the pool.
        # Outcome weight 0 means outcome learning is off, so the gate is off too.
        if (
            self.config.failure_gate is not None
            and self.config.outcome_model is not None
            and self.config.outcome_weight > 0
        ):
            candidates = [m for m in candidates if self._outcome(m) > self.config.failure_gate]

        if not candidates:
            return []

        # Score all candidates. Relative scaling needs the best match among them,
        # so the semantic scores are computed together before anything is combined.
        semantic_scores = self._semantic_scores(candidates, query, query_embedding)
        results = [
            self._score_memory(memory, semantic)
            for memory, semantic in zip(candidates, semantic_scores, strict=True)
        ]

        # See RetrievalConfig.relevance_pool_factor. Relevance is decided before
        # min_score, so a relevant memory with a poor record is not replaced in
        # the pool by a less relevant one that happens to clear the threshold.
        if self.config.relevance_pool_factor is not None:
            results = self._relevance_pool(results, limit)

        results = [result for result in results if result.score >= min_score]

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate similar results
        results = self._deduplicate(results)

        return results[:limit]

    def _relevance_pool(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        """Keep the candidates most relevant to the query, the first retrieval stage.

        Args:
            results: Scored candidates.
            limit: The number of results the search will return.

        Returns:
            Up to ``ceil(limit x relevance_pool_factor)`` results with a semantic
            score above zero, most relevant first.
        """
        factor = self.config.relevance_pool_factor or 1.0
        size = math.ceil(limit * factor)
        relevant = [result for result in results if result.semantic_score > 0]
        relevant.sort(key=lambda r: r.semantic_score, reverse=True)
        return relevant[:size]

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

    def _semantic_scores(
        self,
        candidates: list[Memory],
        query: str,
        query_embedding: EmbeddingVector,
    ) -> list[float]:
        """Score every candidate's relevance to the query, as configured.

        See ``RetrievalConfig.semantic_scaling``. Under relative scaling each part
        is divided by its best value among these candidates, so the best keyword
        match and the best vector match each score 1.0 and no match scores 0.0.
        A memory without a vector, or a query without one, is scored on keywords
        alone, as under fixed scaling.

        Args:
            candidates: Memories that passed the search's filters.
            query: Search query text.
            query_embedding: Query embedding vector, empty without a backend.

        Returns:
            One semantic score in 0-1 per candidate, in the same order.
        """
        if self.config.semantic_scaling == "fixed":
            return [
                self._calculate_semantic_score(memory, query, query_embedding)
                for memory in candidates
            ]

        keyword = [self._bm25.score_document(memory.id, query) for memory in candidates]
        vector = [self._vector_similarity(memory, query_embedding) for memory in candidates]
        best_keyword = max(keyword, default=0.0)
        best_vector = max((max(v, 0.0) for v in vector if v is not None), default=0.0)
        vector_weight = self.config.vector_weight

        scores = []
        for keyword_score, similarity in zip(keyword, vector, strict=True):
            keyword_part = keyword_score / best_keyword if best_keyword > 0 else 0.0
            if similarity is None:
                scores.append(keyword_part)
                continue
            vector_part = max(similarity, 0.0) / best_vector if best_vector > 0 else 0.0
            scores.append((1.0 - vector_weight) * keyword_part + vector_weight * vector_part)
        return scores

    def _vector_similarity(
        self, memory: Memory, query_embedding: EmbeddingVector
    ) -> float | None:
        """Cosine similarity of a memory to the query, or None without both vectors."""
        if not query_embedding or memory.id not in self._embeddings:
            return None
        return self.embedding_provider.cosine_similarity(
            query_embedding, self._embeddings[memory.id]
        )

    def _score_memory(self, memory: Memory, semantic_score: float) -> SearchResult:
        """Combine a memory's signals into its search score.

        Args:
            memory: Memory to score.
            semantic_score: Its relevance to the query, from ``_semantic_scores``.

        Returns:
            SearchResult with scoring breakdown.
        """
        # Calculate recency score
        recency_score = self._calculate_recency_score(memory)

        # Calculate frequency score
        frequency_score = self._calculate_frequency_score(memory)

        # Outcome in -1 to 1, normalized to 0-1.
        outcome_score = (self._outcome(memory) + 1.0) / 2.0

        # Extraction confidence, already 0-1
        confidence_score = memory.confidence

        # Get category boost
        category_boost = self.config.category_boosts.get(memory.category, 1.0)

        # Calculate final weighted score
        final_score = (
            self.config.semantic_weight * semantic_score
            + self.config.outcome_weight * outcome_score
            + self.config.recency_weight * recency_score
            + self.config.frequency_weight * frequency_score
            + self.config.confidence_weight * confidence_score
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
        """Calculate one memory's semantic score under fixed scaling.

        Combines BM25 text matching with vector similarity, each scaled on its
        own: see ``RetrievalConfig.semantic_scaling``.

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

        # Without both a query vector and a stored vector there is nothing to
        # compare, so score on keyword matching alone. This is the path taken
        # when no embedding backend is installed, and also when a memory
        # predates the current provider.
        if not query_embedding or memory.id not in self._embeddings:
            return normalized_bm25

        # Vector similarity score
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

        return bm25_weight * normalized_bm25 + vector_weight * vector_score

    def _outcome(self, memory: Memory) -> float:
        """A memory's outcome signal in -1 to 1. See ``RetrievalConfig.outcome_model``."""
        model = self.config.outcome_model
        if model is None:
            return memory.outcome_score
        return model.score(*model.evidence(memory, datetime.now(UTC)))

    def _calculate_recency_score(self, memory: Memory) -> float:
        """Calculate recency score with exponential decay.

        Uses half-life decay: score = 0.5 ^ (days_old / half_life), from the
        memory's creation unless ``recency_from_created`` is off.

        Args:
            memory: Memory to score.

        Returns:
            Recency score between 0 and 1.
        """
        now = datetime.now(UTC)
        stamp = memory.created_at if self.config.recency_from_created else memory.updated_at
        age = now - stamp
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

        # See RetrievalConfig.outcome_gates_frequency. Without the gate, being
        # retrieved pays the same whether the memory helped or kept failing.
        if self.config.outcome_gates_frequency:
            normalized *= min(1.0, max(0.0, 1.0 + memory.outcome_score))

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

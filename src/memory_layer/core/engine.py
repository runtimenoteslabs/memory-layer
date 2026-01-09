"""Memory Engine orchestrator for Memory Layer.

The MemoryEngine is the main orchestrator that coordinates:
- Storage layer for persistence
- Embedding provider for vector embeddings
- Hybrid retriever for intelligent search
- Outcome tracking for learning from feedback
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory_layer.core.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    MockEmbeddingProvider,
    get_embedding_provider,
)
from memory_layer.core.logging import get_logger
from memory_layer.core.models import (
    ContextResponse,
    Memory,
    MemoryCategory,
    MemoryCreate,
    MemoryScope,
    MemorySource,
    MemoryUpdate,
    Outcome,
    SearchResult,
)
from memory_layer.core.retrieval import HybridRetriever, RetrievalConfig
from memory_layer.core.storage import (
    MemoryNotFoundError,
    MemoryStorage,
    StorageStats,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Type alias to avoid conflict with list() method
_List = builtins.list

logger = get_logger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the Memory Engine."""

    # Storage settings
    db_path: str | Path = "~/.memory-layer/memories.db"
    """Path to SQLite database file."""

    pool_size: int = 5
    """Number of database connections in the pool."""

    secure_permissions: bool = True
    """Whether to set secure file permissions (0600) on database."""

    # Embedding settings
    embedding_provider: str = "local"
    """Embedding provider type: 'local', 'openai', 'voyage', 'mock'."""

    embedding_config: EmbeddingConfig | None = None
    """Optional embedding provider configuration."""

    # Retrieval settings
    retrieval_config: RetrievalConfig | None = None
    """Optional retrieval configuration."""

    # Auto-archival settings
    auto_archive_enabled: bool = True
    """Whether to automatically archive low-score memories."""

    auto_archive_threshold: float = -0.5
    """Outcome score threshold below which memories are archived."""

    auto_archive_on_search: bool = False
    """Whether to run auto-archival after searches (can impact performance)."""

    # Last search tracking
    track_last_search: bool = True
    """Whether to track the last search for easy outcome recording."""

    def __post_init__(self) -> None:
        """Expand paths and set defaults."""
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path).expanduser()


@dataclass
class EngineStats:
    """Statistics from the Memory Engine."""

    storage_stats: StorageStats
    """Underlying storage statistics."""

    indexed_memories: int
    """Number of memories indexed in retriever."""

    indexed_with_embeddings: int
    """Number of memories with embeddings in retriever."""

    last_search_result_count: int
    """Number of results from last search (if tracked)."""


@dataclass
class LastSearchInfo:
    """Information about the last search performed."""

    query: str
    """The search query."""

    results: _List[SearchResult]
    """Search results."""

    memory_ids: _List[str] = field(default_factory=list)
    """IDs of memories returned in search results."""

    def __post_init__(self) -> None:
        """Extract memory IDs from results."""
        self.memory_ids = [r.memory.id for r in self.results]


class EngineError(Exception):
    """Base exception for engine errors."""

    pass


class EngineNotInitializedError(EngineError):
    """Raised when engine is used before initialization."""

    pass


class MemoryEngine:
    """Main orchestrator for the Memory Layer.

    Coordinates storage, embeddings, and retrieval to provide
    a unified interface for memory operations with outcome-based learning.

    Example:
        ```python
        engine = MemoryEngine()
        await engine.initialize()

        # Add a memory
        memory = await engine.add(
            content="Use async/await for I/O operations",
            category=MemoryCategory.PATTERN,
            project="my-project",
        )

        # Search for relevant memories
        results = await engine.search("async patterns")

        # Record outcome feedback
        await engine.record_outcome([results[0].memory.id], Outcome.WORKED)

        # Get context for injection
        context = await engine.get_context(project="my-project")
        print(context.to_markdown())

        await engine.close()
        ```
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        storage: MemoryStorage | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        """Initialize the Memory Engine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
            storage: Optional pre-configured storage instance.
            embedding_provider: Optional pre-configured embedding provider.
            retriever: Optional pre-configured retriever.
        """
        self.config = config or EngineConfig()

        # Components (initialized lazily or provided)
        self._storage = storage
        self._embedding_provider = embedding_provider
        self._retriever = retriever

        # State
        self._initialized = False
        self._last_search: LastSearchInfo | None = None

    async def initialize(self) -> None:
        """Initialize the engine and all components.

        Creates and initializes storage, embedding provider, and retriever.
        Also loads existing memories into the retriever index.
        """
        if self._initialized:
            return

        logger.info("Initializing Memory Engine...")

        # Initialize storage
        if self._storage is None:
            self._storage = MemoryStorage(
                db_path=self.config.db_path,
                pool_size=self.config.pool_size,
                secure_permissions=self.config.secure_permissions,
            )
        await self._storage.initialize()

        # Initialize embedding provider
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider(
                provider_type=self.config.embedding_provider,
                config=self.config.embedding_config,
            )

        # Initialize retriever
        if self._retriever is None:
            self._retriever = HybridRetriever(
                embedding_provider=self._embedding_provider,
                config=self.config.retrieval_config,
            )

        # Load existing memories into retriever
        await self._load_memories_to_retriever()

        self._initialized = True
        logger.info("Memory Engine initialized successfully")

    async def close(self) -> None:
        """Close the engine and release resources."""
        if self._storage:
            await self._storage.close()
        self._initialized = False
        logger.info("Memory Engine closed")

    async def __aenter__(self) -> MemoryEngine:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure the engine is initialized.

        Raises:
            EngineNotInitializedError: If engine is not initialized.
        """
        if not self._initialized:
            raise EngineNotInitializedError(
                "Engine not initialized. Call initialize() first or use async context manager."
            )

    @property
    def storage(self) -> MemoryStorage:
        """Get the storage instance."""
        self._ensure_initialized()
        assert self._storage is not None
        return self._storage

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get the embedding provider instance."""
        self._ensure_initialized()
        assert self._embedding_provider is not None
        return self._embedding_provider

    @property
    def retriever(self) -> HybridRetriever:
        """Get the retriever instance."""
        self._ensure_initialized()
        assert self._retriever is not None
        return self._retriever

    @property
    def last_search(self) -> LastSearchInfo | None:
        """Get information about the last search performed."""
        return self._last_search

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def add(
        self,
        content: str,
        category: MemoryCategory,
        project: str | None = None,
        scope: MemoryScope = MemoryScope.PROJECT,
        source: MemorySource = MemorySource.EXPLICIT,
        confidence: float = 1.0,
        importance: float = 0.5,
        tags: _List[str] | None = None,
        entities: _List[str] | None = None,
        supersedes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a new memory.

        Creates a memory, generates embedding, stores in database, and
        indexes in retriever.

        Args:
            content: The memory content text.
            category: Classification category.
            project: Project scope (None for global).
            scope: Visibility scope.
            source: How the memory was created.
            confidence: Source reliability (0.0-1.0).
            importance: Importance weight (0.0-1.0).
            tags: Optional tags for categorization.
            entities: Detected entities (files, functions, etc.).
            supersedes: ID of memory this one replaces.
            metadata: Additional metadata.

        Returns:
            The created Memory.
        """
        self._ensure_initialized()

        # Validate using Pydantic model
        create_model = MemoryCreate(
            content=content,
            category=category,
            project=project,
            scope=scope,
            source=source,
            confidence=confidence,
            importance=importance,
            tags=tags or [],
            entities=entities or [],
            supersedes=supersedes,
            metadata=metadata or {},
        )

        # Convert to Memory dataclass
        memory = create_model.to_memory()

        # Generate embedding
        embedding_result = await self._embedding_provider.embed(memory.content)
        memory.embedding = embedding_result.embedding

        # Store in database
        await self._storage.create(memory)

        # Add to retriever index
        self._retriever.add_memory(memory, embedding_result.embedding)

        # Archive superseded memory if specified
        if supersedes:
            try:
                await self._storage.archive(supersedes)
                self._retriever.remove_memory(supersedes)
            except MemoryNotFoundError:
                logger.warning(f"Superseded memory {supersedes} not found")

        logger.debug(f"Added memory {memory.id}: {content[:50]}...")
        return memory

    async def add_memory(self, memory: Memory) -> Memory:
        """Add an existing Memory object.

        Useful when you have a pre-constructed Memory object.

        Args:
            memory: The Memory to add.

        Returns:
            The stored Memory (with embedding if not present).
        """
        self._ensure_initialized()

        # Generate embedding if not present
        if memory.embedding is None:
            embedding_result = await self._embedding_provider.embed(memory.content)
            memory.embedding = embedding_result.embedding

        # Store in database
        await self._storage.create(memory)

        # Add to retriever index
        self._retriever.add_memory(memory, memory.embedding)

        logger.debug(f"Added memory {memory.id}")
        return memory

    async def add_many(self, memories: Sequence[Memory]) -> _List[Memory]:
        """Add multiple memories in a batch.

        Args:
            memories: Memories to add.

        Returns:
            List of stored memories.
        """
        self._ensure_initialized()

        # Generate embeddings for memories without them
        texts_to_embed = []
        indices_to_embed = []
        for i, memory in enumerate(memories):
            if memory.embedding is None:
                texts_to_embed.append(memory.content)
                indices_to_embed.append(i)

        if texts_to_embed:
            batch_result = await self._embedding_provider.embed_many(texts_to_embed)
            for i, embedding in zip(indices_to_embed, batch_result.embeddings, strict=True):
                memories[i].embedding = embedding

        # Store all in database
        memories_list = list(memories)
        await self._storage.create_many(memories_list)

        # Add all to retriever
        for memory in memories_list:
            self._retriever.add_memory(memory, memory.embedding)

        logger.debug(f"Added {len(memories_list)} memories")
        return memories_list

    async def get(self, memory_id: str) -> Memory:
        """Get a memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The Memory.

        Raises:
            MemoryNotFoundError: If memory not found.
        """
        self._ensure_initialized()
        return await self._storage.get(memory_id)

    async def get_many(self, memory_ids: _List[str]) -> _List[Memory]:
        """Get multiple memories by ID.

        Args:
            memory_ids: List of memory IDs.

        Returns:
            List of memories (preserves order, excludes missing).
        """
        self._ensure_initialized()
        return await self._storage.get_many(memory_ids)

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        category: MemoryCategory | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        tags: _List[str] | None = None,
        entities: _List[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Update an existing memory.

        Only provided fields are updated. If content is changed, a new
        embedding is generated.

        Args:
            memory_id: ID of memory to update.
            content: New content (optional).
            category: New category (optional).
            confidence: New confidence (optional).
            importance: New importance (optional).
            tags: New tags (optional).
            entities: New entities (optional).
            metadata: New metadata (optional).

        Returns:
            The updated Memory.

        Raises:
            MemoryNotFoundError: If memory not found.
        """
        self._ensure_initialized()

        # Get existing memory
        memory = await self._storage.get(memory_id)

        # Apply updates using Pydantic model
        update_model = MemoryUpdate(
            content=content,
            category=category,
            confidence=confidence,
            importance=importance,
            tags=tags,
            entities=entities,
            metadata=metadata,
        )
        memory = update_model.apply_to(memory)

        # Re-generate embedding if content changed
        new_embedding = None
        if content is not None:
            embedding_result = await self._embedding_provider.embed(memory.content)
            memory.embedding = embedding_result.embedding
            new_embedding = embedding_result.embedding

        # Update in database
        await self._storage.update(memory)

        # Update in retriever
        self._retriever.update_memory(memory, new_embedding)

        logger.debug(f"Updated memory {memory_id}")
        return memory

    async def delete(
        self,
        memory_id: str,
        hard_delete: bool = False,
    ) -> None:
        """Delete a memory.

        By default, performs a soft delete (archive). Use hard_delete=True
        to permanently remove the memory.

        Args:
            memory_id: ID of memory to delete.
            hard_delete: If True, permanently delete. If False, archive.

        Raises:
            MemoryNotFoundError: If memory not found.
        """
        self._ensure_initialized()

        await self._storage.delete(memory_id, hard_delete=hard_delete)
        self._retriever.remove_memory(memory_id)

        action = "deleted" if hard_delete else "archived"
        logger.debug(f"Memory {memory_id} {action}")

    async def archive(self, memory_id: str) -> Memory:
        """Archive a memory (soft delete).

        Args:
            memory_id: ID of memory to archive.

        Returns:
            The archived Memory.
        """
        self._ensure_initialized()

        memory = await self._storage.archive(memory_id)
        self._retriever.remove_memory(memory_id)

        logger.debug(f"Archived memory {memory_id}")
        return memory

    async def unarchive(self, memory_id: str) -> Memory:
        """Unarchive a previously archived memory.

        Args:
            memory_id: ID of memory to unarchive.

        Returns:
            The unarchived Memory.
        """
        self._ensure_initialized()

        memory = await self._storage.unarchive(memory_id)

        # Re-add to retriever
        self._retriever.add_memory(memory, memory.embedding)

        logger.debug(f"Unarchived memory {memory_id}")
        return memory

    # =========================================================================
    # Search and Retrieval
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
        category: MemoryCategory | None = None,
        project: str | None = None,
        include_archived: bool = False,
        min_score: float = 0.0,
        track_usage: bool = True,
    ) -> _List[SearchResult]:
        """Search for relevant memories.

        Uses hybrid retrieval combining BM25 text search and vector
        similarity for intelligent ranking.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            category: Filter by category.
            project: Filter by project.
            include_archived: Whether to include archived memories.
            min_score: Minimum relevance score threshold.
            track_usage: Whether to increment use counts for returned memories.

        Returns:
            List of SearchResults sorted by relevance.
        """
        self._ensure_initialized()

        # Execute search
        results = await self._retriever.search(
            query=query,
            limit=limit,
            category=category,
            project=project,
            include_archived=include_archived,
            min_score=min_score,
        )

        # Track last search if enabled
        if self.config.track_last_search:
            self._last_search = LastSearchInfo(query=query, results=results)

        # Increment use counts for returned memories
        if track_usage and results:
            memory_ids = [r.memory.id for r in results]
            await self._storage.increment_use_counts(memory_ids)

        # Run auto-archival if enabled
        if self.config.auto_archive_enabled and self.config.auto_archive_on_search:
            await self.auto_archive(project=project)

        logger.debug(f"Search for '{query}' returned {len(results)} results")
        return results

    async def list(
        self,
        project: str | None = None,
        category: MemoryCategory | None = None,
        scope: MemoryScope | None = None,
        source: MemorySource | None = None,
        include_archived: bool = False,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> _List[Memory]:
        """List memories with filtering.

        Args:
            project: Filter by project.
            category: Filter by category.
            scope: Filter by scope.
            source: Filter by source.
            include_archived: Whether to include archived memories.
            min_score: Minimum outcome score filter.
            max_score: Maximum outcome score filter.
            limit: Maximum results to return.
            offset: Results offset for pagination.
            order_by: Column to order by.
            descending: Whether to sort descending.

        Returns:
            List of memories matching filters.
        """
        self._ensure_initialized()

        return await self._storage.list(
            project=project,
            category=category,
            scope=scope,
            source=source,
            include_archived=include_archived,
            min_score=min_score,
            max_score=max_score,
            limit=limit,
            offset=offset,
            order_by=order_by,
            descending=descending,
        )

    # =========================================================================
    # Outcome Tracking and Learning
    # =========================================================================

    async def record_outcome(
        self,
        memory_ids: _List[str] | str,
        outcome: Outcome,
    ) -> _List[Memory]:
        """Record outcome feedback for memories.

        Updates outcome scores for specified memories based on feedback.
        This enables the system to learn from experience.

        Args:
            memory_ids: Memory ID(s) to update. Can be a single ID or list.
            outcome: The outcome to record (WORKED, FAILED, PARTIAL).

        Returns:
            List of updated memories.
        """
        self._ensure_initialized()

        # Normalize to list
        if isinstance(memory_ids, str):
            memory_ids = [memory_ids]

        memories = await self._storage.record_outcomes(memory_ids, outcome)

        # Update memories in retriever
        for memory in memories:
            self._retriever.update_memory(memory, memory.embedding)

        logger.debug(f"Recorded {outcome.value} for {len(memories)} memories")
        return memories

    async def record_outcome_for_last_search(self, outcome: Outcome) -> _List[Memory]:
        """Record outcome for the memories from the last search.

        Convenience method for recording feedback on the most recent
        search results.

        Args:
            outcome: The outcome to record.

        Returns:
            List of updated memories.

        Raises:
            EngineError: If no last search is tracked.
        """
        self._ensure_initialized()

        if not self._last_search or not self._last_search.memory_ids:
            raise EngineError("No last search results to record outcome for")

        return await self.record_outcome(self._last_search.memory_ids, outcome)

    async def auto_archive(
        self,
        project: str | None = None,
        threshold: float | None = None,
    ) -> int:
        """Archive memories with low outcome scores.

        Automatically archives memories that have accumulated
        negative feedback below the threshold.

        Args:
            project: Optional project filter.
            threshold: Score threshold (default from config).

        Returns:
            Number of memories archived.
        """
        self._ensure_initialized()

        threshold = threshold or self.config.auto_archive_threshold
        count = await self._storage.archive_low_score_memories(
            threshold=threshold,
            project=project,
        )

        if count > 0:
            # Refresh retriever index to remove archived memories
            await self._refresh_retriever(project=project)
            logger.info(f"Auto-archived {count} memories below score {threshold}")

        return count

    # =========================================================================
    # Context Generation
    # =========================================================================

    async def get_context(
        self,
        query: str | None = None,
        project: str | None = None,
        max_memories: int = 10,
        category_distribution: dict[MemoryCategory, int] | None = None,
    ) -> ContextResponse:
        """Get formatted context for injection into prompts.

        Retrieves relevant memories and formats them for use as context
        in AI agent prompts.

        Args:
            query: Optional query to find relevant memories.
            project: Project filter.
            max_memories: Maximum memories to include.
            category_distribution: Optional category -> count mapping.

        Returns:
            ContextResponse with formatted context.
        """
        self._ensure_initialized()

        # Get relevant memories
        if query:
            results = await self._retriever.get_context_memories(
                query=query,
                project=project,
                max_memories=max_memories,
                category_distribution=category_distribution,
            )
            memories = [r.memory for r in results]
        else:
            # Get most recent/important memories without query
            memories = await self._storage.list(
                project=project,
                include_archived=False,
                limit=max_memories,
                order_by="outcome_score",
                descending=True,
            )

        # Count by category
        categories: dict[str, int] = {}
        for memory in memories:
            cat_name = memory.category.value
            categories[cat_name] = categories.get(cat_name, 0) + 1

        # Get total count
        total_count = await self._storage.count(project=project)

        # Build response
        response = ContextResponse(
            memories=memories,
            project=project,
            total_count=total_count,
            included_count=len(memories),
            categories=categories,
        )

        # Generate formatted string
        response.formatted = response.to_markdown()

        return response

    # =========================================================================
    # Statistics and Health
    # =========================================================================

    async def stats(self, project: str | None = None) -> EngineStats:
        """Get engine statistics.

        Args:
            project: Optional project filter.

        Returns:
            EngineStats with various metrics.
        """
        self._ensure_initialized()

        storage_stats = await self._storage.get_stats(project=project)

        return EngineStats(
            storage_stats=storage_stats,
            indexed_memories=self._retriever.memory_count,
            indexed_with_embeddings=self._retriever.indexed_with_embeddings,
            last_search_result_count=len(self._last_search.results) if self._last_search else 0,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check engine health.

        Returns:
            Health status dictionary.
        """
        status: dict[str, Any] = {
            "initialized": self._initialized,
            "storage": None,
            "embedding_provider": None,
            "retriever": None,
        }

        if not self._initialized:
            status["status"] = "not_initialized"
            return status

        # Check storage health
        status["storage"] = await self._storage.health_check()

        # Check embedding provider
        try:
            # Quick embedding test
            test_result = await self._embedding_provider.embed("test")
            status["embedding_provider"] = {
                "status": "healthy",
                "model": self._embedding_provider.model_name,
                "dimensions": test_result.dimensions,
            }
        except Exception as e:
            status["embedding_provider"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check retriever
        status["retriever"] = {
            "status": "healthy",
            "indexed_memories": self._retriever.memory_count,
            "with_embeddings": self._retriever.indexed_with_embeddings,
        }

        # Overall status
        storage_healthy = status["storage"].get("status") == "healthy"
        embedding_healthy = status["embedding_provider"].get("status") == "healthy"
        status["status"] = "healthy" if (storage_healthy and embedding_healthy) else "degraded"

        return status

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    async def _load_memories_to_retriever(self) -> None:
        """Load existing memories from storage into the retriever."""
        # Load all non-archived memories
        memories = await self._storage.list(
            include_archived=False,
            limit=10000,  # Load up to 10K memories
        )

        for memory in memories:
            self._retriever.add_memory(memory, memory.embedding)

        logger.info(f"Loaded {len(memories)} memories into retriever")

    async def _refresh_retriever(self, project: str | None = None) -> None:
        """Refresh the retriever index from storage.

        Args:
            project: Optional project filter.
        """
        # Clear and reload
        self._retriever.clear()
        await self._load_memories_to_retriever()


# Factory function for easy creation
async def create_engine(
    db_path: str | Path | None = None,
    embedding_provider: str = "local",
    **kwargs: Any,
) -> MemoryEngine:
    """Create and initialize a Memory Engine.

    Convenience factory function that creates and initializes an engine
    in one call.

    Args:
        db_path: Path to database file.
        embedding_provider: Embedding provider type.
        **kwargs: Additional EngineConfig parameters.

    Returns:
        Initialized MemoryEngine.

    Example:
        ```python
        engine = await create_engine(
            db_path="~/.my-app/memories.db",
            embedding_provider="local",
        )
        ```
    """
    config_kwargs: dict[str, Any] = {
        "embedding_provider": embedding_provider,
        **kwargs,
    }
    if db_path:
        config_kwargs["db_path"] = db_path

    config = EngineConfig(**config_kwargs)
    engine = MemoryEngine(config=config)
    await engine.initialize()
    return engine

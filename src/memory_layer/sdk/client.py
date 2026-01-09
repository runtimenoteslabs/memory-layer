"""Python SDK for Memory Layer.

This module provides a unified client interface for accessing Memory Layer
functionality in both local (direct engine access) and remote (REST API) modes.

Example usage:
    ```python
    from memory_layer.sdk import MemoryClient

    # Local mode (direct engine access)
    async with MemoryClient(mode="local") as client:
        memory = await client.add("Use async/await for I/O", category="pattern")
        results = await client.search("async patterns")

    # Remote mode (REST API)
    async with MemoryClient(mode="remote", base_url="http://localhost:8080") as client:
        memory = await client.add("Use async/await for I/O", category="pattern")
        results = await client.search("async patterns")

    # Synchronous wrapper
    from memory_layer.sdk import SyncMemoryClient
    with SyncMemoryClient(mode="local") as client:
        memory = client.add("Use async/await for I/O", category="pattern")
    ```
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx

from memory_layer.core.models import (
    ContextResponse,
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
    SearchResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from memory_layer.core.engine import EngineStats, MemoryEngine

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class ClientMode(str, Enum):
    """SDK client operating mode."""

    LOCAL = "local"
    """Direct access to MemoryEngine (no network)."""

    REMOTE = "remote"
    """Access via REST API."""


@dataclass
class ClientConfig:
    """Configuration for the Memory Client."""

    # Mode selection
    mode: ClientMode = ClientMode.LOCAL
    """Operating mode: local or remote."""

    # Local mode settings
    db_path: str | Path = "~/.memory-layer/memories.db"
    """Path to SQLite database file (local mode only)."""

    embedding_provider: str = "local"
    """Embedding provider type (local mode only)."""

    # Remote mode settings
    base_url: str = "http://127.0.0.1:8080"
    """Base URL for REST API (remote mode only)."""

    api_key: str | None = None
    """Optional API key for authentication."""

    # HTTP client settings (remote mode)
    timeout: float = 30.0
    """Request timeout in seconds."""

    max_retries: int = 3
    """Maximum retry attempts for failed requests."""

    retry_delay: float = 1.0
    """Initial delay between retries (exponential backoff)."""

    # Connection pooling
    max_connections: int = 10
    """Maximum number of concurrent connections."""

    def __post_init__(self) -> None:
        """Normalize configuration values."""
        if isinstance(self.mode, str):
            self.mode = ClientMode(self.mode)
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path).expanduser()
        # Ensure base_url doesn't have trailing slash
        self.base_url = self.base_url.rstrip("/")


# =============================================================================
# Exceptions
# =============================================================================


class SDKError(Exception):
    """Base exception for SDK errors."""

    pass


class ConnectionError(SDKError):
    """Error connecting to remote server."""

    pass


class AuthenticationError(SDKError):
    """Authentication failed."""

    pass


class NotFoundError(SDKError):
    """Resource not found."""

    pass


class ValidationError(SDKError):
    """Request validation failed."""

    pass


class RateLimitError(SDKError):
    """Rate limit exceeded."""

    pass


# =============================================================================
# Response Types (for remote mode)
# =============================================================================


@dataclass
class StatsDict:
    """Statistics response from the API."""

    total_memories: int
    active_memories: int
    archived_memories: int
    by_category: dict[str, int]
    by_scope: dict[str, int]
    by_source: dict[str, int]
    avg_outcome_score: float
    total_uses: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatsDict:
        """Create from API response."""
        return cls(
            total_memories=data.get("total_memories", 0),
            active_memories=data.get("active_memories", 0),
            archived_memories=data.get("archived_memories", 0),
            by_category=data.get("by_category", {}),
            by_scope=data.get("by_scope", {}),
            by_source=data.get("by_source", {}),
            avg_outcome_score=data.get("avg_outcome_score", 0.0),
            total_uses=data.get("total_uses", 0),
        )


# =============================================================================
# Memory Client (Async)
# =============================================================================


class MemoryClient:
    """Async client for Memory Layer.

    Provides a unified interface for memory operations in both local
    (direct engine access) and remote (REST API) modes.

    Example:
        ```python
        # Local mode
        async with MemoryClient(mode="local") as client:
            memory = await client.add(
                content="Use async/await for I/O",
                category="pattern",
            )

        # Remote mode
        async with MemoryClient(
            mode="remote",
            base_url="http://localhost:8080",
            api_key="secret",
        ) as client:
            results = await client.search("async patterns")
        ```
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        *,
        mode: str | ClientMode = ClientMode.LOCAL,
        db_path: str | Path | None = None,
        embedding_provider: str = "local",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Memory Client.

        Args:
            config: Full configuration object (overrides other params).
            mode: Operating mode: "local" or "remote".
            db_path: Database path for local mode.
            embedding_provider: Embedding provider for local mode.
            base_url: API base URL for remote mode.
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        if config:
            self.config = config
        else:
            self.config = ClientConfig(
                mode=ClientMode(mode) if isinstance(mode, str) else mode,
                db_path=db_path or "~/.memory-layer/memories.db",
                embedding_provider=embedding_provider,
                base_url=base_url or "http://127.0.0.1:8080",
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )

        # Internal state
        self._engine: MemoryEngine | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the client.

        For local mode, creates and initializes the MemoryEngine.
        For remote mode, creates the HTTP client with connection pooling.
        """
        if self._initialized:
            return

        if self.config.mode == ClientMode.LOCAL:
            await self._init_local()
        else:
            await self._init_remote()

        self._initialized = True
        logger.info(f"MemoryClient initialized in {self.config.mode.value} mode")

    async def close(self) -> None:
        """Close the client and release resources."""
        if self.config.mode == ClientMode.LOCAL and self._engine:
            await self._engine.close()
            self._engine = None
        elif self.config.mode == ClientMode.REMOTE and self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._initialized = False
        logger.info("MemoryClient closed")

    async def __aenter__(self) -> MemoryClient:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure the client is initialized."""
        if not self._initialized:
            raise SDKError(
                "Client not initialized. Call initialize() first or use async context manager."
            )

    # =========================================================================
    # Local Mode Initialization
    # =========================================================================

    async def _init_local(self) -> None:
        """Initialize local mode with MemoryEngine."""
        from memory_layer.core.engine import EngineConfig, MemoryEngine

        engine_config = EngineConfig(
            db_path=self.config.db_path,
            embedding_provider=self.config.embedding_provider,
        )
        self._engine = MemoryEngine(config=engine_config)
        await self._engine.initialize()

    # =========================================================================
    # Remote Mode Initialization and Helpers
    # =========================================================================

    async def _init_remote(self) -> None:
        """Initialize remote mode with HTTP client."""
        headers = {}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_connections,
        )

        self._http_client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.config.timeout),
            limits=limits,
        )

        # Verify connection with health check
        try:
            response = await self._http_client.get("/health")
            response.raise_for_status()
        except httpx.ConnectError as e:
            await self._http_client.aclose()
            self._http_client = None
            raise ConnectionError(f"Failed to connect to {self.config.base_url}: {e}") from e

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API path.
            json: JSON body for POST/PATCH requests.
            params: Query parameters.

        Returns:
            Response JSON as dictionary.

        Raises:
            Various SDKError subclasses for different error conditions.
        """
        assert self._http_client is not None

        last_error: Exception | None = None
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries):
            try:
                response = await self._http_client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                )

                # Handle different status codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 201:
                    return response.json()
                elif response.status_code == 204:
                    return {}
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status_code == 404:
                    raise NotFoundError(response.json().get("detail", "Not found"))
                elif response.status_code == 422:
                    raise ValidationError(response.json().get("detail", "Validation error"))
                elif response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                else:
                    response.raise_for_status()

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                continue

            except (AuthenticationError, NotFoundError, ValidationError, RateLimitError):
                raise

            except httpx.HTTPStatusError as e:
                raise SDKError(f"HTTP error: {e}") from e

        raise ConnectionError(f"Failed after {self.config.max_retries} attempts: {last_error}")

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def add(
        self,
        content: str,
        category: str | MemoryCategory,
        project: str | None = None,
        scope: str | MemoryScope = MemoryScope.PROJECT,
        source: str | MemorySource = MemorySource.EXPLICIT,
        confidence: float = 1.0,
        importance: float = 0.5,
        tags: list[str] | None = None,
        entities: list[str] | None = None,
        supersedes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a new memory.

        Args:
            content: The memory content text.
            category: Classification category (string or enum).
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

        # Normalize category to enum
        if isinstance(category, str):
            category = MemoryCategory(category)
        if isinstance(scope, str):
            scope = MemoryScope(scope)
        if isinstance(source, str):
            source = MemorySource(source)

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            return await self._engine.add(
                content=content,
                category=category,
                project=project,
                scope=scope,
                source=source,
                confidence=confidence,
                importance=importance,
                tags=tags,
                entities=entities,
                supersedes=supersedes,
                metadata=metadata,
            )
        else:
            # Remote mode
            response = await self._request(
                "POST",
                "/memories",
                json={
                    "content": content,
                    "category": category.value,
                    "project": project,
                    "tags": tags or [],
                    "importance": importance,
                    "entities": entities or [],
                },
            )
            return Memory.from_dict(response)

    async def get(self, memory_id: str) -> Memory:
        """Get a memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The Memory.

        Raises:
            NotFoundError: If memory not found.
        """
        self._ensure_initialized()

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            try:
                return await self._engine.get(memory_id)
            except Exception as e:
                if "not found" in str(e).lower():
                    raise NotFoundError(f"Memory not found: {memory_id}") from e
                raise
        else:
            response = await self._request("GET", f"/memories/{memory_id}")
            return Memory.from_dict(response)

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | MemoryCategory | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Update an existing memory.

        Only provided fields are updated.

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
            NotFoundError: If memory not found.
        """
        self._ensure_initialized()

        # Normalize category to enum
        if isinstance(category, str):
            category = MemoryCategory(category)

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            return await self._engine.update(
                memory_id=memory_id,
                content=content,
                category=category,
                confidence=confidence,
                importance=importance,
                tags=tags,
                entities=entities,
                metadata=metadata,
            )
        else:
            payload: dict[str, Any] = {}
            if content is not None:
                payload["content"] = content
            if category is not None:
                payload["category"] = category.value
            if tags is not None:
                payload["tags"] = tags
            if importance is not None:
                payload["importance"] = importance

            response = await self._request("PATCH", f"/memories/{memory_id}", json=payload)
            return Memory.from_dict(response)

    async def delete(self, memory_id: str, hard_delete: bool = False) -> None:
        """Delete a memory.

        By default, performs a soft delete (archive).

        Args:
            memory_id: ID of memory to delete.
            hard_delete: If True, permanently delete.

        Raises:
            NotFoundError: If memory not found.
        """
        self._ensure_initialized()

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            await self._engine.delete(memory_id, hard_delete=hard_delete)
        else:
            await self._request("DELETE", f"/memories/{memory_id}")

    # =========================================================================
    # Search and List
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
        categories: list[str | MemoryCategory] | None = None,
        project: str | None = None,
        min_score: float = -1.0,
        include_archived: bool = False,
    ) -> list[SearchResult]:
        """Search for relevant memories.

        Uses hybrid retrieval combining BM25 text search and vector
        similarity for intelligent ranking.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            categories: Filter by categories.
            project: Filter by project.
            min_score: Minimum outcome score threshold.
            include_archived: Whether to include archived memories.

        Returns:
            List of SearchResults sorted by relevance.
        """
        self._ensure_initialized()

        # Normalize categories
        if categories:
            categories = [
                MemoryCategory(c) if isinstance(c, str) else c
                for c in categories
            ]

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            # Note: Local mode only supports single category filter
            category = categories[0] if categories and len(categories) == 1 else None
            return await self._engine.search(
                query=query,
                limit=limit,
                category=category,
                project=project,
                min_score=min_score,
                include_archived=include_archived,
            )
        else:
            payload: dict[str, Any] = {
                "query": query,
                "limit": limit,
                "min_score": min_score,
            }
            if categories:
                payload["categories"] = [c.value for c in categories]
            if project:
                payload["project"] = project

            response = await self._request("POST", "/memories/search", json=payload)
            results = []
            for item in response.get("results", []):
                memory = Memory.from_dict(item["memory"])
                results.append(SearchResult(
                    memory=memory,
                    score=item.get("score", 0.0),
                    semantic_score=item.get("semantic_score", 0.0),
                    recency_score=item.get("recency_score", 0.0),
                    frequency_score=item.get("frequency_score", 0.0),
                ))
            return results

    async def list(
        self,
        project: str | None = None,
        category: str | MemoryCategory | None = None,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[Memory]:
        """List memories with filtering.

        Args:
            project: Filter by project.
            category: Filter by category.
            limit: Maximum results to return.
            include_archived: Whether to include archived memories.

        Returns:
            List of memories matching filters.
        """
        self._ensure_initialized()

        # Normalize category
        if isinstance(category, str):
            category = MemoryCategory(category)

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            return await self._engine.list(
                project=project,
                category=category,
                limit=limit,
                include_archived=include_archived,
            )
        else:
            params: dict[str, Any] = {
                "limit": limit,
                "include_archived": include_archived,
            }
            if project:
                params["project"] = project
            if category:
                params["category"] = category.value

            response = await self._request("GET", "/memories", params=params)
            return [Memory.from_dict(m) for m in response.get("memories", [])]

    # =========================================================================
    # Outcome Recording
    # =========================================================================

    async def record_outcome(
        self,
        memory_ids: list[str] | str,
        outcome: str | Outcome,
    ) -> list[Memory]:
        """Record outcome feedback for memories.

        Updates outcome scores for specified memories based on feedback.

        Args:
            memory_ids: Memory ID(s) to update.
            outcome: The outcome (WORKED, FAILED, PARTIAL).

        Returns:
            List of updated memories (local mode only, empty list for remote).
        """
        self._ensure_initialized()

        # Normalize inputs
        if isinstance(memory_ids, str):
            memory_ids = [memory_ids]
        if isinstance(outcome, str):
            outcome = Outcome(outcome)

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            return await self._engine.record_outcome(memory_ids, outcome)
        else:
            await self._request(
                "POST",
                "/memories/outcome",
                json={
                    "memory_ids": memory_ids,
                    "outcome": outcome.value,
                },
            )
            # Remote mode doesn't return updated memories
            return []

    # =========================================================================
    # Context
    # =========================================================================

    async def get_context(
        self,
        project: str | None = None,
        query: str | None = None,
        limit: int = 10,
        format: str = "markdown",
    ) -> ContextResponse:
        """Get formatted context for injection into prompts.

        Args:
            project: Project filter.
            query: Optional query to find relevant memories.
            limit: Maximum memories to include.
            format: Output format (markdown, brief, detailed).

        Returns:
            ContextResponse with formatted context.
        """
        self._ensure_initialized()

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            return await self._engine.get_context(
                query=query,
                project=project,
                max_memories=limit,
            )
        else:
            params: dict[str, Any] = {
                "limit": limit,
                "format": format,
            }
            if project:
                params["project"] = project

            response = await self._request("GET", "/context", params=params)

            memories = [Memory.from_dict(m) for m in response.get("memories", [])]
            return ContextResponse(
                memories=memories,
                project=response.get("project"),
                total_count=response.get("total_count", 0),
                included_count=response.get("included_count", len(memories)),
                formatted=response.get("formatted", ""),
                categories={},
            )

    # =========================================================================
    # Statistics
    # =========================================================================

    async def stats(self, project: str | None = None) -> StatsDict | Any:
        """Get memory statistics.

        Args:
            project: Optional project filter.

        Returns:
            Statistics dictionary (StatsDict for remote, EngineStats for local).
        """
        self._ensure_initialized()

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            return await self._engine.stats(project=project)
        else:
            params: dict[str, Any] = {}
            if project:
                params["project"] = project

            response = await self._request("GET", "/stats", params=params)
            return StatsDict.from_dict(response)

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health(self) -> dict[str, Any]:
        """Check client health.

        Returns:
            Health status dictionary.
        """
        self._ensure_initialized()

        if self.config.mode == ClientMode.LOCAL:
            assert self._engine is not None
            return await self._engine.health_check()
        else:
            response = await self._request("GET", "/health")
            return response


# =============================================================================
# Synchronous Wrapper
# =============================================================================


class SyncMemoryClient:
    """Synchronous wrapper for MemoryClient.

    Provides a sync interface by running async operations in an event loop.

    Example:
        ```python
        with SyncMemoryClient(mode="local") as client:
            memory = client.add(
                content="Use async/await for I/O",
                category="pattern",
            )
            results = client.search("async patterns")
        ```
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        *,
        mode: str | ClientMode = ClientMode.LOCAL,
        db_path: str | Path | None = None,
        embedding_provider: str = "local",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Sync Memory Client.

        Args:
            config: Full configuration object.
            mode: Operating mode: "local" or "remote".
            db_path: Database path for local mode.
            embedding_provider: Embedding provider for local mode.
            base_url: API base URL for remote mode.
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        self._async_client = MemoryClient(
            config=config,
            mode=mode,
            db_path=db_path,
            embedding_provider=embedding_provider,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owns_loop = False

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
                self._owns_loop = False
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                self._owns_loop = True
        return self._loop

    def _run(self, coro: Any) -> Any:
        """Run a coroutine synchronously."""
        loop = self._get_loop()
        if loop.is_running():
            # If loop is running, we need to use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        else:
            return loop.run_until_complete(coro)

    def initialize(self) -> None:
        """Initialize the client."""
        self._run(self._async_client.initialize())

    def close(self) -> None:
        """Close the client."""
        self._run(self._async_client.close())
        if self._owns_loop and self._loop:
            self._loop.close()
            self._loop = None

    def __enter__(self) -> SyncMemoryClient:
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    # =========================================================================
    # Sync Methods (wrap async methods)
    # =========================================================================

    def add(
        self,
        content: str,
        category: str | MemoryCategory,
        project: str | None = None,
        scope: str | MemoryScope = MemoryScope.PROJECT,
        source: str | MemorySource = MemorySource.EXPLICIT,
        confidence: float = 1.0,
        importance: float = 0.5,
        tags: list[str] | None = None,
        entities: list[str] | None = None,
        supersedes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a new memory (sync)."""
        return self._run(self._async_client.add(
            content=content,
            category=category,
            project=project,
            scope=scope,
            source=source,
            confidence=confidence,
            importance=importance,
            tags=tags,
            entities=entities,
            supersedes=supersedes,
            metadata=metadata,
        ))

    def get(self, memory_id: str) -> Memory:
        """Get a memory by ID (sync)."""
        return self._run(self._async_client.get(memory_id))

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | MemoryCategory | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Update an existing memory (sync)."""
        return self._run(self._async_client.update(
            memory_id=memory_id,
            content=content,
            category=category,
            confidence=confidence,
            importance=importance,
            tags=tags,
            entities=entities,
            metadata=metadata,
        ))

    def delete(self, memory_id: str, hard_delete: bool = False) -> None:
        """Delete a memory (sync)."""
        return self._run(self._async_client.delete(memory_id, hard_delete=hard_delete))

    def search(
        self,
        query: str,
        limit: int = 10,
        categories: list[str | MemoryCategory] | None = None,
        project: str | None = None,
        min_score: float = -1.0,
        include_archived: bool = False,
    ) -> list[SearchResult]:
        """Search for relevant memories (sync)."""
        return self._run(self._async_client.search(
            query=query,
            limit=limit,
            categories=categories,
            project=project,
            min_score=min_score,
            include_archived=include_archived,
        ))

    def list(
        self,
        project: str | None = None,
        category: str | MemoryCategory | None = None,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[Memory]:
        """List memories with filtering (sync)."""
        return self._run(self._async_client.list(
            project=project,
            category=category,
            limit=limit,
            include_archived=include_archived,
        ))

    def record_outcome(
        self,
        memory_ids: list[str] | str,
        outcome: str | Outcome,
    ) -> list[Memory]:
        """Record outcome feedback for memories (sync)."""
        return self._run(self._async_client.record_outcome(memory_ids, outcome))

    def get_context(
        self,
        project: str | None = None,
        query: str | None = None,
        limit: int = 10,
        format: str = "markdown",
    ) -> ContextResponse:
        """Get formatted context (sync)."""
        return self._run(self._async_client.get_context(
            project=project,
            query=query,
            limit=limit,
            format=format,
        ))

    def stats(self, project: str | None = None) -> StatsDict | Any:
        """Get memory statistics (sync)."""
        return self._run(self._async_client.stats(project=project))

    def health(self) -> dict[str, Any]:
        """Check client health (sync)."""
        return self._run(self._async_client.health())


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

# Global default client (lazily initialized)
_default_client: MemoryClient | None = None
_default_config: ClientConfig | None = None


def configure(
    mode: str | ClientMode = ClientMode.LOCAL,
    db_path: str | Path | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> None:
    """Configure the default client.

    Call this before using module-level convenience functions.

    Args:
        mode: Operating mode: "local" or "remote".
        db_path: Database path for local mode.
        base_url: API base URL for remote mode.
        api_key: API key for authentication.
        **kwargs: Additional ClientConfig parameters.
    """
    global _default_config, _default_client

    _default_config = ClientConfig(
        mode=ClientMode(mode) if isinstance(mode, str) else mode,
        db_path=db_path or "~/.memory-layer/memories.db",
        base_url=base_url or "http://127.0.0.1:8080",
        api_key=api_key,
        **kwargs,
    )
    _default_client = None  # Reset client to use new config


async def _get_client() -> MemoryClient:
    """Get the default client, initializing if needed."""
    global _default_client

    if _default_client is None:
        _default_client = MemoryClient(config=_default_config or ClientConfig())
        await _default_client.initialize()

    return _default_client


async def add(
    content: str,
    category: str | MemoryCategory,
    project: str | None = None,
    **kwargs: Any,
) -> Memory:
    """Add a new memory using the default client.

    Args:
        content: The memory content text.
        category: Classification category.
        project: Project scope.
        **kwargs: Additional memory parameters.

    Returns:
        The created Memory.

    Example:
        ```python
        from memory_layer.sdk import add, configure

        configure(mode="local")
        memory = await add("Use async/await for I/O", category="pattern")
        ```
    """
    client = await _get_client()
    return await client.add(content=content, category=category, project=project, **kwargs)


async def search(
    query: str,
    limit: int = 10,
    project: str | None = None,
    **kwargs: Any,
) -> list[SearchResult]:
    """Search for relevant memories using the default client.

    Args:
        query: Search query text.
        limit: Maximum number of results.
        project: Filter by project.
        **kwargs: Additional search parameters.

    Returns:
        List of SearchResults.

    Example:
        ```python
        from memory_layer.sdk import search, configure

        configure(mode="local")
        results = await search("async patterns")
        ```
    """
    client = await _get_client()
    return await client.search(query=query, limit=limit, project=project, **kwargs)


async def get_context(
    project: str | None = None,
    limit: int = 10,
    **kwargs: Any,
) -> ContextResponse:
    """Get formatted context using the default client.

    Args:
        project: Project filter.
        limit: Maximum memories to include.
        **kwargs: Additional context parameters.

    Returns:
        ContextResponse with formatted context.
    """
    client = await _get_client()
    return await client.get_context(project=project, limit=limit, **kwargs)


async def record_outcome(
    memory_ids: list[str] | str,
    outcome: str | Outcome,
) -> list[Memory]:
    """Record outcome feedback using the default client.

    Args:
        memory_ids: Memory ID(s) to update.
        outcome: The outcome (worked, failed, partial).

    Returns:
        List of updated memories.
    """
    client = await _get_client()
    return await client.record_outcome(memory_ids=memory_ids, outcome=outcome)


async def close_default_client() -> None:
    """Close the default client if initialized."""
    global _default_client

    if _default_client is not None:
        await _default_client.close()
        _default_client = None

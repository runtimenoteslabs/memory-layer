"""Tests for the REST API server."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from memory_layer import __version__
from memory_layer.core.models import (
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
)
from memory_layer.core.storage import MemoryNotFoundError
from memory_layer.server.api import (
    APIConfig,
    AppState,
    ContextResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
    OutcomeRequest,
    OutcomeResponse,
    RateLimiter,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    StatsResponse,
    create_app,
    _app_state,
)


# =============================================================================
# Fixtures
# =============================================================================


def create_test_memory(**kwargs) -> Memory:
    """Create a test memory with defaults."""
    defaults = {
        "id": "test-memory-id",
        "content": "Test memory content",
        "category": MemoryCategory.GENERAL,
        "outcome_score": 0.0,
        "confidence": 1.0,
        "importance": 0.5,
        "use_count": 0,
        "project": "test-project",
        "scope": MemoryScope.PROJECT,
        "source": MemorySource.EXPLICIT,
        "tags": ["test"],
        "entities": [],
        "supersedes": None,
        "archived": False,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    defaults.update(kwargs)
    return Memory(**defaults)


@pytest.fixture
def mock_engine():
    """Create a mock MemoryEngine."""
    engine = MagicMock()
    # Set up async methods
    engine.add = AsyncMock()
    engine.get = AsyncMock()
    engine.update = AsyncMock()
    engine.delete = AsyncMock()
    engine.list = AsyncMock()
    engine.search = AsyncMock()
    engine.record_outcome = AsyncMock()
    engine.stats = AsyncMock()
    engine.get_context = AsyncMock()
    return engine


@pytest.fixture
def test_client(mock_engine):
    """Create a test client with mock engine."""
    # Reset global state
    _app_state.engine = mock_engine
    _app_state.config = APIConfig(api_key=None)  # No auth
    _app_state.rate_limiter = RateLimiter(max_requests=1000, window_seconds=60)

    app = create_app(config=_app_state.config, engine=mock_engine)
    return TestClient(app)


@pytest.fixture
def auth_test_client(mock_engine):
    """Create a test client with API key authentication."""
    _app_state.engine = mock_engine
    _app_state.config = APIConfig(api_key="test-api-key")
    _app_state.rate_limiter = RateLimiter(max_requests=1000, window_seconds=60)

    app = create_app(config=_app_state.config, engine=mock_engine)
    return TestClient(app)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = APIConfig()
        assert config.rate_limit == 100
        assert config.rate_window == 60.0
        assert config.max_request_size == 1_000_000
        assert config.cors_origins == ["*"]

    def test_custom_config(self):
        """Test custom configuration."""
        config = APIConfig(
            api_key="my-key",
            rate_limit=50,
            rate_window=30.0,
            max_request_size=500_000,
            cors_origins=["https://example.com"],
        )
        assert config.api_key == "my-key"
        assert config.rate_limit == 50
        assert config.rate_window == 30.0
        assert config.max_request_size == 500_000
        assert config.cors_origins == ["https://example.com"]

    def test_api_key_from_env(self):
        """Test API key from environment."""
        with patch.dict(os.environ, {"MEMORY_LAYER_API_KEY": "env-key"}):
            # Need to clear the explicit None to allow env fallback
            config = APIConfig()
            assert config.api_key == "env-key"


# =============================================================================
# Request Model Tests
# =============================================================================


class TestMemoryCreateRequest:
    """Tests for MemoryCreateRequest model."""

    def test_valid_request(self):
        """Test valid request."""
        request = MemoryCreateRequest(content="Test content")
        assert request.content == "Test content"
        assert request.category == MemoryCategory.GENERAL
        assert request.importance == 0.5

    def test_all_fields(self):
        """Test all fields."""
        request = MemoryCreateRequest(
            content="Test content",
            category=MemoryCategory.DECISION,
            project="my-project",
            tags=["tag1", "tag2"],
            importance=0.8,
            entities=["entity1"],
        )
        assert request.content == "Test content"
        assert request.category == MemoryCategory.DECISION
        assert request.project == "my-project"
        assert request.tags == ["tag1", "tag2"]
        assert request.importance == 0.8
        assert request.entities == ["entity1"]

    def test_whitespace_stripped(self):
        """Test whitespace is stripped from content."""
        request = MemoryCreateRequest(content="  Test content  ")
        assert request.content == "Test content"

    def test_empty_content_rejected(self):
        """Test empty content is rejected."""
        # Pydantic's min_length validation triggers first on whitespace-only strings
        # because str_strip_whitespace is applied before validation
        with pytest.raises(ValueError):
            MemoryCreateRequest(content="   ")

    def test_importance_validation(self):
        """Test importance bounds."""
        with pytest.raises(ValueError):
            MemoryCreateRequest(content="Test", importance=1.5)
        with pytest.raises(ValueError):
            MemoryCreateRequest(content="Test", importance=-0.1)


class TestMemoryUpdateRequest:
    """Tests for MemoryUpdateRequest model."""

    def test_partial_update(self):
        """Test partial update."""
        request = MemoryUpdateRequest(content="Updated content")
        assert request.content == "Updated content"
        assert request.category is None
        assert request.tags is None

    def test_all_fields(self):
        """Test all fields."""
        request = MemoryUpdateRequest(
            content="Updated",
            category=MemoryCategory.PATTERN,
            tags=["new-tag"],
            importance=0.9,
        )
        assert request.content == "Updated"
        assert request.category == MemoryCategory.PATTERN
        assert request.tags == ["new-tag"]
        assert request.importance == 0.9


class TestSearchRequest:
    """Tests for SearchRequest model."""

    def test_valid_search(self):
        """Test valid search request."""
        request = SearchRequest(query="test query")
        assert request.query == "test query"
        assert request.limit == 10
        assert request.min_score == -1.0

    def test_all_fields(self):
        """Test all fields."""
        request = SearchRequest(
            query="authentication",
            limit=20,
            categories=[MemoryCategory.PATTERN, MemoryCategory.CONVENTION],
            project="my-project",
            min_score=0.5,
        )
        assert request.query == "authentication"
        assert request.limit == 20
        assert len(request.categories) == 2
        assert request.project == "my-project"
        assert request.min_score == 0.5

    def test_limit_bounds(self):
        """Test limit bounds."""
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=0)
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=101)


class TestOutcomeRequest:
    """Tests for OutcomeRequest model."""

    def test_valid_outcome(self):
        """Test valid outcome request."""
        request = OutcomeRequest(
            memory_ids=["id1", "id2"],
            outcome=Outcome.WORKED,
        )
        assert request.memory_ids == ["id1", "id2"]
        assert request.outcome == Outcome.WORKED

    def test_empty_ids_rejected(self):
        """Test empty memory IDs rejected."""
        with pytest.raises(ValueError):
            OutcomeRequest(memory_ids=[], outcome=Outcome.WORKED)


class TestIngestRequest:
    """Tests for IngestRequest model."""

    def test_valid_ingest(self):
        """Test valid ingest request."""
        request = IngestRequest(transcript="User: Hello\nAssistant: Hi!")
        assert request.transcript == "User: Hello\nAssistant: Hi!"
        assert request.project is None

    def test_with_project(self):
        """Test ingest with project."""
        request = IngestRequest(
            transcript="Conversation...",
            project="my-project",
            session_id="session-123",
        )
        assert request.project == "my-project"
        assert request.session_id == "session-123"


# =============================================================================
# Response Model Tests
# =============================================================================


class TestMemoryResponse:
    """Tests for MemoryResponse model."""

    def test_from_memory(self):
        """Test creating from Memory object."""
        memory = create_test_memory()
        response = MemoryResponse.from_memory(memory)
        assert response.id == "test-memory-id"
        assert response.content == "Test memory content"
        assert response.category == MemoryCategory.GENERAL
        assert response.project == "test-project"


class TestSearchResultResponse:
    """Tests for SearchResultResponse model."""

    def test_search_result(self):
        """Test search result response."""
        memory = create_test_memory()
        result = SearchResultResponse(
            memory=MemoryResponse.from_memory(memory),
            score=0.85,
            semantic_score=0.9,
            recency_score=0.8,
            frequency_score=0.7,
        )
        assert result.score == 0.85
        assert result.semantic_score == 0.9


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self):
        """Test health response."""
        response = HealthResponse(
            status="healthy",
            version="2.0.0",
            timestamp=datetime.now(timezone.utc),
        )
        assert response.status == "healthy"
        assert response.version == "2.0.0"


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_error_response(self):
        """Test error response."""
        response = ErrorResponse(
            error="Not found",
            detail="Memory not found",
            status_code=404,
        )
        assert response.error == "Not found"
        assert response.status_code == 404


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allows_under_limit(self):
        """Test requests allowed under limit."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client1")

    def test_blocks_over_limit(self):
        """Test requests blocked over limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")

    def test_separate_clients(self):
        """Test separate tracking per client."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        assert limiter.is_allowed("client2")

    def test_window_expiry(self):
        """Test requests allowed after window expires."""
        limiter = RateLimiter(max_requests=1, window_seconds=0.1)
        assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        time.sleep(0.15)
        assert limiter.is_allowed("client1")

    def test_get_remaining(self):
        """Test get remaining requests."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.get_remaining("client1") == 5
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.get_remaining("client1") == 3


# =============================================================================
# Health and Stats Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__
        assert "timestamp" in data


class TestStatsEndpoint:
    """Tests for stats endpoint."""

    def test_get_stats(self, test_client, mock_engine):
        """Test get stats endpoint."""
        from memory_layer.core.storage import StorageStats
        from memory_layer.core.engine import EngineStats

        storage_stats = StorageStats(
            total_memories=100,
            active_memories=90,
            archived_memories=10,
            by_category={"general": 50, "pattern": 30, "convention": 20},
            by_scope={"project": 60, "global": 40},
            by_source={"explicit": 80, "extracted": 20},
            avg_outcome_score=0.15,
            total_uses=500,
        )
        mock_engine.stats.return_value = EngineStats(
            storage_stats=storage_stats,
            indexed_memories=100,
            indexed_with_embeddings=100,
            last_search_result_count=0,
        )

        response = test_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_memories"] == 100
        assert data["active_memories"] == 90
        assert data["avg_outcome_score"] == 0.15

    def test_get_stats_with_project(self, test_client, mock_engine):
        """Test get stats with project filter."""
        from memory_layer.core.storage import StorageStats
        from memory_layer.core.engine import EngineStats

        storage_stats = StorageStats(
            total_memories=20,
            active_memories=18,
            archived_memories=2,
            by_category={},
            by_scope={},
            by_source={},
            avg_outcome_score=0.0,
            total_uses=0,
        )
        mock_engine.stats.return_value = EngineStats(
            storage_stats=storage_stats,
            indexed_memories=20,
            indexed_with_embeddings=20,
            last_search_result_count=0,
        )

        response = test_client.get("/stats?project=my-project")
        assert response.status_code == 200
        mock_engine.stats.assert_called_once_with(project="my-project")


# =============================================================================
# Memory CRUD Tests
# =============================================================================


class TestCreateMemory:
    """Tests for create memory endpoint."""

    def test_create_memory(self, test_client, mock_engine):
        """Test creating a memory."""
        memory = create_test_memory()
        mock_engine.add.return_value = memory

        response = test_client.post(
            "/memories",
            json={"content": "Test memory content"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "Test memory content"
        mock_engine.add.assert_called_once()

    def test_create_memory_all_fields(self, test_client, mock_engine):
        """Test creating memory with all fields."""
        memory = create_test_memory(
            category=MemoryCategory.DECISION,
            tags=["tag1", "tag2"],
            entities=["entity1"],
        )
        mock_engine.add.return_value = memory

        response = test_client.post(
            "/memories",
            json={
                "content": "Test memory",
                "category": "decision",
                "project": "my-project",
                "tags": ["tag1", "tag2"],
                "importance": 0.8,
                "entities": ["entity1"],
            },
        )

        assert response.status_code == 201
        call_kwargs = mock_engine.add.call_args.kwargs
        assert call_kwargs["category"] == MemoryCategory.DECISION
        assert call_kwargs["importance"] == 0.8

    def test_create_memory_invalid_content(self, test_client):
        """Test creating memory with empty content."""
        response = test_client.post(
            "/memories",
            json={"content": "   "},
        )
        assert response.status_code == 422


class TestGetMemory:
    """Tests for get memory endpoint."""

    def test_get_memory(self, test_client, mock_engine):
        """Test getting a memory by ID."""
        memory = create_test_memory()
        mock_engine.get.return_value = memory

        response = test_client.get("/memories/test-memory-id")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-memory-id"
        assert data["content"] == "Test memory content"

    def test_get_memory_not_found(self, test_client, mock_engine):
        """Test getting non-existent memory."""
        mock_engine.get.side_effect = MemoryNotFoundError("nonexistent-id")

        response = test_client.get("/memories/nonexistent-id")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestUpdateMemory:
    """Tests for update memory endpoint."""

    def test_update_memory(self, test_client, mock_engine):
        """Test updating a memory."""
        memory = create_test_memory()
        updated_memory = create_test_memory(content="Updated content")
        mock_engine.get.return_value = memory
        mock_engine.update.return_value = updated_memory

        response = test_client.patch(
            "/memories/test-memory-id",
            json={"content": "Updated content"},
        )

        assert response.status_code == 200
        mock_engine.update.assert_called_once()

    def test_update_memory_partial(self, test_client, mock_engine):
        """Test partial memory update."""
        memory = create_test_memory()
        mock_engine.get.return_value = memory
        updated_memory = create_test_memory(importance=0.9)
        mock_engine.update.return_value = updated_memory

        response = test_client.patch(
            "/memories/test-memory-id",
            json={"importance": 0.9},
        )

        assert response.status_code == 200
        # Check that update was called with the right kwargs
        call_kwargs = mock_engine.update.call_args.kwargs
        assert call_kwargs["importance"] == 0.9

    def test_update_memory_not_found(self, test_client, mock_engine):
        """Test updating non-existent memory."""
        mock_engine.update.side_effect = MemoryNotFoundError("nonexistent-id")

        response = test_client.patch(
            "/memories/nonexistent-id",
            json={"content": "Updated"},
        )

        assert response.status_code == 404


class TestDeleteMemory:
    """Tests for delete memory endpoint."""

    def test_delete_memory(self, test_client, mock_engine):
        """Test deleting a memory."""
        mock_engine.delete.return_value = True

        response = test_client.delete("/memories/test-memory-id")

        assert response.status_code == 204
        mock_engine.delete.assert_called_once_with("test-memory-id")

    def test_delete_memory_not_found(self, test_client, mock_engine):
        """Test deleting non-existent memory."""
        mock_engine.delete.side_effect = MemoryNotFoundError("nonexistent-id")

        response = test_client.delete("/memories/nonexistent-id")

        assert response.status_code == 404


class TestListMemories:
    """Tests for list memories endpoint."""

    def test_list_memories(self, test_client, mock_engine):
        """Test listing memories."""
        memories = [
            create_test_memory(id="mem1"),
            create_test_memory(id="mem2"),
        ]
        mock_engine.list.return_value = memories

        response = test_client.get("/memories")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["memories"]) == 2

    def test_list_memories_with_filters(self, test_client, mock_engine):
        """Test listing memories with filters."""
        mock_engine.list.return_value = []

        response = test_client.get(
            "/memories?category=pattern&project=my-project&limit=5&include_archived=true"
        )

        assert response.status_code == 200
        mock_engine.list.assert_called_once_with(
            category=MemoryCategory.PATTERN,
            project="my-project",
            limit=5,
            include_archived=True,
        )


# =============================================================================
# Search Endpoint Tests
# =============================================================================


class TestSearchMemories:
    """Tests for search endpoint."""

    def test_search_memories(self, test_client, mock_engine):
        """Test searching memories."""
        from memory_layer.core.models import SearchResult

        memory = create_test_memory()
        results = [
            SearchResult(
                memory=memory,
                score=0.85,
                semantic_score=0.9,
                recency_score=0.8,
                frequency_score=0.7,
            )
        ]
        mock_engine.search.return_value = results

        response = test_client.post(
            "/memories/search",
            json={"query": "test query"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["score"] == 0.85

    def test_search_with_filters(self, test_client, mock_engine):
        """Test search with filters."""
        mock_engine.search.return_value = []

        response = test_client.post(
            "/memories/search",
            json={
                "query": "authentication",
                "limit": 20,
                "categories": ["pattern", "convention"],
                "project": "my-project",
                "min_score": 0.5,
            },
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.search.call_args.kwargs
        assert call_kwargs["query"] == "authentication"
        assert call_kwargs["limit"] == 20
        # API now passes first category to engine (engine expects singular category)
        assert call_kwargs["category"] == MemoryCategory.PATTERN


# =============================================================================
# Outcome Endpoint Tests
# =============================================================================


class TestRecordOutcome:
    """Tests for outcome endpoint."""

    def test_record_worked_outcome(self, test_client, mock_engine):
        """Test recording worked outcome."""
        mock_engine.record_outcome.return_value = True

        response = test_client.post(
            "/memories/outcome",
            json={
                "memory_ids": ["mem1", "mem2"],
                "outcome": "worked",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["outcome"] == "worked"
        assert data["adjustment"] == "+0.2"

    def test_record_failed_outcome(self, test_client, mock_engine):
        """Test recording failed outcome."""
        mock_engine.record_outcome.return_value = True

        response = test_client.post(
            "/memories/outcome",
            json={
                "memory_ids": ["mem1"],
                "outcome": "failed",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["adjustment"] == "-0.3"

    def test_record_partial_outcome(self, test_client, mock_engine):
        """Test recording partial outcome."""
        mock_engine.record_outcome.return_value = True

        response = test_client.post(
            "/memories/outcome",
            json={
                "memory_ids": ["mem1"],
                "outcome": "partial",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["adjustment"] == "+0.05"

    def test_record_outcome_not_found(self, test_client, mock_engine):
        """Test recording outcome for non-existent memories."""
        mock_engine.record_outcome.return_value = False

        response = test_client.post(
            "/memories/outcome",
            json={
                "memory_ids": ["nonexistent"],
                "outcome": "worked",
            },
        )

        assert response.status_code == 404


# =============================================================================
# Context Endpoint Tests
# =============================================================================


class TestGetContext:
    """Tests for context endpoint."""

    def test_get_context(self, test_client, mock_engine):
        """Test getting context."""
        from memory_layer.core.models import ContextResponse as EngineContextResponse

        memories = [create_test_memory()]
        mock_engine.get_context.return_value = EngineContextResponse(
            memories=memories,
            project=None,
            total_count=1,
            included_count=1,
        )

        response = test_client.get("/context")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert "formatted" in data
        assert len(data["memories"]) == 1

    def test_get_context_with_options(self, test_client, mock_engine):
        """Test getting context with options."""
        from memory_layer.core.models import ContextResponse as EngineContextResponse

        mock_engine.get_context.return_value = EngineContextResponse(
            memories=[],
            project="my-project",
            total_count=0,
            included_count=0,
        )

        response = test_client.get("/context?project=my-project&limit=5&format=brief")

        assert response.status_code == 200
        # API passes limit as max_memories to engine
        mock_engine.get_context.assert_called_once_with(
            project="my-project",
            max_memories=5,
        )


# =============================================================================
# Session Ingest Tests
# =============================================================================


class TestIngestTranscript:
    """Tests for ingest endpoint."""

    def test_ingest_transcript(self, test_client):
        """Test ingesting transcript."""
        response = test_client.post(
            "/sessions/ingest",
            json={
                "transcript": "User: Hello\nAssistant: Hi there!",
                "project": "my-project",
                "session_id": "session-123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["project"] == "my-project"
        assert data["session_id"] == "session-123"


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for API key authentication."""

    def test_no_auth_required(self, test_client):
        """Test endpoints work without auth when not configured."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_auth_required_missing_key(self, auth_test_client, mock_engine):
        """Test auth required but no key provided."""
        mock_engine.stats.return_value = {
            "total_memories": 0,
            "active_memories": 0,
            "archived_memories": 0,
            "by_category": {},
            "by_scope": {},
            "by_source": {},
            "avg_outcome_score": 0.0,
            "total_uses": 0,
        }

        response = auth_test_client.get("/stats")
        assert response.status_code == 401
        assert "API key required" in response.json()["detail"]

    def test_auth_required_invalid_key(self, auth_test_client, mock_engine):
        """Test auth required with invalid key."""
        response = auth_test_client.get(
            "/stats",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_auth_required_valid_key(self, auth_test_client, mock_engine):
        """Test auth required with valid key."""
        from memory_layer.core.storage import StorageStats
        from memory_layer.core.engine import EngineStats

        storage_stats = StorageStats(
            total_memories=0,
            active_memories=0,
            archived_memories=0,
            by_category={},
            by_scope={},
            by_source={},
            avg_outcome_score=0.0,
            total_uses=0,
        )
        mock_engine.stats.return_value = EngineStats(
            storage_stats=storage_stats,
            indexed_memories=0,
            indexed_with_embeddings=0,
            last_search_result_count=0,
        )

        response = auth_test_client.get(
            "/stats",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200

    def test_health_no_auth(self, auth_test_client):
        """Test health endpoint doesn't require auth."""
        response = auth_test_client.get("/health")
        # Health check doesn't use verify_api_key dependency
        assert response.status_code == 200


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""

    def test_rate_limit_headers(self, test_client):
        """Test rate limit headers are included."""
        response = test_client.get("/health")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_rate_limit_exceeded(self, mock_engine):
        """Test rate limit exceeded."""
        # Create app with very low rate limit
        _app_state.engine = mock_engine
        _app_state.config = APIConfig(rate_limit=2, rate_window=60)
        _app_state.rate_limiter = RateLimiter(max_requests=2, window_seconds=60)

        app = create_app(config=_app_state.config, engine=mock_engine)
        client = TestClient(app)

        # Make requests up to and over limit
        client.get("/health")
        client.get("/health")
        response = client.get("/health")

        assert response.status_code == 429
        data = response.json()
        assert "Rate limit exceeded" in data["error"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_validation_error(self, test_client):
        """Test validation error response."""
        response = test_client.post(
            "/memories",
            json={"content": ""},  # Empty content
        )
        assert response.status_code == 422

    def test_not_found_error(self, test_client, mock_engine):
        """Test not found error response."""
        mock_engine.get.side_effect = MemoryNotFoundError("nonexistent")

        response = test_client.get("/memories/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_invalid_json(self, test_client):
        """Test invalid JSON error."""
        response = test_client.post(
            "/memories",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# =============================================================================
# App Factory Tests
# =============================================================================


class TestCreateApp:
    """Tests for create_app factory."""

    def test_create_app_default(self, mock_engine):
        """Test creating app with defaults."""
        _app_state.engine = mock_engine
        _app_state.config = None
        _app_state.rate_limiter = None

        app = create_app()
        assert app.title == "Memory Layer API"
        assert app.version == __version__

    def test_create_app_with_config(self, mock_engine):
        """Test creating app with config."""
        config = APIConfig(api_key="custom-key", rate_limit=50)
        app = create_app(config=config, engine=mock_engine)
        assert _app_state.config == config

    def test_create_app_cors_middleware(self, mock_engine):
        """Test CORS middleware is added."""
        config = APIConfig(cors_origins=["https://example.com"])
        app = create_app(config=config, engine=mock_engine)

        # Check CORS middleware is present
        client = TestClient(app)
        response = client.options(
            "/health",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in response.headers


# =============================================================================
# AppState Tests
# =============================================================================


class TestAppState:
    """Tests for AppState."""

    def test_get_engine_creates_default(self):
        """Test get_engine creates default engine."""
        state = AppState()
        with patch("memory_layer.server.api.MemoryEngine") as mock_cls:
            mock_cls.return_value = MagicMock()
            engine = state.get_engine()
            assert engine is not None
            mock_cls.assert_called_once()

    def test_get_engine_reuses_instance(self):
        """Test get_engine reuses existing instance."""
        state = AppState()
        mock_engine = MagicMock()
        state.engine = mock_engine

        engine = state.get_engine()
        assert engine is mock_engine


# =============================================================================
# Response Model Serialization Tests
# =============================================================================


class TestResponseSerialization:
    """Tests for response model serialization."""

    def test_memory_response_json(self):
        """Test MemoryResponse serializes to JSON."""
        memory = create_test_memory()
        response = MemoryResponse.from_memory(memory)
        data = response.model_dump()

        assert data["id"] == "test-memory-id"
        assert data["category"] == MemoryCategory.GENERAL
        assert isinstance(data["created_at"], datetime)

    def test_search_response_json(self):
        """Test SearchResponse serializes to JSON."""
        memory = create_test_memory()
        response = SearchResponse(
            count=1,
            results=[
                SearchResultResponse(
                    memory=MemoryResponse.from_memory(memory),
                    score=0.85,
                )
            ],
        )
        data = response.model_dump()

        assert data["count"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.85

    def test_stats_response_json(self):
        """Test StatsResponse serializes to JSON."""
        response = StatsResponse(
            total_memories=100,
            active_memories=90,
            archived_memories=10,
            by_category={"general": 50},
            by_scope={"project": 60},
            by_source={"explicit": 80},
            avg_outcome_score=0.15,
            total_uses=500,
        )
        data = response.model_dump()

        assert data["total_memories"] == 100
        assert data["by_category"] == {"general": 50}


# =============================================================================
# Integration-like Tests (using TestClient)
# =============================================================================


class TestEndpointIntegration:
    """Integration-like tests for endpoint workflows."""

    def test_create_then_get_memory(self, test_client, mock_engine):
        """Test creating then getting a memory."""
        memory = create_test_memory(id="new-memory-id")
        mock_engine.add.return_value = memory
        mock_engine.get.return_value = memory

        # Create
        create_response = test_client.post(
            "/memories",
            json={"content": "Test content"},
        )
        assert create_response.status_code == 201
        created_id = create_response.json()["id"]

        # Get
        get_response = test_client.get(f"/memories/{created_id}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == created_id

    def test_create_update_delete_workflow(self, test_client, mock_engine):
        """Test full CRUD workflow."""
        memory = create_test_memory()
        updated_memory = create_test_memory(content="Updated")

        mock_engine.add.return_value = memory
        mock_engine.get.return_value = memory
        mock_engine.update.return_value = updated_memory
        mock_engine.delete.return_value = True

        # Create
        test_client.post("/memories", json={"content": "Test"})

        # Update
        mock_engine.get.return_value = memory
        update_response = test_client.patch(
            f"/memories/{memory.id}",
            json={"content": "Updated"},
        )
        assert update_response.status_code == 200

        # Delete
        delete_response = test_client.delete(f"/memories/{memory.id}")
        assert delete_response.status_code == 204

    def test_search_and_record_outcome(self, test_client, mock_engine):
        """Test search and outcome workflow."""
        from memory_layer.core.models import SearchResult

        memory = create_test_memory()
        results = [
            SearchResult(
                memory=memory,
                score=0.9,
                semantic_score=0.9,
                recency_score=0.8,
                frequency_score=0.7,
            )
        ]
        mock_engine.search.return_value = results
        mock_engine.record_outcome.return_value = True

        # Search
        search_response = test_client.post(
            "/memories/search",
            json={"query": "test"},
        )
        assert search_response.status_code == 200
        memory_id = search_response.json()["results"][0]["memory"]["id"]

        # Record outcome
        outcome_response = test_client.post(
            "/memories/outcome",
            json={
                "memory_ids": [memory_id],
                "outcome": "worked",
            },
        )
        assert outcome_response.status_code == 200
        assert outcome_response.json()["success"] is True

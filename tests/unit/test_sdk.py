"""Tests for the Python SDK."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from memory_layer.core.models import (
    ContextResponse,
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
    SearchResult,
)
from memory_layer.sdk import (
    MemoryClient,
    SyncMemoryClient,
    ClientConfig,
    ClientMode,
    SDKError,
    ConnectionError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    StatsDict,
    configure,
    add,
    search,
    get_context,
    record_outcome,
    close_default_client,
)
from memory_layer.sdk.client import _default_client, _default_config


# =============================================================================
# Test Fixtures
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


def create_test_search_result(memory: Memory | None = None, **kwargs) -> SearchResult:
    """Create a test search result with defaults."""
    if memory is None:
        memory = create_test_memory()
    defaults = {
        "memory": memory,
        "score": 0.85,
        "semantic_score": 0.7,
        "recency_score": 0.9,
        "frequency_score": 0.5,
        "category_boost": 1.0,
    }
    defaults.update(kwargs)
    return SearchResult(**defaults)


@pytest.fixture
def mock_engine():
    """Create a mock MemoryEngine."""
    engine = MagicMock()
    engine.initialize = AsyncMock()
    engine.close = AsyncMock()
    engine.add = AsyncMock(return_value=create_test_memory())
    engine.get = AsyncMock(return_value=create_test_memory())
    engine.update = AsyncMock(return_value=create_test_memory())
    engine.delete = AsyncMock()
    engine.list = AsyncMock(return_value=[create_test_memory()])
    engine.search = AsyncMock(return_value=[create_test_search_result()])
    engine.record_outcome = AsyncMock(return_value=[create_test_memory()])
    engine.get_context = AsyncMock(return_value=ContextResponse(
        memories=[create_test_memory()],
        project="test-project",
        total_count=1,
        included_count=1,
        formatted="# Test Context",
        categories={"general": 1},
    ))
    engine.stats = AsyncMock(return_value=MagicMock(
        storage_stats=MagicMock(
            total_memories=10,
            active_memories=8,
            archived_memories=2,
        ),
        indexed_memories=10,
        indexed_with_embeddings=10,
        last_search_result_count=5,
    ))
    engine.health_check = AsyncMock(return_value={"status": "healthy"})
    return engine


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a path for a temporary database."""
    return tmp_path / "test_memory.db"


# =============================================================================
# ClientConfig Tests
# =============================================================================


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClientConfig()
        assert config.mode == ClientMode.LOCAL
        assert config.embedding_provider == "local"
        assert config.base_url == "http://127.0.0.1:8080"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.max_connections == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = ClientConfig(
            mode=ClientMode.REMOTE,
            base_url="http://custom:9090",
            api_key="test-key",
            timeout=60.0,
            max_retries=5,
        )
        assert config.mode == ClientMode.REMOTE
        assert config.base_url == "http://custom:9090"
        assert config.api_key == "test-key"
        assert config.timeout == 60.0
        assert config.max_retries == 5

    def test_mode_string_conversion(self):
        """Test mode string to enum conversion."""
        config = ClientConfig(mode="remote")
        assert config.mode == ClientMode.REMOTE

        config = ClientConfig(mode="local")
        assert config.mode == ClientMode.LOCAL

    def test_db_path_expansion(self):
        """Test database path expansion."""
        config = ClientConfig(db_path="~/test.db")
        assert str(config.db_path).startswith("/")
        assert "~" not in str(config.db_path)

    def test_base_url_trailing_slash_removed(self):
        """Test that trailing slash is removed from base URL."""
        config = ClientConfig(base_url="http://localhost:8080/")
        assert config.base_url == "http://localhost:8080"


class TestClientMode:
    """Tests for ClientMode enum."""

    def test_local_mode_value(self):
        """Test local mode value."""
        assert ClientMode.LOCAL.value == "local"

    def test_remote_mode_value(self):
        """Test remote mode value."""
        assert ClientMode.REMOTE.value == "remote"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert ClientMode("local") == ClientMode.LOCAL
        assert ClientMode("remote") == ClientMode.REMOTE


# =============================================================================
# StatsDict Tests
# =============================================================================


class TestStatsDict:
    """Tests for StatsDict."""

    def test_from_dict(self):
        """Test creating StatsDict from dictionary."""
        data = {
            "total_memories": 100,
            "active_memories": 90,
            "archived_memories": 10,
            "by_category": {"general": 50, "pattern": 30, "convention": 20},
            "by_scope": {"project": 80, "global": 20},
            "by_source": {"explicit": 70, "extracted": 30},
            "avg_outcome_score": 0.3,
            "total_uses": 500,
        }
        stats = StatsDict.from_dict(data)
        assert stats.total_memories == 100
        assert stats.active_memories == 90
        assert stats.archived_memories == 10
        assert stats.by_category == {"general": 50, "pattern": 30, "convention": 20}
        assert stats.avg_outcome_score == 0.3
        assert stats.total_uses == 500

    def test_from_dict_with_defaults(self):
        """Test creating StatsDict with missing fields."""
        data = {}
        stats = StatsDict.from_dict(data)
        assert stats.total_memories == 0
        assert stats.active_memories == 0
        assert stats.by_category == {}


# =============================================================================
# MemoryClient Initialization Tests
# =============================================================================


class TestMemoryClientInit:
    """Tests for MemoryClient initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default settings."""
        client = MemoryClient()
        assert client.config.mode == ClientMode.LOCAL
        assert client._initialized is False
        assert client._engine is None
        assert client._http_client is None

    def test_init_with_config(self):
        """Test initialization with config object."""
        config = ClientConfig(mode=ClientMode.REMOTE, base_url="http://test:8080")
        client = MemoryClient(config=config)
        assert client.config.mode == ClientMode.REMOTE
        assert client.config.base_url == "http://test:8080"

    def test_init_with_kwargs(self):
        """Test initialization with keyword arguments."""
        client = MemoryClient(
            mode="remote",
            base_url="http://test:9090",
            api_key="my-key",
            timeout=60.0,
        )
        assert client.config.mode == ClientMode.REMOTE
        assert client.config.base_url == "http://test:9090"
        assert client.config.api_key == "my-key"
        assert client.config.timeout == 60.0

    def test_config_overrides_kwargs(self):
        """Test that config object overrides kwargs."""
        config = ClientConfig(mode=ClientMode.REMOTE)
        client = MemoryClient(config=config, mode="local")  # Should be ignored
        assert client.config.mode == ClientMode.REMOTE


# =============================================================================
# MemoryClient Local Mode Tests
# =============================================================================


class TestMemoryClientLocalMode:
    """Tests for MemoryClient in local mode."""

    @pytest.mark.asyncio
    async def test_initialize_local(self, temp_db_path: Path):
        """Test local mode initialization."""
        with patch("memory_layer.core.engine.MemoryEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.initialize = AsyncMock()
            mock_engine.close = AsyncMock()
            MockEngine.return_value = mock_engine

            client = MemoryClient(mode="local", db_path=temp_db_path)
            await client.initialize()

            assert client._initialized is True
            MockEngine.assert_called_once()
            mock_engine.initialize.assert_called_once()

            await client.close()

    @pytest.mark.asyncio
    async def test_close_local(self, mock_engine):
        """Test closing local mode client."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            client = MemoryClient(mode="local")
            await client.initialize()
            await client.close()

            assert client._initialized is False
            mock_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_local(self, mock_engine):
        """Test local mode as async context manager."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                assert client._initialized is True

            assert client._initialized is False

    @pytest.mark.asyncio
    async def test_add_local(self, mock_engine):
        """Test adding memory in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                memory = await client.add(
                    content="Test content",
                    category="pattern",
                    project="test-project",
                )

                assert memory is not None
                mock_engine.add.assert_called_once()
                call_kwargs = mock_engine.add.call_args.kwargs
                assert call_kwargs["content"] == "Test content"
                assert call_kwargs["category"] == MemoryCategory.PATTERN
                assert call_kwargs["project"] == "test-project"

    @pytest.mark.asyncio
    async def test_get_local(self, mock_engine):
        """Test getting memory in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                memory = await client.get("test-memory-id")

                assert memory is not None
                mock_engine.get.assert_called_once_with("test-memory-id")

    @pytest.mark.asyncio
    async def test_update_local(self, mock_engine):
        """Test updating memory in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                memory = await client.update(
                    "test-memory-id",
                    content="Updated content",
                    category="convention",
                )

                assert memory is not None
                mock_engine.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_local(self, mock_engine):
        """Test deleting memory in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                await client.delete("test-memory-id")

                mock_engine.delete.assert_called_once_with(
                    "test-memory-id", hard_delete=False
                )

    @pytest.mark.asyncio
    async def test_search_local(self, mock_engine):
        """Test searching in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                results = await client.search("test query", limit=5)

                assert len(results) == 1
                mock_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_local(self, mock_engine):
        """Test listing memories in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                memories = await client.list(project="test-project", limit=10)

                assert len(memories) == 1
                mock_engine.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_outcome_local(self, mock_engine):
        """Test recording outcome in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                memories = await client.record_outcome(
                    "test-memory-id", Outcome.WORKED
                )

                assert len(memories) == 1
                mock_engine.record_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_local(self, mock_engine):
        """Test getting context in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                context = await client.get_context(project="test-project")

                assert context is not None
                assert len(context.memories) == 1
                mock_engine.get_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_stats_local(self, mock_engine):
        """Test getting stats in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                stats = await client.stats()

                assert stats is not None
                mock_engine.stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_local(self, mock_engine):
        """Test health check in local mode."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                health = await client.health()

                assert health["status"] == "healthy"
                mock_engine.health_check.assert_called_once()


# =============================================================================
# MemoryClient Remote Mode Tests
# =============================================================================


class TestMemoryClientRemoteMode:
    """Tests for MemoryClient in remote mode."""

    @pytest.mark.asyncio
    async def test_initialize_remote(self):
        """Test remote mode initialization."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            ))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            client = MemoryClient(mode="remote", base_url="http://test:8080")
            await client.initialize()

            assert client._initialized is True
            assert client._http_client is not None

            await client.close()

    @pytest.mark.asyncio
    async def test_initialize_remote_connection_error(self):
        """Test remote mode initialization with connection error."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            client = MemoryClient(mode="remote", base_url="http://test:8080")

            with pytest.raises(ConnectionError) as exc_info:
                await client.initialize()

            assert "Failed to connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_add_remote(self):
        """Test adding memory in remote mode."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            mock_response_data = create_test_memory().to_dict()
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            ))
            mock_client.request = AsyncMock(return_value=MagicMock(
                status_code=201,
                json=lambda: mock_response_data,
            ))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            async with MemoryClient(mode="remote") as client:
                memory = await client.add(
                    content="Test content",
                    category="pattern",
                )

                assert memory is not None
                mock_client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_remote(self):
        """Test searching in remote mode."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            mock_response_data = {
                "count": 1,
                "results": [{
                    "memory": create_test_memory().to_dict(),
                    "score": 0.85,
                    "semantic_score": 0.7,
                    "recency_score": 0.9,
                    "frequency_score": 0.5,
                }],
            }
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            ))
            mock_client.request = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: mock_response_data,
            ))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            async with MemoryClient(mode="remote") as client:
                results = await client.search("test query")

                assert len(results) == 1
                assert results[0].score == 0.85

    @pytest.mark.asyncio
    async def test_remote_404_error(self):
        """Test 404 error handling in remote mode."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            ))
            mock_client.request = AsyncMock(return_value=MagicMock(
                status_code=404,
                json=lambda: {"detail": "Memory not found"},
            ))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            async with MemoryClient(mode="remote") as client:
                with pytest.raises(NotFoundError) as exc_info:
                    await client.get("nonexistent-id")

                assert "Memory not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_remote_401_error(self):
        """Test 401 authentication error handling."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            ))
            mock_client.request = AsyncMock(return_value=MagicMock(
                status_code=401,
                json=lambda: {"detail": "Invalid API key"},
            ))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            async with MemoryClient(mode="remote") as client:
                with pytest.raises(AuthenticationError):
                    await client.get("some-id")

    @pytest.mark.asyncio
    async def test_remote_422_error(self):
        """Test 422 validation error handling."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            ))
            mock_client.request = AsyncMock(return_value=MagicMock(
                status_code=422,
                json=lambda: {"detail": "Validation error"},
            ))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            async with MemoryClient(mode="remote") as client:
                with pytest.raises(ValidationError):
                    await client.add(content="", category="general")

    @pytest.mark.asyncio
    async def test_remote_429_error(self):
        """Test 429 rate limit error handling."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            ))
            mock_client.request = AsyncMock(return_value=MagicMock(
                status_code=429,
                json=lambda: {"detail": "Rate limit exceeded"},
            ))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            async with MemoryClient(mode="remote") as client:
                with pytest.raises(RateLimitError):
                    await client.search("query")

    @pytest.mark.asyncio
    async def test_remote_retry_logic(self):
        """Test retry logic on transient failures."""
        with patch("memory_layer.sdk.client.httpx.AsyncClient") as MockClient:
            call_count = 0

            async def mock_request(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise httpx.ConnectError("Temporary failure")
                return MagicMock(
                    status_code=200,
                    json=lambda: create_test_memory().to_dict(),
                )

            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            ))
            mock_client.request = mock_request
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            # Use config to set retry_delay
            config = ClientConfig(mode=ClientMode.REMOTE, max_retries=3, retry_delay=0.01)
            client = MemoryClient(config=config)
            await client.initialize()

            memory = await client.get("test-id")
            assert memory is not None
            assert call_count == 3

            await client.close()


# =============================================================================
# MemoryClient Error Handling Tests
# =============================================================================


class TestMemoryClientErrors:
    """Tests for MemoryClient error handling."""

    @pytest.mark.asyncio
    async def test_not_initialized_error(self):
        """Test error when using client before initialization."""
        client = MemoryClient(mode="local")

        with pytest.raises(SDKError) as exc_info:
            await client.add(content="test", category="general")

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_category_string_conversion(self, mock_engine):
        """Test that category strings are converted to enums."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                await client.add(content="test", category="pattern")

                call_kwargs = mock_engine.add.call_args.kwargs
                assert call_kwargs["category"] == MemoryCategory.PATTERN

    @pytest.mark.asyncio
    async def test_outcome_string_conversion(self, mock_engine):
        """Test that outcome strings are converted to enums."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                await client.record_outcome("test-id", "worked")

                call_args = mock_engine.record_outcome.call_args
                assert call_args[0][1] == Outcome.WORKED


# =============================================================================
# SyncMemoryClient Tests
# =============================================================================


class TestSyncMemoryClient:
    """Tests for SyncMemoryClient."""

    def test_init(self):
        """Test sync client initialization."""
        client = SyncMemoryClient(mode="local")
        assert client._async_client is not None
        assert client._async_client.config.mode == ClientMode.LOCAL

    def test_context_manager(self, mock_engine):
        """Test sync client as context manager."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            with SyncMemoryClient(mode="local") as client:
                assert client._async_client._initialized is True

    def test_add_sync(self, mock_engine):
        """Test adding memory synchronously."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            with SyncMemoryClient(mode="local") as client:
                memory = client.add(content="Test content", category="pattern")

                assert memory is not None
                mock_engine.add.assert_called_once()

    def test_get_sync(self, mock_engine):
        """Test getting memory synchronously."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            with SyncMemoryClient(mode="local") as client:
                memory = client.get("test-memory-id")

                assert memory is not None
                mock_engine.get.assert_called_once()

    def test_search_sync(self, mock_engine):
        """Test searching synchronously."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            with SyncMemoryClient(mode="local") as client:
                results = client.search("test query")

                assert len(results) == 1
                mock_engine.search.assert_called_once()

    def test_list_sync(self, mock_engine):
        """Test listing memories synchronously."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            with SyncMemoryClient(mode="local") as client:
                memories = client.list(project="test-project")

                assert len(memories) == 1
                mock_engine.list.assert_called_once()

    def test_record_outcome_sync(self, mock_engine):
        """Test recording outcome synchronously."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            with SyncMemoryClient(mode="local") as client:
                memories = client.record_outcome("test-id", "worked")

                assert len(memories) == 1
                mock_engine.record_outcome.assert_called_once()

    def test_get_context_sync(self, mock_engine):
        """Test getting context synchronously."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            with SyncMemoryClient(mode="local") as client:
                context = client.get_context(project="test-project")

                assert context is not None
                mock_engine.get_context.assert_called_once()


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        """Clean up default client after each test."""
        yield
        await close_default_client()
        # Reset module state
        import memory_layer.sdk.client as sdk_client
        sdk_client._default_client = None
        sdk_client._default_config = None

    def test_configure(self):
        """Test configure function."""
        configure(mode="local", db_path="/tmp/test.db", api_key="test-key")

        import memory_layer.sdk.client as sdk_client
        assert sdk_client._default_config is not None
        assert sdk_client._default_config.mode == ClientMode.LOCAL
        assert sdk_client._default_config.api_key == "test-key"

    def test_configure_resets_client(self):
        """Test that configure resets the default client."""
        import memory_layer.sdk.client as sdk_client
        sdk_client._default_client = MagicMock()

        configure(mode="remote")

        assert sdk_client._default_client is None

    @pytest.mark.asyncio
    async def test_add_function(self, mock_engine):
        """Test module-level add function."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            configure(mode="local")
            memory = await add("Test content", category="pattern")

            assert memory is not None

    @pytest.mark.asyncio
    async def test_search_function(self, mock_engine):
        """Test module-level search function."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            configure(mode="local")
            results = await search("test query")

            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_context_function(self, mock_engine):
        """Test module-level get_context function."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            configure(mode="local")
            context = await get_context(project="test-project")

            assert context is not None

    @pytest.mark.asyncio
    async def test_record_outcome_function(self, mock_engine):
        """Test module-level record_outcome function."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            configure(mode="local")
            memories = await record_outcome("test-id", "worked")

            assert len(memories) == 1


# =============================================================================
# Integration Tests (Local Mode with Real Engine - Skip if deps unavailable)
# =============================================================================


class TestLocalModeIntegration:
    """Integration tests for local mode with real MemoryEngine."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Skip by default - enable for integration tests
        reason="Integration test - requires full dependencies"
    )
    async def test_full_workflow_local(self, temp_db_path: Path):
        """Test full workflow in local mode."""
        async with MemoryClient(
            mode="local",
            db_path=temp_db_path,
            embedding_provider="mock",
        ) as client:
            # Add a memory
            memory = await client.add(
                content="Use async/await for I/O operations",
                category="pattern",
                project="test-project",
            )
            assert memory.id is not None
            assert memory.content == "Use async/await for I/O operations"

            # Get the memory
            retrieved = await client.get(memory.id)
            assert retrieved.id == memory.id

            # Search for memories
            results = await client.search("async patterns", limit=5)
            assert len(results) >= 1

            # Record outcome
            updated = await client.record_outcome(memory.id, "worked")
            assert len(updated) == 1
            assert updated[0].outcome_score > 0

            # Get context
            context = await client.get_context(project="test-project")
            assert len(context.memories) >= 1

            # Delete the memory
            await client.delete(memory.id)


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_search_results(self, mock_engine):
        """Test handling of empty search results."""
        mock_engine.search = AsyncMock(return_value=[])

        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                results = await client.search("nonexistent query")

                assert results == []

    @pytest.mark.asyncio
    async def test_multiple_memory_ids_outcome(self, mock_engine):
        """Test recording outcome for multiple memories."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                await client.record_outcome(
                    ["id1", "id2", "id3"],
                    Outcome.WORKED,
                )

                call_args = mock_engine.record_outcome.call_args
                assert call_args[0][0] == ["id1", "id2", "id3"]

    @pytest.mark.asyncio
    async def test_single_memory_id_as_string(self, mock_engine):
        """Test recording outcome with single ID as string."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                await client.record_outcome("single-id", "partial")

                call_args = mock_engine.record_outcome.call_args
                assert call_args[0][0] == ["single-id"]
                assert call_args[0][1] == Outcome.PARTIAL

    @pytest.mark.asyncio
    async def test_categories_filter_in_search(self, mock_engine):
        """Test search with multiple categories filter."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                # Local mode only supports single category
                await client.search(
                    "test",
                    categories=["pattern"],  # Single category
                )

                call_kwargs = mock_engine.search.call_args.kwargs
                assert call_kwargs["category"] == MemoryCategory.PATTERN

    @pytest.mark.asyncio
    async def test_categories_multiple_filter_in_search(self, mock_engine):
        """Test search with multiple categories falls back to None."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                # Multiple categories - local mode passes None
                await client.search(
                    "test",
                    categories=["pattern", "convention"],
                )

                call_kwargs = mock_engine.search.call_args.kwargs
                assert call_kwargs["category"] is None

    @pytest.mark.asyncio
    async def test_all_optional_parameters(self, mock_engine):
        """Test add with all optional parameters."""
        with patch("memory_layer.core.engine.MemoryEngine", return_value=mock_engine):
            async with MemoryClient(mode="local") as client:
                await client.add(
                    content="Test content",
                    category="pattern",
                    project="test-project",
                    scope="global",
                    source="extracted",
                    confidence=0.8,
                    importance=0.9,
                    tags=["tag1", "tag2"],
                    entities=["file.py", "function_name"],
                    supersedes="old-memory-id",
                    metadata={"key": "value"},
                )

                call_kwargs = mock_engine.add.call_args.kwargs
                assert call_kwargs["content"] == "Test content"
                assert call_kwargs["scope"] == MemoryScope.GLOBAL
                assert call_kwargs["source"] == MemorySource.EXTRACTED
                assert call_kwargs["confidence"] == 0.8
                assert call_kwargs["tags"] == ["tag1", "tag2"]

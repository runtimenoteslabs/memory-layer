"""Embedding providers for Memory Layer.

Provides vector embeddings with:
- Abstract provider interface
- Local embedding (sentence-transformers)
- API embedding (OpenAI-compatible)
- Caching layer
- Batch processing
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from memory_layer.core.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

# Type alias for embeddings
Embedding = list[float]


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""

    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    dimensions: int | None = None  # None = use model default

    # Cache settings
    cache_enabled: bool = True
    cache_dir: Path | None = None
    cache_ttl_seconds: int = 86400 * 30  # 30 days

    # Batch settings
    batch_size: int = 32

    # API settings (for APIEmbeddingProvider)
    api_base_url: str | None = None
    api_key_env_var: str = "EMBEDDING_API_KEY"
    api_timeout: float = 30.0
    api_max_retries: int = 3


@dataclass
class EmbeddingResult:
    """Result from an embedding operation."""

    embedding: Embedding
    model: str
    dimensions: int
    cached: bool = False
    latency_ms: float = 0.0


@dataclass
class BatchEmbeddingResult:
    """Result from a batch embedding operation."""

    embeddings: list[Embedding]
    model: str
    dimensions: int
    cached_count: int = 0
    computed_count: int = 0
    total_latency_ms: float = 0.0


class EmbeddingError(Exception):
    """Base exception for embedding errors."""

    pass


class ModelNotFoundError(EmbeddingError):
    """Raised when embedding model is not found."""

    pass


class APIError(EmbeddingError):
    """Raised when API call fails."""

    pass


class EmbeddingCache:
    """File-based cache for embeddings."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_seconds: int = 86400 * 30,
    ) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cache/memory-layer/embeddings
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "memory-layer" / "embeddings"
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self._memory_cache: dict[str, tuple[Embedding, float]] = {}

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate a cache key for text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = key[:2]
        return self.cache_dir / subdir / f"{key}.json"

    def get(self, text: str, model: str) -> Embedding | None:
        """Get an embedding from cache.

        Args:
            text: The text that was embedded.
            model: The model used for embedding.

        Returns:
            The cached embedding, or None if not found or expired.
        """
        key = self._get_cache_key(text, model)

        # Check memory cache first
        if key in self._memory_cache:
            embedding, timestamp = self._memory_cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return embedding
            del self._memory_cache[key]

        # Check file cache
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            timestamp = data.get("timestamp", 0)
            if time.time() - timestamp >= self.ttl_seconds:
                cache_path.unlink(missing_ok=True)
                return None

            cached_embedding: Embedding = data["embedding"]
            # Store in memory cache for faster subsequent access
            self._memory_cache[key] = (cached_embedding, timestamp)
            return cached_embedding
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def set(self, text: str, model: str, embedding: Embedding) -> None:
        """Store an embedding in cache.

        Args:
            text: The text that was embedded.
            model: The model used for embedding.
            embedding: The embedding to cache.
        """
        key = self._get_cache_key(text, model)
        timestamp = time.time()

        # Store in memory cache
        self._memory_cache[key] = (embedding, timestamp)

        # Store in file cache
        cache_path = self._get_cache_path(key)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "embedding": embedding,
                        "model": model,
                        "timestamp": timestamp,
                    }
                )
            )
        except OSError as e:
            logger.warning(f"Failed to write embedding cache: {e}")

    def get_many(
        self, texts: list[str], model: str
    ) -> tuple[dict[int, Embedding], list[int]]:
        """Get multiple embeddings from cache.

        Args:
            texts: List of texts to look up.
            model: The model used for embedding.

        Returns:
            Tuple of (cached embeddings by index, list of uncached indices).
        """
        cached: dict[int, Embedding] = {}
        uncached: list[int] = []

        for i, text in enumerate(texts):
            embedding = self.get(text, model)
            if embedding is not None:
                cached[i] = embedding
            else:
                uncached.append(i)

        return cached, uncached

    def set_many(
        self, texts: list[str], model: str, embeddings: list[Embedding]
    ) -> None:
        """Store multiple embeddings in cache.

        Args:
            texts: List of texts that were embedded.
            model: The model used for embedding.
            embeddings: The embeddings to cache.
        """
        for text, embedding in zip(texts, embeddings, strict=True):
            self.set(text, model, embedding)

    def clear(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of entries cleared.
        """
        self._memory_cache.clear()
        count = 0
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.rglob("*.json"):
                cache_file.unlink(missing_ok=True)
                count += 1
        return count

    def clear_expired(self) -> int:
        """Clear expired cache entries.

        Returns:
            Number of entries cleared.
        """
        now = time.time()
        count = 0

        # Clear expired memory cache entries
        expired_keys = [
            key
            for key, (_, timestamp) in self._memory_cache.items()
            if now - timestamp >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            count += 1

        # Clear expired file cache entries
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.rglob("*.json"):
                try:
                    data = json.loads(cache_file.read_text())
                    if now - data.get("timestamp", 0) >= self.ttl_seconds:
                        cache_file.unlink(missing_ok=True)
                        count += 1
                except (json.JSONDecodeError, OSError):
                    cache_file.unlink(missing_ok=True)
                    count += 1

        return count


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the provider.

        Args:
            config: Configuration for the provider.
        """
        self.config = config or EmbeddingConfig()
        self._cache: EmbeddingCache | None = None

        if self.config.cache_enabled:
            self._cache = EmbeddingCache(
                cache_dir=self.config.cache_dir,
                ttl_seconds=self.config.cache_ttl_seconds,
            )

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        ...

    @abstractmethod
    async def _embed_texts(self, texts: list[str]) -> list[Embedding]:
        """Embed multiple texts (internal implementation).

        Args:
            texts: Texts to embed.

        Returns:
            List of embeddings.
        """
        ...

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding result.
        """
        start_time = time.time()

        # Check cache
        if self._cache:
            cached = self._cache.get(text, self.model_name)
            if cached is not None:
                return EmbeddingResult(
                    embedding=cached,
                    model=self.model_name,
                    dimensions=len(cached),
                    cached=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Compute embedding
        embeddings = await self._embed_texts([text])
        embedding = embeddings[0]

        # Apply dimension reduction if configured
        if self.config.dimensions and len(embedding) > self.config.dimensions:
            embedding = embedding[: self.config.dimensions]

        # Cache result
        if self._cache:
            self._cache.set(text, self.model_name, embedding)

        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            dimensions=len(embedding),
            cached=False,
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def embed_many(self, texts: list[str]) -> BatchEmbeddingResult:
        """Embed multiple texts with batching and caching.

        Args:
            texts: Texts to embed.

        Returns:
            Batch embedding result.
        """
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=self.model_name,
                dimensions=self.dimensions,
            )

        start_time = time.time()
        result_embeddings: list[Embedding | None] = [None] * len(texts)
        cached_count = 0

        # Check cache for all texts
        if self._cache:
            cached, uncached_indices = self._cache.get_many(texts, self.model_name)
            for idx, embedding in cached.items():
                result_embeddings[idx] = embedding
                cached_count += 1
            texts_to_embed = [texts[i] for i in uncached_indices]
            indices_to_embed = uncached_indices
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))

        # Embed uncached texts in batches
        computed_count = 0
        for batch_start in range(0, len(texts_to_embed), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[batch_start:batch_end]
            batch_indices = indices_to_embed[batch_start:batch_end]

            batch_embeddings = await self._embed_texts(batch_texts)

            for i, emb in enumerate(batch_embeddings):
                # Apply dimension reduction if configured
                final_emb = emb
                if self.config.dimensions and len(emb) > self.config.dimensions:
                    final_emb = emb[: self.config.dimensions]

                idx = batch_indices[i]
                result_embeddings[idx] = final_emb
                computed_count += 1

            # Cache computed embeddings
            if self._cache:
                self._cache.set_many(batch_texts, self.model_name, batch_embeddings)

        # Ensure all embeddings are present
        final_embeddings: list[Embedding] = []
        for result_emb in result_embeddings:
            if result_emb is None:
                raise EmbeddingError("Missing embedding in result")
            final_embeddings.append(result_emb)

        return BatchEmbeddingResult(
            embeddings=final_embeddings,
            model=self.model_name,
            dimensions=self.dimensions,
            cached_count=cached_count,
            computed_count=computed_count,
            total_latency_ms=(time.time() - start_time) * 1000,
        )

    def cosine_similarity(
        self, embedding1: Embedding, embedding2: Embedding
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding.
            embedding2: Second embedding.

        Returns:
            Cosine similarity score (-1 to 1).
        """
        arr1: NDArray[np.floating[Any]] = np.array(embedding1)
        arr2: NDArray[np.floating[Any]] = np.array(embedding2)

        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    def find_most_similar(
        self,
        query_embedding: Embedding,
        embeddings: list[Embedding],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find the most similar embeddings to a query.

        Args:
            query_embedding: The query embedding.
            embeddings: List of embeddings to search.
            top_k: Number of top results to return.

        Returns:
            List of (index, similarity_score) tuples, sorted by score descending.
        """
        if not embeddings:
            return []

        query_arr: NDArray[np.floating[Any]] = np.array(query_embedding)
        embeddings_arr: NDArray[np.floating[Any]] = np.array(embeddings)

        # Compute all similarities at once
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0:
            return [(i, 0.0) for i in range(min(top_k, len(embeddings)))]

        embeddings_norms = np.linalg.norm(embeddings_arr, axis=1)
        # Avoid division by zero
        embeddings_norms = np.where(embeddings_norms == 0, 1, embeddings_norms)

        similarities = np.dot(embeddings_arr, query_arr) / (embeddings_norms * query_norm)

        # Get top-k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def clear_cache(self) -> int:
        """Clear the embedding cache.

        Returns:
            Number of entries cleared.
        """
        if self._cache:
            return self._cache.clear()
        return 0


class LocalEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers (local models)."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the local provider.

        Args:
            config: Configuration for the provider.
        """
        super().__init__(config)
        self._model: Any = None
        self._model_name: str = self.config.model_name
        self._dimensions: int | None = None

    def _load_model(self) -> Any:
        """Load the sentence-transformers model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as e:
            raise ModelNotFoundError(
                "sentence-transformers not installed. "
                "Install with: pip install 'memory-layer[phase1]'"
            ) from e

        try:
            self._model = SentenceTransformer(self._model_name)
            # Get model dimensions
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded model {self._model_name} with {self._dimensions} dimensions"
            )
            return self._model
        except Exception as e:
            raise ModelNotFoundError(
                f"Failed to load model {self._model_name}: {e}"
            ) from e

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self._dimensions is None:
            self._load_model()
        return self._dimensions or 384  # Default for MiniLM

    async def _embed_texts(self, texts: list[str]) -> list[Embedding]:
        """Embed multiple texts using sentence-transformers.

        Args:
            texts: Texts to embed.

        Returns:
            List of embeddings.
        """
        import asyncio  # noqa: PLC0415

        model = self._load_model()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )

        return [emb.tolist() for emb in embeddings]


class APIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using API (OpenAI-compatible)."""

    # Known model dimensions
    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "voyage-large-2": 1024,
        "voyage-code-2": 1536,
    }

    # Default API endpoints
    DEFAULT_API_URLS: ClassVar[dict[str, str]] = {
        "openai": "https://api.openai.com/v1/embeddings",
        "voyage": "https://api.voyageai.com/v1/embeddings",
    }

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        api_provider: str = "openai",
    ) -> None:
        """Initialize the API provider.

        Args:
            config: Configuration for the provider.
            api_provider: API provider name (openai, voyage).
        """
        super().__init__(config)
        self._api_provider = api_provider

        # Set default model based on provider
        if self.config.model_name == "all-MiniLM-L6-v2":  # Default local model
            if api_provider == "openai":
                self.config.model_name = "text-embedding-3-small"
            elif api_provider == "voyage":
                self.config.model_name = "voyage-large-2"

        # Set API URL
        if not self.config.api_base_url:
            self.config.api_base_url = self.DEFAULT_API_URLS.get(
                api_provider, self.DEFAULT_API_URLS["openai"]
            )

    def _get_api_key(self) -> str:
        """Get the API key from environment."""
        # Try provider-specific env var first
        provider_env_vars = {
            "openai": "OPENAI_API_KEY",
            "voyage": "VOYAGE_API_KEY",
        }
        env_var = provider_env_vars.get(self._api_provider, self.config.api_key_env_var)

        api_key = os.environ.get(env_var) or os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise APIError(
                f"API key not found. Set {env_var} or {self.config.api_key_env_var} "
                "environment variable."
            )
        return api_key

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self.config.dimensions:
            return self.config.dimensions
        return self.MODEL_DIMENSIONS.get(self.config.model_name, 1536)

    async def _embed_texts(self, texts: list[str]) -> list[Embedding]:
        """Embed multiple texts using API.

        Args:
            texts: Texts to embed.

        Returns:
            List of embeddings.
        """
        import httpx  # noqa: PLC0415

        api_key = self._get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "input": texts,
            "model": self.config.model_name,
        }

        # Add dimensions parameter if supported and configured
        if self.config.dimensions and self.config.model_name.startswith("text-embedding-3"):
            payload["dimensions"] = self.config.dimensions

        last_error: Exception | None = None
        for attempt in range(self.config.api_max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.config.api_timeout) as client:
                    response = await client.post(
                        self.config.api_base_url or "",
                        headers=headers,
                        json=payload,
                    )

                    if response.status_code == 429:
                        # Rate limited, wait and retry
                        wait_time = min(2**attempt, 32)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        import asyncio  # noqa: PLC0415

                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = response.json()

                    # Extract embeddings (sorted by index)
                    embedding_data = sorted(data["data"], key=lambda x: x["index"])
                    return [item["embedding"] for item in embedding_data]

            except httpx.HTTPStatusError as e:
                last_error = APIError(f"API request failed: {e.response.status_code} - {e.response.text}")
            except httpx.RequestError as e:
                last_error = APIError(f"API request failed: {e}")
            except (KeyError, json.JSONDecodeError) as e:
                last_error = APIError(f"Invalid API response: {e}")

            if attempt < self.config.api_max_retries - 1:
                import asyncio  # noqa: PLC0415

                await asyncio.sleep(2**attempt)

        raise last_error or APIError("API request failed after retries")


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        fixed_embedding: Embedding | None = None,
    ) -> None:
        """Initialize the mock provider.

        Args:
            config: Configuration for the provider.
            fixed_embedding: Fixed embedding to return (for deterministic tests).
        """
        # Disable cache for mock provider by default
        if config is None:
            config = EmbeddingConfig(cache_enabled=False)
        super().__init__(config)
        self._fixed_embedding = fixed_embedding
        self._call_count = 0
        self._embedded_texts: list[str] = []

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return "mock-embedding-model"

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.config.dimensions or 384

    @property
    def call_count(self) -> int:
        """Get the number of embedding calls made."""
        return self._call_count

    @property
    def embedded_texts(self) -> list[str]:
        """Get all texts that were embedded."""
        return self._embedded_texts

    def reset(self) -> None:
        """Reset call tracking."""
        self._call_count = 0
        self._embedded_texts = []

    async def _embed_texts(self, texts: list[str]) -> list[Embedding]:
        """Generate mock embeddings.

        Args:
            texts: Texts to embed.

        Returns:
            List of mock embeddings.
        """
        self._call_count += 1
        self._embedded_texts.extend(texts)

        embeddings: list[Embedding] = []
        for text in texts:
            if self._fixed_embedding:
                embeddings.append(self._fixed_embedding.copy())
            else:
                # Generate deterministic embedding based on text hash
                # MD5 is used only for creating deterministic seeds, not for security
                text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()  # noqa: S324
                seed = int(text_hash[:8], 16)
                rng = np.random.default_rng(seed)
                embedding = rng.random(self.dimensions).tolist()
                # Normalize
                norm = sum(x**2 for x in embedding) ** 0.5
                embeddings.append([x / norm for x in embedding])

        return embeddings


def get_embedding_provider(
    provider_type: str = "local",
    config: EmbeddingConfig | None = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Factory function to create embedding providers.

    Args:
        provider_type: Type of provider ("local", "openai", "voyage", "mock").
        config: Configuration for the provider.
        **kwargs: Additional arguments passed to the provider.

    Returns:
        Embedding provider instance.
    """
    if provider_type == "local":
        return LocalEmbeddingProvider(config)
    elif provider_type == "openai":
        return APIEmbeddingProvider(config, api_provider="openai")
    elif provider_type == "voyage":
        return APIEmbeddingProvider(config, api_provider="voyage")
    elif provider_type == "mock":
        return MockEmbeddingProvider(config, **kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

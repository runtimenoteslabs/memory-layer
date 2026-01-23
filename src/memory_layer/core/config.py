"""
Configuration management for Memory Layer.

Provides:
- Environment-based configuration
- Validation with Pydantic
- Support for multiple environments (dev, test, prod)
- Secrets management via environment variables
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: Path = Field(
        default_factory=lambda: Path.home() / ".memory-layer" / "memories.db",
        description="Path to SQLite database file",
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL queries (for debugging)",
    )
    pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Connection pool size",
    )
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Database operation timeout in seconds",
    )

    @field_validator("path", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        """Expand user home directory in path."""
        if isinstance(v, str):
            v = Path(v).expanduser()
        return v


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable embedding cache",
    )
    cache_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum number of cached embeddings",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for embedding generation",
    )
    device: str = Field(
        default="cpu",
        description="Device for embedding model (cpu/cuda/mps)",
    )


class RetrievalConfig(BaseModel):
    """Retrieval system configuration."""

    semantic_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for semantic similarity (0-1)",
    )
    outcome_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for outcome score (0-1)",
    )
    recency_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for recency (0-1)",
    )
    frequency_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for usage frequency (0-1)",
    )
    confidence_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for confidence score (0-1)",
    )
    recency_half_life_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Half-life for recency decay in days",
    )
    default_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of results to return",
    )


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(
        default="127.0.0.1",
        description="Server bind address",
    )
    port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Server port number",
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of server workers",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:*", "http://127.0.0.1:*"],
        description="Allowed CORS origins",
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds",
    )


class ExtractionConfig(BaseModel):
    """LLM extraction configuration."""

    model: str = Field(
        default="claude-3-haiku-20240307",
        description="LLM model for extraction",
    )
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum tokens for extraction response",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM temperature for extraction",
    )
    timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="API call timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Maximum retry attempts for API calls",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="WARNING",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    format: str = Field(
        default="text",
        description="Log format (text or json)",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Include timestamp in log output",
    )
    log_file: Path | None = Field(
        default=None,
        description="Optional log file path",
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v


class Settings(BaseSettings):
    """Main application settings.

    Configuration is loaded in priority order:
    1. Environment variables (highest priority)
    2. .env file in current directory
    3. Default values (lowest priority)

    Environment variables use MEMORY_LAYER_ prefix:
    - MEMORY_LAYER_ENV=production
    - MEMORY_LAYER_DB_PATH=/path/to/db
    - MEMORY_LAYER_LOG_LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_LAYER_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Environment
    env: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )

    # API Keys (from environment only, never stored)
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for extraction",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (fallback)",
    )

    # Component configs
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Feature flags
    enable_extraction: bool = Field(
        default=True,
        description="Enable LLM-based extraction",
    )
    enable_metrics: bool = Field(
        default=False,
        description="Enable Prometheus metrics",
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing",
    )

    # Project context
    project_name: str | None = Field(
        default=None,
        description="Current project name (auto-detected if not set)",
    )
    project_path: Path | None = Field(
        default=None,
        description="Current project path",
    )

    @classmethod
    def for_testing(cls) -> "Settings":
        """Create settings optimized for testing."""
        return cls(
            env=Environment.TESTING,
            database=DatabaseConfig(
                path=Path(":memory:"),
                echo=False,
            ),
            embedding=EmbeddingConfig(
                cache_enabled=False,
            ),
            logging=LoggingConfig(
                level="DEBUG",
            ),
            enable_extraction=False,
            enable_metrics=False,
            enable_tracing=False,
        )

    @classmethod
    def for_production(cls) -> "Settings":
        """Create settings optimized for production."""
        return cls(
            env=Environment.PRODUCTION,
            logging=LoggingConfig(
                level="WARNING",
                format="json",
            ),
            enable_metrics=True,
        )

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env == Environment.DEVELOPMENT

    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.env == Environment.TESTING

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env == Environment.PRODUCTION

    def get_db_path(self) -> Path:
        """Get resolved database path."""
        path = self.database.path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def validate_for_extraction(self) -> bool:
        """Check if extraction is properly configured."""
        return bool(self.anthropic_api_key or self.openai_api_key)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings.

    Settings are loaded once and cached for the lifetime of the application.
    Use clear_settings_cache() to reload settings.

    Returns:
        The application settings
    """
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache, forcing reload on next access."""
    get_settings.cache_clear()


def get_config_path() -> Path:
    """Get the configuration directory path."""
    config_dir = Path.home() / ".memory-layer"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_data_path() -> Path:
    """Get the data directory path."""
    data_dir = Path.home() / ".memory-layer" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_cache_path() -> Path:
    """Get the cache directory path."""
    cache_dir = Path.home() / ".memory-layer" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# Environment variable helpers
def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with MEMORY_LAYER_ prefix.

    Args:
        key: Variable name (without prefix)
        default: Default value if not set

    Returns:
        The environment variable value or default
    """
    return os.getenv(f"MEMORY_LAYER_{key}", default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable.

    Recognizes: true, 1, yes, on (case-insensitive)

    Args:
        key: Variable name (without prefix)
        default: Default value if not set

    Returns:
        Boolean value
    """
    value = get_env(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable.

    Args:
        key: Variable name (without prefix)
        default: Default value if not set or invalid

    Returns:
        Integer value
    """
    value = get_env(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable.

    Args:
        key: Variable name (without prefix)
        default: Default value if not set or invalid

    Returns:
        Float value
    """
    value = get_env(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

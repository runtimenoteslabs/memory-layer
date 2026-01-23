"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_layer.core.config import (
    DatabaseConfig,
    EmbeddingConfig,
    Environment,
    ExtractionConfig,
    LoggingConfig,
    RetrievalConfig,
    ServerConfig,
    Settings,
    clear_settings_cache,
    get_cache_path,
    get_config_path,
    get_data_path,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_settings,
)


class TestEnvironment:
    """Tests for Environment enum."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.PRODUCTION.value == "production"


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        assert config.path.name == "memories.db"
        assert config.echo is False
        assert config.pool_size == 5
        assert config.timeout == 30.0

    def test_custom_values(self):
        """Test custom database configuration."""
        config = DatabaseConfig(
            path=Path("/custom/path.db"),
            echo=True,
            pool_size=10,
            timeout=60.0,
        )
        assert config.path == Path("/custom/path.db")
        assert config.echo is True
        assert config.pool_size == 10
        assert config.timeout == 60.0

    def test_path_expansion(self):
        """Test that ~ is expanded in path."""
        config = DatabaseConfig(path="~/test.db")
        assert "~" not in str(config.path)
        assert config.path.is_absolute()

    def test_pool_size_validation(self):
        """Test pool size validation."""
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=0)
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=100)

    def test_timeout_validation(self):
        """Test timeout validation."""
        with pytest.raises(ValueError):
            DatabaseConfig(timeout=0.5)
        with pytest.raises(ValueError):
            DatabaseConfig(timeout=500)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.cache_enabled is True
        assert config.cache_size == 10000
        assert config.batch_size == 32
        assert config.device == "cpu"

    def test_cache_size_validation(self):
        """Test cache size validation."""
        with pytest.raises(ValueError):
            EmbeddingConfig(cache_size=50)
        with pytest.raises(ValueError):
            EmbeddingConfig(cache_size=200000)


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_weights(self):
        """Test default retrieval weights sum to 1."""
        config = RetrievalConfig()
        total = (
            config.semantic_weight
            + config.outcome_weight
            + config.recency_weight
            + config.frequency_weight
            + config.confidence_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_weight_validation(self):
        """Test weight validation."""
        with pytest.raises(ValueError):
            RetrievalConfig(semantic_weight=-0.1)
        with pytest.raises(ValueError):
            RetrievalConfig(semantic_weight=1.5)

    def test_half_life_validation(self):
        """Test half life days validation."""
        with pytest.raises(ValueError):
            RetrievalConfig(recency_half_life_days=0)
        with pytest.raises(ValueError):
            RetrievalConfig(recency_half_life_days=400)


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        """Test default server configuration."""
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.workers == 1
        assert config.request_timeout == 30.0

    def test_port_validation(self):
        """Test port validation."""
        with pytest.raises(ValueError):
            ServerConfig(port=80)  # Below 1024
        with pytest.raises(ValueError):
            ServerConfig(port=70000)  # Above 65535

    def test_cors_origins_default(self):
        """Test CORS origins default."""
        config = ServerConfig()
        assert "http://localhost:*" in config.cors_origins


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_values(self):
        """Test default extraction configuration."""
        config = ExtractionConfig()
        assert "claude" in config.model.lower() or "haiku" in config.model.lower()
        assert config.max_tokens == 1000
        assert config.temperature == 0.0
        assert config.max_retries == 3

    def test_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValueError):
            ExtractionConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            ExtractionConfig(temperature=2.0)


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "WARNING"
        assert config.format == "text"
        assert config.include_timestamp is True
        assert config.log_file is None

    def test_level_case_insensitive(self):
        """Test log level is case insensitive."""
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"

    def test_invalid_level(self):
        """Test invalid log level raises error."""
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingConfig(level="VERBOSE")


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()
        assert settings.env == Environment.DEVELOPMENT
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.embedding, EmbeddingConfig)
        assert isinstance(settings.retrieval, RetrievalConfig)
        assert isinstance(settings.server, ServerConfig)

    def test_for_testing(self):
        """Test testing configuration."""
        settings = Settings.for_testing()
        assert settings.env == Environment.TESTING
        assert settings.database.path == Path(":memory:")
        assert settings.enable_extraction is False
        assert settings.logging.level == "DEBUG"

    def test_for_production(self):
        """Test production configuration."""
        settings = Settings.for_production()
        assert settings.env == Environment.PRODUCTION
        assert settings.logging.level == "WARNING"
        assert settings.logging.format == "json"
        assert settings.enable_metrics is True

    def test_environment_helpers(self):
        """Test environment helper methods."""
        dev_settings = Settings(env=Environment.DEVELOPMENT)
        assert dev_settings.is_development() is True
        assert dev_settings.is_testing() is False
        assert dev_settings.is_production() is False

        prod_settings = Settings(env=Environment.PRODUCTION)
        assert prod_settings.is_production() is True

    def test_validate_for_extraction(self):
        """Test extraction validation."""
        settings = Settings(anthropic_api_key=None, openai_api_key=None)
        assert settings.validate_for_extraction() is False

        settings = Settings(anthropic_api_key="sk-test")
        assert settings.validate_for_extraction() is True

    @patch.dict(os.environ, {"MEMORY_LAYER_ENV": "production"})
    def test_env_override(self):
        """Test environment variable override."""
        clear_settings_cache()
        settings = Settings()
        assert settings.env == Environment.PRODUCTION


class TestSettingsCache:
    """Tests for settings caching."""

    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        clear_settings_cache()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_clear_cache(self):
        """Test cache clearing."""
        clear_settings_cache()
        settings1 = get_settings()
        clear_settings_cache()
        settings2 = get_settings()
        # After clearing, should create new instance
        # (but values should be the same)
        assert settings1.env == settings2.env


class TestPathHelpers:
    """Tests for path helper functions."""

    def test_get_config_path(self):
        """Test config path creation."""
        path = get_config_path()
        assert path.exists()
        assert path.name == ".memory-layer"

    def test_get_data_path(self):
        """Test data path creation."""
        path = get_data_path()
        assert path.exists()
        assert path.name == "data"

    def test_get_cache_path(self):
        """Test cache path creation."""
        path = get_cache_path()
        assert path.exists()
        assert path.name == "cache"


class TestEnvHelpers:
    """Tests for environment variable helpers."""

    def test_get_env(self):
        """Test basic env getter."""
        with patch.dict(os.environ, {"MEMORY_LAYER_TEST_VAR": "test_value"}):
            assert get_env("TEST_VAR") == "test_value"

    def test_get_env_default(self):
        """Test env getter with default."""
        assert get_env("NONEXISTENT_VAR", "default") == "default"

    def test_get_env_bool_true(self):
        """Test boolean env getter with true values."""
        for value in ["true", "True", "1", "yes", "on", "YES"]:
            with patch.dict(os.environ, {"MEMORY_LAYER_BOOL_VAR": value}):
                assert get_env_bool("BOOL_VAR") is True

    def test_get_env_bool_false(self):
        """Test boolean env getter with false values."""
        for value in ["false", "0", "no", "off"]:
            with patch.dict(os.environ, {"MEMORY_LAYER_BOOL_VAR": value}):
                assert get_env_bool("BOOL_VAR") is False

    def test_get_env_bool_default(self):
        """Test boolean env getter default."""
        assert get_env_bool("NONEXISTENT_BOOL") is False
        assert get_env_bool("NONEXISTENT_BOOL", True) is True

    def test_get_env_int(self):
        """Test integer env getter."""
        with patch.dict(os.environ, {"MEMORY_LAYER_INT_VAR": "42"}):
            assert get_env_int("INT_VAR") == 42

    def test_get_env_int_invalid(self):
        """Test integer env getter with invalid value."""
        with patch.dict(os.environ, {"MEMORY_LAYER_INT_VAR": "not_a_number"}):
            assert get_env_int("INT_VAR", 10) == 10

    def test_get_env_float(self):
        """Test float env getter."""
        with patch.dict(os.environ, {"MEMORY_LAYER_FLOAT_VAR": "3.14"}):
            assert get_env_float("FLOAT_VAR") == 3.14

    def test_get_env_float_invalid(self):
        """Test float env getter with invalid value."""
        with patch.dict(os.environ, {"MEMORY_LAYER_FLOAT_VAR": "not_a_float"}):
            assert get_env_float("FLOAT_VAR", 1.0) == 1.0


class TestNestedConfig:
    """Tests for nested configuration with env vars."""

    @patch.dict(
        os.environ,
        {
            "MEMORY_LAYER_DATABASE__POOL_SIZE": "10",
            "MEMORY_LAYER_SERVER__PORT": "9000",
        },
    )
    def test_nested_env_override(self):
        """Test nested configuration via environment."""
        clear_settings_cache()
        settings = Settings()
        assert settings.database.pool_size == 10
        assert settings.server.port == 9000

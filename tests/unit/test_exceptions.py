"""Tests for the centralized exception hierarchy."""

import pytest

from memory_layer.core.exceptions import (
    AuthenticationError,
    BeadsNotFoundError,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseIntegrityError,
    DaemonAlreadyRunningError,
    DaemonError,
    EmbeddingError,
    ErrorCode,
    ExtractionError,
    HookError,
    HookNotInstalledError,
    InitializationError,
    InvalidResponseError,
    LLMAPIError,
    MCPError,
    MemoryLayerError,
    MemoryNotFoundError,
    ModelNotFoundError,
    RateLimitError,
    RetrievalError,
    SDKError,
    SearchTimeoutError,
    ServerError,
    StorageError,
    TaskError,
    TaskLinkError,
    TaskSyncError,
    ValidationError,
    format_error,
    is_recoverable,
)


class TestMemoryLayerError:
    """Tests for base MemoryLayerError."""

    def test_basic_creation(self):
        """Test basic error creation."""
        error = MemoryLayerError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code == ErrorCode.UNKNOWN
        assert error.details == {}
        assert error.recoverable is False

    def test_with_code_and_details(self):
        """Test error with code and details."""
        error = MemoryLayerError(
            "Database error",
            code=ErrorCode.STORAGE_ERROR,
            details={"table": "memories"},
            recoverable=True,
        )
        assert error.code == ErrorCode.STORAGE_ERROR
        assert error.details == {"table": "memories"}
        assert error.recoverable is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = MemoryLayerError(
            "Test error",
            code=ErrorCode.VALIDATION_ERROR,
            details={"field": "content"},
            recoverable=False,
        )
        result = error.to_dict()
        assert result == {
            "error": "Test error",
            "code": "ML6003",
            "details": {"field": "content"},
            "recoverable": False,
        }

    def test_user_message(self):
        """Test user-friendly message."""
        error = MemoryLayerError("Technical error details")
        assert error.user_message() == "Technical error details"


class TestStorageExceptions:
    """Tests for storage-related exceptions."""

    def test_storage_error(self):
        """Test base storage error."""
        error = StorageError("Database failure")
        assert error.code == ErrorCode.STORAGE_ERROR
        assert isinstance(error, MemoryLayerError)

    def test_memory_not_found(self):
        """Test MemoryNotFoundError."""
        error = MemoryNotFoundError("mem-12345")
        assert "mem-12345" in str(error)
        assert error.code == ErrorCode.MEMORY_NOT_FOUND
        assert error.details == {"memory_id": "mem-12345"}
        assert error.recoverable is False

    def test_database_connection_error(self):
        """Test DatabaseConnectionError."""
        error = DatabaseConnectionError("Connection refused")
        assert error.code == ErrorCode.DATABASE_CONNECTION
        assert error.recoverable is True  # Connection errors are recoverable

    def test_database_integrity_error(self):
        """Test DatabaseIntegrityError."""
        error = DatabaseIntegrityError()
        assert error.code == ErrorCode.DATABASE_INTEGRITY
        assert error.recoverable is False  # Integrity errors are not recoverable


class TestRetrievalExceptions:
    """Tests for retrieval-related exceptions."""

    def test_retrieval_error(self):
        """Test base retrieval error."""
        error = RetrievalError("Search failed")
        assert error.code == ErrorCode.RETRIEVAL_ERROR
        assert isinstance(error, MemoryLayerError)

    def test_embedding_error(self):
        """Test EmbeddingError."""
        error = EmbeddingError("Model not loaded")
        assert error.code == ErrorCode.EMBEDDING_ERROR
        assert error.recoverable is True

    def test_model_not_found(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("all-MiniLM-L6-v2")
        assert "all-MiniLM-L6-v2" in str(error)
        assert error.code == ErrorCode.MODEL_NOT_FOUND
        assert error.details == {"model_name": "all-MiniLM-L6-v2"}

    def test_search_timeout(self):
        """Test SearchTimeoutError."""
        error = SearchTimeoutError(30.0)
        assert "30" in str(error)
        assert error.code == ErrorCode.SEARCH_TIMEOUT
        assert error.details == {"timeout_seconds": 30.0}
        assert error.recoverable is True


class TestExtractionExceptions:
    """Tests for extraction-related exceptions."""

    def test_extraction_error(self):
        """Test base extraction error."""
        error = ExtractionError("Extraction failed")
        assert error.code == ErrorCode.EXTRACTION_ERROR
        assert isinstance(error, MemoryLayerError)

    def test_llm_api_error(self):
        """Test LLMAPIError."""
        error = LLMAPIError("API returned 500", status_code=500)
        assert error.code == ErrorCode.LLM_API_ERROR
        assert error.details == {"status_code": 500}
        assert error.recoverable is True

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Too many requests", retry_after=60.0)
        assert error.code == ErrorCode.RATE_LIMIT
        assert error.details == {"retry_after": 60.0}
        assert error.recoverable is True

    def test_invalid_response_error(self):
        """Test InvalidResponseError."""
        error = InvalidResponseError("Cannot parse JSON")
        assert error.code == ErrorCode.INVALID_RESPONSE
        assert error.recoverable is True


class TestServerExceptions:
    """Tests for server-related exceptions."""

    def test_server_error(self):
        """Test base server error."""
        error = ServerError("Server crashed")
        assert error.code == ErrorCode.SERVER_ERROR
        assert isinstance(error, MemoryLayerError)

    def test_mcp_error(self):
        """Test MCPError."""
        error = MCPError("Invalid method", mcp_code=-32601)
        assert error.code == ErrorCode.MCP_ERROR
        assert error.details == {"mcp_code": -32601}

    def test_hook_error(self):
        """Test HookError."""
        error = HookError("SessionStart", "Script failed")
        assert "SessionStart" in str(error)
        assert error.code == ErrorCode.HOOK_ERROR
        assert error.details == {"hook_name": "SessionStart"}
        assert error.recoverable is True

    def test_hook_not_installed(self):
        """Test HookNotInstalledError."""
        error = HookNotInstalledError("PreCompact")
        assert "PreCompact" in str(error)
        assert error.code == ErrorCode.HOOK_ERROR

    def test_daemon_error(self):
        """Test DaemonError."""
        error = DaemonError("Daemon crashed")
        assert error.code == ErrorCode.DAEMON_ERROR

    def test_daemon_already_running(self):
        """Test DaemonAlreadyRunningError."""
        error = DaemonAlreadyRunningError()
        assert "already running" in str(error).lower()


class TestSDKExceptions:
    """Tests for SDK-related exceptions."""

    def test_sdk_error(self):
        """Test base SDK error."""
        error = SDKError("Client error")
        assert error.code == ErrorCode.SDK_ERROR
        assert isinstance(error, MemoryLayerError)

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid API key")
        assert error.code == ErrorCode.AUTHENTICATION_ERROR
        assert error.recoverable is False

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Content too long", field="content")
        assert error.code == ErrorCode.VALIDATION_ERROR
        assert error.details == {"field": "content"}
        assert error.recoverable is False


class TestTaskExceptions:
    """Tests for task integration exceptions."""

    def test_task_error(self):
        """Test base task error."""
        error = TaskError("Task failed")
        assert error.code == ErrorCode.TASK_ERROR
        assert isinstance(error, MemoryLayerError)

    def test_beads_not_found(self):
        """Test BeadsNotFoundError."""
        error = BeadsNotFoundError("/project/.beads")
        assert "/project/.beads" in str(error)
        assert error.code == ErrorCode.BEADS_NOT_FOUND

    def test_beads_not_found_no_path(self):
        """Test BeadsNotFoundError without path."""
        error = BeadsNotFoundError()
        assert "not found" in str(error).lower()

    def test_task_sync_error(self):
        """Test TaskSyncError."""
        error = TaskSyncError("Sync failed", task_id="bd-a3f8")
        assert error.code == ErrorCode.TASK_SYNC_ERROR
        assert error.details == {"task_id": "bd-a3f8"}
        assert error.recoverable is True

    def test_task_link_error(self):
        """Test TaskLinkError."""
        error = TaskLinkError("bd-a3f8", "mem-123", "Memory archived")
        assert "bd-a3f8" in str(error)
        assert "mem-123" in str(error)
        assert error.code == ErrorCode.TASK_LINK_ERROR


class TestConfigurationExceptions:
    """Tests for configuration exceptions."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid port", config_key="port")
        assert error.code == ErrorCode.CONFIGURATION_ERROR
        assert error.details == {"config_key": "port"}

    def test_initialization_error(self):
        """Test InitializationError."""
        error = InitializationError("MemoryEngine", "Database not found")
        assert "MemoryEngine" in str(error)
        assert error.code == ErrorCode.INITIALIZATION_ERROR
        assert error.details == {"component": "MemoryEngine"}


class TestErrorFormatting:
    """Tests for error formatting utilities."""

    def test_format_memory_layer_error(self):
        """Test formatting MemoryLayerError."""
        error = MemoryNotFoundError("mem-123")
        result = format_error(error)
        assert "mem-123" in result

    def test_format_file_not_found(self):
        """Test formatting FileNotFoundError."""
        error = FileNotFoundError("config.json")
        result = format_error(error)
        assert "File not found" in result

    def test_format_permission_error(self):
        """Test formatting PermissionError."""
        error = PermissionError("Access denied")
        result = format_error(error)
        assert "Permission denied" in result

    def test_format_timeout_error(self):
        """Test formatting TimeoutError."""
        error = TimeoutError()
        result = format_error(error)
        assert "timed out" in result.lower()

    def test_format_unknown_error(self):
        """Test formatting unknown error type."""
        error = ValueError("Something went wrong")
        result = format_error(error)
        assert "error occurred" in result.lower()


class TestRecoverableCheck:
    """Tests for is_recoverable utility."""

    def test_recoverable_memory_layer_error(self):
        """Test recoverable MemoryLayerError."""
        error = DatabaseConnectionError()
        assert is_recoverable(error) is True

    def test_non_recoverable_memory_layer_error(self):
        """Test non-recoverable MemoryLayerError."""
        error = MemoryNotFoundError("mem-123")
        assert is_recoverable(error) is False

    def test_recoverable_python_error(self):
        """Test recoverable Python error."""
        error = TimeoutError()
        assert is_recoverable(error) is True

    def test_non_recoverable_python_error(self):
        """Test non-recoverable Python error."""
        error = ValueError()
        assert is_recoverable(error) is False

    def test_connection_refused(self):
        """Test ConnectionRefusedError is recoverable."""
        error = ConnectionRefusedError()
        assert is_recoverable(error) is True


class TestErrorCodeValues:
    """Tests for error code values."""

    def test_error_code_format(self):
        """Test error codes follow ML prefix pattern."""
        for code in ErrorCode:
            assert code.value.startswith("ML")
            # Should be ML followed by 4 digits
            assert len(code.value) == 6
            assert code.value[2:].isdigit()

    def test_error_code_categories(self):
        """Test error codes are in expected ranges."""
        # General: 1xxx
        assert ErrorCode.UNKNOWN.value.startswith("ML1")
        # Storage: 2xxx
        assert ErrorCode.STORAGE_ERROR.value.startswith("ML2")
        # Retrieval: 3xxx
        assert ErrorCode.RETRIEVAL_ERROR.value.startswith("ML3")
        # Extraction: 4xxx
        assert ErrorCode.EXTRACTION_ERROR.value.startswith("ML4")
        # Server: 5xxx
        assert ErrorCode.SERVER_ERROR.value.startswith("ML5")
        # SDK: 6xxx
        assert ErrorCode.SDK_ERROR.value.startswith("ML6")
        # Task: 7xxx
        assert ErrorCode.TASK_ERROR.value.startswith("ML7")

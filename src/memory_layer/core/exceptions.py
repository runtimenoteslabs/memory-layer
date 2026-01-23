"""
Centralized exception hierarchy for Memory Layer.

This module provides a consistent exception hierarchy with:
- Clear categorization (storage, retrieval, extraction, server, SDK)
- User-friendly error messages
- Structured error codes for programmatic handling
- Graceful degradation support
"""

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standardized error codes for programmatic handling."""

    # General errors (1xxx)
    UNKNOWN = "ML1000"
    INVALID_INPUT = "ML1001"
    CONFIGURATION_ERROR = "ML1002"
    INITIALIZATION_ERROR = "ML1003"

    # Storage errors (2xxx)
    STORAGE_ERROR = "ML2000"
    MEMORY_NOT_FOUND = "ML2001"
    DATABASE_CONNECTION = "ML2002"
    DATABASE_INTEGRITY = "ML2003"
    MIGRATION_ERROR = "ML2004"

    # Retrieval errors (3xxx)
    RETRIEVAL_ERROR = "ML3000"
    EMBEDDING_ERROR = "ML3001"
    MODEL_NOT_FOUND = "ML3002"
    SEARCH_TIMEOUT = "ML3003"

    # Extraction errors (4xxx)
    EXTRACTION_ERROR = "ML4000"
    LLM_API_ERROR = "ML4001"
    RATE_LIMIT = "ML4002"
    INVALID_RESPONSE = "ML4003"

    # Server errors (5xxx)
    SERVER_ERROR = "ML5000"
    MCP_ERROR = "ML5001"
    REST_API_ERROR = "ML5002"
    HOOK_ERROR = "ML5003"
    DAEMON_ERROR = "ML5004"

    # SDK errors (6xxx)
    SDK_ERROR = "ML6000"
    CONNECTION_ERROR = "ML6001"
    AUTHENTICATION_ERROR = "ML6002"
    VALIDATION_ERROR = "ML6003"

    # Task integration errors (7xxx)
    TASK_ERROR = "ML7000"
    BEADS_NOT_FOUND = "ML7001"
    TASK_SYNC_ERROR = "ML7002"
    TASK_LINK_ERROR = "ML7003"


class MemoryLayerError(Exception):
    """Base exception for all Memory Layer errors.

    Attributes:
        message: Human-readable error message
        code: Standardized error code for programmatic handling
        details: Additional context about the error
        recoverable: Whether the operation can be retried
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.message,
            "code": self.code.value,
            "details": self.details,
            "recoverable": self.recoverable,
        }

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        return self.message


# =============================================================================
# Storage Exceptions
# =============================================================================


class StorageError(MemoryLayerError):
    """Base exception for storage-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.STORAGE_ERROR,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message, code, details, recoverable)


class MemoryNotFoundError(StorageError):
    """Raised when a requested memory does not exist."""

    def __init__(self, memory_id: str):
        super().__init__(
            message=f"Memory not found: {memory_id}",
            code=ErrorCode.MEMORY_NOT_FOUND,
            details={"memory_id": memory_id},
            recoverable=False,
        )


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails."""

    def __init__(self, message: str = "Database connection failed"):
        super().__init__(
            message=message,
            code=ErrorCode.DATABASE_CONNECTION,
            recoverable=True,
        )


class DatabaseIntegrityError(StorageError):
    """Raised when database integrity check fails."""

    def __init__(self, message: str = "Database integrity check failed"):
        super().__init__(
            message=message,
            code=ErrorCode.DATABASE_INTEGRITY,
            recoverable=False,
        )


# =============================================================================
# Retrieval Exceptions
# =============================================================================


class RetrievalError(MemoryLayerError):
    """Base exception for retrieval-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.RETRIEVAL_ERROR,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message, code, details, recoverable)


class EmbeddingError(RetrievalError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str = "Embedding generation failed"):
        super().__init__(
            message=message,
            code=ErrorCode.EMBEDDING_ERROR,
            recoverable=True,
        )


class ModelNotFoundError(RetrievalError):
    """Raised when embedding model is not available."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Embedding model not found: {model_name}",
            code=ErrorCode.MODEL_NOT_FOUND,
            details={"model_name": model_name},
            recoverable=False,
        )


class SearchTimeoutError(RetrievalError):
    """Raised when search operation times out."""

    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"Search timed out after {timeout_seconds}s",
            code=ErrorCode.SEARCH_TIMEOUT,
            details={"timeout_seconds": timeout_seconds},
            recoverable=True,
        )


# =============================================================================
# Extraction Exceptions
# =============================================================================


class ExtractionError(MemoryLayerError):
    """Base exception for extraction-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.EXTRACTION_ERROR,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message, code, details, recoverable)


class LLMAPIError(ExtractionError):
    """Raised when LLM API call fails."""

    def __init__(
        self,
        message: str = "LLM API request failed",
        status_code: int | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.LLM_API_ERROR,
            details={"status_code": status_code} if status_code else {},
            recoverable=True,
        )


class RateLimitError(ExtractionError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.RATE_LIMIT,
            details={"retry_after": retry_after} if retry_after else {},
            recoverable=True,
        )


class InvalidResponseError(ExtractionError):
    """Raised when LLM response cannot be parsed."""

    def __init__(self, message: str = "Invalid response from LLM"):
        super().__init__(
            message=message,
            code=ErrorCode.INVALID_RESPONSE,
            recoverable=True,
        )


# =============================================================================
# Server Exceptions
# =============================================================================


class ServerError(MemoryLayerError):
    """Base exception for server-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SERVER_ERROR,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message, code, details, recoverable)


class MCPError(ServerError):
    """Raised when MCP protocol operation fails."""

    def __init__(self, message: str, mcp_code: int | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.MCP_ERROR,
            details={"mcp_code": mcp_code} if mcp_code else {},
            recoverable=False,
        )


class HookError(ServerError):
    """Raised when Claude Code hook execution fails."""

    def __init__(self, hook_name: str, message: str):
        super().__init__(
            message=f"Hook '{hook_name}' failed: {message}",
            code=ErrorCode.HOOK_ERROR,
            details={"hook_name": hook_name},
            recoverable=True,
        )


class HookNotInstalledError(HookError):
    """Raised when a required hook is not installed."""

    def __init__(self, hook_name: str):
        super().__init__(
            hook_name=hook_name,
            message="not installed",
        )


class DaemonError(ServerError):
    """Raised when daemon operation fails."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code=ErrorCode.DAEMON_ERROR,
            recoverable=False,
        )


class DaemonAlreadyRunningError(DaemonError):
    """Raised when daemon is already running."""

    def __init__(self):
        super().__init__(message="Daemon is already running")


# =============================================================================
# SDK Exceptions
# =============================================================================


class SDKError(MemoryLayerError):
    """Base exception for SDK client errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SDK_ERROR,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message, code, details, recoverable)


class ConnectionError(SDKError):
    """Raised when connection to server fails."""

    def __init__(self, message: str = "Connection to server failed"):
        super().__init__(
            message=message,
            code=ErrorCode.CONNECTION_ERROR,
            recoverable=True,
        )


class AuthenticationError(SDKError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            code=ErrorCode.AUTHENTICATION_ERROR,
            recoverable=False,
        )


class ValidationError(SDKError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            details={"field": field} if field else {},
            recoverable=False,
        )


# =============================================================================
# Task Integration Exceptions
# =============================================================================


class TaskError(MemoryLayerError):
    """Base exception for task integration errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.TASK_ERROR,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message, code, details, recoverable)


class BeadsNotFoundError(TaskError):
    """Raised when Beads directory is not found."""

    def __init__(self, path: str | None = None):
        msg = "Beads directory not found"
        if path:
            msg += f": {path}"
        super().__init__(
            message=msg,
            code=ErrorCode.BEADS_NOT_FOUND,
            details={"path": path} if path else {},
            recoverable=False,
        )


class TaskSyncError(TaskError):
    """Raised when task synchronization fails."""

    def __init__(self, message: str, task_id: str | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.TASK_SYNC_ERROR,
            details={"task_id": task_id} if task_id else {},
            recoverable=True,
        )


class TaskLinkError(TaskError):
    """Raised when task-memory linking fails."""

    def __init__(self, task_id: str, memory_id: str, reason: str):
        super().__init__(
            message=f"Failed to link task {task_id} to memory {memory_id}: {reason}",
            code=ErrorCode.TASK_LINK_ERROR,
            details={"task_id": task_id, "memory_id": memory_id},
            recoverable=True,
        )


# =============================================================================
# Resilience Exceptions
# =============================================================================


class CircuitOpenError(MemoryLayerError):
    """Raised when circuit breaker is open and rejecting requests."""

    def __init__(self, circuit_name: str):
        super().__init__(
            message=f"Service temporarily unavailable (circuit '{circuit_name}' is open)",
            code=ErrorCode.SERVER_ERROR,
            details={"circuit_name": circuit_name},
            recoverable=True,
        )


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(MemoryLayerError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.CONFIGURATION_ERROR,
            details={"config_key": config_key} if config_key else {},
            recoverable=False,
        )


class InitializationError(MemoryLayerError):
    """Raised when component initialization fails."""

    def __init__(self, component: str, message: str):
        super().__init__(
            message=f"Failed to initialize {component}: {message}",
            code=ErrorCode.INITIALIZATION_ERROR,
            details={"component": component},
            recoverable=False,
        )


# =============================================================================
# Error Formatting Utilities
# =============================================================================


def format_error(error: Exception) -> str:
    """Format an exception for user-friendly display.

    Converts technical exceptions into human-readable messages.

    Args:
        error: The exception to format

    Returns:
        A user-friendly error message
    """
    if isinstance(error, MemoryLayerError):
        return error.user_message()

    # Map common Python exceptions to friendly messages
    error_type = type(error).__name__
    error_msg = str(error)

    friendly_messages = {
        "FileNotFoundError": f"File not found: {error_msg}",
        "PermissionError": f"Permission denied: {error_msg}",
        "TimeoutError": "Operation timed out. Please try again.",
        "json.JSONDecodeError": "Invalid JSON data received.",
        "sqlite3.OperationalError": f"Database error: {error_msg}",
        "sqlite3.IntegrityError": "Database constraint violation.",
        "aiohttp.ClientError": "Network request failed. Check your connection.",
        "httpx.HTTPError": "HTTP request failed. Check the server status.",
    }

    return friendly_messages.get(error_type, f"An error occurred: {error_msg}")


def is_recoverable(error: Exception) -> bool:
    """Check if an error is recoverable (can be retried).

    Args:
        error: The exception to check

    Returns:
        True if the operation can be retried
    """
    if isinstance(error, MemoryLayerError):
        return error.recoverable

    # Common recoverable Python exceptions
    recoverable_types = (
        TimeoutError,
        ConnectionRefusedError,
        ConnectionResetError,
        BrokenPipeError,
    )
    return isinstance(error, recoverable_types)

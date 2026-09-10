"""Python SDK for Runtime Memory.

This module provides a unified client interface for accessing Runtime Memory
functionality in both local (direct engine access) and remote (REST API) modes.

Quick Start:
    ```python
    from runtime_memory.sdk import MemoryClient

    # Async client (recommended)
    async with MemoryClient(mode="local") as client:
        memory = await client.add(
            content="Use async/await for I/O operations",
            category="pattern",
        )
        results = await client.search("async patterns")
        await client.record_outcome(memory.id, "worked")

    # Sync client for simple scripts
    from runtime_memory.sdk import SyncMemoryClient

    with SyncMemoryClient(mode="local") as client:
        memory = client.add("Use async/await for I/O", category="pattern")
        results = client.search("async patterns")
    ```

Remote Mode:
    ```python
    # Connect to a running Runtime Memory server
    async with MemoryClient(
        mode="remote",
        base_url="http://localhost:8080",
        api_key="your-api-key",
    ) as client:
        results = await client.search("error handling")
    ```

Module-Level Functions:
    ```python
    from runtime_memory.sdk import configure, add, search

    configure(mode="local", db_path="~/.my-app/memories.db")

    memory = await add("Use type hints", category="convention")
    results = await search("type hints")
    ```
"""

from runtime_memory.sdk.client import (
    AuthenticationError,
    # Configuration
    ClientConfig,
    ClientMode,
    ConnectionError,
    # Main classes
    MemoryClient,
    NotFoundError,
    RateLimitError,
    # Exceptions
    SDKError,
    # Response types
    StatsDict,
    SyncMemoryClient,
    ValidationError,
    add,
    close_default_client,
    # Module-level functions
    configure,
    get_context,
    record_outcome,
    search,
)

__all__ = [
    # Main classes
    "MemoryClient",
    "SyncMemoryClient",
    # Configuration
    "ClientConfig",
    "ClientMode",
    # Exceptions
    "SDKError",
    "ConnectionError",
    "AuthenticationError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    # Response types
    "StatsDict",
    # Module-level functions
    "configure",
    "add",
    "search",
    "get_context",
    "record_outcome",
    "close_default_client",
]

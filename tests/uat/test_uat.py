#!/usr/bin/env python3
"""
Memory Layer - User Acceptance Test Script

Tests REST API, Python SDK, and MCP tools from an end-user perspective.

Usage:
    # Run all tests
    python tests/uat/test_uat.py

    # Run specific test suite
    python tests/uat/test_uat.py --api      # REST API only
    python tests/uat/test_uat.py --sdk      # Python SDK only
    python tests/uat/test_uat.py --mcp      # MCP tools only

    # Verbose output
    python tests/uat/test_uat.py -v

Prerequisites:
    - For API tests: No server needed (script starts its own)
    - For SDK tests: No prerequisites
    - For MCP tests: No prerequisites
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    name: str
    passed: bool
    message: str = ""
    duration: float = 0.0


class TestRunner:
    """Runs UAT tests and collects results."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.temp_db: Path | None = None

    def log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"  {msg}")

    def record(self, test_id: str, name: str, passed: bool, message: str = "", duration: float = 0.0):
        """Record a test result."""
        result = TestResult(test_id, name, passed, message, duration)
        self.results.append(result)
        status = "✓" if passed else "✗"
        print(f"  [{status}] {test_id}: {name}")
        if not passed and message:
            print(f"      Error: {message}")

    @contextmanager
    def temp_database(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.temp_db = Path(tmpdir) / "test_memories.db"
            yield self.temp_db

    def summary(self) -> tuple[int, int]:
        """Print summary and return (passed, failed) counts."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print("\n" + "=" * 60)
        print(f"UAT Test Summary: {passed}/{total} passed, {failed} failed")
        print("=" * 60)

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.test_id}: {r.name}")
                    if r.message:
                        print(f"    {r.message}")

        return passed, failed


# =============================================================================
# REST API Tests
# =============================================================================

class APITests:
    """REST API UAT tests."""

    def __init__(self, runner: TestRunner, base_url: str = "http://localhost:8080"):
        self.runner = runner
        self.base_url = base_url
        self.created_ids: list[str] = []

    async def run_all(self):
        """Run all API tests with an embedded server."""
        print("\n" + "=" * 60)
        print("REST API Tests")
        print("=" * 60)

        try:
            import httpx
        except ImportError:
            print("  [!] httpx not installed, skipping API tests")
            print("      Install with: pip install httpx")
            return

        with self.runner.temp_database() as db_path:
            # Start embedded test server
            from memory_layer.server.api import create_app
            from memory_layer.core.engine import MemoryEngine, EngineConfig

            config = EngineConfig(db_path=str(db_path), secure_permissions=False)
            engine = MemoryEngine(config=config)
            await engine.initialize()
            app = create_app(engine=engine)

            from httpx import AsyncClient, ASGITransport

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url=self.base_url
            ) as client:
                await self._test_health(client)
                await self._test_create_memory(client)
                await self._test_get_memory(client)
                await self._test_list_memories(client)
                await self._test_search_memories(client)
                await self._test_update_memory(client)
                await self._test_record_outcome(client)
                await self._test_get_context(client)
                await self._test_get_stats(client)
                await self._test_delete_memory(client)
                await self._test_error_handling(client)

            await engine.close()

    async def _test_health(self, client):
        """13.1: Health check endpoint."""
        start = time.time()
        try:
            resp = await client.get("/health")
            data = resp.json()
            passed = resp.status_code == 200 and data.get("status") == "healthy"
            self.runner.record("13.1", "Health check", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.1", "Health check", False, str(e))

    async def _test_create_memory(self, client):
        """13.2: Create memory endpoint."""
        start = time.time()
        try:
            resp = await client.post("/memories", json={
                "content": "Always use async/await in this project",
                "category": "convention"
            })
            data = resp.json()
            passed = resp.status_code == 201 and "id" in data
            if passed:
                self.created_ids.append(data["id"])
            self.runner.record("13.2", "Create memory", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.2", "Create memory", False, str(e))

    async def _test_get_memory(self, client):
        """13.3: Get memory by ID."""
        start = time.time()
        try:
            if not self.created_ids:
                self.runner.record("13.3", "Get memory by ID", False, "No memory created")
                return
            resp = await client.get(f"/memories/{self.created_ids[0]}")
            data = resp.json()
            passed = resp.status_code == 200 and data.get("content") == "Always use async/await in this project"
            self.runner.record("13.3", "Get memory by ID", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.3", "Get memory by ID", False, str(e))

    async def _test_list_memories(self, client):
        """13.6: List memories."""
        start = time.time()
        try:
            resp = await client.get("/memories")
            data = resp.json()
            # Response is MemoryListResponse with count and memories fields
            passed = resp.status_code == 200 and "memories" in data and len(data["memories"]) >= 1
            self.runner.record("13.6", "List memories", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.6", "List memories", False, str(e))

    async def _test_search_memories(self, client):
        """13.8: Search memories."""
        start = time.time()
        try:
            resp = await client.post("/memories/search", json={
                "query": "async await",
                "limit": 5
            })
            data = resp.json()
            # Response is SearchResponse with count and results fields
            passed = resp.status_code == 200 and "results" in data
            self.runner.record("13.8", "Search memories", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.8", "Search memories", False, str(e))

    async def _test_update_memory(self, client):
        """13.4: Update memory."""
        start = time.time()
        try:
            if not self.created_ids:
                self.runner.record("13.4", "Update memory", False, "No memory created")
                return
            resp = await client.patch(f"/memories/{self.created_ids[0]}", json={
                "content": "Always use async/await (updated)"
            })
            passed = resp.status_code == 200
            self.runner.record("13.4", "Update memory", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.4", "Update memory", False, str(e))

    async def _test_record_outcome(self, client):
        """13.9: Record outcome."""
        start = time.time()
        try:
            if not self.created_ids:
                self.runner.record("13.9", "Record outcome", False, "No memory created")
                return
            resp = await client.post("/memories/outcome", json={
                "memory_ids": [self.created_ids[0]],
                "outcome": "worked"
            })
            passed = resp.status_code == 200
            self.runner.record("13.9", "Record outcome", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.9", "Record outcome", False, str(e))

    async def _test_get_context(self, client):
        """13.10: Get context."""
        start = time.time()
        try:
            resp = await client.get("/context")
            passed = resp.status_code == 200
            self.runner.record("13.10", "Get context", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.10", "Get context", False, str(e))

    async def _test_get_stats(self, client):
        """13.11: Get statistics."""
        start = time.time()
        try:
            resp = await client.get("/stats")
            data = resp.json()
            passed = resp.status_code == 200 and "total_memories" in data
            self.runner.record("13.11", "Get statistics", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.11", "Get statistics", False, str(e))

    async def _test_delete_memory(self, client):
        """13.5: Delete memory."""
        start = time.time()
        try:
            # Create a memory to delete
            resp = await client.post("/memories", json={
                "content": "Memory to delete",
                "category": "general"
            })
            delete_id = resp.json()["id"]

            resp = await client.delete(f"/memories/{delete_id}")
            passed = resp.status_code == 204
            self.runner.record("13.5", "Delete memory", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.5", "Delete memory", False, str(e))

    async def _test_error_handling(self, client):
        """13.13-13.14: Error handling."""
        # Test validation error
        start = time.time()
        try:
            resp = await client.post("/memories", json={})
            passed = resp.status_code == 422
            self.runner.record("13.13", "Invalid request (422)", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.13", "Invalid request (422)", False, str(e))

        # Test not found
        start = time.time()
        try:
            resp = await client.get("/memories/nonexistent-id-12345")
            passed = resp.status_code == 404
            self.runner.record("13.14", "Not found (404)", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("13.14", "Not found (404)", False, str(e))


# =============================================================================
# Python SDK Tests
# =============================================================================

class SDKTests:
    """Python SDK UAT tests."""

    def __init__(self, runner: TestRunner):
        self.runner = runner
        self.created_ids: list[str] = []

    async def run_all(self):
        """Run all SDK tests."""
        print("\n" + "=" * 60)
        print("Python SDK Tests")
        print("=" * 60)

        with self.runner.temp_database() as db_path:
            await self._test_async_client(db_path)
            await self._test_sync_client(db_path)
            await self._test_module_functions(db_path)

    async def _test_async_client(self, db_path: Path):
        """14.1.x: Async client tests."""
        from memory_layer.sdk.client import MemoryClient, ClientConfig

        config = ClientConfig(db_path=str(db_path))

        # 14.1.1: Initialize client
        start = time.time()
        try:
            client = MemoryClient(config=config)
            await client.initialize()
            self.runner.record("14.1.1", "Initialize async client", True, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.1.1", "Initialize async client", False, str(e))
            return

        # 14.1.2: Add memory
        start = time.time()
        try:
            memory = await client.add("SDK test memory", category="convention")
            passed = memory is not None and memory.id is not None
            if passed:
                self.created_ids.append(memory.id)  # Store the ID, not the Memory object
            self.runner.record("14.1.2", "Add memory (async)", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.1.2", "Add memory (async)", False, str(e))

        # 14.1.3: Get memory
        start = time.time()
        try:
            if self.created_ids:
                memory = await client.get(self.created_ids[0])
                passed = memory is not None and memory.content == "SDK test memory"
                self.runner.record("14.1.3", "Get memory (async)", passed, duration=time.time() - start)
            else:
                self.runner.record("14.1.3", "Get memory (async)", False, "No memory created")
        except Exception as e:
            self.runner.record("14.1.3", "Get memory (async)", False, str(e))

        # 14.1.6: Search memories
        start = time.time()
        try:
            results = await client.search("SDK test", limit=5)
            passed = isinstance(results, list)
            self.runner.record("14.1.6", "Search memories (async)", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.1.6", "Search memories (async)", False, str(e))

        # 14.1.7: Record outcome
        start = time.time()
        try:
            if self.created_ids:
                await client.record_outcome(self.created_ids, "worked")
                self.runner.record("14.1.7", "Record outcome (async)", True, duration=time.time() - start)
            else:
                self.runner.record("14.1.7", "Record outcome (async)", False, "No memory created")
        except Exception as e:
            self.runner.record("14.1.7", "Record outcome (async)", False, str(e))

        # 14.1.8: Get context
        start = time.time()
        try:
            context = await client.get_context()
            passed = context is not None
            self.runner.record("14.1.8", "Get context (async)", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.1.8", "Get context (async)", False, str(e))

        # 14.1.9: Get stats
        start = time.time()
        try:
            stats = await client.stats()
            # In LOCAL mode, returns EngineStats object; in REMOTE mode, returns dict
            passed = stats is not None and (
                hasattr(stats, 'storage_stats') or  # EngineStats object
                isinstance(stats, dict)  # StatsDict from remote
            )
            self.runner.record("14.1.9", "Get stats (async)", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.1.9", "Get stats (async)", False, str(e))

        await client.close()

    async def _test_sync_client(self, db_path: Path):
        """14.3.x: Sync client tests.

        Note: SyncMemoryClient must be run in a separate thread to avoid
        deadlock when called from async code.
        """
        import concurrent.futures

        def run_sync_tests():
            """Run sync tests in a thread (separate event loop)."""
            from memory_layer.sdk.client import SyncMemoryClient, ClientConfig
            results = []

            config = ClientConfig(db_path=str(db_path))

            # 14.3.1: Initialize sync client
            start = time.time()
            try:
                client = SyncMemoryClient(config=config)
                client.initialize()  # Must initialize before use
                results.append(("14.3.1", "Initialize sync client", True, "", time.time() - start))
            except Exception as e:
                results.append(("14.3.1", "Initialize sync client", False, str(e), 0))
                return results

            # 14.3.2: Add memory sync
            start = time.time()
            try:
                memory = client.add("Sync SDK test", category="decision")
                passed = memory is not None
                results.append(("14.3.2", "Add memory (sync)", passed, "", time.time() - start))
            except Exception as e:
                results.append(("14.3.2", "Add memory (sync)", False, str(e), 0))

            # 14.3.3: Search sync
            start = time.time()
            try:
                search_results = client.search("Sync SDK")
                passed = isinstance(search_results, list)
                results.append(("14.3.3", "Search (sync)", passed, "", time.time() - start))
            except Exception as e:
                results.append(("14.3.3", "Search (sync)", False, str(e), 0))

            # 14.3.4: Context manager
            start = time.time()
            try:
                with SyncMemoryClient(config=config) as ctx_client:
                    ctx_client.add("Context manager test", category="general")
                results.append(("14.3.4", "Context manager", True, "", time.time() - start))
            except Exception as e:
                results.append(("14.3.4", "Context manager", False, str(e), 0))

            return results

        # Run sync tests in a thread pool to avoid event loop issues
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(executor, run_sync_tests)

        # Record results
        for test_id, name, passed, message, duration in results:
            self.runner.record(test_id, name, passed, message, duration)

    async def _test_module_functions(self, db_path: Path):
        """14.4.x: Module-level convenience functions."""
        import memory_layer.sdk.client as sdk

        # Reset global client
        sdk._global_client = None
        sdk._global_config = None

        # 14.4.1: Configure
        start = time.time()
        try:
            sdk.configure(db_path=str(db_path))
            self.runner.record("14.4.1", "Configure", True, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.4.1", "Configure", False, str(e))

        # 14.4.2: Quick add
        start = time.time()
        try:
            memory = await sdk.add("Module function test", "convention")
            passed = memory is not None
            self.runner.record("14.4.2", "Quick add", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.4.2", "Quick add", False, str(e))

        # 14.4.3: Quick search
        start = time.time()
        try:
            results = await sdk.search("Module function")
            passed = isinstance(results, list)
            self.runner.record("14.4.3", "Quick search", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.4.3", "Quick search", False, str(e))

        # 14.4.4: Quick context
        start = time.time()
        try:
            context = await sdk.get_context()
            passed = context is not None
            self.runner.record("14.4.4", "Quick context", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("14.4.4", "Quick context", False, str(e))


# =============================================================================
# MCP Tools Tests
# =============================================================================

class MCPTests:
    """MCP tools UAT tests."""

    def __init__(self, runner: TestRunner):
        self.runner = runner
        self.created_ids: list[str] = []

    async def run_all(self):
        """Run all MCP tests."""
        print("\n" + "=" * 60)
        print("MCP Tools Tests")
        print("=" * 60)

        with self.runner.temp_database() as db_path:
            from memory_layer.server.mcp import MCPServer
            from memory_layer.core.engine import MemoryEngine, EngineConfig

            config = EngineConfig(db_path=str(db_path), secure_permissions=False)
            engine = MemoryEngine(config=config)
            await engine.initialize()

            server = MCPServer(engine=engine)

            await self._test_list_tools(server)
            await self._test_add_memory(server)
            await self._test_search_memories(server)
            await self._test_record_outcome(server)
            await self._test_get_context(server)
            await self._test_update_memory(server)
            await self._test_list_memories(server)
            await self._test_get_stats(server)
            await self._test_delete_memory(server)
            await self._test_error_handling(server)

            await engine.close()

    async def _call_tool(self, server, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Helper to call an MCP tool."""
        from memory_layer.server.mcp import MCPRequest

        request = MCPRequest.from_dict({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        })
        response = await server.handle_request(request)
        return response.to_dict() if hasattr(response, 'to_dict') else response

    async def _test_list_tools(self, server):
        """15.1: List tools."""
        from memory_layer.server.mcp import MCPRequest

        start = time.time()
        try:
            request = MCPRequest.from_dict({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
            response = await server.handle_request(request)
            response_dict = response.to_dict() if hasattr(response, 'to_dict') else response
            tools = response_dict.get("result", {}).get("tools", [])
            passed = len(tools) == 8
            self.runner.record("15.1", "List tools (8 tools)", passed,
                             f"Got {len(tools)} tools" if not passed else "",
                             duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.1", "List tools", False, str(e))

    async def _test_add_memory(self, server):
        """15.5: Add memory."""
        start = time.time()
        try:
            response = await self._call_tool(server, "add_memory", {
                "content": "MCP test memory",
                "category": "gotcha"
            })
            result = response.get("result", {})
            content = result.get("content", [{}])[0]
            memory_id = content.get("text", "")
            passed = "id" in memory_id or len(memory_id) > 10
            if passed:
                # Extract ID from response
                import re
                match = re.search(r'id["\s:]+([a-f0-9-]+)', memory_id)
                if match:
                    self.created_ids.append(match.group(1))
            self.runner.record("15.5", "Add memory", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.5", "Add memory", False, str(e))

    async def _test_search_memories(self, server):
        """15.2: Search memories."""
        start = time.time()
        try:
            response = await self._call_tool(server, "search_memories", {
                "query": "MCP test",
                "limit": 5
            })
            passed = "result" in response and "error" not in response
            self.runner.record("15.2", "Search memories", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.2", "Search memories", False, str(e))

    async def _test_record_outcome(self, server):
        """15.7: Record outcome."""
        start = time.time()
        try:
            if not self.created_ids:
                self.runner.record("15.7", "Record worked outcome", False, "No memory created")
                return
            response = await self._call_tool(server, "record_outcome", {
                "memory_ids": self.created_ids,
                "outcome": "worked"
            })
            passed = "result" in response and "error" not in response
            self.runner.record("15.7", "Record worked outcome", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.7", "Record worked outcome", False, str(e))

    async def _test_get_context(self, server):
        """15.10: Get context."""
        start = time.time()
        try:
            response = await self._call_tool(server, "get_context", {
                "limit": 10
            })
            passed = "result" in response and "error" not in response
            self.runner.record("15.10", "Get context", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.10", "Get context", False, str(e))

    async def _test_update_memory(self, server):
        """15.11: Update memory."""
        start = time.time()
        try:
            if not self.created_ids:
                self.runner.record("15.11", "Update memory", False, "No memory created")
                return
            response = await self._call_tool(server, "update_memory", {
                "id": self.created_ids[0],
                "content": "MCP test memory (updated)"
            })
            passed = "result" in response and "error" not in response
            self.runner.record("15.11", "Update memory", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.11", "Update memory", False, str(e))

    async def _test_list_memories(self, server):
        """15.14: List memories."""
        start = time.time()
        try:
            response = await self._call_tool(server, "list_memories", {
                "limit": 10
            })
            passed = "result" in response and "error" not in response
            self.runner.record("15.14", "List memories", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.14", "List memories", False, str(e))

    async def _test_get_stats(self, server):
        """15.16: Get stats."""
        start = time.time()
        try:
            response = await self._call_tool(server, "get_stats", {})
            passed = "result" in response and "error" not in response
            self.runner.record("15.16", "Get stats", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.16", "Get stats", False, str(e))

    async def _test_delete_memory(self, server):
        """15.13: Delete memory."""
        start = time.time()
        try:
            # Create a memory to delete
            response = await self._call_tool(server, "add_memory", {
                "content": "Memory to delete via MCP",
                "category": "general"
            })

            # Extract ID
            result = response.get("result", {})
            content = result.get("content", [{}])[0]
            text = content.get("text", "")

            import re
            match = re.search(r'id["\s:]+([a-f0-9-]+)', text)
            if match:
                delete_id = match.group(1)
                response = await self._call_tool(server, "delete_memory", {"id": delete_id})
                passed = "result" in response and "error" not in response
                self.runner.record("15.13", "Delete memory", passed, duration=time.time() - start)
            else:
                self.runner.record("15.13", "Delete memory", False, "Could not extract ID")
        except Exception as e:
            self.runner.record("15.13", "Delete memory", False, str(e))

    async def _test_error_handling(self, server):
        """15.17-15.19: Error handling."""
        # 15.18: Missing required param
        start = time.time()
        try:
            response = await self._call_tool(server, "add_memory", {})
            passed = "error" in response
            self.runner.record("15.18", "Invalid params error", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.18", "Invalid params error", False, str(e))

        # 15.19: Invalid outcome value
        start = time.time()
        try:
            response = await self._call_tool(server, "record_outcome", {
                "memory_ids": ["test-id"],
                "outcome": "invalid_value"
            })
            passed = "error" in response
            self.runner.record("15.19", "Invalid outcome error", passed, duration=time.time() - start)
        except Exception as e:
            self.runner.record("15.19", "Invalid outcome error", False, str(e))


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Memory Layer UAT Tests")
    parser.add_argument("--api", action="store_true", help="Run REST API tests only")
    parser.add_argument("--sdk", action="store_true", help="Run SDK tests only")
    parser.add_argument("--mcp", action="store_true", help="Run MCP tests only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # If no specific test selected, run all
    run_all = not (args.api or args.sdk or args.mcp)

    runner = TestRunner(verbose=args.verbose)

    print("=" * 60)
    print("Memory Layer - User Acceptance Tests")
    print("=" * 60)

    if run_all or args.api:
        api_tests = APITests(runner)
        await api_tests.run_all()

    if run_all or args.sdk:
        sdk_tests = SDKTests(runner)
        await sdk_tests.run_all()

    if run_all or args.mcp:
        mcp_tests = MCPTests(runner)
        await mcp_tests.run_all()

    passed, failed = runner.summary()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

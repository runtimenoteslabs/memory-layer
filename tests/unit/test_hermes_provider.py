"""Unit tests for the Hermes Agent memory provider.

Tests for:
- The standalone base shim used when Hermes is not installed
- The sync-to-async bridge
- Provider lifecycle, recall, write gating and tool dispatch
- The evaluation trace
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from runtime_memory.core.attribution import Attribution
from runtime_memory.core.embeddings import MockEmbeddingProvider
from runtime_memory.core.models import Outcome, RelationType
from runtime_memory.extraction.extractor import ExtractionResult, MemoryExtractor, ModelUsage
from runtime_memory.hermes import RuntimeMemoryProvider, register
from runtime_memory.hermes._base import RecallStatus, is_trivial_prompt
from runtime_memory.hermes.bridge import DEFAULT_TIMEOUT, run_sync, spawn
from runtime_memory.hermes.provider import PROVIDER_NAME, _embedding_provider_name
from runtime_memory.hermes.trace import TRACE_ENV_VAR, TraceWriter, summarize


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """An initialized provider backed by a throwaway store."""
    monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
    monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
    monkeypatch.delenv(TRACE_ENV_VAR, raising=False)

    instance = RuntimeMemoryProvider()
    instance.initialize("session-1", agent_context="primary")
    yield instance
    instance.shutdown()


def _hide_sentence_transformers(monkeypatch):
    """Make the embedding extra look uninstalled, as a base install would.

    Blocks both routes to it: the import statement and the spec lookup the
    engine factory uses.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "sentence_transformers" else real_find_spec(name, *args, **kwargs)
        ),
    )


def _call(instance, tool, **args):
    """Invoke a tool and parse its JSON response."""
    return json.loads(instance.handle_tool_call(tool, args))


def _line_for(block, memory_id):
    """Return the injected line for one memory id."""
    return next(line for line in block.splitlines() if memory_id in line)


# =============================================================================
# Base Shim Tests
# =============================================================================


class TestBaseShim:
    """Tests for the standalone fallback contract."""

    @pytest.mark.parametrize("text", ["", "   ", "/reset", "thanks!", "ok", "yes", "got it", "hi"])
    def test_trivial_prompts(self, text):
        """Greetings, acknowledgements and slash commands skip recall."""
        assert is_trivial_prompt(text) is True

    @pytest.mark.parametrize(
        "text", ["k8s", "yolo", "note", "how do I run the tests?", "done deal now"]
    )
    def test_substantive_prompts(self, text):
        """Real questions, and words that merely start like an ack, do not."""
        assert is_trivial_prompt(text) is False

    def test_recall_status_defaults(self):
        """RecallStatus carries a label, a count and a glyph."""
        status = RecallStatus(provider_label="Runtime Memory", count=3)

        assert status.count == 3
        assert status.glyph


# =============================================================================
# Bridge Tests
# =============================================================================


class TestBridge:
    """Tests for the sync-to-async bridge."""

    def test_run_sync_returns_result(self):
        """A coroutine's value comes back to the synchronous caller."""

        async def work():
            await asyncio.sleep(0)
            return 42

        assert run_sync(work()) == 42

    def test_run_sync_propagates_error(self):
        """Exceptions surface to the caller rather than being swallowed."""

        async def boom():
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            run_sync(boom())

    def test_run_sync_times_out(self):
        """A hung coroutine raises instead of blocking the agent forever."""

        async def forever():
            await asyncio.sleep(30)

        with pytest.raises(TimeoutError):
            run_sync(forever(), timeout=0.1)

    def test_spawn_swallows_failure(self):
        """Background work logs its failure instead of raising into the turn."""

        async def boom():
            raise RuntimeError("background")

        spawn(boom(), label="test")  # must not raise

    def test_default_timeout_is_bounded(self):
        """The default timeout is finite, so a stall cannot wedge a turn."""
        assert 0 < DEFAULT_TIMEOUT < 120


# =============================================================================
# Provider Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for identity, availability and initialization."""

    def test_name_matches_entry_point(self, provider):
        """The provider answers to the name Hermes activates it by."""
        assert provider.name == PROVIDER_NAME == "runtimememory"

    def test_available_with_writable_store(self, provider):
        """Availability is a filesystem check, with no network involved."""
        assert provider.is_available() is True

    def test_unavailable_when_unwritable(self, tmp_path, monkeypatch):
        """An unwritable location reports unavailable with a usable reason."""
        blocked = tmp_path / "blocked"
        blocked.mkdir(mode=0o500)
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(blocked / "sub" / "memories.db"))

        instance = RuntimeMemoryProvider()

        assert instance.is_available() is False
        assert "RUNTIME_MEMORY_DB" in instance.unavailable_reason()

    def test_workspace_becomes_project(self, tmp_path, monkeypatch):
        """Memories are scoped to the workspace directory name."""
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        monkeypatch.delenv("RUNTIME_MEMORY_PROJECT", raising=False)

        instance = RuntimeMemoryProvider()
        instance.initialize("s", agent_workspace="/home/user/my-project")
        try:
            assert instance._project == "my-project"
        finally:
            instance.shutdown()

    def test_backup_paths_names_the_store(self, provider):
        """The store sits outside HERMES_HOME, so backup must be told about it."""
        paths = provider.backup_paths()

        assert len(paths) == 1
        assert paths[0].endswith("memories.db")

    def test_shutdown_is_idempotent(self, provider):
        """A second shutdown is harmless."""
        provider.shutdown()
        provider.shutdown()

    def test_session_switch_rebinds(self, provider):
        """A new session id is adopted, and a reset clears recall state."""
        provider._last_ids = ["a"]

        provider.on_session_switch("session-2", reset=True)

        assert provider._session_id == "session-2"
        assert provider._last_ids == []


# =============================================================================
# Embedding Backend Tests
# =============================================================================


class TestDefaultStore:
    """The setup wizard and the provider must not name the pre-3.0 directory.

    `hermes memory setup` offers this default and writes the answer to Hermes'
    .env, so a stale path there is not a cosmetic slip: it points a fresh
    install at a store that does not exist.
    """

    def test_config_schema_offers_the_current_store(self, provider):
        schema = {field["key"]: field for field in provider.get_config_schema()}

        assert schema["db_path"]["default"].endswith("/.runtime-memory/memories.db")
        assert ".memory-layer" not in schema["db_path"]["default"]
        assert schema["db_path"]["env_var"] == "RUNTIME_MEMORY_DB"

    def test_provider_defaults_to_the_current_store(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RUNTIME_MEMORY_DB", raising=False)
        monkeypatch.delenv("MEMORY_LAYER_DB", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        instance = RuntimeMemoryProvider()

        assert instance._db_path == tmp_path / ".runtime-memory" / "memories.db"


class TestEmbeddingBackend:
    """The store is shared, so a missing model must not turn into fake vectors."""

    def test_defaults_to_local_even_without_the_model(self, monkeypatch):
        """The engine factory decides what `local` resolves to, not the provider."""
        monkeypatch.delenv("RUNTIME_MEMORY_EMBEDDING", raising=False)
        _hide_sentence_transformers(monkeypatch)

        assert _embedding_provider_name() == "local"

    def test_override_is_passed_through(self, monkeypatch):
        """An explicit choice wins."""
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "voyage")

        assert _embedding_provider_name() == "voyage"

    def test_missing_model_writes_no_vector(self, tmp_path, monkeypatch):
        """Recall still works, and nothing meaningless lands in the index."""
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
        monkeypatch.delenv("RUNTIME_MEMORY_EMBEDDING", raising=False)
        _hide_sentence_transformers(monkeypatch)

        instance = RuntimeMemoryProvider()
        instance.initialize("session-null", agent_context="primary")
        try:
            assert instance._engine.embedding_provider.available is False

            _call(
                instance,
                "runtimememory_remember",
                content="Run pytest from the repo root",
                category="gotcha",
            )
            block = instance.prefetch("how do I run the tests?")

            assert "Run pytest from the repo root" in block
            assert instance._engine.retriever.indexed_with_embeddings == 0
        finally:
            instance.shutdown()

    def test_missing_model_is_said_out_loud(self, tmp_path, monkeypatch):
        """Keyword-only recall ranks differently, so falling into it is a warning.

        The warning names this interpreter, because the extra usually goes
        missing by being installed into a different environment from Hermes'.
        """
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
        monkeypatch.delenv("RUNTIME_MEMORY_EMBEDDING", raising=False)
        _hide_sentence_transformers(monkeypatch)

        instance = RuntimeMemoryProvider()
        with patch("runtime_memory.hermes.provider.logger") as mock_logger:
            instance.initialize("session-keyword", agent_context="primary")
        try:
            assert instance._engine.search_mode == "keyword"
            warnings = " ".join(str(call) for call in mock_logger.warning.call_args_list)
            assert "keyword matching only" in warnings
            assert sys.executable in warnings
            ready = " ".join(str(call) for call in mock_logger.info.call_args_list)
            assert "search=keyword" in ready
        finally:
            instance.shutdown()

    def test_choosing_keyword_matching_is_quiet(self, tmp_path, monkeypatch):
        """`null` says keyword matching is intended, so there is nothing to warn about."""
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "null")

        instance = RuntimeMemoryProvider()
        with patch("runtime_memory.hermes.provider.logger") as mock_logger:
            instance.initialize("session-null-chosen", agent_context="primary")
        try:
            assert instance._engine.search_mode == "keyword"
            warnings = " ".join(str(call) for call in mock_logger.warning.call_args_list)
            assert "keyword matching only" not in warnings
        finally:
            instance.shutdown()

    def test_vectors_make_it_hybrid(self, provider):
        """A backend that produces vectors reports hybrid search."""
        assert provider._engine.search_mode == "hybrid"


# =============================================================================
# Recall Tests
# =============================================================================


class TestRecall:
    """Tests for the per-turn retrieval path."""

    def test_prefetch_injects_stored_memories(self, provider):
        """A stored memory is retrieved and rendered for the prompt."""
        _call(
            provider,
            "runtimememory_remember",
            content="Run pytest from the repo root",
            category="gotcha",
        )

        block = provider.prefetch("how do I run the tests?")

        assert "Run pytest from the repo root" in block
        assert "[gotcha]" in block

    def test_prefetch_includes_memory_ids(self, provider):
        """Ids are injected so the model can name memories in an outcome."""
        result = _call(
            provider,
            "runtimememory_remember",
            content="Use snake_case here",
            category="convention",
        )

        block = provider.prefetch("naming style?")

        assert result["memory_id"] in block

    def test_prefetch_flags_proven_memories(self, provider):
        """One success is shown, as a count."""
        stored = _call(
            provider, "runtimememory_remember", content="Clear the cache", category="troubleshooting"
        )
        run_sync(provider._engine.record_outcome([stored["memory_id"]], Outcome.WORKED))

        line = _line_for(provider.prefetch("cache problems"), stored["memory_id"])

        assert "(worked 1 time)" in line

    def test_prefetch_flags_discredited_memories(self, provider):
        """One failure is shown, as a count."""
        stored = _call(
            provider, "runtimememory_remember", content="Delete the lockfile", category="workaround"
        )
        run_sync(provider._engine.record_outcome([stored["memory_id"]], Outcome.FAILED))

        line = _line_for(provider.prefetch("lockfile trouble"), stored["memory_id"])

        assert "(failed 1 time)" in line

    def test_mixed_record_shows_both_counts(self, provider):
        """A memory that both worked and failed says so, rather than picking a side."""
        stored = _call(
            provider, "runtimememory_remember", content="Retry the request", category="pattern"
        )
        memory_id = stored["memory_id"]
        run_sync(provider._engine.record_outcome([memory_id], Outcome.WORKED))
        run_sync(provider._engine.record_outcome([memory_id], Outcome.FAILED))

        line = _line_for(provider.prefetch("request keeps failing"), memory_id)

        assert "(worked 1 time, failed 1 time)" in line

    def test_trivial_prompt_skips_recall(self, provider):
        """An acknowledgement does not trigger retrieval."""
        _call(provider, "runtimememory_remember", content="Something", category="general")

        assert provider.prefetch("thanks!") == ""

    def test_empty_store_injects_nothing(self, provider):
        """No memories means no block, not an empty heading."""
        assert provider.prefetch("anything at all?") == ""

    def test_empty_recall_is_still_traced(self, tmp_path, monkeypatch):
        """A recall that found nothing must be distinguishable from no recall."""
        trace_path = tmp_path / "trace.jsonl"
        monkeypatch.setenv(TRACE_ENV_VAR, str(trace_path))
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "empty.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")

        instance = RuntimeMemoryProvider()
        instance.initialize("session-empty", agent_context="primary")
        try:
            block = instance.prefetch("how do I run the tests in this project?")
        finally:
            instance.shutdown()

        assert block == ""
        records = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
        recalls = [r for r in records if r["event"] == "recall"]
        assert len(recalls) == 1
        assert recalls[0]["retrieved"] == []

    def test_recall_searches_by_keyword_while_the_model_loads(self, tmp_path, monkeypatch):
        """A recall does not wait for the model: Hermes stops waiting after 8 seconds."""
        trace_path = tmp_path / "trace.jsonl"
        monkeypatch.setenv(TRACE_ENV_VAR, str(trace_path))
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "loading.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")

        instance = RuntimeMemoryProvider()
        instance.initialize("session-loading", agent_context="primary")
        try:
            _call(
                instance,
                "runtimememory_remember",
                content="Run the ledger tests with .venv/bin/python -m pytest",
                category="command",
            )
            monkeypatch.setattr(MockEmbeddingProvider, "loaded", property(lambda _self: False))

            async def refuse(_text):
                raise AssertionError("recall waited for the model")

            monkeypatch.setattr(instance._engine.embedding_provider, "embed", refuse)
            block = instance.prefetch("how do I run the ledger tests?")
        finally:
            instance.shutdown()

        assert "pytest" in block
        recall = next(
            json.loads(line)
            for line in trace_path.read_text().splitlines()
            if json.loads(line)["event"] == "recall"
        )
        assert recall["search_mode"] == "keyword"
        assert recall["model_loading"] is True

    def test_a_loaded_model_is_used(self, tmp_path, monkeypatch):
        trace_path = tmp_path / "trace.jsonl"
        monkeypatch.setenv(TRACE_ENV_VAR, str(trace_path))
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "loaded.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")

        instance = RuntimeMemoryProvider()
        instance.initialize("session-loaded", agent_context="primary")
        try:
            _call(instance, "runtimememory_remember", content="A useful fact", category="general")
            instance.prefetch("tell me the fact")
        finally:
            instance.shutdown()

        recall = next(
            json.loads(line)
            for line in trace_path.read_text().splitlines()
            if json.loads(line)["event"] == "recall"
        )
        assert (recall["search_mode"], recall["model_loading"]) == ("hybrid", False)

    def test_the_model_loads_on_its_own_thread(self, provider, monkeypatch):
        called = threading.Event()
        loaded_on = []

        def load():
            loaded_on.append(threading.current_thread().name)
            called.set()

        monkeypatch.setattr(MockEmbeddingProvider, "loaded", property(lambda _self: False))
        monkeypatch.setattr(provider._engine.embedding_provider, "load", load)

        provider._start_model_load()

        assert called.wait(timeout=5)
        assert loaded_on == ["runtime-memory-model"]

    def test_recall_survives_engine_failure(self, provider, monkeypatch):
        """A broken search costs the recall, never the turn."""

        def explode(*args, **kwargs):
            raise RuntimeError("storage is down")

        monkeypatch.setattr(provider._engine, "search", explode)

        assert provider.prefetch("a real question") == ""

    def test_recall_status_reflects_last_prefetch(self, provider):
        """The indicator counts this turn's recall, never a stale one."""
        assert provider.recall_status() is None

        _call(provider, "runtimememory_remember", content="A fact", category="general")
        provider.prefetch("fact?")
        assert provider.recall_status().count == 1

        provider.prefetch("thanks!")
        assert provider.recall_status() is None

    def test_queue_prefetch_is_inert(self, provider):
        """Recall is synchronous, so there is nothing to warm."""
        assert provider.queue_prefetch("anything") is None


# =============================================================================
# Write Path Tests
# =============================================================================


class TestWrites:
    """Tests for how memories get into the store."""

    def test_sync_turn_persists_nothing(self, provider):
        """Raw turns are not stored; that would bury the curated memories."""
        provider.sync_turn("a user message", "an assistant reply")

        assert provider.prefetch("user message") == ""

    def test_builtin_writes_are_mirrored(self, provider):
        """A built-in memory write also lands in the shared store."""
        provider.on_memory_write("add", "user", "Prefers tabs over spaces")
        run_sync(asyncio.sleep(0.5))

        found = _call(provider, "runtimememory_recall", query="tabs")

        assert found["count"] == 1
        assert found["memories"][0]["category"] == "preference"

    def test_mirroring_can_be_disabled(self, tmp_path, monkeypatch):
        """Mirroring is a setting, not a fixed behavior."""
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        monkeypatch.setenv("RUNTIME_MEMORY_MIRROR_WRITES", "false")

        instance = RuntimeMemoryProvider()
        instance.initialize("s")
        try:
            instance.on_memory_write("add", "user", "Should not be stored")
            run_sync(asyncio.sleep(0.3))

            assert _call(instance, "runtimememory_recall", query="stored")["count"] == 0
        finally:
            instance.shutdown()

    def test_removals_are_not_mirrored(self, provider):
        """Only additions and replacements mirror; a removal is not content."""
        provider.on_memory_write("remove", "user", "Some old fact")
        run_sync(asyncio.sleep(0.3))

        assert _call(provider, "runtimememory_recall", query="old fact")["count"] == 0

    def test_subagents_do_not_write(self, tmp_path, monkeypatch):
        """Non-primary contexts read the store but never add to it."""
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")

        instance = RuntimeMemoryProvider()
        instance.initialize("s", agent_context="subagent")
        try:
            assert instance._writes_allowed is False

            result = _call(instance, "runtimememory_remember", content="X", category="general")
            assert "error" in result

            instance.on_memory_write("add", "user", "Also blocked")
            run_sync(asyncio.sleep(0.3))
            assert _call(instance, "runtimememory_recall", query="blocked")["count"] == 0
        finally:
            instance.shutdown()

    def test_extraction_is_refused_without_the_package(self, tmp_path, monkeypatch):
        """A missing dependency is said out loud, not discovered a session later."""
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        monkeypatch.setenv("RUNTIME_MEMORY_EXTRACT_ON_END", "true")
        monkeypatch.setitem(sys.modules, "anthropic", None)

        # Asserted on the logger rather than caplog: setup_logging turns off
        # propagation for this package, so caplog is empty whenever another test
        # has configured logging first.
        instance = RuntimeMemoryProvider()
        with patch("runtime_memory.hermes.provider.logger") as mock_logger:
            instance.initialize("session-no-package", agent_context="primary")
        try:
            assert instance._extract_on_end is False
            warnings = " ".join(str(call) for call in mock_logger.warning.call_args_list)
            assert "anthropic package is not installed" in warnings
        finally:
            instance.shutdown()

    def test_extraction_is_refused_without_credentials(self, tmp_path, monkeypatch):
        """The wizard offers this option whether or not a key was ever entered.

        Exercises the real check: the SDK constructs a client happily with no
        credentials at all and only fails at request time, so "did it construct"
        is not a usable test.
        """
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        monkeypatch.setenv("RUNTIME_MEMORY_EXTRACT_ON_END", "true")
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_FEDERATION_RULE_ID"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        instance = RuntimeMemoryProvider()
        with patch("runtime_memory.hermes.provider.logger") as mock_logger:
            instance.initialize("session-no-key", agent_context="primary")
        try:
            assert instance._extract_on_end is False
            warnings = " ".join(str(call) for call in mock_logger.warning.call_args_list)
            assert "no Anthropic credentials" in warnings
        finally:
            instance.shutdown()

    def test_a_key_in_the_environment_is_enough(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        monkeypatch.setenv("RUNTIME_MEMORY_EXTRACT_ON_END", "true")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

        instance = RuntimeMemoryProvider()
        instance.initialize("session-key", agent_context="primary")
        try:
            assert instance._extract_on_end is True
        finally:
            instance.shutdown()

    def test_extraction_stays_on_when_usable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        monkeypatch.setenv("RUNTIME_MEMORY_EXTRACT_ON_END", "true")

        with patch(
            "runtime_memory.hermes.provider._extraction_unavailable_reason",
            return_value=None,
        ):
            instance = RuntimeMemoryProvider()
            instance.initialize("session-ok", agent_context="primary")
            try:
                assert instance._extract_on_end is True
            finally:
                instance.shutdown()

    def test_extraction_runs_when_enabled(self, tmp_path, monkeypatch):
        """Session end extracts, and the trace records how many memories landed."""
        trace_path = tmp_path / "trace.jsonl"
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "extract.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        monkeypatch.setenv("RUNTIME_MEMORY_EXTRACT_ON_END", "true")
        monkeypatch.setenv(TRACE_ENV_VAR, str(trace_path))
        # The startup check refuses extraction without a credential, and the
        # extractor itself is mocked, so nothing reaches the network.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

        extracted = MagicMock()
        extracted.success = True
        extracted.error = None
        extracted.memory_count = 2

        extractor = MagicMock()
        extractor.return_value.extract_and_store = AsyncMock(return_value=extracted)

        instance = RuntimeMemoryProvider()
        instance.initialize("session-extract", agent_context="primary")
        try:
            with patch(
                "runtime_memory.extraction.extractor.MemoryExtractor", extractor
            ):
                instance.on_session_end(
                    [
                        {"role": "user", "content": "money must be Decimal here"},
                        {"role": "assistant", "content": "noted"},
                    ]
                )
                # No polling: the hook blocks until extraction finishes. Hermes
                # tears the provider down the moment it returns, so anything
                # still in flight at that point is lost.
                assert trace_path.exists(), "on_session_end returned before extraction finished"
        finally:
            instance.shutdown()

        transcript = extractor.return_value.extract_and_store.call_args.kwargs["transcript"]
        assert "money must be Decimal here" in transcript

        records = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
        writes = [r for r in records if r["event"] == "write"]
        assert writes and writes[0]["kind"] == "extraction"
        assert writes[0]["count"] == 2

    def test_extraction_is_off_by_default(self, provider):
        """Session end costs nothing unless extraction is switched on."""
        provider.on_session_end([{"role": "user", "content": "hello"}])

        assert provider._extract_on_end is False


# =============================================================================
# Tool Dispatch Tests
# =============================================================================


class TestTools:
    """Tests for the function-calling surface."""

    def test_schemas_are_openai_shaped(self, provider):
        """Hermes wants name/description/parameters, not MCP inputSchema."""
        for schema in provider.get_tool_schemas():
            assert set(schema) == {"name", "description", "parameters"}
            assert schema["parameters"]["type"] == "object"

    def test_unknown_tool_returns_error(self, provider):
        """An unrecognized tool is reported, not raised."""
        assert "error" in _call(provider, "runtimememory_nonexistent")

    def test_bad_category_lists_valid_ones(self, provider):
        """A wrong category tells the model what it may use instead."""
        result = _call(provider, "runtimememory_remember", content="X", category="nonsense")

        assert "nonsense" in result["error"]
        assert "convention" in result["error"]

    def test_missing_content_is_rejected(self, provider):
        """Empty content is refused rather than stored."""
        assert "error" in _call(provider, "runtimememory_remember", content="   ", category="general")

    def test_bad_outcome_is_rejected(self, provider):
        """Only the three defined outcomes are accepted."""
        result = _call(provider, "runtimememory_outcome", outcome="great")

        assert "great" in result["error"]

    def test_stats_reports_the_store(self, provider):
        """Stats summarize what is held, broken down by category."""
        _call(provider, "runtimememory_remember", content="A gotcha", category="gotcha")

        stats = _call(provider, "runtimememory_stats")

        assert stats["total_memories"] == 1
        assert stats["by_category"]["gotcha"] == 1

    def test_explicit_recall_joins_the_turn(self, provider):
        """A tool search returns ids a later outcome can name."""
        _call(provider, "runtimememory_remember", content="A fact", category="general")

        recalled = _call(provider, "runtimememory_recall", query="fact")
        result = _call(
            provider,
            "runtimememory_outcome",
            outcome="worked",
            memory_ids=[m["id"] for m in recalled["memories"]],
        )

        assert result["recorded"] is True


# =============================================================================
# Outcome Tests
# =============================================================================


class TestOutcomes:
    """Tests for the feedback loop, which is the point of the integration."""

    def test_outcome_names_the_memories_it_is_about(self, provider):
        """An outcome reaches the memories named, and scores them."""
        stored = _call(
            provider,
            "runtimememory_remember",
            content="Restart the daemon",
            category="troubleshooting",
        )
        provider.prefetch("daemon is stuck")

        result = _call(
            provider, "runtimememory_outcome", outcome="worked", memory_ids=[stored["memory_id"]]
        )

        assert result["recorded"] is True
        assert result["updated"][0]["worked"] == 1.0
        assert result["updated"][0]["outcome_score"] == pytest.approx(1 / 3, abs=1e-3)

    def test_outcome_without_ids_is_declined(self, provider):
        """Before 4.0.0 this scored every memory recalled in the turn.

        The Tier 2 evaluation measured that contract three times, and each time
        it left the store worse than recording nothing: the memory that misled
        the turn and the memory that was right about the same thing sank together.
        """
        stored = _call(
            provider,
            "runtimememory_remember",
            content="Restart the daemon",
            category="troubleshooting",
        )
        provider.prefetch("daemon is stuck")

        result = _call(provider, "runtimememory_outcome", outcome="worked")

        assert result["recorded"] is False
        assert "memory_ids" in result["reason"]
        after = _call(provider, "runtimememory_recall", query="daemon is stuck")
        assert after["memories"][0]["id"] == stored["memory_id"]
        assert after["memories"][0]["outcome_score"] == 0.0

    def test_outcome_with_an_unknown_id_is_an_error(self, provider):
        result = _call(
            provider, "runtimememory_outcome", outcome="worked", memory_ids=["not-an-id"]
        )

        assert "not-an-id" in result["error"]

    def test_failure_costs_more_than_success_gains(self, provider):
        """The asymmetry that makes bad advice sink is preserved end to end."""
        stored = _call(
            provider, "runtimememory_remember", content="Try turning it off", category="workaround"
        )
        memory_id = stored["memory_id"]

        worked = _call(provider, "runtimememory_outcome", outcome="worked", memory_ids=[memory_id])
        failed = _call(provider, "runtimememory_outcome", outcome="failed", memory_ids=[memory_id])

        # One of each nets negative: (1 - 1.5) / (1 + 1.5 + 2).
        assert worked["updated"][0]["outcome_score"] == pytest.approx(1 / 3, abs=1e-3)
        assert failed["updated"][0]["outcome_score"] == pytest.approx(-0.5 / 4.5, abs=1e-3)

    def test_outcome_without_recall_is_a_no_op(self, provider):
        """With nothing recalled there is nothing to credit or blame."""
        result = _call(provider, "runtimememory_outcome", outcome="worked")

        assert result["recorded"] is False


# =============================================================================
# Trace Tests
# =============================================================================


class TestTrace:
    """Tests for the evaluation trace."""

    def test_disabled_without_a_path(self, monkeypatch):
        """Tracing costs nothing when it is not configured."""
        monkeypatch.delenv(TRACE_ENV_VAR, raising=False)

        assert TraceWriter().enabled is False

    def test_records_recall_and_outcome(self, tmp_path, monkeypatch):
        """A turn's retrieval and its outcome share a turn_id, so they join."""
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        trace_path = tmp_path / "traces" / "run.jsonl"
        monkeypatch.setenv(TRACE_ENV_VAR, str(trace_path))

        instance = RuntimeMemoryProvider()
        instance.initialize("session-9")
        try:
            stored = _call(
                instance, "runtimememory_remember", content="A useful fact", category="general"
            )
            instance.prefetch("tell me the fact")
            _call(
                instance,
                "runtimememory_outcome",
                outcome="worked",
                memory_ids=[stored["memory_id"]],
            )
        finally:
            instance.shutdown()

        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        by_event = {event["event"]: event for event in events}

        assert by_event["recall"]["retrieved"][0]["memory_id"]
        assert by_event["recall"]["search_mode"] == "hybrid"
        assert by_event["recall"]["turn_id"] == by_event["outcome"]["turn_id"]
        assert by_event["outcome"]["outcome"] == "worked"
        assert by_event["outcome"]["origin"] == "tool"

    def test_records_an_outcome_it_declined(self, tmp_path, monkeypatch):
        """A declined outcome is visible, so a silent loop can be told from a quiet one."""
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "mock")
        trace_path = tmp_path / "traces" / "declined.jsonl"
        monkeypatch.setenv(TRACE_ENV_VAR, str(trace_path))

        instance = RuntimeMemoryProvider()
        instance.initialize("session-10")
        try:
            _call(instance, "runtimememory_remember", content="A useful fact", category="general")
            instance.prefetch("tell me the fact")
            _call(instance, "runtimememory_outcome", outcome="worked")
        finally:
            instance.shutdown()

        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        declined = next(e for e in events if e["event"] == "outcome")
        assert declined["origin"] == "declined"
        assert declined["memory_ids"] == []
        assert all("ts" in event for event in events)

    def test_broken_path_does_not_break_a_turn(self, tmp_path):
        """An unwritable trace degrades to no tracing, not to an error."""
        blocked = tmp_path / "ro"
        blocked.mkdir(mode=0o500)

        writer = TraceWriter(blocked / "sub" / "trace.jsonl")

        assert writer.enabled is False
        writer.outcome(turn_id="t", session_id="s", outcome="worked", memory_ids=[], origin="tool")

    def test_write_count_defaults_to_ids(self, tmp_path):
        """Callers that know the ids need not also count them."""
        path = tmp_path / "t.jsonl"
        writer = TraceWriter(path)

        writer.write_turn(turn_id="t", session_id="s", memory_ids=["a", "b"], kind="tool")

        assert json.loads(path.read_text())["count"] == 2

    def test_the_default_path_is_used_when_nothing_is_configured(self, tmp_path, monkeypatch):
        monkeypatch.delenv(TRACE_ENV_VAR, raising=False)

        writer = TraceWriter(default=tmp_path / "hermes-trace.jsonl")

        assert writer.path == tmp_path / "hermes-trace.jsonl"

    @pytest.mark.parametrize("value", ["off", "OFF", "none", "0"])
    def test_off_turns_tracing_off(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv(TRACE_ENV_VAR, value)

        assert TraceWriter(default=tmp_path / "hermes-trace.jsonl").enabled is False

    def test_the_provider_traces_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "null")
        monkeypatch.delenv(TRACE_ENV_VAR, raising=False)
        instance = RuntimeMemoryProvider()
        instance.initialize("session-default-trace", agent_context="primary")
        try:
            instance.prefetch("how are ledger amounts stored?")
        finally:
            instance.shutdown()

        assert (tmp_path / "hermes-trace.jsonl").exists()

    def test_summarize_counts_what_a_trace_holds(self, tmp_path):
        path = tmp_path / "t.jsonl"
        writer = TraceWriter(path)
        writer.usage(turn_id="t", session_id="s", kind="extraction", model="m",
                     calls=1, input_tokens=10, output_tokens=5)
        writer.outcome(turn_id="t", session_id="s", outcome="worked", memory_ids=["a"], origin="cited")
        with path.open("a") as handle:
            handle.write("not json\n")

        summary = summarize(path)

        assert summary["extraction_usage"] == {"m": {"calls": 1, "input_tokens": 10, "output_tokens": 5}}
        assert summary["outcomes_by_origin"] == {"cited": {"events": 1, "memories": 1}}
        assert summary["skipped_lines"] == 1

    def test_confirmations_are_not_writes(self, tmp_path):
        """A memory learned again is its own event, so it never counts as a write."""
        path = tmp_path / "t.jsonl"
        writer = TraceWriter(path)

        writer.confirm(turn_id="t", session_id="s", memory_ids=["a"])

        record = json.loads(path.read_text())
        assert record["event"] == "confirm"
        assert record["memory_ids"] == ["a"]


# =============================================================================
# Conflict Tests
# =============================================================================


@pytest.fixture
def keyword_provider(tmp_path, monkeypatch):
    """A provider on keyword matching alone, so what is recalled is exact.

    A memory that shares no word with the query has no relevance and is never
    recalled, which lets a test keep one side of a contradiction out of recall.
    """
    monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
    monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "null")
    monkeypatch.delenv(TRACE_ENV_VAR, raising=False)

    instance = RuntimeMemoryProvider()
    instance.initialize("session-conflicts", agent_context="primary")
    yield instance
    instance.shutdown()


def _remember(instance, content):
    return _call(instance, "runtimememory_remember", content=content, category="convention")["memory_id"]


def _contradict(instance, first, second):
    run_sync(instance._engine.link(first, second, RelationType.CONFLICTS_WITH))


class TestConflicts:
    """Memories linked as contradicting each other are shown as such."""

    def test_two_recalled_memories_are_both_marked(self, keyword_provider):
        decimal = _remember(keyword_provider, "Ledger amounts are Decimal values")
        floats = _remember(keyword_provider, "Ledger amounts are float values rounded to cents")
        _contradict(keyword_provider, floats, decimal)

        block = keyword_provider.prefetch("how are ledger amounts stored?")

        # Found by content: each line also names the other memory's id.
        assert _line_for(block, "Decimal values").endswith(f"`{decimal}` (contradicts `{floats}`)")
        assert _line_for(block, "float values").endswith(f"`{floats}` (contradicts `{decimal}`)")
        assert "disagree" in block

    def test_the_other_side_is_shown_when_not_recalled(self, keyword_provider):
        """A contested note does not arrive alone."""
        floats = _remember(keyword_provider, "Ledger amounts are float values")
        decimal = _remember(keyword_provider, "Money must be Decimal, never binary")
        _contradict(keyword_provider, floats, decimal)

        block = keyword_provider.prefetch("how are ledger amounts stored?")

        heading, counterpart = block.split("Also stored, and contradicting a memory above:")
        assert floats in heading
        assert f"`{decimal}` (contradicts `{floats}`)" in counterpart
        assert keyword_provider._last_ids == [floats, decimal]

    def test_counterparts_can_be_turned_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "null")
        monkeypatch.setenv("RUNTIME_MEMORY_CONFLICT_COUNTERPARTS", "0")
        instance = RuntimeMemoryProvider()
        instance.initialize("session-no-counterparts", agent_context="primary")
        try:
            floats = _remember(instance, "Ledger amounts are float values")
            decimal = _remember(instance, "Money must be Decimal, never binary")
            _contradict(instance, floats, decimal)

            block = instance.prefetch("how are ledger amounts stored?")

            assert decimal not in block
            assert "contradicts" not in block
        finally:
            instance.shutdown()

    def test_an_archived_counterpart_is_not_shown(self, keyword_provider):
        floats = _remember(keyword_provider, "Ledger amounts are float values")
        decimal = _remember(keyword_provider, "Money must be Decimal, never binary")
        _contradict(keyword_provider, floats, decimal)
        run_sync(keyword_provider._engine.archive(decimal))

        block = keyword_provider.prefetch("how are ledger amounts stored?")

        assert decimal not in block
        assert "contradicts" not in block

    def test_without_conflicts_the_block_is_unchanged(self, keyword_provider):
        memory = _remember(keyword_provider, "Ledger amounts are Decimal values")

        block = keyword_provider.prefetch("how are ledger amounts stored?")

        assert block == (
            f"## Relevant memories\n\n- [convention] Ledger amounts are Decimal values `{memory}`"
            "\n\nWhen you act on one of these memories, name its id in your reply."
        )

    def test_a_failed_lookup_keeps_the_recall(self, keyword_provider, monkeypatch):
        memory = _remember(keyword_provider, "Ledger amounts are Decimal values")

        async def broken(*_args, **_kwargs):
            raise RuntimeError("relations table unreadable")

        monkeypatch.setattr(keyword_provider._engine, "related", broken)
        block = keyword_provider.prefetch("how are ledger amounts stored?")

        assert memory in block

    def test_the_trace_records_what_was_added_and_why(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "null")
        trace_path = tmp_path / "trace.jsonl"
        monkeypatch.setenv(TRACE_ENV_VAR, str(trace_path))
        instance = RuntimeMemoryProvider()
        instance.initialize("session-traced", agent_context="primary")
        try:
            floats = _remember(instance, "Ledger amounts are float values")
            decimal = _remember(instance, "Money must be Decimal, never binary")
            _contradict(instance, floats, decimal)
            instance.prefetch("how are ledger amounts stored?")
        finally:
            instance.shutdown()

        recall = next(
            json.loads(line) for line in trace_path.read_text().splitlines()
            if json.loads(line)["event"] == "recall"
        )
        assert recall["counterparts"] == [decimal]
        assert recall["contradicts"] == {floats: [decimal], decimal: [floats]}

    def test_the_recall_tool_names_contradictions(self, keyword_provider):
        floats = _remember(keyword_provider, "Ledger amounts are float values")
        decimal = _remember(keyword_provider, "Money must be Decimal, never binary")
        _contradict(keyword_provider, floats, decimal)

        found = _call(keyword_provider, "runtimememory_recall", query="ledger amounts")

        (memory,) = found["memories"]
        assert memory["contradicts"] == [decimal]


# =============================================================================
# Session Outcome Tests
# =============================================================================


@pytest.fixture
def session_provider(tmp_path, monkeypatch):
    """A keyword-only provider with a trace, for outcomes recorded at session end."""
    monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
    monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "null")
    monkeypatch.setenv(TRACE_ENV_VAR, str(tmp_path / "trace.jsonl"))
    for name in ("RUNTIME_MEMORY_SESSION_OUTCOMES", "RUNTIME_MEMORY_ASK_CITATIONS"):
        monkeypatch.delenv(name, raising=False)

    instance = RuntimeMemoryProvider()
    instance.initialize("session-outcomes", agent_context="primary")
    yield instance
    instance.shutdown()


def _session(cited_id, test_output="5 passed in 0.12s"):
    """A session whose agent names one memory and whose last test run printed this."""
    return [
        {"role": "user", "content": "how are ledger amounts stored?"},
        {"role": "assistant", "content": f"Following `{cited_id}`, amounts stay Decimal."},
        {"role": "tool", "content": json.dumps({"output": test_output, "exit_code": 0})},
    ]


def _record(instance, memory_id):
    memory = run_sync(instance._engine.get(memory_id))
    return memory.worked, memory.failed


class TestSessionOutcomes:
    """The session's verdict reaches the memories it acted on, and only those."""

    def _two_recalled(self, instance):
        decimal = _remember(instance, "Ledger amounts are Decimal values")
        floats = _remember(instance, "Ledger amounts are float values")
        instance.prefetch("how are ledger amounts stored?")
        return decimal, floats

    def test_the_block_asks_for_citations(self, session_provider):
        _remember(session_provider, "Ledger amounts are Decimal values")

        block = session_provider.prefetch("how are ledger amounts stored?")

        assert "name its id" in block

    def test_the_request_can_be_turned_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNTIME_MEMORY_DB", str(tmp_path / "memories.db"))
        monkeypatch.setenv("RUNTIME_MEMORY_EMBEDDING", "null")
        monkeypatch.setenv("RUNTIME_MEMORY_ASK_CITATIONS", "false")
        instance = RuntimeMemoryProvider()
        instance.initialize("session-quiet", agent_context="primary")
        try:
            _remember(instance, "Ledger amounts are Decimal values")
            assert "name its id" not in instance.prefetch("how are ledger amounts stored?")
        finally:
            instance.shutdown()

    def test_a_cited_memory_takes_the_verdict_and_the_rest_do_not(self, session_provider):
        decimal, floats = self._two_recalled(session_provider)

        session_provider.on_session_end(_session(decimal))

        assert _record(session_provider, decimal) == (1.0, 0.0)
        assert _record(session_provider, floats) == (0.0, 0.0)

    def test_a_failing_last_run_records_a_failure(self, session_provider):
        decimal, _ = self._two_recalled(session_provider)

        session_provider.on_session_end(_session(decimal, "1 failed, 4 passed in 0.3s"))

        assert _record(session_provider, decimal) == (0.0, 1.0)

    def test_no_test_run_records_nothing(self, session_provider):
        decimal, _ = self._two_recalled(session_provider)

        session_provider.on_session_end(_session(decimal, "wrote ledger/entries.py"))

        assert _record(session_provider, decimal) == (0.0, 0.0)

    def test_the_trace_names_where_the_attribution_came_from(self, session_provider, tmp_path):
        decimal, _ = self._two_recalled(session_provider)

        session_provider.on_session_end(_session(decimal))

        events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        (outcome,) = [e for e in events if e["event"] == "outcome"]
        assert (outcome["origin"], outcome["memory_ids"], outcome["outcome"]) == (
            "cited", [decimal], "worked",
        )

    def test_a_memory_the_agent_already_scored_is_not_scored_again(self, session_provider):
        decimal, _ = self._two_recalled(session_provider)
        _call(session_provider, "runtimememory_outcome", outcome="worked", memory_ids=[decimal])

        session_provider.on_session_end(_session(decimal))

        assert _record(session_provider, decimal) == (1.0, 0.0)

    def test_extraction_attributions_are_recorded_too(self, session_provider):
        decimal, floats = self._two_recalled(session_provider)

        async def extract(_messages):
            return [Attribution(floats, "extraction", "used float amounts")]

        session_provider._extract_on_end = True
        session_provider._extract = extract
        session_provider.on_session_end(_session(decimal, "1 failed in 0.1s"))

        assert _record(session_provider, floats) == (0.0, 1.0)
        assert _record(session_provider, decimal) == (0.0, 1.0)

    def test_turned_off_records_nothing(self, session_provider):
        decimal, _ = self._two_recalled(session_provider)
        session_provider._session_outcomes = False

        session_provider.on_session_end(_session(decimal))

        assert _record(session_provider, decimal) == (0.0, 0.0)

    def test_a_subagent_records_nothing(self, session_provider):
        decimal, _ = self._two_recalled(session_provider)
        session_provider._writes_allowed = False

        session_provider.on_session_end(_session(decimal))

        assert _record(session_provider, decimal) == (0.0, 0.0)

    def test_the_recall_trace_records_the_block_size(self, session_provider, tmp_path):
        _remember(session_provider, "Ledger amounts are Decimal values")

        block = session_provider.prefetch("how are ledger amounts stored?")

        events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        (recall,) = [e for e in events if e["event"] == "recall"]
        assert recall["block_chars"] == len(block) > 0

    def test_extraction_usage_is_traced(self, session_provider, tmp_path, monkeypatch):
        async def extract_and_store(self, **kwargs):
            return ExtractionResult(
                memories=[], summary="", transcript_length=10, extraction_time_ms=1.0,
                error="bad JSON", usage=ModelUsage("claude-sonnet-5", 1, 2400, 900),
            )

        monkeypatch.setattr(MemoryExtractor, "extract_and_store", extract_and_store)
        session_provider._extract_on_end = True
        session_provider.on_session_end([{"role": "user", "content": "hello"}])

        events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        (usage,) = [e for e in events if e["event"] == "usage"]
        assert (usage["kind"], usage["calls"], usage["input_tokens"], usage["output_tokens"]) == (
            "extraction", 1, 2400, 900,
        )

    def test_a_timed_out_extraction_is_traced(self, session_provider, tmp_path):
        async def extract(_messages):
            raise TimeoutError

        session_provider._extract_on_end = True
        session_provider._extract = extract
        session_provider.on_session_end([{"role": "user", "content": "hello"}])

        events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        (error,) = [e for e in events if e["event"] == "error"]
        assert (error["kind"], error["message"]) == ("extraction", "TimeoutError")

    def test_a_new_session_starts_with_nothing_recalled(self, session_provider):
        decimal, _ = self._two_recalled(session_provider)

        session_provider.on_session_switch("session-next")
        session_provider.on_session_end(_session(decimal))

        assert _record(session_provider, decimal) == (0.0, 0.0)


# =============================================================================
# Registration Tests
# =============================================================================


class TestRegistration:
    """Tests for how Hermes finds and activates the provider."""

    def test_register_hands_back_a_provider(self):
        """register(ctx) is the contract Hermes' plugin loader calls."""
        captured = []

        class Ctx:
            def register_memory_provider(self, provider):
                captured.append(provider)

        register(Ctx())

        assert isinstance(captured[0], RuntimeMemoryProvider)

    def test_entry_point_is_declared(self):
        """The entry point must name a package, not a bare module.

        Hermes resolves a provider's directory without importing it, so a module
        entry point silently loses its plugin.yaml, dashboard panel and CLI.
        """
        root = Path(__file__).resolve().parents[2]
        config = tomllib.loads((root / "pyproject.toml").read_text())

        group = config["project"]["entry-points"]["hermes_agent.memory_providers"]

        assert group[PROVIDER_NAME] == "runtime_memory.hermes"
        assert (root / "src" / "runtime_memory" / "hermes" / "__init__.py").exists()
        assert (root / "src" / "runtime_memory" / "hermes" / "plugin.yaml").exists()

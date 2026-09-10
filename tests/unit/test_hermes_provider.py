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
import sys
import tomllib
from pathlib import Path

import pytest

from memory_layer.core.models import Outcome
from memory_layer.hermes import MemoryLayerProvider, register
from memory_layer.hermes._base import RecallStatus, is_trivial_prompt
from memory_layer.hermes.bridge import DEFAULT_TIMEOUT, run_sync, spawn
from memory_layer.hermes.provider import PROVIDER_NAME, _embedding_provider_name
from memory_layer.hermes.trace import TRACE_ENV_VAR, TraceWriter


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """An initialized provider backed by a throwaway store."""
    monkeypatch.setenv("MEMORY_LAYER_DB", str(tmp_path / "memories.db"))
    monkeypatch.setenv("MEMORY_LAYER_EMBEDDING", "mock")
    monkeypatch.delenv(TRACE_ENV_VAR, raising=False)

    instance = MemoryLayerProvider()
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
        status = RecallStatus(provider_label="Memory Layer", count=3)

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
        assert provider.name == PROVIDER_NAME == "memorylayer"

    def test_available_with_writable_store(self, provider):
        """Availability is a filesystem check, with no network involved."""
        assert provider.is_available() is True

    def test_unavailable_when_unwritable(self, tmp_path, monkeypatch):
        """An unwritable location reports unavailable with a usable reason."""
        blocked = tmp_path / "blocked"
        blocked.mkdir(mode=0o500)
        monkeypatch.setenv("MEMORY_LAYER_DB", str(blocked / "sub" / "memories.db"))

        instance = MemoryLayerProvider()

        assert instance.is_available() is False
        assert "MEMORY_LAYER_DB" in instance.unavailable_reason()

    def test_workspace_becomes_project(self, tmp_path, monkeypatch):
        """Memories are scoped to the workspace directory name."""
        monkeypatch.setenv("MEMORY_LAYER_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("MEMORY_LAYER_EMBEDDING", "mock")
        monkeypatch.delenv("MEMORY_LAYER_PROJECT", raising=False)

        instance = MemoryLayerProvider()
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


class TestEmbeddingBackend:
    """The store is shared, so a missing model must not turn into fake vectors."""

    def test_defaults_to_local_even_without_the_model(self, monkeypatch):
        """The engine factory decides what `local` resolves to, not the provider."""
        monkeypatch.delenv("MEMORY_LAYER_EMBEDDING", raising=False)
        _hide_sentence_transformers(monkeypatch)

        assert _embedding_provider_name() == "local"

    def test_override_is_passed_through(self, monkeypatch):
        """An explicit choice wins."""
        monkeypatch.setenv("MEMORY_LAYER_EMBEDDING", "voyage")

        assert _embedding_provider_name() == "voyage"

    def test_missing_model_writes_no_vector(self, tmp_path, monkeypatch):
        """Recall still works, and nothing meaningless lands in the index."""
        monkeypatch.setenv("MEMORY_LAYER_DB", str(tmp_path / "memories.db"))
        monkeypatch.delenv("MEMORY_LAYER_EMBEDDING", raising=False)
        _hide_sentence_transformers(monkeypatch)

        instance = MemoryLayerProvider()
        instance.initialize("session-null", agent_context="primary")
        try:
            assert instance._engine.embedding_provider.available is False

            _call(
                instance,
                "memorylayer_remember",
                content="Run pytest from the repo root",
                category="gotcha",
            )
            block = instance.prefetch("how do I run the tests?")

            assert "Run pytest from the repo root" in block
            assert instance._engine.retriever.indexed_with_embeddings == 0
        finally:
            instance.shutdown()


# =============================================================================
# Recall Tests
# =============================================================================


class TestRecall:
    """Tests for the per-turn retrieval path."""

    def test_prefetch_injects_stored_memories(self, provider):
        """A stored memory is retrieved and rendered for the prompt."""
        _call(
            provider,
            "memorylayer_remember",
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
            "memorylayer_remember",
            content="Use snake_case here",
            category="convention",
        )

        block = provider.prefetch("naming style?")

        assert result["memory_id"] in block

    def test_prefetch_flags_proven_memories(self, provider):
        """One success is enough to mark a memory as having worked."""
        stored = _call(
            provider, "memorylayer_remember", content="Clear the cache", category="troubleshooting"
        )
        run_sync(provider._engine.record_outcome([stored["memory_id"]], Outcome.WORKED))

        line = _line_for(provider.prefetch("cache problems"), stored["memory_id"])

        assert "has worked before" in line

    def test_prefetch_flags_discredited_memories(self, provider):
        """One failure is enough to mark a memory as having failed."""
        stored = _call(
            provider, "memorylayer_remember", content="Delete the lockfile", category="workaround"
        )
        run_sync(provider._engine.record_outcome([stored["memory_id"]], Outcome.FAILED))

        line = _line_for(provider.prefetch("lockfile trouble"), stored["memory_id"])

        assert "has failed before" in line

    def test_mixed_record_is_left_unmarked(self, provider):
        """A memory that both worked and failed carries no claim either way."""
        stored = _call(
            provider, "memorylayer_remember", content="Retry the request", category="pattern"
        )
        memory_id = stored["memory_id"]
        run_sync(provider._engine.record_outcome([memory_id], Outcome.WORKED))
        run_sync(provider._engine.record_outcome([memory_id], Outcome.FAILED))

        line = _line_for(provider.prefetch("request keeps failing"), memory_id)

        assert "has worked before" not in line
        assert "has failed before" not in line

    def test_trivial_prompt_skips_recall(self, provider):
        """An acknowledgement does not trigger retrieval."""
        _call(provider, "memorylayer_remember", content="Something", category="general")

        assert provider.prefetch("thanks!") == ""

    def test_empty_store_injects_nothing(self, provider):
        """No memories means no block, not an empty heading."""
        assert provider.prefetch("anything at all?") == ""

    def test_recall_survives_engine_failure(self, provider, monkeypatch):
        """A broken search costs the recall, never the turn."""

        def explode(*args, **kwargs):
            raise RuntimeError("storage is down")

        monkeypatch.setattr(provider._engine, "search", explode)

        assert provider.prefetch("a real question") == ""

    def test_recall_status_reflects_last_prefetch(self, provider):
        """The indicator counts this turn's recall, never a stale one."""
        assert provider.recall_status() is None

        _call(provider, "memorylayer_remember", content="A fact", category="general")
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

        found = _call(provider, "memorylayer_recall", query="tabs")

        assert found["count"] == 1
        assert found["memories"][0]["category"] == "preference"

    def test_mirroring_can_be_disabled(self, tmp_path, monkeypatch):
        """Mirroring is a setting, not a fixed behavior."""
        monkeypatch.setenv("MEMORY_LAYER_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("MEMORY_LAYER_EMBEDDING", "mock")
        monkeypatch.setenv("MEMORY_LAYER_MIRROR_WRITES", "false")

        instance = MemoryLayerProvider()
        instance.initialize("s")
        try:
            instance.on_memory_write("add", "user", "Should not be stored")
            run_sync(asyncio.sleep(0.3))

            assert _call(instance, "memorylayer_recall", query="stored")["count"] == 0
        finally:
            instance.shutdown()

    def test_removals_are_not_mirrored(self, provider):
        """Only additions and replacements mirror; a removal is not content."""
        provider.on_memory_write("remove", "user", "Some old fact")
        run_sync(asyncio.sleep(0.3))

        assert _call(provider, "memorylayer_recall", query="old fact")["count"] == 0

    def test_subagents_do_not_write(self, tmp_path, monkeypatch):
        """Non-primary contexts read the store but never add to it."""
        monkeypatch.setenv("MEMORY_LAYER_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("MEMORY_LAYER_EMBEDDING", "mock")

        instance = MemoryLayerProvider()
        instance.initialize("s", agent_context="subagent")
        try:
            assert instance._writes_allowed is False

            result = _call(instance, "memorylayer_remember", content="X", category="general")
            assert "error" in result

            instance.on_memory_write("add", "user", "Also blocked")
            run_sync(asyncio.sleep(0.3))
            assert _call(instance, "memorylayer_recall", query="blocked")["count"] == 0
        finally:
            instance.shutdown()

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
        assert "error" in _call(provider, "memorylayer_nonexistent")

    def test_bad_category_lists_valid_ones(self, provider):
        """A wrong category tells the model what it may use instead."""
        result = _call(provider, "memorylayer_remember", content="X", category="nonsense")

        assert "nonsense" in result["error"]
        assert "convention" in result["error"]

    def test_missing_content_is_rejected(self, provider):
        """Empty content is refused rather than stored."""
        assert "error" in _call(provider, "memorylayer_remember", content="   ", category="general")

    def test_bad_outcome_is_rejected(self, provider):
        """Only the three defined outcomes are accepted."""
        result = _call(provider, "memorylayer_outcome", outcome="great")

        assert "great" in result["error"]

    def test_stats_reports_the_store(self, provider):
        """Stats summarize what is held, broken down by category."""
        _call(provider, "memorylayer_remember", content="A gotcha", category="gotcha")

        stats = _call(provider, "memorylayer_stats")

        assert stats["total_memories"] == 1
        assert stats["by_category"]["gotcha"] == 1

    def test_explicit_recall_joins_the_turn(self, provider):
        """A tool search is creditable by a later outcome, like automatic recall."""
        _call(provider, "memorylayer_remember", content="A fact", category="general")

        _call(provider, "memorylayer_recall", query="fact")
        result = _call(provider, "memorylayer_outcome", outcome="worked")

        assert result["recorded"] is True


# =============================================================================
# Outcome Tests
# =============================================================================


class TestOutcomes:
    """Tests for the feedback loop, which is the point of the integration."""

    def test_outcome_defaults_to_this_turn(self, provider):
        """The model need not repeat ids back to score what it was given."""
        _call(
            provider,
            "memorylayer_remember",
            content="Restart the daemon",
            category="troubleshooting",
        )
        provider.prefetch("daemon is stuck")

        result = _call(provider, "memorylayer_outcome", outcome="worked")

        assert result["recorded"] is True
        assert result["updated"][0]["outcome_score"] == pytest.approx(0.2)

    def test_failure_costs_more_than_success_gains(self, provider):
        """The asymmetry that makes bad advice sink is preserved end to end."""
        stored = _call(
            provider, "memorylayer_remember", content="Try turning it off", category="workaround"
        )
        memory_id = stored["memory_id"]

        worked = _call(provider, "memorylayer_outcome", outcome="worked", memory_ids=[memory_id])
        failed = _call(provider, "memorylayer_outcome", outcome="failed", memory_ids=[memory_id])

        assert worked["updated"][0]["outcome_score"] == pytest.approx(0.2)
        assert failed["updated"][0]["outcome_score"] == pytest.approx(-0.1)

    def test_outcome_without_recall_is_a_no_op(self, provider):
        """With nothing recalled there is nothing to credit or blame."""
        result = _call(provider, "memorylayer_outcome", outcome="worked")

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
        monkeypatch.setenv("MEMORY_LAYER_DB", str(tmp_path / "m.db"))
        monkeypatch.setenv("MEMORY_LAYER_EMBEDDING", "mock")
        trace_path = tmp_path / "traces" / "run.jsonl"
        monkeypatch.setenv(TRACE_ENV_VAR, str(trace_path))

        instance = MemoryLayerProvider()
        instance.initialize("session-9")
        try:
            _call(instance, "memorylayer_remember", content="A useful fact", category="general")
            instance.prefetch("tell me the fact")
            _call(instance, "memorylayer_outcome", outcome="worked")
        finally:
            instance.shutdown()

        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        by_event = {event["event"]: event for event in events}

        assert by_event["recall"]["retrieved"][0]["memory_id"]
        assert by_event["recall"]["turn_id"] == by_event["outcome"]["turn_id"]
        assert by_event["outcome"]["outcome"] == "worked"
        assert by_event["outcome"]["origin"] == "auto"
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

        assert isinstance(captured[0], MemoryLayerProvider)

    def test_entry_point_is_declared(self):
        """The entry point must name a package, not a bare module.

        Hermes resolves a provider's directory without importing it, so a module
        entry point silently loses its plugin.yaml, dashboard panel and CLI.
        """
        root = Path(__file__).resolve().parents[2]
        config = tomllib.loads((root / "pyproject.toml").read_text())

        group = config["project"]["entry-points"]["hermes_agent.memory_providers"]

        assert group[PROVIDER_NAME] == "memory_layer.hermes"
        assert (root / "src" / "memory_layer" / "hermes" / "__init__.py").exists()
        assert (root / "src" / "memory_layer" / "hermes" / "plugin.yaml").exists()

"""runtime-memory as a Hermes Agent memory provider.

Hermes discovers this through the ``hermes_agent.memory_providers`` entry point
and activates it with ``memory.provider: runtimememory``. Once active it replaces
the built-in note files rather than supplementing them, so recall stops being a
fixed block of text pasted into every prompt and becomes a per-turn retrieval
against the same SQLite store Claude Code and the MCP clients already share.

Two design choices are worth stating up front, because both differ from the
built-in provider:

Recall is synchronous. The Hermes contract expects ``prefetch()`` to hand back a
result warmed in the background on the previous turn, because the bundled
providers talk to network services. This store is local SQLite, so recall runs
against the turn's actual query instead of the one before it. That matters for
measurement as much as for quality: an off-by-one between question and retrieval
would confound any attempt to attribute an outcome to what was recalled.

Writes are explicit. Persisting every turn verbatim would fill a curated store
with conversational debris and degrade the retrieval it exists to serve. Memories
arrive from the ``runtimememory_remember`` tool, from mirrored built-in memory
writes, and - only when switched on - from end-of-session extraction.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runtime_memory.core.logging import get_logger
from runtime_memory.core.models import (
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
)
from runtime_memory.core.paths import default_db_path
from runtime_memory.core.retrieval import RetrievalConfig
from runtime_memory.hermes._base import (
    INDICATOR_GLYPH,
    MemoryProvider,
    RecallStatus,
    is_trivial_prompt,
)
from runtime_memory.hermes.bridge import run_sync, spawn
from runtime_memory.hermes.tools import TOOL_SCHEMAS, dispatch

if TYPE_CHECKING:
    from runtime_memory.core.engine import EngineStats, MemoryEngine
    from runtime_memory.core.models import Memory, SearchResult

logger = get_logger(__name__)

PROVIDER_NAME = "runtimememory"
PROVIDER_LABEL = "Runtime Memory"

DEFAULT_RECALL_LIMIT = 8
DEFAULT_MIN_SCORE = 0.0

_WRITE_CONTEXTS = frozenset({"primary", ""})
"""Agent contexts allowed to write. Subagents, cron and flush runs read only."""

_MIRROR_CATEGORY = {
    "memory": MemoryCategory.CONTEXT,
    "user": MemoryCategory.PREFERENCE,
}
"""Hermes built-in write targets mapped onto Runtime Memory categories."""


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _embedding_provider_name() -> str:
    """Name the embedding backend for the engine to build.

    ``local`` is the right answer even when ``sentence-transformers`` is
    missing: the engine factory degrades it to the null provider, which indexes
    no vectors and leaves retrieval on the BM25 half of the hybrid. Repeating
    that check here would give the store two answers to the same question, and
    picking ``mock`` writes hash-derived vectors alongside real ones.
    """
    return os.environ.get("RUNTIME_MEMORY_EMBEDDING") or "local"


class RuntimeMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by a local Runtime Memory engine."""

    pre_compress_checkpoint_api_version = 1

    def __init__(self) -> None:
        """Create an inactive provider. Nothing is opened until ``initialize``."""
        self._engine: MemoryEngine | None = None
        self._session_id: str = ""
        self._project: str | None = None
        self._writes_allowed: bool = True

        self._db_path: Path = Path(
            os.environ.get("RUNTIME_MEMORY_DB") or default_db_path()
        ).expanduser()
        self._recall_limit: int = int(
            os.environ.get("RUNTIME_MEMORY_RECALL_LIMIT", DEFAULT_RECALL_LIMIT)
        )
        self._min_score: float = float(os.environ.get("RUNTIME_MEMORY_MIN_SCORE", DEFAULT_MIN_SCORE))
        self._mirror_builtin: bool = _env_flag("RUNTIME_MEMORY_MIRROR_WRITES", True)
        self._extract_on_end: bool = _env_flag("RUNTIME_MEMORY_EXTRACT_ON_END", False)

        # Last recall, kept so an outcome can be attributed without the model
        # having to repeat the memory ids back to us.
        self._turn_id: str = ""
        self._last_ids: list[str] = []
        self._last_count: int = 0

        from runtime_memory.hermes.trace import TraceWriter  # noqa: PLC0415

        self._trace = TraceWriter()

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        """Provider name, matched against ``memory.provider``."""
        return PROVIDER_NAME

    def is_available(self) -> bool:
        """Whether the store can be opened. Checks the filesystem only."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            return os.access(self._db_path.parent, os.W_OK)
        except OSError as exc:
            logger.debug(f"Runtime Memory unavailable: {exc}")
            return False

    def unavailable_reason(self) -> str:
        """Explain an unavailable store."""
        return f"Cannot write to {self._db_path.parent}. Set RUNTIME_MEMORY_DB to a writable path."

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Open the engine and warm it before the first turn.

        Args:
            session_id: The Hermes session this provider instance serves.
            **kwargs: Hermes context. ``agent_context`` gates writes;
                ``agent_workspace`` scopes memories to a project.
        """
        from runtime_memory.core.engine import EngineConfig, MemoryEngine  # noqa: PLC0415

        self._session_id = session_id

        agent_context = kwargs.get("agent_context", "primary")
        self._writes_allowed = agent_context in _WRITE_CONTEXTS
        if not self._writes_allowed:
            logger.info(f"Read-only in '{agent_context}' context")

        workspace = kwargs.get("agent_workspace")
        self._project = os.environ.get("RUNTIME_MEMORY_PROJECT") or (
            Path(workspace).name if workspace else None
        )

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        config = EngineConfig(
            db_path=str(self._db_path),
            embedding_provider=_embedding_provider_name(),
            track_last_search=True,
            # Signal weights come from the environment so an evaluation arm can
            # ablate one, such as running with outcome weight 0 to separate a
            # shared store from the learning on top of it.
            retrieval_config=RetrievalConfig.from_env(),
        )

        engine = MemoryEngine(config=config)
        run_sync(engine.initialize(), timeout=120.0)
        self._engine = engine

        # Pull the embedding model into memory now. It costs ~20s on first load,
        # and paying that here rather than inside the user's first turn is the
        # difference between a slow start and a stalled reply.
        spawn(self._warm(), label="warmup")

        logger.info(
            f"Runtime Memory ready (db={self._db_path}, project={self._project}, "
            f"writes={'on' if self._writes_allowed else 'off'})"
        )

    async def _warm(self) -> None:
        """Touch the retrieval path once so the first real query is fast."""
        if self._engine is not None:
            await self._engine.search("warmup", limit=1, track_usage=False)

    def shutdown(self) -> None:
        """Close the engine. The shared event loop deliberately stays up."""
        if self._engine is None:
            return
        try:
            run_sync(self._engine.close(), timeout=10.0)
        except Exception as exc:  # shutdown must not raise
            logger.warning(f"Engine close failed: {exc}")
        finally:
            self._engine = None

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Rebind to a new session id so later writes land in the right record."""
        self._session_id = new_session_id
        if reset:
            self._turn_id, self._last_ids, self._last_count = "", [], 0

    # -- recall --------------------------------------------------------------

    def system_prompt_block(self) -> str:
        """Static guidance. Recalled content is injected by ``prefetch``."""
        return (
            "You have persistent memory across sessions via Runtime Memory. "
            "Relevant memories are retrieved automatically each turn. "
            "Use runtimememory_remember to save a durable fact worth recalling "
            "later, and runtimememory_outcome to report whether recalled memories "
            "actually helped - that feedback decides what surfaces next time."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Retrieve memories for this turn and format them for injection.

        Args:
            query: The user's message for the upcoming turn.
            session_id: Session scope, unused for a single shared store.

        Returns:
            A formatted memory block, or ``""`` when nothing is worth injecting.
        """
        self._turn_id = uuid.uuid4().hex
        self._last_ids, self._last_count = [], 0

        if self._engine is None or is_trivial_prompt(query):
            return ""

        started = time.perf_counter()
        try:
            results = run_sync(
                self._engine.search(
                    query=query,
                    limit=self._recall_limit,
                    project=self._project,
                    min_score=self._min_score,
                ),
                timeout=15.0,
            )
        except Exception as exc:  # a failed recall must never break the turn
            logger.warning(f"Recall failed: {exc}")
            return ""

        if not results:
            return ""

        self._last_ids = [r.memory.id for r in results]
        self._last_count = len(results)

        self._trace.recall(
            turn_id=self._turn_id,
            session_id=session_id or self._session_id,
            query=query,
            results=results,
            project=self._project,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return self._format(results)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op. Recall is synchronous, so there is nothing to warm."""

    def recall_status(self) -> RecallStatus | None:
        """Report what the last recall injected, for the indicator line."""
        if not self._last_count:
            return None
        return RecallStatus(
            provider_label=PROVIDER_LABEL,
            count=self._last_count,
            glyph=INDICATOR_GLYPH,
        )

    def _format(self, results: list[SearchResult]) -> str:
        """Render search results as a promptable block.

        Each line carries its memory id so the model can name specific memories
        when reporting an outcome, and its outcome score so a memory with a poor
        track record reads as weaker evidence than one that keeps working.
        """
        lines = ["## Relevant memories", ""]
        for result in results:
            memory = result.memory
            # Thresholds are inclusive so a single recorded outcome is enough to
            # show: one `worked` lands exactly on +0.2, one `failed` on -0.3.
            marker = ""
            if memory.outcome_score >= 0.2:
                marker = " (has worked before)"
            elif memory.outcome_score <= -0.2:
                marker = " (has failed before)"
            lines.append(f"- [{memory.category.value}] {memory.content}{marker} `{memory.id}`")
        return "\n".join(lines)

    # -- writes --------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Called after every turn. Intentionally does not persist.

        Storing raw turns would bury the curated memories that make retrieval
        useful. Facts reach the store through the remember tool, mirrored
        built-in writes, or opt-in end-of-session extraction.
        """

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror a built-in memory write into the shared store.

        This is what lifts the built-in character cap in practice: Hermes keeps
        its small always-injected note file, and the same fact also lands here,
        where it is retrieved on relevance instead of pasted in wholesale.
        """
        if not (self._mirror_builtin and self._writes_allowed):
            return
        if action not in {"add", "replace"} or not content.strip():
            return

        category = _MIRROR_CATEGORY.get(target, MemoryCategory.CONTEXT)
        spawn(
            self._store(
                content=content.strip(),
                category=category,
                source=MemorySource.IMPORTED,
                tags=["hermes", f"builtin-{target}"],
                metadata={"hermes_action": action, **(metadata or {})},
                kind="mirror",
            ),
            label="mirror write",
        )

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Optionally extract durable facts when a session really ends."""
        if not (self._extract_on_end and self._writes_allowed):
            return
        if self._engine is None or not messages:
            return
        logger.info(f"Session end: extraction over {len(messages)} messages")
        spawn(self._extract(messages), label="extraction")

    async def _extract(self, messages: list[dict[str, Any]]) -> None:
        """Run LLM extraction over a finished session.

        Requires the ``extraction`` extra and an API key. Failures are logged and
        dropped: a missed extraction is a lost convenience, not a lost session.
        """
        from runtime_memory.extraction.extractor import MemoryExtractor  # noqa: PLC0415

        transcript = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages if m.get("content")
        )
        result = await MemoryExtractor().extract_and_store(
            transcript=transcript,
            engine=self._require_engine(),
            project=self._project,
        )
        if not result.success:
            logger.warning(f"Extraction failed: {result.error}")
            return

        # extract_and_store writes through the engine without handing back the
        # stored rows, so the trace records how many landed, not which.
        if result.memory_count:
            self._trace.write_turn(
                turn_id=self._turn_id,
                session_id=self._session_id,
                memory_ids=[],
                kind="extraction",
                count=result.memory_count,
            )

    async def _store(
        self,
        *,
        content: str,
        category: MemoryCategory,
        source: MemorySource,
        tags: list[str],
        metadata: dict[str, Any] | None = None,
        kind: str = "tool",
    ) -> Memory:
        """Write one memory and trace it."""
        memory = await self._engine.add(  # type: ignore[union-attr]
            content=content,
            category=category,
            project=self._project,
            scope=MemoryScope.PROJECT if self._project else MemoryScope.GLOBAL,
            source=source,
            tags=tags,
            metadata={"hermes_session": self._session_id, **(metadata or {})},
        )
        self._trace.write_turn(
            turn_id=self._turn_id,
            session_id=self._session_id,
            memory_ids=[memory.id],
            kind=kind,
        )
        return memory

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Function-calling schemas for the tools this provider handles."""
        return TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        """Handle one tool call, returning a JSON string."""
        return dispatch(self, tool_name, args)

    # -- synchronous helpers used by tool dispatch ---------------------------

    def _require_engine(self) -> MemoryEngine:
        """Return the engine or explain why there isn't one."""
        if self._engine is None:
            raise RuntimeError("Runtime Memory is not initialized")
        return self._engine

    def remember(self, *, content: str, category: MemoryCategory, tags: list[str]) -> Memory:
        """Store a memory on the model's explicit instruction."""
        self._require_engine()
        if not self._writes_allowed:
            raise RuntimeError("Writes are disabled in this agent context")
        return run_sync(
            self._store(
                content=content,
                category=category,
                source=MemorySource.EXPLICIT,
                tags=[*tags, "hermes"],
            )
        )

    def recall(
        self, *, query: str, limit: int, category: MemoryCategory | None
    ) -> list[SearchResult]:
        """Run an explicit search, separate from the automatic per-turn recall."""
        engine = self._require_engine()
        results = run_sync(
            engine.search(query=query, limit=limit, category=category, project=self._project)
        )
        # Fold into the turn's recall set so an outcome can credit these too.
        for result in results:
            if result.memory.id not in self._last_ids:
                self._last_ids.append(result.memory.id)
        return results

    def record_outcome(
        self, *, outcome: Outcome, memory_ids: list[str] | None = None
    ) -> list[Memory]:
        """Apply outcome feedback, defaulting to this turn's recalled memories."""
        engine = self._require_engine()
        targets = memory_ids or self._last_ids
        if not targets:
            return []

        updated = run_sync(engine.record_outcome(targets, outcome))
        self._trace.outcome(
            turn_id=self._turn_id,
            session_id=self._session_id,
            outcome=outcome.value,
            memory_ids=[memory.id for memory in updated],
            origin="tool" if memory_ids else "auto",
        )
        return updated

    def stats(self) -> EngineStats:
        """Return store statistics."""
        return run_sync(self._require_engine().stats(project=self._project))

    # -- configuration -------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        """Fields offered by ``hermes memory setup``."""
        return [
            {
                "key": "db_path",
                "description": "SQLite store shared with Claude Code and MCP clients",
                "default": str(default_db_path()),
                "env_var": "RUNTIME_MEMORY_DB",
                "type": "text",
            },
            {
                "key": "recall_limit",
                "description": "Memories injected per turn",
                "default": DEFAULT_RECALL_LIMIT,
                "env_var": "RUNTIME_MEMORY_RECALL_LIMIT",
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
            },
            {
                "key": "mirror_writes",
                "description": "Mirror built-in memory writes into the store",
                "default": True,
                "env_var": "RUNTIME_MEMORY_MIRROR_WRITES",
                "type": "boolean",
            },
            {
                "key": "extract_on_end",
                "description": "Extract memories at session end (needs an API key)",
                "default": False,
                "env_var": "RUNTIME_MEMORY_EXTRACT_ON_END",
                "type": "boolean",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """No-op: every setting is an environment variable."""

    def backup_paths(self) -> list[str]:
        """The store lives outside HERMES_HOME, so name it for ``hermes backup``."""
        return [str(self._db_path)]

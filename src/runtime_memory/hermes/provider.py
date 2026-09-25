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
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runtime_memory.core.attribution import Attribution, attribute, session_verdict, transcript
from runtime_memory.core.logging import get_logger
from runtime_memory.core.models import (
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
    RelationType,
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
DEFAULT_CONFLICT_COUNTERPARTS = 3
"""Stored memories shown beside the recall because they contradict one in it."""
DEFAULT_MIN_SCORE = 0.0
EXTRACTION_TIMEOUT = 360.0
"""Seconds to wait for session-end extraction, one LLM round trip.

Room for the extraction call's full output, 16,000 tokens with thinking, at about
50 tokens a second.
"""

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


def _extraction_unavailable_reason() -> str | None:
    """Why session-end extraction cannot run, or None when it can.

    Extraction needs a package and a credential, and the failure without either
    is invisible: it happens on a background task at session end, where the
    exception is swallowed so a lost extraction never takes a session with it.
    A provider that reports extraction as on while nothing is ever extracted is
    worse than one that refuses, so the check happens once at startup where it
    can still be said out loud.

    An unset ``ANTHROPIC_API_KEY`` does not mean there is no credential. The SDK
    also accepts an auth token, a signed-in profile on disk, and workload
    identity federation, so each documented source is checked before refusing.

    Returns:
        A reason to show the operator, or None when extraction is usable.
    """
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        return "the anthropic package is not installed (pip install 'runtime-memory[extraction]')"

    # Let the SDK apply its own precedence to the environment rather than
    # reimplementing it. Construction does not raise when nothing resolves.
    try:
        client = anthropic.AsyncAnthropic()
        if getattr(client, "api_key", None) or getattr(client, "auth_token", None):
            return None
    except Exception:  # noqa: BLE001 - an unconstructable client is also a refusal
        pass

    if os.environ.get("ANTHROPIC_FEDERATION_RULE_ID"):
        return None
    if (Path.home() / ".config" / "anthropic").exists():
        return None

    return (
        "no Anthropic credentials are configured (set ANTHROPIC_API_KEY, or sign "
        "in with `ant auth login`)"
    )


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
        self._conflict_counterparts: int = int(
            os.environ.get("RUNTIME_MEMORY_CONFLICT_COUNTERPARTS", DEFAULT_CONFLICT_COUNTERPARTS)
        )
        self._mirror_builtin: bool = _env_flag("RUNTIME_MEMORY_MIRROR_WRITES", True)
        self._extract_on_end: bool = _env_flag("RUNTIME_MEMORY_EXTRACT_ON_END", False)
        self._ask_citations: bool = _env_flag("RUNTIME_MEMORY_ASK_CITATIONS", True)
        self._session_outcomes: bool = _env_flag("RUNTIME_MEMORY_SESSION_OUTCOMES", True)

        # Everything recalled in the session, and what the agent already scored
        # with the outcome tool, for the outcome recorded when the session ends.
        self._session_recalled: dict[str, None] = {}
        self._session_reported: set[str] = set()

        # Last recall, kept so an outcome can be attributed without the model
        # having to repeat the memory ids back to us.
        self._turn_id: str = ""
        self._last_ids: list[str] = []
        self._last_count: int = 0

        from runtime_memory.hermes.trace import TRACE_FILE_NAME, TraceWriter  # noqa: PLC0415

        # On by default: without it there is no record of what memory did in a
        # session, what it cost, or which outcomes came from where.
        self._trace = TraceWriter(default=self._db_path.parent / TRACE_FILE_NAME)

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

        if self._extract_on_end:
            reason = _extraction_unavailable_reason()
            if reason:
                logger.warning(
                    f"Extraction at session end was requested but is off: {reason}"
                )
                self._extract_on_end = False

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

        self._start_model_load()

        logger.info(
            f"Runtime Memory ready (db={self._db_path}, project={self._project}, "
            f"search={engine.search_mode}, writes={'on' if self._writes_allowed else 'off'})"
        )
        # The engine factory degrades a missing model to keyword matching on
        # purpose, and logs it as info so the CLI stays quiet. A provider starts
        # once per session, and the extra usually goes missing by being installed
        # into a different environment from Hermes', so here it is a warning that
        # names the interpreter to install into. Choosing `null` says keyword
        # matching is intended.
        if engine.search_mode == "keyword" and os.environ.get("RUNTIME_MEMORY_EMBEDDING") != "null":
            logger.warning(
                "No embedding backend in this Python environment, so recall uses "
                "keyword matching only. For semantic search, install it into the "
                f"environment Hermes runs in: {sys.executable} -m pip install "
                "'runtime-memory[embedding]'. Set RUNTIME_MEMORY_EMBEDDING=null to "
                "choose keyword matching and silence this."
            )

    def _start_model_load(self) -> None:
        """Load the embedding model on a thread of its own.

        The first embedding in a new process loads the model, which took 14 to 15
        seconds in Hermes' environment on 2026-09-24. Hermes gives a provider's
        prefetch 8 seconds and then skips it. Loaded on the event loop, as it used
        to be, the model held up every search behind it, and in one-shot sessions
        every recall missed the window. Loaded here, it leaves the loop free, and
        ``prefetch`` searches by keyword until it is ready.
        """
        embedder = self._require_engine().embedding_provider
        if embedder.available and not embedder.loaded:
            threading.Thread(
                target=self._load_model, name="runtime-memory-model", daemon=True
            ).start()

    def _load_model(self) -> None:
        """Thread body: load the model, logging rather than raising a failure."""
        try:
            self._require_engine().embedding_provider.load()
        except Exception as exc:  # keyword search carries on without it
            logger.warning(f"Embedding model failed to load; recall stays keyword-only: {exc}")

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
        if new_session_id != self._session_id:
            self._session_recalled, self._session_reported = {}, set()
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

        # Search by keyword while the model is still loading: see
        # _start_model_load. A recall with memories found by keyword is worth
        # more than one that arrives after Hermes has stopped waiting.
        embedder = self._engine.embedding_provider
        loading = embedder.available and not embedder.loaded
        started = time.perf_counter()
        try:
            results = run_sync(
                self._engine.search(
                    query=query,
                    limit=self._recall_limit,
                    project=self._project,
                    min_score=self._min_score,
                    semantic=not loading,
                ),
                timeout=15.0,
            )
        except Exception as exc:  # a failed recall must never break the turn
            logger.warning(f"Recall failed: {exc}")
            return ""

        contradicts, counterparts = self._conflicts_shown(results)
        self._last_ids = [r.memory.id for r in results] + [m.id for m in counterparts]
        self._last_count = len(self._last_ids)
        self._session_recalled.update(dict.fromkeys(self._last_ids))

        block = self._format(results, contradicts, counterparts) if results else ""

        # Traced even when nothing came back. A recall that found nothing and a
        # recall that never happened look identical in an untraced run, and the
        # difference is the whole question when a store is still filling up.
        self._trace.recall(
            turn_id=self._turn_id,
            session_id=session_id or self._session_id,
            query=query,
            results=results,
            project=self._project,
            latency_ms=(time.perf_counter() - started) * 1000,
            search_mode="keyword" if loading else self._engine.search_mode,
            model_loading=loading,
            counterparts=[m.id for m in counterparts],
            contradicts=contradicts,
            block_chars=len(block),
        )
        return block

    def _conflicts_shown(
        self, results: list[SearchResult]
    ) -> tuple[dict[str, list[str]], list[Memory]]:
        """Find the contradictions to show with a recall, and what to add for them.

        When a recalled memory contradicts a stored one that was not recalled,
        the stored one is added, up to ``conflict_counterparts``, so the model
        sees both sides. Ranking alone has not kept the right side of a
        contradiction in the prompt.

        Args:
            results: The recall.

        Returns:
            For each memory shown, the shown memories it contradicts; and the
            counterparts added. Both empty if the lookup fails, which must not
            cost the turn its recall.
        """
        try:
            return run_sync(self._conflicts_for(results), timeout=5.0)
        except Exception as exc:
            logger.warning(f"Conflict lookup failed: {exc}")
            return {}, []

    async def _conflicts_for(
        self, results: list[SearchResult]
    ) -> tuple[dict[str, list[str]], list[Memory]]:
        engine = self._require_engine()
        recalled = [r.memory.id for r in results]
        others = await self._contradicted_by(recalled)
        wanted = list(dict.fromkeys(o for mid in recalled for o in others[mid] if o not in recalled))
        counterparts = [
            memory
            for memory in await engine.get_many(wanted)
            if not memory.archived and (self._project is None or memory.project == self._project)
        ][: self._conflict_counterparts]

        shown = set(recalled) | {m.id for m in counterparts}
        contradicts: dict[str, list[str]] = {}
        for mid in recalled:
            for other in others[mid]:
                if other not in shown:
                    continue
                # Each pair is marked from both sides, and a pair of recalled
                # memories is reached from both, so each side is added once.
                for one, two in ((mid, other), (other, mid)):
                    marked = contradicts.setdefault(one, [])
                    if two not in marked:
                        marked.append(two)
        return contradicts, counterparts

    async def _contradicted_by(self, memory_ids: list[str]) -> dict[str, list[str]]:
        """For each memory, the ids of the memories it is linked as contradicting."""
        engine = self._require_engine()
        found: dict[str, list[str]] = {}
        for memory_id in memory_ids:
            links = await engine.related(memory_id, RelationType.CONFLICTS_WITH)
            found[memory_id] = list(dict.fromkeys(
                link.target_id if link.source_id == memory_id else link.source_id
                for link in links
            ))
        return found

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

    def _format(
        self,
        results: list[SearchResult],
        contradicts: dict[str, list[str]] | None = None,
        counterparts: list[Memory] | None = None,
    ) -> str:
        """Render search results as a promptable block.

        Each line carries its memory id so the model can name specific memories
        when reporting an outcome, and its outcome record so a memory with a poor
        track record reads as weaker evidence than one that keeps working. Two
        memories linked as contradicting each other are both marked, naming the
        other, so the agent chooses between them knowingly rather than taking
        whichever ranked higher.
        """
        contradicts = contradicts or {}
        lines = ["## Relevant memories", ""]
        lines += [self._line(result.memory, contradicts) for result in results]
        if counterparts:
            lines += ["", "Also stored, and contradicting a memory above:"]
            lines += [self._line(memory, contradicts) for memory in counterparts]
        if any(contradicts.values()):
            lines += [
                "",
                "Memories marked as contradicting each other disagree. Check which one "
                "holds here before relying on either.",
            ]
        if self._ask_citations:
            # The only record of which memories a session used, short of a paid
            # judge: in the Tier 3 evaluation the agent never named an id unasked.
            lines += ["", "When you act on one of these memories, name its id in your reply."]
        return "\n".join(lines)

    def _line(self, memory: Memory, contradicts: dict[str, list[str]]) -> str:
        """One memory as a line of the block."""
        others = contradicts.get(memory.id)
        conflict = f" (contradicts {', '.join(f'`{o}`' for o in others)})" if others else ""
        return (
            f"- [{memory.category.value}] {memory.content}{self._record(memory)} "
            f"`{memory.id}`{conflict}"
        )

    def _record(self, memory: Memory) -> str:
        """A memory's outcome record as the model is shown it.

        Shown as counts, because "worked 3 times" says how much evidence there
        is, and "has worked before" read the same for one observation and ten.
        """
        model = self._engine.retriever.config.outcome_model if self._engine else None
        if model is None:
            # 3.x scoring keeps no counts worth showing. Thresholds are
            # inclusive so one outcome shows: +0.2 worked, -0.3 failed.
            if memory.outcome_score >= 0.2:
                return " (has worked before)"
            if memory.outcome_score <= -0.2:
                return " (has failed before)"
            return ""
        worked, failed = (round(n) for n in model.evidence(memory, datetime.now(UTC)))
        parts = [
            f"{label} {n} {'time' if n == 1 else 'times'}"
            for label, n in (("worked", worked), ("failed", failed))
            if n >= 1
        ]
        return f" ({', '.join(parts)})" if parts else ""

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
        """Extract durable facts and record the session's outcome, when it really ends.

        Both are optional. Extraction runs with ``RUNTIME_MEMORY_EXTRACT_ON_END``;
        the outcome, on by default, with ``RUNTIME_MEMORY_SESSION_OUTCOMES``.
        """
        if self._engine is None or not messages or not self._writes_allowed:
            return
        # Blocking, not spawned. Hermes calls this hook and then immediately
        # tears the provider down; in one-shot mode the process exits straight
        # after. Fired and forgotten, the extraction never survived long enough
        # to write anything, which looked exactly like extraction being off.
        # Hermes documents this hook as LLM-bound and runs it on a background
        # worker, so waiting here is what the contract expects.
        followed: list[Attribution] = []
        if self._extract_on_end:
            logger.info(f"Session end: extraction over {len(messages)} messages")
            try:
                followed = run_sync(self._extract(messages), timeout=EXTRACTION_TIMEOUT)
            except Exception as exc:  # a lost extraction must not take the session with it
                logger.warning(f"Extraction failed: {exc}")
                self._trace_error("extraction", exc)
        if self._session_outcomes:
            try:
                run_sync(self._record_session_outcome(messages, followed), timeout=30.0)
            except Exception as exc:  # nor must a lost outcome
                logger.warning(f"Session outcome not recorded: {exc}")
                self._trace_error("session_outcome", exc)

    def _trace_error(self, kind: str, exc: BaseException) -> None:
        """Trace a failure; a timeout's message is empty, so its type is named."""
        self._trace.error(
            turn_id=self._turn_id,
            session_id=self._session_id,
            kind=kind,
            message=str(exc) or type(exc).__name__,
        )

    async def _record_session_outcome(
        self, messages: list[dict[str, Any]], followed: list[Attribution]
    ) -> None:
        """Record the session's verdict against the memories it acted on.

        The verdict is the last test run in the session's tool results. It goes
        to the recalled memories an attributor names (cited by id, a command
        that was run, or named by extraction), once each, and to no others:
        applying a verdict to everything recalled left the store worse than
        recording nothing in three Tier 2 evaluation runs. Memories the agent
        already scored with the outcome tool this session are left alone.

        Args:
            messages: The session's messages.
            followed: Memories the extraction call judged the session followed.
        """
        verdict = session_verdict(messages)
        if verdict is None:
            logger.info("Session end: no test run found, so no outcome recorded")
            return
        engine = self._require_engine()
        recalled = await engine.get_many(list(self._session_recalled))
        found = {a.memory_id: a for a in attribute(messages, recalled)}
        for attribution in followed:
            found.setdefault(attribution.memory_id, attribution)

        by_source: dict[str, list[str]] = {}
        for attribution in found.values():
            if attribution.memory_id not in self._session_reported:
                by_source.setdefault(attribution.source, []).append(attribution.memory_id)
        for source, memory_ids in by_source.items():
            await engine.record_outcome(memory_ids, verdict)
            self._trace.outcome(
                turn_id=self._turn_id,
                session_id=self._session_id,
                outcome=verdict.value,
                memory_ids=memory_ids,
                origin=source,
            )
        logger.info(
            f"Session end: '{verdict.value}' recorded for "
            f"{sum(len(ids) for ids in by_source.values())} of {len(recalled)} recalled memories"
        )

    async def _extract(self, messages: list[dict[str, Any]]) -> list[Attribution]:
        """Run LLM extraction over a finished session.

        Requires the ``extraction`` extra and an API key. Failures are logged and
        dropped: a missed extraction is a lost convenience, not a lost session.

        Returns:
            The recalled memories the extraction call judged the session followed.
        """
        from runtime_memory.extraction.extractor import MemoryExtractor  # noqa: PLC0415

        engine = self._require_engine()
        result = await MemoryExtractor().extract_and_store(
            transcript=transcript(messages),
            engine=engine,
            project=self._project,
            recalled=await engine.get_many(list(self._session_recalled)),
        )
        # Recorded whether or not the extraction succeeded: a failed call is
        # still paid for.
        if result.usage is not None:
            self._trace.usage(
                turn_id=self._turn_id,
                session_id=self._session_id,
                kind="extraction",
                model=result.usage.model,
                calls=result.usage.calls,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            )
        if not result.success:
            logger.warning(f"Extraction failed: {result.error}")
            self._trace.error(
                turn_id=self._turn_id,
                session_id=self._session_id,
                kind="extraction",
                message=str(result.error),
            )
            return []

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
        # What the session learned again rather than stored a second time.
        if result.confirmed_ids:
            self._trace.confirm(
                turn_id=self._turn_id,
                session_id=self._session_id,
                memory_ids=result.confirmed_ids,
            )
        return [
            Attribution(memory_id, "extraction", evidence)
            for memory_id, evidence in result.acted_on
        ]

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
        self._session_recalled.update(dict.fromkeys(r.memory.id for r in results))
        return results

    def contradictions(self, memory_ids: list[str]) -> dict[str, list[str]]:
        """For each memory, the ids of the stored memories it is linked as contradicting."""
        return run_sync(self._contradicted_by(memory_ids))

    def record_outcome(
        self, *, outcome: Outcome, memory_ids: list[str] | None = None
    ) -> list[Memory]:
        """Apply outcome feedback to the memories named, and only those.

        Before 4.0.0 a call without ids applied the outcome to every memory
        recalled in the turn. That is the contract the Tier 2 evaluation measured
        three times, and each time it left the store worse than recording nothing:
        a turn's verdict reached the memory that misled it and the memory that was
        right about the same thing, so both sank together. An outcome now needs to
        name what it is about. A caller that has no attribution records nothing,
        which the trace shows as a declined outcome.
        """
        engine = self._require_engine()
        if not memory_ids:
            logger.info(
                "Outcome '%s' not recorded: no memory ids given. Name the memories "
                "the outcome is about; %d were recalled this turn.",
                outcome.value,
                len(self._last_ids),
            )
            self._trace.outcome(
                turn_id=self._turn_id,
                session_id=self._session_id,
                outcome=outcome.value,
                memory_ids=[],
                origin="declined",
            )
            return []

        updated = run_sync(engine.record_outcome(memory_ids, outcome))
        self._session_reported.update(memory_ids)
        self._trace.outcome(
            turn_id=self._turn_id,
            session_id=self._session_id,
            outcome=outcome.value,
            memory_ids=[memory.id for memory in updated],
            origin="tool",
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

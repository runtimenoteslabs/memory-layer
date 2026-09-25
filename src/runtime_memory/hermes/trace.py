"""Retrieval and outcome tracing for the Hermes provider.

This is the instrumentation the Tier 3 evaluation runs on. Every recall writes
one JSONL record naming the memories it injected and their scores; every recorded
outcome writes another naming the memories it credited or blamed. Joining the two
on ``turn_id`` reconstructs, per turn, what was retrieved and whether it helped -
which is the measurement the outcome-learning claim needs.

The Hermes provider traces by default, to ``hermes-trace.jsonl`` beside the store;
``RUNTIME_MEMORY_HERMES_TRACE`` names another file, or ``off`` turns tracing off.
A broken trace never breaks a turn: writes are best-effort and failures are
logged once. ``summarize`` reads a trace back for ``mem stats``.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runtime_memory.core.logging import get_logger

logger = get_logger(__name__)

TRACE_ENV_VAR = "RUNTIME_MEMORY_HERMES_TRACE"
"""Environment variable holding the trace file path, or ``off``."""

TRACE_FILE_NAME = "hermes-trace.jsonl"
"""The trace's file name beside the store, when no path is configured."""

_OFF = frozenset({"off", "none", "false", "0"})


class TraceWriter:
    """Append-only JSONL writer for retrieval and outcome events."""

    def __init__(self, path: str | Path | None = None, default: Path | None = None) -> None:
        """Initialize the writer.

        Args:
            path: Trace file path. Falls back to ``RUNTIME_MEMORY_HERMES_TRACE``,
                then to ``default``. ``off`` in either turns tracing off, and so
                does having none of the three.
            default: Where to trace when nothing is configured.
        """
        raw = str(path or os.environ.get(TRACE_ENV_VAR) or "").strip()
        if raw.lower() in _OFF:
            self._path = None
        elif raw:
            self._path = Path(raw).expanduser()
        else:
            self._path = default
        self._lock = threading.Lock()
        self._warned = False

        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(f"Trace directory unavailable ({exc}); tracing off")
                self._path = None

    @property
    def enabled(self) -> bool:
        """Whether events are being written anywhere."""
        return self._path is not None

    @property
    def path(self) -> Path | None:
        """The trace file path, or None when tracing is off."""
        return self._path

    def recall(
        self,
        *,
        turn_id: str,
        session_id: str,
        query: str,
        results: list[Any],
        project: str | None,
        latency_ms: float,
        search_mode: str | None = None,
        counterparts: list[str] | None = None,
        contradicts: dict[str, list[str]] | None = None,
        block_chars: int = 0,
        model_loading: bool = False,
    ) -> None:
        """Record what a recall injected.

        Args:
            turn_id: Identifier joining this recall to any later outcome.
            session_id: Hermes session the recall belongs to.
            query: The query text used for retrieval.
            results: ``SearchResult`` objects that were injected.
            project: Project filter in force, if any.
            latency_ms: Wall-clock retrieval time.
            search_mode: ``hybrid`` or ``keyword``, so a run that lost its
                embedding backend can be told apart from one that had it.
            counterparts: Memories shown beside the recall because they
                contradict one in it.
            contradicts: For each memory shown, the shown memories it was
                marked as contradicting.
            block_chars: Length of the injected block, which the agent's model
                reads on every call it makes in the turn. Characters, because
                that model's tokenizer is not Runtime Memory's to know.
            model_loading: The embedding model was still loading, so this recall
                searched by keyword although the store searches in hybrid mode.
        """
        self._write(
            {
                "event": "recall",
                "turn_id": turn_id,
                "session_id": session_id,
                "query": query,
                "project": project,
                "search_mode": search_mode,
                "model_loading": model_loading,
                "latency_ms": round(latency_ms, 2),
                "retrieved": [
                    {
                        "memory_id": r.memory.id,
                        "category": r.memory.category.value,
                        "score": round(r.score, 4),
                        "semantic_score": round(r.semantic_score, 4),
                        "outcome_score": round(r.memory.outcome_score, 4),
                        "worked": round(r.memory.worked, 4),
                        "failed": round(r.memory.failed, 4),
                        "use_count": r.memory.use_count,
                    }
                    for r in results
                ],
                "counterparts": counterparts or [],
                "contradicts": contradicts or {},
                "block_chars": block_chars,
            }
        )

    def outcome(
        self,
        *,
        turn_id: str,
        session_id: str,
        outcome: str,
        memory_ids: list[str],
        origin: str,
    ) -> None:
        """Record an outcome applied to memories.

        Args:
            turn_id: The turn the outcome is attributed to.
            session_id: Hermes session the outcome came from.
            outcome: ``worked``, ``failed`` or ``partial``.
            memory_ids: Memories the score change was applied to.
            origin: What produced the outcome: ``tool`` when the model reported
                it, ``declined`` when the call named no memories, and ``cited``,
                ``command`` or ``extraction`` for an outcome recorded at session
                end, after the attribution that found the memory.
        """
        self._write(
            {
                "event": "outcome",
                "turn_id": turn_id,
                "session_id": session_id,
                "outcome": outcome,
                "memory_ids": memory_ids,
                "origin": origin,
            }
        )

    def write_turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        memory_ids: list[str],
        kind: str,
        count: int | None = None,
    ) -> None:
        """Record memories created from a conversation turn.

        Args:
            turn_id: The turn the write belongs to.
            session_id: Hermes session that produced it.
            memory_ids: Ids written, where the caller knows them.
            kind: What produced the write, e.g. ``tool`` or ``extraction``.
            count: How many memories were written. Defaults to ``len(memory_ids)``;
                pass it explicitly when the ids are not available.
        """
        self._write(
            {
                "event": "write",
                "turn_id": turn_id,
                "session_id": session_id,
                "kind": kind,
                "memory_ids": memory_ids,
                "count": len(memory_ids) if count is None else count,
            }
        )

    def confirm(self, *, turn_id: str, session_id: str, memory_ids: list[str]) -> None:
        """Record stored memories a session learned again instead of storing copies.

        Args:
            turn_id: The turn the confirmation belongs to.
            session_id: Hermes session that produced it.
            memory_ids: The memories confirmed.
        """
        self._write(
            {
                "event": "confirm",
                "turn_id": turn_id,
                "session_id": session_id,
                "memory_ids": memory_ids,
            }
        )

    def usage(
        self,
        *,
        turn_id: str,
        session_id: str,
        kind: str,
        model: str,
        calls: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record the model calls Runtime Memory itself made, such as extraction.

        Args:
            turn_id: The turn the calls belong to.
            session_id: Hermes session they were made for.
            kind: What made them, e.g. ``extraction``.
            model: The model called.
            calls: Number of calls.
            input_tokens: Input tokens across the calls.
            output_tokens: Output tokens across the calls, thinking included.
        """
        self._write(
            {
                "event": "usage",
                "turn_id": turn_id,
                "session_id": session_id,
                "kind": kind,
                "model": model,
                "calls": calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def error(self, *, turn_id: str, session_id: str, kind: str, message: str) -> None:
        """Record something that failed without failing the turn, such as extraction.

        Hermes runs the provider's session-end work where a warning reaches no
        log file, so a timed-out extraction was invisible except as a missing
        write.

        Args:
            turn_id: The turn it belongs to.
            session_id: Hermes session it happened in.
            kind: What failed, e.g. ``extraction`` or ``session_outcome``.
            message: The error, as text.
        """
        self._write(
            {
                "event": "error",
                "turn_id": turn_id,
                "session_id": session_id,
                "kind": kind,
                "message": message,
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        """Append one record, swallowing and logging any failure."""
        if self._path is None:
            return

        record["ts"] = datetime.now(UTC).isoformat()
        line = json.dumps(record, ensure_ascii=False, default=str)

        try:
            with self._lock, self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            if not self._warned:
                logger.warning(f"Trace write failed ({exc}); further errors muted")
                self._warned = True


def summarize(path: Path) -> dict[str, Any]:
    """Totals from a trace file, for ``mem stats``.

    Args:
        path: The trace file.

    Returns:
        Recalls (count, by search mode, mean injected characters), outcomes (by
        origin, as events and as memories), and extraction usage (calls and
        tokens by model). Unreadable lines are skipped and counted.
    """
    recalls = 0
    block_chars = 0
    by_mode: dict[str, int] = {}
    outcomes: dict[str, dict[str, int]] = {}
    usage: dict[str, dict[str, int]] = {}
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            event = record.get("event")
            if event == "recall":
                recalls += 1
                block_chars += int(record.get("block_chars") or 0)
                mode = str(record.get("search_mode") or "unknown")
                by_mode[mode] = by_mode.get(mode, 0) + 1
            elif event == "outcome":
                origin = outcomes.setdefault(str(record.get("origin")), {"events": 0, "memories": 0})
                origin["events"] += 1
                origin["memories"] += len(record.get("memory_ids") or [])
            elif event == "usage":
                model = usage.setdefault(
                    str(record.get("model")), {"calls": 0, "input_tokens": 0, "output_tokens": 0}
                )
                for key in model:
                    model[key] += int(record.get(key) or 0)
    return {
        "recalls": recalls,
        "recalls_by_search_mode": by_mode,
        "mean_block_chars": block_chars / recalls if recalls else 0.0,
        "outcomes_by_origin": outcomes,
        "extraction_usage": usage,
        "skipped_lines": skipped,
    }

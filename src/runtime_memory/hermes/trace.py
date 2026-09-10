"""Retrieval and outcome tracing for the Hermes provider.

This is the instrumentation the Tier 3 evaluation runs on. Every recall writes
one JSONL record naming the memories it injected and their scores; every recorded
outcome writes another naming the memories it credited or blamed. Joining the two
on ``turn_id`` reconstructs, per turn, what was retrieved and whether it helped -
which is the measurement the outcome-learning claim needs.

Tracing is off unless a path is configured, and a broken trace never breaks a
turn: writes are best-effort and failures are logged once.
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
"""Environment variable holding the trace file path. Unset disables tracing."""


class TraceWriter:
    """Append-only JSONL writer for retrieval and outcome events."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize the writer.

        Args:
            path: Trace file path. Falls back to ``RUNTIME_MEMORY_HERMES_TRACE``;
                tracing is disabled when neither is set.
        """
        raw = path or os.environ.get(TRACE_ENV_VAR)
        self._path = Path(raw).expanduser() if raw else None
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
    ) -> None:
        """Record what a recall injected.

        Args:
            turn_id: Identifier joining this recall to any later outcome.
            session_id: Hermes session the recall belongs to.
            query: The query text used for retrieval.
            results: ``SearchResult`` objects that were injected.
            project: Project filter in force, if any.
            latency_ms: Wall-clock retrieval time.
        """
        self._write(
            {
                "event": "recall",
                "turn_id": turn_id,
                "session_id": session_id,
                "query": query,
                "project": project,
                "latency_ms": round(latency_ms, 2),
                "retrieved": [
                    {
                        "memory_id": r.memory.id,
                        "category": r.memory.category.value,
                        "score": round(r.score, 4),
                        "semantic_score": round(r.semantic_score, 4),
                        "outcome_score": round(r.memory.outcome_score, 4),
                        "use_count": r.memory.use_count,
                    }
                    for r in results
                ],
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
            origin: What produced the outcome, e.g. ``tool`` or ``auto``.
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

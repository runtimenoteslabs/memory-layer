"""Tool schemas exposed to Hermes, and their dispatch.

Hermes wants OpenAI function-calling schemas (``name``/``description``/
``parameters``), which is a different shape from memory-layer's MCP schemas, so
these are declared here rather than converted. The set is deliberately small: the
lifecycle hooks already handle recall and persistence, so the tools cover what the
model must ask for explicitly - deliberate saves, targeted lookups, outcome
feedback and a health check.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from memory_layer.core.models import MemoryCategory, Outcome

if TYPE_CHECKING:
    from memory_layer.hermes.provider import MemoryLayerProvider

CATEGORIES = [category.value for category in MemoryCategory]
OUTCOMES = [outcome.value for outcome in Outcome]

REMEMBER = {
    "name": "memorylayer_remember",
    "description": (
        "Save a durable fact to long-term memory: a convention, decision, "
        "gotcha, command or preference worth recalling in a later session. "
        "Do not use it for transient conversation detail."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to store, written to stand alone.",
            },
            "category": {
                "type": "string",
                "enum": CATEGORIES,
                "description": "Classification for the memory.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for later filtering.",
            },
        },
        "required": ["content", "category"],
    },
}

RECALL = {
    "name": "memorylayer_recall",
    "description": (
        "Search long-term memory for facts relevant to a query. Results are "
        "ranked by relevance and by how well each memory has worked before."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {
                "type": "integer",
                "description": "Maximum memories to return (default 10).",
                "minimum": 1,
                "maximum": 50,
            },
            "category": {
                "type": "string",
                "enum": CATEGORIES,
                "description": "Restrict results to one category.",
            },
        },
        "required": ["query"],
    },
}

OUTCOME = {
    "name": "memorylayer_outcome",
    "description": (
        "Report whether recalled memories actually helped. Call it once you "
        "know: 'worked' when the advice solved the problem, 'failed' when it "
        "was wrong or misleading, 'partial' when it helped a little. This is "
        "what teaches the store which memories to surface next time. With no "
        "memory_ids, it applies to the memories recalled for this turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": OUTCOMES,
                "description": "How the recalled memories performed.",
            },
            "memory_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific memories to score. Defaults to this turn's recall.",
            },
        },
        "required": ["outcome"],
    },
}

STATS = {
    "name": "memorylayer_stats",
    "description": "Report how many memories are stored and how they break down by category.",
    "parameters": {"type": "object", "properties": {}},
}

TOOL_SCHEMAS: list[dict[str, Any]] = [REMEMBER, RECALL, OUTCOME, STATS]

TOOL_NAMES = frozenset(schema["name"] for schema in TOOL_SCHEMAS)


def dispatch(provider: MemoryLayerProvider, tool_name: str, args: dict[str, Any]) -> str:
    """Route one tool call to the provider and serialize the result.

    Args:
        provider: The provider handling the call.
        tool_name: Which tool the model invoked.
        args: The model's arguments.

    Returns:
        A JSON string, per the Hermes tool contract. Errors are returned as
        ``{"error": ...}`` rather than raised, so a bad call costs a turn instead
        of the session.
    """
    handlers = {
        REMEMBER["name"]: _remember,
        RECALL["name"]: _recall,
        OUTCOME["name"]: _outcome,
        STATS["name"]: _stats,
    }
    handler = handlers.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        return json.dumps(handler(provider, args), ensure_ascii=False, default=str)
    except Exception as exc:  # surface the failure to the model, not the user
        return json.dumps({"error": str(exc)})


def _remember(provider: MemoryLayerProvider, args: dict[str, Any]) -> dict[str, Any]:
    content = (args.get("content") or "").strip()
    if not content:
        return {"error": "content is required"}

    raw_category = args.get("category") or MemoryCategory.GENERAL.value
    try:
        category = MemoryCategory(raw_category)
    except ValueError:
        return {"error": f"Unknown category '{raw_category}'. Valid: {CATEGORIES}"}

    memory = provider.remember(content=content, category=category, tags=args.get("tags") or [])
    return {
        "stored": True,
        "memory_id": memory.id,
        "category": memory.category.value,
        "project": memory.project,
    }


def _recall(provider: MemoryLayerProvider, args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "query is required"}

    raw_category = args.get("category")
    category = None
    if raw_category:
        try:
            category = MemoryCategory(raw_category)
        except ValueError:
            return {"error": f"Unknown category '{raw_category}'. Valid: {CATEGORIES}"}

    limit = args.get("limit") or 10
    results = provider.recall(query=query, limit=int(limit), category=category)
    return {
        "count": len(results),
        "memories": [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "category": r.memory.category.value,
                "score": round(r.score, 3),
                "outcome_score": round(r.memory.outcome_score, 3),
            }
            for r in results
        ],
    }


def _outcome(provider: MemoryLayerProvider, args: dict[str, Any]) -> dict[str, Any]:
    raw_outcome = args.get("outcome")
    try:
        outcome = Outcome(raw_outcome)
    except ValueError:
        return {"error": f"Unknown outcome '{raw_outcome}'. Valid: {OUTCOMES}"}

    memory_ids = args.get("memory_ids")
    updated = provider.record_outcome(outcome=outcome, memory_ids=memory_ids)
    if not updated:
        return {
            "recorded": False,
            "reason": "No memories to score - none were recalled this turn.",
        }
    return {
        "recorded": True,
        "outcome": outcome.value,
        "updated": [
            {"id": memory.id, "outcome_score": round(memory.outcome_score, 3)} for memory in updated
        ],
    }


def _stats(provider: MemoryLayerProvider, args: dict[str, Any]) -> dict[str, Any]:
    storage = provider.stats().storage_stats
    return {
        "total_memories": storage.total_memories,
        "active_memories": storage.active_memories,
        "avg_outcome_score": round(storage.avg_outcome_score, 3),
        "total_uses": storage.total_uses,
        "by_category": {
            key.value if hasattr(key, "value") else str(key): count
            for key, count in storage.by_category.items()
        },
    }

"""Memory Layer Plugin Module.

Claude Code 2.1.1+ integration utilities for the Memory Layer plugin.

This module provides:
- HookContext: Environment variable handling for native hooks
- ContextFormatter: Memory formatting for context injection
- SkillTriggers: Trigger detection for Agent Skills
- SessionManager: Native Claude Code session ID integration
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from memory_layer.core.models import Memory


@dataclass
class HookContext:
    """Context available during hook execution.

    Captures environment variables set by Claude Code during hook execution:
    - $CLAUDE_SESSION_ID: The current session identifier
    - $PWD: Current working directory (project path)
    - $TOOL_NAME: Name of the tool being used (for PostToolUse)
    - $TOOL_INPUT_FILE_PATH: File path from tool input (for Write/Edit hooks)
    """

    session_id: Optional[str] = None
    """Claude Code session ID from $CLAUDE_SESSION_ID."""

    project_path: str = ""
    """Current working directory from $PWD."""

    tool_name: Optional[str] = None
    """Tool name from $TOOL_NAME (PostToolUse hook)."""

    tool_input: Optional[dict[str, Any]] = None
    """Parsed tool input (if available)."""

    file_path: Optional[str] = None
    """File path from $TOOL_INPUT_FILE_PATH (Write/Edit hooks)."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the hook was triggered."""

    @classmethod
    def from_environment(cls) -> HookContext:
        """Create hook context from environment variables.

        Returns:
            HookContext populated from current environment.
        """
        return cls(
            session_id=os.environ.get("CLAUDE_SESSION_ID"),
            project_path=os.environ.get("PWD", str(Path.cwd())),
            tool_name=os.environ.get("TOOL_NAME"),
            tool_input=None,  # Would need JSON parsing from env
            file_path=os.environ.get("TOOL_INPUT_FILE_PATH"),
        )

    @property
    def has_session(self) -> bool:
        """Check if a session ID is available."""
        return self.session_id is not None and len(self.session_id) > 0

    @property
    def project_name(self) -> str:
        """Extract project name from path."""
        if self.project_path:
            return Path(self.project_path).name
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "project_path": self.project_path,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "file_path": self.file_path,
            "timestamp": self.timestamp.isoformat(),
        }


class ContextFormatter:
    """Format memories for context injection.

    Provides different formatting styles for injecting memories
    into Claude Code's context window.
    """

    STYLES = ("brief", "detailed", "structured", "markdown")

    @staticmethod
    def format_for_injection(
        memories: list[Memory],
        style: str = "brief",
        max_memories: int = 20,
    ) -> str:
        """Format memories for context window injection.

        Args:
            memories: List of Memory objects to format.
            style: Formatting style - "brief", "detailed", "structured", or "markdown".
            max_memories: Maximum number of memories to include.

        Returns:
            Formatted string for context injection.
        """
        if not memories:
            return ""

        memories = memories[:max_memories]

        if style == "brief":
            return ContextFormatter._format_brief(memories)
        elif style == "detailed":
            return ContextFormatter._format_detailed(memories)
        elif style == "structured":
            return ContextFormatter._format_structured(memories)
        elif style == "markdown":
            return ContextFormatter._format_markdown(memories)
        else:
            return ContextFormatter._format_brief(memories)

    @staticmethod
    def _format_brief(memories: list[Memory]) -> str:
        """Brief format: category + content on single lines."""
        lines = ["# Memory Context", ""]
        for m in memories:
            category = m.category.value if hasattr(m.category, "value") else str(m.category)
            lines.append(f"- [{category}] {m.content}")
        return "\n".join(lines)

    @staticmethod
    def _format_detailed(memories: list[Memory]) -> str:
        """Detailed format: includes scores and usage stats."""
        lines = ["# Memory Context (detailed)", ""]
        for m in memories:
            category = m.category.value if hasattr(m.category, "value") else str(m.category)
            lines.append(f"## [{m.id[:8]}] {category.upper()}")
            lines.append(f"{m.content}")
            lines.append(f"Score: {m.outcome_score:.2f} | Used: {m.use_count}x | Confidence: {m.confidence:.1f}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_structured(memories: list[Memory]) -> str:
        """Structured format: grouped by category."""
        by_category: dict[str, list[Memory]] = {}
        for m in memories:
            category = m.category.value if hasattr(m.category, "value") else str(m.category)
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(m)

        lines = ["# Memory Context", ""]
        for category, mems in sorted(by_category.items()):
            lines.append(f"## {category.replace('_', ' ').title()} ({len(mems)})")
            for m in mems:
                score_indicator = ""
                if m.outcome_score > 0.3:
                    score_indicator = " [proven]"
                elif m.outcome_score < -0.2:
                    score_indicator = " [questionable]"
                content = m.content[:100] + "..." if len(m.content) > 100 else m.content
                lines.append(f"- [{m.id[:8]}] {content}{score_indicator}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_markdown(memories: list[Memory]) -> str:
        """Markdown format: full markdown with headers."""
        by_category: dict[str, list[Memory]] = {}
        for m in memories:
            category = m.category.value if hasattr(m.category, "value") else str(m.category)
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(m)

        lines = ["# Project Knowledge", ""]

        # Priority order for categories
        priority_order = [
            "architecture", "decision", "convention", "pattern",
            "gotcha", "workaround", "troubleshooting", "command", "preference"
        ]

        # Sort categories by priority
        sorted_categories = sorted(
            by_category.keys(),
            key=lambda c: priority_order.index(c) if c in priority_order else 99
        )

        for category in sorted_categories:
            mems = by_category[category]
            # Sort by outcome score descending
            mems.sort(key=lambda m: m.outcome_score, reverse=True)

            lines.append(f"## {category.replace('_', ' ').title()}")
            lines.append("")
            for m in mems:
                confidence_note = ""
                if m.outcome_score > 0.3:
                    confidence_note = " *[high confidence]*"
                elif m.outcome_score < -0.2:
                    confidence_note = " *[low confidence]*"
                lines.append(f"- {m.content}{confidence_note}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_single_memory(
        memory: Memory,
        include_feedback_hint: bool = True,
    ) -> str:
        """Format a single memory for display.

        Args:
            memory: The memory to format.
            include_feedback_hint: Whether to include feedback command hint.

        Returns:
            Formatted string.
        """
        category = memory.category.value if hasattr(memory.category, "value") else str(memory.category)
        output = f"[{category}] {memory.content}"

        if include_feedback_hint:
            output += f"\n> Feedback: `/outcome {memory.id} worked|failed|partial`"

        return output

    @staticmethod
    def format_search_results(
        results: list[Any],  # SearchResult type
        include_scores: bool = True,
    ) -> str:
        """Format search results for display.

        Args:
            results: List of SearchResult objects.
            include_scores: Whether to include relevance scores.

        Returns:
            Formatted string.
        """
        if not results:
            return "No relevant memories found."

        lines = ["## Relevant Memories", ""]
        for i, r in enumerate(results, 1):
            m = r.memory
            category = m.category.value if hasattr(m.category, "value") else str(m.category)

            if include_scores:
                lines.append(f"{i}. [{category}] {m.content}")
                lines.append(f"   Score: {r.score:.2f} | Outcome: {m.outcome_score:.2f}")
            else:
                lines.append(f"{i}. [{category}] {m.content}")
            lines.append(f"   ID: {m.id}")
            lines.append("")

        lines.append("Use `/outcome <id> worked|failed|partial` to provide feedback.")
        return "\n".join(lines)


class SkillTriggers:
    """Detect when Agent Skills should activate.

    Agent Skills are automatically loaded by Claude based on conversation
    context. This class provides trigger detection for:
    - memory-retrieval: When user asks about past decisions/conventions
    - outcome-feedback: When user signals success/failure
    - coding-patterns: When user is creating new code
    """

    # Triggers for memory-retrieval skill
    RETRIEVAL_TRIGGERS = [
        "what did we decide",
        "how do we handle",
        "what's our convention",
        "what's the convention",
        "what's the pattern",
        "what pattern do we",
        "last time we",
        "we discussed",
        "as i mentioned",
        "as we discussed",
        "remember when",
        "what's the approach",
        "how should i",
        "how do i usually",
        "what's our standard",
        "what did we agree",
        "why did we choose",
        "what was the decision",
    ]

    # Triggers for coding-patterns skill
    PATTERN_TRIGGERS = [
        "create a new",
        "implement a",
        "implement the",
        "write a",
        "add a new",
        "build a",
        "make a new",
        "how should i structure",
        "what's the best way to",
        "how do i create",
        "scaffold a",
        "generate a",
        "set up a",
        "setup a",
    ]

    # Positive outcome signals
    OUTCOME_POSITIVE = [
        "thanks",
        "thank you",
        "that worked",
        "it worked",
        "works now",
        "perfect",
        "great",
        "excellent",
        "awesome",
        "solved",
        "fixed",
        "that's it",
        "exactly what i needed",
        "you're right",
    ]

    # Negative outcome signals
    OUTCOME_NEGATIVE = [
        "still not working",
        "doesn't work",
        "didn't work",
        "same error",
        "didn't help",
        "not helpful",
        "nope",
        "wrong",
        "that's wrong",
        "still broken",
        "still failing",
        "no luck",
        "try again",
        "that's not right",
    ]

    # Partial outcome signals
    OUTCOME_PARTIAL = [
        "kind of",
        "partially",
        "somewhat",
        "helped but",
        "almost",
        "close but",
        "partly",
        "half working",
        "better but",
    ]

    @classmethod
    def should_retrieve(cls, message: str) -> bool:
        """Check if memory retrieval skill should activate.

        Args:
            message: User message to analyze.

        Returns:
            True if retrieval triggers are detected.
        """
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in cls.RETRIEVAL_TRIGGERS)

    @classmethod
    def should_surface_patterns(cls, message: str) -> bool:
        """Check if coding patterns skill should activate.

        Args:
            message: User message to analyze.

        Returns:
            True if pattern triggers are detected.
        """
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in cls.PATTERN_TRIGGERS)

    @classmethod
    def detect_outcome_signal(cls, message: str) -> Optional[str]:
        """Detect if user is signaling outcome feedback.

        Args:
            message: User message to analyze.

        Returns:
            "worked", "failed", "partial", or None if no signal detected.
        """
        message_lower = message.lower()

        # Check partial first (more specific)
        if any(trigger in message_lower for trigger in cls.OUTCOME_PARTIAL):
            return "partial"

        # Then positive
        if any(trigger in message_lower for trigger in cls.OUTCOME_POSITIVE):
            return "worked"

        # Then negative
        if any(trigger in message_lower for trigger in cls.OUTCOME_NEGATIVE):
            return "failed"

        return None

    @classmethod
    def get_trigger_type(cls, message: str) -> Optional[str]:
        """Get the type of trigger detected in a message.

        Args:
            message: User message to analyze.

        Returns:
            "retrieval", "patterns", "outcome", or None.
        """
        if cls.should_retrieve(message):
            return "retrieval"
        if cls.should_surface_patterns(message):
            return "patterns"
        if cls.detect_outcome_signal(message):
            return "outcome"
        return None

    @classmethod
    def extract_query_keywords(cls, message: str) -> list[str]:
        """Extract relevant keywords from a message for memory search.

        Args:
            message: User message to extract keywords from.

        Returns:
            List of keywords for search query.
        """
        import re

        # Remove common words and trigger phrases
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "i", "me", "my", "we", "our",
            "you", "your", "it", "its", "this", "that", "these", "those",
            "what", "which", "who", "whom",
        }

        # Also remove trigger phrases
        message_lower = message.lower()
        for trigger in cls.RETRIEVAL_TRIGGERS + cls.PATTERN_TRIGGERS:
            message_lower = message_lower.replace(trigger, " ")

        # Extract words
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", message_lower)

        # Filter stop words and short words
        keywords = [
            w for w in words
            if w not in stop_words and len(w) > 2
        ]

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:10]  # Limit to 10 keywords


class SessionManager:
    """Manage memory sessions with Claude Code session IDs.

    Integrates with Claude Code's native session management:
    - Uses $CLAUDE_SESSION_ID environment variable
    - Links memory sessions to Claude Code sessions
    - Supports session resumption with /resume
    """

    def __init__(self, engine: Any = None):
        """Initialize the session manager.

        Args:
            engine: Optional MemoryEngine instance for session operations.
        """
        self.engine = engine
        self._current_session: Optional[str] = None
        self._session_memories: list[str] = []  # Memory IDs used in session
        self._session_start: Optional[datetime] = None

    @property
    def current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._current_session or os.environ.get("CLAUDE_SESSION_ID")

    @property
    def is_active(self) -> bool:
        """Check if a session is currently active."""
        return self._current_session is not None

    def start_session(self, claude_session_id: Optional[str] = None) -> str:
        """Start or resume a memory session.

        If claude_session_id is provided, links to that session.
        Otherwise uses $CLAUDE_SESSION_ID from environment.

        Args:
            claude_session_id: Optional Claude Code session ID.

        Returns:
            The session ID being used.
        """
        session_id = claude_session_id or os.environ.get("CLAUDE_SESSION_ID")

        if session_id:
            self._current_session = session_id
            self._session_start = datetime.now(UTC)
            self._session_memories = []

        return self._current_session or "default"

    def end_session(self, summarize: bool = False) -> Optional[dict[str, Any]]:
        """End the current session.

        Args:
            summarize: Whether to generate a session summary.

        Returns:
            Session summary dict if summarize=True, else None.
        """
        if not self._current_session:
            return None

        summary = None
        if summarize:
            summary = {
                "session_id": self._current_session,
                "start_time": self._session_start.isoformat() if self._session_start else None,
                "end_time": datetime.now(UTC).isoformat(),
                "memories_used": len(self._session_memories),
                "memory_ids": self._session_memories[:20],  # First 20
            }

        self._current_session = None
        self._session_start = None
        self._session_memories = []

        return summary

    def track_memory_use(self, memory_id: str) -> None:
        """Track that a memory was used in this session.

        Args:
            memory_id: The ID of the memory that was used.
        """
        if memory_id not in self._session_memories:
            self._session_memories.append(memory_id)

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics for the current session.

        Returns:
            Dictionary with session statistics.
        """
        return {
            "session_id": self._current_session,
            "is_active": self.is_active,
            "start_time": self._session_start.isoformat() if self._session_start else None,
            "memories_used": len(self._session_memories),
            "duration_seconds": (
                (datetime.now(UTC) - self._session_start).total_seconds()
                if self._session_start else 0
            ),
        }


def get_plugin_root() -> Path:
    """Get the plugin root directory.

    Checks for CLAUDE_PLUGIN_ROOT environment variable first,
    then searches up from the current file for .claude-plugin directory.

    Returns:
        Path to the plugin root directory.
    """
    # Check environment variable first (set by Claude Code)
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if plugin_root:
        return Path(plugin_root)

    # Search up for .claude-plugin directory
    current = Path(__file__).parent
    while current != current.parent:
        if (current / ".claude-plugin").exists():
            return current
        current = current.parent

    # Fall back to the memory-layer package root
    # Go up from plugin/__init__.py -> memory_layer -> src -> memory-layer
    return Path(__file__).parent.parent.parent.parent


def get_hook_context() -> HookContext:
    """Convenience function to get current hook context.

    Returns:
        HookContext from current environment.
    """
    return HookContext.from_environment()


__all__ = [
    "HookContext",
    "ContextFormatter",
    "SkillTriggers",
    "SessionManager",
    "get_plugin_root",
    "get_hook_context",
]

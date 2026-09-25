"""Extraction pipeline for Runtime Memory.

Extracts actionable memories from conversation transcripts using LLM analysis.

Features:
- Structured extraction via LLM prompts
- Category auto-detection
- Confidence and importance scoring
- Entity detection (files, functions, errors)
- Conflict detection with existing memories
- Rate limiting for API calls
- PII detection and filtering
- Prompt injection prevention
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from runtime_memory.core.logging import get_logger
from runtime_memory.core.models import Memory, MemoryCategory, MemorySource, RelationType

if TYPE_CHECKING:
    from runtime_memory.core.engine import MemoryEngine

logger = get_logger(__name__)


# =============================================================================
# Extraction Prompts
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to extract actionable, reusable knowledge from coding conversation transcripts.

Focus on extracting:
1. **Decisions made and their rationale** - WHY something was chosen, not just WHAT
2. **Patterns discovered or established** - Reusable approaches
3. **Gotchas and pitfalls encountered** - Things that caused problems
4. **Solutions to problems** - Error messages → fixes (troubleshooting)
5. **User preferences expressed** - Coding style, tool preferences
6. **Useful commands** - Shell, npm, docker commands that worked
7. **Architecture insights** - System design decisions
8. **Conventions** - Project-specific coding standards

Skip:
- Generic coding advice the AI would already know
- One-time specific fixes unlikely to recur
- Incomplete or abandoned approaches
- Information that's too vague to be actionable
- Personal information (names, emails, API keys, passwords)

Output JSON only. No markdown, no explanations."""

EXTRACTION_USER_PROMPT = """Extract actionable memories from this conversation transcript.

<transcript>
{transcript}
</transcript>
{stored_section}{recalled_section}
Return a JSON object with this exact structure:
{{
  "memories": [
    {{
      "content": "Clear, actionable statement of what was learned",
      "category": "one of: architecture, convention, decision, pattern, gotcha, workaround, troubleshooting, command, preference",
      "importance": 0.0 to 1.0 (how important/reusable is this),
      "confidence": 0.0 to 1.0 (how certain are we this is correct),
      "entities": ["list", "of", "relevant", "entities"],
      "tags": ["optional", "tags"],
      "rationale": "Brief explanation of why this is worth remembering",
      "relates_to": "handle of a stored memory this one updates, conflicts with or extends, or null",
      "relation": "updates, conflicts or extends, or null"
    }}
  ],
  "confirmed": ["handles of stored memories this conversation confirmed without adding to them"],
  "acted_on": [{{"handle": "handle of a recalled memory the agent acted on", "evidence": "short quote from the transcript"}}],
  "summary": "One sentence summary of the conversation"
}}

Guidelines:
- importance: 0.9+ for critical gotchas/decisions, 0.5-0.8 for useful patterns, 0.3-0.5 for minor preferences
- confidence: 0.9+ if explicitly stated and verified, 0.6-0.8 if implied, 0.3-0.5 if inferred
- entities: Include file names, function names, package names, error types
- Each memory should be self-contained and understandable without context
- relates_to, relation and confirmed refer to stored memories listed above; leave them null or empty when none are listed
- acted_on refers to recalled memories listed above; leave it empty when none are listed

Return ONLY the JSON object, no other text."""

STORED_MEMORIES_SECTION = """
These memories are already stored for this project, each with a handle:

<stored>
{stored}
</stored>

- Do not extract anything a stored memory already says, even in other words.
- If the conversation shows a stored memory is wrong or out of date, extract the correct version, set "relates_to" to that memory's handle, and set "relation" to "updates" if the new memory should replace it, or "conflicts" if the two contradict and the conversation does not settle which is right.
- If a new memory adds to a stored one without contradicting it, set "relates_to" to its handle and "relation" to "extends".
- If the conversation confirms a stored memory without adding anything, do not extract it again; put its handle in "confirmed".
"""

RECALLED_MEMORIES_SECTION = """
These memories were shown to the agent during this conversation, each with a handle:

<recalled>
{recalled}
</recalled>

- In "acted_on", list each recalled memory whose advice the agent followed in this conversation, with a short quote from the transcript that shows it.
- Leave out a recalled memory the agent was shown but did not follow, or went against.
"""

CONFLICT_DETECTION_PROMPT = """Compare these two memories and determine their relationship.

EXISTING MEMORY:
Content: {existing_content}
Category: {existing_category}

NEW MEMORY:
Content: {new_content}
Category: {new_category}

What is the relationship between these memories?

Return a JSON object:
{{
  "relationship": "one of: updates, extends, conflicts, unrelated",
  "confidence": 0.0 to 1.0,
  "explanation": "Brief explanation of your reasoning",
  "should_supersede": true or false (should new memory replace existing?)
}}

Definitions:
- updates: New memory provides updated information on the same topic (supersedes old)
- extends: New memory adds complementary information (both should exist)
- conflicts: Memories contradict each other (needs resolution)
- unrelated: Different topics entirely

Return ONLY the JSON object."""


# =============================================================================
# Entity Detection Patterns
# =============================================================================

ENTITY_PATTERNS: dict[str, list[str]] = {
    "file": [
        r'\b([\w/-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|rb|php|swift|kt|scala|vue|svelte|md|json|yaml|yml|toml|sql|sh|bash|zsh))\b',
        r'`([^`]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|rb|php))`',
    ],
    "module": [
        r'\bfrom\s+([\w.]+)\s+import\b',
        r'\bimport\s+([\w.]+)',
        r'\brequire\([\'"]([^\'"]+)[\'"]\)',
    ],
    "error": [
        r'\b([A-Z][a-zA-Z]*Error)\b',
        r'\b([A-Z][a-zA-Z]*Exception)\b',
        r'\b([A-Z][a-zA-Z]*Warning)\b',
    ],
    "function": [
        r'\bdef\s+(\w+)\s*\(',
        r'\bfunction\s+(\w+)\s*\(',
        r'\bconst\s+(\w+)\s*=\s*(?:async\s*)?\(',
        r'\b(\w+)\s*=\s*(?:async\s+)?function',
    ],
    "class": [
        r'\bclass\s+(\w+)',
    ],
    "command": [
        r'\$\s*([^\n]+)',
        r'```(?:bash|sh|shell|zsh)\n([^`]+)```',
        r'`(npm\s+\w+[^`]*)`',
        r'`(pip\s+\w+[^`]*)`',
        r'`(docker\s+\w+[^`]*)`',
        r'`(git\s+\w+[^`]*)`',
    ],
    "package": [
        r'\b(npm|pip|cargo|go)\s+install\s+([\w@/-]+)',
        r'"([\w@/-]+)":\s*"[\d^~]',
    ],
}

# PII detection patterns
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "api_key": re.compile(r'\b(?:sk-|pk-|api[_-]?key[_-]?)[a-zA-Z0-9]{20,}\b', re.IGNORECASE),
    "password": re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*[\'"]?([^\s\'"]+)', re.IGNORECASE),
    "token": re.compile(r'\b(?:token|secret)[_-]?[a-zA-Z0-9]{20,}\b', re.IGNORECASE),
    "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "phone": re.compile(r'\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
}

# Prompt injection patterns to detect and filter
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'ignore\s+(?:all\s+)?(?:previous|above)\s+instructions?', re.IGNORECASE),
    re.compile(r'disregard\s+(?:all\s+)?(?:previous|above)', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:a|an|in)', re.IGNORECASE),
    re.compile(r'new\s+instructions?\s*:', re.IGNORECASE),
    re.compile(r'system\s*:\s*you', re.IGNORECASE),
    re.compile(r'<\|(?:im_start|im_end|system|user|assistant)\|>', re.IGNORECASE),
    re.compile(r'\[INST\]|\[/INST\]', re.IGNORECASE),
]


# =============================================================================
# Data Classes
# =============================================================================

class ConflictRelationship(str, Enum):
    """Relationship between conflicting memories."""

    UPDATES = "updates"
    """New memory updates/supersedes old memory."""

    EXTENDS = "extends"
    """New memory extends/complements old memory."""

    CONFLICTS = "conflicts"
    """Memories conflict with each other."""

    UNRELATED = "unrelated"
    """Memories are about different topics."""


@dataclass
class ExtractedMemory:
    """A memory extracted from a transcript."""

    content: str
    """The extracted memory content."""

    category: MemoryCategory
    """Detected category."""

    importance: float = 0.5
    """Importance score (0.0 to 1.0)."""

    confidence: float = 0.5
    """Confidence score (0.0 to 1.0)."""

    entities: list[str] = field(default_factory=list)
    """Detected entities (files, functions, errors)."""

    tags: list[str] = field(default_factory=list)
    """Optional tags."""

    rationale: str = ""
    """Why this memory is worth remembering."""

    relates_to: str | None = None
    """Handle of the stored memory this one updates, conflicts with or extends,
    as the extraction call named it. Only meaningful against the stored memories
    that call was shown."""

    relation: str | None = None
    """``updates``, ``conflicts`` or ``extends``, with ``relates_to``."""

    def to_memory(self, project: str | None = None) -> Memory:
        """Convert to Memory dataclass.

        Args:
            project: Project scope.

        Returns:
            Memory instance.
        """
        return Memory(
            content=self.content,
            category=self.category,
            source=MemorySource.EXTRACTED,
            confidence=self.confidence,
            importance=self.importance,
            entities=self.entities,
            tags=self.tags,
            project=project,
            metadata={"rationale": self.rationale} if self.rationale else {},
        )


RELATION_OF_CONFLICT: dict[ConflictRelationship, RelationType] = {
    ConflictRelationship.UPDATES: RelationType.UPDATES,
    ConflictRelationship.EXTENDS: RelationType.EXTENDS,
    ConflictRelationship.CONFLICTS: RelationType.CONFLICTS_WITH,
}
"""How the extractor's labels are stored. ``unrelated`` is not a relationship."""


@dataclass
class ConflictResult:
    """Result of conflict detection between memories."""

    existing_id: str
    """ID of existing memory."""

    relationship: ConflictRelationship
    """Detected relationship."""

    confidence: float
    """Confidence in the relationship detection."""

    explanation: str
    """Explanation of the relationship."""

    should_supersede: bool
    """Whether new memory should supersede existing."""

    new_index: int | None = None
    """Position of the new memory in the extraction's memories. A result
    without one is not tied to any new memory, so it never supersedes."""


@dataclass
class ModelUsage:
    """What an extraction's model calls consumed."""

    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ExtractionResult:
    """Result of extracting memories from a transcript."""

    memories: list[ExtractedMemory]
    """Extracted memories."""

    summary: str
    """Summary of the conversation."""

    transcript_length: int
    """Length of the original transcript."""

    extraction_time_ms: float
    """Time taken for extraction."""

    conflicts: list[ConflictResult] = field(default_factory=list)
    """Detected conflicts with existing memories."""

    confirmed_ids: list[str] = field(default_factory=list)
    """Stored memories the conversation confirmed without adding to, which were
    therefore not extracted again."""

    acted_on: list[tuple[str, str]] = field(default_factory=list)
    """Recalled memories the call judged the agent followed, as (memory id,
    quoted evidence). Empty unless the call was shown the recalled memories."""

    usage: ModelUsage | None = None
    """Model calls and tokens this extraction used, including any conflict
    classifier calls. None when it made no call."""

    pii_removed: int = 0
    """Number of PII instances removed."""

    injection_attempts: int = 0
    """Number of potential injection attempts detected."""

    raw_response: str = ""
    """Raw LLM response for debugging."""

    error: str | None = None
    """Error message if extraction failed."""

    @property
    def success(self) -> bool:
        """Whether extraction was successful."""
        return self.error is None

    @property
    def memory_count(self) -> int:
        """Number of memories extracted."""
        return len(self.memories)


@dataclass
class ExtractionConfig:
    """Configuration for the extraction pipeline."""

    # LLM settings
    model: str = "claude-sonnet-5"
    """Model to use for extraction.

    Extraction is a bulk structured-output task, so it runs on a cheaper model
    than a reasoning workload would. Claude 4.x ids still resolve today but are
    a generation behind and will eventually retire.
    """

    max_tokens: int = 16000
    """Maximum tokens in the response, thinking included.

    Claude 5 models think by default, and thinking counts against this limit. At
    4,096, four of eleven extraction calls on Tier 3 sessions of 27,000 to 48,600
    input tokens ran out on 2026-09-24 and returned cut-off JSON, which stores
    nothing. 16,000 stays under the length at which the SDK requires streaming.
    """

    temperature: float = 0.1
    """Ignored since the move to Claude 5.

    ``temperature`` was removed on Claude 4.6 and later and the API rejects it
    with a 400. The field stays so existing configuration keeps loading, and it
    records the original intent of deterministic extraction, but it is no longer
    sent. Determinism now comes from the prompt and the schema.
    """

    # Rate limiting
    rate_limit_rpm: int = 50
    """Rate limit: requests per minute."""

    rate_limit_tpm: int = 100000
    """Rate limit: tokens per minute."""

    # Content limits
    max_transcript_length: int = 100000
    """Maximum transcript length in characters."""

    min_transcript_length: int = 100
    """Minimum transcript length to process."""

    # Filtering
    min_confidence: float = 0.3
    """Minimum confidence to keep a memory."""

    min_importance: float = 0.2
    """Minimum importance to keep a memory."""

    # Security
    enable_pii_filtering: bool = True
    """Whether to filter PII from transcripts."""

    enable_injection_detection: bool = True
    """Whether to detect prompt injection attempts."""

    # Conflict detection
    enable_conflict_detection: bool = True
    """Whether to detect conflicts with existing memories."""

    conflict_similarity_threshold: float = 0.7
    """Similarity threshold for potential conflicts."""

    stored_context_limit: int = 50
    """How many stored memories the extraction call is shown. 0 turns it off.

    The call is told not to extract what a stored memory already says, to name
    the stored memory a new one updates, conflicts with or extends, and to list
    the ones the conversation confirmed. That is dedup and conflict detection in
    the one call extraction already makes. Before, nothing stopped a session
    from storing a rule again in new words, and a store could hold five copies
    of each rule; and conflicts were found by one classifier call per candidate
    pair, among same-category memories sharing words, so restatements filed
    under another category were never compared. With this off, that pairwise
    classifier runs as before.

    A project with more stored memories than this shows the ones most relevant
    to the transcript."""


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute.
            tokens_per_minute: Maximum tokens per minute.
        """
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._request_times: list[float] = []
        self._token_counts: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 0) -> None:
        """Wait until request can proceed.

        Args:
            tokens: Estimated tokens for this request.
        """
        async with self._lock:
            now = time.time()
            minute_ago = now - 60

            # Clean old entries
            self._request_times = [t for t in self._request_times if t > minute_ago]
            self._token_counts = [(t, c) for t, c in self._token_counts if t > minute_ago]

            # Check request limit
            while len(self._request_times) >= self.rpm:
                sleep_time = self._request_times[0] - minute_ago
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = time.time()
                minute_ago = now - 60
                self._request_times = [t for t in self._request_times if t > minute_ago]

            # Check token limit
            current_tokens = sum(c for _, c in self._token_counts)
            while current_tokens + tokens > self.tpm:
                if self._token_counts:
                    sleep_time = self._token_counts[0][0] - minute_ago
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                now = time.time()
                minute_ago = now - 60
                self._token_counts = [(t, c) for t, c in self._token_counts if t > minute_ago]
                current_tokens = sum(c for _, c in self._token_counts)

            # Record this request
            self._request_times.append(now)
            if tokens > 0:
                self._token_counts.append((now, tokens))


# =============================================================================
# Main Extractor Class
# =============================================================================

class MemoryExtractor:
    """Extracts memories from conversation transcripts using LLM analysis."""

    # Category keywords for auto-detection fallback
    CATEGORY_KEYWORDS: ClassVar[dict[MemoryCategory, set[str]]] = {
        MemoryCategory.ARCHITECTURE: {
            "architecture", "design", "microservice", "monolith", "database",
            "system", "component", "layer", "api", "service",
        },
        MemoryCategory.CONVENTION: {
            "convention", "standard", "naming", "format", "style", "lint",
            "rule", "guideline", "best practice",
        },
        MemoryCategory.DECISION: {
            "decided", "chose", "choice", "decision", "why", "because",
            "rationale", "trade-off", "alternative",
        },
        MemoryCategory.PATTERN: {
            "pattern", "approach", "technique", "method", "way to",
            "how to", "idiom", "recipe",
        },
        MemoryCategory.GOTCHA: {
            "gotcha", "watch out", "careful", "warning", "trap", "pitfall",
            "don't", "avoid", "never", "beware", "caution", "caveat",
        },
        MemoryCategory.WORKAROUND: {
            "workaround", "hack", "temporary", "quick fix", "bypass",
            "until", "for now", "interim",
        },
        MemoryCategory.TROUBLESHOOTING: {
            "error", "exception", "bug", "fix", "crash", "fail", "issue",
            "traceback", "debug", "solve", "solution",
        },
        MemoryCategory.COMMAND: {
            "command", "cli", "terminal", "shell", "npm", "pip", "docker",
            "git", "run", "execute", "script",
        },
        MemoryCategory.PREFERENCE: {
            "prefer", "like", "want", "favorite", "always", "usually",
            "habit", "style",
        },
    }

    def __init__(
        self,
        config: ExtractionConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            config: Extraction configuration.
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var).
        """
        self.config = config or ExtractionConfig()
        self._api_key = api_key
        self._client: Any = None
        self._rate_limiter = RateLimiter(
            self.config.rate_limit_rpm,
            self.config.rate_limit_tpm,
        )
        # Reset by each extraction, added to by every model call it makes.
        self._usage = ModelUsage(model=self.config.model)

    def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package required for extraction. "
                    "Install with: pip install 'runtime-memory[phase1]'"
                ) from e

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

        return self._client

    # =========================================================================
    # Security: PII and Injection Detection
    # =========================================================================

    def detect_pii(self, text: str) -> list[tuple[str, str, int, int]]:
        """Detect PII in text.

        Args:
            text: Text to scan.

        Returns:
            List of (pii_type, matched_text, start, end) tuples.
        """
        findings: list[tuple[str, str, int, int]] = []
        for pii_type, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append((pii_type, match.group(), match.start(), match.end()))
        return findings

    def filter_pii(self, text: str) -> tuple[str, int]:
        """Remove PII from text.

        Args:
            text: Text to filter.

        Returns:
            Tuple of (filtered_text, count_removed).
        """
        findings = self.detect_pii(text)
        if not findings:
            return text, 0

        # Sort by position descending to replace from end
        findings.sort(key=lambda x: x[2], reverse=True)

        result = text
        for pii_type, _, start, end in findings:
            placeholder = f"[{pii_type.upper()}_REDACTED]"
            result = result[:start] + placeholder + result[end:]

        return result, len(findings)

    def detect_injection_attempts(self, text: str) -> list[str]:
        """Detect potential prompt injection attempts.

        Args:
            text: Text to scan.

        Returns:
            List of detected injection patterns.
        """
        attempts = []
        for pattern in INJECTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                attempts.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
        return attempts

    def sanitize_for_prompt(self, text: str) -> str:
        """Sanitize text for use in prompts.

        Args:
            text: Text to sanitize.

        Returns:
            Sanitized text.
        """
        # Escape any prompt-like patterns
        text = re.sub(r'<\|', '<｜', text)  # Use fullwidth vertical line
        text = re.sub(r'\|>', '｜>', text)
        text = re.sub(r'\[INST\]', '[inst]', text, flags=re.IGNORECASE)
        text = re.sub(r'\[/INST\]', '[/inst]', text, flags=re.IGNORECASE)
        return text

    # =========================================================================
    # Entity Detection
    # =========================================================================

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract entities from text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary mapping entity type to list of entities.
        """
        entities: dict[str, list[str]] = {}

        for entity_type, patterns in ENTITY_PATTERNS.items():
            found: set[str] = set()
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        # Multi-group pattern, take first non-empty
                        for group in match:
                            if group:
                                found.add(group.strip())
                                break
                    else:
                        found.add(match.strip())

            if found:
                entities[entity_type] = sorted(found)

        return entities

    def flatten_entities(self, entities: dict[str, list[str]]) -> list[str]:
        """Flatten entity dictionary to list.

        Args:
            entities: Entity dictionary.

        Returns:
            Flat list of unique entities.
        """
        all_entities: set[str] = set()
        for entity_list in entities.values():
            all_entities.update(entity_list)
        return sorted(all_entities)

    # =========================================================================
    # Category Detection
    # =========================================================================

    def detect_category(self, content: str) -> tuple[MemoryCategory, float]:
        """Detect memory category from content.

        Args:
            content: Memory content.

        Returns:
            Tuple of (category, confidence).
        """
        content_lower = content.lower()
        scores: dict[MemoryCategory, int] = {}

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            # Default to DECISION if no keywords match
            return MemoryCategory.DECISION, 0.3

        best_category = max(scores, key=lambda k: scores[k])
        max_score = scores[best_category]

        # Calculate confidence based on number of keyword matches
        confidence = min(0.4 + (max_score * 0.15), 0.9)

        return best_category, confidence

    def parse_category(self, category_str: str) -> MemoryCategory:
        """Parse category string to enum.

        Args:
            category_str: Category string from LLM.

        Returns:
            MemoryCategory enum.
        """
        category_str = category_str.lower().strip()
        try:
            return MemoryCategory(category_str)
        except ValueError:
            # Try to match partial
            for category in MemoryCategory:
                if category.value in category_str or category_str in category.value:
                    return category
            return MemoryCategory.DECISION

    # =========================================================================
    # LLM Extraction
    # =========================================================================

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        estimated_tokens: int = 1000,
    ) -> str:
        """Call the LLM with rate limiting.

        Args:
            system_prompt: System prompt.
            user_prompt: User prompt.
            estimated_tokens: Estimated tokens for rate limiting.

        Returns:
            LLM response text.
        """
        await self._rate_limiter.acquire(estimated_tokens)

        client = self._get_client()
        # No temperature: the parameter was removed on Claude 4.6 and later and
        # sending it returns a 400.
        response = await client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        usage = getattr(response, "usage", None)
        self._usage.calls += 1
        self._usage.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self._usage.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        if response.stop_reason == "refusal":
            raise ValueError("The model declined the request")
        if response.stop_reason == "max_tokens":
            logger.warning(
                f"Response hit max_tokens ({self.config.max_tokens}); its JSON may be cut off"
            )
        # Claude 5 models think by default, so the answer is not always the first
        # block: reading content[0] failed whenever a thinking block came first.
        return "".join(block.text for block in response.content if block.type == "text")

    def _parse_extraction_response(self, response: str) -> tuple[list[ExtractedMemory], str]:
        """Parse LLM extraction response.

        Args:
            response: Raw LLM response.

        Returns:
            Tuple of (memories, summary).
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            raise ValueError("No JSON found in response")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        memories: list[ExtractedMemory] = []
        raw_memories = data.get("memories", [])

        for raw in raw_memories:
            if not isinstance(raw, dict):
                continue

            content = raw.get("content", "").strip()
            if not content:
                continue

            # Parse category
            category_str = raw.get("category", "decision")
            category = self.parse_category(category_str)

            # Parse scores with validation
            importance = float(raw.get("importance", 0.5))
            importance = max(0.0, min(1.0, importance))

            confidence = float(raw.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            # Parse lists
            entities = raw.get("entities", [])
            if not isinstance(entities, list):
                entities = []
            entities = [str(e).strip() for e in entities if e]

            tags = raw.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).strip().lower() for t in tags if t]

            rationale = str(raw.get("rationale", "")).strip()

            # Checked against the stored memories later; null, blank or a type
            # that is not a string all mean "none".
            relates_to = raw.get("relates_to")
            relation = raw.get("relation")

            memories.append(ExtractedMemory(
                content=content,
                category=category,
                importance=importance,
                confidence=confidence,
                entities=entities,
                tags=tags,
                rationale=rationale,
                relates_to=relates_to.strip() if isinstance(relates_to, str) and relates_to.strip() else None,
                relation=relation.strip().lower() if isinstance(relation, str) and relation.strip() else None,
            ))

        summary = str(data.get("summary", "")).strip()
        return memories, summary

    async def _relations(
        self,
        memories: list[ExtractedMemory],
        response: str,
        handles: dict[str, Memory],
        existing_memories: list[Memory] | None,
    ) -> tuple[list[ConflictResult], list[str]]:
        """How the new memories relate to stored ones, and which were confirmed.

        Named by the extraction call itself when it was shown stored memories,
        found by the pairwise classifier otherwise, which confirms nothing.

        Args:
            memories: The extracted memories, in the order they will be stored.
            response: The extraction call's raw response.
            handles: The stored memories the call was shown, by handle.
            existing_memories: What the pairwise classifier compares against.

        Returns:
            The conflict results and the confirmed memory ids.
        """
        if handles:
            return self._relations_named(memories, handles), self._confirmed_ids(response, handles)
        if self.config.enable_conflict_detection and existing_memories:
            return await self._detect_conflicts(memories, existing_memories), []
        return [], []

    @staticmethod
    def _confirmed_ids(response: str, handles: dict[str, Memory]) -> list[str]:
        """The stored memories an extraction response listed as confirmed.

        Args:
            response: Raw LLM response, already known to hold valid JSON.
            handles: The stored memories the call was shown, by handle.

        Returns:
            Their ids, once each, in the order listed. Handles that were not
            shown, and a missing or malformed field, give nothing.
        """
        json_match = re.search(r'\{[\s\S]*\}', response)
        confirmed = json.loads(json_match.group()).get("confirmed", []) if json_match else []
        if not isinstance(confirmed, list):
            return []
        shown = (handles.get(str(handle).strip()) for handle in confirmed)
        return list(dict.fromkeys(memory.id for memory in shown if memory is not None))

    def _recalled_section(self, handles: dict[str, Memory]) -> str:
        """Render the memories the agent was shown, by handle."""
        if not handles:
            return ""
        lines = "\n".join(
            f"[{handle}] ({memory.category.value}) {self.sanitize_for_prompt(memory.content)}"
            for handle, memory in handles.items()
        )
        return RECALLED_MEMORIES_SECTION.format(recalled=lines)

    @staticmethod
    def _acted_on(response: str, handles: dict[str, Memory]) -> list[tuple[str, str]]:
        """The recalled memories a response says the agent followed.

        Args:
            response: Raw LLM response, already known to hold valid JSON.
            handles: The recalled memories the call was shown, by handle.

        Returns:
            (memory id, evidence) pairs, once per memory, in the order listed.
            Handles that were not shown, and a missing or malformed field, give
            nothing.
        """
        if not handles:
            return []
        json_match = re.search(r'\{[\s\S]*\}', response)
        listed = json.loads(json_match.group()).get("acted_on", []) if json_match else []
        if not isinstance(listed, list):
            return []
        found: dict[str, str] = {}
        for entry in listed:
            handle, evidence = (
                (entry.get("handle"), entry.get("evidence", "")) if isinstance(entry, dict) else (entry, "")
            )
            memory = handles.get(str(handle).strip())
            if memory is not None and memory.id not in found:
                found[memory.id] = str(evidence).strip()
        return list(found.items())

    def _stored_section(self, handles: dict[str, Memory]) -> str:
        """Render the stored memories an extraction call is shown, by handle."""
        if not handles:
            return ""
        lines = "\n".join(
            f"[{handle}] ({memory.category.value}) {self.sanitize_for_prompt(memory.content)}"
            for handle, memory in handles.items()
        )
        return STORED_MEMORIES_SECTION.format(stored=lines)

    @staticmethod
    def _relations_named(
        memories: list[ExtractedMemory], handles: dict[str, Memory]
    ) -> list[ConflictResult]:
        """Turn the relations an extraction call named into conflict results.

        A handle that was not shown, or a relation other than updates, conflicts
        or extends, is dropped and logged: it cannot be tied to a stored memory.

        Args:
            memories: The extracted memories, in the order they will be stored.
            handles: The stored memories the call was shown, by handle.

        Returns:
            One result per usable relation, carrying the new memory's position.
        """
        named: list[ConflictResult] = []
        allowed = {
            ConflictRelationship.UPDATES.value,
            ConflictRelationship.CONFLICTS.value,
            ConflictRelationship.EXTENDS.value,
        }
        for index, memory in enumerate(memories):
            if memory.relates_to is None and memory.relation is None:
                continue
            stored = handles.get(memory.relates_to or "")
            if stored is None or memory.relation not in allowed:
                logger.warning(
                    f"Extraction named relation {memory.relation!r} to "
                    f"{memory.relates_to!r}, which matches no stored memory shown; ignored"
                )
                continue
            relationship = ConflictRelationship(memory.relation)
            named.append(ConflictResult(
                existing_id=stored.id,
                relationship=relationship,
                confidence=memory.confidence,
                explanation="named by the extraction call",
                should_supersede=relationship == ConflictRelationship.UPDATES,
                new_index=index,
            ))
        return named

    async def extract_from_transcript(
        self,
        transcript: str,
        project: str | None = None,
        existing_memories: list[Memory] | None = None,
        stored_memories: list[Memory] | None = None,
        recalled_memories: list[Memory] | None = None,
    ) -> ExtractionResult:
        """Extract memories from a conversation transcript.

        Args:
            transcript: The conversation transcript.
            project: Optional project context.
            existing_memories: Memories for the pairwise conflict classifier, which
                runs only when ``stored_memories`` is not given.
            stored_memories: Memories to show the extraction call, so it leaves
                out what they already say and names what it updates, conflicts
                with, extends or confirms. See ``stored_context_limit``.
            recalled_memories: Memories the agent was shown during the
                conversation, so the call can name the ones it followed.

        Returns:
            ExtractionResult with extracted memories.
        """
        start_time = time.time()
        transcript_length = len(transcript)
        self._usage = ModelUsage(model=self.config.model)

        # Validate transcript length
        if transcript_length < self.config.min_transcript_length:
            return ExtractionResult(
                memories=[],
                summary="Transcript too short for extraction",
                transcript_length=transcript_length,
                extraction_time_ms=0,
                error="Transcript too short",
            )

        if transcript_length > self.config.max_transcript_length:
            # Truncate to max length
            transcript = transcript[:self.config.max_transcript_length]
            logger.warning(f"Transcript truncated from {transcript_length} to {self.config.max_transcript_length}")

        # Security: Detect injection attempts
        injection_attempts = 0
        if self.config.enable_injection_detection:
            attempts = self.detect_injection_attempts(transcript)
            injection_attempts = len(attempts)
            if injection_attempts > 0:
                logger.warning(f"Detected {injection_attempts} potential injection attempts")

        # Security: Filter PII
        pii_removed = 0
        if self.config.enable_pii_filtering:
            transcript, pii_removed = self.filter_pii(transcript)
            if pii_removed > 0:
                logger.info(f"Removed {pii_removed} PII instances")

        # Sanitize transcript for prompt
        transcript = self.sanitize_for_prompt(transcript)

        # Build prompt. Short handles, not ids, stand for the stored memories:
        # fewer tokens, and nothing for the model to miscopy.
        handles = {f"S{i}": memory for i, memory in enumerate(stored_memories or [], start=1)}
        stored_section = self._stored_section(handles)
        recalled = {f"R{i}": memory for i, memory in enumerate(recalled_memories or [], start=1)}
        recalled_section = self._recalled_section(recalled)
        user_prompt = EXTRACTION_USER_PROMPT.format(
            transcript=transcript,
            stored_section=stored_section,
            recalled_section=recalled_section,
        )
        estimated_tokens = (len(transcript) + len(stored_section) + len(recalled_section)) // 4 + 1000

        try:
            # Call LLM
            response = await self._call_llm(
                EXTRACTION_SYSTEM_PROMPT,
                user_prompt,
                estimated_tokens,
            )

            # Parse response
            memories, summary = self._parse_extraction_response(response)

            # Filter by minimum thresholds
            memories = [
                m for m in memories
                if m.confidence >= self.config.min_confidence
                and m.importance >= self.config.min_importance
            ]

            # Enhance with additional entity detection
            for memory in memories:
                detected = self.extract_entities(memory.content)
                existing_entities = set(memory.entities)
                for entity_list in detected.values():
                    for entity in entity_list:
                        if entity not in existing_entities:
                            memory.entities.append(entity)

            conflicts, confirmed_ids = await self._relations(
                memories, response, handles, existing_memories
            )

            extraction_time = (time.time() - start_time) * 1000

            return ExtractionResult(
                memories=memories,
                summary=summary,
                transcript_length=transcript_length,
                extraction_time_ms=extraction_time,
                conflicts=conflicts,
                confirmed_ids=confirmed_ids,
                acted_on=self._acted_on(response, recalled),
                usage=self._usage_so_far(),
                pii_removed=pii_removed,
                injection_attempts=injection_attempts,
                raw_response=response,
            )

        except Exception as e:
            extraction_time = (time.time() - start_time) * 1000
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(
                memories=[],
                summary="",
                transcript_length=transcript_length,
                extraction_time_ms=extraction_time,
                usage=self._usage_so_far(),
                pii_removed=pii_removed,
                injection_attempts=injection_attempts,
                error=str(e),
            )

    def _usage_so_far(self) -> ModelUsage | None:
        """This extraction's usage, or None when it made no model call."""
        return replace(self._usage) if self._usage.calls else None

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    async def _detect_conflicts(
        self,
        new_memories: list[ExtractedMemory],
        existing_memories: list[Memory],
    ) -> list[ConflictResult]:
        """Detect conflicts between new and existing memories.

        Args:
            new_memories: Newly extracted memories.
            existing_memories: Existing memories to check against.

        Returns:
            List of conflict results.
        """
        conflicts: list[ConflictResult] = []

        for index, new_mem in enumerate(new_memories):
            # Find potentially related existing memories
            for existing in existing_memories:
                # Quick category match check
                if new_mem.category != existing.category:
                    continue

                # Check for entity overlap
                new_entities = set(new_mem.entities)
                existing_entities = set(existing.entities)
                if not new_entities.intersection(existing_entities):
                    # No entity overlap, check content similarity heuristic
                    if not self._content_similar(new_mem.content, existing.content):
                        continue

                # Potential conflict found, use LLM to classify
                try:
                    conflict = await self._classify_conflict(new_mem, existing)
                    if conflict.relationship != ConflictRelationship.UNRELATED:
                        conflict.new_index = index
                        conflicts.append(conflict)
                except Exception as e:
                    logger.warning(f"Conflict classification failed: {e}")

        return conflicts

    def _content_similar(self, content1: str, content2: str) -> bool:
        """Quick heuristic check for content similarity.

        Args:
            content1: First content.
            content2: Second content.

        Returns:
            True if contents appear similar.
        """
        # Simple word overlap check
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1.intersection(words2))
        min_len = min(len(words1), len(words2))

        return overlap / min_len >= 0.3

    async def _classify_conflict(
        self,
        new_memory: ExtractedMemory,
        existing: Memory,
    ) -> ConflictResult:
        """Classify the relationship between new and existing memory.

        Args:
            new_memory: New extracted memory.
            existing: Existing memory.

        Returns:
            ConflictResult with classification.
        """
        prompt = CONFLICT_DETECTION_PROMPT.format(
            existing_content=existing.content,
            existing_category=existing.category.value,
            new_content=new_memory.content,
            new_category=new_memory.category.value,
        )

        response = await self._call_llm(
            "You are a memory conflict analyzer. Classify relationships between memories.",
            prompt,
            500,
        )

        # Parse response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            raise ValueError("No JSON in conflict response")

        data = json.loads(json_match.group())

        relationship_str = data.get("relationship", "unrelated").lower()
        try:
            relationship = ConflictRelationship(relationship_str)
        except ValueError:
            relationship = ConflictRelationship.UNRELATED

        return ConflictResult(
            existing_id=existing.id,
            relationship=relationship,
            confidence=float(data.get("confidence", 0.5)),
            explanation=str(data.get("explanation", "")),
            should_supersede=bool(data.get("should_supersede", False)),
        )

    # =========================================================================
    # High-Level API
    # =========================================================================

    STORED_CONTEXT_QUERY_CHARS: ClassVar[int] = 20000
    """How much of a transcript ranks stored memories when a project has more
    than the call can be shown. BM25 tokenises the query once per memory."""

    async def _stored_context(
        self, engine: MemoryEngine, transcript: str, project: str | None
    ) -> list[Memory]:
        """Choose the stored memories an extraction call is shown.

        All of the project's live memories when they fit in
        ``stored_context_limit``, otherwise the ones most relevant to the
        transcript. Ranked through the retriever directly, so choosing them
        neither counts as a use nor becomes the engine's last search.

        Args:
            engine: The engine the memories are stored in.
            transcript: The conversation about to be extracted.
            project: Its project.

        Returns:
            Up to ``stored_context_limit`` memories.
        """
        limit = self.config.stored_context_limit
        memories = await engine.list(project=project, limit=limit + 1)
        if len(memories) <= limit:
            return memories
        hits = await engine.retriever.search(
            query=transcript[: self.STORED_CONTEXT_QUERY_CHARS], limit=limit, project=project
        )
        return [hit.memory for hit in hits]

    async def extract_and_store(
        self,
        transcript: str,
        engine: MemoryEngine,
        project: str | None = None,
        recalled: list[Memory] | None = None,
    ) -> ExtractionResult:
        """Extract memories and store them in the engine.

        Args:
            transcript: Conversation transcript.
            engine: Memory engine to store in.
            project: Project scope.
            recalled: Memories the agent was shown during the conversation. The
                result's ``acted_on`` names the ones the call judged it followed;
                recording an outcome for them is left to the caller.

        Returns:
            ExtractionResult with stored memories.
        """
        # What the call is shown, or failing that what the pairwise classifier
        # compares against.
        stored: list[Memory] = []
        existing_memories: list[Memory] = []
        if self.config.stored_context_limit > 0:
            stored = await self._stored_context(engine, transcript, project)
        elif self.config.enable_conflict_detection:
            existing_memories = await engine.list(project=project, limit=100)

        # Extract
        result = await self.extract_from_transcript(
            transcript=transcript,
            project=project,
            existing_memories=existing_memories,
            stored_memories=stored or None,
            recalled_memories=recalled or None,
        )

        if not result.success:
            return result

        for memory_id in result.confirmed_ids:
            await engine.confirm(memory_id)

        if not result.memories:
            return result

        # Handle conflicts and store
        for index, memory in enumerate(result.memories):
            # Only a conflict classified for this memory can make it supersede
            # another. Taking any conflict in the batch pointed every new memory
            # at the same superseded one.
            supersedes_id = next(
                (
                    conflict.existing_id
                    for conflict in result.conflicts
                    if conflict.new_index == index
                    and conflict.should_supersede
                    and conflict.relationship == ConflictRelationship.UPDATES
                ),
                None,
            )

            # Store the memory
            stored = await engine.add(
                content=memory.content,
                category=memory.category,
                project=project,
                source=MemorySource.EXTRACTED,
                confidence=memory.confidence,
                importance=memory.importance,
                entities=memory.entities,
                tags=memory.tags,
                supersedes=supersedes_id,
                metadata={"rationale": memory.rationale} if memory.rationale else {},
            )

            # Keep what the classifier found. Before, everything but a supersede
            # was discarded, so a memory that contradicted a stored one was left
            # to be out-ranked by it, and nothing could tell the two apart later.
            for conflict in result.conflicts:
                if conflict.new_index != index:
                    continue
                relation = RELATION_OF_CONFLICT.get(conflict.relationship)
                if relation is None:
                    continue
                await engine.link(
                    stored.id,
                    conflict.existing_id,
                    relation,
                    strength=conflict.confidence,
                    metadata={"source": "extraction", "explanation": conflict.explanation},
                )

        logger.info(f"Stored {len(result.memories)} memories from extraction")
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

async def extract_from_transcript(
    transcript: str,
    project: str | None = None,
    config: ExtractionConfig | None = None,
    api_key: str | None = None,
) -> ExtractionResult:
    """Extract memories from a transcript.

    Convenience function for one-off extraction.

    Args:
        transcript: Conversation transcript.
        project: Project scope.
        config: Extraction configuration.
        api_key: Anthropic API key.

    Returns:
        ExtractionResult.
    """
    extractor = MemoryExtractor(config=config, api_key=api_key)
    return await extractor.extract_from_transcript(transcript, project=project)


def detect_entities(text: str) -> dict[str, list[str]]:
    """Detect entities in text.

    Convenience function for entity detection without LLM.

    Args:
        text: Text to analyze.

    Returns:
        Dictionary of entity types to entity lists.
    """
    extractor = MemoryExtractor()
    return extractor.extract_entities(text)


def detect_pii(text: str) -> list[tuple[str, str, int, int]]:
    """Detect PII in text.

    Convenience function for PII detection.

    Args:
        text: Text to scan.

    Returns:
        List of (pii_type, matched_text, start, end) tuples.
    """
    extractor = MemoryExtractor()
    return extractor.detect_pii(text)


def filter_pii(text: str) -> tuple[str, int]:
    """Filter PII from text.

    Convenience function for PII filtering.

    Args:
        text: Text to filter.

    Returns:
        Tuple of (filtered_text, count_removed).
    """
    extractor = MemoryExtractor()
    return extractor.filter_pii(text)

"""Data models for Memory Layer.

This module defines the core data structures:
- Enums for categories, scopes, sources, and outcomes
- Dataclasses for Memory, SearchResult, and ContextResponse
- Pydantic models for validation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryCategory(str, Enum):
    """Categories for memory classification.

    Each category represents a different type of knowledge that can be stored.
    """

    # Project Knowledge
    ARCHITECTURE = "architecture"
    """System design decisions (microservices, database choices)."""

    CONVENTION = "convention"
    """Coding standards (naming, formatting, style)."""

    DECISION = "decision"
    """Why something was built this way (rationale)."""

    DEPENDENCY = "dependency"
    """What relies on what (v2 addition)."""

    # Implementation Knowledge
    PATTERN = "pattern"
    """Reusable code patterns (repository, factory)."""

    GOTCHA = "gotcha"
    """Non-obvious behaviors and traps to avoid."""

    WORKAROUND = "workaround"
    """Temporary fixes (with context on why)."""

    # Operational Knowledge
    TROUBLESHOOTING = "troubleshooting"
    """Error → solution mappings."""

    COMMAND = "command"
    """Useful shell/npm/etc. commands."""

    ENVIRONMENT = "environment"
    """Setup, config, secrets locations (v2 addition)."""

    # User Preferences
    PREFERENCE = "preference"
    """User preferences (style, tools)."""

    CODING_STYLE = "coding_style"
    """Tabs vs spaces, naming conventions (v2 addition)."""

    TOOL_PREFERENCE = "tool_preference"
    """Preferred libraries, frameworks (v2 addition)."""

    # Meta
    CONTEXT = "context"
    """Current work state (v2 addition)."""

    TODO = "todo"
    """Pending items (v2 addition)."""

    # General
    GENERAL = "general"
    """Uncategorized memories (v2 addition)."""


class MemoryScope(str, Enum):
    """Scope levels for memory visibility.

    Determines where a memory is applicable.
    """

    GLOBAL = "global"
    """Available across all projects."""

    PROJECT = "project"
    """Specific to a single project."""

    SESSION = "session"
    """Only valid for the current session."""


class MemorySource(str, Enum):
    """Source of the memory.

    Indicates how the memory was created.
    """

    EXPLICIT = "explicit"
    """Manually added by user via /remember or API."""

    EXTRACTED = "extracted"
    """Automatically extracted from conversation."""

    IMPORTED = "imported"
    """Imported from external source (file, task system)."""


class Outcome(str, Enum):
    """Outcome feedback for a memory.

    Used to adjust the memory's outcome_score.
    """

    WORKED = "worked"
    """Advice solved the problem. Score adjustment: +0.2"""

    FAILED = "failed"
    """Advice was wrong. Score adjustment: -0.3"""

    PARTIAL = "partial"
    """Advice was on the right track but incomplete. Score adjustment: +0.05"""


# Score adjustments for each outcome
OUTCOME_SCORE_ADJUSTMENTS: dict[Outcome, float] = {
    Outcome.WORKED: 0.2,
    Outcome.FAILED: -0.3,
    Outcome.PARTIAL: 0.05,
}


class RelationType(str, Enum):
    """Types of relationships between memories (v2 addition).

    Used to track how memories relate to each other.
    """

    UPDATES = "updates"
    """New info replaces old (e.g., new decision supersedes old)."""

    EXTENDS = "extends"
    """Enriches without replacing (e.g., adds detail to existing memory)."""

    DERIVES = "derives"
    """Inferred connection (e.g., pattern derived from multiple examples)."""

    RELATES_TO = "relates_to"
    """Soft connection (e.g., same topic, related concepts)."""

    CONFLICTS_WITH = "conflicts_with"
    """Contradictory information (requires resolution)."""


class EntityType(str, Enum):
    """Types of entities that can be extracted from memories (v2 addition)."""

    FILE = "file"
    """A file path or file name."""

    MODULE = "module"
    """A code module or package."""

    FUNCTION = "function"
    """A function or method name."""

    CLASS = "class"
    """A class name."""

    VARIABLE = "variable"
    """A variable or constant name."""

    ERROR = "error"
    """An error message or error type."""

    CONCEPT = "concept"
    """An abstract concept or pattern name."""

    TOOL = "tool"
    """A tool, library, or framework."""

    PERSON = "person"
    """A person's name (e.g., team member)."""


def generate_id() -> str:
    """Generate a unique memory ID.

    Returns:
        UUID string.
    """
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Get current UTC timestamp.

    Returns:
        Current datetime in UTC.
    """
    return datetime.now(UTC)


@dataclass
class Memory:
    """Core memory data structure.

    Represents a single piece of stored knowledge with metadata
    for retrieval and outcome-based learning.
    """

    content: str
    """The memory content text."""

    category: MemoryCategory
    """Classification category."""

    id: str = field(default_factory=generate_id)
    """Unique identifier (UUID)."""

    outcome_score: float = 0.0
    """Accumulated outcome score (-1.0 to 1.0)."""

    confidence: float = 1.0
    """Source reliability (0.0 to 1.0)."""

    importance: float = 0.5
    """Importance weight (0.0 to 1.0)."""

    use_count: int = 0
    """Times retrieved in search."""

    project: str | None = None
    """Project scope (None for global)."""

    scope: MemoryScope = MemoryScope.PROJECT
    """Visibility scope."""

    source: MemorySource = MemorySource.EXPLICIT
    """How the memory was created."""

    tags: list[str] = field(default_factory=list)
    """Optional tags for additional categorization."""

    entities: list[str] = field(default_factory=list)
    """Detected entities (files, functions, errors)."""

    supersedes: str | None = None
    """ID of memory this one replaces."""

    archived: bool = False
    """Whether the memory is archived (soft deleted)."""

    created_at: datetime = field(default_factory=utc_now)
    """Creation timestamp."""

    updated_at: datetime = field(default_factory=utc_now)
    """Last update timestamp."""

    embedding: list[float] | None = None
    """Vector embedding for semantic search."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def apply_outcome(self, outcome: Outcome) -> None:
        """Apply an outcome to adjust the score.

        Args:
            outcome: The outcome to apply.
        """
        adjustment = OUTCOME_SCORE_ADJUSTMENTS[outcome]
        self.outcome_score = max(-1.0, min(1.0, self.outcome_score + adjustment))
        self.updated_at = utc_now()

    def increment_use_count(self) -> None:
        """Increment the use count when memory is retrieved."""
        self.use_count += 1
        self.updated_at = utc_now()

    def archive(self) -> None:
        """Archive the memory (soft delete)."""
        self.archived = True
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all fields.
        """
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category.value,
            "outcome_score": self.outcome_score,
            "confidence": self.confidence,
            "importance": self.importance,
            "use_count": self.use_count,
            "project": self.project,
            "scope": self.scope.value,
            "source": self.source.value,
            "tags": self.tags,
            "entities": self.entities,
            "supersedes": self.supersedes,
            "archived": self.archived,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Create Memory from dictionary.

        Args:
            data: Dictionary with memory fields.

        Returns:
            Memory instance.
        """
        # Handle datetime conversion
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = utc_now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = utc_now()

        # Handle enum conversion
        category_raw = data.get("category")
        if isinstance(category_raw, str):
            category = MemoryCategory(category_raw)
        elif isinstance(category_raw, MemoryCategory):
            category = category_raw
        else:
            raise ValueError("category is required and must be a valid MemoryCategory")

        scope = data.get("scope", MemoryScope.PROJECT)
        if isinstance(scope, str):
            scope = MemoryScope(scope)

        source = data.get("source", MemorySource.EXPLICIT)
        if isinstance(source, str):
            source = MemorySource(source)

        return cls(
            id=data.get("id", generate_id()),
            content=data["content"],
            category=category,
            outcome_score=data.get("outcome_score", 0.0),
            confidence=data.get("confidence", 1.0),
            importance=data.get("importance", 0.5),
            use_count=data.get("use_count", 0),
            project=data.get("project"),
            scope=scope,
            source=source,
            tags=data.get("tags", []),
            entities=data.get("entities", []),
            supersedes=data.get("supersedes"),
            archived=data.get("archived", False),
            created_at=created_at,
            updated_at=updated_at,
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """Result from a memory search operation.

    Contains the memory along with relevance scoring details.
    """

    memory: Memory
    """The matched memory."""

    score: float
    """Final combined relevance score."""

    semantic_score: float = 0.0
    """Semantic similarity component (BM25 + vector)."""

    recency_score: float = 0.0
    """Recency decay component."""

    frequency_score: float = 0.0
    """Usage frequency component."""

    category_boost: float = 1.0
    """Category-specific boost factor."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all fields.
        """
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "semantic_score": self.semantic_score,
            "recency_score": self.recency_score,
            "frequency_score": self.frequency_score,
            "category_boost": self.category_boost,
        }


@dataclass
class ContextResponse:
    """Response containing formatted context for injection.

    Used to provide relevant memories as context to an AI agent.
    """

    memories: list[Memory]
    """List of relevant memories."""

    project: str | None
    """Project scope for the context."""

    total_count: int
    """Total number of memories available."""

    included_count: int
    """Number of memories included in response."""

    formatted: str = ""
    """Pre-formatted context string for injection."""

    categories: dict[str, int] = field(default_factory=dict)
    """Count of memories by category."""

    def to_markdown(self) -> str:
        """Format context as markdown for injection.

        Returns:
            Markdown-formatted context string.
        """
        if not self.memories:
            return "No relevant memories found."

        lines = ["# Project Knowledge", ""]

        # Group by category
        by_category: dict[MemoryCategory, list[Memory]] = {}
        for memory in self.memories:
            if memory.category not in by_category:
                by_category[memory.category] = []
            by_category[memory.category].append(memory)

        # Format each category
        for category in MemoryCategory:
            if category in by_category:
                memories = by_category[category]
                lines.append(f"## {category.value.title()}")
                lines.append("")
                for memory in memories:
                    score_indicator = ""
                    if memory.outcome_score > 0.3:
                        score_indicator = " [high confidence]"
                    elif memory.outcome_score < -0.3:
                        score_indicator = " [low confidence]"
                    lines.append(f"- {memory.content}{score_indicator}")
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all fields.
        """
        return {
            "memories": [m.to_dict() for m in self.memories],
            "project": self.project,
            "total_count": self.total_count,
            "included_count": self.included_count,
            "formatted": self.formatted or self.to_markdown(),
            "categories": self.categories,
        }


# =============================================================================
# V2 Model Additions: Relationships, Entities, Routing Patterns
# =============================================================================


@dataclass
class Relationship:
    """A relationship between two memories (v2 addition).

    Tracks how memories relate to each other, enabling knowledge graph
    capabilities and conflict detection.
    """

    source_id: str
    """ID of the source memory."""

    target_id: str
    """ID of the target memory."""

    relation_type: RelationType
    """Type of relationship."""

    strength: float = 1.0
    """Relationship strength (0.0 to 1.0)."""

    created_at: datetime = field(default_factory=utc_now)
    """When the relationship was created."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional relationship metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        """Create Relationship from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = utc_now()

        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            strength=data.get("strength", 1.0),
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Entity:
    """An entity extracted from memories (v2 addition).

    Represents a named entity (file, module, function, concept, etc.)
    that can be referenced by multiple memories.
    """

    id: str = field(default_factory=generate_id)
    """Unique identifier for the entity."""

    name: str = ""
    """The entity name (e.g., 'auth_service.py', 'UserRepository')."""

    entity_type: EntityType = EntityType.CONCEPT
    """Type of entity."""

    description: str = ""
    """Optional description of the entity."""

    memory_ids: list[str] = field(default_factory=list)
    """IDs of memories that reference this entity."""

    created_at: datetime = field(default_factory=utc_now)
    """When the entity was first detected."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional entity metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "memory_ids": self.memory_ids,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Create Entity from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = utc_now()

        return cls(
            id=data.get("id", generate_id()),
            name=data.get("name", ""),
            entity_type=EntityType(data.get("entity_type", "concept")),
            description=data.get("description", ""),
            memory_ids=data.get("memory_ids", []),
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )


@dataclass
class RoutingPattern:
    """Learned query → category mapping (v2 addition).

    Inspired by Roampal's Routing KG, this tracks which query patterns
    lead to which categories being most useful.
    """

    id: str = field(default_factory=generate_id)
    """Unique identifier."""

    query_pattern: str = ""
    """The query pattern (keywords or regex)."""

    category: MemoryCategory = MemoryCategory.GENERAL
    """Target category for this pattern."""

    success_count: int = 0
    """Times this routing led to a successful outcome."""

    fail_count: int = 0
    """Times this routing led to a failed outcome."""

    created_at: datetime = field(default_factory=utc_now)
    """When the pattern was first detected."""

    updated_at: datetime = field(default_factory=utc_now)
    """When the pattern was last updated."""

    @property
    def success_rate(self) -> float:
        """Calculate the success rate for this routing pattern."""
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5  # Neutral default
        return self.success_count / total

    @property
    def confidence(self) -> float:
        """Calculate confidence in this routing pattern.

        Higher confidence with more data points.
        """
        total = self.success_count + self.fail_count
        if total < 5:
            return 0.3  # Low confidence with few samples
        elif total < 20:
            return 0.6  # Medium confidence
        else:
            return 0.9  # High confidence with many samples

    def record_outcome(self, success: bool) -> None:
        """Record an outcome for this routing pattern.

        Args:
            success: Whether the routing led to a successful outcome.
        """
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "query_pattern": self.query_pattern,
            "category": self.category.value,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "success_rate": self.success_rate,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingPattern:
        """Create RoutingPattern from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = utc_now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = utc_now()

        return cls(
            id=data.get("id", generate_id()),
            query_pattern=data.get("query_pattern", ""),
            category=MemoryCategory(data.get("category", "general")),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            created_at=created_at,
            updated_at=updated_at,
        )


# =============================================================================
# Pydantic Validation Models
# =============================================================================


class MemoryCreate(BaseModel):
    """Pydantic model for creating a new memory."""

    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(..., min_length=1, max_length=10000)
    """The memory content (required, 1-10000 chars)."""

    category: MemoryCategory
    """Classification category (required)."""

    project: str | None = Field(default=None, max_length=255)
    """Project scope (optional)."""

    scope: MemoryScope = MemoryScope.PROJECT
    """Visibility scope."""

    source: MemorySource = MemorySource.EXPLICIT
    """How the memory was created."""

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    """Source reliability (0.0-1.0)."""

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    """Importance weight (0.0-1.0)."""

    tags: list[str] = Field(default_factory=list, max_length=20)
    """Optional tags (max 20)."""

    entities: list[str] = Field(default_factory=list, max_length=50)
    """Detected entities (max 50)."""

    supersedes: str | None = Field(default=None, max_length=36)
    """ID of memory this one replaces."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional metadata."""

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v.strip()

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate and normalize tags."""
        return [tag.strip().lower() for tag in v if tag.strip()]

    def to_memory(self) -> Memory:
        """Convert to Memory dataclass.

        Returns:
            Memory instance.
        """
        return Memory(
            content=self.content,
            category=self.category,
            project=self.project,
            scope=self.scope,
            source=self.source,
            confidence=self.confidence,
            importance=self.importance,
            tags=self.tags,
            entities=self.entities,
            supersedes=self.supersedes,
            metadata=self.metadata,
        )


class MemoryUpdate(BaseModel):
    """Pydantic model for updating an existing memory."""

    model_config = ConfigDict(str_strip_whitespace=True)

    content: str | None = Field(default=None, min_length=1, max_length=10000)
    """New content (optional)."""

    category: MemoryCategory | None = None
    """New category (optional)."""

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """New confidence (optional)."""

    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    """New importance (optional)."""

    tags: list[str] | None = Field(default=None, max_length=20)
    """New tags (optional)."""

    entities: list[str] | None = Field(default=None, max_length=50)
    """New entities (optional)."""

    metadata: dict[str, Any] | None = None
    """New metadata (optional)."""

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str | None) -> str | None:
        """Validate content is not just whitespace if provided."""
        if v is not None and not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v.strip() if v else None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str] | None) -> list[str] | None:
        """Validate and normalize tags."""
        if v is None:
            return None
        return [tag.strip().lower() for tag in v if tag.strip()]

    def apply_to(self, memory: Memory) -> Memory:
        """Apply updates to a memory.

        Args:
            memory: The memory to update.

        Returns:
            Updated memory.
        """
        if self.content is not None:
            memory.content = self.content
        if self.category is not None:
            memory.category = self.category
        if self.confidence is not None:
            memory.confidence = self.confidence
        if self.importance is not None:
            memory.importance = self.importance
        if self.tags is not None:
            memory.tags = self.tags
        if self.entities is not None:
            memory.entities = self.entities
        if self.metadata is not None:
            memory.metadata = self.metadata
        memory.updated_at = utc_now()
        return memory


class SearchQuery(BaseModel):
    """Pydantic model for search requests."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=1000)
    """Search query text."""

    project: str | None = Field(default=None, max_length=255)
    """Filter by project."""

    categories: list[MemoryCategory] | None = None
    """Filter by categories."""

    scope: MemoryScope | None = None
    """Filter by scope."""

    include_archived: bool = False
    """Whether to include archived memories."""

    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    """Minimum outcome score filter."""

    limit: int = Field(default=10, ge=1, le=100)
    """Maximum results to return."""

    offset: int = Field(default=0, ge=0)
    """Results offset for pagination."""


class OutcomeRecord(BaseModel):
    """Pydantic model for recording an outcome."""

    memory_ids: list[str] = Field(..., min_length=1, max_length=50)
    """IDs of memories to apply outcome to."""

    outcome: Outcome
    """The outcome to record."""


class MemoryResponse(BaseModel):
    """Pydantic model for memory API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    content: str
    category: MemoryCategory
    outcome_score: float
    confidence: float
    importance: float
    use_count: int
    project: str | None
    scope: MemoryScope
    source: MemorySource
    tags: list[str]
    entities: list[str]
    supersedes: str | None
    archived: bool
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]

    @classmethod
    def from_memory(cls, memory: Memory) -> MemoryResponse:
        """Create response from Memory dataclass.

        Args:
            memory: The memory to convert.

        Returns:
            MemoryResponse instance.
        """
        return cls(
            id=memory.id,
            content=memory.content,
            category=memory.category,
            outcome_score=memory.outcome_score,
            confidence=memory.confidence,
            importance=memory.importance,
            use_count=memory.use_count,
            project=memory.project,
            scope=memory.scope,
            source=memory.source,
            tags=memory.tags,
            entities=memory.entities,
            supersedes=memory.supersedes,
            archived=memory.archived,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            metadata=memory.metadata,
        )


class SearchResultResponse(BaseModel):
    """Pydantic model for search result API responses."""

    model_config = ConfigDict(from_attributes=True)

    memory: MemoryResponse
    score: float
    semantic_score: float
    recency_score: float
    frequency_score: float
    category_boost: float

    @classmethod
    def from_search_result(cls, result: SearchResult) -> SearchResultResponse:
        """Create response from SearchResult dataclass.

        Args:
            result: The search result to convert.

        Returns:
            SearchResultResponse instance.
        """
        return cls(
            memory=MemoryResponse.from_memory(result.memory),
            score=result.score,
            semantic_score=result.semantic_score,
            recency_score=result.recency_score,
            frequency_score=result.frequency_score,
            category_boost=result.category_boost,
        )


class ContextResponseModel(BaseModel):
    """Pydantic model for context API responses."""

    model_config = ConfigDict(from_attributes=True)

    memories: list[MemoryResponse]
    project: str | None
    total_count: int
    included_count: int
    formatted: str
    categories: dict[str, int]

    @classmethod
    def from_context_response(cls, response: ContextResponse) -> ContextResponseModel:
        """Create response from ContextResponse dataclass.

        Args:
            response: The context response to convert.

        Returns:
            ContextResponseModel instance.
        """
        return cls(
            memories=[MemoryResponse.from_memory(m) for m in response.memories],
            project=response.project,
            total_count=response.total_count,
            included_count=response.included_count,
            formatted=response.formatted or response.to_markdown(),
            categories=response.categories,
        )


class StatsResponse(BaseModel):
    """Pydantic model for statistics API responses."""

    total_memories: int
    """Total number of memories."""

    active_memories: int
    """Non-archived memories."""

    archived_memories: int
    """Archived memories."""

    by_category: dict[str, int]
    """Count by category."""

    by_scope: dict[str, int]
    """Count by scope."""

    by_source: dict[str, int]
    """Count by source."""

    avg_outcome_score: float
    """Average outcome score."""

    total_uses: int
    """Total use count across all memories."""

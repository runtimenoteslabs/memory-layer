"""Unit tests for v2 model additions.

Tests for:
- RelationType enum
- EntityType enum
- Relationship dataclass
- Entity dataclass
- RoutingPattern dataclass
- New MemoryCategory values
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory_layer.core.models import (
    Entity,
    EntityType,
    MemoryCategory,
    Relationship,
    RelationType,
    RoutingPattern,
)


# =============================================================================
# MemoryCategory Tests (v2 additions)
# =============================================================================


class TestMemoryCategoryV2:
    """Tests for v2 MemoryCategory additions."""

    def test_v2_category_dependency(self):
        """Test DEPENDENCY category exists."""
        assert MemoryCategory.DEPENDENCY == "dependency"
        assert MemoryCategory.DEPENDENCY.value == "dependency"

    def test_v2_category_environment(self):
        """Test ENVIRONMENT category exists."""
        assert MemoryCategory.ENVIRONMENT == "environment"
        assert MemoryCategory.ENVIRONMENT.value == "environment"

    def test_v2_category_coding_style(self):
        """Test CODING_STYLE category exists."""
        assert MemoryCategory.CODING_STYLE == "coding_style"
        assert MemoryCategory.CODING_STYLE.value == "coding_style"

    def test_v2_category_tool_preference(self):
        """Test TOOL_PREFERENCE category exists."""
        assert MemoryCategory.TOOL_PREFERENCE == "tool_preference"
        assert MemoryCategory.TOOL_PREFERENCE.value == "tool_preference"

    def test_v2_category_context(self):
        """Test CONTEXT category exists."""
        assert MemoryCategory.CONTEXT == "context"
        assert MemoryCategory.CONTEXT.value == "context"

    def test_v2_category_todo(self):
        """Test TODO category exists."""
        assert MemoryCategory.TODO == "todo"
        assert MemoryCategory.TODO.value == "todo"

    def test_v2_category_general(self):
        """Test GENERAL category exists."""
        assert MemoryCategory.GENERAL == "general"
        assert MemoryCategory.GENERAL.value == "general"

    def test_all_v2_categories_total(self):
        """Test total number of categories (v1 + v2)."""
        # v1: architecture, convention, decision, pattern, gotcha,
        #     workaround, troubleshooting, command, preference
        # v2: dependency, environment, coding_style, tool_preference,
        #     context, todo, general
        assert len(MemoryCategory) == 16

    def test_category_from_string(self):
        """Test creating categories from string values."""
        assert MemoryCategory("dependency") == MemoryCategory.DEPENDENCY
        assert MemoryCategory("environment") == MemoryCategory.ENVIRONMENT
        assert MemoryCategory("coding_style") == MemoryCategory.CODING_STYLE

    def test_invalid_category_raises(self):
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError):
            MemoryCategory("invalid_category")


# =============================================================================
# RelationType Tests
# =============================================================================


class TestRelationType:
    """Tests for RelationType enum."""

    def test_relation_updates(self):
        """Test UPDATES relation type."""
        assert RelationType.UPDATES == "updates"
        assert RelationType.UPDATES.value == "updates"

    def test_relation_extends(self):
        """Test EXTENDS relation type."""
        assert RelationType.EXTENDS == "extends"
        assert RelationType.EXTENDS.value == "extends"

    def test_relation_derives(self):
        """Test DERIVES relation type."""
        assert RelationType.DERIVES == "derives"
        assert RelationType.DERIVES.value == "derives"

    def test_relation_relates_to(self):
        """Test RELATES_TO relation type."""
        assert RelationType.RELATES_TO == "relates_to"
        assert RelationType.RELATES_TO.value == "relates_to"

    def test_relation_conflicts_with(self):
        """Test CONFLICTS_WITH relation type."""
        assert RelationType.CONFLICTS_WITH == "conflicts_with"
        assert RelationType.CONFLICTS_WITH.value == "conflicts_with"

    def test_all_relation_types(self):
        """Test total number of relation types."""
        assert len(RelationType) == 5

    def test_relation_from_string(self):
        """Test creating relation types from string values."""
        assert RelationType("updates") == RelationType.UPDATES
        assert RelationType("conflicts_with") == RelationType.CONFLICTS_WITH


# =============================================================================
# EntityType Tests
# =============================================================================


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_file(self):
        """Test FILE entity type."""
        assert EntityType.FILE == "file"
        assert EntityType.FILE.value == "file"

    def test_entity_module(self):
        """Test MODULE entity type."""
        assert EntityType.MODULE == "module"
        assert EntityType.MODULE.value == "module"

    def test_entity_function(self):
        """Test FUNCTION entity type."""
        assert EntityType.FUNCTION == "function"
        assert EntityType.FUNCTION.value == "function"

    def test_entity_class(self):
        """Test CLASS entity type."""
        assert EntityType.CLASS == "class"
        assert EntityType.CLASS.value == "class"

    def test_entity_variable(self):
        """Test VARIABLE entity type."""
        assert EntityType.VARIABLE == "variable"
        assert EntityType.VARIABLE.value == "variable"

    def test_entity_error(self):
        """Test ERROR entity type."""
        assert EntityType.ERROR == "error"
        assert EntityType.ERROR.value == "error"

    def test_entity_concept(self):
        """Test CONCEPT entity type."""
        assert EntityType.CONCEPT == "concept"
        assert EntityType.CONCEPT.value == "concept"

    def test_entity_tool(self):
        """Test TOOL entity type."""
        assert EntityType.TOOL == "tool"
        assert EntityType.TOOL.value == "tool"

    def test_entity_person(self):
        """Test PERSON entity type."""
        assert EntityType.PERSON == "person"
        assert EntityType.PERSON.value == "person"

    def test_all_entity_types(self):
        """Test total number of entity types."""
        assert len(EntityType) == 9

    def test_entity_from_string(self):
        """Test creating entity types from string values."""
        assert EntityType("file") == EntityType.FILE
        assert EntityType("function") == EntityType.FUNCTION


# =============================================================================
# Relationship Tests
# =============================================================================


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_create_basic_relationship(self):
        """Test creating a basic relationship."""
        rel = Relationship(
            source_id="mem-001",
            target_id="mem-002",
            relation_type=RelationType.EXTENDS,
        )

        assert rel.source_id == "mem-001"
        assert rel.target_id == "mem-002"
        assert rel.relation_type == RelationType.EXTENDS
        assert rel.strength == 1.0  # Default
        assert isinstance(rel.created_at, datetime)
        assert rel.metadata == {}

    def test_create_relationship_with_all_fields(self):
        """Test creating relationship with all fields."""
        created = datetime.now(timezone.utc)
        rel = Relationship(
            source_id="mem-001",
            target_id="mem-002",
            relation_type=RelationType.UPDATES,
            strength=0.8,
            created_at=created,
            metadata={"reason": "new decision"},
        )

        assert rel.strength == 0.8
        assert rel.created_at == created
        assert rel.metadata["reason"] == "new decision"

    def test_relationship_to_dict(self):
        """Test converting relationship to dictionary."""
        rel = Relationship(
            source_id="mem-001",
            target_id="mem-002",
            relation_type=RelationType.CONFLICTS_WITH,
            strength=0.5,
        )

        result = rel.to_dict()

        assert result["source_id"] == "mem-001"
        assert result["target_id"] == "mem-002"
        assert result["relation_type"] == "conflicts_with"
        assert result["strength"] == 0.5
        assert "created_at" in result
        assert result["metadata"] == {}

    def test_relationship_from_dict(self):
        """Test creating relationship from dictionary."""
        data = {
            "source_id": "mem-001",
            "target_id": "mem-002",
            "relation_type": "derives",
            "strength": 0.7,
            "metadata": {"context": "pattern extraction"},
        }

        rel = Relationship.from_dict(data)

        assert rel.source_id == "mem-001"
        assert rel.target_id == "mem-002"
        assert rel.relation_type == RelationType.DERIVES
        assert rel.strength == 0.7
        assert rel.metadata["context"] == "pattern extraction"

    def test_relationship_from_dict_with_datetime_string(self):
        """Test creating relationship with datetime as string."""
        data = {
            "source_id": "mem-001",
            "target_id": "mem-002",
            "relation_type": "extends",
            "created_at": "2024-01-15T10:30:00+00:00",
        }

        rel = Relationship.from_dict(data)

        assert rel.source_id == "mem-001"
        assert isinstance(rel.created_at, datetime)

    def test_relationship_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = Relationship(
            source_id="src-123",
            target_id="tgt-456",
            relation_type=RelationType.RELATES_TO,
            strength=0.9,
            metadata={"notes": "related concepts"},
        )

        data = original.to_dict()
        restored = Relationship.from_dict(data)

        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.relation_type == original.relation_type
        assert restored.strength == original.strength
        assert restored.metadata == original.metadata


# =============================================================================
# Entity Tests
# =============================================================================


class TestEntity:
    """Tests for Entity dataclass."""

    def test_create_basic_entity(self):
        """Test creating a basic entity."""
        entity = Entity(
            name="auth_service.py",
            entity_type=EntityType.FILE,
        )

        assert entity.name == "auth_service.py"
        assert entity.entity_type == EntityType.FILE
        assert entity.description == ""
        assert entity.memory_ids == []
        assert isinstance(entity.id, str)
        assert isinstance(entity.created_at, datetime)

    def test_create_entity_with_all_fields(self):
        """Test creating entity with all fields."""
        entity = Entity(
            id="entity-123",
            name="UserRepository",
            entity_type=EntityType.CLASS,
            description="Handles user data access",
            memory_ids=["mem-001", "mem-002"],
            metadata={"module": "core.models"},
        )

        assert entity.id == "entity-123"
        assert entity.name == "UserRepository"
        assert entity.entity_type == EntityType.CLASS
        assert entity.description == "Handles user data access"
        assert len(entity.memory_ids) == 2
        assert entity.metadata["module"] == "core.models"

    def test_entity_to_dict(self):
        """Test converting entity to dictionary."""
        entity = Entity(
            id="ent-001",
            name="handleError",
            entity_type=EntityType.FUNCTION,
            memory_ids=["mem-001"],
        )

        result = entity.to_dict()

        assert result["id"] == "ent-001"
        assert result["name"] == "handleError"
        assert result["entity_type"] == "function"
        assert result["memory_ids"] == ["mem-001"]
        assert "created_at" in result

    def test_entity_from_dict(self):
        """Test creating entity from dictionary."""
        data = {
            "id": "ent-002",
            "name": "ConnectionError",
            "entity_type": "error",
            "description": "Database connection failure",
            "memory_ids": ["mem-003"],
            "metadata": {"source": "troubleshooting"},
        }

        entity = Entity.from_dict(data)

        assert entity.id == "ent-002"
        assert entity.name == "ConnectionError"
        assert entity.entity_type == EntityType.ERROR
        assert entity.description == "Database connection failure"
        assert "mem-003" in entity.memory_ids

    def test_entity_from_dict_with_defaults(self):
        """Test creating entity with minimal data."""
        data = {
            "name": "simple_concept",
        }

        entity = Entity.from_dict(data)

        assert entity.name == "simple_concept"
        assert entity.entity_type == EntityType.CONCEPT  # Default
        assert entity.description == ""
        assert entity.memory_ids == []

    def test_entity_from_dict_with_datetime_string(self):
        """Test creating entity with datetime as string."""
        data = {
            "name": "test",
            "created_at": "2024-01-15T10:30:00+00:00",
        }

        entity = Entity.from_dict(data)

        assert isinstance(entity.created_at, datetime)

    def test_entity_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = Entity(
            name="PostgreSQL",
            entity_type=EntityType.TOOL,
            description="Primary database",
            memory_ids=["mem-001", "mem-002", "mem-003"],
            metadata={"version": "15.2"},
        )

        data = original.to_dict()
        restored = Entity.from_dict(data)

        assert restored.name == original.name
        assert restored.entity_type == original.entity_type
        assert restored.description == original.description
        assert restored.memory_ids == original.memory_ids
        assert restored.metadata == original.metadata


# =============================================================================
# RoutingPattern Tests
# =============================================================================


class TestRoutingPattern:
    """Tests for RoutingPattern dataclass."""

    def test_create_basic_routing_pattern(self):
        """Test creating a basic routing pattern."""
        pattern = RoutingPattern(
            query_pattern="authentication",
            category=MemoryCategory.ARCHITECTURE,
        )

        assert pattern.query_pattern == "authentication"
        assert pattern.category == MemoryCategory.ARCHITECTURE
        assert pattern.success_count == 0
        assert pattern.fail_count == 0
        assert isinstance(pattern.id, str)

    def test_create_routing_pattern_with_counts(self):
        """Test creating routing pattern with initial counts."""
        pattern = RoutingPattern(
            query_pattern="error handling",
            category=MemoryCategory.TROUBLESHOOTING,
            success_count=15,
            fail_count=3,
        )

        assert pattern.success_count == 15
        assert pattern.fail_count == 3

    def test_success_rate_no_data(self):
        """Test success rate with no outcome data."""
        pattern = RoutingPattern(
            query_pattern="test",
            category=MemoryCategory.GENERAL,
        )

        assert pattern.success_rate == 0.5  # Neutral default

    def test_success_rate_with_data(self):
        """Test success rate calculation."""
        pattern = RoutingPattern(
            query_pattern="database query",
            category=MemoryCategory.PATTERN,
            success_count=8,
            fail_count=2,
        )

        assert pattern.success_rate == 0.8  # 8/10

    def test_success_rate_all_failures(self):
        """Test success rate with all failures."""
        pattern = RoutingPattern(
            query_pattern="broken",
            category=MemoryCategory.GENERAL,
            success_count=0,
            fail_count=5,
        )

        assert pattern.success_rate == 0.0

    def test_success_rate_all_successes(self):
        """Test success rate with all successes."""
        pattern = RoutingPattern(
            query_pattern="great",
            category=MemoryCategory.GENERAL,
            success_count=10,
            fail_count=0,
        )

        assert pattern.success_rate == 1.0

    def test_confidence_low_samples(self):
        """Test confidence with few samples."""
        pattern = RoutingPattern(
            query_pattern="new",
            category=MemoryCategory.GENERAL,
            success_count=2,
            fail_count=1,
        )

        assert pattern.confidence == 0.3  # < 5 samples

    def test_confidence_medium_samples(self):
        """Test confidence with medium samples."""
        pattern = RoutingPattern(
            query_pattern="medium",
            category=MemoryCategory.GENERAL,
            success_count=7,
            fail_count=3,
        )

        assert pattern.confidence == 0.6  # 5-19 samples

    def test_confidence_high_samples(self):
        """Test confidence with many samples."""
        pattern = RoutingPattern(
            query_pattern="established",
            category=MemoryCategory.GENERAL,
            success_count=15,
            fail_count=10,
        )

        assert pattern.confidence == 0.9  # >= 20 samples

    def test_record_outcome_success(self):
        """Test recording successful outcome."""
        pattern = RoutingPattern(
            query_pattern="test",
            category=MemoryCategory.GENERAL,
        )

        old_updated = pattern.updated_at
        pattern.record_outcome(success=True)

        assert pattern.success_count == 1
        assert pattern.fail_count == 0
        assert pattern.updated_at >= old_updated

    def test_record_outcome_failure(self):
        """Test recording failed outcome."""
        pattern = RoutingPattern(
            query_pattern="test",
            category=MemoryCategory.GENERAL,
        )

        pattern.record_outcome(success=False)

        assert pattern.success_count == 0
        assert pattern.fail_count == 1

    def test_record_multiple_outcomes(self):
        """Test recording multiple outcomes."""
        pattern = RoutingPattern(
            query_pattern="test",
            category=MemoryCategory.GENERAL,
        )

        pattern.record_outcome(success=True)
        pattern.record_outcome(success=True)
        pattern.record_outcome(success=False)
        pattern.record_outcome(success=True)

        assert pattern.success_count == 3
        assert pattern.fail_count == 1
        assert pattern.success_rate == 0.75

    def test_routing_pattern_to_dict(self):
        """Test converting routing pattern to dictionary."""
        pattern = RoutingPattern(
            id="rp-001",
            query_pattern="api endpoint",
            category=MemoryCategory.ARCHITECTURE,
            success_count=12,
            fail_count=4,
        )

        result = pattern.to_dict()

        assert result["id"] == "rp-001"
        assert result["query_pattern"] == "api endpoint"
        assert result["category"] == "architecture"
        assert result["success_count"] == 12
        assert result["fail_count"] == 4
        assert result["success_rate"] == 0.75
        assert result["confidence"] == 0.6  # 16 samples = medium
        assert "created_at" in result
        assert "updated_at" in result

    def test_routing_pattern_from_dict(self):
        """Test creating routing pattern from dictionary."""
        data = {
            "id": "rp-002",
            "query_pattern": "debugging tips",
            "category": "troubleshooting",
            "success_count": 20,
            "fail_count": 5,
        }

        pattern = RoutingPattern.from_dict(data)

        assert pattern.id == "rp-002"
        assert pattern.query_pattern == "debugging tips"
        assert pattern.category == MemoryCategory.TROUBLESHOOTING
        assert pattern.success_count == 20
        assert pattern.fail_count == 5

    def test_routing_pattern_from_dict_with_datetime_strings(self):
        """Test creating routing pattern with datetime strings."""
        data = {
            "query_pattern": "test",
            "category": "general",
            "created_at": "2024-01-15T10:30:00+00:00",
            "updated_at": "2024-01-15T11:30:00+00:00",
        }

        pattern = RoutingPattern.from_dict(data)

        assert isinstance(pattern.created_at, datetime)
        assert isinstance(pattern.updated_at, datetime)

    def test_routing_pattern_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = RoutingPattern(
            query_pattern="naming conventions",
            category=MemoryCategory.CONVENTION,
            success_count=30,
            fail_count=5,
        )

        # Record an outcome to update the timestamp
        original.record_outcome(success=True)

        data = original.to_dict()
        restored = RoutingPattern.from_dict(data)

        assert restored.query_pattern == original.query_pattern
        assert restored.category == original.category
        assert restored.success_count == original.success_count
        assert restored.fail_count == original.fail_count


# =============================================================================
# Integration Tests
# =============================================================================


class TestV2ModelIntegration:
    """Integration tests for v2 models working together."""

    def test_entity_references_memories(self):
        """Test that entities can reference multiple memories."""
        entity = Entity(
            name="AuthService",
            entity_type=EntityType.MODULE,
            memory_ids=["mem-001", "mem-002", "mem-003"],
        )

        assert len(entity.memory_ids) == 3
        assert "mem-001" in entity.memory_ids

    def test_relationship_chain(self):
        """Test creating a chain of relationships."""
        # Memory A updates B, B extends C
        rel1 = Relationship(
            source_id="mem-A",
            target_id="mem-B",
            relation_type=RelationType.UPDATES,
        )

        rel2 = Relationship(
            source_id="mem-B",
            target_id="mem-C",
            relation_type=RelationType.EXTENDS,
        )

        assert rel1.target_id == rel2.source_id

    def test_routing_pattern_for_v2_category(self):
        """Test routing pattern with v2 category."""
        pattern = RoutingPattern(
            query_pattern="environment variables",
            category=MemoryCategory.ENVIRONMENT,  # v2 category
        )

        assert pattern.category == MemoryCategory.ENVIRONMENT
        data = pattern.to_dict()
        assert data["category"] == "environment"

    def test_all_v2_categories_in_routing(self):
        """Test that all v2 categories work in routing patterns."""
        v2_categories = [
            MemoryCategory.DEPENDENCY,
            MemoryCategory.ENVIRONMENT,
            MemoryCategory.CODING_STYLE,
            MemoryCategory.TOOL_PREFERENCE,
            MemoryCategory.CONTEXT,
            MemoryCategory.TODO,
            MemoryCategory.GENERAL,
        ]

        for category in v2_categories:
            pattern = RoutingPattern(
                query_pattern=f"test {category.value}",
                category=category,
            )
            assert pattern.category == category

    def test_conflict_relationship(self):
        """Test creating conflict relationship for contradiction."""
        rel = Relationship(
            source_id="mem-old-decision",
            target_id="mem-new-decision",
            relation_type=RelationType.CONFLICTS_WITH,
            metadata={"resolution": "pending"},
        )

        assert rel.relation_type == RelationType.CONFLICTS_WITH
        assert rel.metadata["resolution"] == "pending"

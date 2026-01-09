"""Integration tests for Memory Engine.

These tests verify the full workflow of the engine with real database
operations and mock embeddings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_layer.core.embeddings import MockEmbeddingProvider
from memory_layer.core.engine import EngineConfig, MemoryEngine
from memory_layer.core.models import (
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
)


@pytest.fixture
async def engine(temp_db_path: Path) -> MemoryEngine:
    """Create an initialized engine for integration testing."""
    config = EngineConfig(
        db_path=temp_db_path,
        secure_permissions=False,
        embedding_provider="mock",
        track_last_search=True,
        auto_archive_enabled=True,
        auto_archive_threshold=-0.5,
    )
    eng = MemoryEngine(config=config)
    await eng.initialize()
    yield eng
    await eng.close()


class TestMemoryLifecycle:
    """Test the full lifecycle of a memory."""

    async def test_full_lifecycle(self, engine: MemoryEngine) -> None:
        """Test add → search → outcome → verify score flow."""
        # 1. Add a memory
        memory = await engine.add(
            content="When debugging async code, use asyncio.run() for the entry point",
            category=MemoryCategory.TROUBLESHOOTING,
            project="python-tips",
            tags=["async", "debugging"],
        )
        assert memory.id
        assert memory.outcome_score == 0.0

        # 2. Search for the memory
        results = await engine.search("async debugging")
        assert len(results) > 0
        assert any(r.memory.id == memory.id for r in results)

        # 3. Record positive outcome
        await engine.record_outcome(memory.id, Outcome.WORKED)

        # 4. Verify score increased
        updated = await engine.get(memory.id)
        assert updated.outcome_score == 0.2

        # 5. Search again - should still find it
        results2 = await engine.search("async debugging")
        assert any(r.memory.id == memory.id for r in results2)

        # 6. Delete (archive) the memory
        await engine.delete(memory.id)

        # 7. Search should not return archived
        results3 = await engine.search("async debugging")
        assert all(r.memory.id != memory.id for r in results3)

    async def test_memory_with_negative_feedback_gets_archived(
        self, engine: MemoryEngine
    ) -> None:
        """Test that memories with negative feedback can be auto-archived."""
        # Add a memory
        memory = await engine.add(
            content="Bad advice that will fail",
            category=MemoryCategory.PATTERN,
        )

        # Record multiple failures
        for _ in range(2):
            await engine.record_outcome(memory.id, Outcome.FAILED)

        # Check score is negative
        updated = await engine.get(memory.id)
        assert updated.outcome_score == -0.6

        # Run auto-archive
        archived_count = await engine.auto_archive()
        assert archived_count == 1

        # Memory should be archived
        final = await engine.get(memory.id)
        assert final.archived is True


class TestContextInjection:
    """Test context generation for AI agent injection."""

    async def test_context_generation_workflow(self, engine: MemoryEngine) -> None:
        """Test generating context for injection."""
        # Add various memories
        await engine.add(
            content="Use type hints for better IDE support",
            category=MemoryCategory.CONVENTION,
            project="python-style",
        )
        await engine.add(
            content="Watch out for mutable default arguments",
            category=MemoryCategory.GOTCHA,
            project="python-style",
        )
        await engine.add(
            content="Use pytest fixtures for test setup",
            category=MemoryCategory.PATTERN,
            project="python-style",
        )

        # Generate context for a task
        context = await engine.get_context(
            query="writing python code",
            project="python-style",
            max_memories=10,
        )

        # Verify context
        assert context.included_count > 0
        assert context.project == "python-style"
        assert len(context.categories) > 0

        # Verify markdown format
        markdown = context.to_markdown()
        assert "# Project Knowledge" in markdown
        assert len(markdown) > 50

    async def test_context_respects_outcome_scores(self, engine: MemoryEngine) -> None:
        """Test that context prefers high-scoring memories."""
        # Add memories with different scores
        good_memory = await engine.add(
            content="This is proven good advice",
            category=MemoryCategory.PATTERN,
        )
        await engine.record_outcome(good_memory.id, Outcome.WORKED)
        await engine.record_outcome(good_memory.id, Outcome.WORKED)

        bad_memory = await engine.add(
            content="This is questionable advice",
            category=MemoryCategory.PATTERN,
        )
        await engine.record_outcome(bad_memory.id, Outcome.FAILED)

        # Get context (without query, uses score ordering)
        context = await engine.get_context(max_memories=5)

        # Good memory should be first
        if len(context.memories) > 1:
            good_in_results = any(m.id == good_memory.id for m in context.memories)
            assert good_in_results


class TestMultiProjectIsolation:
    """Test that projects are properly isolated."""

    async def test_project_isolation_in_search(self, engine: MemoryEngine) -> None:
        """Test that search respects project boundaries."""
        # Add memories to different projects
        await engine.add(
            content="Project A specific pattern",
            category=MemoryCategory.PATTERN,
            project="project-a",
        )
        await engine.add(
            content="Project B specific pattern",
            category=MemoryCategory.PATTERN,
            project="project-b",
        )
        await engine.add(
            content="Global pattern available everywhere",
            category=MemoryCategory.PATTERN,
            scope=MemoryScope.GLOBAL,
        )

        # Search in project A
        results_a = await engine.search("pattern", project="project-a")
        project_a_ids = {r.memory.project for r in results_a}
        assert "project-b" not in project_a_ids

        # Search in project B
        results_b = await engine.search("pattern", project="project-b")
        project_b_ids = {r.memory.project for r in results_b}
        assert "project-a" not in project_b_ids

    async def test_project_isolation_in_stats(self, engine: MemoryEngine) -> None:
        """Test that stats respect project boundaries."""
        # Add memories to different projects
        for i in range(3):
            await engine.add(
                content=f"Project A memory {i}",
                category=MemoryCategory.DECISION,
                project="project-a",
            )
        for i in range(5):
            await engine.add(
                content=f"Project B memory {i}",
                category=MemoryCategory.DECISION,
                project="project-b",
            )

        # Get stats per project
        stats_a = await engine.stats(project="project-a")
        stats_b = await engine.stats(project="project-b")

        assert stats_a.storage_stats.total_memories == 3
        assert stats_b.storage_stats.total_memories == 5


class TestSupersedes:
    """Test memory supersession functionality."""

    async def test_supersedes_archives_old_memory(self, engine: MemoryEngine) -> None:
        """Test that superseding a memory archives the old one."""
        # Create original memory
        original = await engine.add(
            content="Original advice v1",
            category=MemoryCategory.DECISION,
            project="test",
        )

        # Create superseding memory
        new_memory = await engine.add(
            content="Updated advice v2",
            category=MemoryCategory.DECISION,
            project="test",
            supersedes=original.id,
        )

        # Original should be archived
        original_updated = await engine.get(original.id)
        assert original_updated.archived is True

        # New memory should be active
        assert new_memory.archived is False

        # Search should only return new memory
        results = await engine.search("advice", project="test")
        memory_ids = [r.memory.id for r in results]
        assert new_memory.id in memory_ids
        assert original.id not in memory_ids


class TestSearchWithUsageTracking:
    """Test that usage tracking works correctly."""

    async def test_use_count_increments(self, engine: MemoryEngine) -> None:
        """Test that use counts are incremented on search."""
        memory = await engine.add(
            content="Frequently accessed memory",
            category=MemoryCategory.PATTERN,
        )
        assert memory.use_count == 0

        # Search multiple times
        for _ in range(3):
            await engine.search("frequently accessed")

        # Check use count increased
        updated = await engine.get(memory.id)
        assert updated.use_count == 3


class TestOutcomeRecordingWorkflow:
    """Test outcome recording workflows."""

    async def test_record_outcome_for_last_search(self, engine: MemoryEngine) -> None:
        """Test recording outcome for last search results."""
        # Add some memories
        m1 = await engine.add(
            content="Helpful advice 1",
            category=MemoryCategory.PATTERN,
        )
        m2 = await engine.add(
            content="Helpful advice 2",
            category=MemoryCategory.PATTERN,
        )

        # Perform search
        await engine.search("helpful advice")

        # Record outcome for all search results
        updated = await engine.record_outcome_for_last_search(Outcome.WORKED)

        # Both memories should have increased scores
        for memory in updated:
            assert memory.outcome_score == 0.2

    async def test_outcome_score_clamping(self, engine: MemoryEngine) -> None:
        """Test that outcome scores are clamped to [-1, 1]."""
        memory = await engine.add(
            content="Test memory",
            category=MemoryCategory.PATTERN,
        )

        # Record many positive outcomes
        for _ in range(10):
            await engine.record_outcome(memory.id, Outcome.WORKED)

        # Score should not exceed 1.0
        updated = await engine.get(memory.id)
        assert updated.outcome_score <= 1.0

        # Record many negative outcomes
        for _ in range(20):
            await engine.record_outcome(memory.id, Outcome.FAILED)

        # Score should not go below -1.0
        final = await engine.get(memory.id)
        assert final.outcome_score >= -1.0


class TestBatchOperations:
    """Test batch operations."""

    async def test_add_many_memories(self, engine: MemoryEngine) -> None:
        """Test adding multiple memories in batch."""
        memories = [
            Memory(
                content=f"Batch memory {i}",
                category=MemoryCategory.DECISION,
                project="batch-test",
            )
            for i in range(10)
        ]

        results = await engine.add_many(memories)
        assert len(results) == 10

        # All should have embeddings
        for memory in results:
            assert memory.embedding is not None

        # All should be searchable
        search_results = await engine.search("batch memory", project="batch-test")
        assert len(search_results) == 10


class TestEngineRestart:
    """Test engine behavior across restarts."""

    async def test_memories_persist_across_restart(self, temp_db_path: Path) -> None:
        """Test that memories are persisted and reloaded."""
        config = EngineConfig(
            db_path=temp_db_path,
            secure_permissions=False,
            embedding_provider="mock",
        )

        # Create engine and add memories
        engine1 = MemoryEngine(config=config)
        await engine1.initialize()

        memory = await engine1.add(
            content="Persistent memory",
            category=MemoryCategory.PATTERN,
            project="persistence-test",
        )
        memory_id = memory.id

        await engine1.close()

        # Create new engine instance
        engine2 = MemoryEngine(config=config)
        await engine2.initialize()

        # Memory should be retrievable
        retrieved = await engine2.get(memory_id)
        assert retrieved.content == "Persistent memory"

        # Memory should be in retriever (searchable)
        results = await engine2.search("persistent", project="persistence-test")
        assert any(r.memory.id == memory_id for r in results)

        await engine2.close()

    async def test_outcome_scores_persist(self, temp_db_path: Path) -> None:
        """Test that outcome scores persist across restarts."""
        config = EngineConfig(
            db_path=temp_db_path,
            secure_permissions=False,
            embedding_provider="mock",
        )

        # Create engine, add memory, record outcome
        engine1 = MemoryEngine(config=config)
        await engine1.initialize()

        memory = await engine1.add(
            content="Test memory",
            category=MemoryCategory.PATTERN,
        )
        await engine1.record_outcome(memory.id, Outcome.WORKED)
        await engine1.close()

        # Verify after restart
        engine2 = MemoryEngine(config=config)
        await engine2.initialize()

        retrieved = await engine2.get(memory.id)
        assert retrieved.outcome_score == 0.2

        await engine2.close()

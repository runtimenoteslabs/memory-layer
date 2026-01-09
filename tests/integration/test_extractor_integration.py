"""Integration tests for the extraction pipeline with real API.

These tests require a valid ANTHROPIC_API_KEY environment variable.
They make actual API calls and are marked with @pytest.mark.integration.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from memory_layer.core.models import MemoryCategory
from memory_layer.extraction.extractor import (
    ExtractionConfig,
    MemoryExtractor,
    extract_from_transcript,
)

# Skip all tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY", "")


@pytest.fixture
def extractor(api_key: str) -> MemoryExtractor:
    """Create an extractor with real API."""
    config = ExtractionConfig(
        model="claude-sonnet-4-20250514",
        min_confidence=0.3,
        min_importance=0.2,
        enable_pii_filtering=True,
        enable_injection_detection=True,
    )
    return MemoryExtractor(config=config, api_key=api_key)


class TestRealExtraction:
    """Integration tests for real extraction."""

    @pytest.mark.integration
    async def test_extract_simple_conversation(self, extractor: MemoryExtractor) -> None:
        """Test extracting from a simple coding conversation."""
        transcript = """
        User: I keep getting a ModuleNotFoundError when importing pandas in my project.

        Assistant: This usually happens when pandas isn't installed in your current
        Python environment. Try running `pip install pandas` in your terminal.

        User: That worked! But now I'm getting a warning about a deprecated function.

        Assistant: You should check which function is deprecated and update to the
        recommended alternative. Pandas often deprecates functions between major versions.
        Always check the migration guide when upgrading.

        User: Good advice. I'll make sure to pin my pandas version in requirements.txt
        to avoid surprise breakages.

        Assistant: That's a great practice! You can use `pandas==2.0.0` format in
        requirements.txt to pin exact versions, or `pandas>=2.0.0,<3.0.0` for
        compatible version ranges.
        """

        result = await extractor.extract_from_transcript(
            transcript=transcript,
            project="test-project",
        )

        assert result.success, f"Extraction failed: {result.error}"
        assert result.memory_count > 0, "Should extract at least one memory"
        assert result.summary, "Should have a summary"

        # Check that we got reasonable memories
        for memory in result.memories:
            assert memory.content, "Memory should have content"
            assert memory.category in MemoryCategory, "Should have valid category"
            assert 0.0 <= memory.confidence <= 1.0
            assert 0.0 <= memory.importance <= 1.0

    @pytest.mark.integration
    async def test_extract_troubleshooting_conversation(self, extractor: MemoryExtractor) -> None:
        """Test extracting troubleshooting information."""
        transcript = """
        User: My Docker container keeps crashing with "OOMKilled" status.

        Assistant: OOMKilled means your container ran out of memory. You need to either:
        1. Increase the memory limit in your docker-compose.yml with `mem_limit: 2g`
        2. Or optimize your application to use less memory

        User: I increased the limit and it's working now. Should I document this?

        Assistant: Yes! This is exactly the kind of gotcha that's worth remembering.
        The default Docker memory limit is often too low for data-intensive applications.

        User: By the way, I noticed the container also needs `--shm-size=1g` for
        PyTorch to work properly with DataLoader workers.

        Assistant: Great catch! That's another common gotcha. PyTorch's DataLoader
        uses shared memory for inter-process communication, and Docker's default
        64MB shared memory is usually insufficient.
        """

        result = await extractor.extract_from_transcript(
            transcript=transcript,
            project="docker-tips",
        )

        assert result.success
        assert result.memory_count > 0

        # Should extract troubleshooting/gotcha type memories
        categories = [m.category for m in result.memories]
        troubleshooting_categories = {
            MemoryCategory.TROUBLESHOOTING,
            MemoryCategory.GOTCHA,
            MemoryCategory.WORKAROUND,
        }
        assert any(
            cat in troubleshooting_categories for cat in categories
        ), f"Expected troubleshooting-related category, got {categories}"

    @pytest.mark.integration
    async def test_extract_architecture_decisions(self, extractor: MemoryExtractor) -> None:
        """Test extracting architecture decisions."""
        transcript = """
        User: Should we use a monolith or microservices for this project?

        Assistant: Given your team size (3 developers) and the project scope,
        I'd recommend starting with a modular monolith. You can always split
        into microservices later if needed.

        User: That makes sense. What about the database choice?

        Assistant: For this use case with complex JSON data and full-text search needs,
        PostgreSQL would be ideal. It has excellent JSON support with JSONB and
        built-in full-text search capabilities.

        User: Agreed. We decided to go with PostgreSQL and a modular monolith architecture.
        The rationale is to keep deployment simple while maintaining code organization.
        """

        result = await extractor.extract_from_transcript(
            transcript=transcript,
            project="architecture-decisions",
        )

        assert result.success
        assert result.memory_count > 0

        # Should extract decision/architecture type memories
        categories = [m.category for m in result.memories]
        decision_categories = {
            MemoryCategory.DECISION,
            MemoryCategory.ARCHITECTURE,
        }
        assert any(
            cat in decision_categories for cat in categories
        ), f"Expected decision/architecture category, got {categories}"

    @pytest.mark.integration
    async def test_extract_with_pii_filtering(self, extractor: MemoryExtractor) -> None:
        """Test that PII is filtered from extraction."""
        transcript = """
        User: I'm setting up the database connection. Here's my config:
        DB_HOST=192.168.1.100
        DB_USER=admin
        DB_PASSWORD=supersecret123

        Assistant: I see you're using a local IP. For production, you should:
        1. Use environment variables instead of hardcoding credentials
        2. Use a connection pool for better performance
        3. Consider using SSL for the database connection

        User: My email is developer@company.com if you need to reach me.

        Assistant: Thanks. Remember to never commit credentials to version control.
        Use a .env file and add it to .gitignore.
        """

        result = await extractor.extract_from_transcript(
            transcript=transcript,
            project="security-test",
        )

        assert result.success
        assert result.pii_removed > 0, "Should have detected and removed PII"

        # Check that extracted memories don't contain raw PII
        all_content = " ".join(m.content for m in result.memories)
        assert "supersecret123" not in all_content
        assert "developer@company.com" not in all_content

    @pytest.mark.integration
    async def test_extract_commands_and_patterns(self, extractor: MemoryExtractor) -> None:
        """Test extracting commands and patterns."""
        transcript = """
        User: How do I set up pre-commit hooks for this Python project?

        Assistant: First install pre-commit:
        $ pip install pre-commit

        Then create a .pre-commit-config.yaml file and run:
        $ pre-commit install

        User: What hooks do you recommend?

        Assistant: For Python projects, I recommend:
        - ruff for linting and formatting
        - mypy for type checking
        - pytest for running tests

        The pattern I follow is: lint -> typecheck -> test, in that order.

        User: Great, I'll set that up. Always run `pre-commit run --all-files`
        before pushing to catch issues early.
        """

        result = await extractor.extract_from_transcript(
            transcript=transcript,
            project="python-setup",
        )

        assert result.success
        assert result.memory_count > 0

        # Should find some entities
        all_entities = []
        for memory in result.memories:
            all_entities.extend(memory.entities)

        # Should detect at least some relevant entities
        assert len(all_entities) > 0 or any(
            m.category in {MemoryCategory.COMMAND, MemoryCategory.PATTERN}
            for m in result.memories
        )


class TestConvenienceFunction:
    """Test the convenience function."""

    @pytest.mark.integration
    async def test_extract_from_transcript_function(self, api_key: str) -> None:
        """Test the standalone extract_from_transcript function."""
        transcript = """
        User: What's the best way to handle errors in async Python code?

        Assistant: Use try/except blocks with specific exception types.
        For async code, remember that exceptions in tasks need to be
        awaited or gathered properly to be caught.

        User: Thanks, I'll use asyncio.gather with return_exceptions=True
        to handle errors gracefully.
        """ * 2  # Repeat to meet minimum length

        config = ExtractionConfig(
            model="claude-sonnet-4-20250514",
            min_confidence=0.2,
            min_importance=0.1,
        )

        result = await extract_from_transcript(
            transcript=transcript,
            project="async-patterns",
            config=config,
            api_key=api_key,
        )

        assert result.success
        # May or may not extract memories depending on LLM judgment
        assert result.transcript_length > 0
        assert result.extraction_time_ms > 0


class TestExtractionEdgeCases:
    """Test edge cases with real API."""

    @pytest.mark.integration
    async def test_extract_minimal_useful_content(self, extractor: MemoryExtractor) -> None:
        """Test extraction from minimal but useful conversation."""
        transcript = """
        User: Why does git push fail with "non-fast-forward"?

        Assistant: This happens when the remote has commits you don't have locally.
        Solution: Run `git pull --rebase` first, then push again.

        User: That fixed it!
        """ * 3  # Repeat to meet minimum length

        result = await extractor.extract_from_transcript(
            transcript=transcript,
            project="git-tips",
        )

        assert result.success
        # This should extract at least one troubleshooting memory

    @pytest.mark.integration
    async def test_extract_technical_discussion(self, extractor: MemoryExtractor) -> None:
        """Test extraction from technical architecture discussion."""
        transcript = """
        User: We need to implement caching for our API. What are our options?

        Assistant: You have several options:
        1. Redis - Great for distributed caching, supports complex data types
        2. Memcached - Simpler, faster for basic key-value caching
        3. In-memory (like lru_cache) - Good for single-instance apps

        User: We're running multiple instances, so Redis sounds right.

        Assistant: Good choice. Use Redis with a TTL (time-to-live) for cache entries.
        For Python, the redis-py library works well. Consider using redis-om for
        object mapping if you're storing complex objects.

        User: We decided on Redis with a 5-minute TTL for most endpoints,
        and 1-hour TTL for expensive computations.

        Assistant: Smart approach! Document this decision. Also consider adding
        cache invalidation logic for when underlying data changes.
        """

        result = await extractor.extract_from_transcript(
            transcript=transcript,
            project="api-caching",
        )

        assert result.success
        assert result.memory_count > 0
        assert result.summary  # Should summarize the caching discussion


class TestRateLimiting:
    """Test rate limiting behavior."""

    @pytest.mark.integration
    async def test_multiple_extractions_rate_limited(self, api_key: str) -> None:
        """Test that multiple extractions respect rate limits."""
        config = ExtractionConfig(
            model="claude-sonnet-4-20250514",
            rate_limit_rpm=10,  # Low limit for testing
        )
        extractor = MemoryExtractor(config=config, api_key=api_key)

        transcript = """
        User: Quick question about Python imports.
        Assistant: Sure, what do you need to know?
        User: Should I use absolute or relative imports?
        Assistant: Prefer absolute imports for clarity. They're more explicit
        and work better with tools like mypy and IDEs.
        """ * 3

        # Make a few extractions - should work without hitting rate limit
        for i in range(2):
            result = await extractor.extract_from_transcript(
                transcript=transcript,
                project=f"test-{i}",
            )
            assert result.success or "rate" in (result.error or "").lower()

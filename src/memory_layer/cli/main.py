"""Memory Layer CLI.

Command-line interface for interacting with the Memory Layer.
Provides commands for memory management, context injection, and hook support.

Usage:
    mem add <content> [-c category]
    mem search <query> [--limit N]
    mem context [--inject] [--format FMT]
    mem outcome <id> worked|failed|partial
    mem stats
    mem serve --mcp|--rest
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import click

from memory_layer import __version__
from memory_layer.core.logging import get_logger, setup_logging
from memory_layer.core.models import MemoryCategory, MemoryScope, MemorySource, Outcome


logger = get_logger(__name__)


def get_engine():
    """Get or create an initialized MemoryEngine instance."""
    from memory_layer.core.engine import EngineConfig, MemoryEngine

    db_path = os.environ.get(
        "MEMORY_LAYER_DB",
        str(Path.home() / ".memory-layer" / "memories.db")
    )
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    config = EngineConfig(db_path=db_path)
    engine = MemoryEngine(config=config)
    # Initialize the engine synchronously
    asyncio.get_event_loop().run_until_complete(engine.initialize())
    return engine


def run_async(coro):
    """Run an async coroutine."""
    return asyncio.get_event_loop().run_until_complete(coro)


def format_error(error: Exception, verbose: bool = False) -> str:
    """Convert exception to user-friendly error message.

    Args:
        error: The exception to format
        verbose: If True, include technical details

    Returns:
        User-friendly error message
    """
    error_str = str(error)

    # Handle Pydantic validation errors
    if "validation error" in error_str.lower():
        # Extract the key issue
        if "content" in error_str.lower() and "at least 1 character" in error_str.lower():
            return "Content cannot be empty."
        if "content" in error_str.lower() and "string_too_short" in error_str.lower():
            return "Content cannot be empty."
        if "importance" in error_str.lower():
            return "Importance must be a number between 0.0 and 1.0."
        if "category" in error_str.lower():
            return "Invalid category. Use: architecture, convention, decision, pattern, gotcha, workaround, troubleshooting, command, preference, general."
        # Generic validation error
        if verbose:
            return error_str
        return "Invalid input. Use -v flag for details."

    # Handle common errors
    if "No memory found" in error_str:
        return error_str  # Already user-friendly
    if "not found" in error_str.lower():
        return error_str
    if "database" in error_str.lower() and "locked" in error_str.lower():
        return "Database is locked. Another process may be using it."
    if "permission denied" in error_str.lower():
        return "Permission denied. Check file permissions."

    # Default: return original message (but strip Pydantic URLs)
    if "pydantic" in error_str.lower() and "https://" in error_str:
        # Remove the URL part
        lines = error_str.split('\n')
        filtered = [l for l in lines if "https://errors.pydantic.dev" not in l]
        return '\n'.join(filtered).strip()

    return error_str


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group()
@click.version_option(version=__version__, prog_name="Memory Layer")
@click.option("--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)")
@click.option("--json-output", is_flag=True, help="Output in JSON format")
@click.pass_context
def cli(ctx: click.Context, verbose: int, json_output: bool) -> None:
    """Memory Layer - Persistent memory for AI coding agents.

    Store, search, and manage memories with outcome-based learning.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["json_output"] = json_output

    # Default to WARNING (silent), -v for INFO, -vv for DEBUG
    if verbose >= 2:
        setup_logging(level="DEBUG")
    elif verbose == 1:
        setup_logging(level="INFO")
    else:
        setup_logging(level="WARNING")


# =============================================================================
# Core Commands
# =============================================================================


@cli.command("add")
@click.argument("content")
@click.option(
    "-c", "--category",
    type=click.Choice([c.value for c in MemoryCategory], case_sensitive=False),
    default="general",
    help="Memory category"
)
@click.option("-p", "--project", default=None, help="Project name/path")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--importance", type=float, default=0.5, help="Importance (0.0-1.0)")
@click.pass_context
def add_memory(
    ctx: click.Context,
    content: str,
    category: str,
    project: Optional[str],
    tags: str,
    importance: float,
) -> None:
    """Add a new memory.

    Example:
        mem add "Use snake_case for Python variables" -c convention
    """
    engine = get_engine()

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Use current directory as project if not specified
    if project is None:
        project = Path.cwd().name

    try:
        memory = run_async(engine.add(
            content=content,
            category=MemoryCategory(category),
            project=project,
            tags=tag_list,
            importance=importance,
            source=MemorySource.EXPLICIT,
        ))

        if ctx.obj.get("json_output"):
            click.echo(json.dumps(memory.to_dict(), indent=2, default=str))
        else:
            click.echo(f"Added memory [{memory.id[:8]}] ({category})")
            click.echo(f"  {content[:80]}{'...' if len(content) > 80 else ''}")
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


@cli.command("search")
@click.argument("query")
@click.option("-l", "--limit", default=5, help="Maximum results")
@click.option(
    "-c", "--category",
    type=click.Choice([c.value for c in MemoryCategory], case_sensitive=False),
    default=None,
    help="Filter by category"
)
@click.option("-p", "--project", default=None, help="Filter by project")
@click.option("--min-score", type=float, default=-1.0, help="Minimum outcome score")
@click.option("--format", "output_format", default="brief", help="Output format (brief/detailed/context)")
@click.option("--full", is_flag=True, help="Show full content (no truncation)")
@click.pass_context
def search_memories(
    ctx: click.Context,
    query: str,
    limit: int,
    category: Optional[str],
    project: Optional[str],
    min_score: float,
    output_format: str,
    full: bool,
) -> None:
    """Search memories.

    Example:
        mem search "authentication" --limit 10
    """
    engine = get_engine()

    cat = MemoryCategory(category) if category else None

    try:
        results = run_async(engine.search(
            query=query,
            limit=limit,
            category=cat,
            project=project,
            min_score=min_score,
        ))

        if ctx.obj.get("json_output"):
            click.echo(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        elif output_format == "context":
            # Format for context injection
            from memory_layer.plugin import ContextFormatter
            memories = [r.memory for r in results]
            click.echo(ContextFormatter.format_for_injection(memories, style="markdown"))
        elif output_format == "detailed":
            for r in results:
                m = r.memory
                project_tag = f"[{m.project}]" if m.project else "[global]"
                click.echo(f"\n[{m.id[:8]}] {m.category.value.upper()} {project_tag} (score: {r.score:.2f})")
                click.echo(f"  {m.content}")
                click.echo(f"  Outcome: {m.outcome_score:.2f} | Used: {m.use_count}x")
        else:
            # Brief format
            if not results:
                click.echo("No memories found.")
            for r in results:
                m = r.memory
                project_tag = f"[{m.project}]" if m.project else "[global]"
                content = m.content if full else f"{m.content[:60]}..."
                click.echo(f"[{m.id[:8]}] [{m.category.value}] {project_tag} {content}")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


@cli.command("show")
@click.argument("memory_id")
@click.pass_context
def show_memory(ctx: click.Context, memory_id: str) -> None:
    """Show details of a specific memory.

    Supports partial ID matching.

    Example:
        mem show abc12345
    """
    engine = get_engine()

    try:
        # Support partial ID matching
        full_id = memory_id
        if len(memory_id) < 32:
            memories = run_async(engine.list(limit=1000))
            matches = [m for m in memories if m.id.startswith(memory_id)]
            if len(matches) == 0:
                raise click.ClickException(f"No memory found matching: {memory_id}")
            elif len(matches) > 1:
                msg = f"Ambiguous ID '{memory_id}' matches {len(matches)} memories:\n"
                for m in matches:
                    msg += f"  [{m.id[:16]}] {m.content[:40]}...\n"
                msg += "Use more characters to disambiguate."
                raise click.ClickException(msg)
            full_id = matches[0].id

        memory = run_async(engine.get(full_id))
        if memory is None:
            raise click.ClickException(f"Memory not found: {memory_id}")

        if ctx.obj.get("json_output"):
            click.echo(json.dumps(memory.to_dict(), indent=2, default=str))
        else:
            click.echo(f"ID: {memory.id}")
            click.echo(f"Category: {memory.category.value}")
            click.echo(f"Content: {memory.content}")
            click.echo(f"Outcome Score: {memory.outcome_score:.2f}")
            click.echo(f"Use Count: {memory.use_count}")
            click.echo(f"Confidence: {memory.confidence:.2f}")
            click.echo(f"Project: {memory.project or 'global'}")
            click.echo(f"Tags: {', '.join(memory.tags) if memory.tags else 'none'}")
            click.echo(f"Created: {memory.created_at}")
    except Exception as e:
        logger.error(f"Failed to show memory: {e}")
        raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


@cli.command("list")
@click.option("-l", "--limit", default=20, help="Maximum results")
@click.option(
    "-c", "--category",
    type=click.Choice([c.value for c in MemoryCategory], case_sensitive=False),
    default=None,
    help="Filter by category"
)
@click.option("-p", "--project", default=None, help="Filter by project")
@click.option("--archived", is_flag=True, help="Include archived memories")
@click.option("--full", is_flag=True, help="Show full content (no truncation)")
@click.pass_context
def list_memories(
    ctx: click.Context,
    limit: int,
    category: Optional[str],
    project: Optional[str],
    archived: bool,
    full: bool,
) -> None:
    """List memories with optional filters.

    Example:
        mem list -c convention --limit 10
    """
    engine = get_engine()

    try:
        memories = run_async(engine.list(
            limit=limit,
            category=MemoryCategory(category) if category else None,
            project=project,
            include_archived=archived,
        ))

        if ctx.obj.get("json_output"):
            click.echo(json.dumps([m.to_dict() for m in memories], indent=2, default=str))
        else:
            if not memories:
                click.echo("No memories found.")
            for m in memories:
                score_indicator = ""
                if m.outcome_score > 0.3:
                    score_indicator = " [+]"
                elif m.outcome_score < -0.2:
                    score_indicator = " [-]"
                project_tag = f"[{m.project}]" if m.project else "[global]"
                content = m.content if full else f"{m.content[:50]}..."
                click.echo(f"[{m.id[:8]}] [{m.category.value}] {project_tag} {content}{score_indicator}")
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


@cli.command("delete")
@click.argument("memory_id")
@click.option("--confirm", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_memory(ctx: click.Context, memory_id: str, confirm: bool) -> None:
    """Archive (soft delete) a memory.

    Supports partial ID matching.

    Example:
        mem delete abc12345 --confirm
    """
    engine = get_engine()

    try:
        # Support partial ID matching
        full_id = memory_id
        if len(memory_id) < 32:
            memories = run_async(engine.list(limit=1000))
            matches = [m for m in memories if m.id.startswith(memory_id)]
            if len(matches) == 0:
                raise click.ClickException(f"No memory found matching: {memory_id}")
            elif len(matches) > 1:
                msg = f"Ambiguous ID '{memory_id}' matches {len(matches)} memories:\n"
                for m in matches:
                    msg += f"  [{m.id[:16]}] {m.content[:40]}...\n"
                msg += "Use more characters to disambiguate."
                raise click.ClickException(msg)
            full_id = matches[0].id

        if not confirm:
            if not click.confirm(f"Archive memory {memory_id}?"):
                click.echo("Cancelled.")
                return

        run_async(engine.delete(full_id))
        click.echo(f"Archived memory {memory_id}")
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


@cli.command("outcome")
@click.argument("memory_id")
@click.argument("result", type=click.Choice(["worked", "failed", "partial"]))
@click.pass_context
def record_outcome(ctx: click.Context, memory_id: str, result: str) -> None:
    """Record outcome feedback for a memory.

    Supports partial ID matching (first 8 chars shown by other commands).

    Example:
        mem outcome abc12345 worked
    """
    engine = get_engine()

    try:
        # Support partial ID matching
        full_id = memory_id
        if len(memory_id) < 32:  # Partial ID provided
            memories = run_async(engine.list(limit=1000))
            matches = [m for m in memories if m.id.startswith(memory_id)]
            if len(matches) == 0:
                raise click.ClickException(f"No memory found matching: {memory_id}")
            elif len(matches) > 1:
                msg = f"Ambiguous ID '{memory_id}' matches {len(matches)} memories:\n"
                for m in matches:
                    msg += f"  [{m.id[:16]}] {m.content[:40]}...\n"
                msg += "Use more characters to disambiguate."
                raise click.ClickException(msg)
            full_id = matches[0].id

        success = run_async(engine.record_outcome(
            memory_ids=[full_id],
            outcome=Outcome(result),
        ))
        if success:
            adjustment = {
                "worked": "+0.2",
                "failed": "-0.3",
                "partial": "+0.05",
            }[result]
            click.echo(f"Recorded '{result}' for {memory_id} ({adjustment})")
        else:
            raise click.ClickException(f"Memory not found: {memory_id}")
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")
        raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


# =============================================================================
# Context Commands (for hooks)
# =============================================================================


@cli.command("context")
@click.option("-p", "--project", default=None, help="Project path")
@click.option("--inject", is_flag=True, help="Format for injection (used by hooks)")
@click.option("-l", "--limit", default=10, help="Maximum memories")
@click.option(
    "--format", "output_format",
    type=click.Choice(["brief", "detailed", "structured", "markdown", "silent", "json"]),
    default="brief",
    help="Output format"
)
@click.pass_context
def get_context(
    ctx: click.Context,
    project: Optional[str],
    inject: bool,
    limit: int,
    output_format: str,
) -> None:
    """Get project memory context.

    Used by SessionStart hook to inject context.

    Example:
        mem context --project /path/to/project --inject --limit 10
    """
    engine = get_engine()

    # Use current directory as project if not specified
    if project is None:
        project = os.environ.get("PWD", str(Path.cwd()))

    # Extract project name from path
    project_name = Path(project).name

    try:
        context_response = run_async(engine.get_context(
            project=project_name,
            max_memories=limit,
        ))

        if output_format == "silent":
            # For hook usage - inject into context without output
            if context_response.memories:
                from memory_layer.plugin import ContextFormatter
                # Write to a context file that Claude Code can read
                context_dir = Path.home() / ".memory-layer" / "context"
                context_dir.mkdir(parents=True, exist_ok=True)
                context_file = context_dir / f"{project_name}.context"
                formatted = ContextFormatter.format_for_injection(
                    context_response.memories,
                    style="markdown"
                )
                context_file.write_text(formatted)
            return

        if output_format == "json" or ctx.obj.get("json_output"):
            click.echo(json.dumps(context_response.to_dict(), indent=2, default=str))
        elif inject:
            from memory_layer.plugin import ContextFormatter
            click.echo(ContextFormatter.format_for_injection(
                context_response.memories,
                style="markdown"
            ))
        else:
            from memory_layer.plugin import ContextFormatter
            click.echo(ContextFormatter.format_for_injection(
                context_response.memories,
                style=output_format
            ))
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        if output_format != "silent":
            raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


@cli.command("extract")
@click.option("--auto", "auto_extract", is_flag=True, help="Auto-extract from context")
@click.option("--session", "session_id", default=None, help="Claude session ID")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
@click.option("--from-context", is_flag=True, help="Read from current context")
@click.option("--stdin", "from_stdin", is_flag=True, help="Read transcript from stdin")
@click.pass_context
def extract_memories(
    ctx: click.Context,
    auto_extract: bool,
    session_id: Optional[str],
    quiet: bool,
    from_context: bool,
    from_stdin: bool,
) -> None:
    """Extract memories from context/transcript.

    Used by PreCompact hook to extract learnings before compaction.

    Examples:
        mem extract --auto --session $CLAUDE_SESSION_ID --quiet
        cat transcript.txt | mem extract --stdin -p myproject
        echo "We decided to use PostgreSQL for the database" | mem extract --stdin
    """
    # Get session ID from environment if not provided
    if session_id is None:
        session_id = os.environ.get("CLAUDE_SESSION_ID")

    # Check for stdin input
    stdin_content = None
    if from_stdin or (not sys.stdin.isatty()):
        try:
            # Read from stdin if available and not a TTY
            if not sys.stdin.isatty():
                stdin_content = sys.stdin.read().strip()
        except Exception:
            pass

    if stdin_content:
        # We have content from stdin - could extract from it
        if not quiet:
            click.echo(f"Received {len(stdin_content)} characters from stdin")
            click.echo(f"Session: {session_id or 'unknown'}")

        # TODO: Use the extractor module to extract memories from stdin_content
        # For now, just acknowledge receipt
        if not quiet:
            click.echo("Note: Extraction from stdin content is pending extractor integration.")
    else:
        # No stdin content - standard extraction trigger
        if not quiet:
            click.echo(f"Extraction triggered for session: {session_id or 'unknown'}")
            click.echo("Note: Auto-extraction requires conversation context access.")

    # TODO: Implement actual extraction when Claude Code provides transcript access
    # This would use the extractor module to extract memories from the transcript


# =============================================================================
# Session Commands (for hooks)
# =============================================================================


@cli.group("session")
def session_group() -> None:
    """Session management commands."""
    pass


@session_group.command("end")
@click.option("--session", "session_id", default=None, help="Claude session ID")
@click.option("--summarize", is_flag=True, help="Generate session summary")
@click.pass_context
def end_session(
    ctx: click.Context,
    session_id: Optional[str],
    summarize: bool,
) -> None:
    """End a memory session.

    Used by SessionEnd hook.

    Example:
        mem session end --session $CLAUDE_SESSION_ID --summarize
    """
    from memory_layer.plugin import SessionManager

    # Get session ID from environment if not provided
    if session_id is None:
        session_id = os.environ.get("CLAUDE_SESSION_ID")

    session_manager = SessionManager()
    session_manager._current_session = session_id

    summary = session_manager.end_session(summarize=summarize)

    if summary and not ctx.obj.get("json_output"):
        click.echo(f"Session ended: {summary.get('session_id', 'unknown')}")
        click.echo(f"  Memories used: {summary.get('memories_used', 0)}")
    elif summary:
        click.echo(json.dumps(summary, indent=2, default=str))


@session_group.command("start")
@click.option("--session", "session_id", default=None, help="Claude session ID")
@click.pass_context
def start_session(ctx: click.Context, session_id: Optional[str]) -> None:
    """Start a memory session.

    Example:
        mem session start --session $CLAUDE_SESSION_ID
    """
    from memory_layer.plugin import SessionManager

    if session_id is None:
        session_id = os.environ.get("CLAUDE_SESSION_ID")

    session_manager = SessionManager()
    active_id = session_manager.start_session(session_id)

    if ctx.obj.get("json_output"):
        click.echo(json.dumps({"session_id": active_id}))
    else:
        click.echo(f"Session started: {active_id}")


# =============================================================================
# File Tracking (for PostToolUse hook)
# =============================================================================


@cli.command("track-file")
@click.argument("file_path")
@click.option("--session", "session_id", default=None, help="Claude session ID")
@click.pass_context
def track_file(
    ctx: click.Context,
    file_path: str,
    session_id: Optional[str],
) -> None:
    """Track a modified file.

    Used by PostToolUse hook for Write/Edit operations.
    Silently succeeds/fails for hook usage.

    Example:
        mem track-file /path/to/file.py --session $CLAUDE_SESSION_ID
    """
    # Get session ID from environment if not provided
    if session_id is None:
        session_id = os.environ.get("CLAUDE_SESSION_ID")

    # TODO: Implement file tracking
    # This would record which files were modified during the session
    # for context-aware memory retrieval
    pass  # Silent success for hook usage


# =============================================================================
# Statistics
# =============================================================================


@cli.command("stats")
@click.option("-p", "--project", default=None, help="Filter by project")
@click.pass_context
def show_stats(ctx: click.Context, project: Optional[str]) -> None:
    """Show memory statistics.

    Example:
        mem stats
    """
    engine = get_engine()

    try:
        stats = run_async(engine.stats(project=project))
        storage = stats.storage_stats

        if ctx.obj.get("json_output"):
            from dataclasses import asdict
            click.echo(json.dumps(asdict(stats), indent=2, default=str))
        else:
            click.echo("Memory Layer Statistics")
            click.echo("=" * 40)
            click.echo(f"Total memories: {storage.total_memories}")
            click.echo(f"Active: {storage.active_memories}")
            click.echo(f"Archived: {storage.archived_memories}")
            click.echo(f"Average outcome score: {storage.avg_outcome_score:.2f}")
            click.echo(f"Total uses: {storage.total_uses}")
            click.echo(f"Indexed in retriever: {stats.indexed_memories}")
            click.echo()
            if storage.by_category:
                click.echo("By Category:")
                for cat, count in storage.by_category.items():
                    click.echo(f"  {cat}: {count}")
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


@cli.command("check")
@click.option("--fix", is_flag=True, help="Attempt to fix issues")
@click.pass_context
def check_health(ctx: click.Context, fix: bool) -> None:
    """Check Memory Layer health and configuration.

    Verifies database, embeddings, and configuration are working correctly.

    Example:
        mem check
        mem check --fix
    """
    issues = []
    fixes_applied = []

    click.echo("Memory Layer Health Check")
    click.echo("=" * 40)

    # Check 1: Database location and access
    db_path = os.environ.get(
        "MEMORY_LAYER_DB",
        str(Path.home() / ".memory-layer" / "memories.db")
    )
    db_exists = Path(db_path).exists()
    db_dir_exists = Path(db_path).parent.exists()

    if db_dir_exists:
        click.echo(f"[OK] Database directory: {Path(db_path).parent}")
    else:
        if fix:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            fixes_applied.append("Created database directory")
            click.echo(f"[FIXED] Created database directory: {Path(db_path).parent}")
        else:
            issues.append(f"Database directory missing: {Path(db_path).parent}")
            click.echo(f"[ISSUE] Database directory missing: {Path(db_path).parent}")

    if db_exists:
        click.echo(f"[OK] Database file: {db_path}")
    else:
        click.echo(f"[INFO] Database file will be created on first use: {db_path}")

    # Check 2: Engine initialization
    try:
        engine = get_engine()
        click.echo("[OK] Engine initialization")

        # Check 3: Memory count
        stats = run_async(engine.stats())
        total = stats.storage_stats.total_memories
        click.echo(f"[OK] Memory count: {total} memories")

        # Check 4: Embedding provider
        if hasattr(engine, '_retriever') and engine._retriever:
            click.echo("[OK] Retriever initialized")
        else:
            click.echo("[INFO] Retriever not yet initialized (will init on first search)")

    except Exception as e:
        issues.append(f"Engine initialization failed: {e}")
        click.echo(f"[ISSUE] Engine initialization failed: {e}")

    # Check 5: Environment variables
    click.echo()
    click.echo("Configuration:")
    click.echo(f"  MEMORY_LAYER_DB: {os.environ.get('MEMORY_LAYER_DB', '(default)')}")
    click.echo(f"  ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'not set'}")

    # Summary
    click.echo()
    click.echo("-" * 40)
    if issues:
        click.echo(f"Issues found: {len(issues)}")
        for issue in issues:
            click.echo(f"  - {issue}")
        if not fix:
            click.echo("\nRun 'mem check --fix' to attempt automatic fixes.")
    else:
        click.echo("All checks passed!")

    if fixes_applied:
        click.echo(f"\nFixes applied: {len(fixes_applied)}")
        for f in fixes_applied:
            click.echo(f"  - {f}")


# =============================================================================
# Server Commands
# =============================================================================


@cli.command("serve")
@click.option("--mcp", "serve_mcp", is_flag=True, help="Start MCP server")
@click.option("--rest", "serve_rest", is_flag=True, help="Start REST API server")
@click.option("--port", default=8080, help="Port for REST server")
@click.option("--host", default="127.0.0.1", help="Host for REST server")
@click.pass_context
def serve(
    ctx: click.Context,
    serve_mcp: bool,
    serve_rest: bool,
    port: int,
    host: str,
) -> None:
    """Start a memory server.

    Example:
        mem serve --mcp
        mem serve --rest --port 8080
    """
    if not serve_mcp and not serve_rest:
        raise click.ClickException("Specify --mcp or --rest")

    if serve_mcp:
        click.echo("Starting MCP server on stdio...", err=True)
        from memory_layer.server import run_mcp_server
        asyncio.run(run_mcp_server())
    elif serve_rest:
        click.echo(f"Starting REST server on {host}:{port}...", err=True)
        from memory_layer.server import APIConfig, run_server

        # Get API key from environment (optional)
        api_key = os.environ.get("MEMORY_LAYER_API_KEY")

        # Get engine
        engine = get_engine()

        # Create config
        config = APIConfig(api_key=api_key)

        # Run the server
        run_server(config=config, engine=engine, host=host, port=port)


# =============================================================================
# Utility Commands
# =============================================================================


@cli.command("ingest")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("-p", "--project", default=None, help="Project name")
@click.pass_context
def ingest_file(ctx: click.Context, file_path: str, project: Optional[str]) -> None:
    """Ingest memories from a file.

    Example:
        mem ingest transcript.txt -p myproject
    """
    click.echo(f"Ingesting from {file_path}...")
    # TODO: Implement file ingestion using extractor
    click.echo("File ingestion not yet implemented")


@cli.command("export")
@click.argument("output_file", type=click.Path())
@click.option(
    "--format", "output_format",
    type=click.Choice(["json", "md", "markdown"]),
    default="json",
    help="Export format"
)
@click.option("-p", "--project", default=None, help="Filter by project")
@click.pass_context
def export_memories(
    ctx: click.Context,
    output_file: str,
    output_format: str,
    project: Optional[str],
) -> None:
    """Export memories to a file.

    Example:
        mem export memories.json --format json
    """
    engine = get_engine()

    try:
        memories = run_async(engine.list(limit=1000, project=project))

        if output_format == "json":
            data = [m.to_dict() for m in memories]
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            # Markdown format
            from memory_layer.plugin import ContextFormatter
            formatted = ContextFormatter.format_for_injection(memories, style="markdown")
            with open(output_file, "w") as f:
                f.write(formatted)

        click.echo(f"Exported {len(memories)} memories to {output_file}")
    except Exception as e:
        logger.error(f"Failed to export: {e}")
        raise click.ClickException(format_error(e, ctx.obj.get("verbose", 0) > 0))


# =============================================================================
# Plugin Installation
# =============================================================================


SETTINGS_JSON = """{
  "enableAllProjectMcpServers": true,
  "enabledMcpjsonServers": [
    "memory-layer"
  ],
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "mem context --project \\"$PWD\\" --limit 10 --format brief"
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "mem session end --summarize 2>/dev/null || true"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "mem track-file \\"$TOOL_INPUT_FILE_PATH\\" 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
"""

PLUGIN_JSON = """{
  "name": "memory-layer",
  "description": "Persistent memory with outcome-based learning for AI coding agents.",
  "version": "__VERSION__",
  "author": "memory-layer-contributors",
  "license": "MIT",
  "capabilities": {
    "hooks": true,
    "commands": true,
    "skills": true,
    "mcp": true
  }
}
"""

MCP_JSON = """{
  "mcpServers": {
    "memory-layer": {
      "command": "mem",
      "args": ["serve", "--mcp"],
      "env": {
        "MEMORY_LAYER_DB": "~/.memory-layer/memories.db"
      }
    }
  }
}
"""

REMEMBER_CMD = """---
description: Store a memory for future reference
allowed-tools: Bash
---

Store information in Memory Layer for future sessions.

## Usage

`/remember <content>` - Store with auto-detected category
`/remember category:<cat> <content>` - Store with specific category

## Categories

architecture, convention, decision, pattern, gotcha, workaround, troubleshooting, command, preference

## Examples

- `/remember Always use async/await in this project`
- `/remember category:gotcha Watch out for N+1 queries`

## Action

```bash
mem add "<content>" -c <category>
```

Confirm storage with the memory ID shown.
"""

RECALL_CMD = """---
description: Search memories for relevant information
allowed-tools: Bash
---

Search Memory Layer for relevant past knowledge.

## Usage

`/recall <query>` - Search memories
`/recall <query> limit:N` - Limit results

## Action

```bash
mem search "<query>" --limit 5 --format context
```

Present results naturally, noting which memories have high confidence (outcome score > 0.3).
"""

OUTCOME_CMD = """---
description: Record feedback on whether a memory helped
allowed-tools: Bash
---

Record outcome feedback to improve memory rankings.

## Usage

`/outcome <id> worked` - Memory was helpful (+0.2)
`/outcome <id> failed` - Memory was wrong (-0.3)
`/outcome <id> partial` - Partially helpful (+0.05)

## Action

```bash
mem outcome <id> <result>
```

Confirm the score adjustment.
"""

MEMORIES_CMD = """---
description: List stored memories
allowed-tools: Bash
---

List all memories in Memory Layer.

## Usage

`/memories` - List all
`/memories -c <category>` - Filter by category

## Action

```bash
mem list --limit 20
```
"""

FORGET_CMD = """---
description: Archive a memory
allowed-tools: Bash
---

Archive (soft-delete) a memory that's no longer relevant.

## Usage

`/forget <id>`

## Action

```bash
mem delete <id> --confirm
```

Confirm archival.
"""

MEMORY_CONTEXT_CMD = """---
description: Get project memory context
allowed-tools: Bash
---

Get formatted memory context for the current project.

## Usage

`/memory-context`

## Action

```bash
mem context --format markdown
```
"""

MEMORY_RETRIEVAL_SKILL = """---
description: Automatically retrieve relevant memories when user asks about past decisions
allowed-tools: Bash, Read
user-invocable: false
---

# Memory Retrieval Skill

Activate when user asks about past decisions, conventions, or patterns:
- "what did we decide about..."
- "what's our convention for..."
- "how do we handle..."
- "last time we..."

## Action

Search memories and incorporate into response:

```bash
mem search "<keywords>" --limit 5 --format context
```

Cite memory IDs for transparency. Suggest `/outcome <id> worked` if helpful.
"""

OUTCOME_FEEDBACK_SKILL = """---
description: Detect natural feedback signals and prompt for outcome recording
allowed-tools: Bash
user-invocable: false
---

# Outcome Feedback Skill

Detect when user signals success or failure:

**Positive**: "thanks!", "that worked!", "perfect!", "solved it"
**Negative**: "still not working", "didn't help", "same error"
**Partial**: "kind of", "partially", "almost"

## Action

When detected, offer to record feedback:

```bash
mem outcome <last-used-memory-id> worked|failed|partial
```

Be non-intrusive - only prompt once per memory per session.
"""

CODING_PATTERNS_SKILL = """---
description: Surface relevant coding patterns when implementing new features
allowed-tools: Bash, Read
user-invocable: false
---

# Coding Patterns Skill

Activate when user is creating new code:
- "create a new..."
- "implement..."
- "write a..."
- "add a new..."

## Action

Search for relevant patterns:

```bash
mem search "<feature-type> pattern" -c pattern --limit 3
```

Incorporate patterns into implementation suggestions.
"""


@cli.command("install-plugin")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
@click.pass_context
def install_plugin(ctx: click.Context, force: bool) -> None:
    """Install Claude Code plugin files in the current directory.

    Creates:
    - .claude/settings.json (hooks configuration)
    - .claude/commands/ (slash commands)
    - .claude/skills/ (agent skills)
    - .claude-plugin/plugin.json (plugin manifest)
    - .mcp.json (MCP server configuration)

    Example:
        cd your-project
        mem install-plugin
        claude
    """
    cwd = Path.cwd()

    # Create directories
    claude_dir = cwd / ".claude"
    commands_dir = claude_dir / "commands"
    skills_dir = claude_dir / "skills"
    plugin_dir = cwd / ".claude-plugin"

    dirs_to_create = [
        claude_dir,
        commands_dir,
        skills_dir / "memory-retrieval",
        skills_dir / "outcome-feedback",
        skills_dir / "coding-patterns",
        plugin_dir,
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # Files to create
    files = [
        (claude_dir / "settings.json", SETTINGS_JSON),
        (plugin_dir / "plugin.json", PLUGIN_JSON),
        (cwd / ".mcp.json", MCP_JSON),
        (commands_dir / "remember.md", REMEMBER_CMD),
        (commands_dir / "recall.md", RECALL_CMD),
        (commands_dir / "outcome.md", OUTCOME_CMD),
        (commands_dir / "memories.md", MEMORIES_CMD),
        (commands_dir / "forget.md", FORGET_CMD),
        (commands_dir / "memory-context.md", MEMORY_CONTEXT_CMD),
        (skills_dir / "memory-retrieval" / "SKILL.md", MEMORY_RETRIEVAL_SKILL),
        (skills_dir / "outcome-feedback" / "SKILL.md", OUTCOME_FEEDBACK_SKILL),
        (skills_dir / "coding-patterns" / "SKILL.md", CODING_PATTERNS_SKILL),
    ]

    created = 0
    skipped = 0

    for filepath, content in files:
        if filepath.exists() and not force:
            skipped += 1
            click.echo(f"  Skipped (exists): {filepath.relative_to(cwd)}")
        else:
            filepath.write_text(content.strip().replace("__VERSION__", __version__) + "\n")
            created += 1
            click.echo(f"  Created: {filepath.relative_to(cwd)}")

    click.echo()
    click.echo(f"Plugin installed: {created} files created, {skipped} skipped")

    if skipped > 0:
        click.echo("Use --force to overwrite existing files")

    click.echo()
    click.echo("Next steps:")
    click.echo("  1. Start Claude Code: claude")
    click.echo("  2. Try: /remember Always use async/await")
    click.echo("  3. Try: /recall async")


# =============================================================================
# Beads Integration Commands
# =============================================================================


@cli.command("beads-sync")
@click.option("--task", "-t", help="Sync specific task ID only")
@click.option("--dry-run", is_flag=True, help="Show what would happen without making changes")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output (for hooks)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def beads_sync(
    ctx: click.Context,
    task: Optional[str],
    dry_run: bool,
    quiet: bool,
    json_output: bool,
) -> None:
    """Sync outcomes for completed Beads tasks.

    Scans completed tasks and auto-records outcomes for linked memories.
    This is the core of Beads integration - when a task completes,
    memories that helped solve it get boosted.

    Examples:
        mem beads-sync                  # Sync all completed tasks
        mem beads-sync --task bd-a3f8   # Sync specific task
        mem beads-sync --dry-run        # Preview changes
    """
    from memory_layer.tasks import BeadsAdapter, BeadsSyncResult

    verbose = ctx.obj.get("verbose", 0)

    try:
        engine = get_engine()
        adapter = BeadsAdapter(engine)
        run_async(adapter.initialize())

        if not adapter.is_available:
            if not quiet:
                if json_output:
                    click.echo(json.dumps({"error": "Beads not found", "success": False}))
                else:
                    click.echo("No .beads/ directory found in project hierarchy.")
            ctx.exit(1)

        if task:
            # Sync specific task
            if dry_run:
                from memory_layer.tasks import TaskMemoryLinker
                linker = adapter._linker
                links = run_async(linker.get_unresolved_links(task))
                if not quiet:
                    if json_output:
                        click.echo(json.dumps({
                            "task_id": task,
                            "memories_to_update": len(links),
                            "dry_run": True,
                        }))
                    else:
                        click.echo(f"Would update {len(links)} memories for task {task}")
            else:
                beads_task = adapter.get_task(task)
                if not beads_task:
                    if not quiet:
                        click.echo(f"Task {task} not found")
                    ctx.exit(1)

                from memory_layer.tasks import BeadsTaskStatus
                if beads_task.status == BeadsTaskStatus.DONE:
                    count = run_async(adapter.on_task_done(task))
                elif beads_task.status == BeadsTaskStatus.CANCELLED:
                    count = run_async(adapter.on_task_cancelled(task))
                elif beads_task.status == BeadsTaskStatus.BLOCKED:
                    count = run_async(adapter.on_task_blocked(task))
                else:
                    count = 0
                    if not quiet:
                        click.echo(f"Task {task} is {beads_task.status.value}, no outcome to record")

                if not quiet and count > 0:
                    if json_output:
                        click.echo(json.dumps({
                            "task_id": task,
                            "outcomes_recorded": count,
                        }))
                    else:
                        click.echo(f"Recorded outcomes for {count} memories")
        else:
            # Sync all completed tasks
            result = run_async(adapter.sync())

            if not quiet:
                if json_output:
                    click.echo(json.dumps(result.to_dict()))
                else:
                    click.echo(f"Tasks found: {result.tasks_found}")
                    click.echo(f"Tasks synced: {result.tasks_synced}")
                    click.echo(f"Outcomes recorded: {result.outcomes_recorded}")
                    if result.errors:
                        click.echo(f"Errors: {len(result.errors)}")
                        if verbose:
                            for err in result.errors:
                                click.echo(f"  - {err}")

    except Exception as e:
        if not quiet:
            click.echo(f"Error: {format_error(e, verbose > 0)}", err=True)
        ctx.exit(1)


@cli.command("beads-context")
@click.option("--task", "-t", help="Task ID to get context for (default: current)")
@click.option("--limit", "-l", default=10, help="Max memories to include")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def beads_context(
    ctx: click.Context,
    task: Optional[str],
    limit: int,
    json_output: bool,
) -> None:
    """Get unified context for a Beads task.

    Combines task info with relevant memories for context injection.

    Examples:
        mem beads-context              # Context for current task
        mem beads-context -t bd-a3f8   # Context for specific task
        mem beads-context --json       # Output as JSON
    """
    from memory_layer.tasks import BeadsAdapter

    verbose = ctx.obj.get("verbose", 0)

    try:
        engine = get_engine()
        adapter = BeadsAdapter(engine)
        run_async(adapter.initialize())

        if not adapter.is_available:
            if json_output:
                click.echo(json.dumps({"error": "Beads not found"}))
            else:
                click.echo("No .beads/ directory found in project hierarchy.")
            ctx.exit(1)

        context = run_async(adapter.get_unified_context(task, limit))

        if not context:
            if json_output:
                click.echo(json.dumps({"error": "No task found"}))
            else:
                click.echo("No active task found.")
            ctx.exit(1)

        if json_output:
            click.echo(json.dumps({
                "task_id": context.task.id,
                "task_title": context.task.title,
                "task_status": context.task.status.value,
                "memories_count": len(context.memories),
                "formatted": context.formatted,
            }))
        else:
            click.echo(context.formatted)

    except Exception as e:
        click.echo(f"Error: {format_error(e, verbose > 0)}", err=True)
        ctx.exit(1)


@cli.command("beads-link")
@click.argument("memory_id")
@click.option("--task", "-t", help="Task ID to link to (default: current)")
@click.option("--context", "-c", help="Context about how memory is used")
@click.pass_context
def beads_link(
    ctx: click.Context,
    memory_id: str,
    task: Optional[str],
    context: Optional[str],
) -> None:
    """Link a memory to a Beads task.

    Manually link a memory to a task for outcome tracking.
    When the task completes, the memory's outcome will be recorded.

    Examples:
        mem beads-link mem-abc123                    # Link to current task
        mem beads-link mem-abc123 -t bd-a3f8        # Link to specific task
        mem beads-link mem-abc123 -c "used for auth"  # With context
    """
    from memory_layer.tasks import BeadsAdapter

    verbose = ctx.obj.get("verbose", 0)

    try:
        engine = get_engine()
        adapter = BeadsAdapter(engine)
        run_async(adapter.initialize())

        if not adapter.is_available:
            click.echo("No .beads/ directory found in project hierarchy.")
            ctx.exit(1)

        # Get task ID
        if task:
            target_task = adapter.get_task(task)
            if not target_task:
                click.echo(f"Task {task} not found")
                ctx.exit(1)
            task_id = task
        else:
            current = adapter.get_current_task()
            if not current:
                click.echo("No current task. Use --task to specify a task ID.")
                ctx.exit(1)
            task_id = current.id

        # Verify memory exists
        try:
            memory = run_async(engine.get(memory_id))
        except Exception:
            click.echo(f"Memory {memory_id} not found")
            ctx.exit(1)

        # Create link
        run_async(adapter.link_memory_to_task(task_id, memory_id, context))
        click.echo(f"Linked memory {memory_id} to task {task_id}")

    except Exception as e:
        click.echo(f"Error: {format_error(e, verbose > 0)}", err=True)
        ctx.exit(1)


@cli.command("beads-stats")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def beads_stats(ctx: click.Context, json_output: bool) -> None:
    """Show Beads integration statistics.

    Displays info about tasks, links, and sync status.

    Examples:
        mem beads-stats
        mem beads-stats --json
    """
    from memory_layer.tasks import BeadsAdapter

    verbose = ctx.obj.get("verbose", 0)

    try:
        engine = get_engine()
        adapter = BeadsAdapter(engine)
        run_async(adapter.initialize())

        stats = run_async(adapter.get_stats())

        if json_output:
            click.echo(json.dumps(stats, indent=2))
        else:
            click.echo("Beads Integration Stats")
            click.echo("=" * 40)
            click.echo(f"Beads available: {stats['beads_available']}")
            if stats['beads_dir']:
                click.echo(f"Beads directory: {stats['beads_dir']}")
            click.echo()

            if stats['beads_available']:
                click.echo("Tasks:")
                click.echo(f"  Total: {stats['tasks']['total_tasks']}")
                for status, count in stats['tasks']['by_status'].items():
                    click.echo(f"  {status}: {count}")
                click.echo()

            click.echo("Task-Memory Links:")
            click.echo(f"  Total links: {stats['links']['total_links']}")
            click.echo(f"  Unique tasks: {stats['links']['unique_tasks']}")
            click.echo(f"  Unique memories: {stats['links']['unique_memories']}")
            if stats['links']['by_outcome']:
                click.echo("  By outcome:")
                for outcome, count in stats['links']['by_outcome'].items():
                    click.echo(f"    {outcome}: {count}")
            click.echo()

            click.echo("Configuration:")
            click.echo(f"  Auto-outcome: {stats['auto_outcome_enabled']}")
            click.echo(f"  Outcome on cancel: {stats['outcome_on_cancel']}")

    except Exception as e:
        click.echo(f"Error: {format_error(e, verbose > 0)}", err=True)
        ctx.exit(1)


# =============================================================================
# Unified Tasks Commands (Phase 7 - Claude Code Tasks Adapter)
# =============================================================================


@cli.command("tasks")
@click.option("--source", "-s", type=click.Choice(["beads", "claude", "all"]), default="all",
              help="Task source filter")
@click.option("--status", type=click.Choice(["pending", "in_progress", "done", "completed", "all"]),
              default="all", help="Status filter")
@click.option("--limit", "-l", default=50, help="Maximum number of tasks to show")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def tasks_list(
    ctx: click.Context,
    source: str,
    status: str,
    limit: int,
    json_output: bool,
) -> None:
    """List tasks from all available sources.

    Shows tasks from both Beads (.beads/) and Claude Code (~/.claude/todos/)
    with unified formatting.

    Examples:
        mem tasks                      # List all tasks
        mem tasks --source claude      # Only Claude Code tasks
        mem tasks --source beads       # Only Beads tasks
        mem tasks --status pending     # Only pending tasks
        mem tasks --json               # Output as JSON
    """
    from memory_layer.tasks import UnifiedTaskAdapter, TaskSource

    verbose = ctx.obj.get("verbose", 0)

    try:
        engine = get_engine()
        adapter = UnifiedTaskAdapter(engine)
        run_async(adapter.initialize())

        # Determine source filter
        source_filter = None
        if source == "beads":
            source_filter = TaskSource.BEADS
        elif source == "claude":
            source_filter = TaskSource.CLAUDE_CODE

        # Determine status filter
        status_filter = None if status == "all" else status

        # Get tasks
        tasks = adapter.list_tasks(source=source_filter, status=status_filter)
        tasks = tasks[:limit]

        if json_output:
            output = {
                "tasks": [t.to_dict() for t in tasks],
                "total": len(tasks),
                "sources": [s.value for s in adapter.available_sources],
            }
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            if not tasks:
                click.echo("No tasks found.")
                click.echo(f"Available sources: {[s.value for s in adapter.available_sources]}")
                return

            click.echo(f"Tasks ({len(tasks)} found)")
            click.echo("=" * 60)

            for task in tasks:
                # Status indicator
                status_icons = {
                    "pending": "○",
                    "in_progress": "►",
                    "done": "✓",
                    "completed": "✓",
                    "blocked": "⊘",
                    "cancelled": "✗",
                }
                icon = status_icons.get(task.status, "?")

                # Source tag
                source_tag = "[B]" if task.source == TaskSource.BEADS else "[C]"

                # Title (truncated)
                title = task.title[:50] + "..." if len(task.title) > 50 else task.title

                click.echo(f"{icon} {source_tag} {task.id}: {title}")

            click.echo()
            click.echo(f"Sources: {[s.value for s in adapter.available_sources]}")

    except Exception as e:
        click.echo(f"Error: {format_error(e, verbose > 0)}", err=True)
        ctx.exit(1)


@cli.command("tasks-sync")
@click.option("--source", "-s", type=click.Choice(["beads", "claude", "all"]), default="all",
              help="Task source to sync")
@click.option("--task", "-t", help="Sync specific task ID only")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def tasks_sync(
    ctx: click.Context,
    source: str,
    task: Optional[str],
    json_output: bool,
) -> None:
    """Sync outcomes for completed tasks from all sources.

    Scans completed tasks and auto-records outcomes for linked memories.
    Works with both Beads and Claude Code tasks.

    Examples:
        mem tasks-sync                  # Sync all sources
        mem tasks-sync --source claude  # Only Claude Code
        mem tasks-sync --task cc-abc-0  # Sync specific task
        mem tasks-sync --json           # Output as JSON
    """
    from memory_layer.tasks import UnifiedTaskAdapter, TaskSource

    verbose = ctx.obj.get("verbose", 0)

    try:
        engine = get_engine()
        adapter = UnifiedTaskAdapter(engine)
        run_async(adapter.initialize())

        if task:
            # Sync specific task
            count = run_async(adapter.on_task_completed(task))
            if json_output:
                click.echo(json.dumps({
                    "task_id": task,
                    "outcomes_recorded": count,
                    "success": True,
                }))
            else:
                click.echo(f"Recorded outcomes for {count} memories on task {task}")
        else:
            # Sync all or by source
            source_filter = None
            if source == "beads":
                source_filter = TaskSource.BEADS
            elif source == "claude":
                source_filter = TaskSource.CLAUDE_CODE

            result = run_async(adapter.sync(source=source_filter))

            if json_output:
                click.echo(json.dumps(result.to_dict(), indent=2))
            else:
                if hasattr(result, 'results'):
                    # UnifiedSyncResult
                    click.echo("Task Sync Results")
                    click.echo("=" * 40)
                    click.echo(f"Total tasks found: {result.total_tasks_found}")
                    click.echo(f"Tasks synced: {result.total_tasks_synced}")
                    click.echo(f"Outcomes recorded: {result.total_outcomes_recorded}")
                    if result.errors:
                        click.echo(f"Errors: {len(result.errors)}")
                        for error in result.errors:
                            click.echo(f"  - {error}")
                else:
                    # Single TaskSyncResult
                    click.echo(f"Source: {result.source.value}")
                    click.echo(f"Tasks found: {result.tasks_found}")
                    click.echo(f"Tasks synced: {result.tasks_synced}")
                    click.echo(f"Outcomes recorded: {result.outcomes_recorded}")

    except Exception as e:
        click.echo(f"Error: {format_error(e, verbose > 0)}", err=True)
        ctx.exit(1)


@cli.command("tasks-context")
@click.option("--task", "-t", help="Task ID to get context for (default: current)")
@click.option("--source", "-s", type=click.Choice(["beads", "claude"]),
              help="Task source (auto-detected from ID)")
@click.option("--limit", "-l", default=10, help="Max memories to include")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def tasks_context(
    ctx: click.Context,
    task: Optional[str],
    source: Optional[str],
    limit: int,
    json_output: bool,
) -> None:
    """Get unified context for a task from any source.

    Combines task info with relevant memories for context injection.

    Examples:
        mem tasks-context               # Context for current task
        mem tasks-context -t cc-abc-0   # Context for Claude Code task
        mem tasks-context -t bd-a3f8    # Context for Beads task
        mem tasks-context --json        # Output as JSON
    """
    from memory_layer.tasks import UnifiedTaskAdapter, TaskSource

    verbose = ctx.obj.get("verbose", 0)

    try:
        engine = get_engine()
        adapter = UnifiedTaskAdapter(engine)
        run_async(adapter.initialize())

        # Determine source
        source_filter = None
        if source == "beads":
            source_filter = TaskSource.BEADS
        elif source == "claude":
            source_filter = TaskSource.CLAUDE_CODE

        context = run_async(adapter.get_unified_context(task, source_filter, limit))

        if not context:
            if json_output:
                click.echo(json.dumps({"error": "No task found"}))
            else:
                click.echo("No task found.")
                if not task:
                    click.echo("Tip: Specify a task ID with --task")
            ctx.exit(1)

        if json_output:
            output = {
                "task": context.task.to_dict(),
                "source": context.source.value,
                "memories": len(context.memories),
                "formatted": context.formatted,
            }
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            click.echo(context.formatted)

    except Exception as e:
        click.echo(f"Error: {format_error(e, verbose > 0)}", err=True)
        ctx.exit(1)


@cli.command("tasks-stats")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def tasks_stats(ctx: click.Context, json_output: bool) -> None:
    """Show statistics for all task integrations.

    Displays info about Beads and Claude Code tasks, links, and sync status.

    Examples:
        mem tasks-stats
        mem tasks-stats --json
    """
    from memory_layer.tasks import UnifiedTaskAdapter, TaskSource

    verbose = ctx.obj.get("verbose", 0)

    try:
        engine = get_engine()
        adapter = UnifiedTaskAdapter(engine)
        run_async(adapter.initialize())

        stats = run_async(adapter.get_stats())

        if json_output:
            click.echo(json.dumps(stats, indent=2))
        else:
            click.echo("Task Integration Statistics")
            click.echo("=" * 50)
            click.echo()
            click.echo(f"Available sources: {stats['available_sources']}")
            click.echo()

            # Beads stats
            if stats['beads']:
                beads = stats['beads']
                click.echo("Beads Integration")
                click.echo("-" * 30)
                click.echo(f"  Available: {beads.get('beads_available', False)}")
                if beads.get('beads_dir'):
                    click.echo(f"  Directory: {beads['beads_dir']}")
                if beads.get('tasks'):
                    click.echo(f"  Total tasks: {beads['tasks'].get('total_tasks', 0)}")
                    for status, count in beads['tasks'].get('by_status', {}).items():
                        click.echo(f"    {status}: {count}")
                click.echo()

            # Claude Code stats
            if stats['claude_code']:
                cc = stats['claude_code']
                click.echo("Claude Code Integration")
                click.echo("-" * 30)
                click.echo(f"  Available: {cc.get('claude_code_available', False)}")
                if cc.get('todos_dir'):
                    click.echo(f"  Directory: {cc['todos_dir']}")
                if cc.get('tasks'):
                    click.echo(f"  Total tasks: {cc['tasks'].get('total_tasks', 0)}")
                    click.echo(f"  Sessions: {cc['tasks'].get('sessions', 0)}")
                    for status, count in cc['tasks'].get('by_status', {}).items():
                        click.echo(f"    {status}: {count}")
                click.echo()

    except Exception as e:
        click.echo(f"Error: {format_error(e, verbose > 0)}", err=True)
        ctx.exit(1)


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()

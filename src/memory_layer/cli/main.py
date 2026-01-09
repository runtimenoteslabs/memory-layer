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


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--json-output", is_flag=True, help="Output in JSON format")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, json_output: bool) -> None:
    """Memory Layer - Persistent memory for AI coding agents.

    Store, search, and manage memories with outcome-based learning.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["json_output"] = json_output

    if verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO")


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
        raise click.ClickException(str(e))


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
@click.pass_context
def search_memories(
    ctx: click.Context,
    query: str,
    limit: int,
    category: Optional[str],
    project: Optional[str],
    min_score: float,
    output_format: str,
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
                click.echo(f"\n[{m.id[:8]}] {m.category.value.upper()} (score: {r.score:.2f})")
                click.echo(f"  {m.content}")
                click.echo(f"  Outcome: {m.outcome_score:.2f} | Used: {m.use_count}x")
        else:
            # Brief format
            if not results:
                click.echo("No memories found.")
            for r in results:
                m = r.memory
                click.echo(f"[{m.id[:8]}] [{m.category.value}] {m.content[:60]}...")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise click.ClickException(str(e))


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
        raise click.ClickException(str(e))


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
@click.pass_context
def list_memories(
    ctx: click.Context,
    limit: int,
    category: Optional[str],
    project: Optional[str],
    archived: bool,
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
                click.echo(f"[{m.id[:8]}] [{m.category.value}] {m.content[:50]}...{score_indicator}")
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        raise click.ClickException(str(e))


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
        raise click.ClickException(str(e))


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
        raise click.ClickException(str(e))


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
            raise click.ClickException(str(e))


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
        raise click.ClickException(str(e))


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
        raise click.ClickException(str(e))


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()

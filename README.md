# Runtime Memory

Persistent memory for AI coding agents with outcome-based learning.

> **New to Runtime Memory?** See the [User Guide](https://github.com/runtimenoteslabs/memory-layer/blob/main/USER_GUIDE.md) for an introduction to using Runtime Memory with Claude Code.

## What It Does

Runtime Memory stores knowledge from your coding sessions and records how often each memory worked and how often it failed. Among the memories relevant to a query, those with a better record rank higher. A memory that keeps failing is left out of retrieval until its failures fade.

## Installation

```bash
pip install runtime-memory
```

Or from source:

```bash
pip install git+https://github.com/runtimenoteslabs/memory-layer.git
```

For development:

```bash
git clone https://github.com/runtimenoteslabs/memory-layer.git
cd memory-layer
pip install -e ".[dev]"
```

The distribution is `runtime-memory` and the import is `runtime_memory`. The
repository is still named memory-layer, which is where the project started; the
package was renamed in 3.0. An unrelated package holds `memory-layer` on PyPI,
so `pip install memory-layer` fetches that one instead of this project.

**Note:** First run downloads an embedding model (~100MB) for semantic search. This happens once and is cached. Subsequent operations are fast (<100ms).

## Quick Start

### Python SDK

```python
from runtime_memory.sdk import MemoryClient

async with MemoryClient() as client:
    # Store a memory
    memory = await client.add(
        content="Use async/await for I/O operations",
        category="pattern",
    )

    # Search memories
    results = await client.search("async patterns", limit=5)

    # Record feedback
    await client.record_outcome(memory.id, "worked")

    # Get context for your project
    context = await client.get_context()
```

### Synchronous Client

```python
from runtime_memory.sdk import SyncMemoryClient

with SyncMemoryClient() as client:
    client.add("Always validate user input", category="convention")
    results = client.search("input validation")
```

### CLI

```bash
# Add a memory
mem add "Use type hints for better IDE support" -c convention

# Search memories
mem search "type hints"

# See why a search returns what it does, and why the rest were left out
mem why "type hints"

# Store statistics: outcome records, search mode, and the Hermes trace if present
mem stats

# Record outcome
mem outcome <memory-id> worked

# Get context
mem context

# Start REST API server
mem serve --rest --port 8080

# Start MCP server
mem serve --mcp
```

### REST API

```bash
# Start server
mem serve --rest --port 8080

# Add a memory
curl -X POST http://localhost:8080/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "Always use pytest", "category": "convention"}'

# Search
curl -X POST http://localhost:8080/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "testing"}'
```

### MCP Server

For multi-agent setups, Runtime Memory provides an MCP server:

```bash
mem serve --mcp
```

Configure in your MCP client:

```json
{
  "memory-layer": {
    "command": "mem",
    "args": ["serve", "--mcp"]
  }
}
```

#### Multi-Agent Configurations

All agents share the same memory store. Memories created in Claude Code appear in Cursor, feedback from OpenCode improves results everywhere.

**OpenCode** (`~/.opencode/config.json`):
```json
{
  "mcpServers": {
    "memory-layer": {
      "command": "mem",
      "args": ["serve", "--mcp"]
    }
  }
}
```

**Cursor** (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "memory-layer": {
      "command": "mem",
      "args": ["serve", "--mcp"]
    }
  }
}
```

**Windsurf** (`~/.windsurf/mcp.json`):
```json
{
  "mcpServers": {
    "memory-layer": {
      "command": "mem",
      "args": ["serve", "--mcp"]
    }
  }
}
```

### Claude Code Integration

Runtime Memory integrates with Claude Code via hooks and skills. For a beginner-friendly walkthrough, see the [User Guide](https://github.com/runtimenoteslabs/memory-layer/blob/main/USER_GUIDE.md).

**Installation:**

```bash
pip install runtime-memory

# Go to your project directory
cd your-project

# Install Claude Code plugin
mem install-plugin

# Start Claude Code
claude
```

The `mem install-plugin` command creates:
- `.claude/settings.json` - Hooks for SessionStart, SessionEnd, PostToolUse
- `.claude/commands/` - Slash commands (/remember, /recall, /outcome, etc.)
- `.claude/skills/` - Agent skills (memory-retrieval, outcome-feedback, coding-patterns)
- `.claude-plugin/plugin.json` - Plugin manifest
- `.mcp.json` - MCP server configuration

**What happens automatically:**

- **SessionStart hook**: Loads relevant memories when you start Claude Code
- **PreCompact hook**: Extracts learnings before context compaction (prevents losing insights)
- **PostToolUse hook**: Tracks files you edit for context
- **SessionEnd hook**: Generates session summary when you exit
- **Skills**: Auto-retrieval when you ask "what's our convention...", feedback detection when you say "thanks, that worked!"

**Slash commands in Claude Code:**

```
/remember <content>              # Store a memory
/remember category:gotcha <content>  # Store with category
/recall <query>                  # Search memories
/memories                        # List all memories
/outcome <id> worked|failed      # Record feedback
/forget <id>                     # Archive a memory
/memory-context                  # Get project context
```

### Task Integration (Beads + Claude Code)

Runtime Memory integrates with task trackers to automatically learn from task outcomes.

**Supported sources:**
- [Beads](https://github.com/steveyegge/beads) - `.beads/` directory
- Claude Code Tasks - `~/.claude/todos/` directory

**How it works:**
1. You work on a task, Claude searches for relevant memories
2. Those memories get linked to your task
3. When you mark the task done, linked memories are automatically boosted

```bash
# Unified task commands (all sources)
mem tasks                    # List all tasks
mem tasks --source beads     # Filter by source
mem tasks --source claude    # Claude Code tasks only
mem tasks-sync               # Sync outcomes
mem tasks-context            # Get task context with memories
mem tasks-stats              # View statistics

# Legacy Beads-specific commands (still supported)
mem beads-sync
mem beads-context
mem beads-stats
```

No setup required - Runtime Memory auto-detects both `.beads/` and `~/.claude/todos/` directories.

**Environment variables:**
- `CLAUDE_CODE_TASK_LIST_ID` - Filter to specific task list
- `CLAUDE_CODE_TODOS_DIR` - Custom todos directory location

### Hermes Agent Integration

Runtime Memory can serve as Hermes Agent's memory provider, replacing its capped
note file with retrieval over the same store Claude Code and MCP clients use.

```bash
# Install into the environment Hermes runs in
~/.hermes/hermes-agent/venv/bin/python -m pip install \
    git+https://github.com/runtimenoteslabs/memory-layer.git

hermes config set memory.provider runtimememory
```

Hermes finds the provider through the `hermes_agent.memory_providers` entry
point, so you do not edit its code or config files by hand. See
[docs/hermes.md](https://github.com/runtimenoteslabs/memory-layer/blob/main/docs/hermes.md) for configuration, the tool surface, and the
evaluation trace format.

### Web UI

Runtime Memory includes a web interface for browsing and managing memories.

```bash
# Start server with Web UI
mem serve --rest --port 8080

# Open http://localhost:8080
```

**Features:**
- Dashboard with category statistics
- Memory list with filtering and search
- Semantic and keyword search modes
- Task viewer (Beads + Claude Code)
- Add/edit memories
- Record outcomes
- Light/dark theme

## Memory Categories

| Category | Use For | Example |
|----------|---------|---------|
| `architecture` | System design | "Microservices with event sourcing" |
| `convention` | Coding standards | "Use snake_case for Python" |
| `decision` | Technical choices | "Chose Postgres for ACID compliance" |
| `pattern` | Reusable solutions | "Repository pattern for data access" |
| `gotcha` | Pitfalls to avoid | "Don't use mutable default arguments" |
| `workaround` | Temporary fixes | "Redis reconnect hack for timeout bug" |
| `troubleshooting` | Error solutions | "Clear cache if tests fail randomly" |
| `command` | Useful commands | "npm run test:coverage" |
| `preference` | User preferences | "Prefer functional style" |

## Outcome Scoring

| Outcome | Adds | When to Use |
|---------|------|-------------|
| `worked` | one success | Advice solved the problem |
| `failed` | one failure | Advice was wrong or unhelpful |
| `partial` | a quarter of a success | Advice was on the right track |

A memory's outcome score is `(worked - 1.5 x failed) / (worked + 1.5 x failed + 2)`,
between -1 and 1. One success gives 0.33 and ten give 0.83, so a single
observation counts for less than a long record. A failure weighs 1.5 successes,
because bad advice wastes debugging time and erodes trust. Each count halves
every 90 days.

Retrieval leaves out a memory whose score is -0.5 or lower, which takes two
failures and no successes. One failure is not enough, because it may have been
blamed on the wrong memory. The memory is retrieved again once its failures
have faded.

To change these values, see `RetrievalConfig.outcome_model` and
`RetrievalConfig.failure_gate`.

## How Retrieval Works

Retrieval runs in two stages. Relevance to your query decides which memories
compete, then the other signals order them.

**Stage 1, the relevance pool.** A search keeps the `ceil(limit x 2)` memories
most relevant to the query and drops any with no relevance at all. A memory that
does not match your query is not returned, however good its record.

**Stage 2, the score.**

| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic | 55% | Vector and keyword similarity to your query |
| Outcome | 25% | Learned effectiveness from feedback |
| Confidence | 10% | Extraction confidence score |
| Recency | 10% | Newer memories weighted higher (30-day half-life on age) |
| Frequency | 0% | Off by default; see below |

Outcome and confidence come from how memories have performed rather than from
the query, so ranking changes as feedback accumulates.

**What changed in 4.0.0, and why.** Tier 2 evaluation runs found the older
scoring deciding retrieval on signals that had nothing to do with the query:

- **Frequency left the default score.** It rewards having been retrieved, which
  is not evidence of having helped, and it compounds: a wrong memory held a top
  place through a whole task sequence on it. Set `frequency_weight` to bring it
  back.
- **Category boosts are neutral.** Multiplying the whole score by a category
  seated a memory that ranked about 25th on relevance at rank 1, and in another
  run kept the one memory that would have prevented a repeated mistake out of
  every prompt. Pass `category_boosts` to set your own.
- **Recency decays from a memory's age,** not from when it was last touched.
  Retrieval no longer moves that clock.

`RetrievalConfig.legacy_3x()` restores the 3.x weights, boosts and single-stage
scoring if you tuned for them.

### Category routing

`CategoryRouter` maps query wording to a category, but no search path calls it.
It is available to callers that want to pass `category=` themselves.

## Results

After 12 weeks of use:

| Metric | Improvement |
|--------|-------------|
| Retrieval precision | 70% → 90% |
| Session start context | 54% token savings |
| Post-compaction recovery | 84% token savings |
| Search latency (P95) | <150ms |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | For LLM-based extraction | Required for extraction features |
| `MEMORY_LAYER_DB` | Database location | `~/.runtime-memory/memories.db` |
| `MEMORY_LAYER_ENV` | Environment (development/testing/production) | development |
| `MEMORY_LAYER_LOG_LEVEL` | Logging level | WARNING |
| `CLAUDE_CODE_TASK_LIST_ID` | Filter Claude Code tasks | None |
| `CLAUDE_CODE_TODOS_DIR` | Custom todos directory | `~/.claude/todos/` |

### Data Location

```
~/.runtime-memory/
└── memories.db    # SQLite database
```

## Project Structure

```
memory-layer/
├── src/runtime_memory/
│   ├── core/           # Storage, retrieval, models, config, resilience
│   ├── extraction/     # LLM-based memory extraction
│   ├── server/         # MCP server, REST API, Web UI
│   ├── tasks/          # Task integration (Beads, Claude Code)
│   ├── cli/            # Command-line interface
│   └── sdk/            # Python SDK
└── tests/
    ├── unit/
    ├── integration/
    └── ...
```

## Security

Runtime Memory is designed for local, single-user use:

- **Local storage**: All data stored in `~/.runtime-memory/` (SQLite database)
- **No external transmission**: Memories never leave your machine (except for LLM extraction if enabled)
- **Parameterized queries**: All database operations use parameterized SQL (no injection risk)
- **Input validation**: Pydantic models validate all API inputs
- **Server binding**: REST API binds to `127.0.0.1` by default (localhost only)

**API Keys**: If using LLM extraction features, set `ANTHROPIC_API_KEY` as an environment variable. Never commit API keys to version control.

**Multi-user warning**: The REST API and MCP server are not designed for multi-user/production deployment. For shared use, deploy behind an authentication proxy.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src tests
mypy src
```

## License

MIT

## Acknowledgments

Runtime Memory was inspired by studying 11 existing AI memory systems:

- [claude-mem](https://github.com/thedotmack/claude-mem) - UX patterns, progressive disclosure, web viewer
- [Claude Diary](https://github.com/rlancemartin/claude-diary) - Reflection synthesis, minimal viable memory
- [Mem0](https://github.com/mem0ai/mem0) - Hybrid storage patterns, community building
- [Graphiti/Zep](https://github.com/getzep/graphiti) - Bi-temporal modeling, research-grade benchmarks
- [CORE](https://github.com/RedPlanetHQ/core) - Knowledge graph architecture, temporal modeling
- [Supermemory](https://github.com/supermemoryai/supermemory) - Relationship types, temporal decay
- [Memvid](https://github.com/memvid/memvid) - Single-file portability, embedded WAL
- [Beads](https://github.com/steveyegge/beads) - Task integration, git-native tracking
- [Roampal](https://github.com/roampal-ai/roampal) - Independent validation of outcome-based learning

And thank you to Anthropic for CLAUDE.md - the right foundation for project memory.

The key insight: none of these systems learn from outcomes. Runtime Memory adds a feedback loop so memories that actually help rise to the top.

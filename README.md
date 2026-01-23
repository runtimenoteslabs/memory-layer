# Memory Layer

Persistent memory for AI coding agents with outcome-based learning.

> **New to Memory Layer?** See the [User Guide](USER_GUIDE.md) for a simple introduction to using Memory Layer with Claude Code.

## What It Does

Memory Layer stores knowledge from your coding sessions and learns which memories actually help. When advice works, it gets boosted (+0.2). When it fails, it gets penalized (-0.3). Over time, good memories rise to the top.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/runtimenoteslabs/memory-layer.git
```

For development:

```bash
git clone https://github.com/runtimenoteslabs/memory-layer.git
cd memory-layer
pip install -e ".[dev]"
```

**Note:** First run downloads an embedding model (~100MB) for semantic search. This happens once and is cached. Subsequent operations are fast (<100ms).

## Quick Start

### Python SDK

```python
from memory_layer.sdk import MemoryClient

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
from memory_layer.sdk import SyncMemoryClient

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

For multi-agent setups, Memory Layer provides an MCP server:

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

Memory Layer integrates with Claude Code via hooks and skills. For a beginner-friendly walkthrough, see the [User Guide](USER_GUIDE.md).

**Installation:**

```bash
# Install from GitHub
pip install git+https://github.com/runtimenoteslabs/memory-layer.git

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

Memory Layer integrates with task trackers to automatically learn from task outcomes.

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

No setup required - Memory Layer auto-detects both `.beads/` and `~/.claude/todos/` directories.

**Environment variables:**
- `CLAUDE_CODE_TASK_LIST_ID` - Filter to specific task list
- `CLAUDE_CODE_TODOS_DIR` - Custom todos directory location

### Web UI

Memory Layer includes a web interface for browsing and managing memories.

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

| Outcome | Score Change | When to Use |
|---------|--------------|-------------|
| `worked` | +0.2 | Advice solved the problem |
| `failed` | -0.3 | Advice was wrong or unhelpful |
| `partial` | +0.05 | Advice was on the right track |

The asymmetric scoring is intentional: bad advice wastes debugging time and erodes trust, so it's penalized more heavily.

## How Retrieval Works

Memory Layer uses a 5-signal hybrid retrieval system that combines multiple relevance signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic | 35% | Vector similarity to your query |
| Outcome | 25% | Learned effectiveness from feedback |
| Recency | 15% | Recent memories weighted higher (30-day half-life) |
| Frequency | 15% | Frequently used memories rise |
| Confidence | 10% | Extraction confidence score |

This hybrid approach outperforms pure vector search by incorporating learned effectiveness and usage patterns.

### Category Boosting

When you ask about errors, troubleshooting memories get a 1.5x boost. Query intent is detected and the right category is prioritized:

| Query Pattern | Boosted Category | Multiplier |
|---------------|------------------|------------|
| "What went wrong..." | troubleshooting | 1.5x |
| "Watch out for..." | gotcha | 1.4x |
| "Why did we choose..." | decision | 1.4x |
| "How should I structure..." | pattern, convention | 1.3x |
| "System design..." | architecture | 1.2x |

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
| `MEMORY_LAYER_DB` | Database location | `~/.memory-layer/memories.db` |
| `MEMORY_LAYER_ENV` | Environment (development/testing/production) | development |
| `MEMORY_LAYER_LOG_LEVEL` | Logging level | WARNING |
| `CLAUDE_CODE_TASK_LIST_ID` | Filter Claude Code tasks | None |
| `CLAUDE_CODE_TODOS_DIR` | Custom todos directory | `~/.claude/todos/` |

### Data Location

```
~/.memory-layer/
└── memories.db    # SQLite database
```

## Project Structure

```
memory-layer/
├── src/memory_layer/
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

Memory Layer is designed for local, single-user use:

- **Local storage**: All data stored in `~/.memory-layer/` (SQLite database)
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

Memory Layer was inspired by studying 11 existing AI memory systems:

- **claude-mem** - UX patterns, progressive disclosure, one-command install
- **Claude Diary** - Reflection synthesis, minimal viable memory
- [Mem0](https://github.com/mem0ai/mem0) - Hybrid storage patterns, community building
- **OpenMemory** - Local-first approach
- [Graphiti/Zep](https://github.com/getzep/graphiti) - Temporal modeling research
- [CORE](https://github.com/redplanethq/core) - Knowledge graph architecture
- **Supermemory** - Relationship types, temporal decay
- [Memvid](https://github.com/memvid/memvid) - Single-file portability
- [Beads](https://github.com/steveyegge/beads) - Task integration concepts
- **Roampal** - Validated outcome-based learning approach

The key insight: none of these systems learn from outcomes. Memory Layer adds a feedback loop so memories that actually help rise to the top.

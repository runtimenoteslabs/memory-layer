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

### Beads Task Tracker Integration

Memory Layer integrates with [Beads](https://github.com/steveyegge/beads) to automatically learn from task outcomes.

**How it works:**
1. You work on a Beads task, Claude searches for relevant memories
2. Those memories get linked to your task
3. When you mark the task done, linked memories are automatically boosted

```bash
# Sync outcomes for completed tasks
mem beads-sync

# See context for current task
mem beads-context

# View integration stats
mem beads-stats
```

No setup required - Memory Layer auto-detects `.beads/` directories.

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

### Data Location

```
~/.memory-layer/
└── memories.db    # SQLite database
```

## Project Structure

```
memory-layer/
├── src/memory_layer/
│   ├── core/           # Storage, retrieval, models
│   ├── extraction/     # LLM-based memory extraction
│   ├── server/         # MCP and REST API servers
│   ├── cli/            # Command-line interface
│   └── sdk/            # Python SDK
└── tests/
    ├── unit/
    ├── integration/
    └── ...
```

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

# Migration Guide: v1 (Daemon) to v2 (Plugin)

This guide helps you migrate from Memory Layer v1 (daemon-based architecture) to v2 (Claude Code 2.1.1 native plugin system).

## Overview

Memory Layer v2 introduces a complete architectural shift from a background daemon process to a native Claude Code plugin system. This provides:

- **Native Integration**: Works directly with Claude Code's hook and skill systems
- **Simpler Setup**: No daemon management required
- **Better Performance**: Direct hook execution without IPC overhead
- **Automatic Context**: Session-aware context injection via native hooks

## What's Changed

### Architecture

| Component | v1 (Daemon) | v2 (Plugin) |
|-----------|-------------|-------------|
| Lifecycle Management | `mem daemon start/stop` | Native Claude Code lifecycle |
| Memory Retrieval | Manual `/recall` command | Automatic via Agent Skills |
| Context Injection | Daemon file watcher | Native `SessionStart` hook |
| Session Tracking | Custom session IDs | Native `$CLAUDE_SESSION_ID` |
| Commands | JSON schema files | `commands/*.md` |
| MCP Server | External process | Inline `.mcp.json` |

### Deprecated Modules

The following modules are deprecated in v2 and will show deprecation warnings:

- `memory_layer.claude_code.daemon` - Use native plugin hooks instead
- `memory_layer.claude_code.hooks` - Replaced by `hooks/hooks.json`

**Note**: These modules still work for backwards compatibility but will be removed in v3.

## Migration Steps

### Step 1: Stop the Daemon (if running)

If you were running the v1 daemon, stop it first:

```bash
# v1 command (deprecated)
mem daemon stop
```

### Step 2: Update Installation

Update to v2:

```bash
pip install --upgrade memory-layer
```

### Step 3: Install as Claude Code Plugin

The v2 plugin installs automatically when Claude Code detects the plugin structure:

```bash
# Navigate to your project
cd /path/to/your/project

# The plugin is detected via .claude-plugin/plugin.json
# No manual installation required
```

### Step 4: Verify Hook Configuration

Check that the native hooks are configured in `hooks/hooks.json`:

```json
{
  "hooks": {
    "SessionStart": [...],
    "PreCompact": [...],
    "SessionEnd": [...],
    "PostToolUse": [...]
  }
}
```

### Step 5: Update Custom Scripts

If you had custom scripts using the daemon, update them:

**Before (v1):**
```bash
# Start daemon
mem daemon start

# Manual context injection
mem context --inject
```

**After (v2):**
```bash
# No daemon needed - hooks run automatically
# Context injection happens via SessionStart hook
```

## Command Changes

### Slash Commands

Commands have moved from JSON schemas to markdown files in `commands/`:

| v1 Command | v2 Command | Notes |
|------------|------------|-------|
| `/remember` | `/remember` | Now uses `commands/remember.md` |
| `/recall` | `/recall` | Now uses `commands/recall.md` |
| `/outcome` | `/outcome` | Now uses `commands/outcome.md` |
| `/context` | `/memory-context` | Renamed for clarity |

### CLI Commands

Most CLI commands remain the same, with some additions:

**New v2 CLI Commands:**
```bash
# Context injection for hooks
mem context --inject --format silent

# Extract with stdin support
cat transcript.txt | mem extract --stdin

# Session management
mem session start --session $CLAUDE_SESSION_ID
mem session end --session $CLAUDE_SESSION_ID --summarize

# File tracking for PostToolUse hook
mem track-file /path/to/file.py --session $CLAUDE_SESSION_ID
```

## Agent Skills (New in v2)

v2 introduces Agent Skills that automatically trigger memory retrieval:

### Memory Retrieval Skill
Activates on phrases like:
- "what did we decide about..."
- "how do we handle..."
- "what's our convention..."

### Outcome Feedback Skill
Detects outcome signals:
- Positive: "thanks", "worked", "perfect"
- Negative: "still broken", "didn't help"
- Partial: "kind of worked", "partially"

### Coding Patterns Skill
Activates on code creation:
- "create a new..."
- "implement..."
- "how should I structure..."

## Data Migration

### Existing Memories

Your existing memories are preserved. The database schema is backwards compatible:

```bash
# Verify your memories are intact
mem list --limit 20
mem stats
```

### New Categories (v2)

v2 adds new memory categories:

- `dependency` - What relies on what
- `environment` - Setup, config, secrets locations
- `coding_style` - Tabs vs spaces, naming conventions
- `tool_preference` - Preferred libraries, frameworks
- `context` - Current work state
- `todo` - Pending items
- `general` - Uncategorized (default)

## Configuration

### Environment Variables

| Variable | v1 | v2 | Notes |
|----------|----|----|-------|
| `MEMORY_LAYER_DB` | Yes | Yes | Database path |
| `MEMORY_LAYER_LOG_LEVEL` | Yes | Yes | Logging level |
| `CLAUDE_SESSION_ID` | No | Yes | Native session ID |

### MCP Configuration

v2 uses inline MCP configuration (`.mcp.json`):

```json
{
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
```

## Troubleshooting

### Deprecation Warnings

If you see deprecation warnings like:

```
DeprecationWarning: The daemon module is deprecated as of v2.0.0.
Use the native plugin system with hooks/hooks.json instead.
```

This means you're importing v1 modules. Update your imports:

**Before:**
```python
from memory_layer.claude_code import start_daemon, stop_daemon
```

**After:**
```python
from memory_layer.plugin import SessionManager, HookContext
```

### Hooks Not Running

If hooks aren't executing:

1. Verify `hooks/hooks.json` exists and is valid JSON
2. Check that Claude Code version is 2.1.1 or higher
3. Ensure the `mem` CLI is in your PATH

### Session ID Issues

If session tracking isn't working:

```bash
# Check if session ID is available
echo $CLAUDE_SESSION_ID

# Manually set for testing
export CLAUDE_SESSION_ID="test-session-123"
```

## API Changes

### Python SDK

The Python SDK remains largely unchanged:

```python
from memory_layer import MemoryEngine

# This still works in v2
engine = MemoryEngine()
await engine.add("Use snake_case", category="convention")
results = await engine.search("naming conventions")
```

### New Plugin Module

v2 adds a new plugin module:

```python
from memory_layer.plugin import (
    HookContext,
    ContextFormatter,
    SkillTriggers,
    SessionManager,
)

# Get hook context from environment
ctx = HookContext.from_environment()
print(f"Session: {ctx.session_id}")
print(f"Project: {ctx.project_path}")

# Format memories for injection
memories = await engine.search("conventions")
formatted = ContextFormatter.format_for_injection(
    [r.memory for r in memories],
    style="markdown"
)

# Detect skill triggers
if SkillTriggers.should_retrieve("what did we decide about auth?"):
    # Trigger memory retrieval
    pass
```

## Getting Help

- GitHub Issues: https://github.com/memory-layer/memory-layer/issues
- Documentation: See `README.md` and `docs/` directory

## Version Compatibility

| Memory Layer | Claude Code | Python |
|--------------|-------------|--------|
| v1.x | Any | >=3.10 |
| v2.x | >=2.1.1 | >=3.10 |

## Rollback (if needed)

If you need to rollback to v1:

```bash
pip install memory-layer==1.0.0

# Restart the daemon
mem daemon start
```

Note: v1 will continue to work but won't receive new features.

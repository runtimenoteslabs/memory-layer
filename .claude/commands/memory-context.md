---
description: Show active memories loaded for the current context
allowed-tools: Bash
---

# Memory Context Command

Display the memories currently loaded in context for this project. This shows what the Memory Layer knows about your current working directory.

## Usage

```
/memory-context [--detailed] [--category <cat>] [--limit N]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--detailed` | Show scores and metadata | false |
| `--category <cat>` | Filter by category | all |
| `--limit N` | Maximum memories to show | 20 |
| `--format <fmt>` | Output format (brief/detailed/structured) | brief |

## Examples

```bash
# Show current context
/memory-context

# Show detailed view with scores
/memory-context --detailed

# Show only conventions
/memory-context --category convention

# Show all with structured format
/memory-context --format structured --limit 50
```

## Implementation

Display project context:

```bash
mem context --project "$PWD" --format brief
```

## Output Formats

### Brief (default)
```
# Memory Context

- [architecture] Microservices with API gateway pattern
- [convention] Use snake_case for Python, camelCase for JS
- [gotcha] Auth service rate limits at 100 req/min
```

### Detailed
```
# Memory Context (detailed)

## [abc123] ARCHITECTURE
Microservices with API gateway pattern
Score: 0.45 | Used: 12x | Confidence: 0.9

## [def456] CONVENTION
Use snake_case for Python, camelCase for JS
Score: 0.60 | Used: 8x | Confidence: 1.0
```

### Structured
```
# Memory Context

## Architecture (2)
- [abc123] Microservices with API gateway pattern [proven]
- [def456] Event-driven communication between services

## Convention (3)
- [ghi789] Use snake_case for Python... [proven]
- [jkl012] Imports ordered: stdlib, third-party, local
```

## When Context is Loaded

Memory context is automatically loaded:
1. **SessionStart**: Top 10 relevant memories injected
2. **On demand**: When Agent Skills detect relevant queries
3. **Explicitly**: When you run `/memory-context` or `/recall`

## Managing Context

- Memories with high outcome scores are prioritized
- Project-specific memories take precedence over global
- Recent memories get a slight boost
- Use `/outcome` to improve future context relevance

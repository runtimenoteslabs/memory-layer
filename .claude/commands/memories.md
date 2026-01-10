---
description: List all stored memories
allowed-tools: Bash
---

# Memories Command

List all memories stored in the Memory Layer.

## Usage

```
/memories [--limit N] [--category <cat>]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--limit N` | Maximum number of results | 20 |
| `--category <cat>` | Filter by category | all |

## Examples

```bash
# List all memories
/memories

# List with limit
/memories --limit 10

# List only gotchas
/memories --category gotcha
```

## Implementation

List memories via CLI:

```bash
mem list --limit 20
```

## Output

Shows each memory with:
- Memory ID (for use with `/outcome` or `/forget`)
- Category
- Content preview
- Outcome score (if feedback received)
- Created date

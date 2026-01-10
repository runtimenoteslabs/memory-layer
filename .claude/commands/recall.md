---
description: Explicitly search memories (usually handled automatically by Agent Skills)
allowed-tools: Bash
---

# Recall Command

Explicitly search the Memory Layer for relevant memories. Note that Agent Skills usually handle retrieval automatically based on conversation context, so this command is rarely needed.

## Usage

```
/recall <query> [--limit N] [--category <cat>]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--limit N` | Maximum number of results | 5 |
| `--category <cat>` | Filter by category | all |
| `--min-score <float>` | Minimum outcome score | -1.0 |

## Examples

```bash
# Search for authentication patterns
/recall "authentication patterns"

# Search with limit
/recall "database connection" --limit 10

# Search within a category
/recall "naming" --category convention

# Search for high-confidence memories only
/recall "error handling" --min-score 0.3
```

## Implementation

Search memories and display results:

```bash
mem search "$QUERY" --format context --limit 5
```

## When to Use

The `memory-retrieval` Agent Skill automatically retrieves memories when you:
- Ask about past decisions ("what did we decide about...")
- Reference conventions ("what's our convention for...")
- Mention previous work ("last time we...", "we discussed...")

Use `/recall` explicitly when:
- You want to browse all memories on a topic
- You need more results than auto-retrieval provides
- You want to filter by specific category or score
- Auto-retrieval didn't surface what you were looking for

## Result Format

Results are ranked by a hybrid score combining:
- Semantic similarity (35%)
- Outcome score (25%) - proven advice ranks higher
- Recency (15%)
- Frequency of use (15%)
- Confidence (10%)

After reviewing results, use `/outcome <id> worked|failed|partial` to provide feedback.

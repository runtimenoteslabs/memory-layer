---
description: Store a memory with optional category for future retrieval
allowed-tools: Bash
---

# Remember Command

Store a piece of knowledge in the Memory Layer for future retrieval. Memories are persisted across sessions and become smarter over time through outcome-based learning.

## Usage

```
/remember <content> [-c <category>]
```

## Categories

| Category | Use For |
|----------|---------|
| `architecture` | System structure, component relationships, design patterns |
| `convention` | Coding standards, naming patterns, style rules |
| `decision` | Technical choices and their rationale |
| `pattern` | Reusable solutions, idioms, best practices |
| `gotcha` | Pitfalls, edge cases, things that don't work as expected |
| `workaround` | Temporary fixes, hacks, technical debt |
| `troubleshooting` | Error solutions, debugging steps, fixes |
| `command` | Useful CLI commands, scripts, one-liners |
| `preference` | User preferences for tools, formatting, workflow |

## Examples

```bash
# Store a coding convention
/remember "Always use snake_case for Python variables" -c convention

# Store a gotcha about the project
/remember "The auth service rate limits at 100 req/min" -c gotcha

# Store a technical decision with rationale
/remember "We chose PostgreSQL over MySQL for JSON support and better concurrency" -c decision

# Store a pattern
/remember "Use the repository pattern for all database access" -c pattern

# Store a troubleshooting tip
/remember "Error 'connection refused' usually means Redis isn't running" -c troubleshooting

# Store without category (auto-categorized)
/remember "npm run build:prod for production builds"
```

## Implementation

Parse the arguments and store via CLI:

```bash
mem add "$ARGUMENTS"
```

After storing, confirm to the user what was saved and its assigned ID. The memory starts with an outcome score of 0.0 and will be adjusted based on feedback.

## Tips

- Be specific and actionable in your memories
- Include context about when/why the advice applies
- Use consistent terminology to improve retrieval
- Record outcomes with `/outcome` to improve future suggestions

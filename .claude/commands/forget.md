---
description: Archive/delete a memory by ID
allowed-tools: Bash
---

# Forget Command

Archive a memory so it no longer appears in searches or context.

## Usage

```
/forget <memory-id>
```

## Examples

```bash
# Forget a specific memory
/forget 47abecc7

# Forget using partial ID
/forget 47ab
```

## Implementation

Delete memory via CLI:

```bash
mem delete "$MEMORY_ID" --confirm
```

## Notes

- Memories are soft-deleted (archived), not permanently removed
- Partial IDs work if they uniquely identify a memory
- If the partial ID matches multiple memories, you'll be asked to be more specific
- Use `/memories` to find memory IDs

## When to Use

Use `/forget` when:
- A memory is outdated or no longer applies
- A memory contains incorrect information
- You want to clean up test/temporary memories

Consider using `/outcome <id> failed` instead if the memory was just unhelpful in a specific context - this helps the system learn rather than removing the memory entirely.

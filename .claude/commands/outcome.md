---
description: Record feedback on whether a memory helped (worked/failed/partial)
allowed-tools: Bash
---

# Outcome Command

Record feedback on whether a memory's advice was helpful. This is the core of outcome-based learning - memories that help get boosted, memories that fail get penalized.

## Usage

```
/outcome <memory_id> worked|failed|partial
```

## Outcome Types

| Outcome | Score Change | Use When |
|---------|--------------|----------|
| `worked` | +0.2 | The advice solved the problem |
| `failed` | -0.3 | The advice was wrong or unhelpful |
| `partial` | +0.05 | The advice was on the right track but incomplete |

The asymmetric scoring (-0.3 for failed vs +0.2 for worked) is intentional: wrong advice wastes more time than good advice saves, so it should sink faster.

## Examples

```bash
# Memory advice solved the problem
/outcome mem_abc123 worked

# Memory advice was wrong
/outcome mem_def456 failed

# Memory advice partially helped
/outcome mem_ghi789 partial
```

## Implementation

Record the outcome:

```bash
mem outcome "$MEMORY_ID" "$OUTCOME"
```

## Score Ranges

Memories have an outcome score from -1.0 to 1.0:

| Score Range | Meaning | Retrieval Impact |
|-------------|---------|------------------|
| > 0.5 | Highly reliable | Prioritized in results |
| 0.3 to 0.5 | Generally helpful | Normal ranking |
| 0.0 to 0.3 | Mixed results | Normal ranking |
| -0.3 to 0.0 | Questionable | Deprioritized |
| < -0.3 | Unreliable | May be auto-archived |

## Impact Over Time

With consistent feedback:
- Week 1: ~70% retrieval precision (baseline)
- Week 4: ~78% precision (some outcome data)
- Week 8: ~85% precision (outcome scoring active)
- Week 12: ~90% precision (bad memories suppressed)

## Tips

- Provide feedback soon after using a memory
- Be honest about partial successes
- The `outcome-feedback` Skill will prompt you naturally after solutions
- Feedback on frequently-used memories has the most impact

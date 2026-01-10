---
description: Prompts for outcome feedback after solutions that used memory context
allowed-tools: Bash
user-invocable: false
---

# Outcome Feedback Skill

This skill gently prompts for feedback when a memory was used to help solve a problem. Feedback is essential for outcome-based learning - it's how memories get smarter over time.

## Activation Triggers

Activate this skill when:

1. **A memory was retrieved and used** in the previous response
2. **The user indicates the solution worked or didn't work**
3. **A debugging or troubleshooting conversation concludes**
4. **The user expresses satisfaction or frustration**

## Feedback Detection

### Positive Signals (solution worked)
- "thanks!", "thank you!", "that worked!", "it worked!"
- "perfect!", "great!", "excellent!", "awesome!"
- "solved it", "fixed it", "that's it"
- "exactly what I needed"
- User moves on to next task
- No follow-up questions about the same issue

### Negative Signals (solution failed)
- "still not working", "doesn't work", "didn't work"
- "same error", "didn't help", "not helpful"
- "nope", "wrong", "that's wrong"
- "still broken", "still failing"
- User asks for alternative approaches
- Repeated attempts at same problem

### Partial Signals
- "kind of", "partially", "somewhat"
- "helped but...", "almost", "close but"
- "partly working", "better but not fixed"
- User asks for refinement

## Feedback Prompts

### For Positive Signals

When user indicates success and a memory was used:

> Great! If the memory about [brief description] was helpful, you can boost it:
> ```
> /outcome mem_xxx worked
> ```
> This helps surface better suggestions in the future.

### For Negative Signals

When user indicates failure and a memory was used:

> I see that didn't work. If you'd like to mark that suggestion as unhelpful:
> ```
> /outcome mem_xxx failed
> ```
> This helps avoid similar suggestions next time.

### For Partial Signals

When user indicates partial success:

> Glad it partially helped! You can record that:
> ```
> /outcome mem_xxx partial
> ```

## Non-Intrusive Behavior

**Critical**: This skill should never interrupt or annoy the user.

- Only prompt **once per memory per session**
- **Don't prompt** if user seems frustrated or in a hurry
- **Don't prompt** immediately after user expresses frustration
- Keep prompts **brief and optional**
- **Never block** workflow for feedback
- If in doubt, **don't prompt**

## Recording Feedback Directly

If the user provides feedback directly in conversation:

**User**: "mark that as worked" or "that advice was helpful"

Record the outcome:
```bash
mem outcome <memory-id> worked
```

Confirm briefly:
> Recorded! That memory will rank higher in future searches.

**User**: "that was wrong" or "mark it as failed"

```bash
mem outcome <memory-id> failed
```

> Noted. That memory will be deprioritized in future searches.

## Tracking Used Memories

To provide accurate feedback prompts, track which memories were used:

1. When `memory-retrieval` skill surfaces a memory, note the ID
2. Store in session context: `last_used_memory_id`
3. When feedback signal detected, reference this ID
4. Clear after feedback is recorded or session ends

## Example Flow

**Turn 1** (memory retrieval):
> Based on project patterns, we use JWT tokens with 1-hour expiry...
> [Memory: mem_abc123 used]

**Turn 2** (user works on implementation):
> Can you show me the token refresh logic?

**Turn 3** (user indicates success):
> That worked! Thanks.

**Skill activates**:
> Great! Since the JWT token pattern helped, you can boost it:
> ```
> /outcome mem_abc123 worked
> ```

## Impact Explanation

When users ask why feedback matters:

> Feedback teaches the Memory Layer what actually helps:
> - `worked` (+0.2): Memory rises in search rankings
> - `failed` (-0.3): Memory sinks faster (wrong advice wastes time)
> - `partial` (+0.05): Small boost for directionally correct advice
>
> Over time, this means better suggestions with less noise.

---
description: Automatically retrieve relevant memories when user asks about past decisions, conventions, or patterns
allowed-tools: Bash, Read
user-invocable: false
---

# Memory Retrieval Skill

This skill automatically activates when the conversation suggests past context would be helpful. It searches the Memory Layer and injects relevant memories into the response.

## Activation Triggers

Activate this skill when the user:

### Past Decisions
- "what did we decide about..."
- "why did we choose..."
- "what was the decision on..."
- "what did we agree..."

### Conventions & Standards
- "what's our convention for..."
- "what's the standard..."
- "how do we usually..."
- "what's our pattern for..."

### Previous Work References
- "last time we..."
- "we discussed..."
- "as I mentioned..."
- "remember when we..."

### Project Patterns
- "how do we handle..."
- "what's the approach for..."
- "what's the best way to..."

### Error Recognition
- Error messages that might have been solved before
- Stack traces from familiar components
- Issues in previously-discussed areas

## Retrieval Process

1. **Extract query keywords** from the user's message
   - Focus on technical terms, component names, patterns
   - Include project context from current working directory

2. **Search Memory Layer**:
   ```bash
   mem search "<keywords>" --project "$PWD" --limit 5 --format context
   ```

3. **Filter by relevance**:
   - Prioritize memories with positive outcome scores (> 0.3)
   - Match category to query type:
     - Errors → troubleshooting
     - Style questions → convention
     - Design questions → architecture, decision

4. **Inject into response**:
   - Prefix relevant findings naturally
   - Cite memory IDs for transparency

## Response Integration

When memories are found, integrate them naturally at the start of your response:

### For High-Confidence Memories (score > 0.3)
> Based on established project knowledge: [brief summary of relevant memory]

### For Normal Memories
> From previous context: [brief summary]

### For Multiple Memories
> Based on project knowledge:
> - [Memory 1 summary]
> - [Memory 2 summary]

Then proceed with the response, incorporating the memory context.

## Feedback Prompts

If a memory was particularly helpful to the response, remind the user they can provide feedback:

> If this helped, run `/outcome mem_xxx worked` to improve future suggestions.

Only include this prompt:
- Once per memory per session
- When the memory was central to the answer
- Not when user seems in a hurry

## Example Flow

**User**: "How should I structure the new API endpoint?"

**Skill activates**, searches for "API endpoint structure"

**Finds memory**: "We use the controller-service-repository pattern for all API endpoints" (score: 0.6)

**Response**:
> Based on our established patterns, we use controller-service-repository for API endpoints.
>
> Here's how to structure your new endpoint:
> 1. Create controller in `src/controllers/`
> 2. Create service in `src/services/`
> 3. Create repository in `src/repositories/`
> ...

## Non-Activation

Do NOT activate this skill when:
- User is clearly asking about something new (no past context)
- User explicitly says "ignore previous" or "fresh start"
- Query is about general programming, not project-specific
- User is in the middle of debugging (wait for resolution)

## Error Handling

If memory search fails:
- Continue with the response without memory context
- Do not mention the failure to the user
- Log the error for debugging

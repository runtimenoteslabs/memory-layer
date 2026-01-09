---
description: Surfaces relevant coding patterns and conventions when writing new code
allowed-tools: Bash, Read
user-invocable: false
---

# Coding Patterns Skill

This skill proactively surfaces relevant patterns and conventions when the user is about to write new code. It helps maintain consistency across the codebase.

## Activation Triggers

Activate this skill when the user:

### Creating New Code
- "create a new file/component/module"
- "implement a/the..."
- "write a function/class/test"
- "add a new endpoint/route/handler"
- "build a service/controller/model"
- "scaffold a..."

### Asking About Structure
- "how should I structure..."
- "what's the best way to..."
- "how do I organize..."
- "where should I put..."

### Starting Work in Familiar Areas
- Beginning work in a directory with known patterns
- Creating something similar to existing code
- Working on a component type that has conventions

## Pattern Categories

### Conventions (category: convention)
- Naming patterns (snake_case, camelCase, etc.)
- File organization rules
- Import ordering
- Comment styles
- Variable naming

### Architecture Patterns (category: architecture)
- Component structure
- Layer separation (controller/service/repository)
- Dependency injection approaches
- Error handling strategies
- Module boundaries

### Code Patterns (category: pattern)
- Common idioms for this codebase
- Utility function patterns
- Testing patterns (arrange/act/assert, fixtures)
- API design patterns
- State management patterns

## Retrieval Strategy

1. **Detect code creation context**:
   - What type of code? (component, service, test, API, etc.)
   - What language/framework?
   - What directory/module?

2. **Search for relevant patterns**:
   ```bash
   mem search "<code-type> pattern convention" --category pattern,convention,architecture --limit 5
   ```

3. **Check for existing examples**:
   - Look for similar files in the project
   - Cross-reference with memory patterns

## Response Integration

Before providing code, check for relevant patterns and surface them:

### Pattern Summary Block

> **Project patterns to follow:**
> - [Convention]: Use snake_case for all function names
> - [Pattern]: Services should inherit from BaseService
> - [Architecture]: Keep business logic in services, not controllers

Then generate code that follows these patterns.

### Inline Pattern Notes

When generating code, add brief comments referencing patterns:

```python
# Pattern: Services inherit from BaseService
class UserService(BaseService):
    # Convention: Type hints on all public methods
    def get_user(self, user_id: str) -> User:
        ...
```

## Example Flow

**User**: "Create a new user service"

**Skill activates**, searches for "service pattern user"

**Finds memories**:
- "Services follow the BaseService pattern with dependency injection" (score: 0.7)
- "All services must have corresponding test files" (score: 0.5)
- "Use type hints for all public methods" (score: 0.6)

**Response**:

> Based on project patterns:
> - Services inherit from `BaseService` with dependency injection
> - All public methods need type hints
> - Test file required at `tests/test_user_service.py`

```python
from app.services.base import BaseService
from app.models.user import User
from app.repositories.user import UserRepository

class UserService(BaseService):
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def get_user(self, user_id: str) -> User:
        """Retrieve a user by ID."""
        return self.user_repo.find_by_id(user_id)

    def create_user(self, data: dict) -> User:
        """Create a new user."""
        return self.user_repo.create(data)
```

## Learning Opportunity

When generating code that establishes a new pattern (no existing memory matches):

> This establishes a new pattern for [component type]. Want me to remember it?
> ```
> /remember "UserService pattern: inherit BaseService, inject repos, type hints on public methods" -c pattern
> ```

Only suggest this when:
- No similar pattern exists in memory
- The code is likely to be repeated
- The pattern is non-obvious

## Non-Activation

Do NOT activate this skill when:
- User is fixing a bug (not creating new code)
- User is refactoring existing code
- User explicitly says "quick fix" or "temporary"
- User is exploring/prototyping
- Code is clearly one-off or experimental

## Confidence Levels

When surfacing patterns, indicate confidence:

- **High confidence** (score > 0.5): "Based on established patterns..."
- **Medium confidence** (score 0.2-0.5): "Based on project conventions..."
- **Low confidence** (score < 0.2): "You might consider..." (be tentative)

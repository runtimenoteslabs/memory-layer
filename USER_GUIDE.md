# Memory Layer - User Guide

A simple guide for using Memory Layer with Claude Code and other AI coding assistants.

---

## What It Does

Memory Layer gives Claude Code (and other AI coding assistants) a persistent memory. Instead of forgetting everything when you close a session, Claude remembers:

- Your project conventions ("we use tabs, not spaces")
- Past decisions ("we chose PostgreSQL because...")
- Gotchas ("the CI server needs Node 18, not 20")
- What advice actually helped you

Over time, it learns which memories are actually useful based on your feedback.

---

## Setup (One Time)

```bash
# 1. Install from GitHub
pip install git+https://github.com/runtimenoteslabs/memory-layer.git

# 2. Go to your project
cd your-project

# 3. Enable for Claude Code
mem install-plugin

# 4. Start Claude Code as usual
claude
```

That's it. Memory Layer now works automatically.

---

## Daily Usage

### The Basics

**You don't need to do anything special.** Claude will:
- Automatically load relevant memories at session start
- Remember things from your conversations
- Learn which advice actually helped

### When You Want to Explicitly Remember Something

Just tell Claude naturally:

> "Remember that we always run tests before committing"

> "Remember: the API rate limits to 100 requests per minute"

Or use the slash command:
```
/remember We use React 18 with TypeScript
```

With a category:
```
/remember category:gotcha The staging server resets every night at 2am
```

### When You Want to Find Past Knowledge

Ask Claude naturally:

> "What's our convention for error handling?"

> "What did we decide about the database?"

Or use the slash command:
```
/recall database setup
```

### Giving Feedback

When Claude's advice helps (or doesn't), tell it:

> "Thanks, that worked!"

> "That didn't work, the tests still fail"

> "That partially helped, but I also needed to restart the server"

This helps Claude learn which memories are actually useful. Good advice gets boosted, bad advice gets penalized.

---

## Available Slash Commands

Use these directly in Claude Code:

| Command | What it does |
|---------|--------------|
| `/remember <content>` | Store a new memory |
| `/remember category:gotcha <content>` | Store with specific category |
| `/recall <query>` | Search for memories |
| `/memories` | List all stored memories |
| `/outcome <id> worked` | Mark advice as helpful |
| `/outcome <id> failed` | Mark advice as unhelpful |
| `/forget <id>` | Archive a memory |
| `/memory-context` | Get summary of project knowledge |

---

## Terminal Commands (Optional)

Most users never need these, but they're available if you prefer the command line:

```bash
# See all memories
mem list

# Search memories
mem search "keyword"

# Add a memory
mem add "Always use async/await" -c convention

# Get project summary
mem context

# View statistics
mem stats

# Give feedback
mem outcome <id> worked
```

---

## Memory Categories

When storing memories, you can specify a category to help organize them:

| Category | Use for | Example |
|----------|---------|---------|
| `convention` | Team coding standards | "Use snake_case for Python variables" |
| `architecture` | System design decisions | "Microservices communicate via RabbitMQ" |
| `decision` | Why we chose X over Y | "Using PostgreSQL for ACID compliance" |
| `pattern` | Reusable code patterns | "Repository pattern for data access" |
| `gotcha` | Things that trip people up | "CI requires Node 18, not Node 20" |
| `workaround` | Temporary fixes | "Restart Redis if connections timeout" |
| `troubleshooting` | How to debug issues | "Clear cache if tests fail randomly" |
| `command` | Useful commands | "npm run test:coverage for coverage report" |
| `preference` | Personal/team preferences | "Prefer functional style over classes" |
| `general` | Everything else | Default if not specified |

---

## Tips for Best Results

### 1. Be Specific

**Good:** "Use async/await for all database calls in this project"

**Less useful:** "use async"

### 2. Include the Why

**Good:** "We use PostgreSQL because we need ACID transactions for payment processing"

**Less useful:** "We use PostgreSQL"

### 3. Give Feedback

The more you say "that worked!" or "that didn't help", the smarter the system gets. It takes just a second and makes a real difference.

### 4. Don't Worry About Perfect Organization

You don't need to categorize everything perfectly. The search is smart enough to find relevant memories even if they're in different categories.

### 5. It Works Across Sessions

Close Claude, come back tomorrow, next week, or next month - your memories are still there. That's the whole point!

---

## How It Works (For the Curious)

1. **Storage**: Memories are stored locally in a SQLite database (`~/.memory-layer/memories.db`)

2. **Retrieval**: When you ask questions, relevant memories are automatically searched using a hybrid approach:
   - Semantic similarity (what you're asking about)
   - Outcome scores (what actually helped before)
   - Recency (recent memories weighted higher)
   - Usage frequency (popular memories rise)

3. **Smart Boosting**: When you ask about errors, troubleshooting memories are automatically prioritized. Ask "what's our convention...", and convention memories get boosted. The system detects your intent.

4. **Learning**: Each memory has a score starting at 0.0
   - "worked" → +0.2 (max 1.0)
   - "failed" → -0.3 (min -1.0)
   - "partial" → +0.05

   Higher-scored memories appear first in search results. Over time, good advice rises and bad advice sinks.

5. **Privacy**: Everything stays on your machine. No data is sent anywhere.

---

## First-Time Performance

The first time you use Memory Layer, a few things happen that may make it seem slow:

1. **Embedding model download** (~100MB): On first search, the system downloads a sentence embedding model for semantic search. This happens once and is cached.

2. **Database creation**: The SQLite database is created on first use at `~/.memory-layer/memories.db`.

3. **Index building**: As you add memories, they get indexed for fast retrieval.

**What to expect:**
- First run: 5-30 seconds (model download)
- Subsequent operations: <100ms

If the first run seems stuck, it's likely downloading the embedding model. You can verify with:
```bash
mem check
```

---

## Task Integration

Memory Layer integrates with task trackers to automatically learn from task outcomes.

### Supported Task Sources

| Source | Location | Auto-detected |
|--------|----------|---------------|
| [Beads](https://github.com/steveyegge/beads) | `.beads/` in project | Yes |
| Claude Code Tasks | `~/.claude/todos/` | Yes |

### How It Works

1. When you work on a task, Claude searches for relevant memories
2. Those memories get linked to your task
3. When you mark the task as done, the memories that helped are automatically boosted

```
Task completed
    → Memories used during this task get +0.2 boost
    → Good advice rises to the top over time
```

### Unified Task Commands

```bash
# List tasks from all sources
mem tasks

# Filter by source
mem tasks --source beads      # Beads tasks only
mem tasks --source claude     # Claude Code tasks only

# Sync outcomes for completed tasks
mem tasks-sync

# Get context with relevant memories
mem tasks-context

# View statistics
mem tasks-stats
```

### Legacy Beads Commands (Still Supported)

```bash
mem beads-sync
mem beads-context
mem beads-stats
mem beads-link <memory_id>
```

### What Gets Recorded

| Task Status | Memory Outcome | Score Change |
|-------------|----------------|--------------|
| completed/done | worked | +0.2 |
| cancelled | failed (if enabled) | -0.3 |
| blocked | partial | +0.05 |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CLAUDE_CODE_TASK_LIST_ID` | Filter to specific task list |
| `CLAUDE_CODE_TODOS_DIR` | Custom todos directory |

---

## Web UI

Memory Layer includes a web interface for browsing and managing memories.

### Starting the Web UI

```bash
# Start the server
mem serve --rest --port 8080

# Open in browser
# http://localhost:8080
```

### Features

- **Dashboard**: Statistics with color-coded category bars
- **Memories**: Sortable list with filters (category, project, search)
- **Search**: Semantic (related concepts) or Keyword (exact match) modes
- **Tasks**: View tasks from Beads and Claude Code with context
- **Add Memory**: Form with category selection
- **Outcomes**: Record feedback on memories
- **Theme**: Light/dark mode toggle

---

## Troubleshooting

### "mem: command not found"

Make sure memory-layer is installed and your PATH includes pip's bin directory:
```bash
pip install git+https://github.com/runtimenoteslabs/memory-layer.git
# or
python -m pip install git+https://github.com/runtimenoteslabs/memory-layer.git
```

### Memories not loading in Claude Code

Re-run the plugin installation:
```bash
cd your-project
mem install-plugin
```

Then restart Claude Code.

### Want to start fresh?

Delete the database:
```bash
rm ~/.memory-layer/memories.db
```

---

## Getting Help

- Report issues: https://github.com/runtimenoteslabs/memory-layer/issues
- See all CLI options: `mem --help`
- See command help: `mem <command> --help`

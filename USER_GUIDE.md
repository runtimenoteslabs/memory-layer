# Runtime Memory user guide

How to use Runtime Memory with Claude Code and other AI coding assistants.

---

## What it does

Runtime Memory gives Claude Code and other AI coding assistants a memory that lasts
between sessions. It keeps:

- Your project conventions ("we use tabs, not spaces")
- Past decisions ("we chose PostgreSQL because...")
- Gotchas ("the CI server needs Node 18, not 20")
- How often each memory's advice worked and how often it failed, from your feedback

---

## Setup

```bash
# 1. Install, with semantic search, extraction and the MCP server
pip install "runtime-memory[all]"

# 2. Go to your project
cd your-project

# 3. Enable for Claude Code
mem install-plugin

# 4. Start Claude Code as usual
claude
```

From the next session on, Claude Code loads relevant memories when it starts.

---

## Daily use

### What happens on its own

The plugin's hooks run without any action from you:
- At session start, relevant memories are loaded into the session.
- Before Claude Code compacts its context, extraction stores what the session learned.
  Extraction needs `ANTHROPIC_API_KEY`.
- At session end, a summary of the session is generated.

### Storing something on purpose

Tell Claude:

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

### Finding past knowledge

Ask Claude:

> "What's our convention for error handling?"

> "What did we decide about the database?"

Or use the slash command:
```
/recall database setup
```

### Giving feedback

When Claude's advice works or fails, tell it:

> "Thanks, that worked!"

> "That didn't work, the tests still fail"

> "That partially helped, but I also needed to restart the server"

Claude records the outcome against the memories the advice came from. Among the
memories relevant to a query, those with a better record rank higher.

---

## Slash commands

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

## Terminal commands

The same operations are available from the terminal:

```bash
# See all memories
mem list

# Search memories
mem search "keyword"

# Add a memory
mem add "Always use async/await" -c convention

# Get project summary
mem context

# View statistics, including the search mode in use
mem stats

# See why a search returns what it does
mem why "keyword"

# Give feedback
mem outcome <id> worked
```

---

## Memory categories

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

## Tips

### Be specific

**Good:** "Use async/await for all database calls in this project"

**Less useful:** "use async"

### Include the reason

**Good:** "We use PostgreSQL because we need ACID transactions for payment processing"

**Less useful:** "We use PostgreSQL"

### Give feedback

Each "that worked" or "that didn't help" records an outcome. Without feedback,
memories rank on relevance, extraction confidence and age alone.

### Categories are optional

Search matches on content, so a memory filed under an unexpected category is still
found.

---

## How it works

1. **Storage**: Memories are stored locally in a SQLite database (`~/.runtime-memory/memories.db`)

2. **Retrieval**: When you ask questions, relevant memories are automatically searched using a hybrid approach:
   - Semantic similarity (what you're asking about)
   - Outcome record (what actually helped before)
   - Recency (recent memories weighted higher)
   - Extraction confidence

3. **Relevance first**: A search keeps the memories most relevant to your query and drops the rest, then ranks those on outcome, confidence and age. Outcome records reorder only the memories that match the query.

4. **Learning**: Each memory counts how often it worked and how often it failed
   - "worked" adds a success
   - "failed" adds a failure, which weighs 1.5 successes
   - "partial" adds a quarter of a success

   The counts give a score between -1 and 1, and a single observation counts for
   less than a long record. Among relevant memories, those with a better record
   rank higher. A memory that failed twice with no successes is left out of
   retrieval until its failures fade; each count halves every 90 days.

5. **Privacy**: Memories are stored only on your machine. Extraction, when enabled, sends session transcripts to Anthropic's API.

---

## The first run

With the `embedding` extra, the first search downloads a sentence embedding model of
about 100 MB and caches it, so that search takes longer than later ones. Without the
extra, search is keyword-only; `mem stats` shows which mode is in use.

The SQLite database is created on first use at `~/.runtime-memory/memories.db`. To
check the database and the engine:
```bash
mem check
```

---

## Task integration

Runtime Memory integrates with task trackers to automatically learn from task outcomes.

### Supported task sources

| Source | Location | Auto-detected |
|--------|----------|---------------|
| [Beads](https://github.com/steveyegge/beads) | `.beads/` in project | Yes |
| Claude Code Tasks | `~/.claude/todos/` | Yes |

### How tasks record outcomes

1. When you work on a task, Claude searches for relevant memories
2. Those memories get linked to your task
3. When you mark the task as done, the linked memories are recorded as having worked

### Task commands

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

### Beads commands, still supported

```bash
mem beads-sync
mem beads-context
mem beads-stats
mem beads-link <memory_id>
```

### What gets recorded

| Task status | Memory outcome | Adds |
|-------------|----------------|------|
| completed/done | worked | one success |
| cancelled | failed (if enabled) | one failure |
| blocked | partial | a quarter of a success |

### Environment variables

| Variable | Description |
|----------|-------------|
| `CLAUDE_CODE_TASK_LIST_ID` | Filter to specific task list |
| `CLAUDE_CODE_TODOS_DIR` | Custom todos directory |

---

## Web UI

Runtime Memory includes a web interface for browsing and managing memories.

### Starting the web UI

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

Make sure runtime-memory is installed and your PATH includes pip's bin directory:
```bash
pip install runtime-memory
# or
python -m pip install runtime-memory
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
rm ~/.runtime-memory/memories.db
```

---

## Getting help

- Report issues: https://github.com/runtimenoteslabs/memory-layer/issues
- See all CLI options: `mem --help`
- See command help: `mem <command> --help`

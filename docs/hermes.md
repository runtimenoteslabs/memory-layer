# Using Runtime Memory with Hermes Agent

Hermes Agent stores memory in a small note file, capped at about 2200 characters.
Everything in that file is pasted into every prompt, so raising the cap makes
every turn more expensive.

Runtime Memory replaces that file with retrieval. The store is uncapped, and each
turn receives only the memories relevant to it, ranked partly by whether they
have worked before. It is the same SQLite database Claude Code and any MCP client
already use, so a fact learned in one tool is available in the others.

## Install

Runtime Memory runs in-process as a Hermes memory provider, so install it into the
Python environment Hermes itself runs in.

```bash
# Find the environment Hermes runs in
realpath "$(command -v hermes)"

# Install into it (adjust the path to match)
~/.hermes/hermes-agent/venv/bin/python -m pip install runtime-memory

# Activate it
hermes config set memory.provider runtimememory
```

The distribution is `runtime-memory` and it imports as `runtime_memory`. The
repository is still named memory-layer, and an unrelated package holds
`memory-layer` on PyPI, so neither of those names installs this project.

Restart Hermes, or run `/reset`. Hermes finds the provider through the
`hermes_agent.memory_providers` entry point, so you do not edit its code or
config files by hand. Once installed, `hermes memory setup` lists `runtimememory`.

The base install adds `aiosqlite`, `numpy`, and `watchdog`. For semantic search,
add the embedding extra:

```bash
~/.hermes/hermes-agent/venv/bin/python -m pip install \
    'runtime-memory[embedding]'
```

Without the extra, retrieval uses only the BM25 half of the hybrid, and the
provider logs a warning at startup naming the interpreter to install into. With
it, the embedding model loads at startup rather than during your first turn. The
startup line reports which you got, `search=hybrid` or `search=keyword`, and so
does every recall in the evaluation trace.

If the Hermes environment already holds memories embedded by a different model,
keep using that model. Mixing embedding models in one store leaves the older
vectors unmatched by semantic search, though BM25 still finds them.

## Configuration

Every setting is an environment variable, so nothing is written to Hermes' config
beyond the provider name.

| Variable | Default | Purpose |
|----------|---------|---------|
| `RUNTIME_MEMORY_DB` | `~/.runtime-memory/memories.db` | The shared store |
| `RUNTIME_MEMORY_EMBEDDING` | `local` | `local`, `null`, `mock`, `openai`, `voyage` |
| `RUNTIME_MEMORY_RECALL_LIMIT` | `8` | Memories injected per turn |
| `RUNTIME_MEMORY_MIN_SCORE` | `0.0` | Floor on the combined score for injection |
| `RUNTIME_MEMORY_RELEVANCE_POOL_FACTOR` | `2` | Two-stage recall, see below |
| `RUNTIME_MEMORY_CONFLICT_COUNTERPARTS` | `3` | Contradicting memories added below a recall, see Outcome feedback |
| `RUNTIME_MEMORY_PROJECT` | workspace name | Project scope for memories |
| `RUNTIME_MEMORY_MIRROR_WRITES` | `true` | Mirror built-in memory writes |
| `RUNTIME_MEMORY_EXTRACT_ON_END` | `false` | Extract memories at session end |
| `RUNTIME_MEMORY_SESSION_OUTCOMES` | `true` | Record the session's test verdict at session end, see Outcome feedback |
| `RUNTIME_MEMORY_ASK_CITATIONS` | `true` | Ask the agent to name the memories it uses |
| `RUNTIME_MEMORY_HERMES_TRACE` | `hermes-trace.jsonl` beside the store | Path for the trace, or `off` |

Recall runs in two stages. The `recall_limit x factor` memories most relevant to
the message form a pool, a memory with no relevance at all never enters it, and
only the pool competes on the combined score. At `1` the other signals can only
reorder what relevance picked. At the default `2` they can also replace a
memory with a less relevant one whose record is much better. Set `off` to let
every memory compete on the combined score, as 3.x did. Only the semantic part
of that score depends on the message, so a memory that has worked before can
then take a slot from one that matches the message far better.

`local` degrades on its own: without `sentence-transformers` it indexes no
vectors and retrieval scores on keywords alone. Set `null` to choose that even
when the model is installed, which also silences the startup warning. Avoid
`mock` against a shared store, since its hash-derived vectors are meaningless
next to real ones.

The store sits outside `HERMES_HOME` so that Claude Code and MCP clients can
share it. The provider reports its path through `backup_paths()`, so
`hermes backup` includes it.

## Tools

| Tool | Purpose |
|------|---------|
| `runtimememory_remember` | Save a durable fact with a category |
| `runtimememory_recall` | Search memory directly |
| `runtimememory_outcome` | Report whether recalled memories helped |
| `runtimememory_stats` | Summarize the store |

## How memories are written

Memories reach the store three ways. None of them writes raw conversation turns,
which would bury the curated memories that retrieval depends on.

- **The remember tool**, when the model judges a fact durable.
- **Mirrored built-in writes.** When Hermes writes to its own note file, the same
  fact also lands in the store. Hermes keeps its small file for always-on
  context, and the store keeps the full history.
- **End-of-session extraction**, off by default. It makes one LLM call over the
  finished session. The call is shown the project's stored memories, skips what
  they already say, counts the ones the session confirmed, and records which
  stored memory each new one updates, contradicts, or extends.

Subagent, cron, and flush contexts read the store but never write to it, so
background runs cannot fill it with duplicates.

## Outcome feedback

Each memory counts the times it worked and the times it failed; `partial` adds a
quarter of a success. A failure weighs 1.5 successes, so a memory that misleads
once needs more than one success to read as reliable again. Each count halves
every 90 days. A memory that failed twice with no successes is not recalled until
its failures fade.

Injected memories carry their id and their record, as counts, so the model can
cite a specific memory and can see how much evidence there is:

```
- [workaround] Delete the lockfile (worked 2 times, failed 1 time) `9a9f`
```

`runtimememory_outcome` needs the ids of the memories the outcome is about; a
call without them is declined, and the trace records it as such.

When a session ends, the provider records an outcome without the agent's help.
The verdict is the last test run in the session's tool results, from pytest,
unittest, Jest, Cargo, or Go. It goes to the recalled memories the session acted
on:

- memories the agent named by id in its replies or tool arguments
- `command` memories whose quoted command the agent ran
- with extraction on, memories the extraction call judged the agent followed

No other recalled memory gets the verdict, a memory the agent already scored with
`runtimememory_outcome` is not scored again, and a session that ran no tests
records nothing. The block ends by asking the agent to name the ids of memories it
acts on, since that is how most attributions are found. To leave that line out,
set `RUNTIME_MEMORY_ASK_CITATIONS=false`. To stop recording session outcomes, set
`RUNTIME_MEMORY_SESSION_OUTCOMES=false`.

When two recalled memories are stored as contradicting each other, each is
marked with the other's id, and the block asks the model to check which one
holds before relying on either. When a recalled memory contradicts one that was
not recalled, that memory is added below the recall, up to
`RUNTIME_MEMORY_CONFLICT_COUNTERPARTS`, so the model sees both sides:

```
- [convention] Amounts are float, rounded to cents `4c1d` (contradicts `c21e`)

Also stored, and contradicting a memory above:
- [convention] Amounts are Decimal, never float `c21e` (contradicts `4c1d`)
```

`runtimememory_recall` lists the same links in each result's `contradicts`.

## Evaluation trace

The provider writes one JSONL record per recall, write, confirmation, outcome,
and extraction call to `hermes-trace.jsonl` beside the store. Set
`RUNTIME_MEMORY_HERMES_TRACE` to write it somewhere else, or to `off` to stop
tracing:

```bash
export RUNTIME_MEMORY_HERMES_TRACE=~/traces/hermes-run.jsonl
```

The trace holds each message the recall searched with, so it contains your
prompts. `mem stats` summarises it: recalls by search mode, the average size of
the injected block, outcomes by origin, and extraction tokens.

Records share a `turn_id`, so joining `recall` to `outcome` on it reconstructs,
per turn, which memories were injected and whether they helped.

```json
{"event": "recall", "turn_id": "a1b2", "query": "how do I run the tests?",
 "retrieved": [{"memory_id": "9a9f", "score": 0.82, "outcome_score": 0.33,
                "worked": 1.0, "failed": 0.0}]}
{"event": "outcome", "turn_id": "a1b2", "outcome": "worked",
 "memory_ids": ["9a9f"], "origin": "tool"}
```

A recall also records its `search_mode` (`hybrid` or `keyword`), any
`counterparts` added below it, the `contradicts` marks shown, and `block_chars`,
the length of the injected block in characters. The block is sent with every
model call in the turn, so its length is what memory adds to the agent's input.
A `usage` record gives the model, calls, and input and output tokens of each
extraction, whether or not it succeeded. An outcome's
`origin` is `tool` when the model reported it, `declined` when the call named no
memories, and `cited`, `command`, or `extraction` for an outcome recorded at
session end, after the attribution that found the memory. A `confirm` record lists the stored memories that an extraction
confirmed instead of storing them again.

A failed trace write never breaks a turn.

## Design notes

- **Recall is synchronous.** Hermes expects `prefetch()` to return a result warmed
  on the previous turn, because its bundled providers call network services. This
  store is local, so recall runs against the current turn's query. Retrieval then
  matches the question actually asked, which also keeps outcome attribution
  correct.
- **The provider tracks Hermes' plugin API.** It imports `MemoryProvider` from
  Hermes and needs updating if that contract changes. Importing
  `runtime_memory.hermes` outside Hermes falls back to a local shim, so the package
  stays importable and testable without Hermes installed.

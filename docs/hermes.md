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
| `RUNTIME_MEMORY_CONFLICT_COUNTERPARTS` | `3` | Contradicting memories added beside a recall, see Outcome feedback |
| `RUNTIME_MEMORY_PROJECT` | workspace name | Project scope for memories |
| `RUNTIME_MEMORY_MIRROR_WRITES` | `true` | Mirror built-in memory writes |
| `RUNTIME_MEMORY_EXTRACT_ON_END` | `false` | Extract memories at session end |
| `RUNTIME_MEMORY_HERMES_TRACE` | unset | Path for the evaluation trace |

Recall runs in two stages. The `recall_limit x factor` memories most relevant to
the message form a pool, a memory with no relevance at all never enters it, and
only the pool competes on the combined score. At `1` the other signals can only
reorder what relevance picked; at the default `2` they can replace up to every
slot with the next most relevant memory, which is what lets a memory that keeps
failing drop out. Set `off` to let every memory compete on the combined score,
as 3.x did. Only the semantic part of that score depends on the message, so a
memory that has worked before can then take a slot from one that matches the
message far better.

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
- **End-of-session extraction**, off by default. It runs one LLM call over the
  finished session, which is shown the project's stored memories: it leaves out
  what they already say, counts the ones the session confirmed, and records
  which stored memory a new one updates, contradicts, or extends.

Subagent, cron, and flush contexts read the store but never write to it, so
background runs cannot fill it with duplicates.

## Outcome feedback

Each memory counts the times it worked and the times it failed; `partial` adds a
quarter of a success. A failure weighs 1.5 successes, so a memory that misleads
once needs more than one success to read as reliable again. Counts halve over 90
days. A memory that failed twice with no successes is not recalled until its
failures fade.

Injected memories carry their id and their record, as counts, so the model can
cite a specific memory and can see how much evidence there is:

```
- [workaround] Delete the lockfile (worked 2 times, failed 1 time) `9a9f`
```

`runtimememory_outcome` needs the ids of the memories the outcome is about; a
call without them is declined, and the trace records it as such.

Two memories stored as contradicting each other are marked in the block, each
naming the other, with a line asking the model to check which holds before
relying on either. When a recalled memory contradicts one that was not recalled,
that one is added under the recall, up to `RUNTIME_MEMORY_CONFLICT_COUNTERPARTS`,
so a contested memory never arrives looking settled:

```
- [convention] Amounts are float, rounded to cents `9a9f` (contradicts `c21e`)

Also stored, and contradicting a memory above:
- [convention] Amounts are Decimal, never float `c21e` (contradicts `9a9f`)
```

`runtimememory_recall` lists the same links in each result's `contradicts`.

## Evaluation trace

Setting `RUNTIME_MEMORY_HERMES_TRACE` writes one JSONL record per recall, write,
confirmation, and outcome:

```bash
export RUNTIME_MEMORY_HERMES_TRACE=~/traces/hermes-run.jsonl
```

Records share a `turn_id`, so joining `recall` to `outcome` on it reconstructs,
per turn, which memories were injected and whether they helped.

```json
{"event": "recall", "turn_id": "a1b2", "query": "how do I run the tests?",
 "retrieved": [{"memory_id": "9a9f", "score": 0.82, "outcome_score": 0.33,
                "worked": 1.0, "failed": 0.0}]}
{"event": "outcome", "turn_id": "a1b2", "outcome": "worked",
 "memory_ids": ["9a9f"], "origin": "auto"}
```

A recall also records its `search_mode` (`hybrid` or `keyword`), any
`counterparts` added beside it, and the `contradicts` marks shown. A `confirm`
record lists stored memories an extraction learned again instead of storing
again.

Tracing stays off until the variable is set, and a failed trace write never
breaks a turn.

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

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

Without the extra, retrieval uses only the BM25 half of the hybrid. With it, the
embedding model loads at startup rather than during your first turn.

If the Hermes environment already holds memories embedded by a different model,
keep using that model. Mixing embedding models in one store leaves the older
vectors unmatched by semantic search, though BM25 still finds them.

## Configuration

Every setting is an environment variable, so nothing is written to Hermes' config
beyond the provider name.

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEMORY_LAYER_DB` | `~/.runtime-memory/memories.db` | The shared store |
| `MEMORY_LAYER_EMBEDDING` | `local` | `local`, `null`, `mock`, `openai`, `voyage` |
| `MEMORY_LAYER_RECALL_LIMIT` | `8` | Memories injected per turn |
| `MEMORY_LAYER_MIN_SCORE` | `0.0` | Relevance floor for injection |
| `MEMORY_LAYER_PROJECT` | workspace name | Project scope for memories |
| `MEMORY_LAYER_MIRROR_WRITES` | `true` | Mirror built-in memory writes |
| `MEMORY_LAYER_EXTRACT_ON_END` | `false` | Extract memories at session end |
| `MEMORY_LAYER_HERMES_TRACE` | unset | Path for the evaluation trace |

`local` degrades on its own: without `sentence-transformers` it indexes no
vectors and retrieval scores on keywords alone. Set `null` to force that even
when the model is installed. Avoid `mock` against a shared store, since its
hash-derived vectors are meaningless next to real ones.

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
- **End-of-session extraction**, off by default. It runs an LLM pass over the
  finished session and costs API calls.

Subagent, cron, and flush contexts read the store but never write to it, so
background runs cannot fill it with duplicates.

## Outcome feedback

A memory that worked scores +0.2, one that failed -0.3, and one that partly
helped +0.05. Failures weigh more than successes, so a memory that misleads once
needs several successes to recover its ranking.

Injected memories carry their id and their track record, so the model can cite a
specific memory and can see that it has failed before. Calling
`runtimememory_outcome` with no ids scores whatever was recalled for that turn.

## Evaluation trace

Setting `MEMORY_LAYER_HERMES_TRACE` writes one JSONL record per recall, write,
and outcome:

```bash
export MEMORY_LAYER_HERMES_TRACE=~/traces/hermes-run.jsonl
```

Records share a `turn_id`, so joining `recall` to `outcome` on it reconstructs,
per turn, which memories were injected and whether they helped.

```json
{"event": "recall", "turn_id": "a1b2", "query": "how do I run the tests?",
 "retrieved": [{"memory_id": "9a9f", "score": 0.82, "outcome_score": 0.2}]}
{"event": "outcome", "turn_id": "a1b2", "outcome": "worked",
 "memory_ids": ["9a9f"], "origin": "auto"}
```

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

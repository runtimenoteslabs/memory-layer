# Changelog

All notable changes to Runtime Memory will be documented in this file.

## [Unreleased] - 4.0.0

Retrieval decides on relevance first, and an outcome has to name what it is
about. Both changes come from four pre-registered Tier 2 evaluation runs; each is
stated below with what it was measured against.

### Changed

- **Retrieval is two-stage by default.** `relevance_pool_factor` defaults to 2.0
  instead of off: a search keeps the `ceil(limit x factor)` most relevant
  candidates, drops those with no relevance, and ranks only those on the full
  score. **A query that matches nothing now returns nothing** rather than
  whatever scored best on the other signals.
- **Frequency left the default score** (`frequency_weight` 0.15 to 0.0), and its
  weight went to semantic (0.35 to 0.55). Frequency counts retrievals, which is
  not evidence that a memory helped, and it compounds. One run found a memory
  that failed every time it was used holding a top-8 place on it; another found
  a correct memory that had taken no outcome falling out of retrieval while
  memories that had taken one rose past it.
- **Category boosts default to neutral.** They decided which memories were
  injected in two runs: once seating a memory that ranked about 25th of 38 on
  relevance at rank 1, once keeping the one memory that would have held a rule
  out of all 45 prompts because its category was multiplied by 0.9. Pass
  `category_boosts` to set your own.
- **Recency decays from `created_at`,** a memory's age, rather than from
  `updated_at`. `recency_from_created=False` restores the old reading.
- **The semantic score is scaled relative to the query's best match**
  (`RetrievalConfig.semantic_scaling`, `relative` by default, `fixed` for 3.x).
  BM25 is divided by the highest BM25 among the candidates, and cosine, clipped
  at zero, by the highest cosine, so the best match scores 1.0 and no match
  still scores 0.0. Under the 3.x scaling, BM25 / (BM25 + 1) and
  (cos + 1) / 2, a task-length query gives raw BM25 in the tens for most
  memories, so the keyword part sat near 1 for all of them. Without an
  embedding backend the top candidates' semantic scores then spanned a few
  hundredths, less than the confidence weight moves a memory, so the
  extractor's self-reported confidence decided their order rather than
  relevance. The order by semantic score alone is unchanged, and on a
  stratified 120-question sample of LongMemEval-S recall@10 of the evidence
  sessions went from 0.982 to 0.988; what changes is how much relevance counts
  against the other signals. `min_vector_similarity` applies to fixed scaling
  only.
- **Outcomes are kept as evidence, not a running score.** Each memory counts the
  times it worked and the times it failed (`worked`, `failed`, schema 3), and
  every count halves over 90 days. Its outcome score is
  `(worked - 1.5 x failed) / (worked + 1.5 x failed + 2)`, read at search time so
  old evidence fades between outcomes. 3.x moved the score +0.2, -0.3 or +0.05
  per outcome and clamped it to [-1, 1]: one observation read like a settled
  record, nothing aged, and a memory at the floor lost nothing from another
  failure. Now one success scores 0.33 and ten score 0.83, and a failure still
  weighs 1.5 successes. `core.outcomes.OutcomeModel` holds the parameters, as
  `RetrievalConfig.outcome_model`; `legacy_3x()` sets it to None, which keeps
  the 3.x steps.
- **A memory that keeps failing is not retrieved** (`RetrievalConfig.failure_gate`,
  -0.5). Once relevance is scaled to the query's best match, a lead in relevance
  outweighs any outcome record, so ranking alone would keep injecting a memory
  the query matches best however often it failed. Two failures and no successes
  read -0.6 and gate it; one failure, which may have been blamed on the wrong
  memory, does not. As the failures decay the memory can be retrieved again.
  The gate is off with outcome weight 0, so an arm that switches outcome
  learning off switches it off too, and in `legacy_3x()`.
- **The Hermes block shows a memory's record as counts,** "worked 2 times,
  failed 1 time", where it said "has worked before" for one observation and
  ten alike. The recall trace and the recall and outcome tools report the
  counts as well.
- **`EngineConfig.counterpart_credit` is in successes:** 0.25, a quarter of one,
  where it was 0.05 of score.
- **A retrieval no longer moves `updated_at`.** Reading a memory is not a change
  to it, and while recency decayed from that field every retrieval made a memory
  look newly written.
- **`RuntimeMemoryProvider.record_outcome` requires memory ids.** A call without
  them previously applied the outcome to every memory recalled in the turn. That
  contract was measured in three Tier 2 runs and left the store worse than
  recording nothing each time: a turn's verdict reached both the memory that
  misled it and the memory that was right about the same thing, so both sank
  together. A call with no ids is now declined, logged, and traced with origin
  `declined`. The `runtimememory_outcome` tool says so, and lists `memory_ids`
  as required.

### Added

- **Memories can record how they relate to each other** (schema 2, migrated on
  open). `MemoryEngine.link()` stores a `Relationship` and `related()` reads
  them from either side, since a conflict has no direction. Extraction already
  classified new memories against stored ones and kept only the supersede; it
  now stores what it found, so a memory that contradicts another is marked as
  such instead of being left to be out-ranked.
- **`EngineConfig.counterpart_credit`**, 0.05 by default, 0.0 to turn off. When
  a memory is recorded as having failed, the memories it conflicts with gain
  that much. A contract that credits only what was acted on leaves the memory
  that was right about the same thing with nothing: it was not followed, so it
  earns nothing, while memories that were followed rise past it. The Tier 2
  evaluation watched a correct memory fall out of retrieval that way, after
  which the rule it spoke to broke on 8 of the next 10 tasks. Only a failure
  credits a counterpart, and only memories not named in the same call.
- **`RetrievalConfig.legacy_3x()`**, the 3.x weights, category boosts,
  single-stage scoring and recency reading, so evaluations run against 3.x stay
  reproducible and callers who tuned for them can ask by name.
- **`RetrievalConfig.recency_from_created`**, on by default.
- **`MemoryEngine.search_mode`**, `hybrid` or `keyword`. Without an embedding
  backend the engine falls back to keyword matching on purpose, and the two rank
  differently enough that anything comparing runs needs to know which it got.
  The Hermes provider reports it in its startup line and in every recall in the
  evaluation trace, and warns at startup when it has fallen back, naming the
  interpreter to install the extra into. `RUNTIME_MEMORY_EMBEDDING=null`
  chooses keyword matching and silences the warning.
- **`RUNTIME_MEMORY_RELEVANCE_POOL_FACTOR=off`** (or `none`) turns the two-stage
  pool off from the environment.
- **Extraction is shown what is already stored**
  (`ExtractionConfig.stored_context_limit`, 50 by default, 0 for the old
  behaviour). The call is told not to extract what a stored memory already
  says, in any words; to name the stored memory a new one updates, conflicts
  with or extends; and to list the stored memories the conversation confirmed.
  Before, nothing stopped a session from storing a rule again in new words, and
  a store written by a few sessions of the same project could hold five copies
  of each rule, which then took several of a recall's slots. Relations were
  found by a separate classifier call per candidate pair, among memories of the
  same category that shared words, so a restatement filed under another
  category was never compared at all. Now they come from the call extraction
  already makes. A project with more stored memories than the limit shows the
  ones most relevant to the transcript.
- **`MemoryEngine.confirm()`** and **`EngineConfig.skip_exact_duplicates`**, on
  by default. `add` with the same text as a live memory in the same project,
  once case, spacing and end punctuation are set aside, returns that memory
  and counts a confirmation (`metadata["confirmations"]`,
  `metadata["last_confirmed"]`) instead of storing a copy. Nothing looser is
  merged: two notes that say opposite things can share every word, so a
  similarity threshold cannot tell a restatement from a contradiction.
  Confirmations from extraction land in the same place, and the Hermes trace
  records them as their own `confirm` event. Nothing ranks on them yet.
- **The Hermes block shows contradictions.** Two recalled memories stored as
  contradicting each other are both marked, each naming the other, and the
  block asks the model to check which holds before relying on either. A
  recalled memory that contradicts one not recalled brings that one in under
  the recall, up to `RUNTIME_MEMORY_CONFLICT_COUNTERPARTS` (3), so a contested
  memory never arrives alone looking settled. Ranking alone did not keep the
  right side of a contradiction in the prompt: the Tier 2 evaluation watched a
  correct memory drop out while the wrong one it contradicted stayed. The
  recall trace records the counterparts added and the marks shown, and
  `runtimememory_recall` lists each result's contradictions.

### Fixed

- **README and USER_GUIDE described category routing that no search performs.**
  `CategoryRouter` exists and is tested, but nothing calls it, so the documented
  1.5x boost for troubleshooting memories on an error-shaped query never
  happened. The docs now say what retrieval does.
- **The Hermes guide still described the relevance pool as off by default.**
- **The Hermes guide said an outcome without ids scores the whole recall,** which
  4.0 declines.
- **Extraction failed whenever the model began its answer with a thinking
  block.** Claude 5 models think by default, and the response was read from its
  first block, so such a response failed the whole extraction with
  `'ThinkingBlock' object has no attribute 'text'`. The conflict classifier went
  through the same call. The answer is now read from the text blocks, and a
  refusal is reported as a failed extraction.

### Migration

- Scoring changes for every caller. To keep 3.x behaviour, build the engine with
  `RetrievalConfig.legacy_3x()`.
- Callers of `record_outcome` that relied on the no-ids default must pass the ids
  the outcome is about. The provider's recall results and the injected block both
  carry them.
- Stores migrate to schema 3 on open. A stored 3.x score becomes the counts it
  stands for, +0.2 per success and -0.3 per failure, dated from the memory's
  last update. The stored score is left as it was until the next outcome, but
  ranking reads the counts.
- Code that asserted exact outcome scores after `record_outcome` will see the
  new values; `legacy_3x()` keeps the old ones.

## [3.1.0] - 2026-09-17

### Added

- **`RetrievalConfig.outcome_gates_frequency`**, off by default. Retrieval
  counts as use, so a memory that keeps being retrieved keeps gaining frequency
  score even when every outcome recorded against it is a failure, and once its
  outcome score reaches the floor of -1.0 further failures cost it nothing. The
  Tier 2 evaluation found a wrong memory holding a top-8 place this way through
  a whole task sequence. With the option on, the frequency score is scaled by
  1 + outcome_score, clamped to [0, 1]: a memory at or above zero keeps its
  boost and one at the floor loses it. Default scoring is unchanged.
- **`RetrievalConfig.relevance_pool_factor`, two-stage retrieval**, off by
  default and settable as `RUNTIME_MEMORY_RELEVANCE_POOL_FACTOR`. Only the
  semantic signal depends on the query, and category boost multiplies the whole
  score, so a memory that barely matches can outrank one that matches well.
  Replaying both Tier 2 runs exactly, the top 8 shared only four memories on
  average with the eight most relevant, and the corpus v1 note that stayed
  injected on all 45 outcome-weight-0 tasks ranked about 25th of 38 on
  relevance; without category boost it would have been injected on 5. With a
  factor set, a search keeps the `ceil(limit x factor)` most relevant
  candidates, drops any with no relevance at all, and ranks only those on the
  full score. Default scoring is unchanged.

### Fixed

- **Session-end extraction was requested and silently did nothing.** It needs
  the `anthropic` package and a credential, and without either it failed on a
  background task at session end where the exception is swallowed, so the
  provider reported extraction as on while nothing was ever extracted. The
  provider now checks at startup and says what is missing, then runs without it.
  An unset `ANTHROPIC_API_KEY` is not taken to mean there is no credential: an
  auth token, a signed-in profile and workload identity federation each count.
- **The extraction model defaults were a generation behind**, and the call sent
  `temperature`, which Claude 4.6 and later reject with a 400. Extraction now
  defaults to `claude-sonnet-5` (and `claude-haiku-4-5` in the settings model)
  and no longer sends the parameter. The `temperature` field remains so existing
  configuration keeps loading; it is documented as ignored.
- An empty recall left no trace record, so a recall that found nothing and a
  recall that never happened were indistinguishable in the log. Both are
  recorded now.
- **Retrieval scored four signals, not the five that are documented, and
  weighted them differently.** The retriever used semantic 0.50, recency 0.25,
  frequency 0.15 and outcome 0.10, with extraction confidence absent from the
  formula. Everything describing the system, the README included, states
  semantic 0.35, outcome 0.25, recency 0.15, frequency 0.15 and confidence
  0.10. The scoring now matches: outcome carries the weight it is supposed to,
  and confidence is a term rather than a category-router detail. Rankings will
  shift, most visibly for memories with a recorded outcome.
- `RetrievalConfig` existed twice with different defaults, once in
  `core/retrieval.py` (which scores) and once in `core/config.py` (which does
  not). That is how the two drifted unnoticed. The settings one is now
  `RetrievalSettings`, and a test asserts the two sets of weights agree.
- The Hermes provider still defaulted to `~/.memory-layer/memories.db`, the
  pre-3.0 store, when `RUNTIME_MEMORY_DB` was unset. `hermes memory setup`
  offered that path as its default and wrote the answer into Hermes'
  environment, so a fresh install was configured to read a database that does
  not exist. It now uses the shared default, which resolves to
  `~/.runtime-memory/` and falls back to the old directory only when that one
  is actually there.

## [3.0.0] - 2026-09-11

The project is now **Runtime Memory**, published as `runtime-memory` and
imported as `runtime_memory`. The repository keeps its original name.

PyPI refused `memory-layer-ai` as too similar to an existing project, the
abandoned `memory-layer` placeholder. Rather than pick another variation on a
generic phrase, the package took a name of its own that matches the
runtimenoteslabs work it belongs to. No release before this one reached PyPI,
so nothing installed from there is affected.

### Changed

- **Distribution is `runtime-memory`**, previously `memory-layer-ai`. It was
  only ever published to TestPyPI under the old name.
- **Import is `runtime_memory`**, previously `memory_layer`. The distribution
  and import names now match, which is what lets Hermes resolve the provider
  with no mapping on its side.
- **Environment variables use the `RUNTIME_MEMORY_` prefix.** Any
  `MEMORY_LAYER_` variable still set is carried onto its current name at import,
  so an agent config written before the rename keeps working. An explicitly set
  current name always wins.
- **The store lives in `~/.runtime-memory/`.** When that directory does not
  exist and `~/.memory-layer/` does, the old one is used, so an existing install
  keeps its memories rather than starting empty. Move the directory when
  convenient.
- **The Hermes provider is `runtimememory`**, and its tools are
  `runtimememory_remember`, `_recall`, `_outcome` and `_stats`. Update Hermes
  with `hermes config set memory.provider runtimememory`.
- The Claude Code plugin and the bundled MCP server entry are both named
  `runtime-memory`.

### Added

- `rmem` as a second console script. `mem` is unchanged and both run the same
  CLI.
- `plugin.yaml` declares `pip_dependencies: ["runtime-memory"]`. Hermes checks a
  dependency by importing `dist_name.replace("-", "_")`, which resolves for this
  package without the upstream name mapping its bundled providers need.

## [2.2.2] - 2026-09-10

### Fixed

- Links in the README pointed at `USER_GUIDE.md` and `docs/hermes.md` by
  relative path, which resolves against the repository but not against the
  package page. They are absolute now.
- Logger names carried the package prefix twice, so every log line read
  `memory_layer.memory_layer.core.embeddings`. `get_logger()` adds the prefix
  only when the caller's name is not already inside the package.

### Added

- `mem check` reports the embedding backend: the model name when one is
  loaded, and otherwise a note that search is keyword-only along with the extra
  to install. A missing backend is reported as information rather than an
  issue, since keyword retrieval is a supported way to run.

### Changed

- The notice about `sentence-transformers` being absent is logged at INFO
  rather than WARNING. The CLI builds an embedding provider once per command,
  so at WARNING it printed on every invocation. Run with `-v` to see it, or
  `mem check` for the same information.

## [2.2.1] - 2026-09-10

### Fixed

- A base install now works without the embedding extra. `sentence-transformers`
  is optional, but asking for the `local` provider without it raised
  `ModelNotFoundError` on the first embed, which took down `mem add` and every
  other write. The factory now falls back to a null provider that indexes no
  vectors, leaving retrieval on the BM25 half of the hybrid. Install
  `memory-layer-ai[embedding]` to turn semantic search back on.
- The Hermes provider no longer selects `mock` embeddings when
  `sentence-transformers` is missing. Mock vectors are hash-derived, so writing
  them into a store that Claude Code and MCP clients share put meaningless
  vectors next to real ones. It now asks for `local` and lets the factory
  decide, which is the single place that check belongs.

### Added

- `null` is an accepted embedding provider name, so semantic search can be
  turned off deliberately. Useful against a store holding vectors from another
  model, where a fresh model's scores would be meaningless.
- The `mem` CLI reads `MEMORY_LAYER_DATABASE__PATH` as well as
  `MEMORY_LAYER_DB`. The settings docs named a third variable,
  `MEMORY_LAYER_DB_PATH`, that was recognised nowhere and silently fell through
  to the default store.

### Changed

- `mem check` reports a missing embedding backend as `unavailable` rather than
  `unhealthy`, and names the extra to install. Keyword retrieval is a working
  state, not a broken one.

## [2.2.0] - 2026-09-10

### Added

- Hermes Agent memory provider. Hermes discovers it through the
  `hermes_agent.memory_providers` entry point, so installing this package into
  the Hermes environment and setting `memory.provider` is enough to activate it,
  with no changes to Hermes itself. It replaces the built-in note file with
  per-turn retrieval over the store Claude Code and MCP clients already share.
  See [docs/hermes.md](docs/hermes.md).
- Four Hermes tools: `memorylayer_remember`, `memorylayer_recall`,
  `memorylayer_outcome`, and `memorylayer_stats`.
- Optional JSONL trace of recalls, writes, and outcomes, enabled by setting
  `MEMORY_LAYER_HERMES_TRACE`. Records share a `turn_id`, so a recall can be
  joined to the outcome it earned.

### Changed

- **The distribution is now named `memory-layer-ai`.** The import name is
  unchanged, so `import memory_layer` continues to work, but installs and
  uninstalls must use the new name. An unrelated package occupies `memory-layer`
  on PyPI. Anything depending on this project by distribution name, including a
  `pip uninstall`, needs updating.
- Provider status in Hermes reports `4 - Beta` rather than `3 - Alpha`, matching
  the state of the test suite and public releases.

### Fixed

- MCP server tool calls. `_ensure_engine()` passed an unsupported `db_path`
  keyword to `MemoryEngine` and never awaited `initialize()`, so `mem serve
  --mcp` listed its tools while every call to one failed. No test covered the
  branch, because every handler test injects a ready-made engine; the branch now
  has regression tests.

## [2.1.1] - 2026-01-24

### Fixed

- Tasks API route ordering, which broke `/tasks/stats` and `/tasks/context`.
- Version number and repository URL in `.claude-plugin/plugin.json`.
- Web UI header alignment against the main content area.

### Changed

- `.mcp.json` simplified: the MCP server advertises its tools at runtime rather
  than declaring them statically.
- Removed `docs/MIGRATION_GUIDE.md`. No public v1 exists to migrate from.

## [2.1.0] - 2026-01-24

### Added

- Claude Code Tasks integration, reading `~/.claude/todos/` and recording
  outcomes when tasks complete. Adds a unified Tasks API over Beads and Claude
  Code sources, `mem tasks*` CLI commands, `/tasks` REST endpoints, and
  `tasks_*` MCP tools.
- Web UI on `localhost:8080`: dashboard, filterable memory list, semantic and
  keyword search, unified tasks view, outcome recording, theme toggle, and JSON
  export.
- Production hardening: a custom exception hierarchy with readable messages,
  configuration handling, and observability.

[3.1.0]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v3.1.0
[3.0.0]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v3.0.0
[2.2.2]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.2.2
[2.2.1]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.2.1
[2.2.0]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.2.0
[2.1.1]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.1.1
[2.1.0]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.1.0

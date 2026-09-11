# Changelog

All notable changes to Runtime Memory will be documented in this file.

## [3.1.0] - 2026-09-11

### Fixed

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

# Changelog

All notable changes to memory-layer will be documented in this file.

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

[2.2.1]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.2.1
[2.2.0]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.2.0
[2.1.1]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.1.1
[2.1.0]: https://github.com/runtimenoteslabs/memory-layer/releases/tag/v2.1.0

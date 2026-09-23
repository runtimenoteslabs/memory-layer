"""SQLite storage layer for Runtime Memory.

Provides async database operations with:
- Connection pooling
- CRUD operations
- Outcome score tracking
- Soft delete (archival)
- Migration support
- Transaction support
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import stat
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from runtime_memory.core.logging import get_logger
from runtime_memory.core.models import (
    OUTCOME_SCORE_ADJUSTMENTS,
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
    Relationship,
    RelationType,
)
from runtime_memory.core.outcomes import EVIDENCE_OF, OutcomeModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

# Type alias to avoid conflict with the list() method
_List = builtins.list

logger = get_logger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 3

# SQL statements for schema creation
SCHEMA_SQL = """
-- Memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    outcome_score REAL DEFAULT 0.0,
    confidence REAL DEFAULT 1.0,
    importance REAL DEFAULT 0.5,
    use_count INTEGER DEFAULT 0,
    project TEXT,
    scope TEXT DEFAULT 'project',
    source TEXT DEFAULT 'explicit',
    tags TEXT DEFAULT '[]',
    entities TEXT DEFAULT '[]',
    supersedes TEXT,
    archived INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    embedding TEXT,
    metadata TEXT DEFAULT '{}'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_outcome_score ON memories(outcome_score);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_project_category ON memories(project, category);
CREATE INDEX IF NOT EXISTS idx_memories_project_archived ON memories(project, archived);

-- How memories relate to each other (schema 2). A conflict recorded here is
-- what lets an outcome reach the memory that contradicts the one that failed,
-- instead of leaving it to be out-ranked.
CREATE TABLE IF NOT EXISTS memory_relations (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    PRIMARY KEY (source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Full-text search virtual table
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id,
    content,
    tags,
    entities,
    content='memories',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, id, content, tags, entities)
    VALUES (new.rowid, new.id, new.content, new.tags, new.entities);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, id, content, tags, entities)
    VALUES ('delete', old.rowid, old.id, old.content, old.tags, old.entities);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, id, content, tags, entities)
    VALUES ('delete', old.rowid, old.id, old.content, old.tags, old.entities);
    INSERT INTO memories_fts(rowid, id, content, tags, entities)
    VALUES (new.rowid, new.id, new.content, new.tags, new.entities);
END;
"""

# Migration SQL statements (version -> SQL)
async def _add_outcome_evidence(conn: aiosqlite.Connection) -> None:
    """Schema 3: decayed counts of the times a memory worked and failed.

    Written as a step rather than a script because SQLite cannot add a column
    only if it is missing, and a store whose version table was reset would
    otherwise fail to open on the duplicate. Stored scores become the counts
    they stand for (see ``core.outcomes.pseudo_counts``, which this mirrors),
    but only when the columns are new, so real counts are never overwritten.
    """
    cursor = await conn.execute("PRAGMA table_info(memories)")
    present = {row[1] for row in await cursor.fetchall()}
    added = False
    for column, ddl in (
        ("worked", "REAL DEFAULT 0.0"),
        ("failed", "REAL DEFAULT 0.0"),
        ("evidence_at", "TEXT"),
    ):
        if column not in present:
            await conn.execute(f"ALTER TABLE memories ADD COLUMN {column} {ddl}")
            added = True
    if added:
        await conn.execute(
            """
            UPDATE memories SET
                worked = CASE WHEN outcome_score > 0 THEN outcome_score / 0.2 ELSE 0.0 END,
                failed = CASE WHEN outcome_score < 0 THEN -outcome_score / 0.3 ELSE 0.0 END,
                evidence_at = CASE WHEN outcome_score != 0 THEN updated_at ELSE NULL END
            """
        )


MIGRATIONS: dict[int, str | Callable[[aiosqlite.Connection], Awaitable[None]]] = {
    2: """
        CREATE TABLE IF NOT EXISTS memory_relations (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            strength REAL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            PRIMARY KEY (source_id, target_id, relation_type)
        );
        CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_id);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_id);
    """,
    # Outcome evidence. Not in SCHEMA_SQL: a new store runs every migration after
    # creating the table, so this is where every store gains the columns.
    3: _add_outcome_evidence,
}


@dataclass
class StorageStats:
    """Statistics about the storage."""

    total_memories: int
    active_memories: int
    archived_memories: int
    by_category: dict[str, int]
    by_scope: dict[str, int]
    by_source: dict[str, int]
    avg_outcome_score: float
    total_uses: int


class StorageError(Exception):
    """Base exception for storage errors."""

    pass


class MemoryNotFoundError(StorageError):
    """Raised when a memory is not found."""

    pass


class ConnectionError(StorageError):
    """Raised when database connection fails."""

    pass


class MemoryStorage:
    """Async SQLite storage for memories.

    Provides connection pooling, CRUD operations, and transaction support.
    """

    def __init__(
        self,
        db_path: str | Path,
        pool_size: int = 5,
        timeout: float = 30.0,
        secure_permissions: bool = True,
    ) -> None:
        """Initialize storage.

        Args:
            db_path: Path to SQLite database file.
            pool_size: Number of connections in the pool.
            timeout: Connection timeout in seconds.
            secure_permissions: Whether to set secure file permissions (0600).
        """
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.timeout = timeout
        self.secure_permissions = secure_permissions

        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=pool_size)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the storage and connection pool.

        Creates the database file, applies schema, and populates connection pool.
        """
        async with self._lock:
            if self._initialized:
                return

            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create initial connection and schema
            conn = await self._create_connection()
            try:
                await self._apply_schema(conn)
                await self._run_migrations(conn)
            finally:
                await conn.close()

            # Set secure file permissions
            if self.secure_permissions and self.db_path.exists():
                self._set_secure_permissions()

            # Populate connection pool
            for _ in range(self.pool_size):
                conn = await self._create_connection()
                await self._pool.put(conn)

            self._initialized = True
            logger.info(f"Storage initialized at {self.db_path}")

    async def close(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except asyncio.QueueEmpty:
                break
        self._initialized = False
        logger.info("Storage closed")

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection.

        Returns:
            New aiosqlite connection.
        """
        try:
            conn = await aiosqlite.connect(
                self.db_path,
                timeout=self.timeout,
            )
            # Enable foreign keys and WAL mode for better concurrency
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA synchronous = NORMAL")
            conn.row_factory = aiosqlite.Row
            return conn
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}") from e

    async def _apply_schema(self, conn: aiosqlite.Connection) -> None:
        """Apply the database schema.

        Args:
            conn: Database connection.
        """
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()

    async def _run_migrations(self, conn: aiosqlite.Connection) -> None:
        """Run pending database migrations.

        Args:
            conn: Database connection.
        """
        # Get current version
        cursor = await conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        row = await cursor.fetchone()
        current_version = row[0] if row and row[0] else 0

        # Apply pending migrations
        for version in sorted(MIGRATIONS.keys()):
            if version > current_version:
                logger.info(f"Applying migration {version}")
                migration = MIGRATIONS[version]
                if callable(migration):
                    await migration(conn)
                else:
                    await conn.executescript(migration)
                await conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, datetime.now(UTC).isoformat()),
                )
                await conn.commit()

        # Record initial schema version if needed
        if current_version == 0:
            await conn.execute(
                "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now(UTC).isoformat()),
            )
            await conn.commit()

    def _set_secure_permissions(self) -> None:
        """Set secure file permissions (0600) on the database file."""
        try:
            self.db_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            # Also secure the WAL and SHM files if they exist
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(self.db_path) + suffix)
                if wal_path.exists():
                    wal_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError as e:
            logger.warning(f"Failed to set secure permissions: {e}")

    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a connection from the pool.

        Yields:
            Database connection.
        """
        if not self._initialized:
            await self.initialize()

        conn = await asyncio.wait_for(self._pool.get(), timeout=self.timeout)
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Start a transaction.

        Yields:
            Database connection with active transaction.

        Example:
            async with storage.transaction() as conn:
                await storage.create(memory1, conn=conn)
                await storage.create(memory2, conn=conn)
        """
        async with self._get_connection() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create(
        self,
        memory: Memory,
        conn: aiosqlite.Connection | None = None,
    ) -> Memory:
        """Create a new memory.

        Args:
            memory: Memory to create.
            conn: Optional connection for transaction.

        Returns:
            Created memory.
        """
        sql = """
            INSERT INTO memories (
                id, content, category, outcome_score, worked, failed, evidence_at,
                confidence, importance,
                use_count, project, scope, source, tags, entities, supersedes,
                archived, created_at, updated_at, embedding, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            memory.id,
            memory.content,
            memory.category.value,
            memory.outcome_score,
            memory.worked,
            memory.failed,
            memory.evidence_at.isoformat() if memory.evidence_at else None,
            memory.confidence,
            memory.importance,
            memory.use_count,
            memory.project,
            memory.scope.value,
            memory.source.value,
            json.dumps(memory.tags),
            json.dumps(memory.entities),
            memory.supersedes,
            1 if memory.archived else 0,
            memory.created_at.isoformat(),
            memory.updated_at.isoformat(),
            json.dumps(memory.embedding) if memory.embedding else None,
            json.dumps(memory.metadata),
        )

        if conn:
            await conn.execute(sql, params)
        else:
            async with self._get_connection() as c:
                await c.execute(sql, params)
                await c.commit()

        logger.debug(f"Created memory {memory.id}")
        return memory

    async def get(self, memory_id: str) -> Memory:
        """Get a memory by ID.

        Args:
            memory_id: Memory ID.

        Returns:
            Memory instance.

        Raises:
            MemoryNotFoundError: If memory not found.
        """
        sql = "SELECT * FROM memories WHERE id = ?"
        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, (memory_id,))
            row = await cursor.fetchone()

        if not row:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        return self._row_to_memory(row)

    async def get_many(self, memory_ids: _List[str]) -> _List[Memory]:
        """Get multiple memories by ID.

        Args:
            memory_ids: List of memory IDs.

        Returns:
            List of memories (in same order as IDs, missing ones excluded).
        """
        if not memory_ids:
            return []

        placeholders = ",".join("?" * len(memory_ids))
        sql = f"SELECT * FROM memories WHERE id IN ({placeholders})"

        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, memory_ids)
            rows = await cursor.fetchall()

        # Create lookup dict and preserve order
        memory_map = {self._row_to_memory(row).id: self._row_to_memory(row) for row in rows}
        return [memory_map[mid] for mid in memory_ids if mid in memory_map]

    async def update(
        self,
        memory: Memory,
        conn: aiosqlite.Connection | None = None,
    ) -> Memory:
        """Update an existing memory.

        Args:
            memory: Memory with updated fields.
            conn: Optional connection for transaction.

        Returns:
            Updated memory.

        Raises:
            MemoryNotFoundError: If memory not found.
        """
        # Ensure memory exists
        await self.get(memory.id)

        sql = """
            UPDATE memories SET
                content = ?, category = ?, outcome_score = ?, worked = ?, failed = ?,
                evidence_at = ?, confidence = ?,
                importance = ?, use_count = ?, project = ?, scope = ?, source = ?,
                tags = ?, entities = ?, supersedes = ?, archived = ?,
                updated_at = ?, embedding = ?, metadata = ?
            WHERE id = ?
        """
        params = (
            memory.content,
            memory.category.value,
            memory.outcome_score,
            memory.worked,
            memory.failed,
            memory.evidence_at.isoformat() if memory.evidence_at else None,
            memory.confidence,
            memory.importance,
            memory.use_count,
            memory.project,
            memory.scope.value,
            memory.source.value,
            json.dumps(memory.tags),
            json.dumps(memory.entities),
            memory.supersedes,
            1 if memory.archived else 0,
            memory.updated_at.isoformat(),
            json.dumps(memory.embedding) if memory.embedding else None,
            json.dumps(memory.metadata),
            memory.id,
        )

        if conn:
            await conn.execute(sql, params)
        else:
            async with self._get_connection() as c:
                await c.execute(sql, params)
                await c.commit()

        logger.debug(f"Updated memory {memory.id}")
        return memory

    async def delete(
        self,
        memory_id: str,
        hard_delete: bool = False,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        """Delete a memory.

        Args:
            memory_id: Memory ID.
            hard_delete: If True, permanently delete. If False, soft delete (archive).
            conn: Optional connection for transaction.

        Raises:
            MemoryNotFoundError: If memory not found.
        """
        # Ensure memory exists
        await self.get(memory_id)

        if hard_delete:
            sql = "DELETE FROM memories WHERE id = ?"
        else:
            sql = "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?"
        # A hard delete takes the memory's relations with it; an archived memory
        # keeps them, because archiving is reversible.
        relations_sql = "DELETE FROM memory_relations WHERE source_id = ? OR target_id = ?"

        if conn:
            if hard_delete:
                await conn.execute(sql, (memory_id,))
                await conn.execute(relations_sql, (memory_id, memory_id))
            else:
                await conn.execute(sql, (datetime.now(UTC).isoformat(), memory_id))
        else:
            async with self._get_connection() as c:
                if hard_delete:
                    await c.execute(sql, (memory_id,))
                    await c.execute(relations_sql, (memory_id, memory_id))
                else:
                    await c.execute(sql, (datetime.now(UTC).isoformat(), memory_id))
                await c.commit()

        action = "deleted" if hard_delete else "archived"
        logger.debug(f"Memory {memory_id} {action}")

    async def archive(
        self,
        memory_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> Memory:
        """Archive a memory (soft delete).

        Args:
            memory_id: Memory ID.
            conn: Optional connection for transaction.

        Returns:
            Archived memory.
        """
        memory = await self.get(memory_id)
        memory.archive()
        return await self.update(memory, conn=conn)

    async def unarchive(
        self,
        memory_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> Memory:
        """Unarchive a memory.

        Args:
            memory_id: Memory ID.
            conn: Optional connection for transaction.

        Returns:
            Unarchived memory.
        """
        memory = await self.get(memory_id)
        memory.archived = False
        memory.updated_at = datetime.now(UTC)
        return await self.update(memory, conn=conn)

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def list(
        self,
        project: str | None = None,
        category: MemoryCategory | None = None,
        scope: MemoryScope | None = None,
        source: MemorySource | None = None,
        include_archived: bool = False,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> _List[Memory]:
        """List memories with filters.

        Args:
            project: Filter by project.
            category: Filter by category.
            scope: Filter by scope.
            source: Filter by source.
            include_archived: Whether to include archived memories.
            min_score: Minimum outcome score.
            max_score: Maximum outcome score.
            limit: Maximum results.
            offset: Results offset.
            order_by: Column to order by.
            descending: Whether to sort descending.

        Returns:
            List of memories.
        """
        conditions = []
        params: list[Any] = []

        if not include_archived:
            conditions.append("archived = 0")

        if project is not None:
            conditions.append("project = ?")
            params.append(project)

        if category is not None:
            conditions.append("category = ?")
            params.append(category.value)

        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope.value)

        if source is not None:
            conditions.append("source = ?")
            params.append(source.value)

        if min_score is not None:
            conditions.append("outcome_score >= ?")
            params.append(min_score)

        if max_score is not None:
            conditions.append("outcome_score <= ?")
            params.append(max_score)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Validate order_by to prevent SQL injection
        valid_columns = {
            "created_at", "updated_at", "outcome_score",
            "use_count", "importance", "confidence",
        }
        if order_by not in valid_columns:
            order_by = "created_at"

        direction = "DESC" if descending else "ASC"
        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY {order_by} {direction}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def count(
        self,
        project: str | None = None,
        category: MemoryCategory | None = None,
        include_archived: bool = False,
    ) -> int:
        """Count memories with filters.

        Args:
            project: Filter by project.
            category: Filter by category.
            include_archived: Whether to include archived memories.

        Returns:
            Count of matching memories.
        """
        conditions = []
        params: list[Any] = []

        if not include_archived:
            conditions.append("archived = 0")

        if project is not None:
            conditions.append("project = ?")
            params.append(project)

        if category is not None:
            conditions.append("category = ?")
            params.append(category.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT COUNT(*) FROM memories WHERE {where_clause}"

        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()

        return row[0] if row else 0

    async def search_fts(
        self,
        query: str,
        project: str | None = None,
        include_archived: bool = False,
        limit: int = 20,
    ) -> _List[Memory]:
        """Full-text search for memories.

        Args:
            query: Search query.
            project: Filter by project.
            include_archived: Whether to include archived.
            limit: Maximum results.

        Returns:
            List of matching memories.
        """
        # Build FTS query
        sql = """
            SELECT m.* FROM memories m
            JOIN memories_fts fts ON m.id = fts.id
            WHERE memories_fts MATCH ?
        """
        params: list[Any] = [query]

        if not include_archived:
            sql += " AND m.archived = 0"

        if project is not None:
            sql += " AND m.project = ?"
            params.append(project)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute(sql, params)
                rows = await cursor.fetchall()
                return [self._row_to_memory(row) for row in rows]
            except aiosqlite.OperationalError:
                # FTS query syntax error, return empty
                return []

    # =========================================================================
    # Outcome and Use Count Operations
    # =========================================================================

    async def record_outcome(
        self,
        memory_id: str,
        outcome: Outcome,
        conn: aiosqlite.Connection | None = None,
        model: OutcomeModel | None = None,
    ) -> Memory:
        """Record an outcome for a memory.

        Adds the outcome to the memory's worked and failed counts, decayed to now,
        and sets its outcome score from them under ``model``. With no model the
        score takes the 3.x step instead (+0.2 worked, -0.3 failed, +0.05 partial,
        clamped to [-1, 1]); the counts are kept either way.

        Args:
            memory_id: Memory ID.
            outcome: Outcome to record.
            conn: Optional connection for transaction.
            model: How evidence becomes a score, or None for the 3.x step.

        Returns:
            Updated memory.
        """
        worked, failed = EVIDENCE_OF[outcome]
        legacy = None if model else OUTCOME_SCORE_ADJUSTMENTS[outcome]
        (memory,) = await self._add_evidence([memory_id], worked, failed, model, legacy, conn)
        logger.debug(
            f"Recorded {outcome.value} for memory {memory_id}: worked {memory.worked:.2f}, "
            f"failed {memory.failed:.2f}, score {memory.outcome_score:.3f}"
        )
        return memory

    async def add_evidence(
        self,
        memory_ids: _List[str],
        worked: float,
        failed: float = 0.0,
        model: OutcomeModel | None = None,
    ) -> _List[Memory]:
        """Add observations to several memories that are not an outcome of their own.

        Used for credit such as the counterpart of a memory that failed. It is
        recorded against the memories named and nothing else.

        Args:
            memory_ids: Memory IDs.
            worked: Successes to add to each.
            failed: Failures to add to each.
            model: How evidence becomes a score, or None for the 3.x step, under
                which ``worked`` moves the score 0.2 per success and ``failed``
                0.3 per failure.

        Returns:
            The updated memories, in the order given.
        """
        legacy = None if model else 0.2 * worked - 0.3 * failed
        async with self.transaction() as conn:
            return await self._add_evidence(memory_ids, worked, failed, model, legacy, conn)

    async def _add_evidence(
        self,
        memory_ids: _List[str],
        worked: float,
        failed: float,
        model: OutcomeModel | None,
        legacy_step: float | None,
        conn: aiosqlite.Connection | None,
    ) -> _List[Memory]:
        """Bring each memory's counts up to date, add to them, and rescore it."""
        now = datetime.now(UTC)
        counts = model or OutcomeModel()
        updated: _List[Memory] = []
        for memory_id in memory_ids:
            memory = await self.get(memory_id)
            new_worked, new_failed, score = counts.add(memory, worked, failed, now)
            if legacy_step is not None:
                score = max(-1.0, min(1.0, memory.outcome_score + legacy_step))
            sql = """
                UPDATE memories
                SET outcome_score = ?, worked = ?, failed = ?, evidence_at = ?, updated_at = ?
                WHERE id = ?
            """
            params = (score, new_worked, new_failed, now.isoformat(), now.isoformat(), memory_id)
            if conn:
                await conn.execute(sql, params)
            else:
                async with self._get_connection() as c:
                    await c.execute(sql, params)
                    await c.commit()
            memory.outcome_score = score
            memory.worked, memory.failed, memory.evidence_at = new_worked, new_failed, now
            memory.updated_at = now
            updated.append(memory)
        return updated

    # =========================================================================
    # Relations
    # =========================================================================

    async def add_relationship(self, relationship: Relationship) -> None:
        """Store how one memory relates to another, replacing any same-typed link.

        Args:
            relationship: The relationship to store.
        """
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO memory_relations
                    (source_id, target_id, relation_type, strength, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    relationship.source_id,
                    relationship.target_id,
                    relationship.relation_type.value,
                    relationship.strength,
                    relationship.created_at.isoformat(),
                    json.dumps(relationship.metadata),
                ),
            )
            await conn.commit()

    async def relationships(
        self,
        memory_id: str,
        relation_type: RelationType | None = None,
    ) -> _List[Relationship]:
        """Every relationship the memory takes part in, in either direction.

        A conflict has no direction, and a caller asking what a memory conflicts
        with should not have to know which side wrote the link.

        Args:
            memory_id: Memory ID.
            relation_type: Only this type, or every type.

        Returns:
            The relationships, oldest first.
        """
        sql = "SELECT * FROM memory_relations WHERE (source_id = ? OR target_id = ?)"
        params: list[Any] = [memory_id, memory_id]
        if relation_type is not None:
            sql += " AND relation_type = ?"
            params.append(relation_type.value)
        sql += " ORDER BY created_at"

        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
        return [
            Relationship(
                source_id=row["source_id"],
                target_id=row["target_id"],
                relation_type=RelationType(row["relation_type"]),
                strength=row["strength"],
                created_at=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    async def related_ids(
        self,
        memory_id: str,
        relation_type: RelationType,
    ) -> _List[str]:
        """The ids on the other side of a memory's relationships of one type.

        Args:
            memory_id: Memory ID.
            relation_type: The type to follow.

        Returns:
            The other memories' ids, without duplicates.
        """
        seen: dict[str, None] = {}
        for link in await self.relationships(memory_id, relation_type):
            other = link.target_id if link.source_id == memory_id else link.source_id
            seen.setdefault(other, None)
        return list(seen)

    async def record_outcomes(
        self,
        memory_ids: _List[str],
        outcome: Outcome,
        model: OutcomeModel | None = None,
    ) -> _List[Memory]:
        """Record an outcome for multiple memories.

        Args:
            memory_ids: Memory IDs.
            outcome: Outcome to record.
            model: How evidence becomes a score, or None for the 3.x step.

        Returns:
            List of updated memories.
        """
        async with self.transaction() as conn:
            memories = []
            for memory_id in memory_ids:
                memory = await self.record_outcome(memory_id, outcome, conn=conn, model=model)
                memories.append(memory)
            return memories

    async def increment_use_count(
        self,
        memory_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> Memory:
        """Increment the use count for a memory.

        Args:
            memory_id: Memory ID.
            conn: Optional connection for transaction.

        Returns:
            Updated memory.
        """
        # Since 4.0.0 a retrieval does not move updated_at. Reading a memory is
        # not a change to it, and while recency decayed from updated_at every
        # retrieval made a memory look newly written.
        sql = """
            UPDATE memories
            SET use_count = use_count + 1
            WHERE id = ?
        """
        params = (memory_id,)

        if conn:
            await conn.execute(sql, params)
        else:
            async with self._get_connection() as c:
                await c.execute(sql, params)
                await c.commit()

        return await self.get(memory_id)

    async def increment_use_counts(
        self,
        memory_ids: _List[str],
    ) -> None:
        """Increment use counts for multiple memories.

        Args:
            memory_ids: Memory IDs.
        """
        if not memory_ids:
            return

        async with self.transaction() as conn:
            for memory_id in memory_ids:
                await self.increment_use_count(memory_id, conn=conn)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def create_many(self, memories: _List[Memory]) -> _List[Memory]:
        """Create multiple memories in a transaction.

        Args:
            memories: Memories to create.

        Returns:
            Created memories.
        """
        async with self.transaction() as conn:
            for memory in memories:
                await self.create(memory, conn=conn)
        return memories

    async def delete_many(
        self,
        memory_ids: _List[str],
        hard_delete: bool = False,
    ) -> None:
        """Delete multiple memories in a transaction.

        Args:
            memory_ids: Memory IDs.
            hard_delete: If True, permanently delete.
        """
        async with self.transaction() as conn:
            for memory_id in memory_ids:
                with contextlib.suppress(MemoryNotFoundError):
                    await self.delete(memory_id, hard_delete=hard_delete, conn=conn)

    async def archive_low_score_memories(
        self,
        threshold: float = -0.5,
        project: str | None = None,
    ) -> int:
        """Archive memories with low outcome scores.

        Args:
            threshold: Score threshold (archive if below).
            project: Optional project filter.

        Returns:
            Number of archived memories.
        """
        conditions = ["archived = 0", "outcome_score < ?"]
        params: list[Any] = [threshold]

        if project is not None:
            conditions.append("project = ?")
            params.append(project)

        where_clause = " AND ".join(conditions)
        sql = f"""
            UPDATE memories
            SET archived = 1, updated_at = ?
            WHERE {where_clause}
        """
        params.insert(0, datetime.now(UTC).isoformat())

        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.rowcount

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self, project: str | None = None) -> StorageStats:
        """Get storage statistics.

        Args:
            project: Optional project filter.

        Returns:
            Storage statistics.
        """
        project_filter = "AND project = ?" if project else ""
        params: list[Any] = [project] if project else []

        async with self._get_connection() as conn:
            # Total and archived counts
            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM memories WHERE 1=1 {project_filter}",
                params,
            )
            row = await cursor.fetchone()
            total = row[0] if row else 0

            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM memories WHERE archived = 1 {project_filter}",
                params,
            )
            row = await cursor.fetchone()
            archived = row[0] if row else 0

            # By category
            cursor = await conn.execute(
                f"""
                SELECT category, COUNT(*) FROM memories
                WHERE archived = 0 {project_filter}
                GROUP BY category
                """,
                params,
            )
            by_category: dict[str, int] = {r[0]: r[1] for r in await cursor.fetchall()}

            # By scope
            cursor = await conn.execute(
                f"""
                SELECT scope, COUNT(*) FROM memories
                WHERE archived = 0 {project_filter}
                GROUP BY scope
                """,
                params,
            )
            by_scope: dict[str, int] = {r[0]: r[1] for r in await cursor.fetchall()}

            # By source
            cursor = await conn.execute(
                f"""
                SELECT source, COUNT(*) FROM memories
                WHERE archived = 0 {project_filter}
                GROUP BY source
                """,
                params,
            )
            by_source: dict[str, int] = {r[0]: r[1] for r in await cursor.fetchall()}

            # Average outcome score
            cursor = await conn.execute(
                f"""
                SELECT AVG(outcome_score) FROM memories
                WHERE archived = 0 {project_filter}
                """,
                params,
            )
            row = await cursor.fetchone()
            avg_score = row[0] if row and row[0] else 0.0

            # Total uses
            cursor = await conn.execute(
                f"""
                SELECT SUM(use_count) FROM memories
                WHERE 1=1 {project_filter}
                """,
                params,
            )
            row = await cursor.fetchone()
            total_uses = row[0] if row and row[0] else 0

        return StorageStats(
            total_memories=total,
            active_memories=total - archived,
            archived_memories=archived,
            by_category=by_category,
            by_scope=by_scope,
            by_source=by_source,
            avg_outcome_score=avg_score,
            total_uses=total_uses,
        )

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check storage health.

        Returns:
            Health status dictionary.
        """
        try:
            async with self._get_connection() as conn:
                # Check we can query
                cursor = await conn.execute("SELECT 1")
                await cursor.fetchone()

                # Check integrity
                cursor = await conn.execute("PRAGMA integrity_check")
                row = await cursor.fetchone()
                integrity = row[0] if row else "unknown"

                # Get database size
                cursor = await conn.execute("PRAGMA page_count")
                row = await cursor.fetchone()
                page_count = row[0] if row else 0
                cursor = await conn.execute("PRAGMA page_size")
                row = await cursor.fetchone()
                page_size = row[0] if row else 0
                db_size = page_count * page_size

            return {
                "status": "healthy" if integrity == "ok" else "degraded",
                "database": str(self.db_path),
                "size_bytes": db_size,
                "integrity": integrity,
                "pool_size": self.pool_size,
                "initialized": self._initialized,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized,
            }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_memory(self, row: aiosqlite.Row) -> Memory:
        """Convert a database row to a Memory object.

        Args:
            row: Database row.

        Returns:
            Memory instance.
        """
        return Memory(
            id=row["id"],
            content=row["content"],
            category=MemoryCategory(row["category"]),
            outcome_score=row["outcome_score"],
            worked=row["worked"] or 0.0,
            failed=row["failed"] or 0.0,
            evidence_at=datetime.fromisoformat(row["evidence_at"]) if row["evidence_at"] else None,
            confidence=row["confidence"],
            importance=row["importance"],
            use_count=row["use_count"],
            project=row["project"],
            scope=MemoryScope(row["scope"]),
            source=MemorySource(row["source"]),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            entities=json.loads(row["entities"]) if row["entities"] else [],
            supersedes=row["supersedes"],
            archived=bool(row["archived"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

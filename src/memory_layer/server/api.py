"""REST API Server for Memory Layer.

This module implements a FastAPI-based REST API that exposes memory operations
for HTTP-based access.

Usage:
    mem serve --rest --port 8080

    Or programmatically:
    from memory_layer.server.api import create_app
    app = create_app()
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_layer.core.engine import MemoryEngine
from memory_layer.core.logging import get_logger
from memory_layer.core.models import (
    Memory,
    MemoryCategory,
    MemoryScope,
    MemorySource,
    Outcome,
)

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class APIConfig:
    """API configuration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 100,
        rate_window: float = 60.0,
        max_request_size: int = 1_000_000,  # 1MB
        cors_origins: list[str] | None = None,
    ):
        self.api_key = api_key or os.environ.get("MEMORY_LAYER_API_KEY")
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.max_request_size = max_request_size
        self.cors_origins = cors_origins or ["*"]


# =============================================================================
# Request/Response Models
# =============================================================================


class MemoryCreateRequest(BaseModel):
    """Request model for creating a memory."""

    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(..., min_length=1, max_length=10000)
    category: MemoryCategory = MemoryCategory.GENERAL
    project: Optional[str] = Field(default=None, max_length=255)
    tags: list[str] = Field(default_factory=list, max_length=20)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()


class MemoryUpdateRequest(BaseModel):
    """Request model for updating a memory."""

    model_config = ConfigDict(str_strip_whitespace=True)

    content: Optional[str] = Field(default=None, min_length=1, max_length=10000)
    category: Optional[MemoryCategory] = None
    tags: Optional[list[str]] = Field(default=None, max_length=20)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    """Request model for searching memories."""

    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    categories: Optional[list[MemoryCategory]] = None
    project: Optional[str] = None
    min_score: float = Field(default=-1.0, ge=-1.0, le=1.0)
    search_type: str = Field(default="semantic", pattern="^(semantic|keyword)$")


class OutcomeRequest(BaseModel):
    """Request model for recording outcome."""

    memory_ids: list[str] = Field(..., min_length=1, max_length=50)
    outcome: Outcome


class IngestRequest(BaseModel):
    """Request model for ingesting transcript."""

    transcript: str = Field(..., min_length=1, max_length=100000)
    project: Optional[str] = None
    session_id: Optional[str] = None


class MemoryResponse(BaseModel):
    """Response model for a single memory."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    content: str
    category: MemoryCategory
    outcome_score: float
    confidence: float
    importance: float
    use_count: int
    project: Optional[str]
    scope: MemoryScope
    source: MemorySource
    tags: list[str]
    entities: list[str]
    supersedes: Optional[str]
    archived: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_memory(cls, memory: Memory) -> "MemoryResponse":
        return cls(
            id=memory.id,
            content=memory.content,
            category=memory.category,
            outcome_score=memory.outcome_score,
            confidence=memory.confidence,
            importance=memory.importance,
            use_count=memory.use_count,
            project=memory.project,
            scope=memory.scope,
            source=memory.source,
            tags=memory.tags,
            entities=memory.entities,
            supersedes=memory.supersedes,
            archived=memory.archived,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )


class SearchResultResponse(BaseModel):
    """Response model for a search result."""

    memory: MemoryResponse
    score: float
    semantic_score: float = 0.0
    recency_score: float = 0.0
    frequency_score: float = 0.0


class SearchResponse(BaseModel):
    """Response model for search results."""

    count: int
    results: list[SearchResultResponse]


class MemoryListResponse(BaseModel):
    """Response model for memory list."""

    count: int
    memories: list[MemoryResponse]


class ContextResponse(BaseModel):
    """Response model for context."""

    project: Optional[str]
    total_count: int
    included_count: int
    formatted: str
    memories: list[MemoryResponse]


class StatsResponse(BaseModel):
    """Response model for statistics."""

    total_memories: int
    active_memories: int
    archived_memories: int
    by_category: dict[str, int]
    by_scope: dict[str, int]
    by_source: dict[str, int]
    avg_outcome_score: float
    total_uses: int


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str
    status: str
    message: str = ""
    duration_ms: float = 0.0


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    timestamp: datetime
    checks: list[ComponentHealth] = Field(default_factory=list)


class ReadinessResponse(BaseModel):
    """Response model for readiness check."""

    ready: bool
    message: str = ""
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    detail: Optional[str] = None
    status_code: int


class OutcomeResponse(BaseModel):
    """Response model for outcome recording."""

    success: bool
    memory_ids: list[str]
    outcome: str
    adjustment: str


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        if len(self._requests[client_id]) >= self.max_requests:
            return False

        self._requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        window_start = now - self.window_seconds
        current = len([t for t in self._requests[client_id] if t > window_start])
        return max(0, self.max_requests - current)


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Application state container."""

    def __init__(self):
        self.engine: Optional[MemoryEngine] = None
        self.config: Optional[APIConfig] = None
        self.rate_limiter: Optional[RateLimiter] = None

    def get_engine(self) -> MemoryEngine:
        """Get or create engine."""
        if self.engine is None:
            from pathlib import Path

            db_path = os.environ.get(
                "MEMORY_LAYER_DB",
                str(Path.home() / ".memory-layer" / "memories.db"),
            )
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.engine = MemoryEngine(db_path=db_path)
        return self.engine


# Global state
_app_state = AppState()


def get_engine() -> MemoryEngine:
    """Dependency for getting the engine."""
    return _app_state.get_engine()


def get_config() -> APIConfig:
    """Dependency for getting the config."""
    if _app_state.config is None:
        _app_state.config = APIConfig()
    return _app_state.config


def get_rate_limiter() -> RateLimiter:
    """Dependency for getting the rate limiter."""
    if _app_state.rate_limiter is None:
        config = get_config()
        _app_state.rate_limiter = RateLimiter(
            max_requests=config.rate_limit,
            window_seconds=config.rate_window,
        )
    return _app_state.rate_limiter


# =============================================================================
# Authentication
# =============================================================================


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    config: APIConfig = Depends(get_config),
) -> Optional[str]:
    """Verify API key if configured."""
    if config.api_key is None:
        return None  # No auth required

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )

    # Constant-time comparison
    if not secrets.compare_digest(x_api_key, config.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key


# =============================================================================
# Middleware
# =============================================================================


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    rate_limiter = get_rate_limiter()
    config = get_config()

    # Get client identifier
    client_id = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "detail": f"Maximum {config.rate_limit} requests per {config.rate_window} seconds",
                "status_code": 429,
            },
        )

    response = await call_next(request)

    # Add rate limit headers
    remaining = rate_limiter.get_remaining(client_id)
    response.headers["X-RateLimit-Limit"] = str(config.rate_limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    return response


async def request_logging_middleware(request: Request, call_next):
    """Request logging middleware."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)"
    )

    return response


# =============================================================================
# Exception Handlers
# =============================================================================


async def validation_exception_handler(request: Request, exc: Exception):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "status_code": 422,
        },
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general errors (not HTTPException)."""
    # Let HTTPException be handled by FastAPI's default handler
    if isinstance(exc, HTTPException):
        raise exc

    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.environ.get("DEBUG") else None,
            "status_code": 500,
        },
    )


# =============================================================================
# Application Factory
# =============================================================================


def create_app(
    config: Optional[APIConfig] = None,
    engine: Optional[MemoryEngine] = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        config: Optional API configuration
        engine: Optional MemoryEngine instance

    Returns:
        Configured FastAPI application
    """
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # Set up state
    if config:
        _app_state.config = config
    if engine:
        _app_state.engine = engine

    config = config or get_config()

    # Create app
    app = FastAPI(
        title="Memory Layer API",
        description="REST API for Memory Layer - Persistent memory for AI coding agents",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(request_logging_middleware)

    # Add exception handlers
    app.add_exception_handler(ValueError, validation_exception_handler)
    # Note: Don't add general Exception handler as it catches HTTPException too

    # Register routes
    app.include_router(router)

    # Static files for Web UI
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            """Serve the Web UI index page."""
            return FileResponse(static_dir / "index.html")

    return app


# =============================================================================
# Routes
# =============================================================================

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    engine: MemoryEngine = Depends(get_engine),
):
    """Comprehensive health check endpoint.

    Returns overall health status and individual component checks.
    """
    checks = []
    overall_healthy = True

    # Check database connectivity
    try:
        start = time.time()
        stats = await engine.stats()
        duration_ms = (time.time() - start) * 1000
        checks.append(
            ComponentHealth(
                name="database",
                status="healthy",
                message=f"{stats.storage_stats.total_memories} memories",
                duration_ms=round(duration_ms, 2),
            )
        )
    except Exception as e:
        checks.append(
            ComponentHealth(
                name="database",
                status="unhealthy",
                message=str(e),
            )
        )
        overall_healthy = False

    # Check embedding model (if available)
    try:
        if engine._embedding_provider:
            start = time.time()
            # Quick embedding test
            await engine._embedding_provider.embed("test")
            duration_ms = (time.time() - start) * 1000
            checks.append(
                ComponentHealth(
                    name="embedding",
                    status="healthy",
                    message="Model loaded",
                    duration_ms=round(duration_ms, 2),
                )
            )
    except Exception as e:
        checks.append(
            ComponentHealth(
                name="embedding",
                status="degraded",
                message=str(e),
            )
        )

    return HealthResponse(
        status="healthy" if overall_healthy else "unhealthy",
        version="2.0.0",
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )


@router.get("/health/live", tags=["System"])
async def liveness_check():
    """Kubernetes liveness probe.

    Returns 200 if the service is running.
    """
    return {"alive": True}


@router.get("/health/ready", response_model=ReadinessResponse, tags=["System"])
async def readiness_check(
    engine: MemoryEngine = Depends(get_engine),
):
    """Kubernetes readiness probe.

    Returns 200 if the service is ready to handle requests.
    """
    checks = {}

    # Check database
    try:
        await engine.stats()
        checks["database"] = True
    except Exception:
        checks["database"] = False

    ready = all(checks.values())

    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    return ReadinessResponse(
        ready=ready,
        message="All systems operational" if ready else "Some systems unavailable",
        checks=checks,
    )


@router.get("/metrics", tags=["System"])
async def get_metrics():
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    from memory_layer.core.observability import get_metrics_collector

    collector = get_metrics_collector()
    metrics_text = collector.to_prometheus()

    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(
        content=metrics_text,
        media_type="text/plain; charset=utf-8",
    )


@router.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats(
    project: Optional[str] = Query(None),
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get memory statistics."""
    stats = await engine.stats(project=project)
    # StatsResponse fields come from storage_stats
    storage = stats.storage_stats
    return StatsResponse(
        total_memories=storage.total_memories,
        active_memories=storage.active_memories,
        archived_memories=storage.archived_memories,
        by_category=storage.by_category,
        by_scope=storage.by_scope,
        by_source=storage.by_source,
        avg_outcome_score=storage.avg_outcome_score,
        total_uses=storage.total_uses,
    )


# -----------------------------------------------------------------------------
# Memory CRUD
# -----------------------------------------------------------------------------


@router.post(
    "/memories",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Memories"],
)
async def create_memory(
    request: MemoryCreateRequest,
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Create a new memory."""
    memory = await engine.add(
        content=request.content,
        category=request.category,
        project=request.project,
        tags=request.tags,
        importance=request.importance,
        entities=request.entities,
        source=MemorySource.EXPLICIT,
    )
    return MemoryResponse.from_memory(memory)


@router.get("/memories/{memory_id}", response_model=MemoryResponse, tags=["Memories"])
async def get_memory(
    memory_id: str,
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get a memory by ID."""
    from memory_layer.core.engine import MemoryNotFoundError

    try:
        memory = await engine.get(memory_id)
    except MemoryNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory not found: {memory_id}",
        )
    return MemoryResponse.from_memory(memory)


@router.patch("/memories/{memory_id}", response_model=MemoryResponse, tags=["Memories"])
async def update_memory(
    memory_id: str,
    request: MemoryUpdateRequest,
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Update a memory."""
    from memory_layer.core.engine import MemoryNotFoundError

    try:
        # Update memory with provided fields
        updated = await engine.update(
            memory_id=memory_id,
            content=request.content,
            category=request.category,
            tags=request.tags,
            importance=request.importance,
        )
    except MemoryNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory not found: {memory_id}",
        )
    return MemoryResponse.from_memory(updated)


@router.delete(
    "/memories/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Memories"],
)
async def delete_memory(
    memory_id: str,
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Archive (soft delete) a memory."""
    from memory_layer.core.engine import MemoryNotFoundError

    try:
        await engine.delete(memory_id)
    except MemoryNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory not found: {memory_id}",
        )
    return None


@router.get("/memories", response_model=MemoryListResponse, tags=["Memories"])
async def list_memories(
    category: Optional[MemoryCategory] = Query(None),
    project: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    include_archived: bool = Query(False),
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """List memories with optional filters."""
    memories = await engine.list(
        category=category,
        project=project,
        limit=limit,
        include_archived=include_archived,
    )
    return MemoryListResponse(
        count=len(memories),
        memories=[MemoryResponse.from_memory(m) for m in memories],
    )


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------


@router.post("/memories/search", response_model=SearchResponse, tags=["Search"])
async def search_memories(
    request: SearchRequest,
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Search memories by query.

    Supports two search types:
    - semantic: Vector similarity search (finds conceptually related memories)
    - keyword: Full-text search (exact keyword matching)
    """
    if request.search_type == "keyword":
        # Use FTS for keyword search
        memories = await engine._storage.search_fts(
            query=request.query,
            project=request.project,
            limit=request.limit,
        )
        # Filter by category if specified
        if request.categories:
            memories = [m for m in memories if m.category in request.categories]

        return SearchResponse(
            count=len(memories),
            results=[
                SearchResultResponse(
                    memory=MemoryResponse.from_memory(m),
                    score=1.0,  # FTS doesn't provide similarity scores
                    semantic_score=0.0,
                    recency_score=0.0,
                    frequency_score=0.0,
                )
                for m in memories
            ],
        )

    # Semantic search (default)
    # Engine only supports single category, use first if list provided
    category = request.categories[0] if request.categories else None
    results = await engine.search(
        query=request.query,
        limit=request.limit,
        category=category,
        project=request.project,
        min_score=request.min_score,
    )
    return SearchResponse(
        count=len(results),
        results=[
            SearchResultResponse(
                memory=MemoryResponse.from_memory(r.memory),
                score=r.score,
                semantic_score=r.semantic_score,
                recency_score=r.recency_score,
                frequency_score=r.frequency_score,
            )
            for r in results
        ],
    )


# -----------------------------------------------------------------------------
# Outcome
# -----------------------------------------------------------------------------


@router.post("/memories/outcome", response_model=OutcomeResponse, tags=["Feedback"])
async def record_outcome(
    request: OutcomeRequest,
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Record outcome feedback for memories."""
    success = await engine.record_outcome(
        memory_ids=request.memory_ids,
        outcome=request.outcome,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more memories not found",
        )

    adjustment = {
        Outcome.WORKED: "+0.2",
        Outcome.FAILED: "-0.3",
        Outcome.PARTIAL: "+0.05",
    }[request.outcome]

    return OutcomeResponse(
        success=True,
        memory_ids=request.memory_ids,
        outcome=request.outcome.value,
        adjustment=adjustment,
    )


# -----------------------------------------------------------------------------
# Context
# -----------------------------------------------------------------------------


@router.get("/context", response_model=ContextResponse, tags=["Context"])
async def get_context(
    project: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    format: str = Query("markdown"),
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get project context."""
    context = await engine.get_context(project=project, max_memories=limit)

    from memory_layer.plugin import ContextFormatter

    formatted = ContextFormatter.format_for_injection(
        context.memories,
        style=format if format in ["brief", "detailed", "markdown"] else "markdown",
    )

    return ContextResponse(
        project=project,
        total_count=context.total_count,
        included_count=context.included_count,
        formatted=formatted,
        memories=[MemoryResponse.from_memory(m) for m in context.memories],
    )


# -----------------------------------------------------------------------------
# Sessions
# -----------------------------------------------------------------------------


@router.post("/sessions/ingest", tags=["Sessions"])
async def ingest_transcript(
    request: IngestRequest,
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Ingest a conversation transcript to extract memories."""
    # TODO: Implement full extraction when extractor integration is ready
    return {
        "status": "accepted",
        "message": "Transcript received for processing",
        "transcript_length": len(request.transcript),
        "project": request.project,
        "session_id": request.session_id,
    }


# -----------------------------------------------------------------------------
# Beads Integration
# -----------------------------------------------------------------------------


class BeadsSyncRequest(BaseModel):
    """Request model for Beads sync."""

    task_id: Optional[str] = Field(default=None, description="Sync specific task only")


class BeadsSyncResponse(BaseModel):
    """Response model for Beads sync."""

    success: bool
    tasks_found: int = 0
    tasks_synced: int = 0
    outcomes_recorded: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BeadsContextResponse(BaseModel):
    """Response model for Beads context."""

    success: bool
    task_id: Optional[str] = None
    task_title: Optional[str] = None
    task_status: Optional[str] = None
    task_description: Optional[str] = None
    memories_count: int = 0
    formatted: str = ""
    error: Optional[str] = None


class BeadsLinkRequest(BaseModel):
    """Request model for linking memory to task."""

    memory_id: str = Field(..., min_length=1)
    task_id: Optional[str] = Field(default=None, description="Task to link to (uses current if not provided)")
    context: Optional[str] = Field(default=None, description="Context about how memory is used")


class BeadsLinkResponse(BaseModel):
    """Response model for link operation."""

    success: bool
    memory_id: Optional[str] = None
    task_id: Optional[str] = None
    error: Optional[str] = None


class BeadsTaskResponse(BaseModel):
    """Response model for a single task."""

    id: str
    title: str
    status: str
    description: str = ""
    is_ready: bool = False
    is_completed: bool = False


class BeadsTasksResponse(BaseModel):
    """Response model for task list."""

    success: bool
    count: int = 0
    tasks: list[BeadsTaskResponse] = Field(default_factory=list)
    error: Optional[str] = None


class BeadsStatsResponse(BaseModel):
    """Response model for Beads stats."""

    beads_available: bool
    beads_dir: Optional[str] = None
    tasks: dict = Field(default_factory=dict)
    links: dict = Field(default_factory=dict)
    auto_outcome_enabled: bool = True
    outcome_on_cancel: bool = False


# Beads adapter singleton
_beads_adapter = None


async def get_beads_adapter(engine: MemoryEngine = Depends(get_engine)):
    """Get or create the Beads adapter."""
    global _beads_adapter
    if _beads_adapter is None:
        from memory_layer.tasks import BeadsAdapter
        _beads_adapter = BeadsAdapter(engine)
    if not _beads_adapter._initialized:
        await _beads_adapter.initialize()
    return _beads_adapter


@router.post("/beads/sync", response_model=BeadsSyncResponse, tags=["Beads"])
async def beads_sync(
    request: BeadsSyncRequest = None,
    adapter = Depends(get_beads_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Sync outcomes for completed Beads tasks.

    When a task completes, memories that helped solve it get their outcome scores boosted.
    """
    if not adapter.is_available:
        return BeadsSyncResponse(
            success=False,
            errors=["Beads not available (no .beads/ directory found)"],
        )

    if request and request.task_id:
        # Sync specific task
        from memory_layer.tasks import BeadsTaskStatus
        task = adapter.get_task(request.task_id)
        if not task:
            return BeadsSyncResponse(success=False, errors=[f"Task {request.task_id} not found"])

        if task.status == BeadsTaskStatus.DONE:
            count = await adapter.on_task_done(request.task_id)
        elif task.status == BeadsTaskStatus.CANCELLED:
            count = await adapter.on_task_cancelled(request.task_id)
        elif task.status == BeadsTaskStatus.BLOCKED:
            count = await adapter.on_task_blocked(request.task_id)
        else:
            count = 0

        return BeadsSyncResponse(
            success=True,
            tasks_found=1,
            tasks_synced=1 if count > 0 else 0,
            outcomes_recorded=count,
        )
    else:
        # Sync all
        result = await adapter.sync()
        return BeadsSyncResponse(
            success=result.success,
            tasks_found=result.tasks_found,
            tasks_synced=result.tasks_synced,
            outcomes_recorded=result.outcomes_recorded,
            errors=result.errors,
            warnings=result.warnings,
        )


@router.get("/beads/context", response_model=BeadsContextResponse, tags=["Beads"])
async def beads_context(
    task_id: Optional[str] = Query(None, description="Task ID (uses current if not provided)"),
    limit: int = Query(10, ge=1, le=50, description="Max memories to include"),
    adapter = Depends(get_beads_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get unified context combining task info and relevant memories."""
    if not adapter.is_available:
        return BeadsContextResponse(
            success=False,
            error="Beads not available (no .beads/ directory found)",
        )

    context = await adapter.get_unified_context(task_id, limit)

    if not context:
        return BeadsContextResponse(
            success=False,
            error="No task found",
        )

    return BeadsContextResponse(
        success=True,
        task_id=context.task.id,
        task_title=context.task.title,
        task_status=context.task.status.value,
        task_description=context.task.description,
        memories_count=len(context.memories),
        formatted=context.formatted,
    )


@router.post("/beads/link", response_model=BeadsLinkResponse, tags=["Beads"])
async def beads_link(
    request: BeadsLinkRequest,
    adapter = Depends(get_beads_adapter),
    engine: MemoryEngine = Depends(get_engine),
    _: Optional[str] = Depends(verify_api_key),
):
    """Link a memory to a Beads task for outcome tracking."""
    if not adapter.is_available:
        return BeadsLinkResponse(
            success=False,
            error="Beads not available (no .beads/ directory found)",
        )

    # Get task ID
    task_id = request.task_id
    if task_id:
        task = adapter.get_task(task_id)
        if not task:
            return BeadsLinkResponse(success=False, error=f"Task {task_id} not found")
    else:
        task = adapter.get_current_task()
        if not task:
            return BeadsLinkResponse(success=False, error="No current task found")
        task_id = task.id

    # Verify memory exists
    try:
        await engine.get(request.memory_id)
    except Exception:
        return BeadsLinkResponse(success=False, error=f"Memory {request.memory_id} not found")

    # Create link
    await adapter.link_memory_to_task(task_id, request.memory_id, request.context)

    return BeadsLinkResponse(
        success=True,
        memory_id=request.memory_id,
        task_id=task_id,
    )


@router.get("/beads/tasks", response_model=BeadsTasksResponse, tags=["Beads"])
async def beads_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Max tasks to return"),
    adapter = Depends(get_beads_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """List Beads tasks with optional status filter."""
    if not adapter.is_available:
        return BeadsTasksResponse(
            success=False,
            error="Beads not available (no .beads/ directory found)",
        )

    # Parse status filter
    status_enum = None
    if status:
        from memory_layer.tasks import BeadsTaskStatus
        try:
            status_enum = BeadsTaskStatus(status)
        except ValueError:
            return BeadsTasksResponse(success=False, error=f"Invalid status: {status}")

    tasks = adapter.list_tasks(status=status_enum)[:limit]

    return BeadsTasksResponse(
        success=True,
        count=len(tasks),
        tasks=[
            BeadsTaskResponse(
                id=t.id,
                title=t.title,
                status=t.status.value,
                description=t.description[:100] if t.description else "",
                is_ready=t.is_ready,
                is_completed=t.is_completed,
            )
            for t in tasks
        ],
    )


@router.get("/beads/tasks/{task_id}", response_model=BeadsTaskResponse, tags=["Beads"])
async def beads_task_detail(
    task_id: str,
    adapter = Depends(get_beads_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get details for a specific task."""
    if not adapter.is_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Beads not available",
        )

    task = adapter.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    return BeadsTaskResponse(
        id=task.id,
        title=task.title,
        status=task.status.value,
        description=task.description,
        is_ready=task.is_ready,
        is_completed=task.is_completed,
    )


@router.get("/beads/tasks/{task_id}/memories", tags=["Beads"])
async def beads_task_memories(
    task_id: str,
    adapter = Depends(get_beads_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get memories linked to a task."""
    if not adapter.is_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Beads not available",
        )

    task = adapter.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    memories = await adapter.get_task_memories(task_id)

    return {
        "task_id": task_id,
        "count": len(memories),
        "memories": [
            {
                "id": m.id,
                "content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
                "category": m.category.value,
                "outcome_score": m.outcome_score,
            }
            for m in memories
        ],
    }


@router.get("/beads/stats", response_model=BeadsStatsResponse, tags=["Beads"])
async def beads_stats(
    adapter = Depends(get_beads_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get Beads integration statistics."""
    stats = await adapter.get_stats()
    return BeadsStatsResponse(**stats)


# -----------------------------------------------------------------------------
# Unified Tasks Integration (Phase 7 - Claude Code Tasks Adapter)
# -----------------------------------------------------------------------------


class TaskResponse(BaseModel):
    """Response model for a unified task."""

    id: str
    title: str
    status: str
    description: str = ""
    source: str  # "beads" or "claude_code"
    is_ready: bool = False
    is_completed: bool = False


class TasksListResponse(BaseModel):
    """Response model for unified task list."""

    success: bool
    count: int = 0
    tasks: list[TaskResponse] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class TasksSyncRequest(BaseModel):
    """Request model for unified tasks sync."""

    source: Optional[str] = Field(
        default=None,
        description="Source to sync: 'beads', 'claude_code', or None for all",
    )
    task_id: Optional[str] = Field(default=None, description="Sync specific task only")


class TasksSyncResponse(BaseModel):
    """Response model for unified tasks sync."""

    success: bool
    total_tasks_found: int = 0
    total_tasks_synced: int = 0
    total_outcomes_recorded: int = 0
    results: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class TasksContextResponse(BaseModel):
    """Response model for unified tasks context."""

    success: bool
    task_id: Optional[str] = None
    task_title: Optional[str] = None
    task_status: Optional[str] = None
    task_description: Optional[str] = None
    source: Optional[str] = None
    memories_count: int = 0
    formatted: str = ""
    error: Optional[str] = None


class TasksStatsResponse(BaseModel):
    """Response model for unified tasks stats."""

    available_sources: list[str] = Field(default_factory=list)
    beads: dict = Field(default_factory=dict)
    claude_code: dict = Field(default_factory=dict)


# Unified adapter singleton
_unified_adapter = None


async def get_unified_adapter(engine: MemoryEngine = Depends(get_engine)):
    """Get or create the unified task adapter."""
    global _unified_adapter
    if _unified_adapter is None:
        from memory_layer.tasks import UnifiedTaskAdapter
        _unified_adapter = UnifiedTaskAdapter(engine)
    if not _unified_adapter._initialized:
        await _unified_adapter.initialize()
    return _unified_adapter


@router.get("/tasks", response_model=TasksListResponse, tags=["Tasks"])
async def list_tasks(
    source: Optional[str] = Query(None, description="Filter by source: beads, claude_code"),
    task_status: Optional[str] = Query(None, description="Filter by status: pending, in_progress, done"),
    limit: int = Query(50, ge=1, le=200, description="Max tasks to return"),
    adapter = Depends(get_unified_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """List tasks from all available sources.

    Combines tasks from Beads (.beads/) and Claude Code (~/.claude/todos/).
    """
    from memory_layer.tasks import TaskSource

    # Parse source filter
    source_filter = None
    if source:
        try:
            source_filter = TaskSource(source)
        except ValueError:
            return TasksListResponse(
                success=False,
                error=f"Invalid source: {source}. Use 'beads' or 'claude_code'",
            )

    tasks = adapter.list_tasks(source=source_filter, status=task_status)[:limit]

    return TasksListResponse(
        success=True,
        count=len(tasks),
        tasks=[
            TaskResponse(
                id=t.id,
                title=t.title,
                status=t.status,
                description=t.description[:100] if t.description else "",
                source=t.source.value,
                is_ready=t.is_ready,
                is_completed=t.is_completed,
            )
            for t in tasks
        ],
        sources=[s.value for s in adapter.available_sources],
    )


@router.get("/tasks/stats", response_model=TasksStatsResponse, tags=["Tasks"])
async def get_tasks_stats(
    adapter = Depends(get_unified_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get statistics for all task integrations."""
    stats = await adapter.get_stats()
    return TasksStatsResponse(
        available_sources=stats.get("available_sources", []),
        beads=stats.get("beads", {}),
        claude_code=stats.get("claude_code", {}),
    )


@router.get("/tasks/context", response_model=TasksContextResponse, tags=["Tasks"])
async def get_tasks_context(
    task_id: Optional[str] = Query(None, description="Task ID (uses current if not provided)"),
    source: Optional[str] = Query(None, description="Source: beads, claude_code"),
    limit: int = Query(10, ge=1, le=50, description="Max memories to include"),
    adapter = Depends(get_unified_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get unified context combining task info and relevant memories."""
    from memory_layer.tasks import TaskSource

    # Parse source filter
    source_filter = None
    if source:
        try:
            source_filter = TaskSource(source)
        except ValueError:
            return TasksContextResponse(
                success=False,
                error=f"Invalid source: {source}",
            )

    context = await adapter.get_unified_context(task_id, source_filter, limit)

    if not context:
        return TasksContextResponse(
            success=False,
            error="No task found",
        )

    return TasksContextResponse(
        success=True,
        task_id=context.task.id,
        task_title=context.task.title,
        task_status=context.task.status.value,
        task_description=context.task.description if hasattr(context.task, 'description') else context.task.content,
        source=context.source.value,
        memories_count=len(context.memories),
        formatted=context.formatted,
    )


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task(
    task_id: str,
    adapter = Depends(get_unified_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get a specific task by ID from any source."""
    task = adapter.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    return TaskResponse(
        id=task.id,
        title=task.title,
        status=task.status,
        description=task.description,
        source=task.source.value,
        is_ready=task.is_ready,
        is_completed=task.is_completed,
    )


@router.post("/tasks/sync", response_model=TasksSyncResponse, tags=["Tasks"])
async def sync_tasks(
    request: TasksSyncRequest = None,
    adapter = Depends(get_unified_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Sync outcomes for completed tasks from all sources.

    When a task completes, memories that helped solve it get their outcome scores boosted.
    """
    from memory_layer.tasks import TaskSource

    if request and request.task_id:
        # Sync specific task
        count = await adapter.on_task_completed(request.task_id)
        return TasksSyncResponse(
            success=True,
            total_tasks_found=1,
            total_tasks_synced=1 if count > 0 else 0,
            total_outcomes_recorded=count,
        )

    # Parse source filter
    source_filter = None
    if request and request.source:
        try:
            source_filter = TaskSource(request.source)
        except ValueError:
            return TasksSyncResponse(
                success=False,
                errors=[f"Invalid source: {request.source}"],
            )

    # Sync all or specific source
    result = await adapter.sync(source=source_filter)

    if hasattr(result, 'results'):
        # UnifiedSyncResult
        return TasksSyncResponse(
            success=result.success,
            total_tasks_found=result.total_tasks_found,
            total_tasks_synced=result.total_tasks_synced,
            total_outcomes_recorded=result.total_outcomes_recorded,
            results={k.value: v.to_dict() for k, v in result.results.items()},
            errors=result.errors,
        )
    else:
        # Single TaskSyncResult
        return TasksSyncResponse(
            success=result.success,
            total_tasks_found=result.tasks_found,
            total_tasks_synced=result.tasks_synced,
            total_outcomes_recorded=result.outcomes_recorded,
            results={result.source.value: result.to_dict()},
            errors=result.errors,
        )


@router.get("/tasks/{task_id}/memories", tags=["Tasks"])
async def get_task_memories(
    task_id: str,
    adapter = Depends(get_unified_adapter),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get memories linked to a task."""
    task = adapter.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    memories = await adapter.get_task_memories(task_id, source=task.source)

    return {
        "success": True,
        "task_id": task_id,
        "source": task.source.value,
        "count": len(memories),
        "memories": [
            {
                "id": m.id,
                "content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
                "category": m.category.value,
                "outcome_score": m.outcome_score,
            }
            for m in memories
        ],
    }


# =============================================================================
# Server Runner
# =============================================================================


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    config: Optional[APIConfig] = None,
    engine: Optional[MemoryEngine] = None,
):
    """Run the REST API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        config: Optional API configuration
        engine: Optional MemoryEngine instance
    """
    import signal
    import uvicorn

    app = create_app(config=config, engine=engine)

    # Configure for graceful shutdown on single Ctrl+C
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",  # Reduce noise
        timeout_graceful_shutdown=5,  # Clean shutdown timeout
    )
    server = uvicorn.Server(uvicorn_config)

    # Handle SIGINT (Ctrl+C) gracefully
    def handle_sigint(sig, frame):
        server.should_exit = True

    signal.signal(signal.SIGINT, handle_sigint)

    server.run()


if __name__ == "__main__":
    run_server()

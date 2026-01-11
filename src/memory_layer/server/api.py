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


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    timestamp: datetime


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

    return app


# =============================================================================
# Routes
# =============================================================================

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now(timezone.utc),
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
    """Search memories by query."""
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
    import uvicorn

    app = create_app(config=config, engine=engine)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

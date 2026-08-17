from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import cast
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response

from .db import Database
from .engine import TermyteDB
from .errors import IdempotencyConflict
from .integrity import check_database
from .provider import ExtractionProvider
from .schemas import (
    BatchEventRequest,
    BatchEventResponse,
    ContextRequest,
    ContextResponse,
    EpisodeStatusRequest,
    EventInput,
    EventReceipt,
    FeedbackRequest,
    FeedbackResponse,
    MemoryHistoryResponse,
    MemoryResponse,
    ProcessRequest,
    ProcessResponse,
    SearchRequest,
    SearchResult,
)


def create_app(
    database_path: str | Path | None = None,
    *,
    database: Database | None = None,
    extraction_provider: ExtractionProvider | None = None,
    namespace_authorizer: Callable[[str], bool] | None = None,
    rate_limit_per_minute: int | None = None,
) -> FastAPI:
    if database is None and database_path is None:
        raise ValueError("create_app requires an explicit database path or database instance")
    engine = TermyteDB(database_path, database=database, extraction_provider=extraction_provider)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title="TermyteDB", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    request_windows: dict[str, deque[float]] = defaultdict(deque)
    request_windows_lock = Lock()

    def require_namespace(namespace_id: str) -> None:
        if namespace_authorizer is not None and not namespace_authorizer(namespace_id):
            raise HTTPException(status_code=403, detail="namespace access denied")

    def enforce_rate_limit(namespace_id: str) -> None:
        if rate_limit_per_minute is None:
            return
        if rate_limit_per_minute < 1:
            raise ValueError("rate_limit_per_minute must be positive")
        now = monotonic()
        with request_windows_lock:
            window = request_windows[namespace_id]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= rate_limit_per_minute:
                raise HTTPException(status_code=429, detail="namespace rate limit exceeded", headers={"retry-after": "60"})
            window.append(now)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: object) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response = cast(Response, await call_next(request))  # type: ignore[operator]
        response.headers["x-request-id"] = request_id
        return response

    @app.post("/v1/events")
    def ingest(event: EventInput) -> EventReceipt:
        require_namespace(event.namespace_id)
        enforce_rate_limit(event.namespace_id)
        try:
            return engine.ingest(event)
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/events:batch", response_model=BatchEventResponse)
    def ingest_batch(request: BatchEventRequest) -> BatchEventResponse:
        for event in request.events:
            require_namespace(event.namespace_id)
            enforce_rate_limit(event.namespace_id)
        try:
            return engine.ingest_batch(request.events)
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/process")
    def process(request: ProcessRequest) -> ProcessResponse:
        require_namespace(request.namespace_id)
        enforce_rate_limit(request.namespace_id)
        return engine.process_with_timeout(request.namespace_id, request.limit, request.lease_seconds, request.timeout_seconds)

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, namespace_id: str = Query(...)) -> dict[str, bool]:
        require_namespace(namespace_id)
        if not engine.cancel_job(namespace_id, job_id):
            raise HTTPException(status_code=404, detail="job not found")
        return {"cancelled": True}

    @app.get("/v1/jobs")
    def jobs(namespace_id: str = Query(...), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[dict[str, object]]:
        require_namespace(namespace_id)
        return engine.jobs(namespace_id, limit, offset)

    @app.get("/v1/events", response_model=list[dict[str, object]])
    def events(namespace_id: str = Query(...), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[dict[str, object]]:
        require_namespace(namespace_id)
        return engine.events(namespace_id, limit, offset)

    @app.get("/v1/events/{event_id}")
    def get_event(event_id: str, namespace_id: str = Query(...)) -> dict[str, object]:
        require_namespace(namespace_id)
        event = engine.event(namespace_id, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return event

    @app.post("/v1/search")
    def search(request: SearchRequest) -> list[SearchResult]:
        require_namespace(request.namespace_id)
        enforce_rate_limit(request.namespace_id)
        return engine.search(request.namespace_id, request.query, request.limit, request.historical)

    @app.post("/v1/context")
    def context(request: ContextRequest) -> ContextResponse:
        require_namespace(request.namespace_id)
        enforce_rate_limit(request.namespace_id)
        return engine.context(request.namespace_id, request.query, request.token_budget, request.limit, request.historical)

    @app.get("/v1/context/requests")
    def context_requests(namespace_id: str = Query(...), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[dict[str, object]]:
        require_namespace(namespace_id)
        return engine.context_requests(namespace_id, limit, offset)

    @app.get("/v1/extraction/runs")
    def extraction_runs(namespace_id: str = Query(...), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[dict[str, object]]:
        require_namespace(namespace_id)
        return engine.extraction_runs(namespace_id, limit, offset)

    @app.get("/v1/extraction/decisions")
    def extraction_decisions(namespace_id: str = Query(...), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[dict[str, object]]:
        require_namespace(namespace_id)
        return engine.extraction_decisions(namespace_id, limit, offset)

    @app.get("/v1/memories", response_model=list[MemoryResponse])
    def memories(namespace_id: str = Query(...), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[MemoryResponse]:
        require_namespace(namespace_id)
        return engine.memories(namespace_id, limit, offset)

    @app.get("/v1/memories/{memory_id}")
    def get_memory(memory_id: str, namespace_id: str) -> MemoryResponse:
        require_namespace(namespace_id)
        memory = engine.get_memory(namespace_id, memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return memory

    @app.get("/v1/memories/{memory_id}/history", response_model=MemoryHistoryResponse)
    def history(memory_id: str, namespace_id: str = Query(...)) -> MemoryHistoryResponse:
        require_namespace(namespace_id)
        versions = engine.history(namespace_id, memory_id)
        if versions is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return MemoryHistoryResponse(memory_id=UUID(memory_id), versions=versions)

    @app.post("/v1/memories/{memory_id}/invalidate")
    def invalidate(memory_id: str, namespace_id: str, reason: str = Query(..., min_length=1)) -> dict[str, bool]:
        require_namespace(namespace_id)
        if not engine.invalidate(namespace_id, memory_id, reason):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"invalidated": True}

    @app.get("/v1/export")
    def export(namespace_id: str = Query(...)) -> dict[str, object]:
        require_namespace(namespace_id)
        return engine.export_namespace(namespace_id)

    @app.post("/v1/import")
    def import_namespace(document: dict[str, object], namespace_id: str = Query(...)) -> dict[str, int]:
        require_namespace(namespace_id)
        try:
            return engine.import_namespace(document, namespace_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/episodes")
    def episodes(namespace_id: str = Query(...), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[dict[str, object]]:
        require_namespace(namespace_id)
        return engine.episodes(namespace_id, limit, offset)

    @app.patch("/v1/episodes/{episode_id}")
    def update_episode(episode_id: str, request: EpisodeStatusRequest) -> dict[str, bool]:
        require_namespace(request.namespace_id)
        if not engine.update_episode(request.namespace_id, episode_id, request.status, request.summary):
            raise HTTPException(status_code=404, detail="episode not found")
        return {"updated": True}

    @app.get("/v1/feedback")
    def feedback_rows(namespace_id: str = Query(...), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[dict[str, object]]:
        require_namespace(namespace_id)
        return engine.feedback_rows(namespace_id, limit, offset)

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        require_namespace(request.namespace_id)
        try:
            feedback_id = engine.feedback(request.namespace_id, str(request.memory_id), request.label, request.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="memory not found") from exc
        return FeedbackResponse(id=UUID(feedback_id), namespace_id=request.namespace_id, memory_id=request.memory_id, label=request.label)

    @app.delete("/v1/namespaces/{namespace_id}")
    def delete_namespace(namespace_id: str) -> dict[str, bool]:
        require_namespace(namespace_id)
        if not engine.delete_namespace(namespace_id):
            raise HTTPException(status_code=404, detail="namespace not found")
        return {"deleted": True}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        report = check_database(engine.database)
        if not report.ok:
            raise HTTPException(status_code=503, detail="database is not ready")
        return {"status": "ready"}

    @app.get("/v1/integrity")
    def integrity() -> dict[str, object]:
        report = check_database(engine.database)
        return {"ok": report.ok, **report.__dict__}

    @app.get("/v1/metrics")
    def metrics(namespace_id: str = Query(...)) -> dict[str, float | int]:
        require_namespace(namespace_id)
        return engine.metrics(namespace_id)

    return app

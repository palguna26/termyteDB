from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
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

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: object) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response = cast(Response, await call_next(request))  # type: ignore[operator]
        response.headers["x-request-id"] = request_id
        return response

    @app.post("/v1/events")
    def ingest(event: EventInput) -> EventReceipt:
        try:
            return engine.ingest(event)
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/events:batch", response_model=BatchEventResponse)
    def ingest_batch(request: BatchEventRequest) -> BatchEventResponse:
        try:
            return engine.ingest_batch(request.events)
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/process")
    def process(request: ProcessRequest) -> ProcessResponse:
        return engine.process(request.namespace_id, request.limit, request.lease_seconds)

    @app.get("/v1/jobs")
    def jobs(namespace_id: str = Query(...)) -> list[dict[str, object]]:
        return engine.jobs(namespace_id)

    @app.get("/v1/events/{event_id}")
    def get_event(event_id: str, namespace_id: str = Query(...)) -> dict[str, object]:
        event = engine.event(namespace_id, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return event

    @app.post("/v1/search")
    def search(request: SearchRequest) -> list[SearchResult]:
        return engine.search(request.namespace_id, request.query, request.limit)

    @app.post("/v1/context")
    def context(request: ContextRequest) -> ContextResponse:
        return engine.context(request.namespace_id, request.query, request.token_budget, request.limit)

    @app.get("/v1/memories/{memory_id}")
    def get_memory(memory_id: str, namespace_id: str) -> MemoryResponse:
        memory = engine.get_memory(namespace_id, memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return memory

    @app.get("/v1/memories/{memory_id}/history", response_model=MemoryHistoryResponse)
    def history(memory_id: str, namespace_id: str = Query(...)) -> MemoryHistoryResponse:
        versions = engine.history(namespace_id, memory_id)
        if versions is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return MemoryHistoryResponse(memory_id=UUID(memory_id), versions=versions)

    @app.post("/v1/memories/{memory_id}/invalidate")
    def invalidate(memory_id: str, namespace_id: str, reason: str = Query(..., min_length=1)) -> dict[str, bool]:
        if not engine.invalidate(namespace_id, memory_id, reason):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"invalidated": True}

    @app.get("/v1/export")
    def export(namespace_id: str = Query(...)) -> dict[str, object]:
        return engine.export_namespace(namespace_id)

    @app.post("/v1/import")
    def import_namespace(document: dict[str, object], namespace_id: str = Query(...)) -> dict[str, int]:
        try:
            return engine.import_namespace(document, namespace_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/episodes")
    def episodes(namespace_id: str = Query(...)) -> list[dict[str, object]]:
        return engine.episodes(namespace_id)

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        try:
            feedback_id = engine.feedback(request.namespace_id, str(request.memory_id), request.label, request.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="memory not found") from exc
        return FeedbackResponse(id=UUID(feedback_id), namespace_id=request.namespace_id, memory_id=request.memory_id, label=request.label)

    @app.delete("/v1/namespaces/{namespace_id}")
    def delete_namespace(namespace_id: str) -> dict[str, bool]:
        if not engine.delete_namespace(namespace_id):
            raise HTTPException(status_code=404, detail="namespace not found")
        return {"deleted": True}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/v1/integrity")
    def integrity() -> dict[str, object]:
        report = check_database(engine.database)
        return {"ok": report.ok, **report.__dict__}

    return app

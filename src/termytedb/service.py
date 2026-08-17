from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .db import Database
from .engine import TermyteDB
from .errors import IdempotencyConflict
from .provider import ExtractionProvider
from .schemas import (
    ContextRequest,
    ContextResponse,
    EventInput,
    EventReceipt,
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

    @app.post("/v1/events")
    def ingest(event: EventInput) -> EventReceipt:
        try:
            return engine.ingest(event)
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/process")
    def process(request: ProcessRequest) -> ProcessResponse:
        return engine.process(request.namespace_id, request.limit, request.lease_seconds)

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

    return app

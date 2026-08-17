from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .engine import TermyteDB
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


def create_app(database_path: str = "termytedb.sqlite") -> FastAPI:
    app = FastAPI(title="TermyteDB", version="0.1.0")
    engine = TermyteDB(database_path)
    app.state.engine = engine

    @app.post("/v1/events")
    def ingest(event: EventInput) -> EventReceipt:
        return engine.ingest(event)

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


app = create_app()

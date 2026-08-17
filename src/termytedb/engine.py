from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from .context import build_context
from .db import Database
from .logging import get_logger, log
from .processor import Processor
from .provider import ExtractionProvider
from .redaction import redact
from .repository import Repository
from .schemas import (
    BatchEventResponse,
    ContextResponse,
    EventInput,
    EventReceipt,
    MemoryResponse,
    ProcessResponse,
    SearchResult,
)


class TermyteDB:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        database: Database | None = None,
        logger: logging.Logger | None = None,
        extraction_provider: ExtractionProvider | None = None,
    ):
        if database is None and path is None:
            raise ValueError("an explicit database path or database instance is required")
        if database is not None and path is not None:
            raise ValueError("provide a database path or database instance, not both")
        self.database = database or Database(path)  # type: ignore[arg-type]
        self.repository = Repository(self.database)
        self.logger = logger or get_logger()
        self.processor = Processor(self.repository, self.logger, extraction_provider)
        self._closed = False

    def ingest(self, event: EventInput | dict[str, Any]) -> EventReceipt:
        parsed = event if isinstance(event, EventInput) else EventInput.model_validate(event)
        redacted_payload = redact(parsed.payload)
        event_id, duplicate, content_hash, job_id = self.repository.ingest(parsed.namespace_id, parsed, redacted_payload)
        log(
            self.logger,
            logging.INFO,
            "ingestion.accepted",
            namespace_id=parsed.namespace_id,
            event_id=event_id,
            duplicate=duplicate,
        )
        return EventReceipt(
            event_id=UUID(event_id),
            namespace_id=parsed.namespace_id,
            content_hash=content_hash,
            duplicate=duplicate,
            job_id=UUID(job_id),
        )

    def ingest_batch(self, events: list[EventInput]) -> BatchEventResponse:
        return BatchEventResponse(receipts=[self.ingest(event) for event in events])

    def history(self, namespace_id: str, memory_id: str) -> list[dict[str, Any]] | None:
        return self.repository.history(namespace_id, memory_id)

    def invalidate(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        return self.repository.invalidate_memory(namespace_id, memory_id, reason)

    def export_namespace(self, namespace_id: str) -> dict[str, Any]:
        return self.repository.export_namespace(namespace_id)

    def import_namespace(self, document: dict[str, Any], namespace_id: str) -> dict[str, int]:
        return self.repository.import_namespace(document, namespace_id)

    def delete_namespace(self, namespace_id: str) -> bool:
        return self.repository.delete_namespace(namespace_id)

    def episodes(self, namespace_id: str) -> list[dict[str, Any]]:
        return self.repository.list_episodes(namespace_id)

    def feedback(self, namespace_id: str, memory_id: str, label: str, note: str | None) -> str:
        return self.repository.record_feedback(namespace_id, memory_id, label, note)

    def process(self, namespace_id: str, limit: int = 100, lease_seconds: int = 30) -> ProcessResponse:
        processed, failed, dead, accepted, rejected = self.processor.process_namespace(namespace_id, limit, lease_seconds)
        return ProcessResponse(processed=processed, failed=failed, dead_lettered=dead, accepted=accepted, rejected=rejected)

    def search(self, namespace_id: str, query: str, limit: int = 10) -> list[SearchResult]:
        results = self.repository.search(namespace_id, query, limit)
        log(
            self.logger,
            logging.INFO,
            "retrieval.completed",
            namespace_id=namespace_id,
            result_count=len(results),
        )
        return results

    def context(self, namespace_id: str, query: str, token_budget: int = 500, limit: int = 10) -> ContextResponse:
        result = build_context(self.repository, namespace_id, query, limit, token_budget)
        log(
            self.logger,
            logging.INFO,
            "context.completed",
            namespace_id=namespace_id,
            result_count=len(result.results),
            abstained=result.abstained,
        )
        return result

    def get_memory(self, namespace_id: str, memory_id: str) -> MemoryResponse | None:
        return self.repository.get_memory(namespace_id, memory_id)

    def close(self) -> None:
        if not self._closed:
            self.database.close()
            self._closed = True

    def checkpoint(self) -> None:
        self.database.checkpoint()

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from .core.logging import get_logger, log
from .core.redaction import redact
from .memory.consolidator import consolidate
from .memory.processor import Processor
from .memory.provider import ExtractionProvider, SessionSummaryProvider
from .models import (
    BatchEventResponse,
    ContextResponse,
    EventInput,
    EventReceipt,
    MemoryResponse,
    ProcessResponse,
    SearchResult,
)
from .retrieval.context import build_context
from .retrieval.embedding import EmbeddingProvider
from .storage.db import Database
from .storage.repository import Repository


class TermyteDB:
    MAX_EVENT_BYTES = 1_048_576

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        database: Database | None = None,
        logger: logging.Logger | None = None,
        extraction_provider: ExtractionProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        summary_provider: SessionSummaryProvider | None = None,
    ):
        if database is None and path is None:
            raise ValueError("an explicit database path or database instance is required")
        if database is not None and path is not None:
            raise ValueError("provide a database path or database instance, not both")
        self.database = database or Database(path)  # type: ignore[arg-type]
        self.repository = Repository(self.database, embedding_provider)
        self.logger = logger or get_logger()
        self.processor = Processor(self.repository, self.logger, extraction_provider, summary_provider)
        self._closed = False

    def ingest(self, event: EventInput | dict[str, Any]) -> EventReceipt:
        result = self.ingest_batch([event])
        return result.receipts[0].model_copy(update={"accepted": result.accepted, "rejected": result.rejected})

    def _prepare_event(self, event: EventInput | dict[str, Any]) -> tuple[EventInput, dict[str, Any]]:
        parsed = event if isinstance(event, EventInput) else EventInput.model_validate(event)
        redacted_payload = redact(parsed.payload)
        payload_bytes = len(json.dumps(redacted_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if payload_bytes > self.MAX_EVENT_BYTES or sum(item.size_bytes for item in parsed.artifacts) > 100_000_000:
            raise ValueError(f"event payload exceeds {self.MAX_EVENT_BYTES} bytes")
        return parsed, redacted_payload

    def ingest_batch(self, events: list[EventInput | dict[str, Any]]) -> BatchEventResponse:
        # One SQLite connection is shared by the embedded engine. Keep a direct
        # ingestion call coherent while separate engine instances use WAL.
        with self.database.lock:
            return self._ingest_batch(events)

    def _ingest_batch(self, events: list[EventInput | dict[str, Any]]) -> BatchEventResponse:
        prepared = [self._prepare_event(event) for event in events]
        if not prepared:
            raise ValueError("at least one event is required")
        namespace_id = prepared[0][0].namespace_id
        if any(parsed.namespace_id != namespace_id for parsed, _ in prepared):
            raise ValueError("all events in one ingestion call must share a namespace")

        receipts: list[EventReceipt] = []
        new_event_ids: list[str] = []
        for parsed, redacted_payload in prepared:
            event_id, duplicate, content_hash = self.repository.ingest(parsed.namespace_id, parsed, redacted_payload)
            if not duplicate:
                new_event_ids.append(event_id)
            receipts.append(
                EventReceipt(
                    event_id=UUID(event_id),
                    namespace_id=parsed.namespace_id,
                    content_hash=content_hash,
                    duplicate=duplicate,
                )
            )
            log(
                self.logger,
                logging.INFO,
                "ingestion.accepted",
                namespace_id=parsed.namespace_id,
                event_id=event_id,
                duplicate=duplicate,
            )

        accepted = rejected = 0
        if new_event_ids:
            _, accepted, rejected = self.processor.process_events(namespace_id, new_event_ids)
        return BatchEventResponse(receipts=receipts, accepted=accepted, rejected=rejected)

    def history(self, namespace_id: str, memory_id: str) -> list[dict[str, Any]] | None:
        return self.repository.history(namespace_id, memory_id)

    def invalidate(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        return self.repository.invalidate_memory(namespace_id, memory_id, reason)

    def forget(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        return self.repository.forget_memory(namespace_id, memory_id, reason)

    def restore(self, namespace_id: str, memory_id: str) -> bool:
        return self.repository.restore_memory(namespace_id, memory_id)

    def export_namespace(self, namespace_id: str) -> dict[str, Any]:
        return self.repository.export_namespace(namespace_id)

    def import_namespace(self, document: dict[str, Any], namespace_id: str) -> dict[str, int]:
        return self.repository.import_namespace(document, namespace_id)

    def delete_namespace(self, namespace_id: str) -> bool:
        return self.repository.delete_namespace(namespace_id)

    def episodes(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.list_episodes(namespace_id, limit, offset)

    def update_episode(self, namespace_id: str, episode_id: str, status: str, summary: str | None) -> bool:
        return self.repository.update_episode(namespace_id, episode_id, status, summary)

    def feedback(self, namespace_id: str, memory_id: str, label: str, note: str | None) -> str:
        return self.repository.record_feedback(namespace_id, memory_id, label, note)

    def event(self, namespace_id: str, event_id: str) -> dict[str, Any] | None:
        return self.repository.get_event(namespace_id, event_id)

    def events(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.list_events(namespace_id, limit, offset)

    def evidence(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.list_evidence(namespace_id, limit, offset)

    def jobs(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.list_jobs(namespace_id, limit, offset)

    def context_requests(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.list_context_requests(namespace_id, limit, offset)

    def extraction_runs(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.list_extraction_runs(namespace_id, limit, offset)

    def extraction_decisions(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.list_extraction_decisions(namespace_id, limit, offset)

    def feedback_rows(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.list_feedback(namespace_id, limit, offset)

    def metrics(self, namespace_id: str) -> dict[str, float | int]:
        return self.repository.metrics(namespace_id)

    def encoding_decisions(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.repository.encoding_decisions(namespace_id, limit, offset)

    def consolidate(self, namespace_id: str, *, limit: int = 5, dry_run: bool = True) -> dict[str, Any]:
        return consolidate(self.repository, namespace_id, limit=limit, mode="dry-run" if dry_run else "apply")

    def procedures(self, namespace_id: str, goal: str, environment: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.repository.retrieve_procedure(namespace_id, goal, environment, limit)

    def save_procedure(
        self,
        namespace_id: str,
        goal: str,
        environment: str,
        preconditions: list[str],
        actions: list[str],
        expected_outcome: str,
        observed_outcome: str | None,
        failures: list[str],
        success: bool,
        evidence: list[tuple[str, str]],
    ) -> str:
        return self.repository.upsert_procedure(
            namespace_id,
            goal,
            environment,
            preconditions,
            actions,
            expected_outcome,
            observed_outcome,
            failures,
            success,
            evidence,
        )

    def refresh_accessibility(self, namespace_id: str) -> int:
        return self.repository.accessibility(namespace_id)

    def process(self, namespace_id: str, limit: int = 100, lease_seconds: int = 180) -> ProcessResponse:
        processed, failed, dead, accepted, rejected = self.processor.process_namespace(namespace_id, limit, lease_seconds)
        return ProcessResponse(processed=processed, failed=failed, dead_lettered=dead, accepted=accepted, rejected=rejected)

    def process_with_timeout(self, namespace_id: str, limit: int = 100, lease_seconds: int = 180, timeout_seconds: float = 30.0) -> ProcessResponse:
        processed, failed, dead, accepted, rejected = self.processor.process_namespace(namespace_id, limit, lease_seconds, timeout_seconds)
        return ProcessResponse(processed=processed, failed=failed, dead_lettered=dead, accepted=accepted, rejected=rejected)

    def cancel_job(self, namespace_id: str, job_id: str) -> bool:
        return self.repository.cancel_job(namespace_id, job_id)

    def search(self, namespace_id: str, query: str, limit: int = 10, historical: bool = False) -> list[SearchResult]:
        results = self.repository.search(namespace_id, query, limit, historical)
        log(
            self.logger,
            logging.INFO,
            "retrieval.completed",
            namespace_id=namespace_id,
            result_count=len(results),
        )
        return results

    def context(self, namespace_id: str, query: str, token_budget: int = 500, limit: int = 10, historical: bool = False) -> ContextResponse:
        result = build_context(self.repository, namespace_id, query, limit, token_budget, historical)
        result.request_id = UUID(self.repository.record_context_request(namespace_id, query, token_budget, result))
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

    def memories(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[MemoryResponse]:
        return self.repository.list_memories(namespace_id, limit, offset)

    def close(self) -> None:
        if not self._closed:
            self.database.close()
            self._closed = True

    def checkpoint(self) -> None:
        self.database.checkpoint()

    def backup(self, destination: str | Path) -> None:
        self.database.backup(destination)

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
    EventInput,
    EventReceipt,
    MemoryResponse,
    ProcessResponse,
    SearchResult,
    SessionSearchResult,
)
from .retrieval.embedding import EmbeddingProvider
from .storage.db import Database
from .storage.repository import Repository


class TermyteDB:
    """Embedded memory engine - simple facade over EventStore + MemoryStore + HybridRetriever.

    Core API:
      ingest / ingest_batch  - add events -> memories with temporal block
      search                 - hybrid FTS+Vector fetch top-N then rerank to top-K (namespace-filtered, rank-fused, optional rerank; returns memories directly)
      get_memory / memories  - read with temporal {valid_from, valid_until}
      update / invalidate / forget / restore - manage memory lifecycle
    Everything below `--- Extended / Debug ---` is backward-compat for tests/benchmarks.
    TermyteDB returns memories. The caller decides how those memories become model context.
    """

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
        event_hashes: dict[str, str] = {}
        for parsed, redacted_payload in prepared:
            event_id, duplicate, content_hash = self.repository.ingest(parsed.namespace_id, parsed, redacted_payload)
            if not duplicate:
                new_event_ids.append(event_id)
                event_hashes[event_id] = content_hash
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

        # Index source-grounded chunks before extraction.
        self.repository.rebuild_chunks(namespace_id)

        # Durability: enqueue jobs before extraction so failures remain retryable
        job_ids: dict[str, str] = {}
        for event_id in new_event_ids:
            job_id = self.repository.create_processing_job(namespace_id, event_id, event_hashes[event_id])
            job_ids[event_id] = job_id

        accepted = rejected = 0
        if new_event_ids:
            try:
                _, accepted, rejected = self.processor.process_events(namespace_id, new_event_ids)
                # Direct success: remove the now-completed jobs (keeps 0 pending after success)
                self.repository.delete_completed_jobs(namespace_id, new_event_ids)
            except Exception as exc:
                from .memory.provider import ProviderError as _ProviderError

                # Ensure jobs are marked as failed/retryable for later process() calls
                retryable = getattr(exc, "retryable", True) if isinstance(exc, _ProviderError) else True
                retry_after = getattr(exc, "retry_after", None) if isinstance(exc, _ProviderError) else None
                # If provider was never configured, treat as retryable ValueError
                if isinstance(exc, ValueError) and "no extraction provider" in str(exc):
                    retryable = True
                    retry_after = None
                    exc = _ProviderError(str(exc), retryable=True, error_class="no_provider")
                # Also try to parse Retry-After from message if not structured (fallback)
                if retry_after is None and "retry_after=" in str(exc):
                    try:
                        import re

                        m = re.search(r"retry_after=([0-9.]+)", str(exc))
                        if m:
                            retry_after = float(m.group(1))
                    except Exception:
                        retry_after = None
                error_msg = str(exc)[:500]
                for event_id in new_event_ids:
                    job_id = job_ids.get(event_id)
                    if job_id:
                        try:
                            self.repository.fail_job(namespace_id, job_id, error_msg, retryable=retryable, retry_after=retry_after)
                        except Exception:
                            pass
                raise exc
        return BatchEventResponse(receipts=receipts, accepted=accepted, rejected=rejected)

    # -- Core read/write with temporal blocks ---------------------------------
    def history(self, namespace_id: str, memory_id: str) -> list[dict[str, Any]] | None:
        return self.repository.history(namespace_id, memory_id)

    def search_sessions(self, namespace_id: str, query: str, limit: int = 20) -> list[SessionSearchResult]:
        """Search original conversation sessions as a fallback to memory search."""
        return self.repository.search_sessions(namespace_id, query, limit)

    def search_context(self, namespace_id: str, query: str, limit: int = 20) -> dict[str, list[Any]]:
        """Return compact memories and their source chunks for answer generation."""
        memories = self.search(namespace_id, query, limit)
        from .retrieval.context import pack_evidence

        packed = pack_evidence(memories, lambda memory: self.repository.chunks_for_events(namespace_id, [str(x) for x in memory.source_event_ids]))
        sessions = [] if memories else self.search_sessions(namespace_id, query, limit)
        return {"memories": memories, "chunks": packed["memories"], "text": packed["text"], "token_count": packed["token_count"], "sessions": sessions}

    def build_answer_context(self, namespace_id: str, query: str, limit: int = 6, token_budget: int = 3000) -> dict[str, Any]:
        memories = self.search(namespace_id, query, limit)
        from .retrieval.context import pack_evidence

        return pack_evidence(memories, lambda memory: self.repository.chunks_for_events(namespace_id, [str(x) for x in memory.source_event_ids]), token_budget=token_budget)

    def invalidate(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        return self.repository.invalidate_memory(namespace_id, memory_id, reason)

    def forget(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        return self.repository.forget_memory(namespace_id, memory_id, reason)

    def restore(self, namespace_id: str, memory_id: str) -> bool:
        return self.repository.restore_memory(namespace_id, memory_id)

    # -- Extended / Debug (backward-compat; not part of 5-method spec) ---------
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

    def get_memory(self, namespace_id: str, memory_id: str) -> MemoryResponse | None:
        return self.repository.get_memory(namespace_id, memory_id)

    def memories(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[MemoryResponse]:
        return self.repository.list_memories(namespace_id, limit, offset)

    def update_memory(self, namespace_id: str, memory_id: str, statement: str, confidence: float | None = None, kind: str | None = None, source_event_ids: list[str] | None = None, evidence_excerpt: str | None = None) -> bool:
        src = source_event_ids[0] if source_event_ids else None
        return self.repository.update_memory(namespace_id, memory_id, statement, confidence=confidence, kind=kind, source_event_id=src, evidence_excerpt=evidence_excerpt)

    def delete_memory(self, namespace_id: str, memory_id: str) -> bool:
        return self.repository.delete_memory(namespace_id, memory_id)

    def close(self) -> None:
        if not self._closed:
            self.database.close()
            self._closed = True

    def checkpoint(self) -> None:
        self.database.checkpoint()

    def backup(self, destination: str | Path) -> None:
        self.database.backup(destination)

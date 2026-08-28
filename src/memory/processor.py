from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any
from uuid import UUID

from ..core.logging import log
from ..core.redaction import redact_text
from ..models import ExtractionRequest
from ..storage.repository import Repository
from .extraction import CandidateRejected, validate_candidate
from .extractor import payload_text
from .provider import ExtractionProvider, ProviderError, SessionSummaryProvider, default_extraction_provider


class Processor:
    def __init__(
        self,
        repository: Repository,
        logger: logging.Logger,
        provider: ExtractionProvider | None = None,
        summary_provider: SessionSummaryProvider | None = None,
    ):
        self.repository = repository
        self.logger = logger
        self.provider = provider
        self.summary_provider = summary_provider

    def process_events(
        self,
        namespace_id: str,
        event_ids: list[str],
        timeout_seconds: float = 30.0,
    ) -> tuple[int, int, int]:
        """Extract and store memories for one direct ingestion call."""
        if not event_ids:
            return 0, 0, 0

        events = self.repository.events_by_id(namespace_id, event_ids)
        if len(events) != len(event_ids):
            raise ValueError("one or more ingestion events are missing")

        provider = self.provider or default_extraction_provider()
        self.provider = provider
        deadline = time.perf_counter() + timeout_seconds
        included: dict[UUID, str] = {}
        current_events: dict[str, Any] = {}
        episode_ids: set[str] = set()

        for event in events:
            event_id = UUID(event["id"])
            included[event_id] = payload_text(json.loads(event["payload_json"]), event["type"])
            current_events[str(event_id)] = event
            if event["episode_id"]:
                episode_ids.add(str(event["episode_id"]))

        # Current input is never split. Add only a small recent-context window.
        context_limit = len(included) + 10
        for event in events:
            for event_key, source in self.repository.extraction_window(namespace_id, event["id"], limit=4).items():
                if len(included) >= context_limit:
                    break
                included.setdefault(UUID(event_key), source)

        input_snapshot = {str(key): value for key, value in included.items()}
        existing_memories = self.repository.related_memory_context(namespace_id, "\n".join(input_snapshot.values()))
        existing_by_ref = {str(item["ref"]): str(item["memory_id"]) for item in existing_memories}
        request = ExtractionRequest(
            namespace_id=namespace_id,
            events=[UUID(event_id) for event_id in event_ids],
            evidence_text=included,
            existing_memories=existing_memories,
        )

        provider_result = provider.extract(
            request,
            timeout_seconds=max(0.001, deadline - time.perf_counter()),
            cancellation=lambda: time.perf_counter() >= deadline,
        )
        response = provider_result.response
        resolved_candidates = []
        for candidate in response.candidates:
            if candidate.existing_memory_ref is None:
                resolved_candidates.append(candidate)
                continue
            existing_id = existing_by_ref.get(candidate.existing_memory_ref)
            resolved_candidates.append(
                candidate if existing_id is None else candidate.model_copy(update={"existing_memory_id": UUID(existing_id)})
            )
        response = response.model_copy(update={"candidates": resolved_candidates})

        run_id = str(uuid.uuid4())
        self.repository.record_run(
            namespace_id,
            {
                "id": run_id,
                "namespace_id": namespace_id,
                "input_hash": hashlib.sha256(json.dumps(input_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "provider_name": provider_result.provider_name,
                "model_name": provider_result.model_name,
                "prompt_version": response.prompt_version,
                "schema_version": response.schema_version,
                "started_at": self._now(),
                "completed_at": None,
                "input_events_json": json.dumps(list(input_snapshot), separators=(",", ":")),
                "input_characters": sum(len(text) for text in input_snapshot.values()),
                "input_tokens": provider_result.input_tokens,
                "output_tokens": provider_result.output_tokens,
                "latency_ms": provider_result.latency_ms,
                "accepted_count": 0,
                "rejected_count": 0,
                "status": "processing",
                "error_class": None,
                "estimated_cost_usd": self._estimated_cost(provider_result.input_tokens, provider_result.output_tokens),
            },
        )

        accepted = rejected = 0
        try:
            fingerprints: set[str] = set()
            validated_candidates: list[tuple[Any, Any, Any]] = []
            for candidate in response.candidates:
                try:
                    if candidate.existing_memory_ref is not None and candidate.existing_memory_id is None:
                        raise CandidateRejected("unknown_existing_memory_ref")
                    validated = validate_candidate(namespace_id, candidate, included)
                    if validated.fingerprint in fingerprints:
                        raise CandidateRejected("duplicate_candidate")
                    fingerprints.add(validated.fingerprint)
                    source_event = current_events.get(str(candidate.evidence[0].event_id), events[0])
                    validated_candidates.append((candidate, validated, source_event))
                except CandidateRejected as exc:
                    rejected += 1
                    self.repository.record_decision(
                        namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT"
                    )

            statements = [validated.candidate.statement for _, validated, _ in validated_candidates]
            embeddings = self.repository.embed_many(statements) if statements else []
            for (candidate, validated, source_event), embedding in zip(validated_candidates, embeddings, strict=True):
                try:
                    memory_id, action, version_id = self.repository.reconcile_candidate(
                        namespace_id, source_event, validated, run_id, embedding
                    )
                    self.repository.record_decision(
                        namespace_id, run_id, candidate, validated.fingerprint, "accepted", None, action, memory_id, version_id
                    )
                    accepted += 1
                except CandidateRejected as exc:
                    rejected += 1
                    self.repository.record_decision(
                        namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT"
                    )

            self.repository.finish_run(namespace_id, run_id, accepted, rejected, "completed")
        except Exception as exc:
            error_class = exc.error_class if isinstance(exc, ProviderError) else type(exc).__name__
            self.repository.finish_run(namespace_id, run_id, accepted, rejected, "failed", error_class)
            raise

        for episode_id in sorted(episode_ids):
            try:
                self.repository.refresh_episode_summary(namespace_id, episode_id, summary_provider=self.summary_provider)
            except Exception:
                log(self.logger, logging.WARNING, "processing.summary_refresh_failed", namespace_id=namespace_id, episode_id=episode_id)

        log(
            self.logger,
            logging.INFO,
            "processing.completed",
            namespace_id=namespace_id,
            event_count=len(events),
            candidates=len(response.candidates),
            accepted=accepted,
            rejected=rejected,
        )
        return len(events), accepted, rejected

    def process_namespace(self, namespace_id: str, limit: int = 100, lease_seconds: int = 180, timeout_seconds: float = 30.0) -> tuple[int, int, int, int, int]:
        deadline = time.perf_counter() + timeout_seconds
        jobs = self.repository.claim_jobs(namespace_id, min(limit, 100), lease_seconds)
        provider = self.provider or default_extraction_provider()
        self.provider = provider
        processed = failed = dead = accepted = rejected = 0
        episode_ids: set[str] = set()
        scoped_jobs: dict[str, list[tuple[Any, Any]]] = {}
        for job in jobs:
            event = self.repository.event_for_job(namespace_id, job["id"])
            # Events without an explicit conversation scope must not be mixed
            # with unrelated events just because they share a namespace.
            scope = str(event["stream_id"] or event["session_id"] or event["id"])
            scoped_jobs.setdefault(scope, []).append((job, event))

        # Extract a bounded session batch in one provider call. This keeps the
        # evidence window coherent and avoids one LLM request per event.
        batches: list[list[tuple[Any, Any]]] = []
        for scoped in scoped_jobs.values():
            for start in range(0, len(scoped), 20):
                batches.append(scoped[start : start + 20])

        for batch in batches:
            if time.perf_counter() >= deadline:
                break
            run_id = str(uuid.uuid4())
            started = time.perf_counter()
            batch_accepted = batch_rejected = 0
            included: dict[UUID, str] = {}
            current_jobs: dict[str, tuple[Any, Any]] = {}
            input_snapshot: dict[str, str] = {}
            try:
                for job, event in batch:
                    if event["episode_id"]:
                        episode_ids.add(str(event["episode_id"]))
                    if not self.repository.heartbeat_job(namespace_id, job["id"], lease_seconds, str(job["lease_token"])):
                        raise RuntimeError("job lease is no longer active")
                    event_id = UUID(event["id"])
                    current_jobs[str(event_id)] = (job, event)
                    source = payload_text(json.loads(event["payload_json"]), event["type"])
                    included[event_id] = source

                # Add prior same-session turns after current events. Every
                # current event stays addressable for exact evidence offsets.
                for _job, event in batch:
                    for event_key, source in self.repository.extraction_window(namespace_id, event["id"], limit=4).items():
                        if len(included) >= 20:
                            break
                        included.setdefault(UUID(event_key), source)

                input_snapshot = {str(key): value for key, value in included.items()}
                existing_memories = self.repository.related_memory_context(namespace_id, "\n".join(input_snapshot.values()))
                existing_by_ref = {str(item["ref"]): str(item["memory_id"]) for item in existing_memories}
                request = ExtractionRequest(
                    namespace_id=namespace_id,
                    events=list(included),
                    evidence_text=included,
                    existing_memories=existing_memories,
                )
                remaining = max(0.001, deadline - time.perf_counter())
                for job, _event in batch:
                    if not self.repository.heartbeat_job(namespace_id, job["id"], lease_seconds, str(job["lease_token"])):
                        raise RuntimeError("job lease is no longer active")
                provider_result = provider.extract(
                    request,
                    timeout_seconds=remaining,
                    cancellation=lambda: time.perf_counter() >= deadline,
                )
                for job, _event in batch:
                    if not self.repository.heartbeat_job(namespace_id, job["id"], lease_seconds, str(job["lease_token"])):
                        raise RuntimeError("job lease is no longer active")
                response = provider_result.response
                resolved_candidates = []
                for candidate in response.candidates:
                    if candidate.existing_memory_ref is None:
                        resolved_candidates.append(candidate)
                        continue
                    existing_id = existing_by_ref.get(candidate.existing_memory_ref)
                    if existing_id is None:
                        resolved_candidates.append(candidate)
                        continue
                    resolved_candidates.append(candidate.model_copy(update={"existing_memory_id": uuid.UUID(existing_id)}))
                response = response.model_copy(update={"candidates": resolved_candidates})
                provider_name, model_name = provider_result.provider_name, provider_result.model_name
                input_tokens, output_tokens, provider_latency = provider_result.input_tokens, provider_result.output_tokens, provider_result.latency_ms
                self.repository.record_run(
                    namespace_id,
                    {
                        "id": run_id,
                        "namespace_id": namespace_id,
                        "input_hash": hashlib.sha256(json.dumps(input_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "provider_name": provider_name,
                        "model_name": model_name,
                        "prompt_version": response.prompt_version,
                        "schema_version": response.schema_version,
                        "started_at": self._now(),
                        "completed_at": None,
                        "input_events_json": json.dumps(list(input_snapshot), separators=(",", ":")),
                        "input_characters": sum(len(text) for text in input_snapshot.values()),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_ms": provider_latency or int((time.perf_counter() - started) * 1000),
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "status": "processing",
                        "error_class": None,
                        "estimated_cost_usd": self._estimated_cost(input_tokens, output_tokens),
                    },
                )
                fingerprints: set[str] = set()
                validated_candidates: list[tuple[Any, Any, Any, Any]] = []
                for candidate in response.candidates:
                    try:
                        if candidate.existing_memory_ref is not None and candidate.existing_memory_id is None:
                            raise CandidateRejected("unknown_existing_memory_ref")
                        validated = validate_candidate(namespace_id, candidate, included)
                        if validated.fingerprint in fingerprints:
                            raise CandidateRejected("duplicate_candidate")
                        fingerprints.add(validated.fingerprint)
                        source_job, source_event = current_jobs.get(str(candidate.evidence[0].event_id), batch[0])
                        validated_candidates.append((candidate, validated, source_job, source_event))
                    except CandidateRejected as exc:
                        rejected += 1
                        batch_rejected += 1
                        self.repository.record_decision(namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT")

                embedding_values = [validated.candidate.statement for _, validated, _job, _event in validated_candidates]
                embedding_vectors = self.repository.embed_many(embedding_values) if embedding_values else []
                for (candidate, validated, source_job, source_event), embedding in zip(validated_candidates, embedding_vectors, strict=True):
                    try:
                        reconciled_memory_id, action, reconciled_version_id = self.repository.reconcile_candidate(
                            namespace_id,
                            source_event,
                            validated,
                            run_id,
                            embedding,
                            job_id=str(source_job["id"]),
                            lease_token=str(source_job["lease_token"]),
                        )
                        memory_id = reconciled_memory_id
                        version_id = reconciled_version_id
                        self.repository.record_decision(namespace_id, run_id, candidate, validated.fingerprint, "accepted", None, action, memory_id, version_id)
                        accepted += 1
                        batch_accepted += 1
                    except CandidateRejected as exc:
                        rejected += 1
                        batch_rejected += 1
                        self.repository.record_decision(namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT")
                self.repository.finish_run(namespace_id, run_id, batch_accepted, batch_rejected, "completed")
                for job, _event in batch:
                    if self.repository.complete_job(namespace_id, job["id"], str(job["lease_token"])) is False:
                        raise RuntimeError("job lease is no longer active")
                processed += len(batch)
                log(
                    self.logger,
                    logging.INFO,
                    "processing.completed",
                    namespace_id=namespace_id,
                    job_id=str(batch[0][0]["id"]),
                    batch_jobs=len(batch),
                    candidates=len(response.candidates),
                    accepted=batch_accepted,
                    rejected=batch_rejected,
                )
            except Exception as exc:
                safe_error = redact_text(str(exc))
                error_class = exc.error_class if isinstance(exc, ProviderError) else type(exc).__name__
                if not self._run_exists(namespace_id, run_id) and input_snapshot and self.provider is not None:
                    self.repository.record_run(
                        namespace_id,
                        {
                            "id": run_id,
                            "namespace_id": namespace_id,
                            "input_hash": hashlib.sha256(json.dumps(input_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                            "provider_name": self.provider.name,
                            "model_name": self.provider.model,
                            "prompt_version": "unknown",
                            "schema_version": "extraction-v1",
                            "started_at": self._now(),
                            "completed_at": None,
                            "input_events_json": json.dumps(list(input_snapshot), separators=(",", ":")),
                            "input_characters": sum(len(text) for text in input_snapshot.values()),
                            "input_tokens": None,
                            "output_tokens": None,
                            "latency_ms": int((time.perf_counter() - started) * 1000),
                            "accepted_count": 0,
                            "rejected_count": 0,
                            "status": "processing",
                            "error_class": None,
                            "estimated_cost_usd": None,
                        },
                    )
                if self._run_exists(namespace_id, run_id):
                    self.repository.finish_run(namespace_id, run_id, batch_accepted, batch_rejected, "failed", error_class)
                for job, _event in batch:
                    status = self.repository.fail_job(
                        namespace_id,
                        job["id"],
                        safe_error,
                        retryable=not isinstance(exc, ProviderError) or exc.retryable,
                        lease_token=str(job["lease_token"]),
                    )
                    failed += 1
                    dead += status == "dead"
                    log(self.logger, logging.ERROR, "processing.failed", namespace_id=namespace_id, job_id=job["id"], status=status, error=safe_error)
        for episode_id in sorted(episode_ids):
            try:
                self.repository.refresh_episode_summary(namespace_id, episode_id, summary_provider=self.summary_provider)
            except Exception:
                log(self.logger, logging.WARNING, "processing.summary_refresh_failed", namespace_id=namespace_id, episode_id=episode_id)
        return processed, failed, dead, accepted, rejected

    @staticmethod
    def _estimated_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
        input_rate = os.environ.get("TERMYTEDB_INPUT_COST_PER_1K_USD")
        output_rate = os.environ.get("TERMYTEDB_OUTPUT_COST_PER_1K_USD")
        if input_tokens is None or output_tokens is None or input_rate is None or output_rate is None:
            return None
        try:
            return round((input_tokens * float(input_rate) + output_tokens * float(output_rate)) / 1000, 8)
        except ValueError:
            return None

    def _run_exists(self, namespace_id: str, run_id: str) -> bool:
        return self.repository.db.execute("SELECT 1 FROM extraction_runs WHERE id=? AND namespace_id=?", (run_id, namespace_id)).fetchone() is not None

    @staticmethod
    def _safe_fingerprint(candidate: Any) -> str:
        try:
            value = json.dumps(candidate.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        except Exception:
            value = redact_text(str(candidate))
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def _now() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()

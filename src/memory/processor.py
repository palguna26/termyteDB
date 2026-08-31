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
from ..models import ExtractionRequest, ReconciliationRequest
from ..storage.repository import Repository
from .extraction import CandidateRejected, validate_candidate
from .extractor import payload_text
from .provider import ExtractionProvider, ProviderError, SessionSummaryProvider, default_extraction_provider


def _get_extraction_stages() -> list[str]:
    # The single prompt covers all memory types. This remains a list only for
    # compatible tracing metadata in the existing processing pipeline.
    return ["facts"]


def _is_reconciliation_enabled() -> bool:
    # Simple extraction is deliberately one LLM call. Reconciliation remains
    # available as an explicit opt-in for applications that need it.
    raw = os.environ.get("TERMYTEDB_RECONCILIATION_ENABLED", "0")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _get_max_calls() -> int:
    raw = os.environ.get("TERMYTEDB_MAX_LLM_CALLS_PER_BATCH")
    if raw is None:
        return 10
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


def _ensure_stage_column(repo: Repository) -> None:
    try:
        cols = {row[1] for row in repo.db.execute("PRAGMA table_info(extraction_runs)").fetchall()}
        if "stage" not in cols:
            repo.db.execute("ALTER TABLE extraction_runs ADD COLUMN stage TEXT")
        if "candidate_count" not in cols:
            repo.db.execute("ALTER TABLE extraction_runs ADD COLUMN candidate_count INTEGER")
        if "input_event_ids" not in cols:
            repo.db.execute("ALTER TABLE extraction_runs ADD COLUMN input_event_ids TEXT")
    except Exception:
        pass


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

    def _collect_multi_pass(
        self,
        provider: Any,
        namespace_id: str,
        events: list[UUID],
        included: dict[UUID, str],
        existing_memories: list[dict[str, Any]],
        deadline: float,
    ) -> tuple[list[Any], list[Any], dict[str, Any]]:
        stages = _get_extraction_stages()
        max_calls = _get_max_calls()
        if len(stages) > max_calls:
            raise ProviderError(f"extraction stages {len(stages)} exceeds max {max_calls}", retryable=False, error_class="max_calls_exceeded")
        all_candidates: list[Any] = []
        provider_results: list[Any] = []
        last_error: Exception | None = None
        retryable_error: ProviderError | None = None
        for stage in stages:
            # Check cancellation deadline
            remaining = max(0.001, deadline - time.perf_counter())
            if time.perf_counter() >= deadline:
                break
            request = ExtractionRequest(
                namespace_id=namespace_id,
                events=events,
                evidence_text=included,
                existing_memories=existing_memories,
                stage=stage,  # type: ignore[arg-type]
            )
            try:
                pr = provider.extract(
                    request,
                    timeout_seconds=remaining,
                    cancellation=lambda: time.perf_counter() >= deadline,
                )
                # Tag candidates with source_stage if missing
                for c in pr.response.candidates:
                    if getattr(c, "source_stage", None) is None:
                        try:
                            # annotate
                            all_candidates.append(c.model_copy(update={"source_stage": stage}))  # type: ignore[arg-type]
                        except Exception:
                            all_candidates.append(c)
                    else:
                        all_candidates.append(c)
                provider_results.append(pr)
                # Pacing between stages to avoid burst 429s when multi-stage is enabled
                if len(stages) > 1 and stage != stages[-1]:
                    time.sleep(0.15)
            except ProviderError as exc:
                last_error = exc
                if exc.retryable:
                    retryable_error = exc
                log(self.logger, logging.WARNING, "processing.stage_failed", namespace_id=namespace_id, stage=stage, error=str(exc), retryable=exc.retryable)
                continue
            except Exception as exc:
                last_error = exc
                log(self.logger, logging.WARNING, "processing.stage_failed", namespace_id=namespace_id, stage=stage, error=str(exc))
                continue
        if not provider_results and last_error is not None:
            raise last_error
        # Rate-limit or other retryable failures must not silently produce partial memories.
        # If any stage failed retryably, fail the whole batch so the job remains retryable.
        if retryable_error is not None and len(provider_results) != len(stages):
            log(self.logger, logging.WARNING, "processing.retryable_partial_suppressed", namespace_id=namespace_id, succeeded=len(provider_results), failed=len(stages) - len(provider_results))
            raise retryable_error
        meta = {"stages": stages, "succeeded": len(provider_results), "failed": len(stages) - len(provider_results)}
        return all_candidates, provider_results, meta

    def _apply_reconciliation(
        self,
        provider: Any,
        namespace_id: str,
        existing_memories: list[dict[str, Any]],
        candidates: list[Any],
        deadline: float,
    ) -> list[Any]:
        if not _is_reconciliation_enabled():
            return candidates
        if not candidates or not existing_memories:
            return candidates
        # Only attempt if provider has reconcile method
        if not hasattr(provider, "reconcile"):
            return candidates
        max_calls = _get_max_calls()
        # Ensure reconciliation call doesn't exceed max
        stages = _get_extraction_stages()
        if len(stages) + 1 > max_calls:
            log(self.logger, logging.WARNING, "processing.reconciliation_skipped_max_calls", namespace_id=namespace_id)
            return candidates
        try:
            req = ReconciliationRequest(
                namespace_id=namespace_id,
                existing_memories=existing_memories,
                new_candidates=candidates,
            )
            remaining = max(0.001, deadline - time.perf_counter())
            result = provider.reconcile(
                req,
                timeout_seconds=remaining,
                cancellation=lambda: time.perf_counter() >= deadline,
            )
            # Apply decisions: map candidate_index -> action/ref
            existing_by_ref = {str(m.get("ref")): m for m in existing_memories}
            # Validate and apply
            for dec in result.response.decisions:
                idx = dec.candidate_index
                if idx < 0 or idx >= len(candidates):
                    log(self.logger, logging.WARNING, "processing.reconciliation_invalid_index", namespace_id=namespace_id, index=idx)
                    continue
                # Validate action names — simplified to 5 actions
                action = str(dec.action).casefold()
                allowed = {"insert", "reinforce", "update", "supersede", "ignore"}
                if action in {"dispute", "contradiction"}:
                    action = "ignore"
                if action not in allowed:
                    log(self.logger, logging.WARNING, "processing.reconciliation_invalid_action", namespace_id=namespace_id, action=action)
                    continue
                cand = candidates[idx]
                # Validate ref if present
                ref = dec.existing_memory_ref
                if ref is not None and ref not in existing_by_ref:
                    log(self.logger, logging.WARNING, "processing.reconciliation_unknown_ref", namespace_id=namespace_id, ref=ref)
                    continue
                # Apply to candidate intent/ref; code enforces referential integrity later (unknown refs rejected)
                try:
                    updated = cand.model_copy(update={"intent": action, "existing_memory_ref": ref})
                    candidates[idx] = updated
                except Exception as exc:
                    log(self.logger, logging.WARNING, "processing.reconciliation_apply_failed", namespace_id=namespace_id, error=str(exc))
                    continue
            # Record reconciliation trace if needed (could be separate run)
            log(self.logger, logging.INFO, "processing.reconciliation_completed", namespace_id=namespace_id, decisions=len(result.response.decisions))
        except ProviderError as exc:
            log(self.logger, logging.WARNING, "processing.reconciliation_failed", namespace_id=namespace_id, error=str(exc), retryable=exc.retryable, retry_after=getattr(exc, "retry_after", None))
            if exc.retryable:
                # Retryable reconciliation failure must not silently persist partial extraction.
                raise
        except Exception as exc:
            log(self.logger, logging.WARNING, "processing.reconciliation_failed", namespace_id=namespace_id, error=str(exc))
        return candidates

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

        # Multi-pass extraction
        all_candidates, provider_results, meta = self._collect_multi_pass(
            provider, namespace_id, [UUID(eid) for eid in event_ids], included, existing_memories, deadline
        )
        # LLM reconciliation (separate call)
        all_candidates = self._apply_reconciliation(provider, namespace_id, existing_memories, all_candidates, deadline)

        # Aggregate provider metadata for tracing
        if provider_results:
            # Merge prompt versions and token counts
            prompt_version = "+".join(pr.prompt_version for pr in provider_results)
            # Use first provider's metadata plus aggregated
            primary = provider_results[0]
            total_input = sum(pr.input_tokens or 0 for pr in provider_results if pr.input_tokens)
            total_output = sum(pr.output_tokens or 0 for pr in provider_results if pr.output_tokens)
            total_latency = sum(pr.latency_ms for pr in provider_results)
            provider_name = primary.provider_name
            model_name = primary.model_name
            # But keep candidates as merged list
        else:
            # No successful stage (should have raised), fallback
            provider_name = getattr(provider, "name", "unknown")
            model_name = getattr(provider, "model", "unknown")
            prompt_version = "unknown"
            total_input = None
            total_output = None
            total_latency = 0

        # Resolve refs to IDs (structural integrity)
        resolved_candidates = []
        for candidate in all_candidates:
            if candidate.existing_memory_ref is None:
                resolved_candidates.append(candidate)
                continue
            existing_id = existing_by_ref.get(candidate.existing_memory_ref)
            resolved_candidates.append(
                candidate if existing_id is None else candidate.model_copy(update={"existing_memory_id": UUID(existing_id)})
            )

        # Build a pseudo response for downstream (keep candidates merged)
        # We need to record a run; use first schema version or default
        schema_version = provider_results[0].response.schema_version if provider_results else "extraction-v1"

        run_id = str(uuid.uuid4())
        # Ensure detailed tracing columns exist (phase 6)
        _ensure_stage_column(self.repository)
        # Record aggregated run; per-stage rows could be added but aggregated keeps backward compat
        self.repository.record_run(
            namespace_id,
            {
                "id": run_id,
                "namespace_id": namespace_id,
                "input_hash": hashlib.sha256(json.dumps(input_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "provider_name": provider_name,
                "model_name": model_name,
                "prompt_version": prompt_version[:100],  # truncate to fit schema
                "schema_version": schema_version,
                "started_at": self._now(),
                "completed_at": None,
                "input_events_json": json.dumps(list(input_snapshot), separators=(",", ":")),
                "input_characters": sum(len(text) for text in input_snapshot.values()),
                "input_tokens": total_input if total_input else None,
                "output_tokens": total_output if total_output else None,
                "latency_ms": total_latency or int((time.perf_counter() - (deadline - timeout_seconds)) * 1000),
                "accepted_count": 0,
                "rejected_count": 0,
                "status": "processing",
                "error_class": None,
                "estimated_cost_usd": self._estimated_cost(total_input if total_input else None, total_output if total_output else None),
            },
        )

        accepted = rejected = 0
        try:
            fingerprints: set[str] = set()
            validated_candidates: list[tuple[Any, Any, Any]] = []
            for candidate in resolved_candidates:
                try:
                    if candidate.existing_memory_ref is not None and candidate.existing_memory_id is None:
                        raise CandidateRejected("unknown_existing_memory_ref")
                    # Validate intent action names — simplified to 5 actions per Phase 2
                    action = str(candidate.intent).casefold() if hasattr(candidate, "intent") else "insert"
                    allowed_intents = {"insert", "reinforce", "update", "supersede", "ignore"}
                    # Backward compat: map dispute/contradiction -> ignore (no longer a distinct state)
                    if action in {"dispute", "contradiction"}:
                        action = "ignore"
                        candidate = candidate.model_copy(update={"intent": "ignore"})
                    if action not in allowed_intents:
                        raise CandidateRejected("invalid_intent")
                    validated = validate_candidate(namespace_id, candidate, included)
                    if validated.fingerprint in fingerprints:
                        raise CandidateRejected("duplicate_candidate")
                    fingerprints.add(validated.fingerprint)
                    if candidate.evidence:
                        source_event = current_events.get(str(candidate.evidence[0].event_id), events[0])
                    else:
                        source_event = events[0]
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
            candidates=len(resolved_candidates),
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
                # Multi-pass extraction for batch
                all_candidates, provider_results, meta = self._collect_multi_pass(
                    provider, namespace_id, list(included), included, existing_memories, deadline
                )
                # Reconciliation
                all_candidates = self._apply_reconciliation(provider, namespace_id, existing_memories, all_candidates, deadline)
                for job, _event in batch:
                    if not self.repository.heartbeat_job(namespace_id, job["id"], lease_seconds, str(job["lease_token"])):
                        raise RuntimeError("job lease is no longer active")
                # Resolve refs
                resolved_candidates: list[Any] = []
                for candidate in all_candidates:
                    if candidate.existing_memory_ref is None:
                        resolved_candidates.append(candidate)
                        continue
                    existing_id = existing_by_ref.get(candidate.existing_memory_ref)
                    if existing_id is None:
                        resolved_candidates.append(candidate)
                        continue
                    resolved_candidates.append(candidate.model_copy(update={"existing_memory_id": uuid.UUID(existing_id)}))
                # Aggregate provider metadata
                if provider_results:
                    prompt_version = "+".join(pr.prompt_version for pr in provider_results)[:100]
                    provider_name, model_name = provider_results[0].provider_name, provider_results[0].model_name
                    total_input = sum(pr.input_tokens or 0 for pr in provider_results if pr.input_tokens)
                    total_output = sum(pr.output_tokens or 0 for pr in provider_results if pr.output_tokens)
                    provider_latency = sum(pr.latency_ms for pr in provider_results)
                    schema_version = provider_results[0].response.schema_version
                else:
                    provider_name = getattr(provider, "name", "unknown")
                    model_name = getattr(provider, "model", "unknown")
                    prompt_version = "unknown"
                    total_input = None
                    total_output = None
                    provider_latency = int((time.perf_counter() - started) * 1000)
                    schema_version = "extraction-v1"
                _ensure_stage_column(self.repository)
                self.repository.record_run(
                    namespace_id,
                    {
                        "id": run_id,
                        "namespace_id": namespace_id,
                        "input_hash": hashlib.sha256(json.dumps(input_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "provider_name": provider_name,
                        "model_name": model_name,
                        "prompt_version": prompt_version,
                        "schema_version": schema_version,
                        "started_at": self._now(),
                        "completed_at": None,
                        "input_events_json": json.dumps(list(input_snapshot), separators=(",", ":")),
                        "input_characters": sum(len(text) for text in input_snapshot.values()),
                        "input_tokens": total_input if total_input else None,
                        "output_tokens": total_output if total_output else None,
                        "latency_ms": provider_latency or int((time.perf_counter() - started) * 1000),
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "status": "processing",
                        "error_class": None,
                        "estimated_cost_usd": self._estimated_cost(total_input if total_input else None, total_output if total_output else None),
                    },
                )
                fingerprints: set[str] = set()
                validated_candidates: list[tuple[Any, Any, Any, Any]] = []
                for candidate in resolved_candidates:
                    try:
                        if candidate.existing_memory_ref is not None and candidate.existing_memory_id is None:
                            raise CandidateRejected("unknown_existing_memory_ref")
                        action = str(candidate.intent).casefold() if hasattr(candidate, "intent") else "insert"
                        allowed_intents = {"insert", "reinforce", "update", "supersede", "ignore"}
                        if action in {"dispute", "contradiction"}:
                            action = "ignore"
                            candidate = candidate.model_copy(update={"intent": "ignore"})
                        if action not in allowed_intents:
                            raise CandidateRejected("invalid_intent")
                        validated = validate_candidate(namespace_id, candidate, included)
                        if validated.fingerprint in fingerprints:
                            raise CandidateRejected("duplicate_candidate")
                        fingerprints.add(validated.fingerprint)
                        if candidate.evidence:
                            source_job, source_event = current_jobs.get(str(candidate.evidence[0].event_id), batch[0])
                        else:
                            source_job, source_event = batch[0]
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
                    candidates=len(resolved_candidates),
                    accepted=batch_accepted,
                    rejected=batch_rejected,
                )
            except Exception as exc:
                safe_error = redact_text(str(exc))
                error_class = exc.error_class if isinstance(exc, ProviderError) else type(exc).__name__
                if not self._run_exists(namespace_id, run_id) and input_snapshot and self.provider is not None:
                    _ensure_stage_column(self.repository)
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
                    retry_after = getattr(exc, "retry_after", None) if isinstance(exc, ProviderError) else None
                    # Fallback parse from message
                    if retry_after is None and "retry_after=" in safe_error:
                        try:
                            import re

                            m = re.search(r"retry_after=([0-9.]+)", safe_error)
                            if m:
                                retry_after = float(m.group(1))
                        except Exception:
                            retry_after = None
                    status = self.repository.fail_job(
                        namespace_id,
                        job["id"],
                        safe_error,
                        retryable=not isinstance(exc, ProviderError) or exc.retryable,
                        retry_after=retry_after,
                        lease_token=str(job["lease_token"]),
                    )
                    failed += 1
                    dead += status == "dead"
                    log(self.logger, logging.ERROR, "processing.failed", namespace_id=namespace_id, job_id=job["id"], status=status, error=safe_error, retry_after=retry_after)
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

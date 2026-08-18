from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, cast

from .extraction import CandidateRejected, rule_candidate_to_contract, validate_candidate
from .extractor import Candidate as RuleCandidate
from .extractor import extract, payload_text
from .logging import log
from .provider import ExtractionProvider, ProviderError
from .redaction import redact_text
from .repository import Repository
from .schemas import ExtractionRequest, ExtractionResponse


class Processor:
    def __init__(self, repository: Repository, logger: logging.Logger, provider: ExtractionProvider | None = None):
        self.repository = repository
        self.logger = logger
        self.provider = provider

    def process_namespace(self, namespace_id: str, limit: int = 100, lease_seconds: int = 30, timeout_seconds: float = 30.0) -> tuple[int, int, int, int, int]:
        deadline = time.perf_counter() + timeout_seconds
        jobs = self.repository.claim_jobs(namespace_id, limit, lease_seconds)
        processed = failed = dead = accepted = rejected = 0
        for job in jobs:
            if time.perf_counter() >= deadline:
                break
            run_id = str(uuid.uuid4())
            started = time.perf_counter()
            try:
                event = self.repository.event_for_job(namespace_id, job["id"])
                if not self.repository.heartbeat_job(namespace_id, job["id"], lease_seconds):
                    raise RuntimeError("job lease is no longer active")
                payload = json.loads(event["payload_json"])
                source = payload_text(payload)
                event_id = uuid.UUID(event["id"])
                included = {event_id: source}
                if self.provider is None:
                    raw_candidates = [rule_candidate_to_contract(item, event_id, source) for item in extract(payload)]
                    response = ExtractionResponse(schema_version="extraction-v1", prompt_version="rule-v1", candidates=raw_candidates)
                    provider_name, model_name, input_tokens, output_tokens = "rule", "rule-v1", None, None
                    provider_latency = 0
                    rule_mode = True
                else:
                    request = ExtractionRequest(namespace_id=namespace_id, events=[event_id], evidence_text=included)
                    remaining = max(0.001, deadline - time.perf_counter())
                    provider_result = self.provider.extract(
                        request,
                        timeout_seconds=remaining,
                        cancellation=lambda: time.perf_counter() >= deadline,
                    )
                    response = provider_result.response
                    provider_name, model_name = provider_result.provider_name, provider_result.model_name
                    input_tokens, output_tokens, provider_latency = provider_result.input_tokens, provider_result.output_tokens, provider_result.latency_ms
                    rule_mode = False
                self.repository.record_run(
                    namespace_id,
                    {
                        "id": run_id,
                        "namespace_id": namespace_id,
                        "input_hash": hashlib.sha256(source.encode()).hexdigest(),
                        "provider_name": provider_name,
                        "model_name": model_name,
                        "prompt_version": response.prompt_version,
                        "schema_version": response.schema_version,
                        "started_at": self._now(),
                        "completed_at": None,
                        "input_events_json": json.dumps([str(event_id)]),
                        "input_characters": len(source),
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
                for candidate in response.candidates:
                    try:
                        validated = validate_candidate(namespace_id, candidate, included)
                        if validated.fingerprint in fingerprints:
                            raise CandidateRejected("duplicate_candidate")
                        fingerprints.add(validated.fingerprint)
                        memory_id: str | None = None
                        version_id: str | None = None
                        if rule_mode:
                            span = validated.candidate.evidence[0]
                            subject_key = validated.candidate.subject
                            previous = self.repository.db.execute(
                                """SELECT v.statement, v.status FROM memories m
                                JOIN memory_versions v ON v.id=m.current_version_id AND v.namespace_id=m.namespace_id
                                WHERE m.namespace_id=? AND m.kind=? AND m.subject_key=?""",
                                (namespace_id, validated.candidate.kind, subject_key),
                            ).fetchone()
                            rule = RuleCandidate(
                                validated.candidate.kind, validated.candidate.subject, validated.candidate.statement, span.start_offset, span.end_offset
                            )
                            memory_id = self.repository.save_candidate(namespace_id, event, rule)
                            row = self.repository.db.execute(
                                "SELECT current_version_id FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)
                            ).fetchone()
                            version_id = cast(str | None, row["current_version_id"] if row else None)
                            if previous and previous["status"] == "active" and previous["statement"] == validated.candidate.statement:
                                action = "REINFORCE"
                            elif previous:
                                action = "SUPERSEDE"
                            else:
                                action = "INSERT"
                        else:
                            reconciled_memory_id, action, reconciled_version_id = self.repository.reconcile_candidate(namespace_id, event, validated, run_id)
                            memory_id = reconciled_memory_id
                            version_id = reconciled_version_id
                        self.repository.record_decision(namespace_id, run_id, candidate, validated.fingerprint, "accepted", None, action, memory_id, version_id)
                        accepted += 1
                    except CandidateRejected as exc:
                        rejected += 1
                        self.repository.record_decision(namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT")
                self.repository.finish_run(namespace_id, run_id, accepted, rejected, "completed")
                self.repository.complete_job(namespace_id, job["id"])
                processed += 1
                log(
                    self.logger,
                    logging.INFO,
                    "processing.completed",
                    namespace_id=namespace_id,
                    job_id=job["id"],
                    candidates=len(response.candidates),
                    accepted=accepted,
                    rejected=rejected,
                )
            except Exception as exc:
                safe_error = redact_text(str(exc))
                if self._run_exists(namespace_id, run_id):
                    error_class = exc.error_class if isinstance(exc, ProviderError) else type(exc).__name__
                    self.repository.finish_run(namespace_id, run_id, accepted, rejected, "failed", error_class)
                status = self.repository.fail_job(namespace_id, job["id"], safe_error)
                failed += 1
                dead += status == "dead"
                log(self.logger, logging.ERROR, "processing.failed", namespace_id=namespace_id, job_id=job["id"], status=status, error=safe_error)
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

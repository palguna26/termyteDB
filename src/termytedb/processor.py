from __future__ import annotations

import json
import logging
from typing import Any

from .extractor import extract, payload_text
from .logging import log
from .redaction import redact_text
from .repository import Repository


class Processor:
    def __init__(self, repository: Repository, logger: logging.Logger):
        self.repository = repository
        self.logger = logger

    def process_namespace(self, namespace_id: str, limit: int = 100, lease_seconds: int = 30) -> tuple[int, int, int]:
        jobs = self.repository.claim_jobs(namespace_id, limit, lease_seconds)
        processed = failed = dead = 0
        for job in jobs:
            try:
                event = self.repository.event_for_job(namespace_id, job["id"])
                payload = json.loads(event["payload_json"])
                candidates = extract(payload)
                for candidate in candidates:
                    self._validate_evidence(namespace_id, event, candidate, payload)
                    self.repository.save_candidate(namespace_id, event, candidate)
                self.repository.complete_job(namespace_id, job["id"])
                processed += 1
                log(
                    self.logger,
                    logging.INFO,
                    "processing.completed",
                    namespace_id=namespace_id,
                    job_id=job["id"],
                    candidates=len(candidates),
                )
            except Exception as exc:  # the job record is the failure boundary
                safe_error = redact_text(str(exc))
                status = self.repository.fail_job(namespace_id, job["id"], safe_error)
                failed += 1
                dead += status == "dead"
                log(
                    self.logger,
                    logging.ERROR,
                    "processing.failed",
                    namespace_id=namespace_id,
                    job_id=job["id"],
                    status=status,
                    error=safe_error,
                )
        return processed, failed, dead

    @staticmethod
    def _validate_evidence(namespace_id: str, event: object, candidate: object, payload: dict[str, Any]) -> None:
        event_namespace = event["namespace_id"]  # type: ignore[index]
        if event_namespace != namespace_id:
            raise ValueError("evidence namespace mismatch")
        text = payload_text(payload)
        if candidate.start_offset < 0 or candidate.end_offset > len(text):  # type: ignore[attr-defined]
            raise ValueError("invalid evidence span")
        if text[candidate.start_offset : candidate.end_offset] != candidate.statement:  # type: ignore[attr-defined]
            raise ValueError("evidence does not match source text")

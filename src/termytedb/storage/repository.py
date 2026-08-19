from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ..api.schemas import EventInput, EvidenceCitation, ExtractionCandidate, MemoryResponse, SearchResult
from ..core.errors import IdempotencyConflict
from ..core.redaction import redact_text
from ..memory.extraction import ValidatedCandidate
from ..memory.extractor import Candidate
from ..retrieval.embedding import EmbeddingProvider, FastEmbedProvider, cosine
from .db import Database


def iso(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()


def stable_uuid(namespace_id: str, idempotency_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"termytedb:event:{namespace_id}:{idempotency_key}"))


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_artifact_uri(uri: str | None, content_hash: str) -> str | None:
    if not uri:
        return None
    expected = f"cas://{content_hash.removeprefix('sha256:')}"
    return uri if uri == expected else "[REDACTED]"


def canonical_event_content(event: EventInput, redacted_payload: dict[str, Any]) -> str:
    artifacts = [
        {
            **item.model_dump(mode="json", exclude={"uri", "metadata"}),
            "uri": safe_artifact_uri(item.uri, item.content_hash),
            "metadata": {key: redact_text(value) for key, value in item.metadata.items()},
        }
        for item in event.artifacts
    ]
    return json.dumps(
        {
            "protocol_version": event.protocol_version,
            "type": event.type,
            "stream_id": event.stream_id,
            "actor_id": event.actor_id,
            "agent_id": event.agent_id,
            "session_id": event.session_id,
            "source_id": event.source_id,
            "artifacts": artifacts,
            "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
            "payload": redacted_payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class Repository:
    def __init__(self, database: Database, embedding: EmbeddingProvider | None = None):
        self.db = database
        self.embedding = embedding or FastEmbedProvider()

    def ensure_namespace(self, namespace_id: str, org_id: str = "default") -> None:
        with self.db.lock, self.db.connection:
            self.db.execute(
                "INSERT OR IGNORE INTO namespaces(id, org_id, created_at) VALUES (?, ?, ?)",
                (namespace_id, org_id, iso()),
            )

    def ingest(self, namespace_id: str, event: EventInput, redacted_payload: dict[str, Any]) -> tuple[str, bool, str, str]:
        if event.namespace_id != namespace_id:
            raise ValueError("event namespace does not match repository namespace")
        with self.db.lock:
            self.ensure_namespace(namespace_id)
            event_id = stable_uuid(namespace_id, event.idempotency_key)
            idempotency_hash = hash_text(event.idempotency_key)
            payload_json = json.dumps(redacted_payload, sort_keys=True, separators=(",", ":"))
            content_hash = hash_text(canonical_event_content(event, redacted_payload))
            occurred = iso(event.occurred_at)
            job_id = str(uuid.uuid4())
            with self.db.connection:
                existing = self.db.execute(
                    "SELECT id, content_hash FROM events WHERE namespace_id = ? AND idempotency_hash = ?",
                    (namespace_id, idempotency_hash),
                ).fetchone()
                if existing:
                    if existing["content_hash"] != content_hash:
                        raise IdempotencyConflict("idempotency key is already used for different content")
                    existing_job = self.db.execute(
                        "SELECT id FROM processing_jobs WHERE namespace_id = ? AND event_id = ? ORDER BY created_at LIMIT 1",
                        (namespace_id, existing["id"]),
                    ).fetchone()
                    if not existing_job:
                        raise RuntimeError("event is missing its processing job")
                    return existing["id"], True, existing["content_hash"], existing_job["id"]
                self.db.execute(
                    """INSERT INTO events
                    (id, namespace_id, protocol_version, stream_id, actor_id, agent_id, session_id, source_id, idempotency_hash, type, occurred_at,
                     payload_json,
                     content_hash, redaction_state, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'redacted', ?)""",
                    (
                        event_id,
                        namespace_id,
                        event.protocol_version,
                        event.stream_id,
                        event.actor_id,
                        event.agent_id,
                        event.session_id,
                        event.source_id,
                        idempotency_hash,
                        event.type,
                        occurred,
                        payload_json,
                        content_hash,
                        iso(),
                    ),
                )
                for artifact in event.artifacts:
                    safe_uri = safe_artifact_uri(artifact.uri, artifact.content_hash)
                    safe_metadata = {key: redact_text(value) for key, value in artifact.metadata.items()}
                    self.db.execute(
                        """INSERT INTO artifacts(id, namespace_id, event_id, media_type, size_bytes, uri, content_hash, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid.uuid4()), namespace_id, event_id, artifact.media_type, artifact.size_bytes,
                            safe_uri, artifact.content_hash, json.dumps(safe_metadata, sort_keys=True, separators=(",", ":")),
                        ),
                    )
                self.db.execute(
                    """INSERT INTO processing_jobs
                    (id, namespace_id, event_id, input_hash, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                    (job_id, namespace_id, event_id, content_hash, iso(), iso()),
                )
                self._assign_episode(namespace_id, event_id, event.stream_id, occurred)
            return event_id, False, content_hash, job_id

    def _assign_episode(self, namespace_id: str, event_id: str, stream_id: str | None, occurred: str) -> str:
        """Assign an event to the nearest deterministic stream episode."""
        event_time = datetime.fromisoformat(occurred)
        candidates = self.db.execute(
            """SELECT ep.id, ep.start_event_id, ep.end_event_id,
                      start.occurred_at AS start_at, end.occurred_at AS end_at
               FROM episodes ep
               JOIN events start ON start.id=ep.start_event_id
               JOIN events end ON end.id=ep.end_event_id
               WHERE ep.namespace_id=? AND (ep.stream_id IS ? OR ep.stream_id=?)
               ORDER BY ep.updated_at DESC""",
            (namespace_id, stream_id, stream_id),
        ).fetchall()
        selected: sqlite3.Row | None = None
        for row in candidates:
            start = datetime.fromisoformat(row["start_at"])
            end = datetime.fromisoformat(row["end_at"])
            if min(abs(event_time - start), abs(event_time - end)) <= timedelta(minutes=30):
                selected = row
                break
        if selected is None:
            episode_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"termytedb:episode:{namespace_id}:{event_id}"))
            self.db.execute(
                """INSERT INTO episodes(id, namespace_id, stream_id, start_event_id, end_event_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (episode_id, namespace_id, stream_id, event_id, event_id, iso(), iso()),
            )
            self.db.execute(
                "INSERT INTO episode_events(episode_id, namespace_id, event_id, ordinal) VALUES (?, ?, ?, 0)",
                (episode_id, namespace_id, event_id),
            )
            return episode_id
        episode_id = cast(str, selected["id"])
        count = int(self.db.execute("SELECT COUNT(*) FROM episode_events WHERE episode_id=?", (episode_id,)).fetchone()[0])
        self.db.execute(
            "INSERT INTO episode_events(episode_id, namespace_id, event_id, ordinal) VALUES (?, ?, ?, ?)",
            (episode_id, namespace_id, event_id, count),
        )
        start_id = selected["start_event_id"]
        end_id = selected["end_event_id"]
        if event_time < datetime.fromisoformat(selected["start_at"]):
            start_id = event_id
        if event_time > datetime.fromisoformat(selected["end_at"]):
            end_id = event_id
        self.db.execute(
            "UPDATE episodes SET start_event_id=?, end_event_id=?, updated_at=? WHERE id=? AND namespace_id=?",
            (start_id, end_id, iso(), episode_id, namespace_id),
        )
        return episode_id

    def list_episodes(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM episodes WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?", (namespace_id, limit, offset)
        ).fetchall()
        return [dict(row) for row in rows]

    def update_episode(self, namespace_id: str, episode_id: str, status: str, summary: str | None) -> bool:
        with self.db.lock, self.db.connection:
            cursor = self.db.execute(
                "UPDATE episodes SET status=?, summary=COALESCE(?, summary), updated_at=? WHERE id=? AND namespace_id=?",
                (status, redact_text(summary) if summary else None, iso(), episode_id, namespace_id),
            )
            return cursor.rowcount == 1

    def record_feedback(self, namespace_id: str, memory_id: str, label: str, note: str | None) -> str:
        feedback_id = str(uuid.uuid4())
        with self.db.lock, self.db.connection:
            exists = self.db.execute("SELECT 1 FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)).fetchone()
            if not exists:
                raise KeyError(memory_id)
            self.db.execute(
                "INSERT INTO feedback(id, namespace_id, memory_id, label, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (feedback_id, namespace_id, memory_id, label, redact_text(note) if note else None, iso()),
            )
        return feedback_id

    def list_feedback(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM feedback WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?", (namespace_id, limit, offset)
        ).fetchall()
        return [dict(row) for row in rows]

    def metrics(self, namespace_id: str) -> dict[str, float | int]:
        def count(table: str) -> int:
            return int(self.db.execute(f"SELECT COUNT(*) FROM {table} WHERE namespace_id=?", (namespace_id,)).fetchone()[0])

        job_counts = {
            str(row["status"]): int(row["count"])
            for row in self.db.execute("SELECT status, COUNT(*) AS count FROM processing_jobs WHERE namespace_id=? GROUP BY status", (namespace_id,))
        }
        latency = self.db.execute(
            "SELECT AVG(latency_ms) AS average, MAX(latency_ms) AS maximum FROM extraction_runs WHERE namespace_id=? AND latency_ms IS NOT NULL",
            (namespace_id,),
        ).fetchone()
        return {
            "events": count("events"),
            "memories": count("memories"),
            "memory_versions": count("memory_versions"),
            "jobs": count("processing_jobs"),
            "extraction_runs": count("extraction_runs"),
            "extraction_decisions": count("extraction_decisions"),
            "jobs_pending": job_counts.get("pending", 0),
            "jobs_processing": job_counts.get("processing", 0),
            "jobs_completed": job_counts.get("completed", 0),
            "jobs_failed": job_counts.get("failed", 0),
            "jobs_dead": job_counts.get("dead", 0),
            "jobs_cancelled": job_counts.get("cancelled", 0),
            "average_extraction_latency_ms": round(float(latency["average"] or 0), 3),
            "maximum_extraction_latency_ms": int(latency["maximum"] or 0),
            "estimated_extraction_cost_usd": float(
                self.db.execute("SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM extraction_runs WHERE namespace_id=?", (namespace_id,)).fetchone()[0]
            ),
        }

    def get_event(self, namespace_id: str, event_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM events WHERE id=? AND namespace_id=?", (event_id, namespace_id)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload_json"] = json.loads(result["payload_json"])
        result["artifacts"] = [
            {**dict(item), "metadata_json": json.loads(item["metadata_json"])}
            for item in self.db.execute("SELECT * FROM artifacts WHERE event_id=? AND namespace_id=? ORDER BY id", (event_id, namespace_id)).fetchall()
        ]
        result["evidence_refs"] = [
            dict(item)
            for item in self.db.execute(
                "SELECT * FROM evidence_refs WHERE event_id=? AND namespace_id=? ORDER BY id", (event_id, namespace_id)
            ).fetchall()
        ]
        return result

    def list_events(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id FROM events WHERE namespace_id=? ORDER BY occurred_at, id LIMIT ? OFFSET ?",
            (namespace_id, limit, offset),
        ).fetchall()
        return [event for row in rows if (event := self.get_event(namespace_id, row["id"])) is not None]

    def list_evidence(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """SELECT * FROM evidence_refs WHERE namespace_id=?
            ORDER BY id LIMIT ? OFFSET ?""",
            (namespace_id, limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_jobs(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM processing_jobs WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?", (namespace_id, limit, offset)
        ).fetchall()
        return [dict(row) for row in rows]

    def related_memory_context(self, namespace_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return bounded current memory context for extraction reconciliation.

        This deliberately reuses the normal namespace-safe search path. The result is
        comparison context for the provider; evidence for new claims must still come
        from the current input events.
        """
        results = self.search(namespace_id, query, max(1, min(limit, 20)), historical=False)
        return [
            {
                "memory_id": str(item.memory_id),
                "memory_version_id": str(item.memory_version_id),
                "kind": item.kind,
                "statement": item.statement,
                "status": item.status,
            }
            for item in results
        ]

    def record_context_request(self, namespace_id: str, query: str, token_budget: int, response: Any) -> str:
        request_id = str(uuid.uuid4())
        with self.db.lock, self.db.connection:
            self.db.execute(
                """INSERT INTO context_requests
                (id, namespace_id, query, token_budget, selected_json, token_count, abstained, diagnostics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id, namespace_id, query, token_budget,
                    json.dumps([str(item.memory_version_id) for item in response.results], separators=(",", ":")),
                    response.token_count, int(response.abstained), json.dumps(response.diagnostics, sort_keys=True, separators=(",", ":")), iso(),
                ),
            )
        return request_id

    def list_context_requests(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM context_requests WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?", (namespace_id, limit, offset)
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["selected_json"] = json.loads(item["selected_json"])
            item["diagnostics_json"] = json.loads(item["diagnostics_json"])
            result.append(item)
        return result

    def claim_jobs(self, namespace_id: str, limit: int, lease_seconds: int) -> list[sqlite3.Row]:
        # SQLite computes the lease consistently and avoids client clock formatting issues.
        with self.db.lock, self.db.connection:
            rows = self.db.execute(
                """SELECT * FROM processing_jobs
                WHERE namespace_id = ? AND ((status = 'pending' OR
                    (status = 'failed' AND (next_attempt_at IS NULL OR next_attempt_at <= datetime('now'))) OR
                    (status = 'processing' AND lease_until < datetime('now'))))
                ORDER BY created_at, rowid LIMIT ?""",
                (namespace_id, limit),
            ).fetchall()
            claimed: list[sqlite3.Row] = []
            for row in rows:
                self.db.execute(
                    """UPDATE processing_jobs SET status='processing', attempts=attempts+1,
                    lease_until=datetime('now', ?), updated_at=datetime('now')
                    WHERE id=? AND namespace_id=?""",
                    (f"+{lease_seconds} seconds", row["id"], namespace_id),
                )
                claimed_row = self.db.execute(
                    "SELECT * FROM processing_jobs WHERE id=? AND namespace_id=?",
                    (row["id"], namespace_id),
                ).fetchone()
                if claimed_row:
                    claimed.append(claimed_row)
        return claimed

    def complete_job(self, namespace_id: str, job_id: str) -> None:
        with self.db.connection:
            self.db.execute(
                "UPDATE processing_jobs SET status='completed', lease_until=NULL, updated_at=? WHERE id=? AND namespace_id=? AND status='processing'",
                (iso(), job_id, namespace_id),
            )

    def heartbeat_job(self, namespace_id: str, job_id: str, lease_seconds: int) -> bool:
        """Extend an active lease without reviving cancelled or completed work."""
        with self.db.lock, self.db.connection:
            cursor = self.db.execute(
                """UPDATE processing_jobs
                SET lease_until=datetime('now', ?), updated_at=?
                WHERE id=? AND namespace_id=? AND status='processing'""",
                (f"+{lease_seconds} seconds", iso(), job_id, namespace_id),
            )
            return cursor.rowcount == 1

    def fail_job(self, namespace_id: str, job_id: str, error: str, *, retryable: bool = True) -> str:
        with self.db.connection:
            row = self.db.execute(
                "SELECT status, attempts, max_attempts FROM processing_jobs WHERE id=? AND namespace_id=?",
                (job_id, namespace_id),
            ).fetchone()
            if not row:
                raise KeyError(job_id)
            if row["status"] == "cancelled":
                return "cancelled"
            status = "dead" if not retryable or row["attempts"] >= row["max_attempts"] else "failed"
            delay = min(300, 2 ** max(1, int(row["attempts"])))
            self.db.execute(
                "UPDATE processing_jobs SET status=?, lease_until=NULL, next_attempt_at=?, last_error=?, updated_at=? WHERE id=? AND namespace_id=?",
                (status, None, error, iso(), job_id, namespace_id),
            )
            if status != "dead":
                self.db.execute(
                    "UPDATE processing_jobs SET next_attempt_at=datetime('now', ?) WHERE id=? AND namespace_id=?",
                    (f"+{delay} seconds", job_id, namespace_id),
                )
            return status

    def cancel_job(self, namespace_id: str, job_id: str) -> bool:
        with self.db.connection:
            cursor = self.db.execute(
                "UPDATE processing_jobs SET status='cancelled', lease_until=NULL, updated_at=? "
                "WHERE id=? AND namespace_id=? AND status IN ('pending', 'failed', 'processing')",
                (iso(), job_id, namespace_id),
            )
            return cursor.rowcount == 1

    def event_for_job(self, namespace_id: str, job_id: str) -> sqlite3.Row:
        row = self.db.execute(
            """SELECT e.* FROM events e JOIN processing_jobs j ON j.event_id=e.id
            WHERE j.id=? AND j.namespace_id=? AND e.namespace_id=?""",
            (job_id, namespace_id, namespace_id),
        ).fetchone()
        if not row:
            raise KeyError(job_id)
        return cast(sqlite3.Row, row)

    def save_candidate(self, namespace_id: str, event: sqlite3.Row, candidate: Candidate) -> str:
        now = iso()
        with self.db.lock, self.db.connection:
            stored_event = self.db.execute(
                "SELECT id, namespace_id, created_at FROM events WHERE id=? AND namespace_id=?",
                (event["id"], namespace_id),
            ).fetchone()
            if not stored_event:
                raise ValueError("evidence event is not in the requested namespace")
            if stored_event["created_at"] > now:
                raise ValueError("evidence postdates derived version")
            memory = self.db.execute(
                "SELECT * FROM memories WHERE namespace_id=? AND kind=? AND subject_key=?",
                (namespace_id, candidate.kind, candidate.subject_key),
            ).fetchone()
            memory_id = memory["id"] if memory else str(uuid.uuid4())
            if not memory:
                self.db.execute(
                    "INSERT INTO memories(id, namespace_id, kind, subject_key, status, confidence, created_at) VALUES (?, ?, ?, ?, 'active', 1.0, ?)",
                    (memory_id, namespace_id, candidate.kind, candidate.subject_key, now),
                )
            current = self.db.execute(
                "SELECT * FROM memory_versions WHERE memory_id=? AND namespace_id=? ORDER BY version DESC LIMIT 1",
                (memory_id, namespace_id),
            ).fetchone()
            if current and current["statement"] == candidate.statement and current["status"] == "active":
                existing_ref = self.db.execute(
                    """SELECT 1 FROM evidence_refs
                    WHERE memory_version_id=? AND namespace_id=? AND event_id=?
                      AND start_offset=? AND end_offset=?""",
                    (
                        current["id"],
                        namespace_id,
                        event["id"],
                        candidate.start_offset,
                        candidate.end_offset,
                    ),
                ).fetchone()
                if not existing_ref:
                    self.db.execute(
                        """INSERT INTO evidence_refs
                        (id, memory_version_id, namespace_id, event_id, start_offset, end_offset, excerpt)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid.uuid4()),
                            current["id"],
                            namespace_id,
                            event["id"],
                            candidate.start_offset,
                            candidate.end_offset,
                            candidate.statement,
                        ),
                    )
                return memory_id
            version = (current["version"] + 1) if current else 1
            if current:
                self.db.execute(
                    "UPDATE memory_versions SET status='superseded', valid_to=? WHERE id=? AND namespace_id=?",
                    (now, current["id"], namespace_id),
                )
                self.db.execute(
                    "DELETE FROM memory_fts WHERE memory_version_id=? AND namespace_id=?",
                    (current["id"], namespace_id),
                )
                # Keep historical embeddings so explicit historical retrieval
                # remains dense-searchable after supersession.
            version_id = str(uuid.uuid4())
            self.db.execute(
                """INSERT INTO memory_versions
                (id, memory_id, namespace_id, source_event_id, evidence_start_offset,
                 evidence_end_offset, evidence_excerpt, version, statement, valid_from,
                 recorded_at, status, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 'deterministic_rule')""",
                (
                    version_id,
                    memory_id,
                    namespace_id,
                    event["id"],
                    candidate.start_offset,
                    candidate.end_offset,
                    candidate.statement,
                    version,
                    candidate.statement,
                    now,
                    now,
                ),
            )
            self.db.execute(
                """INSERT INTO evidence_refs
                (id, memory_version_id, namespace_id, event_id, start_offset, end_offset, excerpt)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    version_id,
                    namespace_id,
                    event["id"],
                    candidate.start_offset,
                    candidate.end_offset,
                    candidate.statement,
                ),
            )
            self.db.execute(
                "UPDATE memories SET current_version_id=?, status='active' WHERE id=? AND namespace_id=?",
                (version_id, memory_id, namespace_id),
            )
            self.db.execute(
                "DELETE FROM memory_fts WHERE memory_version_id=? AND namespace_id=?",
                (version_id, namespace_id),
            )
            self.db.execute(
                "INSERT INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?, ?, ?, ?)",
                (version_id, namespace_id, candidate.statement, candidate.statement),
            )
            self.db.execute(
                "INSERT INTO memory_embeddings(memory_version_id, namespace_id, provider, dimensions, vector_json) VALUES (?, ?, ?, ?, ?)",
                (
                    version_id, namespace_id, self.embedding.name, self.embedding.dimensions,
                    json.dumps(self.embedding.embed(candidate.statement), separators=(",", ":")),
                ),
            )
            return memory_id

    def record_run(self, namespace_id: str, run: dict[str, Any]) -> None:
        with self.db.connection:
            self.db.execute(
                """INSERT INTO extraction_runs
                (id, namespace_id, input_hash, provider_name, model_name, prompt_version, schema_version,
                 started_at, completed_at, input_events_json, input_characters, input_tokens, output_tokens,
                 latency_ms, accepted_count, rejected_count, status, error_class, estimated_cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(run.values()),
            )

    def finish_run(self, namespace_id: str, run_id: str, accepted: int, rejected: int, status: str, error_class: str | None = None) -> None:
        with self.db.connection:
            self.db.execute(
                """UPDATE extraction_runs SET completed_at=?, accepted_count=?, rejected_count=?, status=?, error_class=?
                WHERE id=? AND namespace_id=?""",
                (iso(), accepted, rejected, status, error_class, run_id, namespace_id),
            )

    def list_extraction_runs(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM extraction_runs WHERE namespace_id=? ORDER BY rowid DESC LIMIT ? OFFSET ?",
            (namespace_id, limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_extraction_decisions(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM extraction_decisions WHERE namespace_id=? ORDER BY rowid DESC LIMIT ? OFFSET ?",
            (namespace_id, limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_decision(
        self,
        namespace_id: str,
        run_id: str,
        candidate: ExtractionCandidate,
        fingerprint: str,
        validation_status: str,
        reason: str | None,
        action: str,
        memory_id: str | None = None,
        version_id: str | None = None,
    ) -> None:
        with self.db.connection:
            self.db.execute(
                """INSERT OR IGNORE INTO extraction_decisions
                (id, run_id, namespace_id, candidate_fingerprint, kind, subject, statement,
                 validation_status, rejection_reason, action, memory_id, memory_version_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    run_id,
                    namespace_id,
                    fingerprint,
                    candidate.kind,
                    redact_text(candidate.subject),
                    redact_text(candidate.statement),
                    validation_status,
                    redact_text(reason) if reason else None,
                    action,
                    memory_id,
                    version_id,
                    iso(),
                ),
            )

    def reconcile_candidate(self, namespace_id: str, event: sqlite3.Row, candidate: ValidatedCandidate, run_id: str) -> tuple[str | None, str, str | None]:
        """Atomically apply one validated proposal. Returns memory id, action, version id."""
        item = candidate.candidate
        with self.db.lock, self.db.connection:
            source = self.db.execute(
                "SELECT namespace_id, created_at, occurred_at, payload_json FROM events WHERE id=? AND namespace_id=?",
                (str(item.evidence[0].event_id), namespace_id),
            ).fetchone()
            if not source:
                raise ValueError("evidence event is not in the requested namespace")
            if source["created_at"] > iso():
                raise ValueError("evidence postdates derived version")
            memory = self.db.execute(
                "SELECT * FROM memories WHERE namespace_id=? AND kind=? AND subject_key=?",
                (namespace_id, item.kind, item.subject),
            ).fetchone()
            memory_id = memory["id"] if memory else str(uuid.uuid4())
            if item.intent == "ignore":
                return memory_id if memory else None, "IGNORE", None
            if not memory:
                self.db.execute(
                    "INSERT INTO memories(id, namespace_id, kind, subject_key, status, confidence, importance, created_at) "
                    "VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
                    (memory_id, namespace_id, item.kind, item.subject, item.confidence, item.importance, iso()),
                )
            current = self.db.execute(
                "SELECT * FROM memory_versions WHERE memory_id=? AND namespace_id=? ORDER BY version DESC LIMIT 1", (memory_id, namespace_id)
            ).fetchone()
            if current and current["statement"] == item.statement and current["status"] == "active":
                for span in item.evidence:
                    self.db.execute(
                        """INSERT OR IGNORE INTO evidence_refs
                        (id, memory_version_id, namespace_id, event_id, start_offset, end_offset, excerpt)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (str(uuid.uuid4()), current["id"], namespace_id, str(span.event_id), span.start_offset, span.end_offset, span.excerpt),
                    )
                return memory_id, "REINFORCE", current["id"]
            if item.intent == "dispute":
                action = "DISPUTE"
                status = "contradicted"
            elif current and item.intent in {"update", "supersede"}:
                explicit = any(
                    word in item.statement.casefold() or word in span.excerpt.casefold()
                    for span in item.evidence
                    for word in ("correction", "corrected", "replace", "supersede", "instead")
                )
                action = "UPDATE" if item.intent == "update" else "SUPERSEDE"
                if not explicit:
                    action, status = "DISPUTE", "contradicted"
                else:
                    status = "active"
            elif current:
                action, status = "DISPUTE", "contradicted"
            else:
                action, status = "INSERT", "active"
            version = current["version"] + 1 if current else 1
            if current and status == "active":
                self.db.execute(
                    "UPDATE memory_versions SET status='superseded', valid_to=? WHERE id=? AND namespace_id=?", (iso(), current["id"], namespace_id)
                )
                self.db.execute("DELETE FROM memory_fts WHERE memory_version_id=? AND namespace_id=?", (current["id"], namespace_id))
            span = item.evidence[0]
            version_id = str(uuid.uuid4())
            self.db.execute(
                """INSERT INTO memory_versions
                (id, memory_id, namespace_id, source_event_id, evidence_start_offset, evidence_end_offset, evidence_excerpt,
                 version, statement, valid_from, valid_until, recorded_at, status, reason, durability, model_run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    version_id,
                    memory_id,
                    namespace_id,
                    str(span.event_id),
                    span.start_offset,
                    span.end_offset,
                    span.excerpt,
                    version,
                    item.statement,
                    item.valid_from.astimezone(UTC).isoformat() if item.valid_from else source["occurred_at"],
                    item.valid_until.astimezone(UTC).isoformat() if item.valid_until else None,
                    iso(),
                    status,
                    action,
                    item.durability,
                    run_id,
                ),
            )
            for ref in item.evidence:
                self.db.execute(
                    "INSERT INTO evidence_refs(id, memory_version_id, namespace_id, event_id, start_offset, end_offset, excerpt) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), version_id, namespace_id, str(ref.event_id), ref.start_offset, ref.end_offset, ref.excerpt),
                )
            if status == "active":
                self.db.execute(
                    "UPDATE memories SET current_version_id=?, status='active', confidence=?, importance=? WHERE id=? AND namespace_id=?",
                    (version_id, item.confidence, item.importance, memory_id, namespace_id),
                )
                self.db.execute(
                    "INSERT INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?, ?, ?, ?)",
                    (version_id, namespace_id, item.statement, span.excerpt),
                )
            elif status == "contradicted":
                self.db.execute("UPDATE memories SET status='disputed' WHERE id=? AND namespace_id=?", (memory_id, namespace_id))
            self.db.execute(
                "INSERT INTO memory_embeddings(memory_version_id, namespace_id, provider, dimensions, vector_json) VALUES (?, ?, ?, ?, ?)",
                (version_id, namespace_id, self.embedding.name, self.embedding.dimensions,
                 json.dumps(self.embedding.embed(item.statement), separators=(",", ":"))),
            )
            return memory_id, action, version_id

    def get_memory(self, namespace_id: str, memory_id: str) -> MemoryResponse | None:
        row = self.db.execute(
            """SELECT m.*, v.id AS version_id, v.version, v.statement, v.status AS version_status
            FROM memories m JOIN memory_versions v ON v.id=m.current_version_id
            WHERE m.id=? AND m.namespace_id=? AND v.namespace_id=?""",
            (memory_id, namespace_id, namespace_id),
        ).fetchone()
        if not row:
            return None
        citations = self._citations(namespace_id, row["version_id"])
        return MemoryResponse(
            memory_id=uuid.UUID(row["id"]),
            namespace_id=namespace_id,
            kind=row["kind"],
            subject_key=row["subject_key"],
            status=row["version_status"],
            confidence=row["confidence"],
            importance=row["importance"],
            current_version_id=uuid.UUID(row["version_id"]),
            version=row["version"],
            statement=row["statement"],
            citations=citations,
        )

    def list_memories(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[MemoryResponse]:
        rows = self.db.execute(
            """SELECT m.*, v.id AS version_id, v.version, v.statement, v.status AS version_status
            FROM memories m JOIN memory_versions v ON v.id=m.current_version_id
            WHERE m.namespace_id=? AND v.namespace_id=?
            ORDER BY m.created_at, m.id LIMIT ? OFFSET ?""",
            (namespace_id, namespace_id, limit, offset),
        ).fetchall()
        return [
            MemoryResponse(
                memory_id=uuid.UUID(row["id"]), namespace_id=namespace_id, kind=row["kind"], subject_key=row["subject_key"],
                status=row["version_status"], confidence=row["confidence"], importance=row["importance"], current_version_id=uuid.UUID(row["version_id"]),
                version=row["version"], statement=row["statement"], citations=self._citations(namespace_id, row["version_id"]),
            )
            for row in rows
        ]

    def search(self, namespace_id: str, query: str, limit: int, historical: bool = False) -> list[SearchResult]:
        terms = [part for part in query.split() if part.isalnum()]
        lexical: dict[str, float] = {}
        if terms:
            match = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
            if historical:
                lexical_rows = self.db.execute(
                    "SELECT id AS memory_version_id, 0.0 AS score FROM memory_versions WHERE namespace_id=? AND statement LIKE ? LIMIT ?",
                    (namespace_id, f"%{query}%", max(limit * 5, 20)),
                ).fetchall()
            else:
                lexical_rows = self.db.execute(
                    "SELECT memory_version_id, bm25(memory_fts) AS score FROM memory_fts WHERE namespace_id=? AND memory_fts MATCH ? ORDER BY score LIMIT ?",
                    (namespace_id, match, max(limit * 5, 20)),
                ).fetchall()
            maximum = max((abs(float(row["score"])) for row in lexical_rows), default=0.0) or 1.0
            lexical = {row["memory_version_id"]: min(1.0, abs(float(row["score"])) / maximum) for row in lexical_rows}
        query_vector = self.embedding.embed(query)
        vector_rows = self.db.execute(
            """SELECT e.memory_version_id, e.vector_json
            FROM memory_embeddings e JOIN memory_versions v ON v.id=e.memory_version_id AND v.namespace_id=e.namespace_id
            JOIN memories m ON m.id=v.memory_id AND m.namespace_id=v.namespace_id
            WHERE e.namespace_id=? AND (? OR (v.status='active' AND m.status='active' AND v.valid_to IS NULL
              AND (v.valid_until IS NULL OR julianday(v.valid_until) > julianday('now'))))""",
            (namespace_id, historical),
        ).fetchall()
        vector = {row["memory_version_id"]: cosine(query_vector, json.loads(row["vector_json"])) for row in vector_rows}
        vector_candidates = {
            memory_id
            for memory_id, score in vector.items()
            if score >= 0.6 and (memory_id in lexical or score >= 0.7)
        }
        # Embeddings are required for retrieval. FTS5 remains a useful lexical
        # signal, but must not make a memory searchable when dense retrieval
        # did not select it. This prevents the system from silently reverting
        # to lexical-only behavior when the embedding index is incomplete.
        candidate_ids = vector_candidates
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self.db.execute(
            f"""SELECT m.id, m.kind, m.confidence, m.importance, v.id AS version_id, v.statement, v.status
            FROM memory_versions v JOIN memories m ON m.id=v.memory_id AND m.namespace_id=?
            WHERE v.namespace_id=? AND v.id IN ({placeholders}) AND (? OR (v.status='active' AND m.status='active' AND v.valid_to IS NULL
              AND (v.valid_until IS NULL OR julianday(v.valid_until) > julianday('now'))))""",
            (namespace_id, namespace_id, *candidate_ids, historical),
        ).fetchall()
        ranked = sorted(
            rows,
            key=lambda row: (-(0.6 * lexical.get(row["version_id"], 0.0) + 0.4 * vector.get(row["version_id"], 0.0)), row["version_id"]),
        )[:limit]
        query_terms = {term.casefold() for term in query.split()}
        results: list[SearchResult] = []
        for row in ranked:
            citations = self._citations(namespace_id, row["version_id"])
            lexical_score = round(lexical.get(row["version_id"], 0.0), 6)
            vector_score = round(vector.get(row["version_id"], 0.0), 6)
            results.append(
                SearchResult(
                    memory_id=uuid.UUID(row["id"]),
                    memory_version_id=uuid.UUID(row["version_id"]),
                    statement=row["statement"],
                    kind=row["kind"],
                    score=round(0.6 * lexical_score + 0.4 * vector_score, 6),
                    lexical_score=lexical_score,
                    vector_score=vector_score,
                    component_scores={
                        "confidence": round(float(row["confidence"]), 6),
                        "importance": round(float(row["importance"]), 6),
                        "evidence_quality": round(min(1.0, len(citations) / 3), 6),
                        "memory_type_signal": float(row["kind"].casefold() in query_terms),
                        "temporal_signal": float(row["status"] == "active"),
                        "staleness_penalty": float(row["status"] != "active"),
                    },
                    status=row["status"],
                    citations=citations,
                )
            )
        return results

    def _citations(self, namespace_id: str, version_id: str) -> list[EvidenceCitation]:
        rows = self.db.execute(
            """SELECT r.event_id, r.start_offset, r.end_offset, r.excerpt
            FROM evidence_refs r JOIN events e ON e.id=r.event_id
            WHERE r.memory_version_id=? AND r.namespace_id=? AND e.namespace_id=?
            ORDER BY r.id""",
            (version_id, namespace_id, namespace_id),
        ).fetchall()
        return [
            EvidenceCitation(
                event_id=uuid.UUID(row["event_id"]),
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
                excerpt=row["excerpt"],
            )
            for row in rows
        ]

    def memory_count(self, namespace_id: str) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM memories WHERE namespace_id=?", (namespace_id,)).fetchone()[0])

    def list_versions(self, namespace_id: str, memory_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT * FROM memory_versions
            WHERE memory_id=? AND namespace_id=? ORDER BY version""",
            (memory_id, namespace_id),
        ).fetchall()

    def invalidate_memory(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        with self.db.lock, self.db.connection:
            row = self.db.execute(
                "SELECT current_version_id FROM memories WHERE id=? AND namespace_id=?",
                (memory_id, namespace_id),
            ).fetchone()
            if not row:
                return False
            self.db.execute(
                "UPDATE memories SET status='invalidated' WHERE id=? AND namespace_id=?",
                (memory_id, namespace_id),
            )
            self.db.execute(
                "UPDATE memory_versions SET status='invalidated', reason=? WHERE memory_id=? AND namespace_id=?",
                (reason, memory_id, namespace_id),
            )
            self.db.execute(
                "DELETE FROM memory_fts WHERE namespace_id=? AND memory_version_id=?",
                (namespace_id, row["current_version_id"]),
            )
            return True

    def forget_memory(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        """Tombstone a memory while retaining its audit history and evidence."""
        with self.db.lock, self.db.connection:
            row = self.db.execute(
                "SELECT current_version_id FROM memories WHERE id=? AND namespace_id=?",
                (memory_id, namespace_id),
            ).fetchone()
            if not row:
                return False
            self.db.execute("UPDATE memories SET status='deleted' WHERE id=? AND namespace_id=?", (memory_id, namespace_id))
            self.db.execute(
                "UPDATE memory_versions SET status='deleted', reason=? WHERE memory_id=? AND namespace_id=? AND status NOT IN ('deleted', 'invalidated')",
                (reason, memory_id, namespace_id),
            )
            self.db.execute("DELETE FROM memory_fts WHERE namespace_id=? AND memory_version_id=?", (namespace_id, row["current_version_id"]))
            return True

    def restore_memory(self, namespace_id: str, memory_id: str) -> bool:
        """Restore the newest version that was not explicitly invalidated."""
        with self.db.lock, self.db.connection:
            memory = self.db.execute("SELECT id FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)).fetchone()
            version = self.db.execute(
                "SELECT * FROM memory_versions WHERE memory_id=? AND namespace_id=? AND status != 'invalidated' ORDER BY version DESC LIMIT 1",
                (memory_id, namespace_id),
            ).fetchone()
            if not memory or not version:
                return False
            self.db.execute(
                "UPDATE memory_versions SET status='superseded' WHERE memory_id=? AND namespace_id=? AND id<>? AND status='active'",
                (memory_id, namespace_id, version["id"]),
            )
            self.db.execute("UPDATE memory_versions SET status='active', valid_to=NULL, reason='RESTORE' WHERE id=? AND namespace_id=?", (version["id"], namespace_id))
            self.db.execute("UPDATE memories SET status='active', current_version_id=? WHERE id=? AND namespace_id=?", (version["id"], memory_id, namespace_id))
            self.db.execute(
                "INSERT OR REPLACE INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?, ?, ?, ?)",
                (version["id"], namespace_id, version["statement"], version["evidence_excerpt"] or ""),
            )
            return True

    def history(self, namespace_id: str, memory_id: str) -> list[dict[str, Any]] | None:
        memory = self.db.execute(
            "SELECT id FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)
        ).fetchone()
        if not memory:
            return None
        rows = self.list_versions(namespace_id, memory_id)
        return [dict(row) for row in rows]

    def export_namespace(self, namespace_id: str) -> dict[str, Any]:
        tables = (
            "namespaces", "events", "artifacts", "memories", "memory_versions", "evidence_refs", "processing_jobs", "context_requests",
            "extraction_runs", "extraction_decisions", "episodes", "episode_events", "memory_embeddings", "feedback",
        )
        result: dict[str, Any] = {
            "namespaces": [dict(row) for row in self.db.execute("SELECT * FROM namespaces WHERE id=?", (namespace_id,))],
        }
        for table in tables[1:]:
            result[table] = [dict(row) for row in self.db.execute(f"SELECT * FROM {table} WHERE namespace_id=?", (namespace_id,))]
        return result

    def import_namespace(self, document: dict[str, Any], namespace_id: str) -> dict[str, int]:
        """Replay a namespace export without changing IDs or creating cross-scope rows."""
        expected = {"namespaces", "events", "memories", "memory_versions", "evidence_refs", "processing_jobs"}
        if not expected.issubset(document):
            raise ValueError("export is missing required tables")
        namespace_rows = document["namespaces"]
        if not isinstance(namespace_rows, list) or not any(row.get("id") == namespace_id for row in namespace_rows):
            raise ValueError("export namespace does not match requested namespace")
        ordered = (
            "namespaces", "events", "artifacts", "memories", "extraction_runs", "memory_versions", "processing_jobs", "context_requests",
            "evidence_refs", "extraction_decisions", "episodes", "episode_events", "memory_embeddings", "feedback",
        )
        counts: dict[str, int] = {}
        with self.db.lock, self.db.connection:
            for table in ordered:
                rows = document.get(table, [])
                if not isinstance(rows, list):
                    raise ValueError(f"export table {table} must be a list")
                inserted = 0
                for row in rows:
                    if not isinstance(row, dict):
                        raise ValueError(f"export table {table} contains a non-object row")
                    if table != "namespaces" and row.get("namespace_id") != namespace_id:
                        raise ValueError(f"export row in {table} has the wrong namespace")
                    columns = list(row)
                    if not columns:
                        continue
                    placeholders = ",".join("?" for _ in columns)
                    cursor = self.db.execute(
                        f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                        tuple(row[column] for column in columns),
                    )
                    inserted += cursor.rowcount
                counts[table] = inserted
            self._rebuild_fts(namespace_id)
        return counts

    def _rebuild_fts(self, namespace_id: str) -> None:
        self.db.execute("DELETE FROM memory_fts WHERE namespace_id=?", (namespace_id,))
        self.db.execute("DELETE FROM memory_embeddings WHERE namespace_id=?", (namespace_id,))
        rows = self.db.execute(
            """SELECT v.id, v.statement, COALESCE(v.evidence_excerpt, '') AS evidence_excerpt
               FROM memory_versions v JOIN memories m ON m.id=v.memory_id AND m.namespace_id=v.namespace_id
               WHERE v.namespace_id=? AND v.status='active' AND m.status='active' AND v.valid_to IS NULL""",
            (namespace_id,),
        ).fetchall()
        for row in rows:
            self.db.execute(
                "INSERT INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?, ?, ?, ?)",
                (row["id"], namespace_id, row["statement"], row["evidence_excerpt"]),
            )
            self.db.execute(
                "INSERT INTO memory_embeddings(memory_version_id, namespace_id, provider, dimensions, vector_json) VALUES (?, ?, ?, ?, ?)",
                (
                    row["id"], namespace_id, self.embedding.name, self.embedding.dimensions,
                    json.dumps(self.embedding.embed(row["statement"]), separators=(",", ":")),
                ),
            )

    def delete_namespace(self, namespace_id: str) -> bool:
        with self.db.lock, self.db.connection:
            exists = self.db.execute("SELECT 1 FROM namespaces WHERE id=?", (namespace_id,)).fetchone()
            if not exists:
                return False
            self.db.execute("DELETE FROM memory_fts WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM extraction_decisions WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM extraction_runs WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM episode_events WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM episodes WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM memory_embeddings WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM feedback WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM context_requests WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM artifacts WHERE namespace_id=?", (namespace_id,))
            for table in ("evidence_refs", "memory_versions", "memories", "processing_jobs", "events"):
                self.db.execute(f"DELETE FROM {table} WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM namespaces WHERE id=?", (namespace_id,))
            return True

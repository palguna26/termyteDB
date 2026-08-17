from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from .db import Database
from .errors import IdempotencyConflict
from .extractor import Candidate
from .schemas import EventInput, EvidenceCitation, MemoryResponse, SearchResult


def iso(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()


def stable_uuid(namespace_id: str, idempotency_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"termytedb:event:{namespace_id}:{idempotency_key}"))


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_event_content(event: EventInput, redacted_payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "type": event.type,
            "stream_id": event.stream_id,
            "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
            "payload": redacted_payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class Repository:
    def __init__(self, database: Database):
        self.db = database

    def ensure_namespace(self, namespace_id: str, org_id: str = "default") -> None:
        with self.db.connection:
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
                    (id, namespace_id, stream_id, idempotency_hash, type, occurred_at, payload_json,
                     content_hash, redaction_state, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'redacted', ?)""",
                    (
                        event_id,
                        namespace_id,
                        event.stream_id,
                        idempotency_hash,
                        event.type,
                        occurred,
                        payload_json,
                        content_hash,
                        iso(),
                    ),
                )
                self.db.execute(
                    """INSERT INTO processing_jobs
                    (id, namespace_id, event_id, input_hash, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                    (job_id, namespace_id, event_id, content_hash, iso(), iso()),
                )
            return event_id, False, content_hash, job_id

    def claim_jobs(self, namespace_id: str, limit: int, lease_seconds: int) -> list[sqlite3.Row]:
        # SQLite computes the lease consistently and avoids client clock formatting issues.
        with self.db.connection:
            rows = self.db.execute(
                """SELECT * FROM processing_jobs
                WHERE namespace_id = ? AND (status IN ('pending', 'failed') OR
                    (status = 'processing' AND lease_until < datetime('now')))
                ORDER BY created_at, id LIMIT ?""",
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
                "UPDATE processing_jobs SET status='completed', lease_until=NULL, updated_at=? WHERE id=? AND namespace_id=?",
                (iso(), job_id, namespace_id),
            )

    def fail_job(self, namespace_id: str, job_id: str, error: str) -> str:
        with self.db.connection:
            row = self.db.execute(
                "SELECT attempts, max_attempts FROM processing_jobs WHERE id=? AND namespace_id=?",
                (job_id, namespace_id),
            ).fetchone()
            if not row:
                raise KeyError(job_id)
            status = "dead" if row["attempts"] >= row["max_attempts"] else "failed"
            self.db.execute(
                "UPDATE processing_jobs SET status=?, lease_until=NULL, last_error=?, updated_at=? WHERE id=? AND namespace_id=?",
                (status, error, iso(), job_id, namespace_id),
            )
            return status

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
            return memory_id

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
            current_version_id=uuid.UUID(row["version_id"]),
            version=row["version"],
            statement=row["statement"],
            citations=citations,
        )

    def search(self, namespace_id: str, query: str, limit: int) -> list[SearchResult]:
        terms = [part for part in query.split() if part.isalnum()]
        if not terms:
            return []
        match = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        rows = self.db.execute(
            """SELECT m.id, m.kind, v.id AS version_id, v.statement, v.status, bm25(memory_fts) AS score
            FROM memory_fts JOIN memory_versions v ON v.id=memory_fts.memory_version_id
            JOIN memories m ON m.id=v.memory_id AND m.namespace_id=?
            WHERE memory_fts.namespace_id=? AND memory_fts MATCH ?
              AND v.namespace_id=? AND v.status='active' AND m.status='active' AND v.valid_to IS NULL
            ORDER BY score, v.id LIMIT ?""",
            (namespace_id, namespace_id, match, namespace_id, limit),
        ).fetchall()
        return [
            SearchResult(
                memory_id=uuid.UUID(row["id"]),
                memory_version_id=uuid.UUID(row["version_id"]),
                statement=row["statement"],
                kind=row["kind"],
                score=float(-row["score"]),
                status=row["status"],
                citations=self._citations(namespace_id, row["version_id"]),
            )
            for row in rows
        ]

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

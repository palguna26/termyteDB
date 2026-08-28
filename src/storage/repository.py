from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ..core.errors import IdempotencyConflict
from ..core.redaction import redact_text
from ..memory.encoding import score_observation
from ..memory.extraction import CandidateRejected, ValidatedCandidate
from ..memory.extractor import Candidate, payload_text
from ..memory.provider import SessionSummaryProvider
from ..models import EventInput, EvidenceCitation, ExtractionCandidate, MemoryResponse, SearchResult
from ..retrieval.embedding import EmbeddingProvider, FastEmbedProvider, batch_dot, pack_embedding
from .db import Database
from .vector_index import SQLiteVecIndex

TRANSITION_MARKERS = (
    "correction",
    "corrected",
    "replace",
    "supersede",
    "instead",
    "switched",
    "moved",
    "migrated",
    "changed",
    "no longer",
    "now using",
    "updated",
    "prefer",
    "renamed",
    "deprecated",
    "stopped",
)
SEARCH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "with",
}


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
        self.vector_index = SQLiteVecIndex(self.db, self.embedding)
        self.vector_index.ensure()
        self._claimed_lease_tokens: dict[str, str] = {}

    def _persist_embedding(self, memory_version_id: str, namespace_id: str, vector: list[float] | None, statement: str) -> None:
        packed = pack_embedding(vector if vector is not None else self.embedding.embed(statement))
        self.db.execute(
            "INSERT INTO memory_embeddings(memory_version_id, namespace_id, provider, dimensions, vector) VALUES (?, ?, ?, ?, ?)",
            (memory_version_id, namespace_id, self.embedding.name, self.embedding.dimensions, packed),
        )
        self.vector_index.upsert_row(memory_version_id, namespace_id, packed)

    def embed_many(self, values: list[str]) -> list[list[float]]:
        embed_many = getattr(self.embedding, "embed_many", None)
        if callable(embed_many):
            return cast(list[list[float]], embed_many(values))
        return [self.embedding.embed(value) for value in values]

    def ensure_namespace(self, namespace_id: str, org_id: str = "default") -> None:
        with self.db.lock, self.db.connection:
            self.db.execute(
                "INSERT OR IGNORE INTO namespaces(id, org_id, created_at) VALUES (?, ?, ?)",
                (namespace_id, org_id, iso()),
            )

    def ingest(self, namespace_id: str, event: EventInput, redacted_payload: dict[str, Any]) -> tuple[str, bool, str]:
        if event.namespace_id != namespace_id:
            raise ValueError("event namespace does not match repository namespace")
        with self.db.lock:
            self.ensure_namespace(namespace_id)
            event_id = stable_uuid(namespace_id, event.idempotency_key)
            idempotency_hash = hash_text(event.idempotency_key)
            payload_json = json.dumps(redacted_payload, sort_keys=True, separators=(",", ":"))
            content_hash = hash_text(canonical_event_content(event, redacted_payload))
            occurred = iso(event.occurred_at)
            observation_text = payload_text({**redacted_payload, "__termytedb_event_type": event.type})
            prior_count = (
                int(
                    self.db.execute(
                        "SELECT COUNT(*) FROM events WHERE namespace_id=? AND stream_id IS ? AND payload_json LIKE ?",
                        (namespace_id, event.stream_id, f"%{observation_text[:80]}%"),
                    ).fetchone()[0]
                )
                if observation_text
                else 0
            )
            encoding = score_observation(observation_text, repeated=min(1.0, prior_count / 3))
            with self.db.connection:
                existing = self.db.execute(
                    "SELECT id, content_hash FROM events WHERE namespace_id = ? AND idempotency_hash = ?",
                    (namespace_id, idempotency_hash),
                ).fetchone()
                if existing:
                    if existing["content_hash"] != content_hash:
                        raise IdempotencyConflict("idempotency key is already used for different content")
                    return existing["id"], True, existing["content_hash"]
                self.db.execute(
                    """INSERT INTO events
                    (id, namespace_id, protocol_version, stream_id, actor_id, agent_id, session_id, source_id, idempotency_hash, type, occurred_at,
                     payload_json, content_hash, redaction_state, created_at, observation_hash, importance_score, encoding_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'redacted', ?, ?, ?, ?)""",
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
                        content_hash,
                        encoding.importance_score,
                        encoding.reason,
                    ),
                )
                for artifact in event.artifacts:
                    safe_uri = safe_artifact_uri(artifact.uri, artifact.content_hash)
                    safe_metadata = {key: redact_text(value) for key, value in artifact.metadata.items()}
                    self.db.execute(
                        """INSERT INTO artifacts(id, namespace_id, event_id, media_type, size_bytes, uri, content_hash, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid.uuid4()),
                            namespace_id,
                            event_id,
                            artifact.media_type,
                            artifact.size_bytes,
                            safe_uri,
                            artifact.content_hash,
                            json.dumps(safe_metadata, sort_keys=True, separators=(",", ":")),
                        ),
                    )
                episode_id = self._assign_episode(namespace_id, event_id, event.stream_id, occurred)
                ordinal = int(self.db.execute("SELECT ordinal FROM episode_events WHERE episode_id=? AND event_id=?", (episode_id, event_id)).fetchone()[0])
                self.db.execute(
                    "UPDATE events SET episode_id=?, sequence_number=? WHERE id=? AND namespace_id=?",
                    (episode_id, ordinal, event_id, namespace_id),
                )
                self.db.execute(
                    """INSERT INTO encoding_decisions
                    (id, namespace_id, event_id, importance_score, novelty, surprise, task_relevance,
                     repetition, outcome_signal, correction_signal, future_use, privacy_penalty, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        namespace_id,
                        event_id,
                        encoding.importance_score,
                        encoding.novelty,
                        encoding.surprise,
                        encoding.task_relevance,
                        encoding.repetition,
                        encoding.outcome_signal,
                        encoding.correction_signal,
                        encoding.future_use,
                        encoding.privacy_penalty,
                        encoding.reason,
                        iso(),
                    ),
                )
            return event_id, False, content_hash

    def events_by_id(self, namespace_id: str, event_ids: list[str]) -> list[sqlite3.Row]:
        if not event_ids:
            return []
        placeholders = ",".join("?" for _ in event_ids)
        rows = self.db.execute(
            f"SELECT * FROM events WHERE namespace_id=? AND id IN ({placeholders}) ORDER BY created_at, sequence_number, id",
            (namespace_id, *event_ids),
        ).fetchall()
        by_id = {str(row["id"]): row for row in rows}
        return [by_id[event_id] for event_id in event_ids if event_id in by_id]

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
        rows = self.db.execute("SELECT * FROM episodes WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?", (namespace_id, limit, offset)).fetchall()
        return [dict(row) for row in rows]

    def update_episode(self, namespace_id: str, episode_id: str, status: str, summary: str | None) -> bool:
        with self.db.lock, self.db.connection:
            cursor = self.db.execute(
                "UPDATE episodes SET status=?, summary=COALESCE(?, summary), updated_at=? WHERE id=? AND namespace_id=?",
                (status, redact_text(summary) if summary else None, iso(), episode_id, namespace_id),
            )
            return cursor.rowcount == 1

    def set_episode_summary(self, namespace_id: str, episode_id: str, summary: str | None) -> bool:
        with self.db.lock, self.db.connection:
            cursor = self.db.execute(
                "UPDATE episodes SET summary=?, updated_at=? WHERE id=? AND namespace_id=?",
                (redact_text(summary) if summary else None, iso(), episode_id, namespace_id),
            )
            return cursor.rowcount == 1

    def refresh_episode_summary(
        self,
        namespace_id: str,
        episode_id: str,
        *,
        limit: int = 8,
        summary_provider: SessionSummaryProvider | None = None,
    ) -> str | None:
        rows = self.db.execute(
            """SELECT e.* FROM episode_events ee JOIN events e ON e.id=ee.event_id
            WHERE ee.namespace_id=? AND ee.episode_id=? ORDER BY ee.ordinal LIMIT ?""",
            (namespace_id, episode_id, max(1, min(limit, 20))),
        ).fetchall()
        if not rows:
            return None
        snippets: list[str] = []
        seen: set[str] = set()
        for row in rows:
            text = payload_text({**json.loads(row["payload_json"]), "__termytedb_event_type": row["type"]})
            text = " ".join(text.split())
            if not text:
                continue
            short = text[:180]
            key = short.casefold()
            if key in seen:
                continue
            seen.add(key)
            role = str(row["actor_id"] or row["type"] or "event")
            snippets.append(f"{role}: {short}")
            if len(snippets) >= 4:
                break
        base_text = "\n".join(snippets)
        summary: str | None = None
        if summary_provider is not None:
            try:
                summary = summary_provider.summarize(base_text, namespace_id=namespace_id, episode_id=episode_id)
            except Exception:
                summary = None
        if not summary:
            summary = " | ".join(snippets) if snippets else None
        if summary:
            self.set_episode_summary(namespace_id, episode_id, summary)
        return summary

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
        rows = self.db.execute("SELECT * FROM feedback WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?", (namespace_id, limit, offset)).fetchall()
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
            for item in self.db.execute("SELECT * FROM evidence_refs WHERE event_id=? AND namespace_id=? ORDER BY id", (event_id, namespace_id)).fetchall()
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

    def related_memory_context(self, namespace_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return bounded current memory context for extraction reconciliation.

        This deliberately reuses the normal namespace-safe search path. The result is
        comparison context for the provider; evidence for new claims must still come
        from the current input events.
        Keeps input tokens ~800 (5*~120) vs 20*120=2400 to avoid output>input.
        """
        results = self.search(namespace_id, query, max(1, min(limit, 5)), historical=False, internal=True)
        return [
            {
                "ref": f"m{index}",
                "memory_id": str(item.memory_id),
                "memory_version_id": str(item.memory_version_id),
                "kind": item.kind,
                "subject_key": str(
                    self.db.execute(
                        "SELECT subject_key FROM memories WHERE id=? AND namespace_id=?",
                        (str(item.memory_id), namespace_id),
                    ).fetchone()["subject_key"]
                ),
                "statement": item.statement,
                "status": item.status,
            }
            for index, item in enumerate(results)
        ]

    def extraction_window(self, namespace_id: str, event_id: str, *, limit: int = 4) -> dict[str, str]:
        """Return a small chronological same-session evidence window for extraction.

        The current event is always included. If the event has a stream_id or session_id,
        we also include the most recent earlier events from the same stream/session so
        extraction can resolve pronouns, updates, and temporal qualifiers without
        losing evidence grounding.
        """
        current = self.get_event(namespace_id, event_id)
        if current is None:
            return {}
        scope_stream = current.get("stream_id")
        scope_session = current.get("session_id")
        current_sequence = current.get("sequence_number")
        params: list[Any] = [namespace_id]
        where = ["namespace_id = ?"]
        if scope_stream is not None:
            where.append("stream_id = ?")
            params.append(scope_stream)
        elif scope_session is not None:
            where.append("session_id = ?")
            params.append(scope_session)
        else:
            return {str(current["id"]): payload_text(current["payload_json"], current["type"])}
        if current_sequence is None:
            params.extend([current["occurred_at"], current["occurred_at"], current["id"], max(1, limit)])
            rows = self.db.execute(
                f"""SELECT id, type, payload_json
                FROM events
                WHERE {" AND ".join(where)}
                  AND (occurred_at < ? OR (occurred_at = ? AND id <= ?))
                ORDER BY occurred_at DESC, id DESC
                LIMIT ?""",
                tuple(params),
            ).fetchall()
        else:
            params.extend([current["occurred_at"], current["occurred_at"], current_sequence, current_sequence, current["id"], max(1, limit)])
            rows = self.db.execute(
                f"""SELECT id, type, payload_json
                FROM events
                WHERE {" AND ".join(where)}
                  AND (
                    occurred_at < ?
                    OR (occurred_at = ? AND (
                      COALESCE(sequence_number, -1) < ?
                      OR (COALESCE(sequence_number, -1) = ? AND id <= ?)
                    ))
                  )
                ORDER BY occurred_at DESC, COALESCE(sequence_number, -1) DESC, id DESC
                LIMIT ?""",
                tuple(params),
            ).fetchall()
        window: dict[str, str] = {}
        for row in reversed(rows):
            window[str(row["id"])] = payload_text(json.loads(row["payload_json"]), row["type"])
        if str(current["id"]) not in window:
            window[str(current["id"])] = payload_text(current["payload_json"], current["type"])
        return window

    def record_context_request(self, namespace_id: str, query: str, token_budget: int, response: Any) -> str:
        request_id = str(uuid.uuid4())
        with self.db.lock, self.db.connection:
            self.db.execute(
                """INSERT INTO context_requests
                (id, namespace_id, query, token_budget, selected_json, token_count, abstained, diagnostics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    namespace_id,
                    query,
                    token_budget,
                    json.dumps([str(item.memory_version_id) for item in response.results], separators=(",", ":")),
                    response.token_count,
                    int(response.abstained),
                    json.dumps(response.diagnostics, sort_keys=True, separators=(",", ":")),
                    iso(),
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

    def claim_jobs(self, namespace_id: str, limit: int, lease_seconds: int = 180) -> list[sqlite3.Row]:
        # SQLite computes the lease consistently and avoids client clock formatting issues.
        with self.db.lock:
            try:
                self.db.connection.execute("BEGIN IMMEDIATE")
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
                    lease_token = str(uuid.uuid4())
                    self.db.execute(
                        """UPDATE processing_jobs SET status='processing', attempts=attempts+1,
                        lease_until=datetime('now', ?), lease_token=?, updated_at=datetime('now')
                        WHERE id=? AND namespace_id=?""",
                        (f"+{lease_seconds} seconds", lease_token, row["id"], namespace_id),
                    )
                    claimed_row = self.db.execute(
                        "SELECT * FROM processing_jobs WHERE id=? AND namespace_id=?",
                        (row["id"], namespace_id),
                    ).fetchone()
                    if claimed_row:
                        self._claimed_lease_tokens[str(row["id"])] = lease_token
                        claimed.append(claimed_row)
                self.db.connection.commit()
                return claimed
            except Exception:
                self.db.connection.rollback()
                raise

    def complete_job(self, namespace_id: str, job_id: str, lease_token: str | None = None) -> bool:
        effective_token = lease_token or self._claimed_lease_tokens.get(str(job_id))
        with self.db.connection:
            cursor = self.db.execute(
                """UPDATE processing_jobs SET status='completed', lease_until=NULL, lease_token=NULL, updated_at=?
                WHERE id=? AND namespace_id=? AND status='processing'
                  AND (? IS NULL OR lease_token=?)
                  AND (lease_until IS NULL OR lease_until >= datetime('now'))""",
                (iso(), job_id, namespace_id, effective_token, effective_token),
            )
            if cursor.rowcount == 1:
                self._claimed_lease_tokens.pop(str(job_id), None)
            return cursor.rowcount == 1

    def heartbeat_job(self, namespace_id: str, job_id: str, lease_seconds: int, lease_token: str | None = None) -> bool:
        """Extend an active lease without reviving cancelled or completed work."""
        with self.db.lock, self.db.connection:
            cursor = self.db.execute(
                """UPDATE processing_jobs
                SET lease_until=datetime('now', ?), updated_at=?
                WHERE id=? AND namespace_id=? AND status='processing'
                  AND (? IS NULL OR lease_token=?)
                  AND (lease_until IS NULL OR lease_until >= datetime('now'))""",
                (f"+{lease_seconds} seconds", iso(), job_id, namespace_id, lease_token, lease_token),
            )
            return cursor.rowcount == 1

    def fail_job(
        self,
        namespace_id: str,
        job_id: str,
        error: str,
        *,
        retryable: bool = True,
        lease_token: str | None = None,
    ) -> str:
        with self.db.connection:
            row = self.db.execute(
                "SELECT status, attempts, max_attempts FROM processing_jobs WHERE id=? AND namespace_id=?",
                (job_id, namespace_id),
            ).fetchone()
            if not row:
                raise KeyError(job_id)
            if row["status"] == "cancelled":
                return "cancelled"
            if lease_token is not None:
                owner = self.db.execute(
                    """SELECT 1 FROM processing_jobs WHERE id=? AND namespace_id=? AND status='processing'
                    AND lease_token=? AND (lease_until IS NULL OR lease_until >= datetime('now'))""",
                    (job_id, namespace_id, lease_token),
                ).fetchone()
                if not owner:
                    return "stale"
            status = "dead" if not retryable or row["attempts"] >= row["max_attempts"] else "failed"
            delay = min(300, 2 ** max(1, int(row["attempts"])))
            self.db.execute(
                """UPDATE processing_jobs
                SET status=?, lease_until=NULL, lease_token=NULL, next_attempt_at=?, last_error=?, updated_at=?
                WHERE id=? AND namespace_id=?""",
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
                "UPDATE processing_jobs SET status='cancelled', lease_until=NULL, lease_token=NULL, updated_at=? "
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

    def save_candidate(
        self,
        namespace_id: str,
        event: sqlite3.Row,
        candidate: Candidate,
        embedding: list[float] | None = None,
        *,
        job_id: str | None = None,
        lease_token: str | None = None,
    ) -> str:
        now = iso()
        with self.db.lock, self.db.connection:
            self.db.connection.execute("BEGIN IMMEDIATE")
            if job_id is not None and lease_token is not None:
                owner = self.db.execute(
                    """SELECT 1 FROM processing_jobs WHERE id=? AND namespace_id=? AND status='processing'
                    AND lease_token=? AND lease_until >= datetime('now')""",
                    (job_id, namespace_id, lease_token),
                ).fetchone()
                if not owner:
                    raise RuntimeError("job lease is no longer active")
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
            previous_version_id = (
                str(current["id"])
                if (
                    current := self.db.execute(
                        "SELECT * FROM memory_versions WHERE memory_id=? AND namespace_id=? ORDER BY version DESC LIMIT 1",
                        (memory_id, namespace_id),
                    ).fetchone()
                )
                else None
            )
            if not memory:
                self.db.execute(
                    "INSERT INTO memories(id, namespace_id, kind, subject_key, status, confidence, created_at) VALUES (?, ?, ?, ?, 'active', 1.0, ?)",
                    (memory_id, namespace_id, candidate.kind, candidate.subject_key, now),
                )
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
            self._persist_embedding(version_id, namespace_id, embedding, candidate.statement)
            self.record_graph_links(
                namespace_id,
                episode_id=str(event["episode_id"]) if event["episode_id"] else None,
                memory_version_id=version_id,
                memory_id=memory_id,
                subject_key=candidate.subject_key,
                statement=candidate.statement,
                predicate="updates" if previous_version_id else "contains",
                previous_version_id=previous_version_id,
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

    def reconcile_candidate(
        self,
        namespace_id: str,
        event: sqlite3.Row,
        candidate: ValidatedCandidate,
        run_id: str,
        embedding: list[float] | None = None,
        *,
        job_id: str | None = None,
        lease_token: str | None = None,
    ) -> tuple[str | None, str, str | None]:
        """Atomically apply one validated proposal. Returns memory id, action, version id."""
        with self.db.lock:
            self.db.connection.execute("BEGIN IMMEDIATE")
            try:
                if job_id is not None and lease_token is not None:
                    owner = self.db.execute(
                        """SELECT 1 FROM processing_jobs WHERE id=? AND namespace_id=? AND status='processing'
                        AND lease_token=? AND lease_until >= datetime('now')""",
                        (job_id, namespace_id, lease_token),
                    ).fetchone()
                    if not owner:
                        raise RuntimeError("job lease is no longer active")
                result = self._reconcile_candidate_in_transaction(namespace_id, event, candidate, run_id, embedding)
                self.db.connection.commit()
                return result
            except Exception:
                self.db.connection.rollback()
                raise

    def _reconcile_candidate_in_transaction(
        self,
        namespace_id: str,
        event: sqlite3.Row,
        candidate: ValidatedCandidate,
        run_id: str,
        embedding: list[float] | None,
    ) -> tuple[str | None, str, str | None]:
        item = candidate.candidate
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
        if item.existing_memory_id is not None and (not memory or str(item.existing_memory_id) != memory_id):
            raise CandidateRejected("existing_memory_identity_mismatch")
        if item.intent == "ignore":
            return memory_id if memory else None, "IGNORE", None
        if not memory:
            self.db.execute(
                "INSERT INTO memories(id, namespace_id, kind, subject_key, status, confidence, importance, created_at) VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
                (memory_id, namespace_id, item.kind, item.subject, item.confidence, item.importance, iso()),
            )
        current = self.db.execute(
            "SELECT * FROM memory_versions WHERE memory_id=? AND namespace_id=? ORDER BY version DESC LIMIT 1", (memory_id, namespace_id)
        ).fetchone()
        previous_version_id = str(current["id"]) if current else None
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
            explicit = item.confidence >= 0.85 or any(
                marker in item.statement.casefold() or marker in span.excerpt.casefold() for span in item.evidence for marker in TRANSITION_MARKERS
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
            self.db.execute("UPDATE memory_versions SET status='superseded', valid_to=? WHERE id=? AND namespace_id=?", (iso(), current["id"], namespace_id))
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
        self._persist_embedding(version_id, namespace_id, embedding, item.statement)
        self.record_graph_links(
            namespace_id,
            episode_id=str(event["episode_id"]) if event["episode_id"] else None,
            memory_version_id=version_id,
            memory_id=memory_id,
            subject_key=item.subject,
            statement=item.statement,
            predicate="updates" if previous_version_id else "contains",
            previous_version_id=previous_version_id,
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
                citations=self._citations(namespace_id, row["version_id"]),
            )
            for row in rows
        ]

    def search(self, namespace_id: str, query: str, limit: int, historical: bool = False, *, internal: bool = False) -> list[SearchResult]:
        terms = list(dict.fromkeys(term.casefold() for term in re.findall(r"[\w./:-]+", query) if len(term) > 1 and term.casefold() not in SEARCH_STOP_WORDS))
        prefer_oldest = historical and bool(re.search(r"\b(first|earliest|initial|original|before|used to|previous|previously|former|formerly)\b", query, re.I))
        lexical: dict[str, float] = {}
        lexical_rank: dict[str, int] = {}
        if terms:
            match = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
            if historical:
                lexical_rows = self.db.execute(
                    """SELECT id AS memory_version_id,
                    CASE WHEN lower(statement) LIKE ? THEN 0 ELSE 1 END AS score
                    FROM memory_versions WHERE namespace_id=? AND ("""
                    + " OR ".join("lower(statement) LIKE ?" for _ in terms)
                    + ") ORDER BY score, recorded_at DESC, id LIMIT ?",
                    (f"%{query.casefold()}%", namespace_id, *(f"%{term}%" for term in terms), max(limit * 5, 20)),
                ).fetchall()
            else:
                lexical_rows = self.db.execute(
                    "SELECT memory_version_id, bm25(memory_fts) AS score FROM memory_fts WHERE namespace_id=? AND memory_fts MATCH ? ORDER BY score LIMIT ?",
                    (namespace_id, match, max(limit * 5, 20)),
                ).fetchall()
            lexical_rank = {str(row["memory_version_id"]): index for index, row in enumerate(lexical_rows, start=1)}
            lexical = {memory_id: 1.0 / (1.0 + 0.08 * (rank - 1)) for memory_id, rank in lexical_rank.items()}

        vector: dict[str, float] = {}
        vector_rank: dict[str, int] = {}
        query_vector: list[float] | None = None
        try:
            query_vector = self.embedding.embed(query)
            indexed_rows = self.vector_index.search(namespace_id, query_vector, max(limit * 5, 20))
            if indexed_rows:
                ordered_vector = [(memory_id, score) for memory_id, score in indexed_rows if score >= 0.6]
                vector_rank = {memory_id: index for index, (memory_id, score) in enumerate(ordered_vector, start=1)}
                vector = {memory_id: score for memory_id, score in ordered_vector}
        except Exception:
            query_vector = None
        if not vector and query_vector is not None:
            vector_rows = self.db.execute(
                """SELECT e.memory_version_id, e.vector
                FROM memory_embeddings e JOIN memory_versions v ON v.id=e.memory_version_id AND v.namespace_id=e.namespace_id
                JOIN memories m ON m.id=v.memory_id AND m.namespace_id=v.namespace_id
                WHERE e.namespace_id=? AND e.provider=? AND e.dimensions=? AND (? OR (v.status='active' AND m.status='active'
                  AND m.accessibility >= 0.05 AND v.valid_to IS NULL
                  AND (v.valid_from IS NULL OR julianday(v.valid_from) <= julianday('now'))
                  AND (v.valid_until IS NULL OR julianday(v.valid_until) > julianday('now'))))""",
                (namespace_id, self.embedding.name, self.embedding.dimensions, historical),
            ).fetchall()
            if vector_rows:
                try:
                    scores = batch_dot(query_vector, [bytes(row["vector"]) for row in vector_rows], self.embedding.dimensions)
                    raw_vector = {str(row["memory_version_id"]): float(score) for row, score in zip(vector_rows, scores, strict=True)}
                    ordered_vector = sorted(raw_vector.items(), key=lambda item: (-item[1], item[0]))
                    vector_rank = {memory_id: index for index, (memory_id, score) in enumerate(ordered_vector, start=1) if score >= 0.6}
                    vector = {memory_id: score for memory_id, score in raw_vector.items() if memory_id in vector_rank}
                except Exception:
                    # Lexical retrieval is the complete local fallback. Embedding
                    # failures must not make already indexed evidence disappear.
                    vector = {}
                    vector_rank = {}

        # Keep semantic hits above the search floor in the candidate pool. The
        # reranker and evidence scores decide the final order; dropping 0.60-
        # 0.70 matches here loses paraphrased temporal and preference answers.
        candidate_ids = set(lexical) | {memory_id for memory_id, score in vector.items() if score >= 0.6}
        alias_rank: dict[str, int] = {}
        alias_terms = [term for term in terms if len(term) > 2]
        if alias_terms:
            alias_match = " OR ".join("lower(ea.alias) LIKE ?" for _ in alias_terms)
            alias_rows = self.db.execute(
                f"""SELECT DISTINCT r.memory_version_id, MIN(ea.alias) AS alias
                FROM relationships r
                JOIN entity_aliases ea ON ea.entity_id IN (r.subject_entity_id, r.object_entity_id)
                WHERE r.namespace_id=? AND ea.namespace_id=? AND r.memory_version_id IS NOT NULL
                  AND ({alias_match})
                GROUP BY r.memory_version_id
                ORDER BY r.memory_version_id""",
                (namespace_id, namespace_id, *(f"%{term}%" for term in alias_terms)),
            ).fetchall()
            alias_rank = {str(row["memory_version_id"]): index for index, row in enumerate(alias_rows, start=1)}
            candidate_ids |= set(alias_rank)
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self.db.execute(
            f"""SELECT m.id, m.kind, m.confidence, m.importance, v.id AS version_id, v.statement, v.status, v.valid_from, v.valid_to, v.recorded_at
            FROM memory_versions v JOIN memories m ON m.id=v.memory_id AND m.namespace_id=?
            WHERE v.namespace_id=? AND v.id IN ({placeholders}) AND (? OR (v.status='active' AND m.status='active'
              AND m.accessibility >= 0.05 AND v.valid_to IS NULL
              AND (v.valid_from IS NULL OR julianday(v.valid_from) <= julianday('now'))
              AND (v.valid_until IS NULL OR julianday(v.valid_until) > julianday('now'))))""",
            (namespace_id, namespace_id, *candidate_ids, historical),
        ).fetchall()
        graph_cache: dict[str, float] = {}
        summary_cache: dict[str, float] = {}

        def graph_signal(version_id: str) -> float:
            cached = graph_cache.get(version_id)
            if cached is not None:
                return cached
            score = 0.0
            rels = self.db.execute(
                """SELECT s.label AS subject_label, s_alias.alias AS subject_alias, o.label AS object_label, o_alias.alias AS object_alias, r.predicate
                FROM relationships r
                JOIN entities s ON s.id=r.subject_entity_id
                JOIN entities o ON o.id=r.object_entity_id
                LEFT JOIN entity_aliases s_alias ON s_alias.entity_id=s.id
                LEFT JOIN entity_aliases o_alias ON o_alias.entity_id=o.id
                WHERE r.namespace_id=? AND r.memory_version_id=? AND r.status='active'""",
                (namespace_id, version_id),
            ).fetchall()
            lowered_terms = [term.casefold() for term in terms]
            for rel in rels:
                labels = [
                    str(rel["subject_label"] or ""),
                    str(rel["object_label"] or ""),
                    str(rel["subject_alias"] or ""),
                    str(rel["object_alias"] or ""),
                ]
                if any(term and any(term in label.casefold() for label in labels) for term in lowered_terms):
                    score = 0.015
                    break
                if rel["predicate"] in {"updates", "supersedes", "contradicts", "contains"}:
                    score = max(score, 0.01)
            graph_cache[version_id] = score
            return score

        def summary_signal(version_id: str) -> float:
            cached = summary_cache.get(version_id)
            if cached is not None:
                return cached
            row = self.db.execute(
                """SELECT ep.summary
                FROM memory_versions v
                JOIN events e ON e.id=v.source_event_id AND e.namespace_id=v.namespace_id
                JOIN episodes ep ON ep.id=e.episode_id AND ep.namespace_id=v.namespace_id
                WHERE v.id=? AND v.namespace_id=?""",
                (version_id, namespace_id),
            ).fetchone()
            if not row or not row["summary"]:
                summary_cache[version_id] = 0.0
                return 0.0
            summary_text = str(row["summary"]).casefold()
            score = 0.012 if any(term in summary_text for term in lowered_terms) else 0.0
            summary_cache[version_id] = score
            return score

        lowered_terms = [term.casefold() for term in terms]
        query_years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", query)]

        def score_row(row: sqlite3.Row) -> float:
            version_id = str(row["version_id"])
            lexical_rrf = 1.0 / (60 + lexical_rank[version_id]) if version_id in lexical_rank else 0.0
            vector_rrf = 1.0 / (60 + vector_rank[version_id]) if version_id in vector_rank else 0.0
            alias_rrf = 1.0 / (60 + alias_rank[version_id]) if version_id in alias_rank else 0.0
            channel_count = sum((version_id in lexical_rank, version_id in vector_rank, version_id in alias_rank))
            fused = 60 * (lexical_rrf + vector_rrf + alias_rrf) / max(1, channel_count)
            exact = 0.08 if query.casefold() in str(row["statement"]).casefold() else 0.0
            support = min(0.04, 0.01 * len(self._citations(namespace_id, version_id)))
            quality = 0.03 * float(row["confidence"]) + 0.02 * float(row["importance"])
            graph = graph_signal(version_id)
            summary = summary_signal(version_id)
            history = 0.0
            if historical and row["valid_to"] is not None:
                history = 0.02 if prefer_oldest else 0.01
            temporal = 0.0
            if query_years:
                valid_from = str(row["valid_from"] or "")
                valid_to = str(row["valid_to"] or "")
                if any(str(year) in valid_from or str(year) in valid_to for year in query_years):
                    temporal = 0.05
            # Small recency tie-breaker inspired by graphiti temporal handling:
            # when scores are close, prefer the more recent valid_from for ordinary
            # agent recency. Bounded to 0.02 to avoid overriding lexical/semantic.
            recency = 0.0
            try:
                if row["valid_from"]:
                    # Use ISO string ordering as proxy for recency without extra parsing;
                    # newer ISO strings are lexicographically larger.
                    recency = 0.01  # placeholder for future decay; kept small to avoid gaming
            except Exception:
                recency = 0.0
            return min(1.0, fused + exact + support + quality + graph + summary + history + temporal + recency)

        # Primary sort by score, secondary by valid_from. Historical queries that
        # ask for "first" or "before" should prefer older evidence instead of the
        # default newest-first tie break.
        rows_by_recency = sorted(rows, key=lambda row: row["valid_from"] or "", reverse=not prefer_oldest)
        ranked = sorted(rows_by_recency, key=lambda row: -score_row(row))[:limit]
        query_terms = set(terms)
        results: list[SearchResult] = []
        for row in ranked:
            citations = self._citations(namespace_id, row["version_id"])
            lexical_score = round(lexical.get(row["version_id"], 0.0), 6)
            vector_score = round(vector.get(row["version_id"], 0.0), 6)
            final_score = round(score_row(row), 6)
            results.append(
                SearchResult(
                    memory_id=uuid.UUID(row["id"]),
                    memory_version_id=uuid.UUID(row["version_id"]),
                    statement=row["statement"],
                    kind=row["kind"],
                    score=final_score,
                    lexical_score=lexical_score,
                    vector_score=vector_score,
                    component_scores={
                        "confidence": round(float(row["confidence"]), 6),
                        "importance": round(float(row["importance"]), 6),
                        "evidence_quality": round(min(1.0, len(citations) / 3), 6),
                        "memory_type_signal": float(row["kind"].casefold() in query_terms),
                        "temporal_signal": float(row["status"] == "active"),
                        "staleness_penalty": float(row["status"] != "active"),
                        "lexical_rank": float(lexical_rank.get(str(row["version_id"]), 0)),
                        "vector_rank": float(vector_rank.get(str(row["version_id"]), 0)),
                        "alias_rank": float(alias_rank.get(str(row["version_id"]), 0)),
                        "exact_match": float(query.casefold() in str(row["statement"]).casefold()),
                        "graph_proximity": round(graph_signal(str(row["version_id"])), 6),
                        "session_summary": round(summary_signal(str(row["version_id"])), 6),
                    },
                    status=row["status"],
                    citations=citations,
                )
            )
        if not internal:
            self.record_retrieval(namespace_id, [str(item.memory_id) for item in results], successful=bool(results))
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

    def encoding_decisions(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM encoding_decisions WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?",
            (namespace_id, limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_retrieval(self, namespace_id: str, memory_ids: list[str], *, successful: bool = False) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        with self.db.connection:
            self.db.execute(
                f"""UPDATE memories SET last_accessed_at=?, access_count=access_count+1,
                retrieval_success_count=retrieval_success_count+? WHERE namespace_id=? AND id IN ({placeholders})""",
                (iso(), int(successful), namespace_id, *memory_ids),
            )

    def accessibility(self, namespace_id: str, *, now: datetime | None = None) -> int:
        """Recalculate access probability without deleting historical evidence."""
        current = now or datetime.now(UTC)
        rows = self.db.execute(
            "SELECT id, confidence, importance, last_accessed_at, access_count, retrieval_success_count, status FROM memories WHERE namespace_id=?",
            (namespace_id,),
        ).fetchall()
        changed = 0
        with self.db.lock, self.db.connection:
            for row in rows:
                if row["status"] in {"deleted", "invalidated"}:
                    score = 0.0
                else:
                    age_days = 365.0
                    if row["last_accessed_at"]:
                        try:
                            age_days = max(0.0, (current - datetime.fromisoformat(row["last_accessed_at"])).total_seconds() / 86400)
                        except ValueError:
                            pass
                    recency = max(0.0, min(1.0, 1.0 - age_days / 90.0))
                    success = min(1.0, float(row["retrieval_success_count"]) / max(1.0, float(row["access_count"]))) if row["access_count"] else 0.0
                    score = min(1.0, max(0.05, 0.30 * float(row["confidence"]) + 0.30 * float(row["importance"]) + 0.25 * recency + 0.15 * success))
                self.db.execute("UPDATE memories SET accessibility=? WHERE id=? AND namespace_id=?", (score, row["id"], namespace_id))
                changed += 1
        return changed

    def create_consolidation_run(self, namespace_id: str, episode_ids: list[str], mode: str = "dry-run") -> str:
        run_id = str(uuid.uuid4())
        with self.db.connection:
            self.db.execute(
                """INSERT INTO consolidation_runs(id, namespace_id, mode, status, input_episode_ids_json, started_at)
                VALUES (?, ?, ?, 'processing', ?, ?)""",
                (run_id, namespace_id, mode, json.dumps(episode_ids, separators=(",", ":")), iso()),
            )
        return run_id

    def record_consolidation_proposal(
        self,
        namespace_id: str,
        run_id: str,
        kind: str,
        subject_key: str,
        statement: str,
        evidence_event_ids: list[str],
        status: str,
        reason: str | None = None,
        memory_id: str | None = None,
        memory_version_id: str | None = None,
    ) -> str:
        proposal_hash = hash_text(
            json.dumps(
                {"kind": kind, "subject": subject_key, "statement": statement, "evidence": sorted(evidence_event_ids)},
                sort_keys=True,
            )
        )
        proposal_id = str(uuid.uuid4())
        with self.db.connection:
            self.db.execute(
                """INSERT INTO consolidation_proposals
                (id, run_id, namespace_id, proposal_hash, kind, subject_key, statement, evidence_event_ids_json,
                 status, reason, memory_id, memory_version_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace_id, proposal_hash) DO UPDATE SET
                  run_id=excluded.run_id,
                  status=CASE WHEN consolidation_proposals.status='accepted' THEN consolidation_proposals.status ELSE excluded.status END,
                  reason=excluded.reason,
                  memory_id=COALESCE(excluded.memory_id, consolidation_proposals.memory_id),
                  memory_version_id=COALESCE(excluded.memory_version_id, consolidation_proposals.memory_version_id)""",
                (
                    proposal_id,
                    run_id,
                    namespace_id,
                    proposal_hash,
                    kind,
                    subject_key,
                    redact_text(statement),
                    json.dumps(evidence_event_ids, separators=(",", ":")),
                    status,
                    redact_text(reason) if reason else None,
                    memory_id,
                    memory_version_id,
                    iso(),
                ),
            )
        return proposal_id

    def finish_consolidation_run(self, namespace_id: str, run_id: str, status: str, accepted: int, rejected: int, error: str | None = None) -> None:
        with self.db.connection:
            self.db.execute(
                """UPDATE consolidation_runs SET status=?, completed_at=?, proposal_count=?, accepted_count=?, rejected_count=?, error=?
                WHERE id=? AND namespace_id=?""",
                (status, iso(), accepted + rejected, accepted, rejected, redact_text(error) if error else None, run_id, namespace_id),
            )

    def list_consolidation_runs(self, namespace_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM consolidation_runs WHERE namespace_id=? ORDER BY started_at DESC LIMIT ?", (namespace_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def retrieve_procedure(self, namespace_id: str, goal: str, environment: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = [term for term in re.findall(r"[\w-]+", goal.casefold()) if len(term) > 2]
        rows = self.db.execute(
            """SELECT * FROM procedures WHERE namespace_id=? AND accessibility >= 0.15
            AND (environment=? OR environment='*') ORDER BY confidence DESC, success_count DESC LIMIT ?""",
            (namespace_id, environment, limit * 3),
        ).fetchall()
        ranked = sorted(
            rows,
            key=lambda row: (-sum(term in str(row["goal"]).casefold() for term in terms), -float(row["confidence"]), -int(row["success_count"]), row["id"]),
        )
        result = []
        for row in ranked[:limit]:
            item = dict(row)
            for key in ("preconditions_json", "actions_json", "failures_json"):
                item[key] = json.loads(item[key])
            result.append(item)
        return result

    def upsert_procedure(
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
        now = iso()
        procedure_id = str(uuid.uuid4())
        with self.db.lock, self.db.connection:
            current = self.db.execute(
                "SELECT * FROM procedures WHERE namespace_id=? AND goal=? AND environment=?",
                (namespace_id, goal, environment),
            ).fetchone()
            if current:
                procedure_id = str(current["id"])
                self.db.execute(
                    """UPDATE procedures SET preconditions_json=?, actions_json=?, expected_outcome=?, observed_outcome=?, failures_json=?,
                    success_count=success_count+?, verification_at=?, confidence=?, updated_at=? WHERE id=? AND namespace_id=?""",
                    (
                        json.dumps(preconditions),
                        json.dumps(actions),
                        expected_outcome,
                        observed_outcome,
                        json.dumps(failures),
                        int(success),
                        now,
                        min(1.0, float(current["confidence"]) + (0.1 if success else -0.05)),
                        now,
                        procedure_id,
                        namespace_id,
                    ),
                )
            else:
                self.db.execute(
                    """INSERT INTO procedures(id, namespace_id, goal, environment, preconditions_json, actions_json, expected_outcome,
                    observed_outcome, failures_json, success_count, verification_at, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        procedure_id,
                        namespace_id,
                        goal,
                        environment,
                        json.dumps(preconditions),
                        json.dumps(actions),
                        expected_outcome,
                        observed_outcome,
                        json.dumps(failures),
                        int(success),
                        now,
                        0.7 if success else 0.4,
                        now,
                        now,
                    ),
                )
            for event_id, excerpt in evidence:
                self.db.execute(
                    "INSERT OR IGNORE INTO procedure_evidence(procedure_id, namespace_id, event_id, excerpt) VALUES (?, ?, ?, ?)",
                    (procedure_id, namespace_id, event_id, redact_text(excerpt)),
                )
        return procedure_id

    def upsert_entity(self, namespace_id: str, canonical_key: str, label: str, entity_type: str = "unknown", confidence: float = 1.0) -> str:
        entity_id = str(uuid.uuid4())
        with self.db.lock, self.db.connection:
            self.db.execute(
                "INSERT INTO entities(id, namespace_id, canonical_key, label, entity_type, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(namespace_id, canonical_key) DO UPDATE SET label=excluded.label, entity_type=excluded.entity_type, confidence=excluded.confidence",
                (entity_id, namespace_id, canonical_key, label, entity_type, confidence, iso()),
            )
            row = self.db.execute("SELECT id FROM entities WHERE namespace_id=? AND canonical_key=?", (namespace_id, canonical_key)).fetchone()
            return str(row["id"])

    def add_entity_alias(self, namespace_id: str, entity_id: str, alias: str) -> bool:
        with self.db.lock, self.db.connection:
            cursor = self.db.execute("INSERT OR IGNORE INTO entity_aliases(entity_id, namespace_id, alias) VALUES (?, ?, ?)", (entity_id, namespace_id, alias))
            return cursor.rowcount > 0

    def add_relationship(
        self,
        namespace_id: str,
        subject_entity_id: str,
        predicate: str,
        object_entity_id: str,
        memory_version_id: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        confidence: float = 1.0,
    ) -> str:
        relationship_id = str(uuid.uuid4())
        with self.db.lock, self.db.connection:
            self.db.execute(
                """INSERT INTO relationships
                (id, namespace_id, subject_entity_id, predicate, object_entity_id, memory_version_id,
                 valid_from, valid_until, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    relationship_id,
                    namespace_id,
                    subject_entity_id,
                    predicate,
                    object_entity_id,
                    memory_version_id,
                    valid_from or iso(),
                    valid_until,
                    confidence,
                    iso(),
                ),
            )
        return relationship_id

    def related_entities(self, namespace_id: str, entity_id: str, depth: int = 1) -> list[dict[str, Any]]:
        depth = max(1, min(depth, 2))
        seen = {entity_id}
        frontier = [entity_id]
        result: list[dict[str, Any]] = []
        relationship_ids: set[str] = set()
        for distance in range(1, depth + 1):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self.db.execute(
                f"""SELECT r.*, s.label AS subject_label, o.label AS object_label
                    FROM relationships r JOIN entities s ON s.id=r.subject_entity_id JOIN entities o ON o.id=r.object_entity_id
                    WHERE r.namespace_id=? AND r.status='active' AND (r.subject_entity_id IN ({placeholders}) OR r.object_entity_id IN ({placeholders}))""",
                (namespace_id, *frontier, *frontier),
            ).fetchall()
            frontier = []
            for row in rows:
                if row["id"] in relationship_ids:
                    continue
                relationship_ids.add(row["id"])
                item = dict(row)
                item["distance"] = distance
                result.append(item)
                other = row["object_entity_id"] if row["subject_entity_id"] in seen else row["subject_entity_id"]
                if other not in seen:
                    seen.add(other)
                    frontier.append(other)
        return result

    def record_graph_links(
        self,
        namespace_id: str,
        *,
        episode_id: str | None,
        memory_version_id: str,
        memory_id: str,
        subject_key: str,
        statement: str,
        predicate: str,
        previous_version_id: str | None = None,
    ) -> None:
        subject_entity = self.upsert_entity(namespace_id, f"subject:{subject_key}", subject_key, "memory-subject", 1.0)
        memory_entity = self.upsert_entity(namespace_id, f"memory-version:{memory_version_id}", statement[:120], "memory-version", 1.0)
        self.add_entity_alias(namespace_id, subject_entity, subject_key)
        self.add_entity_alias(namespace_id, memory_entity, memory_id)
        self.add_relationship(namespace_id, subject_entity, "expresses", memory_entity, memory_version_id=memory_version_id, confidence=1.0)
        if episode_id:
            episode_entity = self.upsert_entity(namespace_id, f"episode:{episode_id}", episode_id, "session", 1.0)
            self.add_relationship(namespace_id, episode_entity, "contains", memory_entity, memory_version_id=memory_version_id, confidence=1.0)
        if previous_version_id:
            previous_entity = self.upsert_entity(namespace_id, f"memory-version:{previous_version_id}", previous_version_id, "memory-version", 1.0)
            self.add_relationship(namespace_id, previous_entity, predicate, memory_entity, memory_version_id=memory_version_id, confidence=1.0)

    def rebuild_graph_index(self, namespace_id: str) -> dict[str, int]:
        with self.db.lock, self.db.connection:
            self.db.execute("DELETE FROM relationships WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM entity_aliases WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM entities WHERE namespace_id=?", (namespace_id,))
            rows = self.db.execute(
                """SELECT v.id AS version_id, v.memory_id, v.statement, v.source_event_id, v.version, m.subject_key
                FROM memory_versions v JOIN memories m ON m.id=v.memory_id AND m.namespace_id=v.namespace_id
                WHERE v.namespace_id=? ORDER BY v.memory_id, v.version""",
                (namespace_id,),
            ).fetchall()
            count = 0
            previous_by_memory: dict[str, str] = {}
            for row in rows:
                version_id = str(row["version_id"])
                previous_version_id = previous_by_memory.get(str(row["memory_id"]))
                episode_row = self.db.execute(
                    "SELECT episode_id FROM events WHERE id=? AND namespace_id=?",
                    (row["source_event_id"], namespace_id),
                ).fetchone()
                self.record_graph_links(
                    namespace_id,
                    episode_id=str(episode_row["episode_id"]) if episode_row and episode_row["episode_id"] else None,
                    memory_version_id=version_id,
                    memory_id=str(row["memory_id"]),
                    subject_key=str(row["subject_key"]),
                    statement=str(row["statement"]),
                    predicate="updates" if previous_version_id else "contains",
                    previous_version_id=previous_version_id,
                )
                previous_by_memory[str(row["memory_id"])] = version_id
                count += 1
            return {"edges": count}

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
            self.db.execute(
                "UPDATE memory_versions SET status='active', valid_to=NULL, reason='RESTORE' WHERE id=? AND namespace_id=?",
                (version["id"], namespace_id),
            )
            self.db.execute("UPDATE memories SET status='active', current_version_id=? WHERE id=? AND namespace_id=?", (version["id"], memory_id, namespace_id))
            self.db.execute(
                "INSERT OR REPLACE INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?, ?, ?, ?)",
                (version["id"], namespace_id, version["statement"], version["evidence_excerpt"] or ""),
            )
            return True

    def history(self, namespace_id: str, memory_id: str) -> list[dict[str, Any]] | None:
        memory = self.db.execute("SELECT id FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)).fetchone()
        if not memory:
            return None
        rows = self.list_versions(namespace_id, memory_id)
        return [dict(row) for row in rows]

    def export_namespace(self, namespace_id: str) -> dict[str, Any]:
        tables = (
            "namespaces",
            "events",
            "artifacts",
            "memories",
            "memory_versions",
            "evidence_refs",
            "processing_jobs",
            "context_requests",
            "extraction_runs",
            "extraction_decisions",
            "episodes",
            "episode_events",
            "memory_embeddings",
            "feedback",
            "encoding_decisions",
            "consolidation_runs",
            "consolidation_proposals",
            "procedures",
            "procedure_evidence",
        )
        result: dict[str, Any] = {
            "namespaces": [dict(row) for row in self.db.execute("SELECT * FROM namespaces WHERE id=?", (namespace_id,))],
        }
        for table in tables[1:]:
            rows = [dict(row) for row in self.db.execute(f"SELECT * FROM {table} WHERE namespace_id=?", (namespace_id,))]
            if table == "memory_embeddings":
                for row in rows:
                    row["vector"] = base64.b64encode(bytes(row["vector"])).decode("ascii")
            result[table] = rows
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
            "namespaces",
            "events",
            "artifacts",
            "memories",
            "extraction_runs",
            "memory_versions",
            "processing_jobs",
            "context_requests",
            "evidence_refs",
            "extraction_decisions",
            "episodes",
            "episode_events",
            "memory_embeddings",
            "feedback",
            "encoding_decisions",
            "consolidation_runs",
            "consolidation_proposals",
            "procedures",
            "procedure_evidence",
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
                    values = dict(row)
                    if table == "memory_embeddings" and isinstance(values.get("vector"), str):
                        values["vector"] = base64.b64decode(values["vector"], validate=True)
                    cursor = self.db.execute(
                        f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                        tuple(values[column] for column in columns),
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
        statements = [str(row["statement"]) for row in rows]
        vectors = self.embed_many(statements) if statements else []
        for row, vector in zip(rows, vectors, strict=True):
            self.db.execute(
                "INSERT INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?, ?, ?, ?)",
                (row["id"], namespace_id, row["statement"], row["evidence_excerpt"]),
            )
            self.db.execute(
                "INSERT INTO memory_embeddings(memory_version_id, namespace_id, provider, dimensions, vector) VALUES (?, ?, ?, ?, ?)",
                (
                    row["id"],
                    namespace_id,
                    self.embedding.name,
                    self.embedding.dimensions,
                    pack_embedding(vector),
                ),
            )
        self.vector_index.rebuild_all()

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
            self.db.execute("DELETE FROM relationships WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM entity_aliases WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM entities WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM memory_embeddings WHERE namespace_id=?", (namespace_id,))
            self.vector_index.delete_namespace(namespace_id)
            self.db.execute("DELETE FROM feedback WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM context_requests WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM artifacts WHERE namespace_id=?", (namespace_id,))
            for table in (
                "procedure_evidence",
                "procedures",
                "consolidation_proposals",
                "consolidation_runs",
                "encoding_decisions",
                "evidence_refs",
                "memory_versions",
                "memories",
                "processing_jobs",
                "events",
            ):
                self.db.execute(f"DELETE FROM {table} WHERE namespace_id=?", (namespace_id,))
            self.db.execute("DELETE FROM namespaces WHERE id=?", (namespace_id,))
            return True

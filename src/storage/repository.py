from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ..config.settings import RETRIEVAL as _RETRIEVAL_CFG
from ..core.errors import IdempotencyConflict
from ..core.redaction import redact_text
from ..memory.encoding import score_observation
from ..memory.extraction import CandidateRejected, ValidatedCandidate
from ..memory.extractor import Candidate, payload_text
from ..memory.provider import SessionSummaryProvider
from ..models import EventInput, EvidenceCitation, ExtractionCandidate, MemoryResponse, SearchResult, SessionSearchResult, temporal_recency_score
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
def _normalize_state_key(value: str | None) -> str | None:
    if not value:
        return None
    normalized = " ".join(str(value).strip().split()).casefold()
    if "." not in normalized:
        return None
    return normalized


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

_RERANKER_LOCK = threading.Lock()
_RERANKERS: dict[str, Any] = {}


def _cached_reranker(model_name: str) -> Any | None:
    """Create a FlashRank model once per process, not once per query."""
    with _RERANKER_LOCK:
        if model_name in _RERANKERS:
            return _RERANKERS[model_name]
        try:
            from flashrank import Ranker  # type: ignore[import-untyped]

            _RERANKERS[model_name] = Ranker(model_name=model_name)
        except Exception:
            return None
        return _RERANKERS[model_name]


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
        self._ensure_extraction_runs_tracing_columns()
        self._ensure_chunk_event_index()
        self._ensure_v3_columns()
        self._claimed_lease_tokens: dict[str, str] = {}

    def _ensure_extraction_runs_tracing_columns(self) -> None:
        try:
            cols = {row[1] for row in self.db.execute("PRAGMA table_info(extraction_runs)").fetchall()}
            if "stage" not in cols:
                self.db.execute("ALTER TABLE extraction_runs ADD COLUMN stage TEXT")
            if "candidate_count" not in cols:
                self.db.execute("ALTER TABLE extraction_runs ADD COLUMN candidate_count INTEGER")
            if "input_event_ids" not in cols:
                self.db.execute("ALTER TABLE extraction_runs ADD COLUMN input_event_ids TEXT")
            if "existing_memory_count" not in cols:
                self.db.execute("ALTER TABLE extraction_runs ADD COLUMN existing_memory_count INTEGER")
            if "input_hash" not in cols:
                pass
        except Exception:
            pass

    def _ensure_v3_columns(self) -> None:
        try:
            cols = {row[1] for row in self.db.execute("PRAGMA table_info(memory_versions)").fetchall()}
            if "state_key" not in cols:
                self.db.execute("ALTER TABLE memory_versions ADD COLUMN state_key TEXT")
            if "lifecycle" not in cols:
                self.db.execute("ALTER TABLE memory_versions ADD COLUMN lifecycle TEXT")
            if "observed_at" not in cols:
                self.db.execute("ALTER TABLE memory_versions ADD COLUMN observed_at TEXT")
            if "source_event_ids_json" not in cols:
                self.db.execute("ALTER TABLE memory_versions ADD COLUMN source_event_ids_json TEXT")
            if "event_dates_json" not in cols:
                self.db.execute("ALTER TABLE memory_versions ADD COLUMN event_dates_json TEXT")
            # extraction_decisions extra diagnostics for v3
            dcols = {row[1] for row in self.db.execute("PRAGMA table_info(extraction_decisions)").fetchall()}
            if "v3_type" not in dcols:
                self.db.execute("ALTER TABLE extraction_decisions ADD COLUMN v3_type TEXT")
            if "v3_lifecycle" not in dcols:
                self.db.execute("ALTER TABLE extraction_decisions ADD COLUMN v3_lifecycle TEXT")
            if "v3_state_key" not in dcols:
                self.db.execute("ALTER TABLE extraction_decisions ADD COLUMN v3_state_key TEXT")
            if "source_event_ids_json" not in dcols:
                self.db.execute("ALTER TABLE extraction_decisions ADD COLUMN source_event_ids_json TEXT")
            if "source_chunk_ids_json" not in dcols:
                self.db.execute("ALTER TABLE extraction_decisions ADD COLUMN source_chunk_ids_json TEXT")
        except Exception:
            pass

    def _ensure_chunk_event_index(self) -> None:
        """Backfill normalized chunk/event rows for databases created pre-v2."""
        try:
            rows = self.db.execute(
                "SELECT c.chunk_id, c.namespace_id, c.session_id, c.event_ids_json FROM chunks c "
                "WHERE NOT EXISTS (SELECT 1 FROM chunk_events ce WHERE ce.chunk_id=c.chunk_id)"
            ).fetchall()
            values: list[tuple[str, str, str, str]] = []
            for row in rows:
                values.extend((str(row["chunk_id"]), str(row["namespace_id"]), str(event_id), str(row["session_id"])) for event_id in json.loads(row["event_ids_json"]))
            if values:
                with self.db.connection:
                    self.db.executemany("INSERT OR IGNORE INTO chunk_events(chunk_id, namespace_id, event_id, session_id) VALUES (?,?,?,?)", values)
        except Exception:
            # A partially migrated legacy database remains usable through the
            # legacy JSON paths until its next successful open.
            pass
        # Phase 3: evidence becomes optional — drop strict guard that required offsets/excerpts
        try:
            self.db.execute("DROP TRIGGER IF EXISTS memory_versions_evidence_guard")
        except Exception:
            pass

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

    def create_processing_job(self, namespace_id: str, event_id: str, content_hash: str) -> str:
        """Create a pending processing job for an event. Idempotent per event."""
        job_id = str(uuid.uuid4())
        with self.db.lock, self.db.connection:
            existing = self.db.execute(
                "SELECT id FROM processing_jobs WHERE namespace_id=? AND event_id=? AND status IN ('pending','processing','failed')",
                (namespace_id, event_id),
            ).fetchone()
            if existing:
                return existing["id"]
            self.db.execute(
                """INSERT INTO processing_jobs
                (id, namespace_id, event_id, input_hash, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (job_id, namespace_id, event_id, content_hash, iso(), iso()),
            )
        return job_id

    def enqueue_failed_job(self, namespace_id: str, event_id: str, content_hash: str, error: str, retryable: bool = True) -> str:
        job_id = self.create_processing_job(namespace_id, event_id, content_hash)
        # Transition pending -> failed/dead with backoff
        self.fail_job(namespace_id, job_id, error, retryable=retryable)
        return job_id

    def delete_completed_jobs(self, namespace_id: str, event_ids: list[str]) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        with self.db.lock, self.db.connection:
            self.db.execute(
                f"DELETE FROM processing_jobs WHERE namespace_id=? AND event_id IN ({placeholders}) AND status IN ('completed','pending')",
                (namespace_id, *event_ids),
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

    def formation_metrics(self, namespace_id: str) -> dict[str, Any]:
        """Phase 0 memory-formation metrics for benchmark diagnostics."""
        total_decisions = int(self.db.execute("SELECT COUNT(*) FROM extraction_decisions WHERE namespace_id=?", (namespace_id,)).fetchone()[0])
        accepted = int(self.db.execute("SELECT COUNT(*) FROM extraction_decisions WHERE namespace_id=? AND validation_status='accepted'", (namespace_id,)).fetchone()[0])
        rejected = total_decisions - accepted
        # "accepted" also includes IGNORE decisions, which do not produce a
        # memory record.  Grounding must be measured only for persisted or
        # reinforced records, and an absent JSON value must never be counted
        # as evidence.
        persisted = int(
            self.db.execute(
                """SELECT COUNT(*) FROM extraction_decisions
                WHERE namespace_id=? AND validation_status='accepted'
                  AND memory_version_id IS NOT NULL""",
                (namespace_id,),
            ).fetchone()[0]
        )
        grounded = int(
            self.db.execute(
                """SELECT COUNT(*) FROM extraction_decisions
                WHERE namespace_id=? AND validation_status='accepted'
                  AND memory_version_id IS NOT NULL
                  AND source_event_ids_json IS NOT NULL
                  AND source_event_ids_json != '[]'""",
                (namespace_id,),
            ).fetchone()[0]
        )
        grounded_rate = round(grounded / persisted, 3) if persisted else 0.0
        # Records per session
        mem_count = int(self.db.execute("SELECT COUNT(*) FROM memories WHERE namespace_id=?", (namespace_id,)).fetchone()[0])
        session_count = int(self.db.execute("SELECT COUNT(DISTINCT session_id) FROM chunks WHERE namespace_id=?", (namespace_id,)).fetchone()[0] or 1)
        records_per_session = round(mem_count / max(1, session_count), 2)
        # Duplicate rate
        dup = int(self.db.execute("SELECT COUNT(*) FROM extraction_decisions WHERE namespace_id=? AND rejection_reason='duplicate_candidate'", (namespace_id,)).fetchone()[0])
        dup_rate = round(dup / max(1, total_decisions), 3)
        # Type distribution
        type_rows = self.db.execute("SELECT kind, COUNT(*) as cnt FROM memories WHERE namespace_id=? GROUP BY kind", (namespace_id,)).fetchall()
        type_dist = {row["kind"]: int(row["cnt"]) for row in type_rows}
        # v3 type distribution if available
        try:
            v3_rows = self.db.execute("SELECT v3_type, COUNT(*) as cnt FROM extraction_decisions WHERE namespace_id=? AND v3_type IS NOT NULL GROUP BY v3_type", (namespace_id,)).fetchall()
            v3_dist = {row["v3_type"]: int(row["cnt"]) for row in v3_rows}
            if v3_dist:
                type_dist["v3"] = v3_dist
        except Exception:
            pass
        # Unsupported/fabricated rejection rate
        unsup = int(self.db.execute("SELECT COUNT(*) FROM extraction_decisions WHERE namespace_id=? AND rejection_reason IN ('unsupported_statement','unknown_source_chunk_id','unknown_source_label','context_only_source')", (namespace_id,)).fetchone()[0])
        unsup_rate = round(unsup / max(1, total_decisions), 3)
        # Answer-session extraction coverage placeholder (requires benchmark answer_session_ids, not available here)
        return {
            "grounded_record_rate": grounded_rate,
            "records_per_session": records_per_session,
            "duplicate_rate": dup_rate,
            "type_distribution": type_dist,
            "unsupported_rejection_rate": unsup_rate,
            "total_decisions": total_decisions,
            "accepted": accepted,
            "rejected": rejected,
            "persisted_records": persisted,
            "grounded_records": grounded,
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

    def rebuild_chunks(
        self,
        namespace_id: str,
        *,
        window: int = 4,
        overlap: int = 1,
        event_ids: list[str] | None = None,
    ) -> int:
        """Reindex only sessions changed by an ingestion batch.

        The old full rebuild remains available when ``event_ids`` is omitted,
        which keeps repair and migration callers compatible.
        """
        from ..retrieval.chunking import build_chunks

        events = self.list_events(namespace_id, limit=1_000_000)
        source = [{**event, "text": payload_text(event["payload_json"], event["type"], include_roles=True)} for event in events]
        chunks = build_chunks(source, window=window, overlap=overlap)
        changed_sessions: set[str] | None = None
        if event_ids is not None:
            wanted = set(event_ids)
            changed_sessions = {
                str(event.get("session_id") or event.get("stream_id") or event["id"])
                for event in events
                if str(event["id"]) in wanted
            }
            chunks = [chunk for chunk in chunks if chunk.session_id in changed_sessions]
        with self.db.lock, self.db.connection:
            if changed_sessions is None:
                self.db.execute("DELETE FROM chunks WHERE namespace_id=?", (namespace_id,))
            elif changed_sessions:
                placeholders = ",".join("?" for _ in changed_sessions)
                self.db.execute(f"DELETE FROM chunks WHERE namespace_id=? AND session_id IN ({placeholders})", (namespace_id, *sorted(changed_sessions)))
            if chunks:
                self.db.executemany(
                    "INSERT INTO chunks(chunk_id,namespace_id,session_id,ordinal,event_ids_json,raw_text,document_date,event_dates_json,contextual_text,content_hash,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [(c.chunk_id, namespace_id, c.session_id, c.ordinal, json.dumps(c.event_ids), c.text, c.document_date, json.dumps(c.event_dates), c.contextual_text, hashlib.sha256(c.text.encode()).hexdigest(), iso()) for c in chunks],
                )
                self.db.executemany(
                    "INSERT INTO chunk_events(chunk_id, namespace_id, event_id, session_id) VALUES (?,?,?,?)",
                    [(chunk.chunk_id, namespace_id, event_id, chunk.session_id) for chunk in chunks for event_id in chunk.event_ids],
                )
            # Every configured embedding provider indexes chunks.  This is
            # essential for OpenRouter-backed embeddings, not only FastEmbed.
            if chunks:
                try:
                    vectors = self.embed_many([value for c in chunks for value in (c.text, c.contextual_text)])
                    self.db.executemany(
                        "INSERT INTO chunk_embeddings VALUES (?,?,?,?,?,?)",
                        [(c.chunk_id, namespace_id, self.embedding.name, len(vectors[index * 2]), pack_embedding(vectors[index * 2]), 0) for index, c in enumerate(chunks)]
                        + [(c.chunk_id, namespace_id, self.embedding.name, len(vectors[index * 2 + 1]), pack_embedding(vectors[index * 2 + 1]), 1) for index, c in enumerate(chunks)],
                    )
                except Exception:
                    pass
        return len(chunks)

    def chunks_for_namespace(self, namespace_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        result = []
        for row in self.db.execute("SELECT * FROM chunks WHERE namespace_id=? ORDER BY session_id, ordinal LIMIT ?", (namespace_id, limit)).fetchall():
            result.append({**dict(row), "event_ids": json.loads(row["event_ids_json"]), "event_dates": json.loads(row["event_dates_json"])})
        return result

    def chunk_ids_for_namespace(self, namespace_id: str) -> set[str]:
        return {str(row["chunk_id"]) for row in self.db.execute("SELECT chunk_id FROM chunks WHERE namespace_id=?", (namespace_id,)).fetchall()}

    def chunk_session_map(self, namespace_id: str) -> dict[str, str]:
        return {str(row["chunk_id"]): str(row["session_id"]) for row in self.db.execute("SELECT chunk_id, session_id FROM chunks WHERE namespace_id=?", (namespace_id,)).fetchall()}

    def chunks_for_events(self, namespace_id: str, event_ids: list[str], limit: int = 2) -> list[dict[str, Any]]:
        wanted = set(event_ids)
        if not wanted:
            return []
        placeholders = ",".join("?" for _ in wanted)
        rows = self.db.execute(
            f"""SELECT DISTINCT c.* FROM chunks c
            JOIN chunk_events ce ON ce.chunk_id=c.chunk_id AND ce.namespace_id=c.namespace_id
            WHERE c.namespace_id=? AND ce.event_id IN ({placeholders})
            ORDER BY c.session_id, c.ordinal LIMIT ?""",
            (namespace_id, *sorted(wanted), limit),
        ).fetchall()
        return [{**dict(row), "event_ids": json.loads(row["event_ids_json"]), "event_dates": json.loads(row["event_dates_json"])} for row in rows]

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
        # Reconciliation receives an ingestion batch, not a user query. Bound
        # it before search so a long transcript cannot generate a SQLite query
        # with more than its 1,000-expression limit.
        results = self.search(namespace_id, query[:12000], max(1, min(limit, 5)), historical=False, internal=True)
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
        import warnings

        warnings.warn("record_context_request is deprecated; context_requests are no longer recorded", DeprecationWarning, stacklevel=2)
        # No longer creates new rows — table retained only for reading legacy databases.
        return str(uuid.uuid4())

    def list_context_requests(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        import warnings

        warnings.warn("list_context_requests is deprecated; context_requests table is retained for legacy reads only", DeprecationWarning, stacklevel=2)
        rows = self.db.execute(
            "SELECT * FROM context_requests WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?", (namespace_id, limit, offset)
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["selected_json"] = json.loads(item["selected_json"])
            except Exception:
                pass
            try:
                item["diagnostics_json"] = json.loads(item["diagnostics_json"])
            except Exception:
                pass
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
        retry_after: float | None = None,
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
            # Honor Retry-After if provided (e.g. 429 retry_after=60 should not be immediately claimable)
            if retry_after is not None and retry_after > 0:
                delay = min(3600.0, max(0.0, float(retry_after)))
            elif row["status"] == "pending" and int(row["attempts"]) == 0:
                delay = 0
            else:
                delay = min(300, 2 ** max(1, int(row["attempts"])))
            self.db.execute(
                """UPDATE processing_jobs
                SET status=?, lease_until=NULL, lease_token=NULL, next_attempt_at=?, last_error=?, updated_at=?
                WHERE id=? AND namespace_id=?""",
                (status, None, error, iso(), job_id, namespace_id),
            )
            if status != "dead":
                if delay == 0:
                    self.db.execute(
                        "UPDATE processing_jobs SET next_attempt_at=datetime('now') WHERE id=? AND namespace_id=?",
                        (job_id, namespace_id),
                    )
                else:
                    # SQLite datetime modifier requires integer seconds; round up
                    secs = int(delay) if delay == int(delay) else int(delay) + 1
                    self.db.execute(
                        "UPDATE processing_jobs SET next_attempt_at=datetime('now', ?) WHERE id=? AND namespace_id=?",
                        (f"+{secs} seconds", job_id, namespace_id),
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
        # Build v3 diagnostics if available
        v3_type = getattr(candidate, "v3_type", None)
        v3_lifecycle = getattr(candidate, "v3_lifecycle", None)
        v3_state_key = getattr(candidate, "v3_state_key", None)
        source_event_ids_json = None
        source_chunk_ids_json = None
        try:
            if getattr(candidate, "evidence", None):
                source_event_ids_json = json.dumps([str(sp.event_id) for sp in candidate.evidence])
            if getattr(candidate, "source_chunk_ids", None):
                source_chunk_ids_json = json.dumps(list(candidate.source_chunk_ids))
        except Exception:
            pass
        with self.db.connection:
            # Try to include v3 columns if they exist (added via _ensure_v3_columns)
            try:
                self.db.execute(
                    """INSERT OR IGNORE INTO extraction_decisions
                    (id, run_id, namespace_id, candidate_fingerprint, kind, subject, statement,
                     validation_status, rejection_reason, action, memory_id, memory_version_id, created_at,
                     v3_type, v3_lifecycle, v3_state_key, source_event_ids_json, source_chunk_ids_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        v3_type,
                        v3_lifecycle,
                        v3_state_key,
                        source_event_ids_json,
                        source_chunk_ids_json,
                    ),
                )
            except Exception:
                # Fallback for DBs without v3 columns (legacy)
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
        # Persisted memories must have real source evidence.  The processor
        # already validates the span; this protects direct repository callers.
        if not item.evidence:
            raise CandidateRejected("missing_source_evidence")
        source = self.db.execute(
            "SELECT namespace_id, created_at, occurred_at, payload_json FROM events WHERE id=? AND namespace_id=?",
            (str(item.evidence[0].event_id), namespace_id),
        ).fetchone()
        if not source:
            raise ValueError("evidence event is not in the requested namespace")
        if source["created_at"] > iso():
            raise ValueError("evidence postdates derived version")
        # A current state is one timeline.  Stable and historical records are
        # separate observations, not updates to that timeline.  Give them a
        # deterministic identity so they are retained and repeat ingestion
        # reinforces the existing record instead of creating duplicates.
        v3_lifecycle = getattr(item, "v3_lifecycle", None)
        v3_state_key = _normalize_state_key(getattr(item, "v3_state_key", None))
        memory_subject = item.subject
        if v3_lifecycle in {"stable", "historical", "instruction", "task"}:
            statement_key = hashlib.sha256(item.statement.casefold().encode("utf-8")).hexdigest()[:16]
            memory_subject = f"{item.subject}::observation::{statement_key}"
        memory = self.db.execute(
            "SELECT * FROM memories WHERE namespace_id=? AND kind=? AND subject_key=?",
            (namespace_id, item.kind, memory_subject),
        ).fetchone()
        memory_id = memory["id"] if memory else str(uuid.uuid4())
        if item.existing_memory_id is not None and (not memory or str(item.existing_memory_id) != memory_id):
            raise CandidateRejected("existing_memory_identity_mismatch")
        if item.intent == "ignore":
            return memory_id if memory else None, "IGNORE", None
        if not memory:
            self.db.execute(
                "INSERT INTO memories(id, namespace_id, kind, subject_key, status, confidence, importance, created_at) VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
                (memory_id, namespace_id, item.kind, memory_subject, item.confidence, item.importance, iso()),
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
        # Compute the latest world-time observation before deciding whether a
        # state can replace the current value.  Ingestion time is not a valid
        # substitute: old transcripts may be imported after newer ones.
        latest_source_time = source["occurred_at"]
        for span in item.evidence[1:]:
            row = self.db.execute(
                "SELECT occurred_at FROM events WHERE id=? AND namespace_id=?",
                (str(span.event_id), namespace_id),
            ).fetchone()
            if row and row["occurred_at"] and (not latest_source_time or str(row["occurred_at"]) > str(latest_source_time)):
                latest_source_time = row["occurred_at"]

        # Phase 2 safe current-state versioning for v3.
        v3_current_update = False
        if current and v3_lifecycle is not None:
            # Non-current lifecycles have already received independent memory
            # identities above.  Only `current` may replace a current value.
            if v3_lifecycle == "current":
                current_key = _normalize_state_key(current["state_key"])
                current_time = current["observed_at"] or current["valid_from"]
                if (
                    not v3_state_key
                    or current_key != v3_state_key
                    or not latest_source_time
                    or (current_time and str(latest_source_time) <= str(current_time))
                ):
                    return memory_id, "IGNORE", None
                v3_current_update = True
        # Simplified reconciliation: only insert, reinforce, update, supersede, ignore
        if current and (item.intent in {"update", "supersede"} or v3_current_update):
            explicit = item.confidence >= 0.85 or any(
                marker in item.statement.casefold() or marker in span.excerpt.casefold() for span in item.evidence for marker in TRANSITION_MARKERS
            ) or v3_current_update
            action = "UPDATE" if v3_current_update or item.intent == "update" else "SUPERSEDE"
            if not explicit:
                # Without explicit marker and low confidence, treat as ignore rather than dispute
                return memory_id, "IGNORE", None
            status = "active"
        elif current:
            # Different wording alone is not proof that a fact changed.  Keep
            # the existing version unless v2 explicitly marks an update or the
            # same canonical identity is backed by a newer source event.
            if item.statement == current["statement"]:
                return memory_id, "REINFORCE", current["id"]
            if source["occurred_at"] and current["valid_from"] and str(source["occurred_at"]) > str(current["valid_from"]):
                action, status = "UPDATE", "active"
            else:
                return memory_id, "IGNORE", None
        else:
            action, status = "INSERT", "active"
        version = current["version"] + 1 if current else 1
        if current and status == "active":
            self.db.execute(
                "UPDATE memory_versions SET status='superseded', valid_to=? WHERE id=? AND namespace_id=?",
                (latest_source_time or iso(), current["id"], namespace_id),
            )
            self.db.execute("DELETE FROM memory_fts WHERE memory_version_id=? AND namespace_id=?", (current["id"], namespace_id))
        span = item.evidence[0]
        src_id, start_off, end_off, excerpt = str(span.event_id), span.start_offset, span.end_offset, span.excerpt
        # For multi-event records, observed_at should be latest source event time
        latest_occurred = latest_source_time
        if len(item.evidence) > 1:
            try:
                for sp in item.evidence[1:]:
                    row2 = self.db.execute("SELECT occurred_at FROM events WHERE id=? AND namespace_id=?", (str(sp.event_id), namespace_id)).fetchone()
                    if row2 and row2["occurred_at"]:
                        if not latest_occurred or str(row2["occurred_at"]) > str(latest_occurred):
                            latest_occurred = row2["occurred_at"]
                            # also update src_id to latest for primary provenance when v3
                            if getattr(item, "v3_lifecycle", None) == "current":
                                src_id = str(sp.event_id)
                                start_off, end_off, excerpt = sp.start_offset, sp.end_offset, sp.excerpt
            except Exception:
                pass
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
                src_id,
                start_off,
                end_off,
                excerpt,
                version,
                item.statement,
                item.valid_from.astimezone(UTC).isoformat() if item.valid_from else latest_occurred,
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
        # Store v3 diagnostics and multi-event provenance
        try:
            state_key = getattr(item, "v3_state_key", None)
            lifecycle = getattr(item, "v3_lifecycle", None)
            source_ids_json = json.dumps([str(sp.event_id) for sp in item.evidence])
            # event_dates from evidence events
            event_dates = []
            for sp in item.evidence:
                r = self.db.execute("SELECT occurred_at FROM events WHERE id=? AND namespace_id=?", (str(sp.event_id), namespace_id)).fetchone()
                if r and r["occurred_at"]:
                    event_dates.append(str(r["occurred_at"]))
            event_dates_json = json.dumps(event_dates)
            self.db.execute(
                "UPDATE memory_versions SET state_key=?, lifecycle=?, observed_at=?, source_event_ids_json=?, event_dates_json=? WHERE id=? AND namespace_id=?",
                (v3_state_key, lifecycle, latest_occurred, source_ids_json, event_dates_json, version_id, namespace_id),
            )
        except Exception:
            pass
        if status == "active":
            self.db.execute(
                "UPDATE memories SET current_version_id=?, status='active', confidence=?, importance=? WHERE id=? AND namespace_id=?",
                (version_id, item.confidence, item.importance, memory_id, namespace_id),
            )
            self.db.execute(
                "INSERT INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?, ?, ?, ?)",
                (version_id, namespace_id, item.statement, excerpt or ""),
            )
        self._persist_embedding(version_id, namespace_id, embedding, item.statement)
        self.record_graph_links(
            namespace_id,
            episode_id=str(event["episode_id"]) if event["episode_id"] else None,
            memory_version_id=version_id,
            memory_id=memory_id,
            subject_key=memory_subject,
            statement=item.statement,
            predicate="updates" if previous_version_id else "contains",
            previous_version_id=previous_version_id,
        )
        return memory_id, action, version_id

    def get_memory(self, namespace_id: str, memory_id: str) -> MemoryResponse | None:
        row = self.db.execute(
            """SELECT m.*, v.id AS version_id, v.version, v.statement, v.status AS version_status,
                      v.valid_from, v.valid_until, v.recorded_at, v.source_event_id, v.evidence_excerpt,
                      m.created_at AS mem_created_at, m.current_version_id
            FROM memories m JOIN memory_versions v ON v.id=m.current_version_id
            WHERE m.id=? AND m.namespace_id=? AND v.namespace_id=?""",
            (memory_id, namespace_id, namespace_id),
        ).fetchone()
        if not row:
            return None
        citations = self._citations(namespace_id, row["version_id"])
        from ..models import TemporalBlock

        temporal = None
        try:
            temporal = TemporalBlock(
                valid_from=datetime.fromisoformat(str(row["valid_from"])) if row["valid_from"] else None,
                valid_until=datetime.fromisoformat(str(row["valid_until"])) if row["valid_until"] else None,
                recorded_at=datetime.fromisoformat(str(row["recorded_at"])) if row["recorded_at"] else None,
            )
        except Exception:
            temporal = None
        source_ids = [c.event_id for c in citations]
        if row["source_event_id"] and uuid.UUID(row["source_event_id"]) not in source_ids:
            source_ids.append(uuid.UUID(row["source_event_id"]))
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
            temporal=temporal,
            source_event_ids=source_ids,
            evidence_excerpt=row["evidence_excerpt"],
            created_at=row["mem_created_at"],
            updated_at=row["recorded_at"],
        )

    def list_memories(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[MemoryResponse]:
        rows = self.db.execute(
            """SELECT m.*, v.id AS version_id, v.version, v.statement, v.status AS version_status,
                      v.source_event_id, v.evidence_excerpt, v.recorded_at
            FROM memories m JOIN memory_versions v ON v.id=m.current_version_id
            WHERE m.namespace_id=? AND v.namespace_id=?
            ORDER BY m.created_at, m.id LIMIT ? OFFSET ?""",
            (namespace_id, namespace_id, limit, offset),
        ).fetchall()
        result: list[MemoryResponse] = []
        for row in rows:
            citations = self._citations(namespace_id, row["version_id"])
            source_ids = [c.event_id for c in citations]
            if row["source_event_id"] and uuid.UUID(row["source_event_id"]) not in source_ids:
                source_ids.append(uuid.UUID(row["source_event_id"]))
            result.append(
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
                    citations=citations,
                    source_event_ids=source_ids,
                    evidence_excerpt=row["evidence_excerpt"],
                    created_at=row["created_at"],
                    updated_at=row["recorded_at"],
                )
            )
        return result

    def _query_weights(self, query: str) -> dict[str, float]:
        """Deterministic query-aware weights (no LLM call).

        - identifiers/names/codes/quoted text -> increase FTS weight
        - conceptual questions -> increase dense-vector weight
        - latest/current/now/newest -> recency boost
        - first/previous/before/original -> historical boost
        - multi-session -> allow more independent evidence chunks
        """
        ql = query.casefold()
        has_identifiers = bool(re.search(r'["\'][\w\s./:-]+["\']|\b[A-Z]{2,}\b|\b\w+_\w+\b|\b\d{2,}\b', query))
        has_quoted = '"' in query or "'" in query
        has_codes = bool(re.search(r"\b(code|id|version|v\d|http|api|key)\b", ql))
        is_conceptual = bool(re.search(r"\b(what|why|how|explain|describe|summarize|preference|feel|think)\b", ql))
        is_latest = bool(re.search(r"\b(latest|current|now|newest|recent|today)\b", ql))
        is_historical = bool(re.search(r"\b(first|earliest|initial|original|before|used to|previous|previously|former|formerly)\b", ql))
        is_multi = bool(re.search(r"\b(both|all|each|every|multiple|across|between)\b", ql)) or query.count("?") > 1
        fts_w = 1.0
        vec_w = 1.0
        if has_identifiers or has_quoted or has_codes:
            fts_w = _RETRIEVAL_CFG.fts_weight_identifiers
        if is_conceptual:
            vec_w = _RETRIEVAL_CFG.vector_weight_conceptual
        return {"fts": fts_w, "vector": vec_w, "is_latest": float(is_latest), "is_historical": float(is_historical), "is_multi": float(is_multi)}

    def _chunk_hits(self, namespace_id: str, query: str, query_vector: list[float] | None, limit: int = 20) -> dict[str, float]:
        """Score chunks via lexical + dense (raw + contextual) and map to memory versions."""
        chunk_rows = self.db.execute("SELECT chunk_id, raw_text, contextual_text, event_ids_json FROM chunks WHERE namespace_id=?", (namespace_id,)).fetchall()
        if not chunk_rows:
            return {}
        terms = [t.casefold() for t in re.findall(r"[\w./:-]+", query) if len(t) > 1 and t.casefold() not in SEARCH_STOP_WORDS]
        # Batch fetch chunk embeddings
        chunk_vectors: dict[str, list[bytes]] = {}
        if query_vector is not None:
            try:
                for row in self.db.execute("SELECT chunk_id, vector FROM chunk_embeddings WHERE namespace_id=? AND provider=?", (namespace_id, self.embedding.name)).fetchall():
                    chunk_vectors.setdefault(str(row["chunk_id"]), []).append(bytes(row["vector"]))
            except Exception:
                pass
        chunk_scores: dict[str, float] = {}
        for row in chunk_rows:
            lexical = 0.0
            if terms:
                text = (str(row["raw_text"] or "") + " " + str(row["contextual_text"] or "")).casefold()
                lexical = sum(t in text for t in terms) / max(1, len(terms))
            dense = 0.0
            if query_vector is not None and str(row["chunk_id"]) in chunk_vectors:
                import array as _array

                from ..retrieval.embedding import cosine as _cosine

                for vec_bytes in chunk_vectors[str(row["chunk_id"])]:
                    try:
                        vec = list(_array.array("f", vec_bytes))
                        dense = max(dense, _cosine(query_vector, vec))
                    except Exception:
                        continue
            score = _RETRIEVAL_CFG.chunk_vector_weight * dense + _RETRIEVAL_CFG.chunk_lexical_weight * lexical
            if score > 0.15:
                chunk_scores[str(row["chunk_id"])] = score
        # Map chunk scores to memory version IDs via event overlap
        if not chunk_scores:
            return {}
        # Build event->memory version map
        top_chunks = sorted(chunk_scores.items(), key=lambda x: -x[1])[: limit * 2]
        # Resolve chunk -> event through the normalized mapping, avoiding a
        # Python scan of all chunks for every query.
        top_chunk_ids = [chunk_id for chunk_id, _ in top_chunks]
        placeholders = ",".join("?" for _ in top_chunk_ids)
        chunk_event_map: dict[str, list[str]] = {}
        for row in self.db.execute(
            f"SELECT chunk_id, event_id FROM chunk_events WHERE namespace_id=? AND chunk_id IN ({placeholders})",
            (namespace_id, *top_chunk_ids),
        ).fetchall():
            chunk_event_map.setdefault(str(row["chunk_id"]), []).append(str(row["event_id"]))
        memory_chunk_scores: dict[str, float] = {}
        for chunk_id, cscore in top_chunks:
            for event_id in chunk_event_map.get(chunk_id, []):
                rows = self.db.execute(
                    "SELECT v.id AS version_id FROM memory_versions v JOIN evidence_refs r ON r.memory_version_id=v.id WHERE v.namespace_id=? AND r.event_id=?", (namespace_id, event_id)
                ).fetchall()
                for vrow in rows:
                    vid = str(vrow["version_id"])
                    memory_chunk_scores[vid] = max(memory_chunk_scores.get(vid, 0.0), cscore)
                # Also check source_event_id
                rows2 = self.db.execute("SELECT id AS version_id FROM memory_versions WHERE namespace_id=? AND source_event_id=?", (namespace_id, event_id)).fetchall()
                for vrow in rows2:
                    vid = str(vrow["version_id"])
                    memory_chunk_scores[vid] = max(memory_chunk_scores.get(vid, 0.0), cscore)
        return memory_chunk_scores

    def _rerank_candidates(
        self, query: str, results: list[SearchResult], namespace_id: str
    ) -> list[SearchResult]:
        """Cross-encoder rerank using question + memory + contextual chunk text. Configurable via FlashRank."""
        if not _RETRIEVAL_CFG.reranker_enabled or not results:
            return results
        try:
            from flashrank import Ranker, RerankRequest  # type: ignore[import-untyped]
        except ImportError:
            return results
        # Build chunk text map for richer passages
        chunk_texts: dict[str, str] = {}
        for row in self.db.execute("SELECT chunk_id, contextual_text, raw_text FROM chunks WHERE namespace_id=?", (namespace_id,)).fetchall():
            chunk_texts[str(row["chunk_id"])] = str(row["contextual_text"] or row["raw_text"] or "")
        # Build passages: memory statement + contextual chunk
        ranker_model = _RETRIEVAL_CFG.reranker_model
        ranker = _cached_reranker(ranker_model)
        if ranker is None:
            return results
        max_candidates = _RETRIEVAL_CFG.reranker_max_candidates
        max_chars = _RETRIEVAL_CFG.reranker_max_chars
        candidates = results[:max_candidates]
        tail = results[max_candidates:]
        # Map chunk contexts to each result via source events
        passages = []
        for hit in candidates:
            extra = ""
            for event_id in hit.source_event_ids[:2]:
                for row in self.db.execute(
                    "SELECT chunk_id FROM chunk_events WHERE namespace_id=? AND event_id=? LIMIT 1",
                    (namespace_id, str(event_id)),
                ).fetchall():
                    ctx = chunk_texts.get(str(row["chunk_id"]), "")
                    if ctx:
                        extra = ctx[:300]
                        break
                if extra:
                    break
            text = hit.statement if not extra else f"{hit.statement} | {extra}"
            passages.append({"id": str(hit.memory_version_id), "text": text[:max_chars]})
        try:
            reranked_raw = ranker.rerank(RerankRequest(query=query, passages=passages))
        except Exception:
            return results
        score_by_id = {str(entry["id"]): float(entry.get("score", 0.0)) for entry in reranked_raw}
        # Apply reranker scores, keep threshold filtering for abstention only at API layer
        reranked = sorted(candidates, key=lambda h: (-score_by_id.get(str(h.memory_version_id), 0.0), -h.score))
        # Update component_scores with reranker
        for hit in reranked:
            hit.component_scores["reranker"] = score_by_id.get(str(hit.memory_version_id), 0.0)
        return reranked + tail

    def search(self, namespace_id: str, query: str, limit: int, historical: bool = False, *, internal: bool = False) -> list[SearchResult]:
        # Each term becomes one SQL/FTS expression. SQLite has a hard maximum
        # expression depth, so cap user and internal transcript queries alike.
        terms = list(dict.fromkeys(term.casefold() for term in re.findall(r"[\w./:-]+", query) if len(term) > 1 and term.casefold() not in SEARCH_STOP_WORDS))[:64]
        qweights = self._query_weights(query)
        prefer_oldest = historical and bool(qweights["is_historical"]) or bool(re.search(r"\b(first|earliest|initial|original|before|used to|previous|previously|former|formerly)\b", query, re.I))
        prefer_latest = bool(qweights["is_latest"])
        is_multi = bool(qweights["is_multi"]) or len(re.findall(r"\b(session|conversation)\b", query.casefold())) > 1
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
                    (f"%{query.casefold()}%", namespace_id, *(f"%{term}%" for term in terms), max(limit * 5, 50)),
                ).fetchall()
            else:
                lexical_rows = self.db.execute(
                    "SELECT memory_version_id, bm25(memory_fts) AS score FROM memory_fts WHERE namespace_id=? AND memory_fts MATCH ? ORDER BY score LIMIT ?",
                    (namespace_id, match, max(limit * 5, 50)),
                ).fetchall()
            lexical_rank = {str(row["memory_version_id"]): index for index, row in enumerate(lexical_rows, start=1)}
            lexical = {memory_id: 1.0 / (1.0 + 0.08 * (rank - 1)) for memory_id, rank in lexical_rank.items()}

        vector: dict[str, float] = {}
        vector_rank: dict[str, int] = {}
        query_vector: list[float] | None = None
        try:
            query_vector = self.embedding.embed(query)
            indexed_rows = self.vector_index.search(namespace_id, query_vector, max(limit * 5, 50))
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

        # Phase 3: contextual chunk embeddings stream (third hybrid channel)
        chunk_scores: dict[str, float] = {}
        chunk_rank: dict[str, int] = {}
        try:
            chunk_scores = self._chunk_hits(namespace_id, query, query_vector, limit=max(limit * 3, 30))
            if chunk_scores:
                ordered_chunks = sorted(chunk_scores.items(), key=lambda x: (-x[1], x[0]))
                chunk_rank = {vid: idx for idx, (vid, _) in enumerate(ordered_chunks, start=1)}
        except Exception:
            chunk_scores = {}
            chunk_rank = {}

        # Keep semantic hits above the search floor in the candidate pool. The
        # reranker and evidence scores decide the final order; dropping 0.60-
        # 0.70 matches here loses paraphrased temporal and preference answers.
        candidate_ids = set(lexical) | {memory_id for memory_id, score in vector.items() if score >= 0.6} | set(chunk_scores.keys())
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
            f"""SELECT m.id, m.kind, m.confidence, m.importance, v.id AS version_id, v.statement, v.status, v.valid_from, v.valid_to, v.recorded_at, v.source_event_id, v.evidence_excerpt
            FROM memory_versions v JOIN memories m ON m.id=v.memory_id AND m.namespace_id=?
            WHERE v.namespace_id=? AND v.id IN ({placeholders}) AND (? OR (v.status='active' AND m.status='active'
              AND m.accessibility >= 0.05 AND v.valid_to IS NULL
              AND (v.valid_from IS NULL OR julianday(v.valid_from) <= julianday('now'))
              AND (v.valid_until IS NULL OR julianday(v.valid_until) > julianday('now'))))""",
            (namespace_id, namespace_id, *candidate_ids, historical),
        ).fetchall()
        # Debt 2 & 5 fix: removed per-hit graph/summary N+1 queries.
        # Hybrid retrieval is now strictly FTS + Vector + Temporal recency.
        # Graph/episode signals are gated behind explicit extension flag and
        # no longer fetched in hot path. See `record_graph_links` for opt-in.
        query_years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", query)]

        def score_row(row: sqlite3.Row) -> float:
            version_id = str(row["version_id"])
            k = _RETRIEVAL_CFG.rrf_k
            # Query-aware weights: boost FTS for identifiers, vector for conceptual
            fts_w = qweights["fts"]
            vec_w = qweights["vector"]
            lexical_rrf = (1.0 / (k + lexical_rank[version_id]) * fts_w) if version_id in lexical_rank else 0.0
            vector_rrf = (1.0 / (k + vector_rank[version_id]) * vec_w) if version_id in vector_rank else 0.0
            alias_rrf = 1.0 / (k + alias_rank[version_id]) if version_id in alias_rank else 0.0
            chunk_rrf = 1.0 / (k + chunk_rank[version_id]) if version_id in chunk_rank else 0.0
            active_channels = sum((version_id in lexical_rank, version_id in vector_rank, version_id in alias_rank, version_id in chunk_rank))
            # Weighted fusion
            raw_fused = lexical_rrf + vector_rrf + alias_rrf + chunk_rrf
            fused = k * raw_fused / max(1, active_channels)
            # Multi-channel bonus: memories that appear in multiple streams get small boost
            if active_channels >= 2:
                fused += 0.02 * (active_channels - 1)
            exact = 0.08 if query.casefold() in str(row["statement"]).casefold() else 0.0
            support = min(0.04, 0.01 * len(self._citations(namespace_id, version_id)))
            quality = 0.03 * float(row["confidence"]) + 0.02 * float(row["importance"])
            history = 0.0
            if historical and row["valid_to"] is not None:
                history = 0.02 if prefer_oldest else 0.01
            # Temporal versioning: boost latest or historical based on query intent
            temporal_boost = 0.0
            if prefer_latest and row["valid_to"] is None and row["status"] == "active":
                temporal_boost = 0.03
            elif prefer_oldest and row["valid_to"] is not None:
                temporal_boost = 0.02
            year_match = 0.0
            if query_years:
                valid_from = str(row["valid_from"] or "")
                valid_to = str(row["valid_to"] or "")
                if any(str(year) in valid_from or str(year) in valid_to for year in query_years):
                    year_match = 0.05
            # Debt 3 fix: real temporal recency decay via TemporalBlock, not placeholder.
            recency = 0.0
            try:
                if row["valid_from"]:
                    recency = temporal_recency_score(datetime.fromisoformat(str(row["valid_from"])))
                    # Amplify recency for "latest" queries
                    if prefer_latest:
                        recency *= 1.5
            except Exception:
                recency = 0.0
            chunk_signal = chunk_scores.get(version_id, 0.0) * 0.05
            return min(1.0, fused + exact + support + quality + history + temporal_boost + year_match + recency + chunk_signal)

        # Primary sort by score, secondary by valid_from. Historical queries that
        # ask for "first" or "before" should prefer older evidence instead of the
        # default newest-first tie break.
        rows_by_recency = sorted(rows, key=lambda row: row["valid_from"] or "", reverse=not prefer_oldest)
        # Fuse top 50-100 candidates then rerank. Keep larger pool for reranker.
        ranked = sorted(rows_by_recency, key=lambda row: -score_row(row))
        query_terms = set(terms)
        results: list[SearchResult] = []
        for row in ranked:
            citations = self._citations(namespace_id, row["version_id"])
            lexical_score = round(lexical.get(row["version_id"], 0.0), 6)
            vector_score = round(vector.get(row["version_id"], 0.0), 6)
            chunk_score = round(chunk_scores.get(str(row["version_id"]), 0.0), 6)
            final_score = round(score_row(row), 6)
            source_ids = [c.event_id for c in citations]
            sid = row["source_event_id"]
            excerpt_val = row["evidence_excerpt"]
            if sid and uuid.UUID(sid) not in source_ids:
                source_ids.append(uuid.UUID(sid))
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
                        "chunk_score": chunk_score,
                        "chunk_rank": float(chunk_rank.get(str(row["version_id"]), 0)),
                        "exact_match": float(query.casefold() in str(row["statement"]).casefold()),
                        "temporal_boost": float(0.03 if prefer_latest and row["valid_to"] is None else 0.0),
                        "recency": round(temporal_recency_score(datetime.fromisoformat(str(row["valid_from"]))) if row["valid_from"] else 0.0, 6) if row["valid_from"] else 0.0,
                        "reranker": 0.0,
                        "graph_proximity": 0.0,
                        "session_summary": 0.0,
                    },
                    status=row["status"],
                    citations=citations,
                    source_event_ids=source_ids,
                    evidence_excerpt=excerpt_val,
                )
            )
        # Phase 3c: Rerank chunks (not whole raw sessions, not only memory titles)
        # Use FlashRank if available; configurable via RETRIEVAL settings
        results = self._rerank_candidates(query, results, namespace_id)
        # Re-sort by final score (reranker may have reordered)
        results = sorted(results, key=lambda r: (r.component_scores.get("reranker", 0.0), r.score), reverse=True)
        # Phase 3c: diversity + dedup — no more than 2 chunks from one session unless multi-session
        # Deduplicate equivalent memories/chunks by statement
        seen_statements: set[str] = set()
        deduped: list[SearchResult] = []
        for r in results:
            key = r.statement.strip().casefold()
            if key in seen_statements:
                continue
            seen_statements.add(key)
            deduped.append(r)
        results = deduped
        # Diversity: limit emitted chunks per session, not memories
        if not is_multi:
            max_chunks_per_session = _RETRIEVAL_CFG.diversity_max_per_session
            # Track chunk counts per session (2 chunks per memory * 2 memories = 4 chunks would exceed cap)
            chunk_session_counts: dict[str, int] = {}
            event_session: dict[str, str] = {}
            for row in self.db.execute("SELECT id, COALESCE(stream_id, session_id, id) AS sid FROM events WHERE namespace_id=?", (namespace_id,)).fetchall():
                event_session[str(row["id"])] = str(row["sid"])
            # Also build chunk_id -> session map
            chunk_session: dict[str, str] = {}
            for row in self.db.execute("SELECT chunk_id, session_id FROM chunks WHERE namespace_id=?", (namespace_id,)).fetchall():
                chunk_session[str(row["chunk_id"])] = str(row["session_id"])
            diverse: list[SearchResult] = []
            for r in results:
                # Count how many chunks this result would emit (from chunks_for_events, up to 2)
                # Use source_event_ids to derive chunk sessions
                chunk_sessions_for_result: list[str] = []
                for eid in r.source_event_ids[:2]:
                    for crow in self.db.execute(
                        "SELECT chunk_id FROM chunk_events WHERE namespace_id=? AND event_id=? LIMIT 1",
                        (namespace_id, str(eid)),
                    ).fetchall():
                        sid = chunk_session.get(str(crow["chunk_id"]), "")
                        if sid:
                            chunk_sessions_for_result.append(sid)
                            break
                if not chunk_sessions_for_result:
                    # Fallback to event session
                    for eid in r.source_event_ids:
                        sid = event_session.get(str(eid), "")
                        if sid:
                            chunk_sessions_for_result.append(sid)
                            break
                # If any session would exceed chunk cap, skip this memory
                would_exceed = False
                for sid in chunk_sessions_for_result:
                    # Each result emits up to len(chunk_sessions_for_result) chunks (1-2)
                    would_be = chunk_session_counts.get(sid, 0) + len(chunk_sessions_for_result)
                    if would_be > max_chunks_per_session and sid:
                        # But if this is the only session with results, allow overflow by 1 to avoid empty
                        if len(diverse) == 0:
                            break
                        would_exceed = True
                        break
                if would_exceed:
                    continue
                for sid in chunk_sessions_for_result:
                    if sid:
                        chunk_session_counts[sid] = chunk_session_counts.get(sid, 0) + 1
                # Also handle case where no chunk sessions found - count by event session once
                if not chunk_sessions_for_result:
                    primary = ""
                    for eid in r.source_event_ids:
                        primary = event_session.get(str(eid), "")
                        if primary:
                            break
                    if primary and chunk_session_counts.get(primary, 0) >= max_chunks_per_session:
                        continue
                    if primary:
                        chunk_session_counts[primary] = chunk_session_counts.get(primary, 0) + 1
                diverse.append(r)
            results = diverse
        results = results[:limit]
        if not internal:
            self.record_retrieval(namespace_id, [str(item.memory_id) for item in results], successful=bool(results))
        return results

    def search_sessions(self, namespace_id: str, query: str, limit: int = 20) -> list[SessionSearchResult]:
        """Search raw conversation sessions without requiring LLM extraction.

        Events are the durable source of truth. This small lexical fallback is
        deliberately independent of embeddings so a failed/empty extraction
        never makes a session disappear from retrieval.
        """
        terms = [term.casefold() for term in re.findall(r"[\w./:-]+", query) if len(term) > 1 and term.casefold() not in SEARCH_STOP_WORDS]
        rows = self.db.execute(
            "SELECT id, stream_id, session_id, occurred_at, payload_json, type FROM events WHERE namespace_id=? ORDER BY occurred_at, id",
            (namespace_id,),
        ).fetchall()
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            session_id = str(row["stream_id"] or row["session_id"] or row["id"])
            item = grouped.setdefault(session_id, {"event_ids": [], "parts": [], "occurred_at": row["occurred_at"]})
            item["event_ids"].append(uuid.UUID(row["id"]))
            try:
                item["parts"].append(payload_text(json.loads(row["payload_json"]), row["type"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        hits: list[SessionSearchResult] = []
        for session_id, item in grouped.items():
            text = "\n".join(part for part in item["parts"] if part).strip()
            if not text:
                continue
            lowered = text.casefold()
            matched = sum(term in lowered for term in terms)
            if terms and not matched:
                continue
            score = matched / max(1, len(terms))
            hits.append(SessionSearchResult(session_id=session_id, event_ids=item["event_ids"], text=text, occurred_at=item["occurred_at"], score=score))
        return sorted(hits, key=lambda hit: (-hit.score, hit.session_id))[:limit]

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

    def expand_relationships(
        self,
        namespace_id: str,
        seed_memory_version_ids: list[str],
        query: str = "",
        max_hops: int = 1,
        max_results: int = 5,
        token_budget: int = 500,
    ) -> list[dict[str, Any]]:
        """Bounded relationship expansion after high-confidence hit.

        1. resolve linked entities from seed memories
        2. traverse one hop (default) and at most two for multi-session questions
        3. filter by allowed relation types and valid time range
        4. include only expanded evidence with source chunks
        Never traverses full namespace graph.
        """
        if not seed_memory_version_ids:
            return []
        # Determine hops: at most 2 for multi-session
        is_multi = bool(__import__("re").search(r"\b(both|all|each|every|multiple|across|between|multi.session)\b", query.casefold())) if query else False
        hops = max(1, min(max_hops, 2 if is_multi else 1))
        allowed_predicates = {"contains", "updates", "expresses", "related", "extends", "derives"}
        expanded: list[dict[str, Any]] = []
        seen_versions: set[str] = set(seed_memory_version_ids)
        # Resolve entities linked to seed memory versions
        seed_entities: set[str] = set()
        for vid in seed_memory_version_ids[:3]:  # limit seed fanout
            for row in self.db.execute(
                "SELECT subject_entity_id, object_entity_id FROM relationships WHERE namespace_id=? AND memory_version_id=? AND status='active'",
                (namespace_id, vid),
            ).fetchall():
                seed_entities.add(str(row["subject_entity_id"]))
                seed_entities.add(str(row["object_entity_id"]))
        if not seed_entities:
            return []
        frontier = list(seed_entities)
        for _ in range(hops):
            if not frontier or len(expanded) >= max_results:
                break
            placeholders = ",".join("?" for _ in frontier)
            predicate_placeholders = ",".join("?" for _ in allowed_predicates)
            # Build NOT IN clause for seen_versions; SQLite cannot have UNION inside IN list
            if seen_versions:
                not_in_clause = f"AND r.memory_version_id NOT IN ({','.join('?' for _ in seen_versions)})"
                not_in_params: list[Any] = list(seen_versions)
            else:
                not_in_clause = ""
                not_in_params = []
            try:
                rows = self.db.execute(
                    f"""SELECT r.*, s.label AS subject_label, o.label AS object_label
                        FROM relationships r JOIN entities s ON s.id=r.subject_entity_id JOIN entities o ON o.id=r.object_entity_id
                        WHERE r.namespace_id=? AND r.status='active' AND r.predicate IN ({predicate_placeholders})
                          AND (r.subject_entity_id IN ({placeholders}) OR r.object_entity_id IN ({placeholders}))
                          AND r.memory_version_id IS NOT NULL
                          {not_in_clause}
                        ORDER BY r.confidence DESC LIMIT ?""",
                    (namespace_id, *allowed_predicates, *frontier, *frontier, *not_in_params, max_results - len(expanded)),
                ).fetchall() if seed_entities else []
            except sqlite3.OperationalError:
                # Fallback without predicate filter if schema differs
                rows = self.db.execute(
                    f"""SELECT r.*, s.label AS subject_label, o.label AS object_label
                        FROM relationships r JOIN entities s ON s.id=r.subject_entity_id JOIN entities o ON o.id=r.object_entity_id
                        WHERE r.namespace_id=? AND r.status='active'
                          AND (r.subject_entity_id IN ({placeholders}) OR r.object_entity_id IN ({placeholders}))
                        LIMIT ?""",
                    (namespace_id, *frontier, *frontier, max_results - len(expanded)),
                ).fetchall() if seed_entities else []
            next_frontier: list[str] = []
            for row in rows:
                vid = str(row["memory_version_id"] or "")
                if not vid or vid in seen_versions:
                    continue
                if row["predicate"] not in allowed_predicates:
                    continue
                # Only include if source chunk exists for this memory's evidence
                # Correlate candidate version's evidence/source event with a chunk
                v_source_event = self.db.execute(
                    "SELECT source_event_id FROM memory_versions WHERE id=? AND namespace_id=?",
                    (vid, namespace_id),
                ).fetchone()
                source_event_id = str(v_source_event["source_event_id"]) if v_source_event and v_source_event["source_event_id"] else None
                # Check evidence_refs first, then source_event_id
                evidence_event_ids: list[str] = []
                if source_event_id:
                    evidence_event_ids.append(source_event_id)
                for er in self.db.execute("SELECT event_id FROM evidence_refs WHERE memory_version_id=? AND namespace_id=?", (vid, namespace_id)).fetchall():
                    evidence_event_ids.append(str(er["event_id"]))
                has_chunk = False
                for eid in evidence_event_ids:
                    exists = self.db.execute(
                        "SELECT 1 FROM chunks WHERE namespace_id=? AND event_ids_json LIKE ? LIMIT 1",
                        (namespace_id, f"%{eid}%"),
                    ).fetchone()
                    if exists:
                        has_chunk = True
                        break
                if not has_chunk:
                    continue
                seen_versions.add(vid)
                # Fetch memory version for evidence
                vrow = self.db.execute(
                    "SELECT m.id, v.statement, v.source_event_id, v.evidence_excerpt, v.valid_from, v.valid_until, v.recorded_at FROM memory_versions v JOIN memories m ON m.id=v.memory_id WHERE v.id=? AND v.namespace_id=?",
                    (vid, namespace_id),
                ).fetchone()
                if not vrow:
                    continue
                item: dict[str, Any] = dict(row)
                item["statement"] = vrow["statement"]
                item["source_event_id"] = vrow["source_event_id"]
                item["memory_id"] = vrow["id"]
                item["memory_version_id"] = vid
                # Token guard
                if sum(len(str(e.get("statement", "")).split()) for e in expanded) + len(str(vrow["statement"]).split()) > token_budget:
                    break
                expanded.append(item)
                # Expand frontier
                other = str(row["object_entity_id"]) if str(row["subject_entity_id"]) in frontier else str(row["subject_entity_id"])
                if other not in seen_versions:
                    next_frontier.append(other)
            frontier = next_frontier
        return expanded[:max_results]

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
        # Debt 5: graph is opt-in extension, not hot-path. Keep write amplification
        # out of default ingest by gating behind env flag. Tests that use
        # upsert_entity/related_entities still work because they call the API
        # directly; ingest now skips the 3-entity upsert + 2 relationships.
        import os

        if os.environ.get("TERMYTEDB_ENABLE_GRAPH", "0") not in {"1", "true", "True"}:
            return
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

    def update_memory(self, namespace_id: str, memory_id: str, statement: str, confidence: float | None = None, kind: str | None = None, source_event_id: str | None = None, evidence_excerpt: str | None = None) -> bool:
        """Phase 3: direct memory update creates a new version and supersedes the old.
        Stores optional provenance without requiring evidence offsets."""
        if not statement or not statement.strip():
            raise ValueError("statement must be non-empty")
        if confidence is not None and not (0 <= confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
        with self.db.lock, self.db.connection:
            mem = self.db.execute("SELECT * FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)).fetchone()
            if not mem:
                return False
            current = self.db.execute("SELECT * FROM memory_versions WHERE id=? AND namespace_id=?", (mem["current_version_id"], namespace_id)).fetchone()
            if not current:
                return False
            new_kind = kind or mem["kind"]
            new_conf = confidence if confidence is not None else mem["confidence"]
            # Create new version
            version = current["version"] + 1
            version_id = str(uuid.uuid4())
            now = iso()
            src_id = source_event_id
            if src_id is None:
                src_id = current["source_event_id"]
            # P1: validate provenance event is in the same namespace before attaching
            if src_id is not None:
                provenance = self.db.execute(
                    "SELECT 1 FROM events WHERE id=? AND namespace_id=?",
                    (src_id, namespace_id),
                ).fetchone()
                if not provenance:
                    raise ValueError("source_event_id is not in the requested namespace")
            excerpt = evidence_excerpt if evidence_excerpt is not None else current["evidence_excerpt"] or statement[:500]
            # Normalize provenance offsets to be consistent with excerpt; synthetic when evidence is optional
            if src_id is not None:
                if not excerpt:
                    excerpt = statement[:500] or "provenance"
                start_off = 0
                end_off = len(excerpt)
                if end_off <= start_off:
                    end_off = start_off + 1
            else:
                start_off = current["evidence_start_offset"]
                end_off = current["evidence_end_offset"]
            # Supersede old
            self.db.execute("UPDATE memory_versions SET status='superseded', valid_to=? WHERE id=? AND namespace_id=?", (now, current["id"], namespace_id))
            self.db.execute("DELETE FROM memory_fts WHERE memory_version_id=? AND namespace_id=?", (current["id"], namespace_id))
            self.db.execute(
                """INSERT INTO memory_versions
                (id, memory_id, namespace_id, source_event_id, evidence_start_offset, evidence_end_offset, evidence_excerpt,
                 version, statement, valid_from, recorded_at, status, reason, durability, model_run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 'update', 'permanent', NULL)""",
                (version_id, memory_id, namespace_id, src_id, start_off, end_off, excerpt, version, statement, now, now),
            )
            # P2: preserve citations for updated version when provenance is available
            if src_id is not None:
                # Ensure evidence_refs has a corresponding row so get_memory/list_memories return citations
                ev_excerpt = excerpt
                ev_start = start_off if isinstance(start_off, int) else 0
                ev_end = end_off if isinstance(end_off, int) else len(ev_excerpt)
                if ev_end <= ev_start:
                    ev_end = ev_start + (len(ev_excerpt) if ev_excerpt else 1)
                self.db.execute(
                    """INSERT OR IGNORE INTO evidence_refs
                    (id, memory_version_id, namespace_id, event_id, start_offset, end_offset, excerpt)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), version_id, namespace_id, src_id, ev_start, ev_end, ev_excerpt),
                )
            # Update memory row
            self.db.execute(
                "UPDATE memories SET kind=?, confidence=?, current_version_id=?, status='active' WHERE id=? AND namespace_id=?",
                (new_kind, new_conf, version_id, memory_id, namespace_id),
            )
            self.db.execute(
                "INSERT INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?, ?, ?, ?)",
                (version_id, namespace_id, statement, excerpt or ""),
            )
            # Persist embedding for new version
            self._persist_embedding(version_id, namespace_id, None, statement)
            return True

    def delete_memory(self, namespace_id: str, memory_id: str) -> bool:
        """Target API alias for forget — tombstones memory while retaining history."""
        return self.forget_memory(namespace_id, memory_id, "delete_memory")

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

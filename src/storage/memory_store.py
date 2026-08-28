"""Focused memory store - owns memories, versions, FTS and embeddings.

Single responsibility: persisting and reconciling ExtractionCandidates.
Expensive graph/procedure extensions are gated to keep hot path simple.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from ..core.redaction import redact_text
from ..memory.extraction import CandidateRejected, ValidatedCandidate
from ..models import EvidenceCitation, MemoryResponse, TemporalBlock, temporal_recency_score
from ..retrieval.embedding import EmbeddingProvider, FastEmbedProvider, pack_embedding
from .db import Database
from .vector_index import SQLiteVecIndex


def iso(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()


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


class MemoryStore:
    def __init__(self, db: Database, embedding: EmbeddingProvider | None = None):
        self.db = db
        self.embedding = embedding or FastEmbedProvider()
        self.vector_index = SQLiteVecIndex(self.db, self.embedding)
        self.vector_index.ensure()

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

    def get_memory(self, namespace_id: str, memory_id: str) -> MemoryResponse | None:
        row = self.db.execute(
            """SELECT m.*, v.id AS version_id, v.version, v.statement, v.status AS version_status,
                      v.valid_from, v.valid_until, v.recorded_at
            FROM memories m JOIN memory_versions v ON v.id=m.current_version_id
            WHERE m.id=? AND m.namespace_id=? AND v.namespace_id=?""",
            (memory_id, namespace_id, namespace_id),
        ).fetchone()
        if not row:
            return None
        temporal = None
        try:
            temporal = TemporalBlock(
                valid_from=datetime.fromisoformat(str(row["valid_from"])) if row["valid_from"] else None,
                valid_until=datetime.fromisoformat(str(row["valid_until"])) if row["valid_until"] else None,
                recorded_at=datetime.fromisoformat(str(row["recorded_at"])) if row["recorded_at"] else None,
            )
        except Exception:
            temporal = None
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
            citations=self._citations(namespace_id, row["version_id"]),
            temporal=temporal,
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

    def _citations(self, namespace_id: str, version_id: str) -> list[EvidenceCitation]:
        rows = self.db.execute(
            """SELECT r.event_id, r.start_offset, r.end_offset, r.excerpt
            FROM evidence_refs r JOIN events e ON e.id=r.event_id
            WHERE r.memory_version_id=? AND r.namespace_id=? AND e.namespace_id=? ORDER BY r.id""",
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
                result = self._reconcile_in_tx(namespace_id, event, candidate, run_id, embedding)
                self.db.connection.commit()
                return result
            except Exception:
                self.db.connection.rollback()
                raise

    def _reconcile_in_tx(
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
                    """INSERT OR IGNORE INTO evidence_refs (id, memory_version_id, namespace_id, event_id, start_offset, end_offset, excerpt)
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
        # Graph links are now opt-in (TERMYTEDB_ENABLE_GRAPH=1) to avoid write amplification.
        return memory_id, action, version_id

    def memory_count(self, namespace_id: str) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM memories WHERE namespace_id=?", (namespace_id,)).fetchone()[0])

    def list_versions(self, namespace_id: str, memory_id: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM memory_versions WHERE memory_id=? AND namespace_id=? ORDER BY version", (memory_id, namespace_id)).fetchall()

    def invalidate_memory(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        with self.db.lock, self.db.connection:
            row = self.db.execute("SELECT current_version_id FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)).fetchone()
            if not row:
                return False
            self.db.execute("UPDATE memories SET status='invalidated' WHERE id=? AND namespace_id=?", (memory_id, namespace_id))
            self.db.execute("UPDATE memory_versions SET status='invalidated', reason=? WHERE memory_id=? AND namespace_id=?", (reason, memory_id, namespace_id))
            self.db.execute("DELETE FROM memory_fts WHERE namespace_id=? AND memory_version_id=?", (namespace_id, row["current_version_id"]))
            return True

    def forget_memory(self, namespace_id: str, memory_id: str, reason: str) -> bool:
        with self.db.lock, self.db.connection:
            row = self.db.execute("SELECT current_version_id FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)).fetchone()
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
        with self.db.lock, self.db.connection:
            memory = self.db.execute("SELECT id FROM memories WHERE id=? AND namespace_id=?", (memory_id, namespace_id)).fetchone()
            version = self.db.execute(
                "SELECT * FROM memory_versions WHERE memory_id=? AND namespace_id=? AND status != 'invalidated' ORDER BY version DESC LIMIT 1",
                (memory_id, namespace_id),
            ).fetchone()
            if not memory or not version:
                return False
            self.db.execute("UPDATE memory_versions SET status='superseded' WHERE memory_id=? AND namespace_id=? AND id<>? AND status='active'", (memory_id, namespace_id, version["id"]))
            self.db.execute("UPDATE memory_versions SET status='active', valid_to=NULL, reason='RESTORE' WHERE id=? AND namespace_id=?", (version["id"], namespace_id))
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

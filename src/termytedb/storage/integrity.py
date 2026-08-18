from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from ..api.schemas import ArtifactInput, EventInput
from ..retrieval.embedding import LocalHashEmbedding
from .db import MIGRATIONS, Database
from .repository import canonical_event_content, hash_text


@dataclass(frozen=True)
class IntegrityReport:
    schema_version: int
    foreign_key_errors: list[str]
    sqlite_errors: list[str]
    orphan_evidence: int
    missing_evidence: int
    orphan_fts: int
    missing_fts: int
    orphan_embeddings: int
    missing_embeddings: int
    event_hash_mismatches: int
    schema_compatible: bool

    @property
    def ok(self) -> bool:
        return not (
            self.foreign_key_errors
            or self.sqlite_errors
            or self.orphan_evidence
            or self.missing_evidence
            or self.orphan_fts
            or self.missing_fts
            or self.orphan_embeddings
            or self.missing_embeddings
            or self.event_hash_mismatches
            or not self.schema_compatible
        )


def check_database(database: Database) -> IntegrityReport:
    connection = database.connection
    schema_rows = connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    versions = [int(row[0]) for row in schema_rows]
    schema_version = max(versions, default=0)
    compatible = versions == list(range(1, len(MIGRATIONS) + 1)) and schema_version == len(MIGRATIONS)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_versions)")}
    compatible = compatible and {
        "source_event_id",
        "evidence_start_offset",
        "evidence_end_offset",
        "evidence_excerpt",
    }.issubset(columns)
    foreign_key_errors = [str(tuple(row)) for row in connection.execute("PRAGMA foreign_key_check")]
    sqlite_errors = [str(row[0]) for row in connection.execute("PRAGMA integrity_check") if row[0] != "ok"]
    orphan_evidence = int(
        connection.execute(
            """SELECT COUNT(*) FROM evidence_refs r
            LEFT JOIN memory_versions v ON v.id=r.memory_version_id AND v.namespace_id=r.namespace_id
            LEFT JOIN events e ON e.id=r.event_id AND e.namespace_id=r.namespace_id
            WHERE v.id IS NULL OR e.id IS NULL"""
        ).fetchone()[0]
    )
    orphan_fts = int(
        connection.execute(
            """SELECT COUNT(*) FROM memory_fts f
            LEFT JOIN memory_versions v ON v.id=f.memory_version_id AND v.namespace_id=f.namespace_id
            WHERE v.id IS NULL OR v.status != 'active'"""
        ).fetchone()[0]
    )
    missing_evidence = int(
        connection.execute(
            """SELECT COUNT(*) FROM memory_versions v
            LEFT JOIN evidence_refs r ON r.memory_version_id=v.id AND r.namespace_id=v.namespace_id
            WHERE r.id IS NULL OR v.source_event_id IS NULL
               OR v.evidence_start_offset IS NULL OR v.evidence_end_offset IS NULL
               OR v.evidence_excerpt IS NULL"""
        ).fetchone()[0]
    )
    missing_fts = int(
        connection.execute(
            """SELECT COUNT(*) FROM memory_versions v
            JOIN memories m ON m.current_version_id=v.id AND m.namespace_id=v.namespace_id
            LEFT JOIN memory_fts f ON f.memory_version_id=v.id AND f.namespace_id=v.namespace_id
            WHERE v.status='active' AND m.status='active' AND v.valid_to IS NULL AND f.memory_version_id IS NULL"""
        ).fetchone()[0]
    )
    orphan_embeddings = int(
        connection.execute(
            """SELECT COUNT(*) FROM memory_embeddings e
            LEFT JOIN memory_versions v ON v.id=e.memory_version_id AND v.namespace_id=e.namespace_id
            WHERE v.id IS NULL OR v.status != 'active'"""
        ).fetchone()[0]
    )
    missing_embeddings = int(
        connection.execute(
            """SELECT COUNT(*) FROM memory_versions v
            JOIN memories m ON m.current_version_id=v.id AND m.namespace_id=v.namespace_id
            LEFT JOIN memory_embeddings e ON e.memory_version_id=v.id AND e.namespace_id=v.namespace_id
            WHERE v.status='active' AND m.status='active' AND v.valid_to IS NULL AND e.memory_version_id IS NULL"""
        ).fetchone()[0]
    )
    event_hash_mismatches = 0
    for event in connection.execute("SELECT * FROM events").fetchall():
        artifacts = connection.execute(
            "SELECT content_hash, media_type, size_bytes, uri, metadata_json FROM artifacts WHERE event_id=? AND namespace_id=? ORDER BY id",
            (event["id"], event["namespace_id"]),
        ).fetchall()
        event_input = EventInput(
            namespace_id=event["namespace_id"],
            protocol_version=event["protocol_version"],
            idempotency_key="integrity-check",
            type=event["type"],
            payload=json.loads(event["payload_json"]),
            # Omitted server ingestion time is intentionally excluded from identity.
            occurred_at=None,
            stream_id=event["stream_id"],
            actor_id=event["actor_id"],
            agent_id=event["agent_id"],
            session_id=event["session_id"],
            source_id=event["source_id"],
            artifacts=[ArtifactInput.model_validate({
                    "content_hash": item["content_hash"], "media_type": item["media_type"], "size_bytes": item["size_bytes"],
                    "uri": item["uri"], "metadata": json.loads(item["metadata_json"]),
                })
                for item in artifacts
            ],
        )
        if hash_text(canonical_event_content(event_input, event_input.payload)) != event["content_hash"]:
            event_hash_mismatches += 1
    return IntegrityReport(
        schema_version,
        foreign_key_errors,
        sqlite_errors,
        orphan_evidence,
        missing_evidence,
        orphan_fts,
        missing_fts,
        orphan_embeddings,
        missing_embeddings,
        event_hash_mismatches,
        compatible,
    )


def repair_fts(database: Database, embedding: LocalHashEmbedding | None = None) -> None:
    """Rebuild deterministic FTS and local embedding indexes from authority."""
    embedding = embedding or LocalHashEmbedding()
    with database.lock, database.connection:
        database.execute("DELETE FROM memory_fts")
        database.execute("DELETE FROM memory_embeddings")
        database.execute(
            """INSERT INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text)
            SELECT v.id, v.namespace_id, v.statement, v.statement
            FROM memory_versions v JOIN memories m
              ON m.current_version_id=v.id AND m.namespace_id=v.namespace_id
            WHERE v.status='active' AND m.status='active' AND v.valid_to IS NULL"""
        )
        rows = database.execute(
            """SELECT v.id, v.namespace_id, v.statement FROM memory_versions v JOIN memories m
            ON m.current_version_id=v.id AND m.namespace_id=v.namespace_id
            WHERE v.status='active' AND m.status='active' AND v.valid_to IS NULL"""
        ).fetchall()
        for row in rows:
            database.execute(
                "INSERT INTO memory_embeddings(memory_version_id, namespace_id, provider, dimensions, vector_json) VALUES (?, ?, ?, ?, ?)",
                (row["id"], row["namespace_id"], embedding.name, embedding.dimensions, json.dumps(embedding.embed(row["statement"]), separators=(",", ":"))),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or safely repair TermyteDB SQLite integrity")
    parser.add_argument("--database", required=True)
    parser.add_argument("--repair-fts", action="store_true")
    args = parser.parse_args()
    database = Database(args.database)
    try:
        if args.repair_fts:
            repair_fts(database)
        report = check_database(database)
        print(json.dumps(asdict(report), sort_keys=True))
        return 0 if report.ok else 1
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())

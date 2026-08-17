from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE namespaces (
      id TEXT PRIMARY KEY,
      org_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      deleted_at TEXT
    );
    CREATE TABLE events (
      id TEXT PRIMARY KEY,
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      stream_id TEXT,
      idempotency_hash TEXT NOT NULL,
      type TEXT NOT NULL,
      occurred_at TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      redaction_state TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(namespace_id, idempotency_hash)
    );
    CREATE INDEX events_namespace_idx ON events(namespace_id, occurred_at, id);
    CREATE TABLE memories (
      id TEXT PRIMARY KEY,
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      kind TEXT NOT NULL,
      subject_key TEXT NOT NULL,
      status TEXT NOT NULL,
      confidence REAL NOT NULL,
      current_version_id TEXT,
      created_at TEXT NOT NULL,
      UNIQUE(namespace_id, kind, subject_key)
    );
    CREATE INDEX memories_namespace_status_idx ON memories(namespace_id, status);
    CREATE TABLE memory_versions (
      id TEXT PRIMARY KEY,
      memory_id TEXT NOT NULL REFERENCES memories(id),
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      version INTEGER NOT NULL,
      statement TEXT NOT NULL,
      valid_from TEXT NOT NULL,
      valid_to TEXT,
      recorded_at TEXT NOT NULL,
      status TEXT NOT NULL,
      reason TEXT NOT NULL,
      UNIQUE(memory_id, version)
    );
    CREATE INDEX versions_namespace_status_idx ON memory_versions(namespace_id, status);
    CREATE TABLE evidence_refs (
      id TEXT PRIMARY KEY,
      memory_version_id TEXT NOT NULL REFERENCES memory_versions(id),
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      event_id TEXT NOT NULL REFERENCES events(id),
      start_offset INTEGER NOT NULL,
      end_offset INTEGER NOT NULL,
      excerpt TEXT NOT NULL,
      UNIQUE(memory_version_id, event_id, start_offset, end_offset)
    );
    CREATE INDEX evidence_namespace_idx ON evidence_refs(namespace_id, event_id);
    CREATE TABLE processing_jobs (
      id TEXT PRIMARY KEY,
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      event_id TEXT NOT NULL REFERENCES events(id),
      input_hash TEXT NOT NULL,
      status TEXT NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      max_attempts INTEGER NOT NULL DEFAULT 3,
      lease_until TEXT,
      last_error TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(namespace_id, event_id, input_hash)
    );
    CREATE INDEX jobs_claim_idx ON processing_jobs(namespace_id, status, lease_until, created_at);
    """,
    """
    CREATE VIRTUAL TABLE memory_fts USING fts5(
      memory_version_id UNINDEXED,
      namespace_id UNINDEXED,
      statement,
      evidence_text
    );
    """,
    """
    ALTER TABLE memory_versions ADD COLUMN source_event_id TEXT REFERENCES events(id);
    UPDATE memory_versions
    SET source_event_id = (
      SELECT event_id FROM evidence_refs
      WHERE evidence_refs.memory_version_id = memory_versions.id
      ORDER BY evidence_refs.id LIMIT 1
    )
    WHERE source_event_id IS NULL;
    CREATE TRIGGER evidence_refs_namespace_guard
    BEFORE INSERT ON evidence_refs
    BEGIN
      SELECT CASE WHEN
        (SELECT namespace_id FROM memory_versions WHERE id=NEW.memory_version_id) != NEW.namespace_id
        OR (SELECT namespace_id FROM events WHERE id=NEW.event_id) != NEW.namespace_id
        THEN RAISE(ABORT, 'evidence namespace or source mismatch') END;
    END;
    ALTER TABLE memory_versions ADD COLUMN evidence_start_offset INTEGER;
    ALTER TABLE memory_versions ADD COLUMN evidence_end_offset INTEGER;
    ALTER TABLE memory_versions ADD COLUMN evidence_excerpt TEXT;
    UPDATE memory_versions
    SET evidence_start_offset = (SELECT start_offset FROM evidence_refs WHERE memory_version_id=memory_versions.id ORDER BY id LIMIT 1),
        evidence_end_offset = (SELECT end_offset FROM evidence_refs WHERE memory_version_id=memory_versions.id ORDER BY id LIMIT 1),
        evidence_excerpt = (SELECT excerpt FROM evidence_refs WHERE memory_version_id=memory_versions.id ORDER BY id LIMIT 1)
    WHERE evidence_start_offset IS NULL;
    CREATE TRIGGER memory_versions_evidence_guard
    BEFORE INSERT ON memory_versions
    BEGIN
      SELECT CASE WHEN
        NEW.source_event_id IS NULL OR NEW.evidence_start_offset IS NULL OR
        NEW.evidence_end_offset IS NULL OR NEW.evidence_excerpt IS NULL OR
        NEW.evidence_end_offset <= NEW.evidence_start_offset OR
        (SELECT namespace_id FROM events WHERE id=NEW.source_event_id) != NEW.namespace_id
        THEN RAISE(ABORT, 'memory version requires same-namespace evidence') END;
    END;
    """,
)


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.migrate()

    def migrate(self) -> None:
        with self.connection:
            self.connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
            applied = {int(row[0]) for row in self.connection.execute("SELECT version FROM schema_migrations")}
            for version, sql in enumerate(MIGRATIONS, start=1):
                if version in applied:
                    continue
                self.connection.executescript(sql)
                self.connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'))",
                    (version,),
                )

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, parameters)

    def close(self) -> None:
        with self.lock:
            self.checkpoint()
            self.connection.close()

    def checkpoint(self) -> None:
        with self.lock:
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

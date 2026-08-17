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
    """
    CREATE TABLE extraction_runs (
      id TEXT PRIMARY KEY,
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      input_hash TEXT NOT NULL,
      provider_name TEXT NOT NULL,
      model_name TEXT NOT NULL,
      prompt_version TEXT NOT NULL,
      schema_version TEXT NOT NULL,
      started_at TEXT NOT NULL,
      completed_at TEXT,
      input_events_json TEXT NOT NULL,
      input_characters INTEGER NOT NULL,
      input_tokens INTEGER,
      output_tokens INTEGER,
      latency_ms INTEGER,
      accepted_count INTEGER NOT NULL DEFAULT 0,
      rejected_count INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL,
      error_class TEXT
    );
    CREATE INDEX extraction_runs_namespace_idx ON extraction_runs(namespace_id, started_at);
    CREATE TABLE extraction_decisions (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES extraction_runs(id),
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      candidate_fingerprint TEXT NOT NULL,
      kind TEXT NOT NULL,
      subject TEXT NOT NULL,
      statement TEXT NOT NULL,
      validation_status TEXT NOT NULL,
      rejection_reason TEXT,
      action TEXT NOT NULL,
      memory_id TEXT,
      memory_version_id TEXT,
      created_at TEXT NOT NULL,
      UNIQUE(run_id, candidate_fingerprint)
    );
    CREATE INDEX extraction_decisions_namespace_idx ON extraction_decisions(namespace_id, created_at);
    ALTER TABLE memory_versions ADD COLUMN valid_until TEXT;
    ALTER TABLE memory_versions ADD COLUMN durability TEXT NOT NULL DEFAULT 'session';
    ALTER TABLE memory_versions ADD COLUMN model_run_id TEXT;
    """,
    """
    CREATE TABLE episodes (
      id TEXT PRIMARY KEY,
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      stream_id TEXT,
      start_event_id TEXT NOT NULL REFERENCES events(id),
      end_event_id TEXT NOT NULL REFERENCES events(id),
      status TEXT NOT NULL DEFAULT 'active',
      summary TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE INDEX episodes_namespace_stream_idx ON episodes(namespace_id, stream_id, updated_at);
    CREATE TABLE episode_events (
      episode_id TEXT NOT NULL REFERENCES episodes(id),
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      event_id TEXT NOT NULL REFERENCES events(id),
      ordinal INTEGER NOT NULL,
      PRIMARY KEY (episode_id, event_id),
      UNIQUE(namespace_id, event_id)
    );
    CREATE INDEX episode_events_order_idx ON episode_events(episode_id, ordinal);
    """,
    """
    CREATE TABLE memory_embeddings (
      memory_version_id TEXT PRIMARY KEY REFERENCES memory_versions(id),
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      provider TEXT NOT NULL,
      dimensions INTEGER NOT NULL,
      vector_json TEXT NOT NULL
    );
    CREATE INDEX memory_embeddings_namespace_idx ON memory_embeddings(namespace_id);
    """,
    """
    CREATE TABLE feedback (
      id TEXT PRIMARY KEY,
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      memory_id TEXT NOT NULL REFERENCES memories(id),
      label TEXT NOT NULL CHECK(label IN ('useful', 'not_useful', 'wrong', 'stale')),
      note TEXT,
      created_at TEXT NOT NULL
    );
    CREATE INDEX feedback_namespace_created_idx ON feedback(namespace_id, created_at);
    """,
    """
    ALTER TABLE events ADD COLUMN protocol_version TEXT NOT NULL DEFAULT 'event-v1';
    ALTER TABLE events ADD COLUMN actor_id TEXT;
    ALTER TABLE events ADD COLUMN agent_id TEXT;
    ALTER TABLE events ADD COLUMN session_id TEXT;
    ALTER TABLE events ADD COLUMN source_id TEXT;
    """,
    """
    CREATE TABLE artifacts (
      id TEXT PRIMARY KEY,
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      event_id TEXT NOT NULL REFERENCES events(id),
      media_type TEXT NOT NULL,
      size_bytes INTEGER NOT NULL,
      uri TEXT,
      content_hash TEXT NOT NULL,
      metadata_json TEXT NOT NULL,
      UNIQUE(namespace_id, event_id, content_hash)
    );
    CREATE INDEX artifacts_namespace_event_idx ON artifacts(namespace_id, event_id);
    """,
    """
    CREATE TABLE context_requests (
      id TEXT PRIMARY KEY,
      namespace_id TEXT NOT NULL REFERENCES namespaces(id),
      query TEXT NOT NULL,
      token_budget INTEGER NOT NULL,
      selected_json TEXT NOT NULL,
      token_count INTEGER NOT NULL,
      abstained INTEGER NOT NULL,
      diagnostics_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE INDEX context_requests_namespace_created_idx ON context_requests(namespace_id, created_at);
    """,
    """
    ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5;
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

    def backup(self, destination: str | Path) -> None:
        target = Path(destination)
        if target.resolve() == Path(self.path).resolve():
            raise ValueError("backup destination must differ from the live database")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, sqlite3.connect(str(target)) as backup_connection:
            self.connection.backup(backup_connection)

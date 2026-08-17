from __future__ import annotations

import pytest

from termytedb.integrity import check_database

from .conftest import event


def test_memory_requires_evidence(db):
    db.repository.ensure_namespace("n1")
    with pytest.raises(Exception):
        with db.database.connection:
            db.database.execute(
                """INSERT INTO memory_versions
                (id, memory_id, namespace_id, version, statement, valid_from, recorded_at, status, reason)
                VALUES ('v', 'm', 'n1', 1, 'orphan', datetime('now'), datetime('now'), 'active', 'test')"""
            )


def test_integrity_tool_detects_version_without_evidence(db):
    db.repository.ensure_namespace("n1")
    event_id = "event-without-memory-evidence"
    with db.database.connection:
        db.database.execute(
            """INSERT INTO events
            (id, namespace_id, idempotency_hash, type, occurred_at, payload_json, content_hash, redaction_state, created_at)
            VALUES (?, 'n1', 'hash', 'test', datetime('now'), '{}', 'hash', 'redacted', datetime('now'))""",
            (event_id,),
        )
        db.database.execute(
            """INSERT INTO memories
            (id, namespace_id, kind, subject_key, status, confidence, created_at)
            VALUES ('m-without-evidence', 'n1', 'test', 'test', 'active', 1.0, datetime('now'))"""
        )
        db.database.execute(
            """INSERT INTO memory_versions
            (id, memory_id, namespace_id, source_event_id, evidence_start_offset,
             evidence_end_offset, evidence_excerpt, version, statement, valid_from,
             recorded_at, status, reason)
            VALUES ('v-without-evidence', 'm-without-evidence', 'n1', ?, 0, 5, 'claim',
                    1, 'claim', datetime('now'), datetime('now'), 'active', 'test')""",
            (event_id,),
        )
    report = check_database(db.database)
    assert report.missing_evidence == 1
    assert report.ok is False


def test_new_version_supersedes_old_and_search_excludes_it(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.process("n1")
    old = db.search("n1", "SQLite")[0]
    db.ingest(event("n1", "two", "Decision: storage uses PostgreSQL."))
    db.process("n1")
    assert db.search("n1", "SQLite") == []
    current = db.search("n1", "PostgreSQL")[0]
    assert current.memory_id == old.memory_id
    assert db.database.execute("SELECT COUNT(*) FROM memory_versions WHERE memory_id=?", (str(old.memory_id),)).fetchone()[0] == 2


def test_historical_search_can_request_superseded_truth(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.process("n1")
    db.ingest(event("n1", "two", "Decision: storage uses PostgreSQL."))
    db.process("n1")
    assert db.search("n1", "SQLite") == []
    historical = db.repository.search("n1", "SQLite", 10, historical=True)
    assert historical
    assert historical[0].status == "superseded"


def test_integrity_detects_tampered_event_payload(db):
    receipt = db.ingest(event("n1", "tamper", "Decision: storage uses SQLite."))
    with db.database.connection:
        db.database.execute("UPDATE events SET payload_json=? WHERE id=?", ('{"text":"tampered"}', str(receipt.event_id)))
    report = check_database(db.database)
    assert report.event_hash_mismatches == 1
    assert report.ok is False

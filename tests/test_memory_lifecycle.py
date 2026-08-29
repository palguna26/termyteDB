from __future__ import annotations

import pytest

from src.storage.integrity import check_database

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
    # Phase 3: evidence is now optional — a version without an evidence_refs row is valid
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
    assert report.missing_evidence == 0
    # still not ok due to event hash mismatch, but not due to missing evidence
    assert report.event_hash_mismatches == 1
    assert report.ok is False


def test_new_version_supersedes_old_and_search_excludes_it(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.process("n1")
    old = db.search("n1", "SQLite")[0]
    db.ingest(event("n1", "two", "Decision: storage uses PostgreSQL."))
    db.process("n1")
    assert all(result.memory_version_id != old.memory_version_id for result in db.search("n1", "SQLite"))
    current = db.search("n1", "PostgreSQL")[0]
    assert current.memory_id == old.memory_id
    assert db.database.execute("SELECT COUNT(*) FROM memory_versions WHERE memory_id=?", (str(old.memory_id),)).fetchone()[0] == 2


def test_historical_search_can_request_superseded_truth(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.process("n1")
    db.ingest(event("n1", "two", "Decision: storage uses PostgreSQL."))
    db.process("n1")
    assert all(result.status == "active" for result in db.search("n1", "SQLite"))
    historical = db.repository.search("n1", "SQLite", 10, historical=True)
    assert historical
    assert any(result.status == "superseded" and "SQLite" in result.statement for result in historical)


def test_first_name_query_does_not_force_historical_search(db):
    db.ingest(event("n1", "one", "My first name is Alice."))
    db.process("n1")
    db.ingest(event("n1", "two", "My first name is Alicia."))
    db.process("n1")
    results = db.search("n1", "first name")
    assert results
    assert results[0].status == "active"
    assert "Alicia" in results[0].statement


def test_forget_tombstones_memory_and_restore_reindexes_it(db):
    db.ingest(event("n1", "forget", "Decision: storage uses SQLite."))
    db.process("n1")
    result = db.search("n1", "SQLite")[0]
    memory_id = str(result.memory_id)
    assert db.forget("n1", memory_id, "user requested forgetting") is True
    assert db.search("n1", "SQLite") == []
    assert db.repository.history("n1", memory_id)[0]["status"] == "deleted"
    assert db.restore("n1", memory_id) is True
    assert db.search("n1", "SQLite")


def test_integrity_detects_tampered_event_payload(db):
    receipt = db.ingest(event("n1", "tamper", "Decision: storage uses SQLite."))
    with db.database.connection:
        db.database.execute("UPDATE events SET payload_json=? WHERE id=?", ('{"text":"tampered"}', str(receipt.event_id)))
    report = check_database(db.database)
    assert report.event_hash_mismatches == 1
    assert report.ok is False

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .conftest import event


def test_duplicate_event_ingestion_is_idempotent(db):
    first = db.ingest(event("n1", "same", "Decision: Use SQLite."))
    second = db.ingest(event("n1", "same", "Decision: Use SQLite."))
    assert first.event_id == second.event_id
    assert first.job_id == second.job_id
    assert second.duplicate is True
    assert db.database.execute("SELECT COUNT(*) FROM events WHERE namespace_id='n1'").fetchone()[0] == 1


def test_different_identical_events_remain_distinct_and_both_are_evidence(db):
    first = db.ingest(event("n1", "one", "Decision: Use SQLite."))
    second = db.ingest(event("n1", "two", "Decision: Use SQLite."))
    assert first.event_id != second.event_id
    db.process("n1")
    assert db.database.execute("SELECT COUNT(*) FROM events WHERE namespace_id='n1'").fetchone()[0] == 2
    memory = db.search("n1", "SQLite")[0]
    assert {str(c.event_id) for c in memory.citations} == {
        str(first.event_id),
        str(second.event_id),
    }


def test_redacted_values_never_reach_persistent_storage(db):
    db.ingest(event("n1", "secret", "Decision: Use API_KEY=SUPERSECRET123456789."))
    row = db.database.execute("SELECT payload_json FROM events WHERE namespace_id='n1'").fetchone()
    assert "SUPERSECRET123456789" not in row[0]
    assert "[REDACTED]" in row[0]


def test_event_payload_is_redacted_before_extraction(db):
    db.ingest(event("n1", "secret", "Decision: Use API_KEY=SUPERSECRET123456789."))
    db.process("n1")
    result = db.search("n1", "Decision")
    assert result
    assert "SUPERSECRET123456789" not in result[0].statement


def test_concurrent_namespaces_keep_event_counts_and_isolation(db):
    namespaces = [f"parallel-{index}" for index in range(4)]

    def ingest_namespace(namespace_id: str) -> None:
        for index in range(5):
            db.ingest(event(namespace_id, str(index), f"Decision: {namespace_id} item {index}."))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(ingest_namespace, namespaces))
    for namespace_id in namespaces:
        assert db.database.execute("SELECT COUNT(*) FROM events WHERE namespace_id=?", (namespace_id,)).fetchone()[0] == 5
        event_id = db.database.execute("SELECT id FROM events WHERE namespace_id=? LIMIT 1", (namespace_id,)).fetchone()[0]
        assert db.event(namespace_id, event_id)["namespace_id"] == namespace_id
        assert db.event("other-namespace", event_id) is None

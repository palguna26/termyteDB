from __future__ import annotations

from .conftest import event


def test_fts_results_are_namespace_scoped(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.ingest(event("n2", "two", "Decision: storage uses SQLite."))
    db.process("n1")
    db.process("n2")
    memory = db.search("n1", "SQLite")[0]
    assert db.search("n2", "SQLite")[0].memory_id != memory.memory_id


def test_direct_ingestion_does_not_create_processing_jobs(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.ingest(event("n2", "two", "Decision: storage uses PostgreSQL."))
    assert db.repository.claim_jobs("n1", 10, 30) == []
    assert db.database.execute("SELECT COUNT(*) FROM processing_jobs").fetchone()[0] == 0


def test_history_remains_inspectable_after_supersession(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.process("n1")
    first = db.search("n1", "SQLite")[0]
    db.ingest(event("n1", "two", "Decision: storage uses PostgreSQL."))
    db.process("n1")
    history = db.repository.list_versions("n1", str(first.memory_id))
    assert [row["version"] for row in history] == [1, 2]
    assert history[0]["status"] == "superseded"
    assert history[1]["status"] == "active"

from __future__ import annotations

from .conftest import event


def test_failed_jobs_retry_without_duplicate_memories(db, monkeypatch):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))
    import termytedb.processor as processor_module

    original = processor_module.extract
    calls = {"count": 0}

    def fail_once(payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary")
        return original(payload)

    monkeypatch.setattr(processor_module, "extract", fail_once)
    assert db.process("n1").failed == 1
    assert db.process("n1").processed == 0
    with db.database.connection:
        db.database.execute("UPDATE processing_jobs SET next_attempt_at='2000-01-01T00:00:00+00:00' WHERE namespace_id='n1'")
    assert db.process("n1").processed == 1
    assert db.repository.memory_count("n1") == 1


def test_expired_leases_are_reclaimed(db):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))
    job = db.repository.claim_jobs("n1", 1, 30)[0]
    with db.database.connection:
        db.database.execute(
            "UPDATE processing_jobs SET lease_until=? WHERE id=?",
            ("2000-01-01T00:00:00+00:00", job["id"]),
        )
    reclaimed = db.repository.claim_jobs("n1", 1, 30)
    assert reclaimed
    assert reclaimed[0]["id"] == job["id"]


def test_active_job_heartbeat_extends_lease_and_respects_namespace(db):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))
    job = db.repository.claim_jobs("n1", 1, 1)[0]
    before = db.database.execute("SELECT lease_until FROM processing_jobs WHERE id=?", (job["id"],)).fetchone()[0]
    assert db.repository.heartbeat_job("n1", job["id"], 60) is True
    after = db.database.execute("SELECT lease_until FROM processing_jobs WHERE id=?", (job["id"],)).fetchone()[0]
    assert after != before
    assert db.repository.heartbeat_job("other", job["id"], 60) is False


def test_permanently_failed_jobs_enter_dead_letter(db, monkeypatch):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))
    import termytedb.processor as processor_module

    monkeypatch.setattr(
        processor_module,
        "extract",
        lambda payload: (_ for _ in ()).throw(RuntimeError("permanent")),
    )
    assert db.process("n1").dead_lettered == 0
    with db.database.connection:
        db.database.execute("UPDATE processing_jobs SET next_attempt_at='2000-01-01T00:00:00+00:00' WHERE namespace_id='n1'")
    assert db.process("n1").dead_lettered == 0
    with db.database.connection:
        db.database.execute("UPDATE processing_jobs SET next_attempt_at='2000-01-01T00:00:00+00:00' WHERE namespace_id='n1'")
    assert db.process("n1").dead_lettered == 1
    status = db.database.execute("SELECT status FROM processing_jobs WHERE namespace_id='n1'").fetchone()[0]
    assert status == "dead"

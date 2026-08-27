from __future__ import annotations

from termytedb.memory.provider import FakeExtractionProvider, ProviderError

from .conftest import event


def test_failed_jobs_retry_without_duplicate_memories(db):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))
    calls = {"count": 0}

    class FlakyProvider:
        name = "flaky"
        model = "flaky-v1"

        def __init__(self, delegate):
            self.delegate = delegate

        def extract(self, request, timeout_seconds=30.0, cancellation=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary")
            return self.delegate.extract(request, timeout_seconds=timeout_seconds, cancellation=cancellation)

    db.processor.provider = FlakyProvider(FakeExtractionProvider())
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


def test_stale_lease_owner_cannot_complete_or_fail_reclaimed_job(db):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))
    first = db.repository.claim_jobs("n1", 1, 30)[0]
    first_token = str(first["lease_token"])
    with db.database.connection:
        db.database.execute("UPDATE processing_jobs SET lease_until='2000-01-01 00:00:00' WHERE id=?", (first["id"],))
    second = db.repository.claim_jobs("n1", 1, 30)[0]
    assert second["lease_token"] != first_token
    assert db.repository.complete_job("n1", first["id"], first_token) is False
    assert db.repository.fail_job("n1", first["id"], "late failure", lease_token=first_token) == "stale"
    row = db.database.execute("SELECT status, lease_token FROM processing_jobs WHERE id=?", (first["id"],)).fetchone()
    assert tuple(row) == ("processing", second["lease_token"])


def test_permanently_failed_jobs_enter_dead_letter(db):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))

    class PermanentFailureProvider:
        name = "permanent"
        model = "permanent-v1"

        def extract(self, request, timeout_seconds=30.0, cancellation=None):
            raise ProviderError("permanent", retryable=True, error_class="permanent")

    db.processor.provider = PermanentFailureProvider()
    assert db.process("n1").dead_lettered == 0
    with db.database.connection:
        db.database.execute("UPDATE processing_jobs SET next_attempt_at='2000-01-01T00:00:00+00:00' WHERE namespace_id='n1'")
    assert db.process("n1").dead_lettered == 0
    with db.database.connection:
        db.database.execute("UPDATE processing_jobs SET next_attempt_at='2000-01-01T00:00:00+00:00' WHERE namespace_id='n1'")
    assert db.process("n1").dead_lettered == 1
    status = db.database.execute("SELECT status FROM processing_jobs WHERE namespace_id='n1'").fetchone()[0]
    assert status == "dead"

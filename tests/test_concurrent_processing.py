from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from termytedb import TermyteDB


def test_concurrent_workers_claim_each_job_once(tmp_path):
    db = TermyteDB(tmp_path / "workers.sqlite")
    for index in range(20):
        db.ingest(
            {"namespace_id": "workers", "idempotency_key": str(index), "type": "note", "payload": {"text": f"Decision: item {index}."}}
        )

    barrier = Barrier(2)

    def claim() -> list[str]:
        barrier.wait()
        return [job["id"] for job in db.repository.claim_jobs("workers", 10, 30)]

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = list(executor.map(lambda _: claim(), range(2)))
    assert len(claimed[0]) == 10
    assert len(claimed[1]) == 10
    assert set(claimed[0]).isdisjoint(claimed[1])
    with db.database.connection:
        db.database.execute("UPDATE processing_jobs SET status='pending', lease_until=NULL WHERE namespace_id='workers'")
    assert db.process("workers", limit=20).processed == 20
    assert db.database.execute("SELECT COUNT(*) FROM processing_jobs WHERE namespace_id='workers' AND status='completed'").fetchone()[0] == 20
    assert db.database.execute("SELECT COUNT(*) FROM memories WHERE namespace_id='workers'").fetchone()[0] == 20
    db.close()


def test_separate_database_connections_do_not_claim_the_same_job(tmp_path):
    path = tmp_path / "separate-workers.sqlite"
    owner = TermyteDB(path)
    for index in range(4):
        owner.ingest(
            {"namespace_id": "workers", "idempotency_key": str(index), "type": "note", "payload": {"text": f"Item {index}."}}
        )

    workers = [TermyteDB(path) for _ in range(4)]
    barrier = Barrier(4)

    def claim(worker: TermyteDB) -> str:
        barrier.wait()
        return str(worker.repository.claim_jobs("workers", 1, 30)[0]["id"])

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            claimed = list(executor.map(claim, workers))
        assert len(set(claimed)) == 4
        attempts = owner.database.execute(
            "SELECT attempts FROM processing_jobs WHERE namespace_id='workers' ORDER BY id"
        ).fetchall()
        assert [row[0] for row in attempts] == [1, 1, 1, 1]
    finally:
        for worker in workers:
            worker.close()
        owner.close()


def test_separate_workers_reinforce_one_memory_without_version_conflicts(tmp_path):
    path = tmp_path / "same-memory.sqlite"
    owner = TermyteDB(path)
    for index in range(2):
        owner.ingest(
            {
                "namespace_id": "workers",
                "idempotency_key": str(index),
                "type": "decision",
                "payload": {"text": "Decision: use SQLite."},
            }
        )
    workers = [TermyteDB(path) for _ in range(2)]
    barrier = Barrier(2)

    def process(worker: TermyteDB):
        barrier.wait()
        return worker.process("workers", limit=1)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(process, workers))
        assert sum(response.processed for response in responses) == 2
        assert sum(response.failed for response in responses) == 0
        assert owner.database.execute("SELECT COUNT(*) FROM memories WHERE namespace_id='workers'").fetchone()[0] == 1
        assert owner.database.execute("SELECT COUNT(*) FROM memory_versions WHERE namespace_id='workers'").fetchone()[0] == 1
        assert owner.database.execute("SELECT COUNT(*) FROM evidence_refs WHERE namespace_id='workers'").fetchone()[0] == 2
    finally:
        for worker in workers:
            worker.close()
        owner.close()

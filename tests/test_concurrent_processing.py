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

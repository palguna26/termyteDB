from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from src import TermyteDB


def test_concurrent_direct_ingestion_on_one_engine_is_safe(tmp_path):
    db = TermyteDB(tmp_path / "direct.sqlite")
    namespaces = [f"worker-{index}" for index in range(4)]

    def ingest(namespace_id: str) -> None:
        db.ingest_batch(
            [
                {
                    "namespace_id": namespace_id,
                    "idempotency_key": str(index),
                    "type": "decision",
                    "payload": {"text": f"Decision: {namespace_id} item {index}."},
                }
                for index in range(5)
            ]
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(ingest, namespaces))

    for namespace_id in namespaces:
        assert len(db.memories(namespace_id)) == 5
    assert db.database.execute("SELECT COUNT(*) FROM processing_jobs").fetchone()[0] == 0
    db.close()


def test_separate_connections_reinforce_memory_without_version_conflicts(tmp_path):
    path = tmp_path / "same-memory.sqlite"
    owner = TermyteDB(path)
    workers = [TermyteDB(path) for _ in range(2)]
    barrier = Barrier(2)

    def ingest(index: int) -> None:
        barrier.wait()
        workers[index].ingest(
            {
                "namespace_id": "workers",
                "idempotency_key": str(index),
                "type": "decision",
                "payload": {"text": "Decision: use SQLite."},
            }
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(ingest, range(2)))
        assert owner.database.execute("SELECT COUNT(*) FROM memories WHERE namespace_id='workers'").fetchone()[0] == 1
        assert owner.database.execute("SELECT COUNT(*) FROM memory_versions WHERE namespace_id='workers'").fetchone()[0] == 1
        assert owner.database.execute("SELECT COUNT(*) FROM evidence_refs WHERE namespace_id='workers'").fetchone()[0] == 2
        assert owner.database.execute("SELECT COUNT(*) FROM processing_jobs").fetchone()[0] == 0
    finally:
        for worker in workers:
            worker.close()
        owner.close()

from __future__ import annotations

from fastapi.testclient import TestClient
from termytedb.service import create_app

from .conftest import event


def test_direct_id_access_and_fts_collision_are_namespace_scoped(db, tmp_path):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.ingest(event("n2", "two", "Decision: storage uses SQLite."))
    db.process("n1")
    db.process("n2")
    memory = db.search("n1", "SQLite")[0]
    assert db.search("n2", "SQLite")[0].memory_id != memory.memory_id
    client = TestClient(create_app(tmp_path / "api.sqlite"))
    client.post("/v1/events", json=event("n1", "one", "Decision: storage uses SQLite."))
    client.post("/v1/process", json={"namespace_id": "n1"})
    api_memory = client.post("/v1/search", json={"namespace_id": "n1", "query": "SQLite"}).json()[0]["memory_id"]
    assert client.get(f"/v1/memories/{api_memory}", params={"namespace_id": "n2"}).status_code == 404


def test_job_claims_do_not_cross_namespaces(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.ingest(event("n2", "two", "Decision: storage uses PostgreSQL."))
    claimed = db.repository.claim_jobs("n1", 10, 30)
    assert claimed
    assert all(row["namespace_id"] == "n1" for row in claimed)


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

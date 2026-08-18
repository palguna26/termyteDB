import sqlite3

from fastapi.testclient import TestClient
from termytedb.service import create_app


def test_readiness_reports_healthy_database(tmp_path):
    client = TestClient(create_app(str(tmp_path / "ready.sqlite")))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_fails_after_event_tampering(tmp_path):
    client = TestClient(create_app(str(tmp_path / "tampered.sqlite")))
    receipt = client.post(
        "/v1/events",
        json={"namespace_id": "ready", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: use SQLite."}},
    ).json()
    with sqlite3.connect(tmp_path / "tampered.sqlite") as database:
        database.execute("UPDATE events SET payload_json=? WHERE id=?", ('{"text":"tampered"}', receipt["event_id"]))
    response = client.get("/ready")
    assert response.status_code == 503

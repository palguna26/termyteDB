from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_evidence_jobs_and_integrity_are_inspectable(tmp_path):
    client = TestClient(create_app(str(tmp_path / "diagnostics.sqlite")))
    response = client.post(
        "/v1/events",
        json={"namespace_id": "diagnostics", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}},
    )
    event_id = response.json()["event_id"]
    client.post("/v1/process", json={"namespace_id": "diagnostics"})
    evidence = client.get("/v1/evidence", params={"namespace_id": "diagnostics"})
    assert evidence.status_code == 200
    assert evidence.json()[0]["namespace_id"] == "diagnostics"
    assert client.get("/v1/evidence", params={"namespace_id": "other"}).json() == []
    assert client.get(f"/v1/events/{event_id}", params={"namespace_id": "diagnostics"}).json()["payload_json"]["text"] == "Decision: use SQLite."
    assert client.get("/v1/jobs", params={"namespace_id": "diagnostics"}).json()[0]["status"] == "completed"
    report = client.get("/v1/integrity")
    assert report.status_code == 200
    assert report.json()["ok"] is True
    assert client.get(f"/v1/events/{event_id}", params={"namespace_id": "other"}).status_code == 404

from __future__ import annotations

from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_http_vertical_slice(tmp_path):
    client = TestClient(create_app(str(tmp_path / "api.sqlite")))
    event = {
        "namespace_id": "api-ns",
        "idempotency_key": "http-1",
        "type": "decision",
        "payload": {"text": "Decision: Use SQLite for local persistence."},
    }
    receipt = client.post("/v1/events", json=event).json()
    assert client.post("/v1/process", json={"namespace_id": "api-ns"}).json()["processed"] == 1
    response = client.post("/v1/context", json={"namespace_id": "api-ns", "query": "SQLite", "token_budget": 100})
    assert response.status_code == 200
    assert response.json()["abstained"] is False
    assert str(receipt["event_id"]) in response.json()["text"]

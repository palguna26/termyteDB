from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_extraction_runs_and_decisions_are_paginated_and_namespace_scoped(tmp_path):
    client = TestClient(create_app(str(tmp_path / "inspection.sqlite")))
    response = client.post(
        "/v1/events",
        json={
            "namespace_id": "inspect-a",
            "idempotency_key": "one",
            "type": "decision",
            "payload": {"text": "Decision: use SQLite."},
        },
    )
    assert response.status_code == 200
    assert client.post("/v1/process", json={"namespace_id": "inspect-a", "limit": 10}).status_code == 200

    runs = client.get("/v1/extraction/runs", params={"namespace_id": "inspect-a", "limit": 1})
    decisions = client.get("/v1/extraction/decisions", params={"namespace_id": "inspect-a", "limit": 1})
    assert runs.status_code == decisions.status_code == 200
    assert len(runs.json()) == len(decisions.json()) == 1
    assert runs.json()[0]["namespace_id"] == decisions.json()[0]["namespace_id"] == "inspect-a"
    assert decisions.json()[0]["action"] == "INSERT"
    assert client.get("/v1/extraction/decisions", params={"namespace_id": "inspect-b"}).json() == []
    assert client.get("/v1/extraction/decisions", params={"namespace_id": "inspect-a", "offset": 1}).json() == []


def test_extraction_inspection_rejects_invalid_pagination(tmp_path):
    client = TestClient(create_app(str(tmp_path / "inspection-validation.sqlite")))
    assert client.get("/v1/extraction/runs", params={"namespace_id": "x", "limit": 0}).status_code == 422

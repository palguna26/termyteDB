from __future__ import annotations

from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_openapi_contract_contains_versioned_operations(tmp_path):
    client = TestClient(create_app(str(tmp_path / "openapi.sqlite")))
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    required = {
        ("/v1/events", "post"), ("/v1/events:batch", "post"), ("/v1/process", "post"),
        ("/v1/search", "post"), ("/v1/context", "post"), ("/v1/export", "get"),
        ("/v1/import", "post"), ("/v1/integrity", "get"), ("/ready", "get"),
    }
    assert required.issubset({(path, method) for path, operations in paths.items() for method in operations})


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


def test_http_batch_history_invalidation_export_and_delete(tmp_path):
    client = TestClient(create_app(str(tmp_path / "lifecycle.sqlite")))
    batch = client.post("/v1/events:batch", json={"events": [
        {"namespace_id": "lifecycle", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}},
        {"namespace_id": "lifecycle", "idempotency_key": "two", "type": "failure", "payload": {"text": "Failure: old cache broke."}},
    ]})
    assert batch.status_code == 200
    assert len(batch.json()["receipts"]) == 2
    assert client.post("/v1/process", json={"namespace_id": "lifecycle"}).json()["accepted"] == 2
    result = client.post("/v1/search", json={"namespace_id": "lifecycle", "query": "SQLite"}).json()[0]
    memory_id = result["memory_id"]
    assert client.get(f"/v1/memories/{memory_id}/history", params={"namespace_id": "lifecycle"}).status_code == 200
    assert client.post(f"/v1/memories/{memory_id}/invalidate", params={"namespace_id": "lifecycle", "reason": "test"}).json() == {"invalidated": True}
    assert client.get("/v1/export", params={"namespace_id": "lifecycle"}).json()["events"]
    assert client.delete("/v1/namespaces/lifecycle").json() == {"deleted": True}
    assert client.get(f"/v1/memories/{memory_id}", params={"namespace_id": "lifecycle"}).status_code == 404


def test_http_historical_search_is_explicit(tmp_path):
    client = TestClient(create_app(str(tmp_path / "historical.sqlite")))
    for key, text in (("one", "Decision: storage uses SQLite."), ("two", "Decision: storage uses PostgreSQL.")):
        client.post("/v1/events", json={"namespace_id": "history", "idempotency_key": key, "type": "decision", "payload": {"text": text}})
    client.post("/v1/process", json={"namespace_id": "history"})
    assert client.post("/v1/search", json={"namespace_id": "history", "query": "SQLite"}).json() == []
    response = client.post("/v1/search", json={"namespace_id": "history", "query": "SQLite", "historical": True})
    assert response.status_code == 200
    assert response.json()[0]["status"] == "superseded"

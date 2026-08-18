from fastapi.testclient import TestClient
from termytedb.service import create_app


def test_memory_collection_is_paginated_and_namespace_scoped(tmp_path):
    client = TestClient(create_app(str(tmp_path / "memories.sqlite")))
    for index in range(3):
        response = client.post(
            "/v1/events",
            json={"namespace_id": "memories-a", "idempotency_key": str(index), "type": "note", "payload": {"text": f"Decision: item {index}."}},
        )
        assert response.status_code == 200
    client.post(
        "/v1/events",
        json={"namespace_id": "memories-b", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: other."}},
    )
    assert client.post("/v1/process", json={"namespace_id": "memories-a"}).status_code == 200
    assert client.post("/v1/process", json={"namespace_id": "memories-b"}).status_code == 200
    first = client.get("/v1/memories", params={"namespace_id": "memories-a", "limit": 2})
    second = client.get("/v1/memories", params={"namespace_id": "memories-a", "limit": 2, "offset": 2})
    assert first.status_code == second.status_code == 200
    assert len(first.json()) == 2
    assert len(second.json()) == 1
    assert client.get("/v1/memories", params={"namespace_id": "memories-b"}).json()[0]["namespace_id"] == "memories-b"
    assert client.get("/v1/memories", params={"namespace_id": "memories-a", "limit": 0}).status_code == 422

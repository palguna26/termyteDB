from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_event_collection_is_paginated_and_preserves_evidence_projection(tmp_path):
    client = TestClient(create_app(str(tmp_path / "events.sqlite")))
    for index in range(3):
        assert client.post(
            "/v1/events",
            json={"namespace_id": "events-a", "idempotency_key": str(index), "type": "note", "payload": {"text": f"Decision: item {index}."}},
        ).status_code == 200
    first = client.get("/v1/events", params={"namespace_id": "events-a", "limit": 2})
    second = client.get("/v1/events", params={"namespace_id": "events-a", "limit": 2, "offset": 2})
    assert len(first.json()) == 2
    assert len(second.json()) == 1
    assert first.json()[0]["payload_json"]["text"].startswith("Decision:")
    assert "evidence_refs" in first.json()[0]
    assert client.get("/v1/events", params={"namespace_id": "other"}).json() == []

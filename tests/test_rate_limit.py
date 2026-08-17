from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_http_rate_limit_is_per_namespace(tmp_path):
    client = TestClient(create_app(str(tmp_path / "rate.sqlite"), rate_limit_per_minute=1))
    event = {"namespace_id": "limited", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: one."}}
    assert client.post("/v1/events", json=event).status_code == 200
    assert client.post("/v1/events", json={**event, "idempotency_key": "two"}).status_code == 429
    other = {**event, "namespace_id": "other"}
    assert client.post("/v1/events", json=other).status_code == 200

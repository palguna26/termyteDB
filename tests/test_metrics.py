from fastapi.testclient import TestClient
from termytedb.service import create_app


def test_metrics_are_namespace_scoped_and_track_processing(tmp_path):
    client = TestClient(create_app(str(tmp_path / "metrics.sqlite")))
    client.post(
        "/v1/events",
        json={"namespace_id": "metrics-a", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: use SQLite."}},
    )
    before = client.get("/v1/metrics", params={"namespace_id": "metrics-a"}).json()
    assert before["events"] == 1
    assert before["jobs_pending"] == 1
    assert client.get("/v1/metrics", params={"namespace_id": "metrics-b"}).json()["events"] == 0
    client.post("/v1/process", json={"namespace_id": "metrics-a"})
    after = client.get("/v1/metrics", params={"namespace_id": "metrics-a"}).json()
    assert after["memories"] == 1
    assert after["jobs_completed"] == 1

from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_job_cancellation_is_namespace_scoped(tmp_path):
    client = TestClient(create_app(str(tmp_path / "jobs.sqlite")))
    receipt = client.post(
        "/v1/events",
        json={"namespace_id": "jobs-a", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: use SQLite."}},
    ).json()
    job_id = receipt["job_id"]
    assert client.post(f"/v1/jobs/{job_id}/cancel", params={"namespace_id": "jobs-b"}).status_code == 404
    assert client.post(f"/v1/jobs/{job_id}/cancel", params={"namespace_id": "jobs-a"}).json() == {"cancelled": True}
    assert client.get("/v1/jobs", params={"namespace_id": "jobs-a"}).json()[0]["status"] == "cancelled"
    assert client.post("/v1/process", json={"namespace_id": "jobs-a", "timeout_seconds": 0}).status_code == 422


def test_processing_timeout_leaves_unclaimed_jobs_pending(tmp_path):
    client = TestClient(create_app(str(tmp_path / "timeout.sqlite")))
    for index in range(2):
        assert client.post(
            "/v1/events",
            json={"namespace_id": "timeout", "idempotency_key": str(index), "type": "note", "payload": {"text": "Decision: use SQLite."}},
        ).status_code == 200
    response = client.post("/v1/process", json={"namespace_id": "timeout", "limit": 2, "timeout_seconds": 0.001})
    assert response.status_code == 200
    assert response.json()["processed"] <= 2

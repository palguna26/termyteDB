from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_feedback_is_namespace_scoped_and_redacted(tmp_path):
    client = TestClient(create_app(str(tmp_path / "feedback.sqlite")))
    receipt = client.post(
        "/v1/events",
        json={"namespace_id": "feedback", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}},
    ).json()
    assert receipt["event_id"]
    assert client.get("/health", headers={"x-request-id": "request-1"}).headers["x-request-id"] == "request-1"
    client.post("/v1/process", json={"namespace_id": "feedback"})
    memory_id = client.post("/v1/search", json={"namespace_id": "feedback", "query": "SQLite"}).json()[0]["memory_id"]
    response = client.post(
        "/v1/feedback",
        json={"namespace_id": "feedback", "memory_id": memory_id, "label": "useful", "note": "token=do-not-store"},
    )
    assert response.status_code == 200
    rows = client.app.state.engine.repository.list_feedback("feedback")
    assert rows[0]["note"] == "token=[REDACTED]"
    assert client.post(
        "/v1/feedback",
        json={"namespace_id": "other", "memory_id": memory_id, "label": "useful"},
    ).status_code == 404

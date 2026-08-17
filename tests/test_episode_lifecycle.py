from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_episode_lifecycle_statuses_are_constrained_and_scoped(tmp_path):
    client = TestClient(create_app(str(tmp_path / "episode-status.sqlite")))
    client.post(
        "/v1/events",
        json={"namespace_id": "episodes", "idempotency_key": "one", "type": "decision", "stream_id": "s", "payload": {"text": "Decision: use SQLite."}},
    )
    episode = client.get("/v1/episodes", params={"namespace_id": "episodes"}).json()[0]
    update = {"namespace_id": "episodes", "status": "completed", "summary": "finished"}
    assert client.patch(f"/v1/episodes/{episode['id']}", json=update).status_code == 200
    assert client.get("/v1/episodes", params={"namespace_id": "episodes"}).json()[0]["status"] == "completed"
    assert client.patch(f"/v1/episodes/{episode['id']}", json={**update, "status": "unknown"}).status_code == 422
    assert client.patch(f"/v1/episodes/{episode['id']}", json={**update, "namespace_id": "other"}).status_code == 404

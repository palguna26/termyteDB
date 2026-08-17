from fastapi.testclient import TestClient

from termytedb.service import create_app


def test_collection_endpoints_support_bounded_pagination(tmp_path):
    client = TestClient(create_app(str(tmp_path / "pages.sqlite")))
    for index in range(3):
        client.post(
            "/v1/events",
            json={"namespace_id": "pages", "idempotency_key": str(index), "type": "decision", "payload": {"text": f"Decision: item {index}."}},
        )
    jobs = client.get("/v1/jobs", params={"namespace_id": "pages", "limit": 1, "offset": 1}).json()
    assert len(jobs) == 1
    assert client.get("/v1/jobs", params={"namespace_id": "other", "limit": 1}).json() == []
    assert client.get("/v1/jobs", params={"namespace_id": "pages", "limit": 101}).status_code == 422

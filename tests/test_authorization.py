from fastapi import Request
from fastapi.testclient import TestClient
from termytedb.service import create_app


def test_http_namespace_authorizer_blocks_reads_and_writes(tmp_path):
    client = TestClient(create_app(str(tmp_path / "auth.sqlite"), namespace_authorizer=lambda namespace: namespace == "allowed"))
    denied_event = {"namespace_id": "denied", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: secret."}}
    assert client.post("/v1/events", json=denied_event).status_code == 403
    assert client.post("/v1/search", json={"namespace_id": "denied", "query": "secret"}).status_code == 403
    assert client.get("/v1/events/not-real", params={"namespace_id": "denied"}).status_code == 403
    allowed_event = {**denied_event, "namespace_id": "allowed"}
    assert client.post("/v1/events", json=allowed_event).status_code == 200


def test_http_request_authorizer_is_an_optional_authentication_boundary(tmp_path):
    def authorize(request: Request) -> bool:
        return request.headers.get("authorization") == "Bearer test-token"

    client = TestClient(create_app(str(tmp_path / "auth-boundary.sqlite"), request_authorizer=authorize))
    denied = client.get("/health")
    assert denied.status_code == 401
    assert denied.headers["x-request-id"]
    allowed = client.get("/health", headers={"authorization": "Bearer test-token"})
    assert allowed.status_code == 200

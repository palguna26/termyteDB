from fastapi.testclient import TestClient
from termytedb.service import create_app

from termytedb import TermyteDB


def test_context_request_is_persisted_and_survives_restart(tmp_path):
    path = tmp_path / "context-audit.sqlite"
    first = TermyteDB(path)
    first.ingest({"namespace_id": "audit", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}})
    first.process("audit")
    response = first.context("audit", "SQLite", token_budget=100)
    assert response.request_id is not None
    first.close()
    second = TermyteDB(path)
    rows = second.context_requests("audit")
    assert rows[0]["id"] == str(response.request_id)
    assert rows[0]["selected_json"]
    second.delete_namespace("audit")
    assert second.context_requests("audit") == []
    second.close()


def test_context_request_api_requires_namespace_authorization(tmp_path):
    client = TestClient(create_app(str(tmp_path / "context-api.sqlite"), namespace_authorizer=lambda value: value == "allowed"))
    assert client.get("/v1/context/requests", params={"namespace_id": "denied"}).status_code == 403

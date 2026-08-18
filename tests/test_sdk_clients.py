import io
import json
from email.message import Message

import pytest

from clients.python.termytedb_client import TermyteDBClient, TermyteDBError


class Response:
    def __init__(self, payload: object):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_python_client_sends_request_id_and_retries(monkeypatch):
    calls = []

    def open_request(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            headers = Message()
            headers["x-request-id"] = "server-request"
            from urllib.error import HTTPError
            raise HTTPError(request.full_url, 503, "busy", headers, io.BytesIO(b'{"detail":"busy"}'))
        return Response({"status": "ok"})

    monkeypatch.setattr("termytedb.client.urlopen", open_request)
    assert TermyteDBClient("http://localhost", retries=1, auth_token="test-token").health() == {"status": "ok"}
    assert len(calls) == 2
    assert calls[0][0].get_header("X-request-id")
    assert calls[0][0].get_header("Authorization") == "Bearer test-token"


def test_python_client_exposes_structured_errors(monkeypatch):
    def open_request(request, timeout):
        from urllib.error import HTTPError
        headers = Message()
        headers["x-request-id"] = "request-1"
        raise HTTPError(request.full_url, 422, "bad", headers, io.BytesIO(b'{"detail":"invalid"}'))

    monkeypatch.setattr("termytedb.client.urlopen", open_request)
    with pytest.raises(TermyteDBError) as error:
        TermyteDBClient("http://localhost", retries=0).health()
    assert error.value.status == 422
    assert error.value.request_id == "request-1"

import json
from uuid import uuid4

import pytest
from termytedb.provider import HttpExtractionProvider, ProviderError
from termytedb.schemas import ExtractionRequest


def request() -> ExtractionRequest:
    event_id = uuid4()
    return ExtractionRequest(namespace_id="provider", events=[event_id], evidence_text={event_id: "Decision: use SQLite."})


def test_http_provider_validates_strict_output(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"schema_version": "extraction-v1", "prompt_version": "p1", "candidates": []}).encode()

    monkeypatch.setattr("termytedb.provider.urlopen", lambda *_args, **_kwargs: Response())
    result = HttpExtractionProvider("http://provider", "model").extract(request())
    assert result.provider_name == "http"
    assert result.response.schema_version == "extraction-v1"


def test_http_provider_rejects_malformed_output_without_leaking_response(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"not-json secret=should-not-appear"

    monkeypatch.setattr("termytedb.provider.urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(ProviderError, match="invalid extraction-v1 JSON") as error:
        HttpExtractionProvider("http://provider").extract(request())
    assert "should-not-appear" not in str(error.value)

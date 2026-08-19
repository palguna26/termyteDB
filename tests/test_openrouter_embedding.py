import json

import pytest

from termytedb.embedding import OpenAICompatibleEmbeddingProvider


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}).encode()


def test_openrouter_provider_batches_and_orders_embeddings(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("termytedb.embedding.urlopen", fake_urlopen)
    provider = OpenAICompatibleEmbeddingProvider("test/model", api_key="secret", dimensions=2, retries=1)
    assert provider.embed_many(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert captured["body"] == {"model": "test/model", "input": ["a", "b"], "encoding_format": "float"}
    assert captured["authorization"] == "Bearer secret"


def test_openrouter_provider_rejects_wrong_dimensions(monkeypatch):
    class WrongResponse(Response):
        def read(self):
            return b'{"data":[{"index":0,"embedding":[1.0]}]}'

    monkeypatch.setattr("termytedb.embedding.urlopen", lambda *_args, **_kwargs: WrongResponse())
    provider = OpenAICompatibleEmbeddingProvider("test/model", api_key="secret", dimensions=2, retries=1)
    with pytest.raises(ValueError, match="dimension"):
        provider.embed("a")

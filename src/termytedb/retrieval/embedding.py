from __future__ import annotations

import json
import os
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, value: str) -> list[float]: ...

    def embed_many(self, values: list[str]) -> list[list[float]]: ...


class FastEmbedProvider:
    """Local CPU embedding provider used by the retrieval path."""

    name = "fastembed-bge-small-en-v1.5"
    dimensions = 384

    def __init__(self, *, batch_size: int = 256, threads: int | None = None) -> None:
        from fastembed import TextEmbedding
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=threads)

    def embed(self, value: str) -> list[float]:
        return self.embed_many([value])[0]

    def embed_many(self, values: list[str]) -> list[list[float]]:
        if not values:
            return []
        return [[float(item) for item in vector] for vector in self.model.embed(values, batch_size=self.batch_size)]


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider for OpenAI-compatible APIs such as OpenRouter."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
        retries: int = 3,
    ) -> None:
        self.model = model or os.environ.get("TERMYTEDB_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TERMYTEDB_EMBEDDING_API_KEY")
        if not self.api_key:
            raise ValueError("an embedding API key is required")
        self.base_url = (base_url or os.environ.get("TERMYTEDB_EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        configured_dimensions = dimensions or int(os.environ.get("TERMYTEDB_EMBEDDING_DIMENSIONS", "1536"))
        self.dimensions = configured_dimensions
        if configured_dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        self.timeout = timeout
        self.retries = max(1, retries if retries != 3 else int(os.environ.get("TERMYTEDB_EMBEDDING_RETRIES", "6")))
        self.name = f"openai-compatible-{self.model}"

    def embed(self, value: str) -> list[float]:
        return self.embed_many([value])[0]

    def embed_many(self, values: list[str]) -> list[list[float]]:
        if not values:
            return []
        body = json.dumps({"model": self.model, "input": values, "encoding_format": "float"}).encode("utf-8")
        request = Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                data = payload.get("data")
                if not isinstance(data, list) or len(data) != len(values):
                    raise ValueError("embedding provider returned an unexpected item count")
                ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
                vectors = [[float(number) for number in item["embedding"]] for item in ordered]
                if any(len(vector) != self.dimensions for vector in vectors):
                    raise ValueError(f"embedding dimension does not match configured dimensions {self.dimensions}")
                return vectors
            except HTTPError as error:
                last_error = error
                if error.code not in {408, 429, 500, 502, 503, 504} or attempt == self.retries - 1:
                    raise RuntimeError(f"embedding provider returned HTTP {error.code}") from error
            except (URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt == self.retries - 1:
                    raise RuntimeError("embedding provider request failed") from error
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError("embedding provider request failed") from last_error

def cosine(left: list[float], right: list[float]) -> float:
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))

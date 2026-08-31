"""Local and remote embedding providers.

Embedding sizes and model names are defined in `src/config/settings.py` and
`src/config/embeddings.py`. Edit there to change dimensions without touching
provider logic.
"""

from __future__ import annotations

import json
import os
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import numpy.typing as npt

from ..config import embeddings as _embed_cfg
from ..config.settings import EMBEDDING as _EMBEDDING_SETTINGS


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, value: str) -> list[float]: ...

    def embed_many(self, values: list[str]) -> list[list[float]]: ...


class FastEmbedProvider:
    """Local CPU embedding provider used by the retrieval path."""

    name = _embed_cfg.FASTEMBED_MODEL
    dimensions = _embed_cfg.FASTEMBED_DIMENSIONS

    def __init__(self, *, batch_size: int | None = None, threads: int | None = None) -> None:
        from fastembed import TextEmbedding

        self.batch_size = batch_size if batch_size is not None else _EMBEDDING_SETTINGS.fastembed_batch_size
        self.model = TextEmbedding(model_name=_embed_cfg.FASTEMBED_MODEL, threads=threads)

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
        self.model = model or os.environ.get("TERMYTEDB_EMBEDDING_MODEL", "")
        if not self.model:
            raise ValueError("TERMYTEDB_EMBEDDING_MODEL is required")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TERMYTEDB_EMBEDDING_API_KEY")
        if not self.api_key:
            raise ValueError("an embedding API key is required")
        self.base_url = (base_url or os.environ.get("TERMYTEDB_EMBEDDING_BASE_URL", _embed_cfg.OPENAI_DEFAULT_BASE_URL)).rstrip("/")
        configured_dimensions = dimensions or int(os.environ.get("TERMYTEDB_EMBEDDING_DIMENSIONS", str(_EMBEDDING_SETTINGS.openai_default_dimensions)))
        self.dimensions = configured_dimensions
        if configured_dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        self.timeout = timeout if timeout != 60.0 else _EMBEDDING_SETTINGS.openai_timeout
        self.retries = max(1, retries if retries != 3 else _EMBEDDING_SETTINGS.openai_retries)
        self.name = f"openai-compatible-{self.model}"

    def embed(self, value: str) -> list[float]:
        return self.embed_many([value])[0]

    def embed_many(self, values: list[str]) -> list[list[float]]:
        if not values:
            return []
        body = json.dumps(
            {
                "model": self.model,
                "input": values,
                "dimensions": self.dimensions,
                "encoding_format": "float",
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter attribution headers are optional, but use its documented
        # casing so requests show up correctly in the provider dashboard.
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "https://termyte.dev")
        title = os.environ.get("OPENROUTER_TITLE", "TermyteDB")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
        request = Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers=headers,
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
                detail = error.read().decode("utf-8", errors="replace").strip()
                retry_after = error.headers.get("Retry-After")
                suffix = f" response={detail}" if detail else ""
                if retry_after:
                    suffix += f" retry_after={retry_after}"
                if error.code not in {408, 429, 500, 502, 503, 504} or attempt == self.retries - 1:
                    raise RuntimeError(f"embedding provider returned HTTP {error.code}{suffix}") from error
            except (URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt == self.retries - 1:
                    raise RuntimeError("embedding provider request failed") from error
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError("embedding provider request failed") from last_error


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    score = float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))
    return max(0.0, min(1.0, score))


def pack_embedding(vector: list[float]) -> bytes:
    """Serialize one embedding as compact little-endian float32 bytes."""
    return np.asarray(vector, dtype="<f4").tobytes()


def batch_dot(query: list[float], vectors: list[bytes], dimensions: int) -> npt.NDArray[np.float32]:
    """Score fixed-width float32 vector buffers in one NumPy operation."""
    if not vectors:
        return np.empty(0, dtype=np.float32)
    query_array = np.asarray(query, dtype=np.float32)
    if query_array.size != dimensions:
        raise ValueError("query embedding dimensions do not match stored vectors")
    expected_bytes = dimensions * np.dtype("<f4").itemsize
    if any(len(vector) != expected_bytes for vector in vectors):
        raise ValueError("stored embedding BLOB has an invalid size")
    matrix = np.frombuffer(b"".join(vectors), dtype="<f4").reshape(len(vectors), dimensions)
    return np.clip(matrix @ query_array, 0.0, 1.0)

from __future__ import annotations

from typing import Protocol

class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, value: str) -> list[float]: ...


class FastEmbedProvider:
    """Local CPU embedding provider used by the retrieval path."""

    name = "fastembed-bge-small-en-v1.5"
    dimensions = 384

    def __init__(self) -> None:
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(self, value: str) -> list[float]:
        return [float(item) for item in next(iter(self.model.embed([value])))]

def cosine(left: list[float], right: list[float]) -> float:
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))

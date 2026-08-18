from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

DIMENSIONS = 32


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, value: str) -> list[float]: ...


class LocalHashEmbedding:
    name = "local-hash-v1"
    dimensions = DIMENSIONS

    def embed(self, value: str) -> list[float]:
        return embed_text(value)


class FastEmbedProvider:
    """Optional CPU dense embeddings. The hash provider remains the offline fallback."""

    name = "fastembed-bge-small-en-v1.5"
    dimensions = 384

    def __init__(self) -> None:
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(self, value: str) -> list[float]:
        return [float(item) for item in next(iter(self.model.embed([value])))]


def embed_text(value: str) -> list[float]:
    """Small deterministic local embedding for offline operation and tests."""
    vector = [0.0] * DIMENSIONS
    for token in re.findall(r"[\w./:-]+", value.casefold()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % DIMENSIONS
        vector[index] += 1.0 if digest[2] & 1 else -1.0
    norm = math.sqrt(sum(item * item for item in vector))
    return [round(item / norm, 8) for item in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))

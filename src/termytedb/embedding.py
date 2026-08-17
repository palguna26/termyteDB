from __future__ import annotations

import hashlib
import math
import re

DIMENSIONS = 32


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

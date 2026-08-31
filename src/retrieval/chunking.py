"""Lossless, session-bounded semantic chunking.

Chunks are an index over events.  The event payload remains the source of truth;
the contextual string is only used for retrieval.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticChunk:
    chunk_id: str
    namespace_id: str
    session_id: str
    ordinal: int
    event_ids: tuple[str, ...]
    text: str
    contextual_text: str
    document_date: str | None
    event_dates: tuple[str, ...] = ()


def build_chunks(events: list[dict[str, Any]], *, window: int = 4, overlap: int = 1) -> list[SemanticChunk]:
    """Build deterministic 2-6 turn windows without crossing sessions."""
    if not 2 <= window <= 6:
        raise ValueError("window must be between 2 and 6 turns")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be smaller than window")
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        sid = str(event.get("session_id") or event.get("stream_id") or event["id"])
        groups.setdefault(sid, []).append(event)
    result: list[SemanticChunk] = []
    for sid, rows in groups.items():
        rows = sorted(rows, key=lambda r: (str(r.get("occurred_at") or ""), str(r["id"])))
        step = max(1, window - overlap)
        ordinal = 0
        for start in range(0, len(rows), step):
            part = rows[start : start + window]
            if not part:
                break
            texts = [str(r.get("text") or r.get("payload_text") or "").strip() for r in part]
            texts = [t for t in texts if t]
            raw = "\n".join(texts)
            if not raw:
                continue
            before = str(rows[start - 1].get("text") or "").strip() if start else ""
            after = str(rows[start + len(part)].get("text") or "").strip() if start + len(part) < len(rows) else ""
            contextual = "\n".join(x for x in (before, raw, after) if x)
            ids = tuple(str(r["id"]) for r in part)
            digest = hashlib.sha256(json.dumps([sid, ordinal, ids, raw], separators=(",", ":")).encode()).hexdigest()[:24]
            result.append(SemanticChunk(f"chunk_{digest}", str(part[0].get("namespace_id", "")), sid, ordinal, ids, raw, contextual, str(part[0].get("occurred_at")) if part[0].get("occurred_at") else None, tuple(str(r["occurred_at"]) for r in part if r.get("occurred_at"))))
            ordinal += 1
            if start + window >= len(rows):
                break
    return result

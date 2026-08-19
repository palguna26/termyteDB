from __future__ import annotations

import array
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from ..storage.db import Database
from .embedding import FastEmbedProvider, cosine

RRF_K = 60
HISTORY_RE = re.compile(r"\b(previously|used to|former|formerly|before|previous|history|historical)\b", re.I)


@dataclass(frozen=True)
class AtomHit:
    atom_id: str
    session_id: str
    fact: str
    timestamp: str | None
    source_role: str
    score: float


def rrf_merge(lists: list[list[AtomHit]], k: int = RRF_K) -> list[AtomHit]:
    merged: dict[str, tuple[AtomHit, float]] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked):
            score = 1.0 / (k + rank + 1)
            previous = merged.get(item.atom_id)
            merged[item.atom_id] = (item, score if previous is None else previous[1] + score)
    return [AtomHit(item.atom_id, item.session_id, item.fact, item.timestamp, item.source_role, score)
            for item, score in sorted(merged.values(), key=lambda pair: (-pair[1], pair[0].atom_id))]


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w./:-]+", query, re.UNICODE)
    return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms if term)


def search_atoms(db: Database, query: str, limit: int = 20,
                 vector_search: Callable[[str, int], list[AtomHit]] | None = None) -> list[AtomHit]:
    historical = bool(HISTORY_RE.search(query))
    match = _fts_query(query)
    lexical: list[AtomHit] = []
    if match:
        rows = db.execute(
            """SELECT a.atom_id, a.session_id, a.fact, a.timestamp, a.source_role,
                      bm25(atoms_fts) AS rank
               FROM atoms_fts JOIN atoms a ON a.atom_id=atoms_fts.atom_id
               WHERE atoms_fts MATCH ? AND (? OR a.invalid_at IS NULL)
               ORDER BY rank LIMIT ?""",
            (match, historical, max(limit * 3, 20)),
        ).fetchall()
        lexical = [AtomHit(r["atom_id"], r["session_id"], r["fact"], r["timestamp"], r["source_role"], float(r["rank"])) for r in rows]
    semantic = vector_search(query, max(limit * 3, 20)) if vector_search else dense_search_atoms(db, query, max(limit * 3, 20))
    if not historical:
        semantic = [item for item in semantic if db.execute("SELECT invalid_at FROM atoms WHERE atom_id=?", (item.atom_id,)).fetchone()[0] is None]
    return rrf_merge([lexical, semantic])[:limit]


def dense_search_atoms(db: Database, query: str, limit: int = 20, provider: EmbeddingProvider | None = None) -> list[AtomHit]:
    if provider is None:
        try:
            provider = _cached_fastembed_provider()
        except ImportError:
            return []
    available = db.execute("SELECT 1 FROM atom_embeddings WHERE provider=? LIMIT 1", (provider.name,)).fetchone()
    if available is None:
        return []
    query_vector = provider.embed(query)
    rows = db.execute(
        "SELECT a.atom_id, a.session_id, a.fact, a.timestamp, a.source_role, e.vector "
        "FROM atoms a JOIN atom_embeddings e ON e.atom_id=a.atom_id WHERE e.provider=?",
        (provider.name,),
    ).fetchall()
    hits: list[AtomHit] = []
    for row in rows:
        vector = list(array.array("f", row["vector"]))
        hits.append(AtomHit(row["atom_id"], row["session_id"], row["fact"], row["timestamp"], row["source_role"], cosine(query_vector, vector)))
    return sorted(hits, key=lambda item: (-item.score, item.atom_id))[:limit]


@lru_cache(maxsize=1)
def _cached_fastembed_provider() -> FastEmbedProvider:
    """Reuse the local model across queries instead of reloading it per search."""
    return FastEmbedProvider()


def rerank_and_filter(query: str, hits: list[AtomHit], threshold: float = 0.25) -> list[AtomHit] | None:
    """Optional FlashRank reranking with hard abstention."""
    try:
        from flashrank import Ranker, RerankRequest  # type: ignore[import-untyped]
    except ImportError:
        return hits if hits else None
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    passages = [{"id": hit.atom_id, "text": hit.fact, "meta": hit} for hit in hits]
    results = ranker.rerank(RerankRequest(query=query, passages=passages))
    if not results or float(results[0].get("score", 0.0)) < threshold:
        return None
    return [item["meta"] for item in results]


def pack_context(hits: list[AtomHit], token_budget: int = 3000) -> str:
    groups: dict[str, list[AtomHit]] = {}
    for hit in hits:
        groups.setdefault(hit.session_id, []).append(hit)
    lines: list[str] = []
    for session_id in sorted(groups, key=lambda sid: min((h.timestamp or "9999") for h in groups[sid])):
        lines.append(f"[Session {session_id}]")
        for hit in sorted(groups[session_id], key=lambda item: (item.timestamp or "9999", item.atom_id)):
            lines.append(f"[Timestamp: {hit.timestamp or 'unknown'}] [Role: {hit.source_role}] {hit.fact}")
    packed: list[str] = []
    count = 0
    for line in lines:
        cost = len(line.split())
        if count + cost > token_budget:
            break
        packed.append(line)
        count += cost
    return "\n".join(packed)
